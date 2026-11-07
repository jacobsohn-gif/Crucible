"""Multi-account Hybrid Credential Vault & Zero-Config Auth Discovery (`ush auth`).

Supports:
1. Zero-config Google Cloud / Vertex AI ADC (`gcloud` CLI + ADC refresh tokens — unlocks Gemini & Vertex Claude!).
2. Zero-config localhost inference probing (`127.0.0.1` Ollama / vLLM / LM Studio).
3. Standard environment variable discovery (GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, etc.).
4. OS Keychain primary vault on macOS (`/usr/bin/security`) + Linux `secret-tool`, with atomic `0600` headless fallback.
5. Official OAuth 2.0 + PKCE (`127.0.0.1` loopback + SSH `--no-browser` copy-URL) and RFC 8628 Device-Code flow.
6. Multi-profile management and automatic HTTP 429 cooldown failover.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import getpass
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
import pathlib
import platform
import re
import secrets
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
import webbrowser


KEYCHAIN_SERVICE = "ush-cli-credentials"

ENV_PROVIDER_MAP: tuple[tuple[str, str, str, str], ...] = (
    # (env_var, profile_name, provider, base_url)
    ("GEMINI_API_KEY", "env-gemini", "google", "https://generativelanguage.googleapis.com"),
    ("GOOGLE_API_KEY", "env-google", "google", "https://generativelanguage.googleapis.com"),
    ("GEMINI_ENTERPRISE_API_KEY", "env-gemini-enterprise", "gemini-enterprise", "https://aiplatform.googleapis.com"),
    ("VERTEX_API_KEY", "env-vertex-express", "gemini-enterprise", "https://aiplatform.googleapis.com"),
    ("GOOGLE_CLOUD_API_KEY", "env-gcloud-apikey", "gemini-enterprise", "https://aiplatform.googleapis.com"),
    ("ANTIGRAVITY_API_KEY", "env-antigravity", "antigravity", "https://generativelanguage.googleapis.com"),
    ("ANTHROPIC_API_KEY", "env-anthropic", "anthropic", "https://api.anthropic.com"),
    ("OPENAI_API_KEY", "env-openai", "openai", "https://api.openai.com/v1"),
    ("LITELLM_API_KEY", "env-litellm", "litellm", "http://127.0.0.1:4000/v1"),
    ("LITELLM_MASTER_KEY", "env-litellm-master", "litellm", "http://127.0.0.1:4000/v1"),
    ("OPENROUTER_API_KEY", "env-openrouter", "openrouter", "https://openrouter.ai/api/v1"),
    ("DEEPSEEK_API_KEY", "env-deepseek", "openrouter", "https://api.deepseek.com/v1"),
    ("DASHSCOPE_API_KEY", "env-qwen", "openrouter", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("MOONSHOT_API_KEY", "env-kimi", "openrouter", "https://api.moonshot.ai/v1"),
    ("ZHIPUAI_API_KEY", "env-glm", "openrouter", "https://open.bigmodel.cn/api/paas/v4"),
)

PROVIDER_DEFAULT_URLS: dict[str, str] = {
    "google": "https://generativelanguage.googleapis.com",
    "vertex": "https://aiplatform.googleapis.com",
    "gemini-enterprise": "https://aiplatform.googleapis.com",
    "code-assist": "https://cloudcode-pa.googleapis.com",
    "antigravity": "https://aiplatform.googleapis.com",
    "vertex-anthropic": "https://aiplatform.googleapis.com",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "litellm": "http://127.0.0.1:4000/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "local": "http://127.0.0.1:11434/v1",
}

PROVIDER_FAMILY_MAP: dict[str, str] = {
    "google": "gemini",
    "vertex": "gemini",
    "gemini-enterprise": "gemini",
    "antigravity": "gemini",
    "anthropic": "anthropic",
    "vertex-anthropic": "anthropic",
    "openai": "openai",
    "litellm": "litellm",
    "openrouter": "openrouter",
    "local": "local_oss",
}


def mask_secret(value: str) -> str:
    if not value:
        return "<empty>"
    if len(value) <= 8:
        return "****"
    return f"{value[:4]}...{value[-4:]}"


def secure_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    return ctx


def is_trusted_host_local_file(path: pathlib.Path) -> bool:
    """Verifies `path` is a regular non-symlink file owned by the current user and not group/world-writable."""
    try:
        if path.is_symlink() or not path.is_file():
            return False
        st = path.stat()
        if hasattr(os, "getuid") and st.st_uid != os.getuid():
            return False
        if hasattr(os, "getuid") and (st.st_mode & 0o022) != 0:
            return False
        return True
    except OSError:
        return False


@dataclass
class AuthProfile:
    name: str
    # google | vertex | gemini-enterprise | antigravity | vertex-anthropic | anthropic | openai | litellm | openrouter | local | <plugin>
    provider: str
    # api_key | bearer | oauth | gcloud_adc | command | none
    auth_kind: str
    base_url: str
    project_id: str | None = None
    region: str = "global"
    default_model: str | None = None
    source: str = "vault"  # vault | keychain | workstation | env:* | gcloud | localhost | codex_cli | claude_cli
    # Wire sub-mode: "" (auto) | "express" | "code_assist" | "responses" | "chat_completions"
    api_mode: str = ""
    # Optional service tier (e.g., "priority" on Gemini Enterprise / Antigravity / OpenAI)
    service_tier: str = ""
    # Optional non-secret organizational/tenant headers (e.g., OpenAI-Organization, OpenAI-Project, ChatGPT-Account-Id)
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Host-local credential helper command (`shlex.split`, `shell=False`) strictly from `~/.ush/` when `auth_kind == "command"`
    credential_command: str = ""
    token_ttl_seconds: int = 1800
    # Optional host marker path predicate: profile is eligible only on machines where this path exists
    require_host_path: str = ""
    # Optional host-local transport plugin module name in `~/.ush/providers/<transport_plugin>.py`
    transport_plugin: str = ""
    # Ephemeral runtime cooldown timestamp after HTTP 429
    cooldown_until: float = 0.0

    def is_available_on_host(self) -> bool:
        if self.api_mode == "agy_acp" or (self.base_url or "").startswith("agy://"):
            from ush.brain.agy_acp import resolve_agy_binary

            if not resolve_agy_binary():
                return False
        if not self.require_host_path:
            return True
        try:
            return pathlib.Path(self.require_host_path).expanduser().exists()
        except OSError:
            return False

    def is_available(self, now: float | None = None) -> bool:
        if not self.is_available_on_host():
            return False
        return (now if now is not None else time.time()) >= self.cooldown_until

    def to_metadata_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "provider": self.provider,
            "auth_kind": self.auth_kind,
            "base_url": self.base_url,
            "project_id": self.project_id,
            "region": self.region,
            "default_model": self.default_model,
            "source": self.source,
        }
        if self.api_mode:
            d["api_mode"] = self.api_mode
        if self.service_tier:
            d["service_tier"] = self.service_tier
        if self.extra_headers:
            d["extra_headers"] = dict(self.extra_headers)
        if self.credential_command:
            d["credential_command"] = self.credential_command
        if self.token_ttl_seconds != 1800:
            d["token_ttl_seconds"] = int(self.token_ttl_seconds)
        if self.require_host_path:
            d["require_host_path"] = self.require_host_path
        if self.transport_plugin:
            d["transport_plugin"] = self.transport_plugin
        return d


@dataclass
class StoredSecret:
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    client_id: str = ""
    client_secret: str = ""
    token_url: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": self.expires_at,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "token_url": self.token_url,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> StoredSecret:
        d = json.loads(raw)
        return cls(
            access_token=str(d.get("access_token", "")),
            refresh_token=str(d.get("refresh_token", "")),
            expires_at=float(d.get("expires_at", 0.0)),
            client_id=str(d.get("client_id", "")),
            client_secret=str(d.get("client_secret", "")),
            token_url=str(d.get("token_url", "")),
        )


class CredentialVault:
    """Hybrid OS-Keychain + `0600` headless credential manager with multi-account failover."""

    def __init__(self, ush_home: pathlib.Path | None = None, probe_ambient: bool = True) -> None:
        self.ush_home = (ush_home or pathlib.Path(os.environ.get("USH_HOME", "~/.ush"))).expanduser().resolve()
        self.profiles_path = self.ush_home / "auth_profiles.json"
        self.workstation_path = self.ush_home / "workstation.json"
        self.fallback_vault_path = self.ush_home / "credentials.vault.json"
        self._profiles: dict[str, AuthProfile] = {}
        self._active_profile: str | None = None
        self._gcloud_token_cache: tuple[str, float] | None = None
        self._cmd_token_cache: dict[str, tuple[str, float]] = {}
        self.security_warnings: list[str] = []
        self._lock = threading.Lock()
        self._load_profiles()
        if probe_ambient:
            self.discover_ambient_credentials()

    def _ensure_home(self) -> None:
        self.ush_home.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.ush_home, 0o700)
        except OSError:
            pass

    def _atomic_write_0600(self, target: pathlib.Path, content: str) -> None:
        self._ensure_home()
        fd, tmp_name = tempfile.mkstemp(dir=str(self.ush_home), prefix=".tmp_auth_")
        try:
            os.chmod(tmp_name, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    # ---- OS Keychain + 0600 Fallback Secret Storage ----

    def _use_macos_keychain(self) -> bool:
        if os.environ.get("USH_DISABLE_KEYCHAIN") == "1":
            return False
        return platform.system() == "Darwin" and pathlib.Path("/usr/bin/security").is_file()

    def store_secret(self, profile_name: str, secret: StoredSecret) -> str:
        payload = secret.to_json()
        if self._use_macos_keychain():
            try:
                res = subprocess.run(
                    [
                        "/usr/bin/security",
                        "add-generic-password",
                        "-U",
                        "-s",
                        KEYCHAIN_SERVICE,
                        "-a",
                        profile_name,
                        "-w",
                        payload,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0:
                    return "keychain"
            except (subprocess.SubprocessError, OSError):
                pass
        # Headless SSH / container fallback (`0600` strict permissions)
        vault: dict[str, Any] = {}
        if self.fallback_vault_path.is_file():
            try:
                vault = json.loads(self.fallback_vault_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                vault = {}
        vault[profile_name] = json.loads(payload)
        self._atomic_write_0600(self.fallback_vault_path, json.dumps(vault, indent=2))
        return "vault"

    def load_secret(self, profile_name: str) -> StoredSecret | None:
        if self._use_macos_keychain():
            try:
                res = subprocess.run(
                    [
                        "/usr/bin/security",
                        "find-generic-password",
                        "-s",
                        KEYCHAIN_SERVICE,
                        "-a",
                        profile_name,
                        "-w",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if res.returncode == 0 and res.stdout.strip():
                    try:
                        return StoredSecret.from_json(res.stdout.strip())
                    except json.JSONDecodeError:
                        return StoredSecret(access_token=res.stdout.strip())
            except (subprocess.SubprocessError, OSError):
                pass
        if self.fallback_vault_path.is_file():
            try:
                vault = json.loads(self.fallback_vault_path.read_text(encoding="utf-8"))
                if profile_name in vault:
                    return StoredSecret.from_json(json.dumps(vault[profile_name]))
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def delete_secret(self, profile_name: str) -> None:
        if self._use_macos_keychain():
            try:
                subprocess.run(
                    [
                        "/usr/bin/security",
                        "delete-generic-password",
                        "-s",
                        KEYCHAIN_SERVICE,
                        "-a",
                        profile_name,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
            except (subprocess.SubprocessError, OSError):
                pass
        if self.fallback_vault_path.is_file():
            try:
                vault = json.loads(self.fallback_vault_path.read_text(encoding="utf-8"))
                if profile_name in vault:
                    del vault[profile_name]
                    self._atomic_write_0600(self.fallback_vault_path, json.dumps(vault, indent=2))
            except (OSError, json.JSONDecodeError):
                pass

    # ---- Profile Metadata Persistence ----

    def _load_profiles(self) -> None:
        valid_keys = set(AuthProfile.__dataclass_fields__.keys()) - {"cooldown_until"}
        if self.profiles_path.is_file():
            trusted_main = is_trusted_host_local_file(self.profiles_path)
            if not trusted_main:
                self.security_warnings.append(
                    f"Ignoring privileged command/plugin hooks in `{self.profiles_path}` (unsafe permissions/owner; run `chmod 600`)."
                )
            try:
                data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
                self._active_profile = data.get("active_profile")
                for item in data.get("profiles", []):
                    if not isinstance(item, dict):
                        continue
                    filtered = {k: v for k, v in item.items() if k in valid_keys}
                    if not trusted_main:
                        filtered.pop("credential_command", None)
                        filtered.pop("transport_plugin", None)
                        if filtered.get("auth_kind") == "command":
                            continue
                    prof = AuthProfile(**filtered)
                    self._profiles[prof.name] = prof
            except (OSError, json.JSONDecodeError, TypeError):
                pass

        # Host-local workstation overlay (`~/.ush/workstation.json`, strictly outside Git, 0600/non-group-writable)
        if self.workstation_path.exists() or self.workstation_path.is_symlink():
            if not is_trusted_host_local_file(self.workstation_path):
                self.security_warnings.append(
                    f"Refused `{self.workstation_path}`: must be a regular user-owned file without group/world write bits (`chmod 600`)."
                )
            else:
                try:
                    ws_data = json.loads(self.workstation_path.read_text(encoding="utf-8"))
                    ws_items = ws_data.get("profiles", []) if isinstance(ws_data, dict) else []
                    for item in ws_items:
                        if not isinstance(item, dict):
                            continue
                        filtered = {k: v for k, v in item.items() if k in valid_keys}
                        filtered["source"] = "workstation"
                        prof = AuthProfile(**filtered)
                        self._profiles[prof.name] = prof
                    ws_active = ws_data.get("active_profile") if isinstance(ws_data, dict) else None
                    if ws_active and ws_active in self._profiles and self._profiles[ws_active].is_available_on_host():
                        cur = self._profiles.get(self._active_profile or "")
                        if cur is None or not cur.is_available_on_host() or bool(ws_data.get("prefer_active", False)):
                            self._active_profile = str(ws_active)
                except (OSError, json.JSONDecodeError, TypeError):
                    pass

    def _save_profiles(self) -> None:
        persisted = [
            p.to_metadata_dict()
            for p in self._profiles.values()
            if p.source in ("vault", "keychain")
        ]
        self._atomic_write_0600(
            self.profiles_path,
            json.dumps({"active_profile": self._active_profile, "profiles": persisted}, indent=2),
        )

    # ---- Zero-Config Ambient Discovery ----

    @staticmethod
    def _resolve_env_base_url(provider: str, default_url: str) -> str:
        if provider == "anthropic":
            return (os.environ.get("ANTHROPIC_BASE_URL") or default_url).rstrip("/")
        if provider == "openai":
            return (os.environ.get("OPENAI_BASE_URL") or default_url).rstrip("/")
        if provider in ("vertex", "gemini-enterprise"):
            return (
                os.environ.get("GOOGLE_VERTEX_BASE_URL")
                or os.environ.get("VERTEX_BASE_URL")
                or default_url
            ).rstrip("/")
        if provider == "antigravity":
            return (
                os.environ.get("ANTIGRAVITY_BASE_URL")
                or os.environ.get("ANTIGRAVITY_ENDPOINT")
                or os.environ.get("ANTIGRAVITY_LITERT_URL")
                or default_url
            ).rstrip("/")
        if provider == "litellm":
            return (
                os.environ.get("LITELLM_BASE_URL")
                or os.environ.get("LITELLM_PROXY_URL")
                or default_url
            ).rstrip("/")
        if provider == "google":
            return (os.environ.get("GEMINI_BASE_URL") or default_url).rstrip("/")
        return default_url.rstrip("/")

    @staticmethod
    def _openai_env_headers() -> dict[str, str]:
        hdrs: dict[str, str] = {}
        org = (os.environ.get("OPENAI_ORG_ID") or os.environ.get("OPENAI_ORGANIZATION") or "").strip()
        proj = (os.environ.get("OPENAI_PROJECT") or "").strip()
        if org:
            hdrs["OpenAI-Organization"] = org
        if proj:
            hdrs["OpenAI-Project"] = proj
        return hdrs

    def discover_ambient_credentials(self) -> None:
        self._ambient_secrets: dict[str, StoredSecret] = getattr(self, "_ambient_secrets", {})

        # 1. Standard Environment Variables (kept in memory only; never written to disk)
        for env_var, prof_name, provider, base_url in ENV_PROVIDER_MAP:
            if os.environ.get(env_var):
                actual_provider = "openrouter" if provider == "openrouter" else provider
                resolved_url = self._resolve_env_base_url(actual_provider, base_url)
                api_mode = ""
                svc_tier = ""
                extra_hdrs: dict[str, str] = {}
                if actual_provider == "gemini-enterprise":
                    api_mode = "express"
                    svc_tier = os.environ.get("GEMINI_SERVICE_TIER", "").strip()
                elif actual_provider == "antigravity":
                    svc_tier = os.environ.get("ANTIGRAVITY_SERVICE_TIER", "").strip()
                elif actual_provider == "openai":
                    extra_hdrs = self._openai_env_headers()
                    api_mode = os.environ.get("OPENAI_API_MODE", "").strip()
                self._profiles.setdefault(
                    prof_name,
                    AuthProfile(
                        name=prof_name,
                        provider=actual_provider,
                        auth_kind="api_key",
                        base_url=resolved_url,
                        source=f"env:{env_var}",
                        api_mode=api_mode,
                        service_tier=svc_tier,
                        extra_headers=extra_hdrs,
                    ),
                )

        # 1b. Anthropic Bearer / OAuth Setup Tokens (`ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_OAUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN`)
        for env_var, prof_name in (
            ("ANTHROPIC_AUTH_TOKEN", "env-anthropic-bearer"),
            ("ANTHROPIC_OAUTH_TOKEN", "env-anthropic-oauth"),
            ("CLAUDE_CODE_OAUTH_TOKEN", "env-claude-oauth"),
        ):
            if os.environ.get(env_var):
                self._profiles.setdefault(
                    prof_name,
                    AuthProfile(
                        name=prof_name,
                        provider="anthropic",
                        auth_kind="bearer",
                        base_url=self._resolve_env_base_url("anthropic", "https://api.anthropic.com"),
                        default_model="claude-sonnet-4-20250514",
                        source=f"env:{env_var}",
                    ),
                )

        # 1c. Antigravity Bearer Token or Custom/LiteRT Endpoint env vars
        if os.environ.get("ANTIGRAVITY_AUTH_TOKEN"):
            self._profiles.setdefault(
                "env-antigravity-token",
                AuthProfile(
                    name="env-antigravity-token",
                    provider="antigravity",
                    auth_kind="bearer",
                    base_url=self._resolve_env_base_url("antigravity", "https://aiplatform.googleapis.com"),
                    service_tier=os.environ.get("ANTIGRAVITY_SERVICE_TIER", "").strip(),
                    default_model="gemini-3.1-pro-preview",
                    source="env:ANTIGRAVITY_AUTH_TOKEN",
                ),
            )
        elif any(os.environ.get(k) for k in ("ANTIGRAVITY_BASE_URL", "ANTIGRAVITY_ENDPOINT", "ANTIGRAVITY_LITERT_URL")) and "env-antigravity" not in self._profiles:
            ag_url = self._resolve_env_base_url("antigravity", "http://127.0.0.1:8080/v1")
            is_google_host = "googleapis.com" in ag_url
            self._profiles.setdefault(
                "env-antigravity-endpoint",
                AuthProfile(
                    name="env-antigravity-endpoint",
                    provider="antigravity",
                    auth_kind="gcloud_adc" if is_google_host else "none",
                    base_url=ag_url,
                    project_id=os.environ.get("GOOGLE_CLOUD_PROJECT") or self._read_gcloud_default_project(),
                    region=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                    service_tier=os.environ.get("ANTIGRAVITY_SERVICE_TIER", "").strip(),
                    source="env:ANTIGRAVITY_BASE_URL",
                ),
            )

        # 1d. LiteLLM Proxy Base URL without explicit API key (e.g. unauthenticated team/local gateway)
        if (os.environ.get("LITELLM_BASE_URL") or os.environ.get("LITELLM_PROXY_URL")) and not any(
            p.provider == "litellm" for p in self._profiles.values()
        ):
            self._profiles.setdefault(
                "env-litellm-url",
                AuthProfile(
                    name="env-litellm-url",
                    provider="litellm",
                    auth_kind="none",
                    base_url=self._resolve_env_base_url("litellm", "http://127.0.0.1:4000/v1"),
                    source="env:LITELLM_BASE_URL",
                ),
            )

        # 1b. Sanctioned local Antigravity CLI (`agy acp` — zero GCP/Argolis project, zero scraped tokens)
        from ush.brain.agy_acp import resolve_agy_binary

        if resolve_agy_binary():
            self._profiles.setdefault(
                "agy-antigravity",
                AuthProfile(
                    name="agy-antigravity",
                    provider="antigravity",
                    auth_kind="none",
                    base_url="agy://acp",
                    default_model="gemini-4-argon",
                    source="agy_cli",
                    api_mode="agy_acp",
                ),
            )

        # 2. Google Cloud ADC (`gcloud` / ADC file) -> Vertex, Gemini Enterprise, Antigravity & Vertex Claude
        adc_file = pathlib.Path(
            os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "~/.config/gcloud/application_default_credentials.json",
            )
        ).expanduser()
        gcloud_bin = shutil.which("gcloud") or str(
            pathlib.Path("~/google-cloud-sdk/bin/gcloud").expanduser()
        )
        project = (
            os.environ.get("GOOGLE_CLOUD_PROJECT")
            or os.environ.get("CLOUD_ML_PROJECT_ID")
            or os.environ.get("OPENCODE_GEMINI_PROJECT_ID")
            or self._read_gcloud_default_project()
        )
        has_gcp_creds = adc_file.is_file() or pathlib.Path(gcloud_bin).is_file()
        if has_gcp_creds and project:
            region = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            v_base = self._resolve_env_base_url("vertex", "https://aiplatform.googleapis.com")
            self._profiles.setdefault(
                "gcloud-antigravity",
                AuthProfile(
                    name="gcloud-antigravity",
                    provider="antigravity",
                    auth_kind="gcloud_adc",
                    base_url=self._resolve_env_base_url("antigravity", v_base),
                    project_id=project,
                    region=region,
                    default_model="gemini-3.1-pro-preview",
                    service_tier=os.environ.get("ANTIGRAVITY_SERVICE_TIER", "").strip(),
                    source="gcloud",
                ),
            )
            self._profiles.setdefault(
                "gcloud-gemini-enterprise",
                AuthProfile(
                    name="gcloud-gemini-enterprise",
                    provider="gemini-enterprise",
                    auth_kind="gcloud_adc",
                    base_url=v_base,
                    project_id=project,
                    region=region,
                    default_model="gemini-3.1-pro-preview",
                    service_tier=os.environ.get("GEMINI_SERVICE_TIER", "").strip(),
                    source="gcloud",
                ),
            )
            self._profiles.setdefault(
                "gcloud-vertex",
                AuthProfile(
                    name="gcloud-vertex",
                    provider="vertex",
                    auth_kind="gcloud_adc",
                    base_url=v_base,
                    project_id=project,
                    region=region,
                    default_model="gemini-3.1-pro-preview",
                    source="gcloud",
                ),
            )
            self._profiles.setdefault(
                "gcloud-vertex-claude",
                AuthProfile(
                    name="gcloud-vertex-claude",
                    provider="vertex-anthropic",
                    auth_kind="gcloud_adc",
                    base_url=v_base,
                    project_id=project,
                    region=os.environ.get("VERTEX_ANTHROPIC_REGION", "us-east5"),
                    default_model="claude-sonnet-4@20250514",
                    source="gcloud",
                ),
            )
            if os.environ.get("CODE_ASSIST_ENDPOINT") or os.environ.get("GEMINI_CODE_ASSIST") == "1":
                ca_url = (os.environ.get("CODE_ASSIST_ENDPOINT") or "https://cloudcode-pa.googleapis.com").rstrip("/")
                self._profiles.setdefault(
                    "gcloud-code-assist",
                    AuthProfile(
                        name="gcloud-code-assist",
                        provider="gemini-enterprise",
                        auth_kind="gcloud_adc",
                        base_url=ca_url,
                        project_id=project,
                        region=region,
                        default_model="gemini-2.5-pro",
                        api_mode="code_assist",
                        source="gcloud",
                    ),
                )

        # 2b. Ambient OpenAI Codex CLI (`~/.codex/auth.json`) & Claude CLI (`~/.claude/.credentials.json`)
        self._discover_ambient_cli_tokens()

        # 3. Localhost zero-auth inference & LiteLLM servers (`127.0.0.1` only, <80ms check)
        for port, label, prov in (
            (4000, "local-litellm", "litellm"),
            (11434, "local-ollama", "local"),
            (8000, "local-vllm", "local"),
            (1234, "local-lmstudio", "local"),
        ):
            if _is_loopback_port_open(port):
                self._profiles.setdefault(
                    label,
                    AuthProfile(
                        name=label,
                        provider=prov,
                        auth_kind="none",
                        base_url=f"http://127.0.0.1:{port}/v1",
                        source="localhost",
                    ),
                )

        if (
            not self._active_profile
            or self._active_profile not in self._profiles
            or not self._profiles[self._active_profile].is_available_on_host()
        ):
            avail_first = next((k for k, p in self._profiles.items() if p.is_available_on_host()), None)
            if avail_first:
                self._active_profile = avail_first

    def _discover_ambient_cli_tokens(self) -> None:
        """Discovers user-owned `~/.codex/auth.json` and `~/.claude/.credentials.json` tokens strictly in memory."""
        self._ambient_secrets = getattr(self, "_ambient_secrets", {})

        # OpenAI Codex CLI (`~/.codex/auth.json`)
        codex_home = pathlib.Path(os.environ.get("CODEX_HOME", "~/.codex")).expanduser()
        codex_auth = codex_home / "auth.json"
        if codex_auth.is_file():
            try:
                raw = json.loads(codex_auth.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    api_k = str(raw.get("OPENAI_API_KEY") or "").strip()
                    tokens = raw.get("tokens") if isinstance(raw.get("tokens"), dict) else {}
                    acc_tok = str((tokens or {}).get("access_token") or "").strip()
                    ref_tok = str((tokens or {}).get("refresh_token") or "").strip()
                    acct_id = str((tokens or {}).get("account_id") or "").strip()
                    hdrs = self._openai_env_headers()
                    if acct_id:
                        hdrs["ChatGPT-Account-Id"] = acct_id
                    if api_k and "env-openai" not in self._profiles:
                        self._ambient_secrets["codex-openai"] = StoredSecret(access_token=api_k)
                        self._profiles.setdefault(
                            "codex-openai",
                            AuthProfile(
                                name="codex-openai",
                                provider="openai",
                                auth_kind="api_key",
                                base_url=self._resolve_env_base_url("openai", "https://api.openai.com/v1"),
                                default_model="gpt-5",
                                source="codex_cli",
                                api_mode="responses",
                                extra_headers=hdrs,
                            ),
                        )
                    elif acc_tok:
                        self._ambient_secrets["codex-oauth"] = StoredSecret(
                            access_token=acc_tok,
                            refresh_token=ref_tok,
                        )
                        self._profiles.setdefault(
                            "codex-oauth",
                            AuthProfile(
                                name="codex-oauth",
                                provider="openai",
                                auth_kind="oauth",
                                base_url=self._resolve_env_base_url("openai", "https://api.openai.com/v1"),
                                default_model="gpt-5",
                                source="codex_cli",
                                api_mode="responses",
                                extra_headers=hdrs,
                            ),
                        )
            except (OSError, json.JSONDecodeError):
                pass

        # Anthropic Claude CLI (`~/.claude/.credentials.json`)
        claude_dir = pathlib.Path(os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")).expanduser()
        claude_creds = claude_dir / ".credentials.json"
        if claude_creds.is_file():
            try:
                raw = json.loads(claude_creds.read_text(encoding="utf-8"))
                oauth_obj = raw.get("claudeAiOauth") if isinstance(raw, dict) else None
                if isinstance(oauth_obj, dict):
                    acc_tok = str(oauth_obj.get("accessToken") or "").strip()
                    ref_tok = str(oauth_obj.get("refreshToken") or "").strip()
                    exp_ms = float(oauth_obj.get("expiresAt") or 0.0)
                    exp_s = exp_ms / 1000.0 if exp_ms > 1e11 else exp_ms
                    if acc_tok and (exp_s <= 0 or exp_s > time.time()):
                        self._ambient_secrets["claude-oauth"] = StoredSecret(
                            access_token=acc_tok,
                            refresh_token=ref_tok,
                            expires_at=exp_s,
                        )
                        self._profiles.setdefault(
                            "claude-oauth",
                            AuthProfile(
                                name="claude-oauth",
                                provider="anthropic",
                                auth_kind="oauth",
                                base_url=self._resolve_env_base_url("anthropic", "https://api.anthropic.com"),
                                default_model="claude-sonnet-4-20250514",
                                source="claude_cli",
                            ),
                        )
            except (OSError, json.JSONDecodeError, ValueError):
                pass

    @staticmethod
    def _read_gcloud_default_project() -> str | None:
        cfg_dir = pathlib.Path("~/.config/gcloud/configurations").expanduser()
        active_file = pathlib.Path("~/.config/gcloud/active_config").expanduser()
        cfg_name = "config_default"
        if active_file.is_file():
            try:
                name = active_file.read_text(encoding="utf-8").strip()
                if name and "/" not in name and "\\" not in name:
                    cfg_name = f"config_{name}"
            except OSError:
                pass
        target = cfg_dir / cfg_name
        if target.is_file():
            try:
                for line in target.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("project"):
                        _, _, val = line.partition("=")
                        proj = val.strip()
                        if proj:
                            return proj
            except OSError:
                pass
        return None

    # ---- Credential Resolution, Refresh & 429 Failover ----

    def list_profiles(self) -> list[AuthProfile]:
        return list(self._profiles.values())

    @property
    def active_profile_name(self) -> str | None:
        return self._active_profile

    def get_active_profile(self) -> AuthProfile | None:
        if self._active_profile and self._active_profile in self._profiles:
            prof = self._profiles[self._active_profile]
            if prof.is_available():
                return prof
        # Fail over to any available profile
        for prof in self._profiles.values():
            if prof.is_available():
                return prof
        fallback = self._profiles.get(self._active_profile or "")
        if fallback is not None and not fallback.is_available_on_host():
            return next((p for p in self._profiles.values() if p.is_available_on_host()), None)
        return fallback

    def select_failover_profile(
        self,
        failed_profile: AuthProfile,
        *,
        target_family: str | None = None,
    ) -> AuthProfile | None:
        """Selects an available backup profile on HTTP 429/503, preferring exact provider then same model family."""
        now = time.time()
        wanted_fam = target_family or PROVIDER_FAMILY_MAP.get(failed_profile.provider, failed_profile.provider)
        # Pass 1: exact same provider (e.g. second API key or second GCP project)
        for prof in self._profiles.values():
            if prof.name != failed_profile.name and prof.is_available(now) and prof.provider == failed_profile.provider:
                return prof
        # Pass 2: compatible provider serving the same model family (e.g. vertex <-> gemini-enterprise <-> google <-> antigravity)
        for prof in self._profiles.values():
            if prof.name == failed_profile.name or not prof.is_available(now):
                continue
            prof_fam = PROVIDER_FAMILY_MAP.get(prof.provider, prof.provider)
            if prof_fam == wanted_fam:
                return prof
        return None

    def select_cross_family_profile(self, exclude_family: str, exclude_profile: str | None = None) -> AuthProfile | None:
        """Selects an authenticated profile from a DIFFERENT model family for blind judging."""
        for prof in self._profiles.values():
            if prof.name == exclude_profile or not prof.is_available():
                continue
            fam = PROVIDER_FAMILY_MAP.get(prof.provider, prof.provider)
            if fam != exclude_family:
                return prof
        return None

    def switch_profile(self, name: str) -> AuthProfile:
        if name not in self._profiles:
            raise KeyError(f"Unknown auth profile '{name}'. Available: {', '.join(self._profiles) or 'none'}")
        self._active_profile = name
        self._save_profiles()
        return self._profiles[name]

    def mark_rate_limited(self, name: str, cooldown_seconds: float = 45.0) -> AuthProfile | None:
        with self._lock:
            failed = self._profiles.get(name)
            if failed is not None:
                failed.cooldown_until = time.time() + max(1.0, cooldown_seconds)
                alt = self.select_failover_profile(failed)
                if alt is not None:
                    return alt
            return self.get_active_profile()

    def invalidate_token_cache(self, profile: AuthProfile | None = None) -> None:
        """Clears cached OAuth/ADC/command tokens after an HTTP 401 so the next retry fetches a fresh token."""
        with self._lock:
            self._gcloud_token_cache = None
            if profile is not None:
                self._cmd_token_cache.pop(profile.name, None)
            else:
                self._cmd_token_cache.clear()
            if profile and profile.auth_kind == "oauth":
                secret = self.load_secret(profile.name)
                if secret and secret.refresh_token and secret.token_url:
                    secret.expires_at = 1.0
                    self.store_secret(profile.name, secret)

    def register_api_key_profile(
        self,
        name: str,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        default_model: str | None = None,
        *,
        auth_kind: str = "api_key",
        api_mode: str = "",
        service_tier: str = "",
        project_id: str | None = None,
        region: str = "global",
        extra_headers: dict[str, str] | None = None,
    ) -> AuthProfile:
        if auth_kind != "none" and not api_key.strip():
            raise ValueError("API key / token must not be empty.")
        backend_provider = (
            "openrouter"
            if provider in ("deepseek", "qwen", "kimi", "glm")
            else ("gemini-enterprise" if provider == "code-assist" else provider)
        )
        resolved_mode = api_mode or (
            "code_assist"
            if provider == "code-assist"
            else ("express" if backend_provider == "gemini-enterprise" and auth_kind == "api_key" else "")
        )
        url = (base_url or PROVIDER_DEFAULT_URLS.get(provider, PROVIDER_DEFAULT_URLS["openrouter"])).rstrip("/")
        src = "vault"
        if auth_kind != "none":
            src = self.store_secret(name, StoredSecret(access_token=api_key.strip()))
        prof = AuthProfile(
            name=name,
            provider=backend_provider,
            auth_kind=auth_kind,
            base_url=url,
            project_id=project_id,
            region=region,
            default_model=default_model,
            source=src,
            api_mode=resolved_mode,
            service_tier=service_tier,
            extra_headers=dict(extra_headers or {}),
        )
        self._profiles[name] = prof
        self._active_profile = name
        self._save_profiles()
        return prof

    def register_vertex_profile(
        self,
        name: str,
        project_id: str,
        region: str = "global",
        include_claude: bool = True,
        *,
        provider: str = "vertex",
        base_url: str = "https://aiplatform.googleapis.com",
        api_mode: str = "",
        service_tier: str = "",
    ) -> AuthProfile:
        prof = AuthProfile(
            name=name,
            provider=provider,
            auth_kind="gcloud_adc",
            base_url=base_url.rstrip("/"),
            project_id=project_id.strip() if project_id else None,
            region=region.strip() or "global",
            default_model="gemini-3.1-pro-preview",
            source="vault",
            api_mode=api_mode,
            service_tier=service_tier,
        )
        self._profiles[name] = prof
        if include_claude and project_id.strip():
            c_reg = region if region not in ("global", "us-central1") else "us-east5"
            self._profiles[f"{name}-claude"] = AuthProfile(
                name=f"{name}-claude",
                provider="vertex-anthropic",
                auth_kind="gcloud_adc",
                base_url="https://aiplatform.googleapis.com",
                project_id=project_id.strip(),
                region=c_reg,
                default_model="claude-sonnet-4@20250514",
                source="vault",
            )
        self._active_profile = name
        self._save_profiles()
        return prof

    def register_command_profile(
        self,
        name: str,
        provider: str,
        credential_command: str,
        base_url: str,
        *,
        default_model: str | None = None,
        project_id: str | None = None,
        region: str = "global",
        api_mode: str = "",
        service_tier: str = "",
        token_ttl_seconds: int = 1800,
        require_host_path: str = "",
        transport_plugin: str = "",
        extra_headers: dict[str, str] | None = None,
        persist_to_workstation_overlay: bool = False,
    ) -> AuthProfile:
        """Registers a host-local `auth_kind='command'` profile in `~/.ush/workstation.json` or `auth_profiles.json`."""
        cmd_clean = credential_command.strip()
        if not cmd_clean or not shlex.split(cmd_clean):
            raise ValueError("`credential_command` must not be empty.")
        prof = AuthProfile(
            name=name,
            provider=provider,
            auth_kind="command",
            base_url=base_url.rstrip("/"),
            project_id=project_id.strip() if project_id else None,
            region=region.strip() or "global",
            default_model=default_model,
            source="workstation" if persist_to_workstation_overlay else "vault",
            api_mode=api_mode,
            service_tier=service_tier,
            extra_headers=dict(extra_headers or {}),
            credential_command=cmd_clean,
            token_ttl_seconds=max(1, int(token_ttl_seconds)),
            require_host_path=require_host_path.strip(),
            transport_plugin=transport_plugin.strip(),
        )
        self._profiles[name] = prof
        self._active_profile = name
        if persist_to_workstation_overlay:
            existing_ws: dict[str, Any] = {"active_profile": name, "prefer_active": True, "profiles": []}
            if is_trusted_host_local_file(self.workstation_path):
                try:
                    loaded = json.loads(self.workstation_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        existing_ws = loaded
                except (OSError, json.JSONDecodeError):
                    pass
            kept = [
                p for p in (existing_ws.get("profiles") or [])
                if isinstance(p, dict) and p.get("name") != name
            ]
            kept.append(prof.to_metadata_dict())
            existing_ws["active_profile"] = name
            existing_ws["profiles"] = kept
            self._atomic_write_0600(self.workstation_path, json.dumps(existing_ws, indent=2))
        self._save_profiles()
        return prof

    def logout(self, name: str) -> bool:
        if name not in self._profiles:
            return False
        self.delete_secret(name)
        del self._profiles[name]
        if self._active_profile == name:
            self._active_profile = next(iter(self._profiles.keys()), None)
        self._save_profiles()
        return True

    def resolve_bearer_or_key(self, profile: AuthProfile) -> str:
        if profile.auth_kind == "none":
            if profile.provider == "litellm":
                return os.environ.get("LITELLM_API_KEY") or os.environ.get("LITELLM_MASTER_KEY") or ""
            return os.environ.get("USH_LOCAL_API_KEY", "")
        if profile.source.startswith("env:"):
            env_key = profile.source.split(":", 1)[1]
            return os.environ.get(env_key, "")
        if profile.auth_kind == "gcloud_adc":
            return self._get_gcloud_access_token()
        if profile.auth_kind == "command":
            return self._resolve_command_token(profile)
        ambient = getattr(self, "_ambient_secrets", {}).get(profile.name)
        secret = ambient or self.load_secret(profile.name)
        if not secret:
            raise RuntimeError(f"Missing stored credentials for profile '{profile.name}'. Run `crux auth login`.")
        # Auto-refresh OAuth token 5 minutes prior to expiry (or when only a refresh_token is cached)
        if (
            profile.auth_kind == "oauth"
            and secret.refresh_token
            and secret.token_url
            and (not secret.access_token or secret.expires_at <= 0 or time.time() >= (secret.expires_at - 300))
        ):
            secret = self._refresh_oauth_token(profile.name, secret, persist=(ambient is None))
            if ambient is not None:
                self._ambient_secrets[profile.name] = secret
        return secret.access_token

    def _resolve_command_token(self, profile: AuthProfile) -> str:
        """Executes a user-owned host-local credential helper (`shell=False`) and caches the returned token."""
        if profile.source not in ("vault", "keychain", "workstation"):
            raise PermissionError(
                f"Refused `credential_command` on profile '{profile.name}': allowed strictly from `~/.ush/`."
            )
        cfg_file = self.workstation_path if profile.source == "workstation" else self.profiles_path
        if cfg_file.exists() and not is_trusted_host_local_file(cfg_file):
            raise PermissionError(
                f"Refused `credential_command` on profile '{profile.name}': `{cfg_file}` has unsafe permissions (`chmod 600` required)."
            )
        with self._lock:
            now = time.time()
            cached = self._cmd_token_cache.get(profile.name)
            if cached and now < cached[1]:
                return cached[0]

        cmd_raw = (profile.credential_command or "").strip()
        if not cmd_raw:
            raise RuntimeError(f"Profile '{profile.name}' has `auth_kind='command'` without `credential_command`.")
        argv = shlex.split(cmd_raw)
        if not argv:
            raise RuntimeError(f"Invalid `credential_command` for profile '{profile.name}'.")

        from ush.kernel.security import scrub_child_environment

        env = scrub_child_environment()
        for ws_env in ("KRB5CCNAME", "CLOUDSDK_CONFIG", "GOOGLE_APPLICATION_CREDENTIALS", "XDG_CONFIG_HOME"):
            if ws_env in os.environ:
                env[ws_env] = os.environ[ws_env]
        env.pop("ANTIGRAVITY_SOURCE_METADATA", None)
        self._ensure_home()

        try:
            res = subprocess.run(
                argv,
                cwd=str(self.ush_home),
                env=env,
                capture_output=True,
                text=True,
                timeout=12,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            raise RuntimeError(f"Credential helper failed for '{profile.name}': {type(exc).__name__}") from exc

        if res.returncode != 0:
            err_msg = (res.stderr or "").strip()[:240]
            raise RuntimeError(
                f"Credential helper for '{profile.name}' exited with status {res.returncode}: {err_msg}"
            )

        out = (res.stdout or "").strip()
        if not out:
            raise RuntimeError(f"Credential helper for '{profile.name}' returned empty output.")

        tok = ""
        ttl = max(1.0, float(profile.token_ttl_seconds))
        if out.startswith("{"):
            try:
                parsed = json.loads(out)
                if isinstance(parsed, dict):
                    status_obj = parsed.get("status") if isinstance(parsed.get("status"), dict) else {}
                    tok = str(
                        parsed.get("access_token")
                        or parsed.get("token")
                        or parsed.get("id_token")
                        or parsed.get("SessionToken")
                        or (status_obj or {}).get("token")
                        or ""
                    ).strip()
                    if "expires_in" in parsed:
                        exp_in = float(parsed["expires_in"])
                        ttl = max(1.0, exp_in - min(60.0, exp_in * 0.1))
                    elif "expires_at" in parsed:
                        rem = float(parsed["expires_at"]) - time.time()
                        ttl = max(1.0, rem - min(60.0, max(0.0, rem * 0.1)))
            except (json.JSONDecodeError, TypeError, ValueError):
                tok = ""
        if not tok:
            tok = out.splitlines()[0].strip()

        with self._lock:
            self._cmd_token_cache[profile.name] = (tok, time.time() + ttl)
        return tok

    def _try_refresh_adc_file_direct(self) -> tuple[str, float] | None:
        """Refreshes `authorized_user` ADC JSON directly over TLS without spawning a `gcloud` subprocess."""
        adc_file = pathlib.Path(
            os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS",
                "~/.config/gcloud/application_default_credentials.json",
            )
        ).expanduser()
        if not adc_file.is_file():
            return None
        try:
            data = json.loads(adc_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or data.get("type") != "authorized_user":
            return None
        cid = str(data.get("client_id") or "").strip()
        csec = str(data.get("client_secret") or "").strip()
        rtok = str(data.get("refresh_token") or "").strip()
        tok_uri = str(data.get("token_uri") or "https://oauth2.googleapis.com/token").strip()
        if not (cid and csec and rtok):
            return None
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": cid,
                "client_secret": csec,
                "refresh_token": rtok,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            tok_uri,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, context=secure_ssl_context(), timeout=10) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            tok = str(payload.get("access_token") or "").strip()
            exp_in = float(payload.get("expires_in", 3600))
            if tok:
                return tok, time.time() + max(60.0, exp_in - 300.0)
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
            return None
        return None

    def _get_gcloud_access_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._gcloud_token_cache and now < self._gcloud_token_cache[1]:
                return self._gcloud_token_cache[0]

            # Fast path: direct HTTPS OAuth refresh from ADC JSON (zero child-process overhead/hang risk)
            direct = self._try_refresh_adc_file_direct()
            if direct is not None:
                self._gcloud_token_cache = direct
                return direct[0]

            # Fallback path: `gcloud` CLI wrapped against hangs (`TimeoutExpired`) or broken Python envs (`OSError`)
            gcloud_bin = shutil.which("gcloud") or str(
                pathlib.Path("~/google-cloud-sdk/bin/gcloud").expanduser()
            )
            for args in (
                [gcloud_bin, "auth", "application-default", "print-access-token"],
                [gcloud_bin, "auth", "print-access-token"],
            ):
                if not pathlib.Path(args[0]).is_file():
                    continue
                try:
                    res = subprocess.run(args, capture_output=True, text=True, timeout=12)
                except (subprocess.SubprocessError, OSError):
                    continue
                tok = res.stdout.strip()
                if res.returncode == 0 and tok:
                    self._gcloud_token_cache = (tok, time.time() + 3000)
                    return tok
            raise RuntimeError(
                "Google Cloud credentials unavailable or expired. Run `gcloud auth application-default login`."
            )

    def _refresh_oauth_token(self, profile_name: str, secret: StoredSecret, *, persist: bool = True) -> StoredSecret:
        params: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": secret.refresh_token,
            "client_id": secret.client_id,
        }
        if secret.client_secret:
            params["client_secret"] = secret.client_secret
        body = urllib.parse.urlencode(params).encode("utf-8")
        req = urllib.request.Request(
            secret.token_url,
            data=body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )
        with urllib.request.urlopen(req, context=secure_ssl_context(), timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        updated = StoredSecret(
            access_token=str(payload["access_token"]),
            refresh_token=str(payload.get("refresh_token") or secret.refresh_token),
            expires_at=time.time() + float(payload.get("expires_in", 3600)),
            client_id=secret.client_id,
            client_secret=secret.client_secret,
            token_url=secret.token_url,
        )
        if persist:
            self.store_secret(profile_name, updated)
        return updated


# ---- Official OAuth 2.0 PKCE Loopback (127.0.0.1 ONLY) & RFC 8628 Device Flow ----


def generate_pkce_pair() -> tuple[str, str]:
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def run_oauth_pkce_login(
    vault: CredentialVault,
    *,
    profile_name: str,
    provider: str,
    authorize_url: str,
    token_url: str,
    client_id: str,
    scopes: list[str],
    base_url: str,
    no_browser: bool = False,
    client_secret: str = "",
    extra_authorize_params: dict[str, str] | None = None,
    redirect_path: str = "/oauth/callback",
    no_browser_redirect_uri: str = "",
    api_mode: str = "",
    project_id: str | None = None,
    default_model: str | None = None,
) -> AuthProfile:
    """Runs OAuth 2.0 Authorization Code + PKCE bound strictly to 127.0.0.1."""
    verifier, challenge = generate_pkce_pair()
    expected_state = secrets.token_urlsafe(24)
    received: dict[str, str] = {}
    clean_redirect_path = redirect_path if redirect_path.startswith("/") else f"/{redirect_path}"

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path != clean_redirect_path:
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy", "default-src 'none'")
                self.end_headers()
                self.wfile.write(b"Not Found")
                return
            qs = urllib.parse.parse_qs(parsed.query)
            received["code"] = (qs.get("code") or [""])[0]
            received["state"] = (qs.get("state") or [""])[0]
            received["error"] = (qs.get("error") or [""])[0]
            received["done"] = "1"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'")
            self.end_headers()
            self.wfile.write(
                b"<!doctype html><html><body><h2>Crucible (crux) Authentication Complete</h2>"
                b"<p>You may close this tab and return to your terminal.</p></body></html>"
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return  # Never log query params containing auth codes

    is_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    use_manual = bool(no_browser or is_ssh or os.environ.get("NO_BROWSER") in ("1", "true", "TRUE"))

    server: HTTPServer | None = None
    if use_manual and no_browser_redirect_uri:
        redirect_uri = no_browser_redirect_uri
    else:
        # Strictly bind 127.0.0.1 (never 0.0.0.0)
        server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
        port = server.server_address[1]
        redirect_uri = f"http://127.0.0.1:{port}{clean_redirect_path}"

    auth_params: dict[str, str] = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": expected_state,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    if extra_authorize_params:
        auth_params.update(extra_authorize_params)
    full_auth_url = f"{authorize_url}?{urllib.parse.urlencode(auth_params)}"

    if use_manual:
        if server is not None:
            server.server_close()
        print(f"\nOpen this URL in any browser:\n  {full_auth_url}\n")
        pasted = input("Paste the authorization code (or final redirected callback URL): ").strip()
        if pasted.startswith("http"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
            code = (qs.get("code") or [""])[0]
            state = (qs.get("state") or [""])[0]
            if state != expected_state:
                raise RuntimeError("OAuth state mismatch — login aborted.")
        else:
            code = pasted
    else:
        assert server is not None
        print(f"Opening browser for OAuth PKCE sign-in (fallback URL: {full_auth_url})...")
        webbrowser.open(full_auth_url)
        deadline = time.monotonic() + 180.0
        try:
            while "done" not in received and time.monotonic() < deadline:
                server.timeout = max(0.5, min(5.0, deadline - time.monotonic()))
                server.handle_request()
        finally:
            server.server_close()
        if received.get("error"):
            raise RuntimeError(f"OAuth provider returned error: {received['error']}")
        if received.get("state") != expected_state:
            raise RuntimeError("OAuth CSRF state mismatch — login aborted.")
        code = received.get("code", "")

    if not code:
        raise RuntimeError("No authorization code received.")

    token_params: dict[str, str] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    if client_secret:
        token_params["client_secret"] = client_secret
    token_body = urllib.parse.urlencode(token_params).encode("utf-8")
    req = urllib.request.Request(
        token_url,
        data=token_body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=secure_ssl_context(), timeout=20) as resp:
        tok = json.loads(resp.read().decode("utf-8"))

    src = vault.store_secret(
        profile_name,
        StoredSecret(
            access_token=str(tok["access_token"]),
            refresh_token=str(tok.get("refresh_token", "")),
            expires_at=time.time() + float(tok.get("expires_in", 3600)),
            client_id=client_id,
            client_secret=client_secret,
            token_url=token_url,
        ),
    )
    prof = AuthProfile(
        name=profile_name,
        provider=provider,
        auth_kind="oauth",
        base_url=base_url.rstrip("/"),
        project_id=project_id,
        default_model=default_model,
        source=src,
        api_mode=api_mode,
    )
    vault._profiles[profile_name] = prof
    vault.switch_profile(profile_name)
    return prof


GOOGLE_OAUTH_CLIENT_VAULT_KEY = "__google_oauth_client__"
GOOGLE_CODE_ASSIST_SCOPES: list[str] = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]


@dataclass
class CodeAssistUserSetup:
    """Result of Code Assist `:loadCodeAssist` / `:onboardUser` tier & companion-project provisioning."""

    project_id: str = ""
    tier_id: str = ""
    tier_name: str = ""
    has_paid_tier: bool = False
    google_one_credits: int = 0
    ineligible_reasons: list[str] = field(default_factory=list)
    validation_url: str = ""
    project_id_required: bool = False


def resolve_google_oauth_client_credentials(vault: CredentialVault) -> tuple[str, str]:
    """Resolves Desktop OAuth `(client_id, client_secret)` without hardcoding credentials in source.

    Checks in priority order:
    1. `USH_GOOGLE_OAUTH_CLIENT_ID` / `GOOGLE_OAUTH_CLIENT_ID` (+ `_SECRET`) environment variables.
    2. OS Keychain / `0600` vault entry `__google_oauth_client__`.
    3. Trusted `0600` `~/.ush/workstation.json` (`{"google_oauth": {"client_id": ..., "client_secret": ...}}`).
    4. Locally installed Apache-2.0 `@google/gemini-cli` (`gemini` in `PATH`) if present on the machine.
    """
    cid = (
        os.environ.get("USH_GOOGLE_OAUTH_CLIENT_ID")
        or os.environ.get("GOOGLE_OAUTH_CLIENT_ID")
        or ""
    ).strip()
    csec = (
        os.environ.get("USH_GOOGLE_OAUTH_CLIENT_SECRET")
        or os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET")
        or ""
    ).strip()
    if cid:
        return cid, csec

    stored = vault.load_secret(GOOGLE_OAUTH_CLIENT_VAULT_KEY)
    if stored and stored.client_id:
        return stored.client_id.strip(), stored.client_secret.strip()

    if is_trusted_host_local_file(vault.workstation_path):
        try:
            ws = json.loads(vault.workstation_path.read_text(encoding="utf-8"))
            goauth = ws.get("google_oauth") if isinstance(ws, dict) else None
            if isinstance(goauth, dict) and goauth.get("client_id"):
                return str(goauth["client_id"]).strip(), str(goauth.get("client_secret") or "").strip()
        except (OSError, json.JSONDecodeError):
            pass

    gemini_bin = shutil.which("gemini")
    if gemini_bin:
        try:
            real_bin = pathlib.Path(gemini_bin).resolve()
            candidates: list[pathlib.Path] = []
            for anc in list(real_bin.parents)[:5]:
                candidates.append(
                    anc / "node_modules" / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist" / "oauth2.js"
                )
                candidates.append(
                    anc / "@google" / "gemini-cli-core" / "dist" / "src" / "code_assist" / "oauth2.js"
                )
            for js_path in candidates:
                if js_path.is_file():
                    src = js_path.read_text(encoding="utf-8", errors="replace")
                    m_id = re.search(r"OAUTH_CLIENT_ID\s*=\s*['\"]([^'\"]+\.apps\.googleusercontent\.com)['\"]", src)
                    m_sec = re.search(r"OAUTH_CLIENT_SECRET\s*=\s*['\"]([^'\"]+)['\"]", src)
                    if m_id:
                        return m_id.group(1).strip(), (m_sec.group(1).strip() if m_sec else "")
        except OSError:
            pass

    return "", ""


def fetch_google_userinfo_email(access_token: str, timeout: float = 5.0) -> str:
    """Fetches the authenticated Google account email from `userinfo` (non-fatal on error)."""
    if not access_token:
        return ""
    req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}", "Accept": "application/json"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, context=secure_ssl_context(), timeout=timeout) as resp:
            data = json.loads(resp.read(32_768).decode("utf-8", errors="replace"))
        return str(data.get("email") or "").strip() if isinstance(data, dict) else ""
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return ""


def _extract_companion_project_id(raw_val: Any) -> str:
    if isinstance(raw_val, str):
        return raw_val.strip()
    if isinstance(raw_val, dict):
        return str(raw_val.get("id") or "").strip()
    return ""


def _sum_google_one_credits(tier_obj: dict[str, Any] | None) -> int:
    if not isinstance(tier_obj, dict):
        return 0
    total = 0
    for item in tier_obj.get("availableCredits") or []:
        if isinstance(item, dict) and item.get("creditType") == "GOOGLE_ONE_AI":
            try:
                total += int(item.get("creditAmount") or 0)
            except (TypeError, ValueError):
                pass
    return total


def setup_code_assist_user(
    access_token: str,
    *,
    base_url: str = "https://cloudcode-pa.googleapis.com",
    api_version: str = "v1internal",
    explicit_project_id: str | None = None,
    extra_headers: dict[str, str] | None = None,
    timeout: float = 15.0,
    poll_interval_seconds: float = 2.0,
    max_poll_attempts: int = 20,
    sleep_fn: Any = time.sleep,
) -> CodeAssistUserSetup:
    """Provisions or resolves a Google-managed `cloudaicompanionProject` and tier (`Free` / `Pro` / `Ultra` / `Standard`).

    Implements the Code Assist `:loadCodeAssist` + `:onboardUser` + LRO polling handshake without requiring a user GCP project
    for eligible individual / Google AI Pro / Google AI Ultra accounts.
    """
    base_clean = base_url.rstrip("/")
    ver_clean = api_version.strip("/") or "v1internal"
    proj_in = (explicit_project_id or "").strip() or None
    if proj_in and proj_in.isdigit():
        raise ValueError(
            f"Invalid numeric Google Cloud Project ID '{proj_in}': provide the string Project ID, not project number."
        )

    core_meta: dict[str, Any] = {
        "ideType": "IDE_UNSPECIFIED",
        "platform": "PLATFORM_UNSPECIFIED",
        "pluginType": "GEMINI",
    }
    load_meta = {**core_meta, "duetProject": proj_in} if proj_in else dict(core_meta)
    load_payload: dict[str, Any] = {"metadata": load_meta}
    if proj_in:
        load_payload["cloudaicompanionProject"] = proj_in

    hdrs: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "crux-cli/0.1",
        **dict(extra_headers or {}),
    }
    if access_token:
        hdrs["Authorization"] = f"Bearer {access_token}"

    ctx = secure_ssl_context() if base_clean.startswith("https://") else None

    def _post_json(method_name: str, body_dict: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{base_clean}/{ver_clean}:{method_name}",
            data=json.dumps(body_dict).encode("utf-8"),
            headers=hdrs,
            method="POST",
        )
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read(131_072)
        parsed = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
        return parsed if isinstance(parsed, dict) else {}

    def _get_operation(op_name: str) -> dict[str, Any]:
        clean_op = op_name.lstrip("/")
        if not clean_op or ".." in clean_op or not re.fullmatch(r"[A-Za-z0-9_./-]+", clean_op):
            raise ValueError(f"Refused malformed Code Assist operation name: {op_name!r}")
        req = urllib.request.Request(
            f"{base_clean}/{ver_clean}/{clean_op}",
            headers=hdrs,
            method="GET",
        )
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            raw = resp.read(131_072)
        parsed = json.loads(raw.decode("utf-8", errors="replace")) if raw else {}
        return parsed if isinstance(parsed, dict) else {}

    load_res = _post_json("loadCodeAssist", load_payload)

    ineligible_list = [x for x in (load_res.get("ineligibleTiers") or []) if isinstance(x, dict)]
    reasons = [
        f"{it.get('tierId') or it.get('tierName') or 'tier'}: {it.get('reasonCode') or ''} {it.get('reasonMessage') or ''}".strip()
        for it in ineligible_list
        if it.get("reasonMessage") or it.get("reasonCode")
    ]
    val_tier = next(
        (
            it
            for it in ineligible_list
            if it.get("reasonCode") == "VALIDATION_REQUIRED" and it.get("validationUrl")
        ),
        None,
    )
    validation_url = str(val_tier.get("validationUrl") or "").strip() if val_tier else ""

    current_tier = load_res.get("currentTier") if isinstance(load_res.get("currentTier"), dict) else None
    paid_tier = load_res.get("paidTier") if isinstance(load_res.get("paidTier"), dict) else None
    direct_companion = _extract_companion_project_id(load_res.get("cloudaicompanionProject"))
    allowed = [t for t in (load_res.get("allowedTiers") or []) if isinstance(t, dict)]

    # Return immediately only when `currentTier` is already active (or when minimal server response omits `allowedTiers`)
    if current_tier is not None or (direct_companion and not allowed):
        companion_proj = direct_companion or (proj_in or "")
        tier_id = str((paid_tier or {}).get("id") or (current_tier or {}).get("id") or "standard-tier")
        tier_name = str((paid_tier or {}).get("name") or (current_tier or {}).get("name") or tier_id)
        credits = _sum_google_one_credits(paid_tier) or _sum_google_one_credits(current_tier)
        return CodeAssistUserSetup(
            project_id=companion_proj,
            tier_id=tier_id,
            tier_name=tier_name,
            has_paid_tier=paid_tier is not None,
            google_one_credits=credits,
            ineligible_reasons=reasons,
            validation_url=validation_url,
            project_id_required=not bool(companion_proj),
        )

    if validation_url:
        return CodeAssistUserSetup(
            ineligible_reasons=reasons,
            validation_url=validation_url,
            project_id_required=True,
        )

    chosen_tier = next((t for t in allowed if t.get("isDefault")), None) or (
        allowed[0] if allowed else {"id": "legacy-tier", "userDefinedCloudaicompanionProject": True}
    )
    tier_id = str(chosen_tier.get("id") or "standard-tier")
    tier_name = str(chosen_tier.get("name") or (paid_tier or {}).get("name") or tier_id)

    # Managed/free tiers forbid passing `cloudaicompanionProject` in `:onboardUser` (`Precondition Failed`),
    # whereas standard/managed-org tiers with a companion or explicit project MUST pass `onboard_proj` to activate the license.
    onboard_proj = proj_in or direct_companion or None
    user_defined_proj = bool(chosen_tier.get("userDefinedCloudaicompanionProject")) or bool(direct_companion)
    onboard_body: dict[str, Any] = {"tierId": tier_id}
    if tier_id != "free-tier" and onboard_proj and user_defined_proj:
        onboard_body["cloudaicompanionProject"] = onboard_proj
        onboard_body["metadata"] = {**core_meta, "duetProject": onboard_proj}
    else:
        onboard_body["metadata"] = dict(core_meta)

    lro = _post_json("onboardUser", onboard_body)
    attempts = 0
    while not lro.get("done") and lro.get("name") and attempts < max_poll_attempts:
        sleep_fn(poll_interval_seconds)
        lro = _get_operation(str(lro["name"]))
        attempts += 1

    resp_obj = lro.get("response") if isinstance(lro.get("response"), dict) else {}
    provisioned_proj = _extract_companion_project_id(resp_obj.get("cloudaicompanionProject")) or (onboard_proj or "")
    credits = _sum_google_one_credits(paid_tier) or _sum_google_one_credits(chosen_tier)
    return CodeAssistUserSetup(
        project_id=provisioned_proj,
        tier_id=str((paid_tier or {}).get("id") or tier_id),
        tier_name=str((paid_tier or {}).get("name") or tier_name),
        has_paid_tier=paid_tier is not None,
        google_one_credits=credits,
        ineligible_reasons=reasons,
        validation_url=validation_url,
        project_id_required=not bool(provisioned_proj),
    )


def login_google_code_assist_oauth(
    vault: CredentialVault,
    *,
    profile_name: str = "antigravity",
    provider: str = "antigravity",
    base_url: str = "https://cloudcode-pa.googleapis.com",
    default_model: str = "gemini-3.1-pro-preview",
    explicit_project_id: str | None = None,
    no_browser: bool = False,
) -> AuthProfile:
    """Browser OAuth PKCE login for Antigravity / Google AI Pro & Ultra (`cloudcode-pa`, zero GCP project required)."""
    client_id, client_secret = resolve_google_oauth_client_credentials(vault)
    if not client_id:
        print(
            "\nConfigure Desktop OAuth Client for Google OAuth (`cloudcode-pa.googleapis.com`).\n"
            "Tip: set `USH_GOOGLE_OAUTH_CLIENT_ID` / `USH_GOOGLE_OAUTH_CLIENT_SECRET`, install `@google/gemini-cli`,\n"
            "or enter a Google Desktop App OAuth Client ID once (stored encrypted in your OS Keychain)."
        )
        client_id = input("Google Desktop OAuth Client ID: ").strip()
        if not client_id:
            raise ValueError("Google OAuth Client ID is required for Google OAuth login.")
        client_secret = getpass.getpass("Google Desktop OAuth Client Secret (optional if PKCE-only): ").strip()
        vault.store_secret(
            GOOGLE_OAUTH_CLIENT_VAULT_KEY,
            StoredSecret(client_id=client_id, client_secret=client_secret),
        )

    prof = run_oauth_pkce_login(
        vault,
        profile_name=profile_name,
        provider=provider,
        authorize_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        client_id=client_id,
        client_secret=client_secret,
        scopes=GOOGLE_CODE_ASSIST_SCOPES,
        base_url=base_url,
        no_browser=no_browser,
        extra_authorize_params={"access_type": "offline", "prompt": "consent"},
        redirect_path="/oauth2callback",
        no_browser_redirect_uri="https://codeassist.google.com/authcode",
        api_mode="code_assist",
        project_id=(explicit_project_id or "").strip() or None,
        default_model=default_model,
    )

    tok = vault.resolve_bearer_or_key(prof)
    email = fetch_google_userinfo_email(tok)
    if email:
        print(f"Signed into Google OAuth as `{email}`.")

    setup = setup_code_assist_user(
        tok,
        base_url=base_url,
        explicit_project_id=prof.project_id,
    )
    if setup.validation_url and sys.stdin.isatty():
        print(f"\nAccount verification required: {setup.validation_url}")
        if setup.ineligible_reasons:
            print(f"  Details: {'; '.join(setup.ineligible_reasons)}")
        input("Complete verification in your browser and press Enter to retry (or Ctrl-C to cancel)...")
        setup = setup_code_assist_user(tok, base_url=base_url, explicit_project_id=prof.project_id)

    if setup.project_id_required and not setup.project_id:
        reason_str = "; ".join(setup.ineligible_reasons) or "Account requires an explicit GOOGLE_CLOUD_PROJECT."
        print(
            f"\nNote: `{email or 'This Google account'}` did not receive a Google-managed companion project "
            f"({reason_str}).\n"
            "  • Personal `@gmail.com` accounts with Google AI Ultra / Pro / Free receive a managed project automatically.\n"
            "  • Corporate Workspace / `@google.com` (`DASHER_USER`) accounts require either an admin-assigned companion\n"
            "    project or an explicit GCP Project ID."
        )
        if sys.stdin.isatty():
            fallback_proj = input(
                "Optional GCP Project ID for this corp account (or press Enter to leave unset / switch account): "
            ).strip()
            if fallback_proj:
                setup = setup_code_assist_user(tok, base_url=base_url, explicit_project_id=fallback_proj)

    if setup.project_id:
        prof.project_id = setup.project_id
        vault._profiles[prof.name] = prof
        vault._save_profiles()
        credits_note = f" • Google One AI Credits: {setup.google_one_credits}" if setup.google_one_credits else ""
        label = "Antigravity" if provider == "antigravity" else provider
        print(
            f"{label} ready — Tier: `{setup.tier_name or setup.tier_id}` "
            f"• Companion Project: `{setup.project_id}`{credits_note}"
        )

    return prof


def run_oauth_device_code_login(
    vault: CredentialVault,
    *,
    profile_name: str,
    provider: str,
    device_code_url: str,
    token_url: str,
    client_id: str,
    scopes: list[str],
    base_url: str,
    api_mode: str = "",
) -> AuthProfile:
    """Runs standard RFC 8628 OAuth 2.0 Device Authorization Grant for headless / SSH terminals."""
    req_body = urllib.parse.urlencode(
        {"client_id": client_id, "scope": " ".join(scopes)}
    ).encode("utf-8")
    req = urllib.request.Request(
        device_code_url,
        data=req_body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=secure_ssl_context(), timeout=20) as resp:
        dev = json.loads(resp.read().decode("utf-8"))

    device_code = str(dev["device_code"])
    user_code = str(dev.get("user_code", ""))
    verify_uri = str(dev.get("verification_uri_complete") or dev.get("verification_uri") or "")
    interval = max(1.0, float(dev.get("interval", 5)))
    expires_in = float(dev.get("expires_in", 600))

    print(f"\nAuthorize {provider} by visiting:\n  {verify_uri}")
    if user_code:
        print(f"Enter one-time code: {user_code}\n")

    deadline = time.time() + expires_in
    tok: dict[str, Any] | None = None
    while time.time() < deadline:
        time.sleep(interval)
        poll_body = urllib.parse.urlencode(
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "device_code": device_code,
                "client_id": client_id,
            }
        ).encode("utf-8")
        poll_req = urllib.request.Request(
            token_url,
            data=poll_body,
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(poll_req, context=secure_ssl_context(), timeout=20) as presp:
                data = json.loads(presp.read().decode("utf-8"))
        except urllib.error.HTTPError as hexc:
            try:
                data = json.loads(hexc.read().decode("utf-8"))
            except (OSError, json.JSONDecodeError):
                raise hexc from None
        err = str(data.get("error") or "")
        if not err and data.get("access_token"):
            tok = data
            break
        if err == "authorization_pending":
            continue
        if err == "slow_down":
            interval += 5.0
            continue
        raise RuntimeError(f"OAuth device flow failed: {err or data}")

    if not tok:
        raise RuntimeError("OAuth device authorization timed out.")

    src = vault.store_secret(
        profile_name,
        StoredSecret(
            access_token=str(tok["access_token"]),
            refresh_token=str(tok.get("refresh_token", "")),
            expires_at=time.time() + float(tok.get("expires_in", 3600)),
            client_id=client_id,
            token_url=token_url,
        ),
    )
    prof = AuthProfile(
        name=profile_name,
        provider=provider,
        auth_kind="oauth",
        base_url=base_url.rstrip("/"),
        source=src,
        api_mode=api_mode,
    )
    vault._profiles[profile_name] = prof
    vault.switch_profile(profile_name)
    return prof


def fetch_openai_compat_models(
    base_url: str,
    token: str = "",
    extra_headers: dict[str, str] | None = None,
    timeout: float = 2.5,
    *,
    api_key: str = "",
) -> list[str]:
    """Queries `GET {base_url}/models` on LiteLLM / OpenAI / Ollama / vLLM endpoints; returns `[]` on failure."""
    eff_tok = token or api_key
    parsed = urllib.parse.urlparse(base_url.rstrip("/"))
    url = f"{base_url.rstrip('/')}/v1/models" if parsed.path in ("", "/") else f"{base_url.rstrip('/')}/models"
    hdrs: dict[str, str] = {"Accept": "application/json", "User-Agent": "crux-cli/0.1"}
    if eff_tok:
        hdrs["Authorization"] = f"Bearer {eff_tok}"
        hdrs["x-litellm-api-key"] = eff_tok
    if extra_headers:
        hdrs.update(extra_headers)
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        ctx = secure_ssl_context() if url.startswith("https://") else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            payload = json.loads(resp.read(262_144).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return []
    models_raw = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(models_raw, list):
        return []
    out: list[str] = []
    for item in models_raw:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
        elif isinstance(item, str):
            out.append(item)
    return sorted(dict.fromkeys(out))


def fetch_code_assist_quota_models(
    base_url: str = "https://cloudcode-pa.googleapis.com",
    *,
    project_id: str = "",
    token: str = "",
    api_version: str = "v1internal",
    timeout: float = 1.2,
) -> list[tuple[str, float | None]]:
    """Queries `POST /{api_version}:retrieveUserQuota` on `cloudcode-pa` to discover live models & remaining quota."""
    if not token or not project_id:
        return []
    url = f"{base_url.rstrip('/')}/{api_version.strip('/')}:retrieveUserQuota"
    hdrs = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "crux-cli/0.1",
    }
    req = urllib.request.Request(
        url,
        data=json.dumps({"project": project_id}).encode("utf-8"),
        headers=hdrs,
        method="POST",
    )
    try:
        ctx = secure_ssl_context() if url.startswith("https://") else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            payload = json.loads(resp.read(131_072).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return []
    buckets = payload.get("buckets") if isinstance(payload, dict) else None
    if not isinstance(buckets, list):
        return []
    found: dict[str, float | None] = {}
    for b in buckets:
        if not isinstance(b, dict):
            continue
        mid = str(b.get("modelId") or "").strip()
        if not mid or mid == "none":
            continue
        # Only auto-surface active Gemini/Gemma buckets (or buckets carrying an active quota reset window)
        if not (mid.startswith(("gemini-", "gemma-")) and b.get("resetTime")):
            continue
        frac_raw = b.get("remainingFraction")
        frac = float(frac_raw) if isinstance(frac_raw, (int, float)) else None
        if mid not in found or (frac is not None and (found[mid] is None or frac < found[mid])):  # type: ignore[operator]
            found[mid] = frac

    priority = {
        "gemini-3.1-pro-preview": 0,
        "gemini-3.8-flash": 1,
        "gemini-3.5-flash": 2,
        "gemini-3-pro-preview": 3,
        "gemini-3-flash-preview": 4,
        "gemini-3-flash": 5,
        "gemini-2.5-pro": 6,
        "gemini-2.5-flash": 7,
    }
    return sorted(found.items(), key=lambda kv: (priority.get(kv[0], 50), kv[0]))


def fetch_gemini_api_models(
    base_url: str = "https://generativelanguage.googleapis.com",
    *,
    token: str = "",
    auth_kind: str = "api_key",
    timeout: float = 1.2,
) -> list[str]:
    """Queries `GET {base_url}/v1beta/models` to discover available Gemini models without provider labels."""
    if not token:
        return []
    url = f"{base_url.rstrip('/')}/v1beta/models?pageSize=100"
    hdrs = {"Accept": "application/json", "User-Agent": "crux-cli/0.1"}
    if auth_kind in ("bearer", "oauth", "gcloud_adc"):
        hdrs["Authorization"] = f"Bearer {token}"
    else:
        hdrs["x-goog-api-key"] = token
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        ctx = secure_ssl_context() if url.startswith("https://") else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            payload = json.loads(resp.read(262_144).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return []
    models_raw = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models_raw, list):
        return []
    out: list[str] = []
    for item in models_raw:
        if not isinstance(item, dict):
            continue
        methods = item.get("supportedGenerationMethods") or []
        if methods and "generateContent" not in methods:
            continue
        raw_name = str(item.get("name") or "").strip()
        mid = raw_name[len("models/") :] if raw_name.startswith("models/") else raw_name
        if mid and mid.startswith(("gemini-", "gemma-")) and "embedding" not in mid and "tts" not in mid:
            out.append(mid)
    return list(dict.fromkeys(out))


def fetch_anthropic_models(
    base_url: str = "https://api.anthropic.com",
    *,
    token: str = "",
    auth_kind: str = "api_key",
    timeout: float = 1.2,
) -> list[tuple[str, str]]:
    """Queries `GET {base_url}/v1/models` to discover available Anthropic models."""
    if not token:
        return []
    base = base_url.rstrip("/")
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    hdrs = {
        "Accept": "application/json",
        "anthropic-version": "2023-06-01",
        "User-Agent": "crux-cli/0.1",
    }
    if auth_kind in ("bearer", "oauth") or (not token.startswith("sk-ant-api") and auth_kind != "api_key"):
        hdrs["Authorization"] = f"Bearer {token}"
    else:
        hdrs["x-api-key"] = token
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        ctx = secure_ssl_context() if url.startswith("https://") else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            payload = json.loads(resp.read(262_144).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return []
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    out: list[tuple[str, str]] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            out.append((str(item["id"]), str(item.get("display_name") or "")))
    return out


_MODEL_DESCRIPTIONS: dict[str, str] = {
    "gemini-4-argon": "Deep reasoning · 1M context",
    "gemini-3.8-flash": "Latest fast reasoning · 1M context",
    "gemini-3.7-flash": "Fast reasoning · 1M context",
    "gemini-3.6-flash": "Fast reasoning · 1M context",
    "gemini-3.5-flash": "Balanced reasoning · 1M context",
    "gemini-3.1-pro": "Deep reasoning · 1M context",
    "gemini-3.1-pro-preview": "Deep reasoning · 1M context",
    "gemini-3-pro-preview": "Deep reasoning · 1M context",
    "gemini-3-flash-preview": "Fast reasoning preview · 1M context",
    "gemini-3-flash": "Fast reasoning · 1M context",
    "gemini-3.1-flash-lite": "Lightweight low-latency · 1M context",
    "gemini-3.1-flash-lite-preview": "Lightweight preview · 1M context",
    "gemini-2.5-pro": "Deep reasoning · 1M context",
    "gemini-2.5-flash": "Fast high-throughput · 1M context",
    "gemini-2.5-flash-lite": "Lightweight low-latency · 1M context",
    "claude-opus-4-20250514": "Deep interleaved reasoning · 200K context",
    "claude-opus-4@20250514": "Deep interleaved reasoning · 200K context",
    "claude-sonnet-4-20250514": "Balanced agentic coding · 200K context",
    "claude-sonnet-4@20250514": "Balanced agentic coding · 200K context",
    "claude-3-7-sonnet-latest": "Extended thinking · 200K context",
    "claude-3-7-sonnet@20250219": "Extended thinking · 200K context",
    "claude-3-5-haiku-latest": "Fast low-latency · 200K context",
    "gpt-5": "Deep reasoning · 400K context",
    "o3": "Multi-step reasoning · 200K context",
    "o4-mini": "Fast reasoning · 200K context",
    "gpt-4.1": "High-context coding · 1M context",
    "deepseek-reasoner": "Chain-of-thought reasoning · 128K context",
    "deepseek-chat": "Fast coding & general chat · 128K context",
    "qwen3-coder": "Agentic coding · 128K context",
}


@dataclass(frozen=True)
class DiscoveredModelOption:
    """Clean, provider-label-free runtime model entry for `/model`."""

    model_id: str
    display_name: str
    description: str


def format_model_display_name(model_id: str, server_name: str = "") -> str:
    """Formats a clean human-readable model label without provider tags."""
    clean_srv = server_name.strip()
    if clean_srv and not clean_srv.startswith("models/") and "/" not in clean_srv:
        return clean_srv
    low = model_id.strip().lower()
    if low.startswith("gemini-"):
        tokens = model_id.strip().split("-")
        pretty: list[str] = []
        for t in tokens:
            tl = t.lower()
            if tl == "gemini":
                pretty.append("Gemini")
            elif tl == "preview":
                pretty.append("(Preview)")
            elif tl in ("pro", "flash", "lite", "argon", "ultra", "nano"):
                pretty.append(tl.capitalize())
            else:
                pretty.append(t)
        return " ".join(pretty)
    return model_id.strip()


def describe_model(model_id: str, quota_fraction: float | None = None) -> str:
    """Returns a clean, provider-agnostic description for `model_id` (with optional remaining quota)."""
    base = _MODEL_DESCRIPTIONS.get(model_id, "")
    if not base:
        low = model_id.lower()
        if "flash-lite" in low:
            base = "Lightweight low-latency"
        elif "flash" in low or "mini" in low or "haiku" in low:
            base = "Fast reasoning"
        elif any(k in low for k in ("argon", "pro", "opus", "gpt-5", "o3", "reasoner", "r1")):
            base = "Deep reasoning"
        else:
            base = "Available model"
    if quota_fraction is not None:
        return f"{base} · {quota_fraction * 100:.0f}% remaining"
    return base


_AGY_DISCOVERY_CACHE_LOCK = threading.Lock()
_AGY_DISCOVERY_CACHE: dict[str, tuple[float, str, list[tuple[str, str]]]] = {}


def fetch_agy_models(
    ush_home: pathlib.Path,
    *,
    timeout: float = 5.0,
    ttl_seconds: float = 300.0,
) -> tuple[str, list[tuple[str, str]]]:
    """Queries live Antigravity models at runtime via `agy acp` / `agy models` (`(default_id, [(id, name)])`)."""
    from ush.brain.agy_acp import get_shared_agy_bridge, parse_agy_models_cli_output, resolve_agy_binary

    agy_bin = resolve_agy_binary()
    if not agy_bin:
        return "", []
    now = time.monotonic()
    with _AGY_DISCOVERY_CACHE_LOCK:
        cached = _AGY_DISCOVERY_CACHE.get(agy_bin)
        if cached is not None and (now - cached[0]) < ttl_seconds and cached[2]:
            return cached[1], list(cached[2])

    default_mid = ""
    pairs: list[tuple[str, str]] = []
    try:
        bridge = get_shared_agy_bridge(ush_home, agy_bin)
        default_mid, opts = bridge.discover_models(timeout=max(timeout, 4.5))
        pairs = [(o.model_id, o.display_name) for o in opts]
    except Exception:  # noqa: BLE001
        pairs = []

    if not pairs:
        try:
            env = dict(os.environ)
            env.pop("ANTIGRAVITY_SOURCE_METADATA", None)
            proc = subprocess.run(
                [agy_bin, "models"],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                timeout=max(timeout, 5.0),
                check=False,
                env=env,
            )
            if proc.returncode == 0 and proc.stdout:
                opts_cli = parse_agy_models_cli_output(proc.stdout)
                pairs = [(o.model_id, o.display_name) for o in opts_cli]
                if pairs and not default_mid:
                    default_mid = pairs[0][0]
        except (OSError, subprocess.SubprocessError):
            pairs = []

    if pairs:
        with _AGY_DISCOVERY_CACHE_LOCK:
            _AGY_DISCOVERY_CACHE[agy_bin] = (time.monotonic(), default_mid or pairs[0][0], pairs)
    return default_mid, pairs


def fetch_vertex_publisher_models(
    base_url: str = "https://aiplatform.googleapis.com",
    *,
    token: str = "",
    auth_kind: str = "gcloud_adc",
    timeout: float = 1.5,
) -> list[tuple[str, str]]:
    """Queries live `GET {base_url}/v1beta1/publishers/google/models` to discover available Vertex / Antigravity models."""
    if not token:
        return []
    host = (base_url or "https://aiplatform.googleapis.com").rstrip("/")
    url = f"{host}/v1beta1/publishers/google/models?pageSize=100"
    hdrs = {"Accept": "application/json", "User-Agent": "crux-cli/0.1"}
    if auth_kind == "api_key":
        hdrs["x-goog-api-key"] = token
    else:
        hdrs["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=hdrs, method="GET")
    try:
        ctx = secure_ssl_context() if url.startswith("https://") else None
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            payload = json.loads(resp.read(524_288).decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ValueError):
        return []
    raw_list = payload.get("publisherModels") if isinstance(payload, dict) else None
    if not isinstance(raw_list, list):
        return []

    skip_substrings = (
        "embedding",
        "-tts",
        "-transcribe",
        "-image",
        "-live",
        "gemini-1.0",
        "gemini-1.5",
        "gemini-experimental",
        "aqa",
        "imagen",
        "veo",
        "chirp",
        "medlm",
    )
    found: dict[str, str] = {}
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        raw_name = str(item.get("name") or "").strip()
        mid = raw_name.rsplit("/", 1)[-1] if "/" in raw_name else raw_name
        low = mid.lower()
        if not mid or not low.startswith(("gemini-", "gemma-")):
            continue
        if any(s in low for s in skip_substrings):
            continue
        disp = str(item.get("displayName") or "").strip()
        found[mid] = format_model_display_name(mid, disp)

    def _rank(mid: str) -> tuple[int, str]:
        low = mid.lower()
        if low.startswith("gemini-4"):
            return (0, mid)
        if low.startswith("gemini-3.8"):
            return (1, mid)
        if low.startswith("gemini-3.7"):
            return (2, mid)
        if low.startswith("gemini-3.6"):
            return (3, mid)
        if low.startswith("gemini-3.5"):
            return (4, mid)
        if low.startswith("gemini-3.1-pro"):
            return (5, mid)
        if low.startswith("gemini-3"):
            return (6, mid)
        if low.startswith("gemini-2.5-pro"):
            return (7, mid)
        if low.startswith("gemini-2.5"):
            return (8, mid)
        return (20, mid)

    ordered = sorted(found.items(), key=lambda kv: _rank(kv[0]))
    return ordered


def discover_model_options(
    vault: CredentialVault,
    prof: AuthProfile,
    *,
    timeout: float = 1.2,
) -> list[DiscoveredModelOption]:
    """Dynamically discovers live `(model_id, display_name, description)` for `prof` with zero provider labels."""
    base_clean = (prof.base_url or "").rstrip("/")

    # 0. Sanctioned local Antigravity CLI (`agy acp` / `agy models`) -> exact live Antigravity models!
    if prof.api_mode == "agy_acp" or base_clean.startswith("agy://"):
        _, agy_pairs = fetch_agy_models(vault.ush_home, timeout=max(timeout, 4.5))
        if agy_pairs:
            return [
                DiscoveredModelOption(
                    model_id=mid,
                    display_name=disp,
                    description=describe_model(mid),
                )
                for mid, disp in agy_pairs
            ]

    tok = ""
    if prof.auth_kind != "none":
        try:
            tok = vault.resolve_bearer_or_key(prof)
        except Exception:  # noqa: BLE001
            tok = ""

    is_code_assist = prof.api_mode == "code_assist" or "cloudcode-pa.googleapis.com" in base_clean

    # 1. Google OAuth / Code Assist (`cloudcode-pa.googleapis.com`) -> live `retrieveUserQuota`
    if is_code_assist:
        proj = prof.project_id or ""
        if tok and not proj:
            try:
                setup = setup_code_assist_user(tok, base_url=base_clean or "https://cloudcode-pa.googleapis.com", timeout=timeout)
                proj = setup.project_id
            except Exception:  # noqa: BLE001
                proj = ""
        live_q = fetch_code_assist_quota_models(
            base_clean or "https://cloudcode-pa.googleapis.com",
            project_id=proj,
            token=tok,
            timeout=timeout,
        )
        if live_q:
            return [
                DiscoveredModelOption(
                    model_id=mid,
                    display_name=format_model_display_name(mid),
                    description=describe_model(mid, frac),
                )
                for mid, frac in live_q
            ]
        fallback_gemini = (
            "gemini-3.1-pro-preview",
            "gemini-3.8-flash",
            "gemini-3.5-flash",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
            "gemini-2.5-pro",
            "gemini-2.5-flash",
        )
        return [
            DiscoveredModelOption(mid, format_model_display_name(mid), describe_model(mid))
            for mid in fallback_gemini
        ]

    # 2. Google AI Studio (`generativelanguage.googleapis.com`) -> live `/v1beta/models`
    if prof.provider == "google" or "generativelanguage.googleapis.com" in base_clean:
        live_g = fetch_gemini_api_models(base_clean or "https://generativelanguage.googleapis.com", token=tok, auth_kind=prof.auth_kind, timeout=timeout)
        if live_g:
            return [
                DiscoveredModelOption(mid, format_model_display_name(mid), describe_model(mid))
                for mid in live_g[:20]
            ]
        return [
            DiscoveredModelOption(mid, format_model_display_name(mid), describe_model(mid))
            for mid in ("gemini-3.1-pro-preview", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite")
        ]

    # 3. Anthropic (`api.anthropic.com`) -> live `/v1/models`
    if prof.provider == "anthropic":
        live_a = fetch_anthropic_models(base_clean or "https://api.anthropic.com", token=tok, auth_kind=prof.auth_kind, timeout=timeout)
        if live_a:
            return [
                DiscoveredModelOption(mid, disp or mid, describe_model(mid))
                for mid, disp in live_a[:20]
            ]
        return [
            DiscoveredModelOption(mid, mid, describe_model(mid))
            for mid in ("claude-opus-4-20250514", "claude-sonnet-4-20250514", "claude-3-7-sonnet-latest", "claude-3-5-haiku-latest")
        ]

    # 4. Vertex AI Claude (`vertex-anthropic`)
    if prof.provider == "vertex-anthropic":
        return [
            DiscoveredModelOption(mid, mid, describe_model(mid))
            for mid in ("claude-opus-4@20250514", "claude-sonnet-4@20250514", "claude-3-7-sonnet@20250219")
        ]

    # 5. OpenAI / LiteLLM / OpenRouter / DeepSeek / Qwen / Kimi / Local (`GET /models`)
    if prof.provider in ("openai", "litellm", "openrouter", "deepseek", "qwen", "kimi", "local") or prof.api_mode in ("responses", "chat_completions"):
        live_o = fetch_openai_compat_models(base_clean, token=tok, extra_headers=prof.extra_headers, timeout=timeout)
        if live_o:
            if prof.provider == "openai":
                skip = ("whisper", "tts", "dall-e", "embedding", "moderation", "babbage", "davinci", "audio", "realtime")
                live_o = [m for m in live_o if not any(s in m.lower() for s in skip)]
            if live_o:
                return [DiscoveredModelOption(mid, mid, describe_model(mid)) for mid in live_o[:25]]
        defaults_by_prov: dict[str, tuple[str, ...]] = {
            "openai": ("gpt-5", "o3", "o4-mini", "gpt-4.1"),
            "deepseek": ("deepseek-reasoner", "deepseek-chat"),
            "qwen": ("qwen3-coder",),
        }
        fallbacks = defaults_by_prov.get(prof.provider, (prof.default_model,) if prof.default_model else ())
        return [DiscoveredModelOption(mid, mid, describe_model(mid)) for mid in fallbacks if mid]

    # 6. Vertex AI / Gemini Enterprise / Hosted Antigravity (`aiplatform.googleapis.com`) -> live `/v1beta1/publishers/google/models`
    live_v = fetch_vertex_publisher_models(
        base_clean or "https://aiplatform.googleapis.com",
        token=tok,
        auth_kind=prof.auth_kind,
        timeout=timeout,
    )
    if live_v:
        return [
            DiscoveredModelOption(mid, disp, describe_model(mid))
            for mid, disp in live_v[:25]
        ]
    vertex_fallbacks = [
        "gemini-4-argon",
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.1-pro-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]
    if prof.default_model and prof.default_model not in vertex_fallbacks:
        vertex_fallbacks.insert(0, prof.default_model)
    return [
        DiscoveredModelOption(mid, format_model_display_name(mid), describe_model(mid))
        for mid in vertex_fallbacks
    ]


def discover_available_models(
    vault: CredentialVault,
    prof: AuthProfile,
    *,
    timeout: float = 1.0,
) -> list[tuple[str, str]]:
    """Discovers `(model_id, description)` for `prof` cleanly without cross-provider clutter or provider labels."""
    return [(m.model_id, m.description) for m in discover_model_options(vault, prof, timeout=timeout)]


def login_antigravity_acp(
    vault: CredentialVault,
    *,
    profile_name: str = "antigravity",
    run_login_if_needed: bool = True,
) -> AuthProfile:
    """Registers an Antigravity ACP profile using the installed `agy` CLI (`agy acp --login` if unauthenticated)."""
    from ush.brain.agy_acp import resolve_agy_binary

    agy_bin = resolve_agy_binary()
    if not agy_bin:
        raise FileNotFoundError("Antigravity CLI (`agy`) is not installed on PATH.")

    def_mid, pairs = fetch_agy_models(vault.ush_home, timeout=6.0, ttl_seconds=0.0)
    if not pairs and run_login_if_needed:
        subprocess.run([agy_bin, "acp", "--login"], check=True)
        def_mid, pairs = fetch_agy_models(vault.ush_home, timeout=8.0, ttl_seconds=0.0)

    resolved_default = def_mid or (pairs[0][0] if pairs else "gemini-4-argon")
    prof = AuthProfile(
        name=profile_name,
        provider="antigravity",
        auth_kind="none",
        base_url="agy://acp",
        default_model=resolved_default,
        source="vault",
        api_mode="agy_acp",
    )
    vault._profiles[profile_name] = prof
    vault.switch_profile(profile_name)
    return prof


def interactive_login_wizard(
    vault: CredentialVault, provider_arg: str | None = None, *, no_browser: bool = False
) -> AuthProfile:
    """Interactive terminal login wizard supporting Google OAuth (Ultra/Pro/Code Assist), Gemini Enterprise, Antigravity, Claude, OpenAI, LiteLLM, GCP ADC & local."""
    choices = [
        ("antigravity", "Antigravity Endpoints (Local `agy` ACP, Google Sign-In, or Priority Vertex)"),
        ("google-oauth", "Login with Google (Antigravity / Google AI Pro & Ultra — Zero GCP/Argolis Project)"),
        ("gemini-enterprise", "Gemini Enterprise (Standard GCP ADC, Vertex Express API Key, or Code Assist)"),
        ("vertex", "Google Cloud Vertex AI (gcloud ADC — unlocks Gemini + Vertex Claude Model Garden)"),
        ("google", "Google Gemini API (AI Studio API Key)"),
        ("anthropic", "Anthropic Claude (Default API Key `x-api-key`, Bearer/OAuth Token, or Custom URL)"),
        ("openai", "OpenAI (Default `/v1/responses` + `/v1/chat/completions`, API Key or OAuth Token)"),
        ("litellm", "LiteLLM Proxy Gateway (`http://127.0.0.1:4000/v1` or Team Gateway + Master/API Key)"),
        ("openrouter", "OpenRouter (300+ Models via API Key)"),
        ("deepseek", "DeepSeek API (DeepSeek-V3 / R1)"),
        ("qwen", "Alibaba Qwen / DashScope API (Qwen3-Coder)"),
        ("kimi", "Moonshot Kimi API (Kimi-K2)"),
        ("local", "Local / Self-Hosted GPU Endpoint (Ollama / vLLM / SGLang)"),
        ("workstation", "Host-Local Workstation Adapter (`~/.ush/workstation.json` — command helper / local plugin)"),
    ]
    provider = provider_arg.lower() if provider_arg else None
    if not provider:
        print("\n=== Crucible (crux v0.1) — Authentication Setup ===")
        for idx, (slug, desc) in enumerate(choices, 1):
            print(f"  [{idx:2d}] {slug:18s} — {desc}")
        raw = input(f"\nSelect provider [1-{len(choices)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            provider = choices[int(raw) - 1][0]
        else:
            provider = raw.lower()

    if provider in ("google-oauth", "code-assist", "code-assist-oauth"):
        ca_url = os.environ.get("CODE_ASSIST_ENDPOINT", "https://cloudcode-pa.googleapis.com").strip() or "https://cloudcode-pa.googleapis.com"
        explicit_proj = os.environ.get("USH_CODE_ASSIST_PROJECT", "").strip() or None
        name = "antigravity"
        if not provider_arg and sys.stdin.isatty():
            name = input("Profile name [antigravity]: ").strip() or "antigravity"
        return login_google_code_assist_oauth(
            vault,
            profile_name=name,
            provider="antigravity",
            base_url=ca_url,
            default_model="gemini-3.1-pro-preview",
            explicit_project_id=explicit_proj,
            no_browser=no_browser,
        )

    if provider == "gemini-enterprise":
        print(
            "  Modes: [1] Standard Enterprise (gcloud ADC + Project)  [2] Express API Key  "
            "[3] Code Assist OAuth (Browser — Zero GCP Project)  [4] Code Assist (gcloud ADC + Project)"
        )
        mode_in = input("Select Gemini Enterprise mode [1]: ").strip() or "1"
        if mode_in == "3":
            ca_url = input("Code Assist Endpoint [https://cloudcode-pa.googleapis.com]: ").strip() or "https://cloudcode-pa.googleapis.com"
            name = input("Profile name [google-oauth]: ").strip() or "google-oauth"
            return login_google_code_assist_oauth(
                vault,
                profile_name=name,
                base_url=ca_url,
                default_model="gemini-3.1-pro-preview",
                no_browser=no_browser,
            )
        if mode_in == "2":
            url = input("Enterprise Base URL [https://aiplatform.googleapis.com]: ").strip() or "https://aiplatform.googleapis.com"
            name = input("Profile name [gemini-enterprise-express]: ").strip() or "gemini-enterprise-express"
            key = getpass.getpass("Enter Gemini Enterprise / Vertex Express API key: ").strip()
            return vault.register_api_key_profile(
                name=name,
                provider="gemini-enterprise",
                api_key=key,
                base_url=url,
                default_model="gemini-2.5-pro",
                api_mode="express",
            )
        detected = vault._read_gcloud_default_project() or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        proj = input(f"GCP Project ID [{detected}]: ").strip() or detected
        reg = input("Region [global]: ").strip() or "global"
        if mode_in == "4":
            ca_url = input("Code Assist Endpoint [https://cloudcode-pa.googleapis.com]: ").strip() or "https://cloudcode-pa.googleapis.com"
            name = input("Profile name [gemini-code-assist]: ").strip() or "gemini-code-assist"
            return vault.register_vertex_profile(
                name=name,
                project_id=proj,
                region=reg,
                include_claude=False,
                provider="gemini-enterprise",
                base_url=ca_url,
                api_mode="code_assist",
            )
        url = input("Vertex / VPC-SC Base URL [https://aiplatform.googleapis.com]: ").strip() or "https://aiplatform.googleapis.com"
        tier = input("Optional Service Tier (`priority` or blank) []: ").strip()
        name = input("Profile name [gemini-enterprise]: ").strip() or "gemini-enterprise"
        return vault.register_vertex_profile(
            name=name,
            project_id=proj,
            region=reg,
            include_claude=False,
            provider="gemini-enterprise",
            base_url=url,
            service_tier=tier,
        )

    if provider == "antigravity":
        from ush.brain.agy_acp import resolve_agy_binary

        has_agy = bool(resolve_agy_binary())
        if not os.environ.get("ANTIGRAVITY_BASE_URL"):
            if has_agy:
                print(
                    "  Modes: [1] Antigravity Local Session (`agy acp` — Argon + Flash, Zero GCP Project)  "
                    "[2] Google OAuth (`cloudcode-pa`)  [3] Custom / Vertex / LiteRT Endpoint"
                )
                ag_mode = input("Select Antigravity mode [1]: ").strip() or "1"
                if ag_mode == "1":
                    name = input("Profile name [antigravity]: ").strip() or "antigravity"
                    return login_antigravity_acp(vault, profile_name=name)
                if ag_mode == "2":
                    ca_url = os.environ.get("CODE_ASSIST_ENDPOINT", "https://cloudcode-pa.googleapis.com").strip() or "https://cloudcode-pa.googleapis.com"
                    name = input("Profile name [antigravity]: ").strip() or "antigravity"
                    return login_google_code_assist_oauth(
                        vault,
                        profile_name=name,
                        provider="antigravity",
                        base_url=ca_url,
                        default_model="gemini-3.1-pro-preview",
                        no_browser=no_browser,
                    )
            else:
                print(
                    "  Modes: [1] Google Account Sign-In (`cloudcode-pa` — Zero GCP/Argolis Project)  "
                    "[2] Custom / Vertex / Local LiteRT Endpoint"
                )
                ag_mode = input("Select Antigravity mode [1]: ").strip() or "1"
                if ag_mode == "1":
                    ca_url = os.environ.get("CODE_ASSIST_ENDPOINT", "https://cloudcode-pa.googleapis.com").strip() or "https://cloudcode-pa.googleapis.com"
                    name = input("Profile name [antigravity]: ").strip() or "antigravity"
                    return login_google_code_assist_oauth(
                        vault,
                        profile_name=name,
                        provider="antigravity",
                        base_url=ca_url,
                        default_model="gemini-3.1-pro-preview",
                        no_browser=no_browser,
                    )
        default_url = os.environ.get("ANTIGRAVITY_BASE_URL") or "https://aiplatform.googleapis.com"
        url = input(f"Antigravity Endpoint URL [{default_url}]: ").strip() or default_url
        tier = input("Service Tier (`priority` or blank) [priority]: ").strip()
        name = input("Profile name [antigravity]: ").strip() or "antigravity"
        key = getpass.getpass("Optional API Key or Bearer Token (press Enter to use gcloud ADC / none): ").strip()
        if key:
            kind = "api_key" if key.startswith("AIza") else "bearer"
            return vault.register_api_key_profile(
                name=name,
                provider="antigravity",
                api_key=key,
                base_url=url,
                default_model="gemini-3.1-pro-preview",
                auth_kind=kind,
                service_tier=tier,
            )
        is_google = "googleapis.com" in url
        proj = (vault._read_gcloud_default_project() or os.environ.get("GOOGLE_CLOUD_PROJECT", "")) if is_google else ""
        if is_google:
            proj = input(f"GCP Project ID [{proj}]: ").strip() or proj
        return vault.register_api_key_profile(
            name=name,
            provider="antigravity",
            api_key="",
            base_url=url,
            default_model="gemini-3.1-pro-preview" if is_google else "litert-local",
            auth_kind="gcloud_adc" if is_google else "none",
            project_id=proj or None,
            service_tier=tier,
        )

    if provider == "vertex":
        detected = vault._read_gcloud_default_project() or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        proj = input(f"GCP Project ID [{detected}]: ").strip() or detected
        if not proj:
            raise ValueError("GCP Project ID is required for Vertex AI.")
        reg = input("Vertex Region [global]: ").strip() or "global"
        name = input(f"Profile name [vertex-{proj[:16]}]: ").strip() or f"vertex-{proj[:16]}"
        return vault.register_vertex_profile(name=name, project_id=proj, region=reg)

    if provider == "anthropic":
        default_url = os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
        url = input(f"Anthropic Base URL [{default_url}]: ").strip() or default_url
        print("  Auth: [1] API Key (`x-api-key`)  [2] Auth / OAuth Bearer Token (`Authorization: Bearer`)")
        ak = "bearer" if input("Select auth kind [1]: ").strip() == "2" else "api_key"
        name = input("Profile name [anthropic]: ").strip() or "anthropic"
        key = getpass.getpass("Enter Anthropic credential (hidden): ").strip()
        return vault.register_api_key_profile(
            name=name,
            provider="anthropic",
            api_key=key,
            base_url=url,
            default_model="claude-sonnet-4-20250514",
            auth_kind=ak,
        )

    if provider == "openai":
        default_url = os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        url = input(f"OpenAI Base URL [{default_url}]: ").strip() or default_url
        org = input("Optional OpenAI Organization ID (press Enter to skip): ").strip()
        proj = input("Optional OpenAI Project ID (press Enter to skip): ").strip()
        hdrs: dict[str, str] = {}
        if org:
            hdrs["OpenAI-Organization"] = org
        if proj:
            hdrs["OpenAI-Project"] = proj
        name = input("Profile name [openai]: ").strip() or "openai"
        key = getpass.getpass("Enter OpenAI API key or OAuth access token (hidden): ").strip()
        ak = "api_key" if key.startswith("sk-") else "oauth"
        return vault.register_api_key_profile(
            name=name,
            provider="openai",
            api_key=key,
            base_url=url,
            default_model="gpt-5",
            auth_kind=ak,
            api_mode="responses",
            extra_headers=hdrs,
        )

    if provider == "litellm":
        default_url = os.environ.get("LITELLM_BASE_URL") or "http://127.0.0.1:4000/v1"
        url = input(f"LiteLLM Proxy Base URL [{default_url}]: ").strip() or default_url
        name = input("Profile name [litellm]: ").strip() or "litellm"
        key = getpass.getpass("LiteLLM API / Master Key (press Enter if unauthenticated): ").strip()
        mdl = input("Default LiteLLM model (e.g. anthropic/claude-sonnet-4-20250514) []: ").strip() or None
        return vault.register_api_key_profile(
            name=name,
            provider="litellm",
            api_key=key,
            base_url=url,
            default_model=mdl,
            auth_kind="api_key" if key else "none",
        )

    if provider == "workstation":
        name = input("Workstation profile name [workstation]: ").strip() or "workstation"
        wire_prov = input("Wire family (`gemini-enterprise`, `antigravity`, `openai`, `anthropic`, or plugin name) [gemini-enterprise]: ").strip() or "gemini-enterprise"
        url = input("Workstation Endpoint Base URL: ").strip()
        cmd = input("Token helper command (executed locally via `shell=False`, e.g. `gcloud auth print-access-token`): ").strip()
        mode_s = input("Optional wire `api_mode` (`code_assist`, `express`, `responses`, `chat_completions`, or blank) []: ").strip()
        proj = input("Optional project/tenant ID (press Enter to auto-resolve or skip) []: ").strip() or None
        mdl = input("Default model [gemini-3.1-pro-preview]: ").strip() or "gemini-3.1-pro-preview"
        host_marker = input("Optional host marker path required to activate profile (press Enter for always) []: ").strip()
        plugin = input("Optional `~/.ush/providers/<plugin>.py` module name (press Enter for built-in wire) []: ").strip()
        return vault.register_command_profile(
            name=name,
            provider=wire_prov,
            credential_command=cmd,
            base_url=url,
            default_model=mdl,
            project_id=proj,
            api_mode=mode_s,
            require_host_path=host_marker,
            transport_plugin=plugin,
            persist_to_workstation_overlay=True,
        )

    if provider == "local":
        url = input("Local/Cluster Base URL [http://127.0.0.1:11434/v1]: ").strip() or "http://127.0.0.1:11434/v1"
        name = input("Profile name [local]: ").strip() or "local"
        key = getpass.getpass("Optional Bearer token (press Enter for none): ").strip()
        if key:
            return vault.register_api_key_profile(name=name, provider="local", api_key=key, base_url=url)
        prof = AuthProfile(name=name, provider="local", auth_kind="none", base_url=url, source="vault")
        vault._profiles[name] = prof
        vault.switch_profile(name)
        return prof

    name = input(f"Profile name [{provider}]: ").strip() or provider
    key = getpass.getpass(f"Enter {provider.upper()} API key (input hidden, saved to Keychain/0600 vault): ").strip()
    return vault.register_api_key_profile(name=name, provider=provider, api_key=key)


def _is_loopback_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.08):
            return True
    except OSError:
        return False
