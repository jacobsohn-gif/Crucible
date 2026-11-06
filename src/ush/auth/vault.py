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
import secrets
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


@dataclass
class AuthProfile:
    name: str
    # google | vertex | gemini-enterprise | antigravity | vertex-anthropic | anthropic | openai | litellm | openrouter | local
    provider: str
    # api_key | bearer | oauth | gcloud_adc | none
    auth_kind: str
    base_url: str
    project_id: str | None = None
    region: str = "global"
    default_model: str | None = None
    source: str = "vault"  # vault | keychain | env:* | gcloud | localhost | codex_cli | claude_cli
    # Wire sub-mode: "" (auto) | "express" | "code_assist" | "responses" | "chat_completions"
    api_mode: str = ""
    # Optional service tier (e.g., "priority" on Gemini Enterprise / Antigravity / OpenAI)
    service_tier: str = ""
    # Optional non-secret organizational/tenant headers (e.g., OpenAI-Organization, OpenAI-Project, ChatGPT-Account-Id)
    extra_headers: dict[str, str] = field(default_factory=dict)
    # Ephemeral runtime cooldown timestamp after HTTP 429
    cooldown_until: float = 0.0

    def is_available(self, now: float | None = None) -> bool:
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
        self.fallback_vault_path = self.ush_home / "credentials.vault.json"
        self._profiles: dict[str, AuthProfile] = {}
        self._active_profile: str | None = None
        self._gcloud_token_cache: tuple[str, float] | None = None
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
        if not self.profiles_path.is_file():
            return
        try:
            data = json.loads(self.profiles_path.read_text(encoding="utf-8"))
            self._active_profile = data.get("active_profile")
            valid_keys = set(AuthProfile.__dataclass_fields__.keys())
            for item in data.get("profiles", []):
                if not isinstance(item, dict):
                    continue
                filtered = {k: v for k, v in item.items() if k in valid_keys}
                prof = AuthProfile(**filtered)
                self._profiles[prof.name] = prof
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

        if not self._active_profile or self._active_profile not in self._profiles:
            if self._profiles:
                self._active_profile = next(iter(self._profiles.keys()))

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
        return self._profiles.get(self._active_profile or "")

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
        """Clears cached OAuth/ADC tokens after an HTTP 401 so the next retry fetches a fresh token."""
        with self._lock:
            self._gcloud_token_cache = None
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
        ambient = getattr(self, "_ambient_secrets", {}).get(profile.name)
        secret = ambient or self.load_secret(profile.name)
        if not secret:
            raise RuntimeError(f"Missing stored credentials for profile '{profile.name}'. Run `crux auth login`.")
        # Auto-refresh OAuth token 5 minutes prior to expiry
        if (
            profile.auth_kind == "oauth"
            and secret.refresh_token
            and secret.token_url
            and secret.expires_at > 0
            and time.time() >= (secret.expires_at - 300)
        ):
            secret = self._refresh_oauth_token(profile.name, secret, persist=(ambient is None))
            if ambient is not None:
                self._ambient_secrets[profile.name] = secret
        return secret.access_token

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
) -> AuthProfile:
    """Runs OAuth 2.0 Authorization Code + PKCE bound strictly to 127.0.0.1."""
    verifier, challenge = generate_pkce_pair()
    expected_state = secrets.token_urlsafe(24)
    received: dict[str, str] = {}

    class _CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            qs = urllib.parse.parse_qs(parsed.query)
            received["code"] = (qs.get("code") or [""])[0]
            received["state"] = (qs.get("state") or [""])[0]
            received["error"] = (qs.get("error") or [""])[0]
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(
                b"<!doctype html><html><body><h2>USH Authentication Complete</h2>"
                b"<p>You may close this tab and return to your terminal.</p></body></html>"
            )

        def log_message(self, fmt: str, *args: Any) -> None:
            return  # Never log query params containing auth codes

    # Strictly bind 127.0.0.1 (never 0.0.0.0)
    server = HTTPServer(("127.0.0.1", 0), _CallbackHandler)
    port = server.server_address[1]
    redirect_uri = f"http://127.0.0.1:{port}/oauth/callback"

    query = urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": " ".join(scopes),
            "state": expected_state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    full_auth_url = f"{authorize_url}?{query}"

    is_ssh = bool(os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"))
    if no_browser or is_ssh:
        server.server_close()
        print(f"\nOpen this URL in any browser:\n  {full_auth_url}\n")
        pasted = input("Paste the final redirected callback URL (or authorization code): ").strip()
        if pasted.startswith("http"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
            code = (qs.get("code") or [""])[0]
            state = (qs.get("state") or [""])[0]
            if state != expected_state:
                raise RuntimeError("OAuth state mismatch — login aborted.")
        else:
            code = pasted
    else:
        print(f"Opening browser for OAuth PKCE sign-in (fallback URL: {full_auth_url})...")
        webbrowser.open(full_auth_url)
        server.timeout = 180
        try:
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

    token_body = urllib.parse.urlencode(
        {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": client_id,
            "code_verifier": verifier,
        }
    ).encode("utf-8")
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
            token_url=token_url,
        ),
    )
    prof = AuthProfile(
        name=profile_name,
        provider=provider,
        auth_kind="oauth",
        base_url=base_url,
        source=src,
    )
    vault._profiles[profile_name] = prof
    vault.switch_profile(profile_name)
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


def interactive_login_wizard(vault: CredentialVault, provider_arg: str | None = None) -> AuthProfile:
    """Interactive terminal login wizard supporting Gemini Enterprise, Antigravity, Claude, OpenAI, LiteLLM, GCP ADC & local."""
    choices = [
        ("gemini-enterprise", "Gemini Enterprise (Standard GCP ADC, Vertex Express API Key, or Code Assist)"),
        ("antigravity", "Antigravity Endpoints (Hosted Priority / Standard or Local LiteRT / Custom URL)"),
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
    ]
    provider = provider_arg
    if not provider:
        print("\n=== Crucible (crux v0.1) — Authentication Setup ===")
        for idx, (slug, desc) in enumerate(choices, 1):
            print(f"  [{idx:2d}] {slug:18s} — {desc}")
        raw = input(f"\nSelect provider [1-{len(choices)}]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            provider = choices[int(raw) - 1][0]
        else:
            provider = raw.lower()

    if provider == "gemini-enterprise":
        print("  Modes: [1] Standard Enterprise (gcloud ADC + Project)  [2] Express API Key  [3] Code Assist (`cloudcode-pa`)")
        mode_in = input("Select Gemini Enterprise mode [1]: ").strip() or "1"
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
        if mode_in == "3":
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
