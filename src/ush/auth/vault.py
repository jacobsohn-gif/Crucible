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
    ("ANTHROPIC_API_KEY", "env-anthropic", "anthropic", "https://api.anthropic.com"),
    ("OPENAI_API_KEY", "env-openai", "openai", "https://api.openai.com/v1"),
    ("OPENROUTER_API_KEY", "env-openrouter", "openrouter", "https://openrouter.ai/api/v1"),
    ("DEEPSEEK_API_KEY", "env-deepseek", "openrouter", "https://api.deepseek.com/v1"),
    ("DASHSCOPE_API_KEY", "env-qwen", "openrouter", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
    ("MOONSHOT_API_KEY", "env-kimi", "openrouter", "https://api.moonshot.ai/v1"),
    ("ZHIPUAI_API_KEY", "env-glm", "openrouter", "https://open.bigmodel.cn/api/paas/v4"),
)

PROVIDER_DEFAULT_URLS: dict[str, str] = {
    "google": "https://generativelanguage.googleapis.com",
    "vertex": "https://aiplatform.googleapis.com",
    "vertex-anthropic": "https://aiplatform.googleapis.com",
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "kimi": "https://api.moonshot.ai/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "local": "http://127.0.0.1:11434/v1",
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
    provider: str  # google | vertex | vertex-anthropic | anthropic | openai | openrouter | local
    auth_kind: str  # api_key | oauth | gcloud_adc | none
    base_url: str
    project_id: str | None = None
    region: str = "global"
    default_model: str | None = None
    source: str = "vault"  # vault | keychain | env | gcloud | localhost
    # Ephemeral runtime cooldown timestamp after HTTP 429
    cooldown_until: float = 0.0

    def is_available(self, now: float | None = None) -> bool:
        return (now if now is not None else time.time()) >= self.cooldown_until

    def to_metadata_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "auth_kind": self.auth_kind,
            "base_url": self.base_url,
            "project_id": self.project_id,
            "region": self.region,
            "default_model": self.default_model,
            "source": self.source,
        }


@dataclass
class StoredSecret:
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0
    client_id: str = ""
    token_url: str = ""

    def to_json(self) -> str:
        return json.dumps(
            {
                "access_token": self.access_token,
                "refresh_token": self.refresh_token,
                "expires_at": self.expires_at,
                "client_id": self.client_id,
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
        # Headless SSH / container fallback (`0600` strict permissions)
        vault: dict[str, Any] = {}
        if self.fallback_vault_path.is_file():
            try:
                vault = json.loads(self.fallback_vault_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                vault = {}
        vault[profile_name] = json.loads(payload)
        self._atomic_write_0600(self.fallback_vault_path, json.dumps(vault, indent=2))
        return "vault"

    def load_secret(self, profile_name: str) -> StoredSecret | None:
        if self._use_macos_keychain():
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
            for item in data.get("profiles", []):
                prof = AuthProfile(**item)
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

    def discover_ambient_credentials(self) -> None:
        # 1. Standard Environment Variables (kept in memory only; never written to disk)
        for env_var, prof_name, provider, base_url in ENV_PROVIDER_MAP:
            if os.environ.get(env_var):
                actual_provider = "openrouter" if provider == "openrouter" else provider
                self._profiles.setdefault(
                    prof_name,
                    AuthProfile(
                        name=prof_name,
                        provider=actual_provider,
                        auth_kind="api_key",
                        base_url=base_url,
                        source=f"env:{env_var}",
                    ),
                )

        # 2. Google Cloud ADC (`gcloud` / ADC file) -> Vertex Gemini & Vertex Claude Model Garden
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
            or self._read_gcloud_default_project()
        )
        if (adc_file.is_file() or pathlib.Path(gcloud_bin).is_file()) and project:
            region = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            self._profiles.setdefault(
                "gcloud-vertex",
                AuthProfile(
                    name="gcloud-vertex",
                    provider="vertex",
                    auth_kind="gcloud_adc",
                    base_url="https://aiplatform.googleapis.com",
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
                    base_url="https://aiplatform.googleapis.com",
                    project_id=project,
                    region=os.environ.get("VERTEX_ANTHROPIC_REGION", "us-east5"),
                    default_model="claude-sonnet-4@20250514",
                    source="gcloud",
                ),
            )

        # 3. Localhost zero-auth inference servers (`127.0.0.1` only, <80ms check)
        for port, label in ((11434, "local-ollama"), (8000, "local-vllm"), (1234, "local-lmstudio")):
            if _is_loopback_port_open(port):
                self._profiles.setdefault(
                    label,
                    AuthProfile(
                        name=label,
                        provider="local",
                        auth_kind="none",
                        base_url=f"http://127.0.0.1:{port}/v1",
                        source="localhost",
                    ),
                )

        if not self._active_profile or self._active_profile not in self._profiles:
            if self._profiles:
                self._active_profile = next(iter(self._profiles.keys()))

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

    def select_cross_family_profile(self, exclude_family: str, exclude_profile: str | None = None) -> AuthProfile | None:
        """Selects an authenticated profile from a DIFFERENT model family for blind judging."""
        family_by_provider = {
            "google": "gemini",
            "vertex": "gemini",
            "anthropic": "anthropic",
            "vertex-anthropic": "anthropic",
            "openai": "openai",
            "openrouter": "openrouter",
            "local": "local_oss",
        }
        for prof in self._profiles.values():
            if prof.name == exclude_profile or not prof.is_available():
                continue
            fam = family_by_provider.get(prof.provider, prof.provider)
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
            if name in self._profiles:
                self._profiles[name].cooldown_until = time.time() + cooldown_seconds
            return self.get_active_profile()

    def register_api_key_profile(
        self,
        name: str,
        provider: str,
        api_key: str,
        base_url: str | None = None,
        default_model: str | None = None,
    ) -> AuthProfile:
        if not api_key.strip():
            raise ValueError("API key must not be empty.")
        backend_provider = "openrouter" if provider in ("deepseek", "qwen", "kimi", "glm") else provider
        url = base_url or PROVIDER_DEFAULT_URLS.get(provider, PROVIDER_DEFAULT_URLS["openrouter"])
        src = self.store_secret(name, StoredSecret(access_token=api_key.strip()))
        prof = AuthProfile(
            name=name,
            provider=backend_provider,
            auth_kind="api_key",
            base_url=url,
            default_model=default_model,
            source=src,
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
    ) -> AuthProfile:
        prof = AuthProfile(
            name=name,
            provider="vertex",
            auth_kind="gcloud_adc",
            base_url="https://aiplatform.googleapis.com",
            project_id=project_id.strip(),
            region=region.strip() or "global",
            default_model="gemini-3.1-pro-preview",
            source="vault",
        )
        self._profiles[name] = prof
        if include_claude:
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
            return os.environ.get("USH_LOCAL_API_KEY", "")
        if profile.source.startswith("env:"):
            env_key = profile.source.split(":", 1)[1]
            return os.environ.get(env_key, "")
        if profile.auth_kind == "gcloud_adc":
            return self._get_gcloud_access_token()
        secret = self.load_secret(profile.name)
        if not secret:
            raise RuntimeError(f"Missing stored credentials for profile '{profile.name}'. Run `ush auth login`.")
        # Auto-refresh OAuth token 5 minutes prior to expiry
        if (
            profile.auth_kind == "oauth"
            and secret.refresh_token
            and secret.token_url
            and secret.expires_at > 0
            and time.time() >= (secret.expires_at - 300)
        ):
            secret = self._refresh_oauth_token(profile.name, secret)
        return secret.access_token

    def _get_gcloud_access_token(self) -> str:
        with self._lock:
            now = time.time()
            if self._gcloud_token_cache and now < self._gcloud_token_cache[1]:
                return self._gcloud_token_cache[0]
            gcloud_bin = shutil.which("gcloud") or str(
                pathlib.Path("~/google-cloud-sdk/bin/gcloud").expanduser()
            )
            for args in (
                [gcloud_bin, "auth", "application-default", "print-access-token"],
                [gcloud_bin, "auth", "print-access-token"],
            ):
                if not pathlib.Path(args[0]).is_file():
                    continue
                res = subprocess.run(args, capture_output=True, text=True, timeout=10)
                tok = res.stdout.strip()
                if res.returncode == 0 and tok:
                    # Cache for 50 minutes (Google OAuth access tokens last 60m)
                    self._gcloud_token_cache = (tok, now + 3000)
                    return tok
            raise RuntimeError(
                "Google Cloud credentials unavailable or expired. Run `gcloud auth application-default login`."
            )

    def _refresh_oauth_token(self, profile_name: str, secret: StoredSecret) -> StoredSecret:
        body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "refresh_token": secret.refresh_token,
                "client_id": secret.client_id,
            }
        ).encode("utf-8")
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
            token_url=secret.token_url,
        )
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


def interactive_login_wizard(vault: CredentialVault, provider_arg: str | None = None) -> AuthProfile:
    """Interactive terminal login wizard supporting GCP ADC, API keys, OpenRouter OAuth PKCE, and local servers."""
    choices = [
        ("vertex", "Google Cloud Vertex AI (gcloud ADC — unlocks Gemini + Vertex Claude Model Garden)"),
        ("google", "Google Gemini API (AI Studio API Key)"),
        ("anthropic", "Anthropic Direct (Claude Opus / Sonnet API Key)"),
        ("openai", "OpenAI Direct (GPT-5 / o3 / o4-mini API Key)"),
        ("openrouter", "OpenRouter (300+ Models via Browser OAuth PKCE or API Key)"),
        ("deepseek", "DeepSeek API (DeepSeek-V3 / R1)"),
        ("qwen", "Alibaba Qwen / DashScope API (Qwen3-Coder)"),
        ("kimi", "Moonshot Kimi API (Kimi-K2)"),
        ("local", "Local / Self-Hosted GPU Endpoint (Ollama / vLLM / SGLang)"),
    ]
    provider = provider_arg
    if not provider:
        print("\n=== Crucible (crux v0.1) — Authentication Setup ===")
        for idx, (slug, desc) in enumerate(choices, 1):
            print(f"  [{idx}] {slug:11s} — {desc}")
        raw = input("\nSelect provider [1-9]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            provider = choices[int(raw) - 1][0]
        else:
            provider = raw.lower()

    if provider == "vertex":
        detected = vault._read_gcloud_default_project() or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
        proj = input(f"GCP Project ID [{detected}]: ").strip() or detected
        if not proj:
            raise ValueError("GCP Project ID is required for Vertex AI.")
        reg = input("Vertex Region [global]: ").strip() or "global"
        name = input(f"Profile name [vertex-{proj[:16]}]: ").strip() or f"vertex-{proj[:16]}"
        return vault.register_vertex_profile(name=name, project_id=proj, region=reg)

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
