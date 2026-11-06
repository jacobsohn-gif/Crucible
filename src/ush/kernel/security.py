"""3-Layer Context Security Firewall, Repo Trust Store & 4-Tier Privilege-Clamped `PolicyEngine`.

Implements CT-12 and CT-16:
1. Child-process environment secret scrubbing (`*KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*CREDENTIAL*`).
2. Strict path-boundary verification (`Path.resolve().is_relative_to(...)`) + `.ushignore`/`.clineignore`/`.geminiignore`.
3. Output secret token redactor (prevents leaked bearer/API keys entering prompts or `/learn` patches).
4. Host-stored SHA-256 `RepoTrustStore` quarantining untrusted workspace skills, hooks, and `.mcp.json`.
5. 4-Tier `PolicyEngine` (`admin > user > workspace > extension`) with shell-chain unwrapping (`sudo`, `env`, `&&`, `|`),
   MCP wildcard matching (`mcp_server_*`), and hard privilege clamping dropping `allow`/`yolo` from extensions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import json
import os
import pathlib
import re
import shlex
import tomllib
from typing import Iterable

from ush.core.types import PermissionMode, PolicyDecision, sha256_hex

SENSITIVE_ENV_SUBSTRINGS: tuple[str, ...] = (
    "KEY",
    "SECRET",
    "TOKEN",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "PRIVATE",
    "BEARER",
    "COOKIE",
    "AUTH",
)

SAFE_ENV_ALLOWLIST: frozenset[str] = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "PWD",
        "EDITOR",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "GOPATH",
        "UV_CACHE_DIR",
        "SSH_AUTH_SOCK",  # Needed ONLY if user explicitly passes ssh; excluded by default below
    }
    - {"SSH_AUTH_SOCK"}
)

DEFAULT_DENIED_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "credentials.json",
    "credentials.vault.json",
    "credentials.enc",
    "application_default_credentials.json",
)

DEFAULT_ALLOWED_EXCEPTIONS: tuple[str, ...] = (
    ".env.example",
    ".env.sample",
    ".env.template",
)

SECRET_TOKEN_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"ya29\.[0-9A-Za-z._-]{24,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{24,}=*"),
)

CATASTROPHIC_SHELL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\s)rm\s+-[a-zA-Z]*r[a-zA-Z]*f[a-zA-Z]*\s+/(?:\s|$|\*)"),
    re.compile(r"(?:^|\s)mkfs(?:\.[a-z0-9]+)?\s+"),
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;"),
)

SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
SHELL_WRAPPERS: frozenset[str] = frozenset({"sudo", "env", "command", "nohup", "nice", "time"})


def redact_secrets(text: str) -> str:
    """Scrubs high-entropy API keys and bearer tokens before prompt or memory ingestion."""
    out = text
    for rx in SECRET_TOKEN_REGEXES:
        if "bearer" in rx.pattern.lower():
            out = rx.sub(r"\1[REDACTED_SECRET]", out)
        else:
            out = rx.sub("[REDACTED_SECRET]", out)
    return out


def scrub_child_environment(
    base_env: dict[str, str] | None = None,
    explicit_allowlist: Iterable[str] = (),
    extra_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    """Strips secret env vars (`*KEY*`, `*SECRET*`, `*TOKEN*`, etc.) and sets anti-hang terminal guards."""
    src = base_env if base_env is not None else dict(os.environ)
    allow = set(SAFE_ENV_ALLOWLIST) | set(explicit_allowlist)
    clean: dict[str, str] = {}
    for k, v in src.items():
        uk = k.upper()
        if k in allow:
            clean[k] = v
            continue
        if uk.startswith("USH_INTERNAL_"):
            continue
        if any(tok in uk for tok in SENSITIVE_ENV_SUBSTRINGS):
            continue
        clean[k] = v

    # Anti-hang non-interactive CLI defaults
    clean.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_EDITOR": "true",
            "PAGER": "cat",
            "GIT_PAGER": "cat",
            "DEBIAN_FRONTEND": "noninteractive",
            "CI": clean.get("CI", "true"),
        }
    )
    if extra_overrides:
        clean.update(extra_overrides)
    return clean


class SecurityFirewall:
    """Enforces workspace read/write sandbox boundaries and `.ushignore`/`.clineignore`/`.geminiignore` rules."""

    def __init__(self, workspace_root: pathlib.Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        # Dynamically expandable read roots (e.g. granted when `activate_skill` activates an external skill!)
        self.extra_read_roots: list[pathlib.Path] = []
        self.extra_write_roots: list[pathlib.Path] = []

    def grant_read_directory(self, directory: pathlib.Path) -> pathlib.Path:
        resolved = directory.expanduser().resolve()
        if resolved not in self.extra_read_roots:
            self.extra_read_roots.append(resolved)
        return resolved

    def grant_write_directory(self, directory: pathlib.Path) -> pathlib.Path:
        resolved = directory.expanduser().resolve()
        if resolved not in self.extra_write_roots:
            self.extra_write_roots.append(resolved)
        return resolved

    def _load_ignore_rules(self) -> tuple[list[str], list[str]]:
        deny = list(DEFAULT_DENIED_FILE_PATTERNS)
        negate = list(DEFAULT_ALLOWED_EXCEPTIONS)
        for fname in (".ushignore", ".clineignore", ".geminiignore"):
            fpath = self.workspace_root / fname
            if not fpath.is_file():
                continue
            try:
                for raw in fpath.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.startswith("!"):
                        negate.append(line[1:].strip())
                    else:
                        deny.append(line)
            except OSError:
                continue
        return deny, negate

    def is_ignored_secret_path(self, candidate: pathlib.Path) -> bool:
        deny, negate = self._load_ignore_rules()
        name = candidate.name
        try:
            rel = candidate.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            rel = name

        for pat in negate:
            if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
                return False
        for pat in deny:
            clean = pat.rstrip("/")
            if (
                fnmatch.fnmatch(name, clean)
                or fnmatch.fnmatch(rel, clean)
                or any(fnmatch.fnmatch(part, clean) for part in rel.split("/"))
            ):
                return True
        return False

    def resolve_and_validate(self, user_path: str | pathlib.Path, *, write: bool = False) -> pathlib.Path:
        raw = pathlib.Path(user_path).expanduser()
        resolved = (raw if raw.is_absolute() else (self.workspace_root / raw)).resolve()

        allowed_roots = [self.workspace_root] + (
            self.extra_write_roots if write else (self.extra_read_roots + self.extra_write_roots)
        )
        # Uses `Path.is_relative_to` which enforces exact path segment boundary (blocks `/repo-evil` prefix bypass!)
        if not any(resolved.is_relative_to(root) for root in allowed_roots):
            mode_lbl = "write" if write else "read"
            raise PermissionError(
                f"Sandbox boundary violation: {mode_lbl} access to `{resolved}` is outside allowed workspace `{self.workspace_root}`."
            )
        if self.is_ignored_secret_path(resolved):
            raise PermissionError(
                f"Context security firewall blocked access to restricted path `{resolved.name}` (.ushignore policy)."
            )
        return resolved


# ============================================================================
# Host-Stored Workspace Trust Quarantine (`trusted_repos.json`)
# ============================================================================


class RepoTrustStore:
    """Prevents untrusted repositories from auto-executing `.mcp.json`, `hooks.json`, or workspace skills."""

    def __init__(self, ush_home: pathlib.Path) -> None:
        self.path = ush_home.expanduser().resolve() / "trusted_repos.json"
        self._records: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.path.is_file():
            try:
                self._records = dict(json.loads(self.path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                self._records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def fingerprint_workspace_configs(workspace_root: pathlib.Path) -> str:
        ws = workspace_root.resolve()
        watched = [
            ws / ".mcp.json",
            ws / ".ush" / "mcp.json",
            ws / ".ush" / "hooks.json",
            ws / ".gemini" / "settings.json",
            ws / ".ush" / "policies.toml",
        ]
        chunks: list[str] = []
        for f in watched:
            if f.is_file():
                chunks.append(f"{f.relative_to(ws)}:{sha256_hex(f.read_bytes())}")
        return sha256_hex("|".join(chunks))

    def is_trusted(self, workspace_root: pathlib.Path) -> bool:
        ws = workspace_root.resolve()
        key = str(ws)
        if key not in self._records:
            return False
        return self._records[key] == self.fingerprint_workspace_configs(ws)

    def trust(self, workspace_root: pathlib.Path) -> str:
        ws = workspace_root.resolve()
        fp = self.fingerprint_workspace_configs(ws)
        self._records[str(ws)] = fp
        self._save()
        return fp

    def untrust(self, workspace_root: pathlib.Path) -> None:
        self._records.pop(str(workspace_root.resolve()), None)
        self._save()


# ============================================================================
# 4-Tier Privilege-Clamped TOML Policy Engine (CT-16)
# ============================================================================

TIER_WEIGHT: dict[str, int] = {
    "extension": 1,
    "workspace": 2,
    "user": 3,
    "admin": 4,
}


@dataclass(frozen=True)
class PolicyRule:
    tier: str  # admin | user | workspace | extension
    tool_pattern: str  # e.g. "exec_pty", "edit_file", "mcp_github_*", "*"
    decision: PolicyDecision
    priority: int = 100
    command_prefixes: tuple[str, ...] = ()
    command_regex: str | None = None
    modes: tuple[str, ...] = ()
    reason: str = ""

    @property
    def sort_key(self) -> tuple[int, int]:
        return (TIER_WEIGHT.get(self.tier, 0), self.priority)


def unwrap_shell_commands(command: str) -> list[str]:
    """Splits chained commands (`&&`, `||`, `;`, `|`) and peels `sudo`/`env` wrappers to stop bypasses."""
    segments = [seg.strip() for seg in SHELL_SPLIT_RE.split(command) if seg.strip()]
    normalized: list[str] = []
    for seg in segments:
        try:
            tokens = shlex.split(seg)
        except ValueError:
            normalized.append(seg)
            continue
        idx = 0
        while idx < len(tokens):
            tok = tokens[idx]
            # Strip leading VAR=val assignments and sudo/env flags
            if "=" in tok and not tok.startswith("-") and idx == 0:
                idx += 1
                continue
            if tok in SHELL_WRAPPERS:
                idx += 1
                while idx < len(tokens) and tokens[idx].startswith("-"):
                    # sudo -u root cmd -> skip option value for -u
                    flag = tokens[idx]
                    idx += 1
                    if flag in ("-u", "-g", "-C") and idx < len(tokens):
                        idx += 1
                while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
                    idx += 1
                continue
            break
        rem = " ".join(tokens[idx:]).strip()
        if rem:
            normalized.append(rem)
    return normalized or [command.strip()]


class PolicyEngine:
    """Gemini-CLI-inspired declarative TOML policy engine with extension privilege clamping."""

    READ_ONLY_TOOLS: frozenset[str] = frozenset(
        {"read_file", "grep_files", "glob_files", "lsp_query", "activate_skill"}
    )

    def __init__(self) -> None:
        self._rules: list[PolicyRule] = []
        self.clamped_warnings: list[str] = []

    @property
    def rules(self) -> list[PolicyRule]:
        return sorted(self._rules, key=lambda r: r.sort_key, reverse=True)

    def add_rule(self, rule: PolicyRule) -> bool:
        if rule.tier == "extension" and rule.decision not in (PolicyDecision.DENY, PolicyDecision.ASK_USER):
            self.clamped_warnings.append(
                f"Ignored extension policy attempting `{rule.decision.value}` on `{rule.tool_pattern}` (privilege-clamped)."
            )
            return False
        self._rules.append(rule)
        return True

    def load_toml_text(self, content: str, *, tier: str, source_label: str = "policies.toml") -> int:
        data = tomllib.loads(content)
        loaded = 0
        for raw in data.get("rule", []):
            dec_raw = str(raw.get("decision", "")).lower().strip()
            # Privilege clamp: Extension tier can NEVER grant allow or yolo! (CT-16)
            if tier == "extension" and dec_raw not in ("deny", "ask_user"):
                self.clamped_warnings.append(
                    f"[{source_label}] Dropped disallowed extension decision `{dec_raw}` on `{raw.get('toolName', '*')}`."
                )
                continue
            if dec_raw not in ("allow", "ask_user", "deny"):
                continue
            tool_Field = raw.get("toolName", "*")
            tool_names = [tool_Field] if isinstance(tool_Field, str) else list(tool_Field)
            prefixes = raw.get("commandPrefix", [])
            if isinstance(prefixes, str):
                prefixes = [prefixes]
            modes = tuple(str(m).lower() for m in raw.get("modes", []))
            for tname in tool_names:
                ok = self.add_rule(
                    PolicyRule(
                        tier=tier,
                        tool_pattern=str(tname),
                        decision=PolicyDecision(dec_raw),
                        priority=int(raw.get("priority", 100)),
                        command_prefixes=tuple(str(p) for p in prefixes),
                        command_regex=raw.get("commandRegex"),
                        modes=modes,
                        reason=str(raw.get("reason", f"{tier} policy ({source_label})")),
                    )
                )
                if ok:
                    loaded += 1
        return loaded

    def load_toml_file(self, path: pathlib.Path, *, tier: str) -> int:
        if not path.is_file():
            return 0
        return self.load_toml_text(path.read_text(encoding="utf-8"), tier=tier, source_label=str(path))

    def check(
        self,
        tool_name: str,
        *,
        command: str | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
    ) -> tuple[PolicyDecision, str]:
        # 1. Hard non-overridable catastrophic shell safety gate (blocks even in YOLO!)
        if command:
            for rx in CATASTROPHIC_SHELL_PATTERNS:
                if rx.search(command):
                    return PolicyDecision.DENY, "Blocked catastrophic destructive command."

        # 2. Plan Mode strictly blocks mutating tools before user leaves `/plan`
        if mode == PermissionMode.PLAN and tool_name not in self.READ_ONLY_TOOLS and tool_name != "ask_user":
            return PolicyDecision.DENY, "Blocked mutating tool while in read-only `/plan` mode."

        # 3. Evaluate explicit TOML rules across unwrapped shell segments (most restrictive segment wins)
        segments = unwrap_shell_commands(command) if command else [""]
        segment_decisions: list[tuple[PolicyDecision, str]] = []

        for seg in segments:
            matched: tuple[PolicyDecision, str] | None = None
            for rule in self.rules:
                if rule.modes and mode.value not in rule.modes:
                    continue
                if not fnmatch.fnmatch(tool_name, rule.tool_pattern):
                    continue
                if rule.command_prefixes:
                    if not seg or not any(
                        seg == pref or seg.startswith(pref + " ") for pref in rule.command_prefixes
                    ):
                        continue
                if rule.command_regex:
                    if not seg or not re.search(rule.command_regex, seg):
                        continue
                matched = (rule.decision, rule.reason)
                break
            if matched:
                segment_decisions.append(matched)

        if segment_decisions:
            # Across chained `cmd1 && cmd2`, DENY beats ASK_USER beats ALLOW
            for prio in (PolicyDecision.DENY, PolicyDecision.ASK_USER):
                for dec, rsn in segment_decisions:
                    if dec == prio:
                        return dec, rsn
            # Only ALLOW if *every* chained segment matched an explicit ALLOW rule!
            if len(segment_decisions) == len(segments) and all(
                d == PolicyDecision.ALLOW for d, _ in segment_decisions
            ):
                return segment_decisions[0]
            # Mixed allowed + unmatched segments -> default mode check below

        # 4. Mode defaults
        if tool_name in self.READ_ONLY_TOOLS or tool_name in ("ask_user", "swarm_dispatch"):
            return PolicyDecision.ALLOW, "Read-only / safe builtin tool."
        if mode == PermissionMode.YOLO:
            return PolicyDecision.ALLOW, "Auto-approved by explicit `yolo` mode."
        if mode == PermissionMode.AUTO_EDIT and tool_name in ("edit_file", "write_file"):
            return PolicyDecision.ALLOW, "Workspace file edit approved by `auto_edit` mode."
        return PolicyDecision.ASK_USER, f"`{tool_name}` requires approval in `{mode.value}` mode."
