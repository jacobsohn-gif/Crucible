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
    "*.jks",
    "*.keystore",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "credentials.json",
    ".credentials.json",
    "credentials.vault.json",
    "credentials.enc",
    "application_default_credentials.json",
    "auth.json",
    "auth_profiles.json",
    "workstation.json",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "kubeconfig",
)

DEFAULT_ALLOWED_EXCEPTIONS: tuple[str, ...] = (
    ".env.example",
    ".env.sample",
    ".env.template",
)

# Protected execution/hook control files inside a workspace that tools (`write_file`/`edit_file`) must never mutate directly
PROTECTED_WORKSPACE_WRITE_PATHS: tuple[str, ...] = (
    ".git/config",
    ".git/hooks",
    ".ush/hooks.json",
    ".ush/policies.toml",
    ".ush/mcp.json",
    ".mcp.json",
    ".gemini/settings.json",
    ".claude/settings.json",
    ".claude/settings.local.json",
)

SECRET_TOKEN_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-ant-[A-Za-z0-9_-]{16,}"),
    re.compile(r"sk-litellm-[A-Za-z0-9_-]{10,}"),
    re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{30,}"),
    re.compile(r"AQ\.[0-9A-Za-z._-]{24,}"),
    re.compile(r"ya29\.[0-9A-Za-z._-]{24,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{24,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]{20,}=*"),
)

FORKBOMB_RE = re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;")
RAW_DISK_OF_RE = re.compile(r"^of=/dev/(?:disk|rdisk|sd|nvme|mmcblk|vd)")


def _strip_quoted_literals(command: str) -> str:
    """Strips quoted string bodies so unquoted structural checks never false-positive on commit messages or literals."""
    out: list[str] = []
    in_sq = False
    in_dq = False
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if ch == "\\" and not in_sq and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            i += 1
            continue
        if not in_sq and not in_dq:
            out.append(ch)
        i += 1
    return "".join(out)


def _is_catastrophic_segment(seg: str) -> bool:
    """Token-level detector for recursive `rm /`, `mkfs`, and `dd of=/dev/...` regardless of flag ordering or wrappers."""
    if FORKBOMB_RE.search(_strip_quoted_literals(seg)):
        return True
    try:
        tokens = shlex.split(seg)
    except ValueError:
        unq = _strip_quoted_literals(seg)
        return bool(
            re.search(r"(?:^|\s)mkfs(?:\.[a-z0-9]+)?\s+", unq)
            or re.search(r"(?:^|\s)dd\s+[^\n;|&]*\bof=/dev/(?:disk|rdisk|sd|nvme|mmcblk|vd)", unq)
        )
    if not tokens:
        return False
    head = pathlib.PurePosixPath(tokens[0]).name
    if head == "mkfs" or head.startswith("mkfs."):
        return len(tokens) > 1
    if head == "dd":
        return any(RAW_DISK_OF_RE.search(tok) for tok in tokens[1:])
    if head != "rm":
        return False
    has_recursive = False
    targets: list[str] = []
    after_ddash = False
    for tok in tokens[1:]:
        if not after_ddash and tok == "--":
            after_ddash = True
            continue
        if not after_ddash and tok.startswith("--"):
            if tok in ("--recursive", "--no-preserve-root"):
                has_recursive = True
            continue
        if not after_ddash and tok.startswith("-") and len(tok) > 1:
            if "r" in tok or "R" in tok:
                has_recursive = True
            continue
        norm = re.sub(r"/+", "/", tok)
        targets.append(norm)
    if not has_recursive:
        return False
    return any(t in ("/", "/*") for t in targets)

SHELL_WRAPPERS: frozenset[str] = frozenset(
    {"sudo", "env", "command", "nohup", "nice", "time", "stdbuf", "timeout"}
)
SUBSHELL_LAUNCHERS: frozenset[str] = frozenset({"sh", "bash", "zsh", "dash", "ksh", "/bin/sh", "/bin/bash", "/bin/zsh"})


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
        repo_deny: list[str] = []
        repo_negate: list[str] = []
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
                        repo_negate.append(line[1:].strip())
                    else:
                        repo_deny.append(line)
            except OSError:
                continue
        return repo_deny, repo_negate

    @staticmethod
    def _matches_any(name: str, rel: str, patterns: Iterable[str]) -> bool:
        for pat in patterns:
            clean = pat.rstrip("/")
            if (
                fnmatch.fnmatch(name, clean)
                or fnmatch.fnmatch(rel, clean)
                or any(fnmatch.fnmatch(part, clean) for part in rel.split("/"))
            ):
                return True
        return False

    def is_ignored_secret_path(self, candidate: pathlib.Path) -> bool:
        name = candidate.name
        try:
            rel = candidate.resolve().relative_to(self.workspace_root).as_posix()
        except ValueError:
            rel = name

        # 1. System-level secret denylist CANNOT be negated by a workspace `.ushignore` (`!.env` escalation blocked!)
        if self._matches_any(name, rel, DEFAULT_ALLOWED_EXCEPTIONS):
            return False
        if self._matches_any(name, rel, DEFAULT_DENIED_FILE_PATTERNS):
            return True

        # 2. Workspace-level `.ushignore`/`.clineignore`/`.geminiignore` rules
        repo_deny, repo_negate = self._load_ignore_rules()
        if self._matches_any(name, rel, repo_negate):
            return False
        return self._matches_any(name, rel, repo_deny)

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
        if write:
            try:
                rel_w = resolved.relative_to(self.workspace_root).as_posix()
            except ValueError:
                rel_w = ""
            if rel_w and any(rel_w == p or rel_w.startswith(p + "/") for p in PROTECTED_WORKSPACE_WRITE_PATHS):
                raise PermissionError(
                    f"Context security firewall blocked autonomous write to protected control path `{rel_w}`."
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
        watched_files = [
            ws / ".mcp.json",
            ws / ".ush" / "mcp.json",
            ws / ".ush" / "hooks.json",
            ws / ".gemini" / "settings.json",
            ws / ".claude" / "settings.json",
            ws / ".claude" / "settings.local.json",
            ws / ".ush" / "policies.toml",
        ]
        watched_dirs = [
            ws / ".ush" / "commands",
            ws / ".gemini" / "commands",
            ws / ".claude" / "commands",
            ws / ".ush" / "skills",
            ws / ".gemini" / "skills",
            ws / ".claude" / "skills",
            ws / ".agents" / "skills",
        ]
        all_targets: list[pathlib.Path] = [f for f in watched_files if f.is_file()]
        for d in watched_dirs:
            if d.is_dir():
                for sub in sorted(d.rglob("*")):
                    if sub.is_file():
                        all_targets.append(sub)
        seen_rels: set[str] = set()
        chunks: list[str] = []
        for f in sorted(all_targets, key=lambda p: p.relative_to(ws).as_posix()):
            rel_s = f.relative_to(ws).as_posix()
            if rel_s in seen_rels:
                continue
            seen_rels.add(rel_s)
            try:
                chunks.append(f"{rel_s}:{sha256_hex(f.read_bytes())}")
            except OSError:
                chunks.append(f"{rel_s}:unreadable")
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


def _extract_balanced_parens(text: str, open_idx: int) -> tuple[str, int]:
    """Extracts content inside `(` at `open_idx` up to its matching `)`, respecting quotes & escapes."""
    depth = 1
    i = open_idx + 1
    in_sq = False
    in_dq = False
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == "\\" and not in_sq and i + 1 < n:
            i += 2
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    return text[open_idx + 1 : i], i + 1
        i += 1
    return text[open_idx + 1 :], n


def _scan_shell_structure(command: str) -> tuple[list[str], list[str], bool]:
    """Quote-aware scanner returning `(top_segments, subshell_commands, has_file_redirect)`."""
    segments: list[str] = []
    subshells: list[str] = []
    has_file_redirect = False
    cur: list[str] = []
    in_sq = False
    in_dq = False
    i = 0
    n = len(command)

    def _flush() -> None:
        s = "".join(cur).strip()
        cur.clear()
        if s:
            segments.append(s)

    while i < n:
        ch = command[i]
        if ch == "\\" and not in_sq and i + 1 < n:
            cur.append(command[i : i + 2])
            i += 2
            continue
        if ch == "'" and not in_dq:
            in_sq = not in_sq
            cur.append(ch)
            i += 1
            continue
        if ch == '"' and not in_sq:
            in_dq = not in_dq
            cur.append(ch)
            i += 1
            continue

        # Command substitutions `$(...)` and `` `...` `` execute BOTH unquoted and inside double quotes!
        if not in_sq:
            if ch == "$" and i + 1 < n and command[i + 1] == "(":
                if i + 2 < n and command[i + 2] == "(":
                    # `$(( ... ))` arithmetic expansion — scan inside only if nested `$` or `` ` `` exists
                    inner_arith, next_i = _extract_balanced_parens(command, i + 1)
                    if "$(" in inner_arith or "`" in inner_arith:
                        subshells.append(inner_arith)
                    cur.append(command[i:next_i])
                    i = next_i
                    continue
                inner_cmd, next_i = _extract_balanced_parens(command, i + 1)
                if inner_cmd.strip():
                    subshells.append(inner_cmd.strip())
                cur.append(command[i:next_i])
                i = next_i
                continue
            if ch == "`":
                j = i + 1
                bt_buf: list[str] = []
                while j < n:
                    if command[j] == "\\" and j + 1 < n:
                        bt_buf.append(command[j + 1])
                        j += 2
                        continue
                    if command[j] == "`":
                        break
                    bt_buf.append(command[j])
                    j += 1
                bt_cmd = "".join(bt_buf).strip()
                if bt_cmd:
                    subshells.append(bt_cmd)
                cur.append(command[i : min(n, j + 1)])
                i = min(n, j + 1)
                continue

        if not in_sq and not in_dq:
            # Process substitution `<(...)` / `>(...)`
            if ch in ("<", ">") and i + 1 < n and command[i + 1] == "(":
                inner_proc, next_i = _extract_balanced_parens(command, i + 1)
                if inner_proc.strip():
                    subshells.append(inner_proc.strip())
                cur.append(command[i:next_i])
                i = next_i
                continue

            # Separators: `\n`, `\r`, `;`, `&&`, `||`, `|&`, `|`, and background `&`
            if ch in ("\n", "\r", ";"):
                _flush()
                i += 1
                continue
            if command.startswith("&&", i) or command.startswith("||", i) or command.startswith("|&", i):
                _flush()
                i += 2
                continue
            if ch == "|":
                _flush()
                i += 1
                continue
            if ch == "&":
                prev_ch = command[i - 1] if i > 0 else ""
                next_ch = command[i + 1] if i + 1 < n else ""
                if prev_ch != ">" and next_ch != ">":
                    _flush()
                    i += 1
                    continue

            # Output file redirection (`>`, `>>`, `>|`, `&>`, `&>>`)
            if ch == ">":
                op_end = i + 1
                if op_end < n and command[op_end] in (">", "|"):
                    op_end += 1
                rest = command[op_end:].lstrip(" \t")
                # Ignore fd duplication (`>&2`, `2>&1`) and `/dev/null`
                if not rest.startswith("&"):
                    target_m = re.match(r"([^\s;|&<>]+)", rest)
                    target_tok = (target_m.group(1).strip("'\"") if target_m else "")
                    if target_tok != "/dev/null":
                        has_file_redirect = True
                cur.append(command[i:op_end])
                i = op_end
                continue

        cur.append(ch)
        i += 1

    _flush()
    return segments, subshells, has_file_redirect


def unwrap_shell_commands(command: str, *, _depth: int = 0) -> list[str]:
    """Splits chained/multiline/background/subshell commands and peels wrappers (`sudo`, `env`, `bash -c`, `eval`)."""
    if _depth > 6 or not command.strip():
        return [command.strip()] if command.strip() else []

    segments, subshells, has_file_redirect = _scan_shell_structure(command)
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
            # Strip leading VAR=val assignments
            if "=" in tok and not tok.startswith("-") and not tok.startswith("$"):
                idx += 1
                continue
            if tok in SHELL_WRAPPERS:
                wrapper = tok
                idx += 1
                while idx < len(tokens) and tokens[idx].startswith("-"):
                    flag = tokens[idx]
                    idx += 1
                    if flag in ("-u", "-g", "-C", "-n", "-k", "-s") and idx < len(tokens):
                        idx += 1
                while idx < len(tokens) and "=" in tokens[idx] and not tokens[idx].startswith("-"):
                    idx += 1
                if wrapper == "timeout" and idx < len(tokens):
                    # Skip timeout duration positional token (e.g. `timeout 10s rm ...`)
                    idx += 1
                continue
            break

        rem_tokens = tokens[idx:]
        rem = " ".join(rem_tokens).strip()
        if rem:
            normalized.append(rem)

        # Unwrap `eval "..."` and `<sh|bash|zsh> ... -c "<script>"`
        if rem_tokens:
            head = rem_tokens[0]
            if head == "eval" and len(rem_tokens) > 1:
                normalized.extend(unwrap_shell_commands(" ".join(rem_tokens[1:]), _depth=_depth + 1))
            elif head in SUBSHELL_LAUNCHERS and "-c" in rem_tokens[1:]:
                c_pos = rem_tokens.index("-c", 1)
                if c_pos + 1 < len(rem_tokens):
                    normalized.extend(unwrap_shell_commands(rem_tokens[c_pos + 1], _depth=_depth + 1))

    for sub in subshells:
        normalized.extend(unwrap_shell_commands(sub, _depth=_depth + 1))

    if has_file_redirect:
        normalized.append("__shell_redirect__")

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
        # 1. Unwrap shell segments once so both catastrophic safety gates and TOML rules inspect inner commands
        segments = unwrap_shell_commands(command) if command else [""]

        # Hard non-overridable catastrophic shell safety gate (blocks even in YOLO!)
        if command:
            if FORKBOMB_RE.search(_strip_quoted_literals(command)) or any(
                _is_catastrophic_segment(seg) for seg in segments
            ):
                return PolicyDecision.DENY, "Blocked catastrophic destructive command."

        # 2. Plan Mode strictly blocks mutating tools before user leaves `/plan`
        if mode == PermissionMode.PLAN and tool_name not in self.READ_ONLY_TOOLS and tool_name != "ask_user":
            return PolicyDecision.DENY, "Blocked mutating tool while in read-only `/plan` mode."

        # 3. Evaluate explicit TOML rules across unwrapped shell segments (most restrictive segment wins)
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
