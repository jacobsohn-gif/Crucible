"""4-Loader Unified Slash Command Engine & Injection-Safe `!{}` / `@{}` Macro Pipeline.

Implements CT-15:
1. Four loaders (`Builtin`, `File (.toml & .md)`, `Skill`, `McpPrompt`) resolved by `SlashCommandResolver`.
2. `CommandKind.BUILTIN` ALWAYS retains bare `/name` — repos and extensions can NEVER hijack built-ins!
3. Safe 3-Stage Macro Processor:
   - `@{path}`: ignore-checked multimodal workspace file injector (`SecurityFirewall`).
   - `!{cmd {{args}}}`: substitutes `shlex.quote(args)` inside shell blocks (zero command injection!) and checks
     the resolved command against `PolicyEngine` BEFORE executing.
   - `{{args}}` / `$ARGUMENTS`: raw prose substitution outside shell blocks.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import mimetypes
import pathlib
import re
import shlex
import subprocess
import tomllib
from typing import Callable, Iterable

from ush.core.types import CanonicalPart, PermissionMode, PolicyDecision
from ush.kernel.security import PolicyEngine, SecurityFirewall, redact_secrets, scrub_child_environment
from ush.platform.skills import SkillManager, parse_colon_safe_frontmatter

AT_FILE_RE = re.compile(r"@\{([^}\r\n]+)\}")
BANG_CMD_RE = re.compile(r"!\{([^}\r\n]+)\}")
IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})


class CommandKind(str, Enum):
    BUILTIN = "builtin"
    FILE = "file"
    SKILL = "skill"
    MCP_PROMPT = "mcp"


@dataclass(frozen=True)
class SlashCommand:
    name: str
    description: str
    kind: CommandKind
    source: str  # builtin | user | workspace | ext:<name> | skill | mcp:<server>
    template: str = ""
    skill_name: str | None = None
    available_during_task: bool = False


BUILTIN_COMMAND_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("help", "Show all built-in, custom file, skill, and MCP slash commands", True),
    ("auth", "Inspect, sign in, switch, or log out auth profiles (`/auth [status|login|switch|logout]`)", False),
    ("model", "Hot-swap active model mid-session via lossless `transform_messages` (`/model <id>`)", False),
    ("thinking", "Set or inspect native reasoning depth (`/thinking [off|low|medium|high|max]`)", True),
    ("threads", "Open visual Thinking Threads navigator & search (`Ctrl+O` or `/threads [query]`)", True),
    ("trace", "Control & query long-form thought traces (`/trace [full|compact|off|show|search|stats|export|path]`)", True),
    ("steer", "Inject live mid-turn guidance at next safe tool boundary (`Alt+S` or `/steer <msg>`)", True),
    ("queue", "Inspect, edit, or clear queued follow-up prompts (`/queue [list|pop|clear|<msg>]`)", True),
    ("steps", "Set max reasoning/tool steps per turn (`/steps <1..200>`, default=80)", True),
    ("sessions", "Browse all recent sessions across workspaces sorted newest-first (`/sessions [local]`)", True),
    ("resume", "Resume any recent session & full thought ledger sorted newest-first (`/resume [last|<id>]`)", False),
    ("profile", "Inspect active `ModelCapabilityProfile` (Tier S/A/B, context cap, parallel batch)", True),
    ("plan", "Toggle read-only `/plan` specification mode (`plan` <-> `supervised`)", False),
    ("permissions", "Switch execution mode (`/permissions plan|supervised|auto_edit|autonomous`)", True),
    ("turbo", "Configure autonomous turbo execution (`/turbo [on|fast|max|off]`)", True),
    ("btw", "Ask a 100%-KV-cached zero-history side question even while tools run (`/btw <question>`)", True),
    ("grill-me", "Socratically stress-test an architecture or plan against repo facts", False),
    ("goal", "Run bounded autonomous multi-step objective execution (`/goal <objective>`)", False),
    ("swarm", "Dispatch task to the 4-Gear Elastic Stigmergic Swarm (`/swarm [auto|g1..g4] <task>`)", False),
    ("arena", "Race multiple models head-to-head in isolated Git worktrees (`/arena m1,m2 <task>`)", False),
    ("review", "Run provenance-blind cross-family diff review (`/review`)", False),
    ("restore", "Rewind Shadow Git + DAG tree (`/restore [code+chat|code-only|chat-only]`)", False),
    ("tree", "Display conversation DAG history and checkpoints", True),
    ("compact", "Trigger epistemic context compaction keeping recent turns verbatim", False),
    ("clear", "Start a fresh conversation leaf in the current workspace", False),
    ("diff", "Show uncommitted Shadow Git diff from session start", True),
    ("skills", "List discovered 4-tier skills and activation states", True),
    ("commands", "List all resolved slash commands and any rename notices", True),
    ("hooks", "List active 9-event lifecycle hooks", True),
    ("evaluators", "List, scaffold, reload, or run independent Crux-Evals plugins (`/evaluators [list|new <id>|run [lens]|reload]`)", True),
    ("policies", "Inspect loaded 4-tier `policies.toml` rules and privilege-clamp warnings", True),
    ("trust", "Approve or revoke SHA-256 trust for repo skills, hooks, and `.mcp.json` (`/trust [grant|revoke]`)", False),
    ("memory", "Inspect 4-scope memory or review/apply `.inbox` patches (`/memory [show|inbox|apply <id>]`)", True),
    ("learn", "Stage a secret-redacted `.inbox` memory patch (`/learn <title> :: <fact>`)", False),
    ("extensions", "List or install extensions (`/extensions [list|install <path>]`)", False),
    ("stats", "Display token usage, prefix KV-cache hit rate, and edit reliability KPIs", True),
    ("doctor", "Run environment, auth, PTY, Git, and sandbox health diagnostics", True),
    ("reload", "Hot-reload skills, custom slash commands, hooks, and policies from disk", False),
    ("quit", "Cleanly close PTY sessions and exit `ush`", False),
)


class SlashCommandRegistry:
    """Loads & merges Builtin + `.toml`/`.md` File + Skill + MCP Prompt commands with zero built-in shadowing."""

    def __init__(
        self,
        workspace_root: pathlib.Path,
        *,
        skill_manager: SkillManager,
        ush_home: pathlib.Path | None = None,
        is_workspace_trusted: bool = False,
        extension_command_dirs: Iterable[tuple[str, pathlib.Path]] = (),
        mcp_prompts: Iterable[tuple[str, str, str, str]] = (),
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.skill_manager = skill_manager
        self.ush_home = (ush_home or pathlib.Path("~/.ush")).expanduser().resolve()
        self.is_workspace_trusted = is_workspace_trusted
        self.extension_command_dirs = list(extension_command_dirs)
        self.mcp_prompts = list(mcp_prompts)  # (server, prompt_name, desc, template)

        self._resolved: dict[str, SlashCommand] = {}
        self.collision_notices: list[str] = []
        self.reload()

    def reload(self) -> int:
        self._resolved.clear()
        self.collision_notices.clear()

        # Loader 1: Built-in Commands (ALWAYS own their primary bare `/name`! CT-15)
        builtin_names: set[str] = set()
        for name, desc, during in BUILTIN_COMMAND_SPECS:
            self._resolved[name] = SlashCommand(
                name=name,
                description=desc,
                kind=CommandKind.BUILTIN,
                source="builtin",
                available_during_task=during,
            )
            builtin_names.add(name)

        candidates: list[SlashCommand] = []

        # Loader 2: File Commands (`.toml` Gemini-CLI format + `.md` Claude/OpenCode format)
        file_sources: list[tuple[str, pathlib.Path]] = [("user", self.ush_home / "commands")]
        if self.is_workspace_trusted:
            for rel in (".ush/commands", ".gemini/commands", ".claude/commands"):
                file_sources.append(("workspace", self.workspace_root / rel))
        for ext_name, cdir in self.extension_command_dirs:
            file_sources.append((f"ext.{ext_name}", cdir))

        for src_label, root in file_sources:
            candidates.extend(self._load_command_dir(root, src_label))

        # Loader 3: Skill Commands (`/<skill-name> [args]`)
        for sk in self.skill_manager.user_invocable_skills():
            candidates.append(
                SlashCommand(
                    name=sk.name,
                    description=f"[Skill] {sk.description}",
                    kind=CommandKind.SKILL,
                    source="skill",
                    skill_name=sk.name,
                )
            )

        # Loader 4: MCP Prompt Commands (`/<server>:<prompt>`)
        for srv, pname, desc, tmpl in self.mcp_prompts:
            candidates.append(
                SlashCommand(
                    name=f"{srv}:{pname}",
                    description=f"[MCP {srv}] {desc}",
                    kind=CommandKind.MCP_PROMPT,
                    source=f"mcp.{srv}",
                    template=tmpl,
                )
            )

        # Deterministic Zero-Shadow `SlashCommandResolver`
        by_bare: dict[str, list[SlashCommand]] = {}
        for cmd in candidates:
            if cmd.name in builtin_names:
                prefixed = f"{cmd.source}.{cmd.name}"
                self.collision_notices.append(
                    f"`/{cmd.name}` ({cmd.source}) renamed to `/{prefixed}` — built-in commands cannot be shadowed."
                )
                self._resolved[prefixed] = SlashCommand(
                    name=prefixed,
                    description=cmd.description,
                    kind=cmd.kind,
                    source=cmd.source,
                    template=cmd.template,
                    skill_name=cmd.skill_name,
                )
            else:
                by_bare.setdefault(cmd.name, []).append(cmd)

        for bare, group in by_bare.items():
            if len(group) == 1:
                self._resolved[bare] = group[0]
            else:
                # Prefix colliding non-builtins by source so nothing silently disappears
                for cmd in group:
                    prefixed = f"{cmd.source}.{cmd.name}"
                    self.collision_notices.append(f"Collision on `/{bare}` -> exposed `/{prefixed}`.")
                    self._resolved[prefixed] = SlashCommand(
                        name=prefixed,
                        description=cmd.description,
                        kind=cmd.kind,
                        source=cmd.source,
                        template=cmd.template,
                        skill_name=cmd.skill_name,
                    )

        return len(self._resolved)

    @staticmethod
    def _load_command_dir(root: pathlib.Path, source: str) -> list[SlashCommand]:
        if not root.is_dir():
            return []
        found: list[SlashCommand] = []
        for fp in sorted(root.rglob("*")):
            if not fp.is_file() or fp.suffix not in (".toml", ".md"):
                continue
            rel_parts = list(fp.relative_to(root).with_suffix("").parts)
            cmd_name = ":".join(p.lower() for p in rel_parts)
            try:
                raw = fp.read_text(encoding="utf-8", errors="replace")
                if fp.suffix == ".toml":
                    data = tomllib.loads(raw)
                    prompt = str(data.get("prompt", "")).strip()
                    desc = str(data.get("description") or f"Custom command from {fp.name}")
                else:
                    meta, body = parse_colon_safe_frontmatter(raw)
                    prompt = body
                    desc = str(meta.get("description") or f"Markdown command from {fp.name}")
                if prompt:
                    found.append(
                        SlashCommand(
                            name=cmd_name,
                            description=desc,
                            kind=CommandKind.FILE,
                            source=source,
                            template=prompt,
                        )
                    )
            except (OSError, tomllib.TOMLDecodeError):
                continue
        return found

    def get(self, name: str) -> SlashCommand | None:
        return self._resolved.get(name.lstrip("/"))

    def list_commands(self) -> list[SlashCommand]:
        return [self._resolved[k] for k in sorted(self._resolved)]


# ============================================================================
# Injection-Safe 3-Stage Macro Interpolation (`@{file}` + `!{cmd}` + `{{args}}`)
# ============================================================================


def expand_command_template(
    template: str,
    raw_args: str,
    *,
    firewall: SecurityFirewall,
    policy_engine: PolicyEngine,
    mode: PermissionMode = PermissionMode.DEFAULT,
    approval_callback: Callable[[str, str], bool] | None = None,
) -> tuple[str, list[CanonicalPart]]:
    """Safely expands `@{file}`, shell-escaped + policy-checked `!{cmd {{args}}}`, and prose `{{args}}` (CT-15)."""
    extra_parts: list[CanonicalPart] = []

    # Stage 1: `@{path}` ignore-safe multimodal expansion
    def _replace_at(match: re.Match[str]) -> str:
        rel_path = match.group(1).strip()
        target = firewall.resolve_and_validate(rel_path, write=False)
        if not target.is_file():
            return f"[File not found: {rel_path}]"
        raw = target.read_bytes()
        if target.suffix.lower() in IMAGE_EXTS:
            mime = mimetypes.guess_type(target.name)[0] or "image/png"
            extra_parts.append(
                CanonicalPart(
                    kind="image",
                    mime_type=mime,
                    base64_data=base64.b64encode(raw).decode("ascii"),
                    alt_text=rel_path,
                )
            )
            return f"[Attached image: {rel_path}]"
        txt = redact_secrets(raw.decode("utf-8", errors="replace"))
        return f"\n<file path=\"{rel_path}\">\n{txt[:32_000]}\n</file>\n"

    text = AT_FILE_RE.sub(_replace_at, template)

    # Stage 2: `!{cmd {{args}}}` — ALWAYS shell-escape `{{args}}`/`$ARGUMENTS` inside `!{...}` (CT-15)!
    escaped_args = shlex.quote(raw_args)

    def _replace_bang(match: re.Match[str]) -> str:
        raw_cmd = match.group(1).strip()
        safe_cmd = raw_cmd.replace("{{args}}", escaped_args).replace("$ARGUMENTS", escaped_args)
        decision, reason = policy_engine.check("exec_pty", command=safe_cmd, mode=mode)
        if decision == PolicyDecision.DENY:
            raise PermissionError(f"Policy denied `!{{{safe_cmd}}}`: {reason}")
        if decision == PolicyDecision.ASK_USER:
            if approval_callback is None or not approval_callback(safe_cmd, reason):
                raise PermissionError(f"Confirmation required for macro shell command `!{{{safe_cmd}}}`: {reason}")
        res = subprocess.run(
            ["/bin/sh", "-c", safe_cmd],
            cwd=str(firewall.workspace_root),
            env=scrub_child_environment(),
            capture_output=True,
            text=True,
            timeout=20,
        )
        combined = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
        return redact_secrets(combined)[:24_000]

    had_arg_placeholder = ("{{args}}" in template) or ("$ARGUMENTS" in template)
    text = BANG_CMD_RE.sub(_replace_bang, text)

    # Stage 3: Raw prose `{{args}}` / `$ARGUMENTS` substitution outside shell blocks
    text = text.replace("{{args}}", raw_args).replace("$ARGUMENTS", raw_args)
    if raw_args.strip() and not had_arg_placeholder:
        text = f"{text}\n\n{raw_args.strip()}"
    return text.strip(), extra_parts
