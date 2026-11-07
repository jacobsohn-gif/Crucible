"""Main `ush` CLI Executable: Rich Interactive Terminal TUI, Headless `-p`, `ush auth`, `ush swarm` & `ush doctor`."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import readline
import select
import shutil
import sys
import threading
import time
from typing import Any

from ush import __version__
from ush.auth.vault import AuthProfile, CredentialVault, interactive_login_wizard
from ush.brain.model_tiers import resolve_model_profile
from ush.brain.providers import HttpProviderTransport, UniversalModelClient
from ush.brain.trace import StreamTracePrinter, ThoughtTraceLedger, format_relative_age, list_workspace_sessions
from ush.core.types import CanonicalPart, Gear, PermissionMode, StreamEvent, ThinkingEffort, TraceMode, new_id
from ush.kernel.fsm import SessionRunner
from ush.kernel.security import PolicyEngine, RepoTrustStore, redact_secrets
from ush.platform.commands import CommandKind, SlashCommandRegistry, expand_command_template
from ush.platform.hooks_memory_ext import ExtensionManager, HookSystem, MemoryManager
from ush.platform.skills import SkillManager
from ush.swarm.engine import ElasticSwarmCoordinator, select_initial_gear
from ush.swarm.judge import CandidatePatch, CascadingJudge

# ANSI styling helpers (automatically disabled when NO_COLOR is set or non-TTY)
USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def c_bold(t: str) -> str:
    return _c("1", t)


def c_cyan(t: str) -> str:
    return _c("36", t)


def c_green(t: str) -> str:
    return _c("32", t)


def c_yellow(t: str) -> str:
    return _c("33", t)


def c_red(t: str) -> str:
    return _c("31", t)


def c_dim(t: str) -> str:
    return _c("2", t)


# ============================================================================
# Environment Bootstrapper (Wires Vault + Policies + Hooks + Skills + Slash)
# ============================================================================


class HarnessRuntime:
    def __init__(
        self,
        workspace: pathlib.Path,
        *,
        model_override: str | None = None,
        profile_override: str | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
        gear: Gear = Gear.AUTO,
        thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
        trace_mode: TraceMode = TraceMode.FULL,
        max_steps: int = 80,
        session_id: str | None = None,
    ) -> None:
        self.workspace = workspace.expanduser().resolve()
        self.ush_home = pathlib.Path(os.environ.get("USH_HOME", "~/.ush")).expanduser().resolve()
        self.gear = gear
        self.trace_mode = trace_mode
        self.max_steps = max(1, max_steps)
        self.print_lock = threading.RLock()

        self.vault = CredentialVault(self.ush_home, probe_ambient=True)
        if profile_override:
            self.vault.switch_profile(profile_override)

        self.auth_profile = self.vault.get_active_profile() or AuthProfile(
            name="unconfigured",
            provider="vertex",
            auth_kind="none",
            base_url="https://aiplatform.googleapis.com",
            default_model="gemini-2.5-pro",
            source="none",
        )

        self.trust_store = RepoTrustStore(self.ush_home)
        self.is_trusted = self.trust_store.is_trusted(self.workspace)

        self.policy_engine = PolicyEngine()
        self.hook_system = HookSystem(self.workspace, is_workspace_trusted=self.is_trusted)
        self.ext_manager = ExtensionManager(self.ush_home)
        exts = self.ext_manager.discover_and_wire(self.policy_engine, self.hook_system)

        # Load user & trusted-workspace policies/hooks (`Admin > User > Workspace > Extension`)
        self.policy_engine.load_toml_file(self.ush_home / "policies.toml", tier="user")
        self.hook_system.load_hooks_json(self.ush_home / "hooks.json", source="user")
        if self.is_trusted:
            self.policy_engine.load_toml_file(self.workspace / ".ush" / "policies.toml", tier="workspace")
            self.hook_system.load_hooks_json(self.workspace / ".ush" / "hooks.json", source="workspace")

        ext_skill_dirs = [e.skill_dir for e in exts if e.skill_dir]
        ext_cmd_dirs = [(e.name, e.command_dir) for e in exts if e.command_dir]
        ext_ctx_files = [e.context_file for e in exts if e.context_file]

        self.skill_manager = SkillManager(
            self.workspace,
            is_workspace_trusted=self.is_trusted,
            ush_home=self.ush_home,
            extension_skill_dirs=ext_skill_dirs,
        )
        self.memory_manager = MemoryManager(
            self.workspace,
            ush_home=self.ush_home,
            extension_context_files=ext_ctx_files,
        )
        self.slash_registry = SlashCommandRegistry(
            self.workspace,
            skill_manager=self.skill_manager,
            ush_home=self.ush_home,
            is_workspace_trusted=self.is_trusted,
            extension_command_dirs=ext_cmd_dirs,
        )

        model_id = model_override or self.auth_profile.default_model or "gemini-3.1-pro-preview"
        m_prof = resolve_model_profile(model_id, self.auth_profile.provider)
        transport = HttpProviderTransport(self.vault, thinking_effort=thinking_effort)
        self.client = UniversalModelClient(self.vault, transport)

        sess_id = session_id or new_id("sess")
        journal = self.ush_home / "sessions" / f"{sess_id}.jsonl"
        self.runner = SessionRunner(
            workspace_root=self.workspace,
            client=self.client,
            auth_profile=self.auth_profile,
            model_profile=m_prof,
            mode=mode,
            session_id=sess_id,
            journal_path=journal,
            ush_home=self.ush_home,
            policy_engine=self.policy_engine,
            hook_runner=self.hook_system,
            skill_manager=self.skill_manager,
            memory_manager=self.memory_manager,
            thinking_effort=thinking_effort,
        )
        self.swarm = ElasticSwarmCoordinator(
            self.workspace,
            judge=CascadingJudge(self.vault, self.client),
        )
        self.runner.swarm_callback = lambda args: str(
            self.swarm.blackboard.post(
                str(args.get("event_type", "NOTE")),
                self.runner.session_id,
                str(args.get("summary", "")),
                str(args.get("scope", "*")),
            )
        )

    def reload_extensibility(self) -> tuple[int, int]:
        self.is_trusted = self.trust_store.is_trusted(self.workspace)
        self.skill_manager.is_workspace_trusted = self.is_trusted
        self.slash_registry.is_workspace_trusted = self.is_trusted
        n_sk = self.skill_manager.reload()
        n_cmd = self.slash_registry.reload()
        return n_sk, n_cmd

    def switch_session(self, target_session_id: str) -> SessionRunner:
        """Hot-switches the active interactive session, loading its full SessionDAG + long-form `.trace.jsonl` ledger."""
        old_mode = self.runner.mode
        old_effort = self.runner.thinking_effort
        old_prof = self.runner.model_profile
        old_appr = self.runner.approval_callback
        old_ask = self.runner.ask_user_callback
        self.runner.close()
        journal = self.ush_home / "sessions" / f"{target_session_id}.jsonl"
        self.runner = SessionRunner(
            workspace_root=self.workspace,
            client=self.client,
            auth_profile=self.auth_profile,
            model_profile=old_prof,
            mode=old_mode,
            session_id=target_session_id,
            journal_path=journal,
            ush_home=self.ush_home,
            policy_engine=self.policy_engine,
            hook_runner=self.hook_system,
            skill_manager=self.skill_manager,
            memory_manager=self.memory_manager,
            thinking_effort=old_effort,
        )
        self.runner.approval_callback = old_appr
        self.runner.ask_user_callback = old_ask
        self.runner.swarm_callback = lambda args: str(
            self.swarm.blackboard.post(
                str(args.get("event_type", "NOTE")),
                self.runner.session_id,
                str(args.get("summary", "")),
                str(args.get("scope", "*")),
            )
        )
        return self.runner


# ============================================================================
# Diagnostics (`ush doctor` / `/doctor`) & Auth Formatter
# ============================================================================


def render_doctor_report(rt: HarnessRuntime) -> str:
    profs = rt.vault.list_profiles()
    git_ok = shutil.which("git") is not None
    gcloud_ok = shutil.which("gcloud") or str(
        pathlib.Path("~/google-cloud-sdk/bin/gcloud").expanduser()
    )
    has_gcloud = pathlib.Path(gcloud_ok).is_file()
    lines = [
        f"Crucible (crux v{__version__}) Health Check ({platform.system()} {platform.machine()} | Python {platform.python_version()})",
        f"  Workspace         : {rt.workspace} ({'TRUSTED' if rt.is_trusted else 'untrusted quarantine — run `/trust grant` to load repo hooks/skills'})",
        f"  Active Auth       : {rt.runner.auth_profile.name} ({rt.runner.auth_profile.provider} via {rt.runner.auth_profile.source}"
        + (f", project={rt.runner.auth_profile.project_id}" if rt.runner.auth_profile.project_id else "")
        + ")",
        f"  All Auth Profiles : {', '.join(p.name for p in profs) if profs else 'NONE — run `crux auth login`'}",
        f"  Active Model      : {rt.runner.model_profile.model_id} (Tier {rt.runner.model_profile.tier.value}, ctx={rt.runner.model_profile.context_window_tokens:,}, max_parallel_reads={rt.runner.model_profile.max_parallel_read_tools})",
        f"  Git / Shadow Git  : {'OK' if git_ok else 'MISSING'} -> {rt.runner.shadow_git.git_dir}",
        f"  gcloud SDK        : {'OK (' + str(gcloud_ok) + ')' if has_gcloud else 'not installed'}",
        f"  Keychain Vault    : {'macOS Keychain (`security`)' if platform.system() == 'Darwin' else '0600 `~/.ush/credentials.vault.json`'}",
        f"  Loaded Skills     : {len(rt.skill_manager.all_skills())} skills across 4 tiers",
        f"  Slash Commands    : {len(rt.slash_registry.list_commands())} commands (`Builtin + File + Skill + MCP`)",
        f"  Memory Files      : {len(rt.memory_manager.loaded_paths)} loaded | {len(rt.memory_manager.list_inbox())} pending `.inbox` patch(es)",
    ]
    return "\n".join(lines)


def render_auth_status(vault: CredentialVault) -> str:
    profs = vault.list_profiles()
    if not profs:
        return "No authentication profiles configured or auto-discovered.\nRun `crux auth login` (or `gcloud auth application-default login`)."
    active = vault.active_profile_name
    now = time.time()
    lines = ["Configured & Auto-Discovered Authentication Profiles:"]
    for p in profs:
        star = "*" if p.name == active else " "
        extras: list[str] = []
        if p.project_id:
            extras.append(f"project={p.project_id} region={p.region}")
        if p.api_mode:
            extras.append(f"mode={p.api_mode}")
        if p.service_tier:
            extras.append(f"tier={p.service_tier}")
        if p.base_url:
            extras.append(f"url={p.base_url}")
        if p.transport_plugin:
            extras.append(f"plugin={p.transport_plugin}")
        if p.require_host_path and not p.is_available_on_host():
            extras.append("host-gated=inactive")
        if p.cooldown_until > now:
            extras.append(f"cooldown={int(p.cooldown_until - now)}s")
        sfx = f" [{' · '.join(extras)}]" if extras else ""
        lines.append(
            f" {star} {p.name:26s} | provider={p.provider:18s} | kind={p.auth_kind:10s} | src={p.source}{sfx}"
        )
    for warn in getattr(vault, "security_warnings", ()):
        lines.append(f" ! {warn}")
    return "\n".join(lines)


# ============================================================================
# Interactive Terminal TUI + Slash Command Interpreter
# ============================================================================


def _setup_tab_completion(rt: HarnessRuntime) -> None:
    def _completer(text: str, state: int) -> str | None:
        buf = readline.get_line_buffer().lstrip()
        options: list[str] = []
        if text.startswith("/") or (buf.startswith("/") and " " not in buf):
            q = text.lstrip("/")
            options = [f"/{c.name} " for c in rt.slash_registry.list_commands() if c.name.startswith(q)]
        elif text.startswith("$"):
            q = text[1:]
            options = [f"${s.name} " for s in rt.skill_manager.user_invocable_skills() if s.name.startswith(q)]
        elif text.startswith("@"):
            q = text[1:]
            for child in sorted(rt.workspace.iterdir()):
                if child.name.startswith(".") and not q.startswith("."):
                    continue
                if child.name.startswith(q):
                    sfx = "/" if child.is_dir() else " "
                    options.append(f"@{child.name}{sfx}")
        return options[state] if state < len(options) else None

    try:
        readline.set_completer_delims(" \t\n")
        readline.set_completer(_completer)
        if "libedit" in (readline.__doc__ or ""):
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")
    except Exception:  # noqa: BLE001
        pass


def _render_status_bar(rt: HarnessRuntime) -> str:
    r = rt.runner
    ctx_pct = 100.0 * r.dag.approx_context_tokens() / max(1, r.model_profile.context_window_tokens)
    bar = (
        f"[{r.session_id} • {r.model_profile.model_id} • think:{r.thinking_effort.value} "
        f"• trace:{rt.trace_mode.value} • thought:{r.total_usage.reasoning_tokens:,}t "
        f"• ctx:{ctx_pct:.1f}% • mode:{r.mode.value}]"
    )
    return c_dim(bar)


def _interactive_approval(
    rt: HarnessRuntime,
    tool_name: str,
    args: dict[str, Any],
    reason: str,
    stdin_claimed: threading.Event | None = None,
    stdin_lock: threading.Lock | None = None,
) -> bool:
    if stdin_claimed is not None:
        stdin_claimed.set()
    try:
        if stdin_lock is not None:
            stdin_lock.acquire()
        try:
            with rt.print_lock:
                summary = args.get("command") or args.get("path") or json.dumps(args)[:120]
                sys.stdout.write(f"\n{c_yellow('? Approval required')} [{c_bold(tool_name)}]: `{summary}`\n")
                sys.stdout.write(c_dim(f"  Reason: {reason}\n"))
                sys.stdout.flush()
            ans = input("  Allow? [y]es / [e] auto-edit session / [n]o (default=n): ").strip().lower()
        finally:
            if stdin_lock is not None:
                stdin_lock.release()
    finally:
        if stdin_claimed is not None:
            stdin_claimed.clear()
    if ans == "e":
        rt.runner.mode = PermissionMode.AUTO_EDIT
        return True
    return ans in ("y", "yes")


def _stream_printer(rt: HarnessRuntime, mode_override: TraceMode | None = None) -> StreamTracePrinter:
    return StreamTracePrinter(
        mode=mode_override or rt.trace_mode,
        use_color=USE_COLOR,
        out=sys.stdout,
        lock=rt.print_lock,
    )


def handle_slash_command(
    rt: HarnessRuntime, line: str, *, headless: bool = False
) -> tuple[bool, str | None, list[CanonicalPart]]:
    """Returns `(should_exit, prompt_to_submit_to_model_or_None, extra_parts)`."""
    parts = line[1:].strip().split(None, 1)
    cmd_name = parts[0].lower() if parts else "help"
    arg_str = parts[1].strip() if len(parts) > 1 else ""

    cmd = rt.slash_registry.get(cmd_name)
    if cmd is None:
        print(c_red(f"Unknown slash command `/{cmd_name}`. Type `/help` to list all commands."))
        return False, None, []

    # Custom `.toml`/`.md` File or MCP Prompt command -> safe 3-stage macro expansion
    if cmd.kind in (CommandKind.FILE, CommandKind.MCP_PROMPT):
        expanded, mparts = expand_command_template(
            cmd.template,
            arg_str,
            firewall=rt.runner.firewall,
            policy_engine=rt.policy_engine,
            mode=rt.runner.mode,
            approval_callback=(
                None
                if headless
                else lambda scmd, rsn: _interactive_approval(rt, "macro_shell", {"command": scmd}, rsn)
            ),
        )
        return False, expanded, mparts

    # `/skill-name [args]` -> directly activates skill & forwards optional task prompt
    if cmd.kind == CommandKind.SKILL and cmd.skill_name:
        activated = rt.skill_manager.activate(cmd.skill_name, rt.runner.firewall)
        prompt = f"{activated}\n\nUser request: {arg_str or f'Apply `{cmd.skill_name}` skill to the workspace.'}"
        return False, prompt, []

    # Built-in commands
    if cmd_name in ("quit", "exit"):
        return True, None, []

    if cmd_name == "help":
        print(c_bold("\nBuilt-in, Custom File, Skill & MCP Slash Commands:"))
        for sc in rt.slash_registry.list_commands():
            during = c_green(" [live]") if sc.available_during_task else ""
            print(f"  /{sc.name:20s} {sc.description}{during}")
        print(c_dim("\nShortcuts: `!cmd` runs in persistent PTY • `$skill` activates skill inline • `Tab` autocompletes"))
        return False, None, []

    if cmd_name == "auth":
        sub = arg_str.split()
        action = sub[0] if sub else "status"
        if action == "status":
            print(render_auth_status(rt.vault))
        elif action == "login":
            prov = sub[1] if len(sub) > 1 else None
            prof = interactive_login_wizard(rt.vault, prov)
            rt.runner.switch_model(prof.default_model or rt.runner.model_profile.model_id, prof)
            print(c_green(f"Authenticated & switched to `{prof.name}` ({prof.provider})."))
        elif action == "switch" and len(sub) > 1:
            prof = rt.vault.switch_profile(sub[1])
            rt.runner.switch_model(prof.default_model or rt.runner.model_profile.model_id, prof)
            print(c_green(f"Switched active auth profile to `{prof.name}`."))
        elif action == "logout" and len(sub) > 1:
            ok = rt.vault.logout(sub[1])
            print(f"Logged out `{sub[1]}`: {ok}")
        else:
            print("Usage: `/auth [status | login [provider] | switch <profile> | logout <profile>]`")
        return False, None, []

    if cmd_name == "model":
        from ush.auth.vault import discover_model_options, format_model_display_name

        if not arg_str:
            cur_m = rt.runner.model_profile.model_id
            avail = discover_model_options(rt.vault, rt.runner.auth_profile, timeout=1.2)
            for opt in avail:
                is_cur = opt.model_id == cur_m or cur_m in (f"{opt.model_id}-high", f"{opt.model_id}-medium", f"{opt.model_id}-low")
                mark = ">" if is_cur else " "
                cur_sfx = c_dim("   (current)") if is_cur else ""
                print(f"{mark} {opt.display_name}{cur_sfx}")
        else:
            np = rt.runner.switch_model(arg_str)
            print(c_green(f"Model -> {format_model_display_name(np.model_id)}"))
        return False, None, []

    if cmd_name == "thinking":
        if not arg_str:
            print(
                f"Active thinking effort: `{rt.runner.thinking_effort.value}` "
                f"(options: `off | low | medium | high | max`)"
            )
            return False, None, []
        try:
            eff = ThinkingEffort(arg_str.lower())
        except ValueError:
            print("Usage: `/thinking [off | low | medium | high | max]`")
            return False, None, []
        rt.runner.set_thinking_effort(eff)
        print(c_green(f"Thinking effort -> `{eff.value}` (applies on the very next model invocation)."))
        return False, None, []

    if cmd_name == "steps":
        if not arg_str or not arg_str.isdigit():
            print(f"Active max steps per turn: `{rt.max_steps}` (Usage: `/steps <1..200>`)")
            return False, None, []
        rt.max_steps = max(1, min(200, int(arg_str)))
        print(c_green(f"Max reasoning/tool steps per turn -> `{rt.max_steps}`"))
        return False, None, []

    if cmd_name == "threads":
        ledger = rt.runner.trace_ledger
        print(ledger.format_trace_view(arg_str or "all") if ledger else "No thinking threads recorded yet.")
        return False, None, []

    if cmd_name == "steer":
        if not arg_str:
            print("Usage: `/steer <guidance>`")
            return False, None, []
        rt.runner.enqueue_steer(arg_str)
        print(c_green(f"Queued live mid-turn steer (#{rt.runner.pending_steer_count()}): {arg_str[:80]}"))
        return False, None, []

    if cmd_name == "queue":
        print("Follow-up prompt queue is active automatically in visual interactive mode (`Enter` while model thinks).")
        return False, None, []

    if cmd_name == "trace":
        ledger = rt.runner.trace_ledger
        sub_parts = arg_str.split(None, 1)
        sub_cmd = sub_parts[0].lower() if sub_parts else "stats"
        sub_arg = sub_parts[1].strip() if len(sub_parts) > 1 else ""
        if sub_cmd in ("full", "compact", "off"):
            rt.trace_mode = TraceMode(sub_cmd)
            print(c_green(f"Live thought trace verbosity -> `{rt.trace_mode.value}` (uncompacted `.trace.jsonl` always saved)."))
        elif sub_cmd == "path":
            print(str(ledger.trace_path) if ledger else "(in-memory session — no trace file)")
        elif sub_cmd == "stats":
            print(ledger.stats_summary() if ledger else "No trace ledger attached.")
        elif sub_cmd == "show":
            print(ledger.format_trace_view(sub_arg or "last") if ledger else "No trace ledger attached.")
        elif sub_cmd == "search":
            if not sub_arg:
                print("Usage: `/trace search <query>`")
            elif ledger:
                hits = ledger.search(sub_arg)
                if not hits:
                    print(f"No matches in session thoughts for `{sub_arg}`.")
                for h in hits:
                    print(f"  [turn {h['turn']} step {h['step']} • {h['type']} • {h['reasoning_tokens']}t] {h['snippet']}")
        elif sub_cmd == "export":
            if ledger:
                dest = pathlib.Path(sub_arg) if sub_arg else None
                out_f = ledger.export_markdown(dest)
                print(c_green(f"Exported full uncompacted Markdown reasoning trace -> {out_f}"))
        else:
            print(
                "Usage: `/trace [full | compact | off | show [last|all|<step>|t<turn>:<step>] "
                "| search <query> | stats | export [out.md] | path]`"
            )
        return False, None, []

    if cmd_name == "sessions":
        # Default to ALL recent sessions sorted strictly newest-first (`/sessions local` filters to cwd)
        ws_f = rt.workspace if arg_str.strip().lower() == "local" else None
        slist = list_workspace_sessions(rt.ush_home / "sessions", workspace_filter=ws_f, include_empty=False)
        if not slist:
            print("No saved sessions found yet.")
            return False, None, []
        for s in slist[:30]:
            cur = "*" if s.session_id == rt.runner.session_id else " "
            age = format_relative_age(s.updated_at)
            m_lbl = s.models[-1] if s.models else "-"
            ws_lbl = pathlib.Path(s.workspace).name if s.workspace else "-"
            print(
                f" {cur} {age:<8s} {s.session_id}  {s.turns:>2d}t/{s.steps:<3d}s "
                f"◈ {s.thought_blocks:<2d} ({s.reasoning_tokens:,}t)  [{m_lbl} · {ws_lbl}]  {s.last_prompt[:52]}"
            )
        print(c_dim("Sorted most-recent-first · `/resume` opens picker (`--plain`: `/resume last` or `/resume <id>`)."))
        return False, None, []

    if cmd_name == "resume":
        target_id = arg_str.strip()
        # Pull ALL recent sessions sorted strictly newest-first so cross-directory sessions are never hidden
        slist = list_workspace_sessions(rt.ush_home / "sessions", workspace_filter=None, include_empty=False)
        if not target_id or target_id == "last":
            candidates = [s for s in slist if s.session_id != rt.runner.session_id and (s.turns > 0 or s.steps > 0)]
            if not candidates:
                print("No previous session found to resume.")
                return False, None, []
            target_id = candidates[0].session_id
        dag_file = rt.ush_home / "sessions" / f"{target_id}.jsonl"
        trace_file = rt.ush_home / "sessions" / f"{target_id}.trace.jsonl"
        if not dag_file.is_file() and not trace_file.is_file():
            print(c_red(f"Session `{target_id}` not found in `{rt.ush_home / 'sessions'}`."))
            return False, None, []
        matched_summary = next((s for s in slist if s.session_id == target_id), None)
        rt.switch_session(target_id)
        if matched_summary and matched_summary.models:
            # Sync active model to the session's last used model when still on startup default
            default_mid = rt.auth_profile.default_model or "gemini-3.1-pro-preview"
            if rt.runner.model_profile.model_id == default_mid and matched_summary.models[-1] != default_mid:
                rt.runner.switch_model(matched_summary.models[-1])
        msgs = len(rt.runner.dag.active_messages())
        print(
            c_green(
                f"Resumed `{target_id}` ({msgs} active messages · model={rt.runner.model_profile.model_id} "
                f"· trace={rt.runner.trace_ledger.trace_path if rt.runner.trace_ledger else 'none'})."
            )
        )
        return False, None, []

    if cmd_name == "profile":
        print(json.dumps(rt.runner.model_profile.__dict__, default=str, indent=2))
        return False, None, []

    if cmd_name == "plan":
        rt.runner.mode = (
            PermissionMode.DEFAULT if rt.runner.mode == PermissionMode.PLAN else PermissionMode.PLAN
        )
        print(c_cyan(f"Switched mode -> `{rt.runner.mode.value}`."))
        if arg_str:
            rt.runner.mode = PermissionMode.PLAN
            return False, f"[PLAN MODE] Inspect the codebase and write a concrete step-by-step architecture plan for: {arg_str}", []
        return False, None, []

    if cmd_name == "permissions":
        norm_p = arg_str.strip().lower()
        alias_map = {"autonomous": "yolo", "turbo": "yolo", "supervised": "default", "auto-edit": "auto_edit"}
        norm_p = alias_map.get(norm_p, norm_p)
        try:
            rt.runner.mode = PermissionMode(norm_p)
            print(c_cyan(f"Execution mode -> `{norm_p}`"))
        except ValueError:
            print("Usage: `/permissions plan | supervised | auto_edit | autonomous`")
        return False, None, []

    if cmd_name == "turbo":
        sub = arg_str.strip().lower()
        if not sub:
            sub = "off" if rt.runner.mode == PermissionMode.YOLO else "on"
        if sub == "off":
            rt.runner.mode = PermissionMode.DEFAULT
            rt.runner.set_thinking_effort(ThinkingEffort.HIGH)
            rt.max_steps = 80
            print(c_cyan("Turbo OFF -> mode=`supervised` • thinking=`high` • steps=`80`"))
        elif sub == "on":
            rt.runner.mode = PermissionMode.YOLO
            rt.max_steps = max(rt.max_steps, 160)
            print(
                c_green(
                    f"▸▸ Turbo ON (autonomous; hard security firewall active) "
                    f"• thinking=`{rt.runner.thinking_effort.value}` • steps=`{rt.max_steps}`"
                )
            )
        elif sub == "max":
            rt.runner.mode = PermissionMode.YOLO
            rt.runner.set_thinking_effort(ThinkingEffort.MAX)
            rt.max_steps = 200
            print(c_green("✦ Turbo MAX (autonomous) • thinking=`max` • steps=`200`"))
        elif sub == "fast":
            rt.runner.mode = PermissionMode.YOLO
            rt.runner.set_thinking_effort(ThinkingEffort.LOW)
            if rt.runner.model_profile.family == "gemini" and "flash" not in rt.runner.model_profile.model_id:
                rt.runner.switch_model("gemini-3-flash-preview")
            print(
                c_green(
                    f"» Turbo FAST (autonomous) • model=`{rt.runner.model_profile.model_id}` • thinking=`low`"
                )
            )
        else:
            print("Usage: `/turbo [on | fast | max | off]`")
        return False, None, []

    if cmd_name == "btw":
        if not arg_str:
            print("Usage: `/btw <side question>`")
            return False, None, []
        print(c_cyan("[/btw zero-history KV-cached side answer]:"))
        rt.runner.ask_btw(arg_str, on_event=_stream_printer(rt, mode_override=TraceMode.COMPACT))
        return False, None, []

    if cmd_name == "grill-me":
        act = rt.skill_manager.activate("grill-me", rt.runner.firewall)
        return False, f"{act}\n\n{arg_str or 'Stress-test our current plan and design assumptions.'}", []

    if cmd_name == "goal":
        if not arg_str:
            print("Usage: `/goal <long-running objective>`")
            return False, None, []
        return (
            False,
            (
                "[LONG-FORM SCIENTIFIC GOAL MODE]\n"
                "Execute an exhaustive multi-step Popperian investigation:\n"
                "1. Formulate explicit hypotheses and invariants in your thinking before touching code.\n"
                "2. Read exact source files/tests and attempt to falsify your first hypothesis before committing.\n"
                "3. Implement minimal surgical changes (if needed) and empirically verify via tests/commands.\n"
                "4. Do NOT stop early until every requirement and edge case is verified.\n\n"
                f"Objective: {arg_str}"
            ),
            [],
        )

    if cmd_name == "review":
        first_snap = next((c.shadow_snapshot for c in rt.runner.checkpoints if c.shadow_snapshot), None)
        diff = rt.runner.shadow_git.diff_from(first_snap) if first_snap else ""
        if not diff.strip():
            res = rt.runner.pty.execute("git diff HEAD", wait_seconds=10.0)
            diff = res.output
        if not diff.strip():
            print("No uncommitted changes detected to review.")
            return False, None, []
        cand = CandidatePatch("local_head", (rt.runner.session_id,), rt.runner.model_profile.family, diff)
        sa = rt.swarm.judge.run_stage_a(cand)
        print(c_bold(f"Stage-A Clean-Patch Sieve: {'PASS' if sa.passed else 'FAIL'}"))
        for r in sa.reasons:
            print(f"  - {c_red(r)}")
        verdict = rt.swarm.judge.run_stage_b_bars(arg_str or "Code review of current workspace diff", cand)
        print(
            c_bold(
                f"Stage-B Blind Judge ({verdict.judge_id} | family={verdict.judge_family}): "
                f"BARS={verdict.score:.2f} ({'PASS' if verdict.passed else 'NEEDS_WORK'})"
            )
        )
        print(f"  Criteria : {verdict.criterion_scores}")
        print(f"  Critique : {verdict.critique}")
        return False, None, []

    if cmd_name == "swarm":
        if not arg_str:
            print("Usage: `/swarm [auto|g1|g2|g3|g4|peer|argonclaw] <task>`")
            return False, None, []
        toks = arg_str.split(None, 1)
        if toks[0].lower() in ("peer", "--peer-cli") and len(toks) > 1:
            peer_res = rt.swarm.dispatch_to_peer_cli(toks[1])
            if peer_res.conversation_id:
                print(c_green(f"Peer conversation: `{peer_res.conversation_id}`"))
            print(peer_res.output or f"Peer CLI exited with status {peer_res.returncode}")
            return False, None, []
        if toks[0].lower() == "argonclaw" and len(toks) > 1:
            rc, out = rt.swarm.dispatch_to_argonclaw_cluster(
                task_id=new_id("ush"), title=toks[1][:64], description=toks[1]
            )
            print(out or f"Dispatched (exit={rc})")
            return False, None, []
        g = Gear.AUTO
        task_txt = arg_str
        if toks[0].lower() in ("auto", "g1", "g2", "g3", "g4") and len(toks) > 1:
            g = Gear(toks[0].lower())
            task_txt = toks[1]
        rt.gear = select_initial_gear(task_txt, g)
        print(c_cyan(f"[Gearbox selected `{rt.gear.value.upper()}` for task]"))
        return False, task_txt, []

    if cmd_name == "arena":
        toks = arg_str.split(None, 1)
        if len(toks) < 2 or "," not in toks[0]:
            print("Usage: `/arena modelA,modelB <task>`")
            return False, None, []
        models = [m.strip() for m in toks[0].split(",") if m.strip()]
        task_txt = toks[1]
        _run_arena_race(rt, models, task_txt)
        return False, None, []

    if cmd_name == "restore":
        rmode = arg_str if arg_str in ("code+chat", "code-only", "chat-only") else "code+chat"
        files, lesson = rt.runner.restore_checkpoint(-1, mode=rmode, reason="User triggered /restore")
        print(c_green(f"Restored ({rmode}) — reverted {len(files)} file(s): {', '.join(files[:8])}"))
        if lesson:
            print(c_dim(lesson.plain_text()))
        return False, None, []

    if cmd_name == "tree":
        nodes = rt.runner.dag.list_nodes()
        leaf = rt.runner.dag.leaf_id
        print(c_bold(f"Session DAG ({len(nodes)} node(s), active leaf={leaf}):"))
        for n in nodes[-20:]:
            mark = "*" if n.entry_id == leaf else " "
            prev = n.plain_text().replace("\n", " ")[:72]
            print(f" {mark} {n.entry_id} <- {str(n.parent_entry_id):17s} [{n.role:9s}] {prev}")
        return False, None, []

    if cmd_name == "compact":
        did = rt.runner.dag.maybe_compact(context_limit_tokens=1, ratio=0.01)
        print(c_green("Epistemic compaction committed.") if did else "History too short to compact.")
        return False, None, []

    if cmd_name == "clear":
        rt.runner.dag = type(rt.runner.dag)(rt.runner.dag.journal_path)
        print(c_green("Cleared active session context."))
        return False, None, []

    if cmd_name == "diff":
        first_snap = next((c.shadow_snapshot for c in rt.runner.checkpoints if c.shadow_snapshot), None)
        d = rt.runner.shadow_git.diff_from(first_snap) if first_snap else ""
        if not d.strip():
            d = rt.runner.pty.execute("git diff HEAD", wait_seconds=10.0).output
        print(d if d.strip() else "Working tree matches session start.")
        return False, None, []

    if cmd_name == "skills":
        for s in rt.skill_manager.all_skills():
            p_info = f" paths={list(s.paths)}" if s.paths else ""
            print(f"  - {c_bold(s.name):24s} [{s.tier:9s}]{p_info} — {s.description[:80]}")
        return False, None, []

    if cmd_name == "commands":
        for sc in rt.slash_registry.list_commands():
            print(f"  /{sc.name:22s} ({sc.kind.value:7s} | {sc.source}) — {sc.description[:72]}")
        for notice in rt.slash_registry.collision_notices:
            print(c_yellow(f"  [notice] {notice}"))
        return False, None, []

    if cmd_name == "hooks":
        if not rt.hook_system.shell_hooks:
            print("No shell hooks configured (`~/.ush/hooks.json` or trusted `.ush/hooks.json`).")
        for h in rt.hook_system.shell_hooks:
            print(f"  - {h.event:20s} matcher=`{h.matcher}` ({h.source}) -> `{h.command}`")
        return False, None, []

    if cmd_name == "policies":
        for r in rt.policy_engine.rules:
            print(f"  [{r.tier:9s} p={r.priority:3d}] {r.decision.value:8s} tool=`{r.tool_pattern}` prefixes={list(r.command_prefixes)} ({r.reason})")
        for w in rt.policy_engine.clamped_warnings:
            print(c_yellow(f"  [clamped] {w}"))
        return False, None, []

    if cmd_name == "trust":
        if arg_str == "grant":
            fp = rt.trust_store.trust(rt.workspace)
            n_sk, n_cmd = rt.reload_extensibility()
            print(c_green(f"Trusted `{rt.workspace}` (sha256={fp[:16]}...); reloaded {n_sk} skills & {n_cmd} commands."))
        elif arg_str == "revoke":
            rt.trust_store.untrust(rt.workspace)
            rt.reload_extensibility()
            print(c_yellow(f"Revoked workspace trust for `{rt.workspace}`."))
        else:
            print(f"Workspace trust: {'TRUSTED' if rt.is_trusted else 'UNTRUSTED'} (`/trust grant` or `/trust revoke`)")
        return False, None, []

    if cmd_name == "memory":
        toks = arg_str.split()
        act = toks[0] if toks else "show"
        if act == "inbox":
            inbox = rt.memory_manager.list_inbox()
            if not inbox:
                print("Memory `.inbox` is empty.")
            for p in inbox:
                print(f"  [{p.patch_id}] -> {p.target_file}\n{p.diff_text}")
        elif act == "apply" and len(toks) > 1:
            ok = rt.memory_manager.apply_inbox_patch(toks[1])
            print(c_green(f"Applied `{toks[1]}`.") if ok else c_red(f"Patch `{toks[1]}` not found."))
        else:
            loaded = rt.memory_manager.loaded_paths
            print("Loaded 4-scope memory files:\n" + ("\n".join(f"  - {x}" for x in loaded) or "  (none)"))
        return False, None, []

    if cmd_name == "learn":
        if "::" in arg_str:
            title, fact = [x.strip() for x in arg_str.split("::", 1)]
        else:
            title, fact = "Session Insight", (arg_str or f"Workspace `{rt.workspace.name}` durable notes")
        prop = rt.memory_manager.stage_learning_patch(title, [fact])
        print(c_green(f"Staged secret-redacted patch `{prop.patch_id}` into `/memory inbox`:\n{prop.diff_text}"))
        return False, None, []

    if cmd_name == "extensions":
        toks = arg_str.split(None, 1)
        if toks and toks[0] == "install" and len(toks) > 1:
            ext = rt.ext_manager.install_local(pathlib.Path(toks[1]))
            print(c_green(f"Installed extension `{ext.name}` v{ext.version} -> {ext.root}"))
        else:
            if not rt.ext_manager.extensions:
                print("No extensions installed (`~/.ush/extensions/`).")
            for e in rt.ext_manager.extensions:
                print(f"  - {e.name} v{e.version} ({e.root})")
        return False, None, []

    if cmd_name == "stats":
        u = rt.runner.total_usage
        efr = (100.0 * rt.runner.edits_failed / rt.runner.edits_attempted) if rt.runner.edits_attempted else 0.0
        print(
            f"Tokens: in={u.input_tokens:,} (cached={u.cached_input_tokens:,}, hit={100*u.cache_hit_rate:.1f}%) "
            f"| out={u.output_tokens:,} | reasoning={u.reasoning_tokens:,}\n"
            f"KPIs  : edits={rt.runner.edits_attempted} (failure_rate={efr:.1f}%) | compactions={rt.runner.dag.compaction_count}"
        )
        return False, None, []

    if cmd_name == "doctor":
        print(render_doctor_report(rt))
        return False, None, []

    if cmd_name == "reload":
        n_sk, n_cmd = rt.reload_extensibility()
        print(c_green(f"Reloaded {n_sk} skills and {n_cmd} slash commands."))
        return False, None, []

    return False, None, []


def _run_arena_race(rt: HarnessRuntime, models: list[str], task: str) -> None:
    print(c_cyan(f"Starting `/arena` across {len(models)} isolated Git worktrees: {', '.join(models)}"))
    wts = [rt.swarm.worktrees.create(f"arena_{i}_{m.replace('/', '_')}") for i, m in enumerate(models)]
    candidates: list[CandidatePatch] = []
    try:
        for m, wt in zip(models, wts):
            prof = resolve_model_profile(m, rt.runner.auth_profile.provider)
            sub = SessionRunner(
                workspace_root=wt.path,
                client=rt.client,
                auth_profile=rt.runner.auth_profile,
                model_profile=prof,
                mode=PermissionMode.AUTO_EDIT,
                ush_home=rt.ush_home,
            )
            try:
                sub.run_turn(task, max_steps=20)
            finally:
                sub.close()
            diff = rt.swarm.worktrees.worktree_diff(wt)
            candidates.append(CandidatePatch(m, (f"builder:{m}",), prof.family, diff, str(wt.path)))

        valid = [c for c in candidates if rt.swarm.judge.run_stage_a(c).passed]
        if not valid:
            print(c_red("No arena candidate produced a Stage-A clean diff."))
            return
        cmp_fn = rt.swarm.judge.build_llm_pairwise_comparator(task)
        ranked = rt.swarm.judge.rank_bradley_terry(valid, cmp_fn)
        print(c_bold("\nPosition-Swapped Bradley-Terry Arena Standings:"))
        for idx, (cand, strength) in enumerate(ranked, 1):
            print(f"  #{idx} {cand.candidate_id:28s} strength={strength:.3f} ({len(cand.unified_diff)} diff chars)")
    finally:
        for wt in wts:
            rt.swarm.worktrees.cleanup(wt)


AT_INLINE_FILE_RE = re.compile(r"(?:(?<=^)|(?<=\s))@([A-Za-z0-9_./-]{1,200})")


def expand_user_prompt_pills(rt: HarnessRuntime, prompt_text: str) -> tuple[str, list[CanonicalPart]]:
    """Safely expands inline `$skill-name` triggers & `@path/to/file` attachments via `SecurityFirewall`."""
    expanded, skill_blocks = rt.skill_manager.expand_inline_dollar_mentions(prompt_text, rt.runner.firewall)
    extra_parts: list[CanonicalPart] = []
    file_blocks: list[str] = []
    seen_files: set[str] = set()

    for m in AT_INLINE_FILE_RE.finditer(expanded):
        rel_cand = m.group(1).rstrip(".,;:)")
        if not rel_cand or rel_cand in seen_files:
            continue
        try:
            resolved = rt.runner.firewall.resolve_and_validate(rel_cand, write=False)
        except Exception:  # noqa: BLE001
            continue
        if not resolved.is_file():
            continue
        seen_files.add(rel_cand)
        suf = resolved.suffix.lower()
        if suf in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            import base64
            import mimetypes

            raw = resolved.read_bytes()
            if len(raw) <= 8 * 1024 * 1024:
                extra_parts.append(
                    CanonicalPart(
                        kind="image",
                        mime_type=mimetypes.guess_type(resolved.name)[0] or "image/png",
                        base64_data=base64.b64encode(raw).decode("ascii"),
                        alt_text=rel_cand,
                    )
                )
        else:
            txt = resolved.read_text(encoding="utf-8", errors="replace")[:49_152]
            file_blocks.append(f'<attached_file path="{rel_cand}">\n{txt}\n</attached_file>')

    prefixes = [*skill_blocks, *file_blocks]
    if prefixes:
        expanded = "\n\n".join(prefixes) + "\n\n" + expanded
    return expanded, extra_parts


def _load_and_bind_history(ush_home: pathlib.Path) -> pathlib.Path:
    hist_path = ush_home / "readline_history"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    if not hist_path.exists():
        fd = os.open(str(hist_path), os.O_WRONLY | os.O_CREAT, 0o600)
        os.close(fd)
    try:
        readline.set_history_length(2_000)
        readline.read_history_file(str(hist_path))
    except Exception:  # noqa: BLE001
        pass
    return hist_path


def _save_history_safe(hist_path: pathlib.Path) -> None:
    try:
        readline.write_history_file(str(hist_path))
        os.chmod(hist_path, 0o600)
    except Exception:  # noqa: BLE001
        pass


def _read_interactive_input(prompt_str: str, is_tty: bool) -> str:
    """Reads interactive prompt, coalescing multi-line terminal pastes & `\"\"\"` / trailing-`\\` blocks cleanly."""
    first = input(prompt_str)
    stripped = first.strip()

    # Explicit `"""` fenced multiline entry (helpful for long proofs, specs, and tracebacks)
    if stripped.startswith('"""') and (stripped == '"""' or not stripped.endswith('"""') or len(stripped) == 3):
        lines = [first[first.find('"""') + 3 :]]
        while True:
            nxt = input("... ")
            if nxt.rstrip().endswith('"""'):
                lines.append(nxt.rstrip()[:-3])
                break
            lines.append(nxt)
        return "\n".join(lines).strip()

    lines = [first]
    while lines[-1].endswith("\\"):
        lines[-1] = lines[-1][:-1]
        lines.append(input("... "))

    # Paste-burst coalescing on interactive TTY (`select` drains buffered lines pasted at once!)
    if is_tty and first and not stripped.startswith("/") and not stripped.startswith("!"):
        try:
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], 0.015)
                if not ready:
                    break
                extra = sys.stdin.readline()
                if not extra:
                    break
                lines.append(extra.rstrip("\r\n"))
        except (OSError, ValueError):
            pass
    return "\n".join(lines)


def _handle_mid_turn_line(rt: HarnessRuntime, line: str, printer: StreamTracePrinter) -> None:
    """Processes live user input typed WHILE a long-form multi-step reasoning turn is actively running."""
    s = line.strip()
    if not s:
        return
    if s.startswith("/"):
        parts = s[1:].strip().split(None, 1)
        cname = parts[0].lower() if parts else ""
        carg = parts[1].strip() if len(parts) > 1 else ""
        if cname == "thinking" and carg:
            try:
                eff = ThinkingEffort(carg.lower())
                rt.runner.set_thinking_effort(eff)
                printer.emit_external_note(f"Thinking effort switched live -> `{eff.value}` (takes effect next step)")
                return
            except ValueError:
                printer.emit_external_note("Usage: `/thinking off|low|medium|high|max`")
                return
        if cname == "trace" and carg in ("full", "compact", "off"):
            rt.trace_mode = TraceMode(carg)
            printer.mode = rt.trace_mode
            printer.emit_external_note(f"Live trace display -> `{rt.trace_mode.value}`")
            return
        if cname in ("stats", "trace", "profile", "tree", "btw"):
            with rt.print_lock:
                sys.stdout.write("\n")
                handle_slash_command(rt, s, headless=False)
            return
        printer.emit_external_note(
            f"`/{cname}` waits for idle; type plain text to steer live, `/thinking <lvl>`, `/trace full|compact|off`, `/btw <q>`, or Ctrl+C to stop turn."
        )
        return

    # Plain text typed during active execution -> protocol-safe TurnSteerQueue!
    rt.runner.enqueue_steer(s)
    qlen = rt.runner.pending_steer_count()
    printer.emit_external_note(
        f"Live steer queued (#{qlen}) -> will inject cleanly at next safe step boundary: {s[:80]}"
    )


def _run_turn_interactive(
    rt: HarnessRuntime,
    prompt_text: str,
    extra_parts: list[CanonicalPart],
    *,
    is_tty: bool,
    stdin_claimed: threading.Event,
    stdin_lock: threading.Lock,
) -> None:
    printer = _stream_printer(rt)
    if not is_tty:
        # Piped/scripted stdin: execute synchronously so subsequent piped REPL commands stay ordered
        rt.runner.run_turn(
            prompt_text,
            extra_parts=extra_parts,
            max_steps=rt.max_steps,
            on_event=printer,
        )
        return

    err_holder: list[BaseException] = []
    done_evt = threading.Event()

    def _worker() -> None:
        try:
            rt.runner.run_turn(
                prompt_text,
                extra_parts=extra_parts,
                max_steps=rt.max_steps,
                on_event=printer,
            )
        except BaseException as exc:  # noqa: BLE001
            err_holder.append(exc)
        finally:
            done_evt.set()

    worker = threading.Thread(target=_worker, name="ush-turn-runner", daemon=True)
    worker.start()

    try:
        while not done_evt.is_set():
            if stdin_claimed.is_set():
                done_evt.wait(timeout=0.05)
                continue
            try:
                ready, _, _ = select.select([sys.stdin], [], [], 0.08)
            except (OSError, ValueError):
                done_evt.wait(timeout=0.08)
                continue
            if not ready or stdin_claimed.is_set() or done_evt.is_set():
                continue
            if not stdin_lock.acquire(blocking=False):
                continue
            try:
                if stdin_claimed.is_set() or done_evt.is_set():
                    continue
                live_line = sys.stdin.readline()
            finally:
                stdin_lock.release()
            if live_line:
                _handle_mid_turn_line(rt, live_line, printer)
    except KeyboardInterrupt:
        rt.runner.cancel_event.set()
        printer.emit_external_note("Canceling active turn cleanly (preserving completed thoughts & closing tools)...")
        worker.join(timeout=4.0)
        return

    worker.join(timeout=1.0)
    if err_holder:
        exc = err_holder[0]
        if isinstance(exc, KeyboardInterrupt):
            rt.runner.cancel_event.set()
            print(c_yellow("\n[Turn canceled cleanly]"))
        else:
            print(c_red(f"\n[Error]: {redact_secrets(str(exc))}"))


def run_interactive_tui(rt: HarnessRuntime) -> int:
    is_tty = sys.stdin.isatty()
    stdin_claimed = threading.Event()
    stdin_lock = threading.Lock()

    def _ask_user_cb(q: str, opts: list[str]) -> str:
        stdin_claimed.set()
        try:
            with stdin_lock:
                with rt.print_lock:
                    opt_lines = "\n".join(f"  [{i + 1}] {o}" for i, o in enumerate(opts))
                    sys.stdout.write(
                        f"\n{c_cyan('? Question')}: {q}\n"
                        + (f"{opt_lines}\n" if opt_lines else "")
                    )
                    sys.stdout.flush()
                raw_ans = input("  Select 1-9 or type custom answer: ").strip()
                if raw_ans.isdigit():
                    idx = int(raw_ans) - 1
                    if 0 <= idx < len(opts):
                        return str(opts[idx])
                return raw_ans
        finally:
            stdin_claimed.clear()

    rt.runner.approval_callback = lambda t, a, r: _interactive_approval(
        rt, t, a, r, stdin_claimed=stdin_claimed, stdin_lock=stdin_lock
    )
    rt.runner.ask_user_callback = _ask_user_cb

    hist_path: pathlib.Path | None = None
    if is_tty:
        _setup_tab_completion(rt)
        hist_path = _load_and_bind_history(rt.ush_home)
    rt.hook_system.fire("SessionStart", {"workspace": str(rt.workspace)})

    trace_file = rt.runner.trace_ledger.trace_path if rt.runner.trace_ledger else None
    print(
        f"{c_bold('crux')} v{__version__} (Crucible) • interactive long-form mode "
        f"(thinking={rt.runner.thinking_effort.value}, trace={rt.trace_mode.value}, max_steps={rt.max_steps})"
    )
    print(
        c_dim(
            f"workspace: {rt.workspace}\n"
            f"trace log: {trace_file}\n"
            f"controls : type mid-turn to steer live • Ctrl+C cancels turn • `\"\"\"` multiline • `/trace` • `/thinking` • `/btw`"
        )
    )

    try:
        while True:
            try:
                print(_render_status_bar(rt))
                raw = _read_interactive_input("crux > ", is_tty=is_tty)
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print(c_dim("\n(Ctrl+D or `/quit` to exit session)"))
                continue

            line = raw.strip()
            if not line:
                continue

            # Instant `!cmd` shell escape inside the persistent stateful PTY
            if line.startswith("!"):
                sh_cmd = line[1:].strip()
                if sh_cmd:
                    res = rt.runner.pty.execute(sh_cmd, wait_seconds=30.0)
                    if res.output:
                        print(res.output)
                    print(c_dim(f"[exit={res.exit_code} cwd={res.cwd}]"))
                continue

            extra_parts: list[CanonicalPart] = []
            if line.startswith("/"):
                try:
                    should_exit, prompt_or_none, extra_parts = handle_slash_command(rt, line, headless=False)
                except Exception as exc:  # noqa: BLE001
                    print(c_red(f"Command error: {exc}"))
                    continue
                if should_exit:
                    break
                if not prompt_or_none:
                    continue
                prompt_text = prompt_or_none
            else:
                prompt_text = line

            # Expand inline `$skill-name` mentions
            prompt_text, skill_blocks = rt.skill_manager.expand_inline_dollar_mentions(
                prompt_text, rt.runner.firewall
            )
            if skill_blocks:
                prompt_text = "\n\n".join(skill_blocks) + "\n\n" + prompt_text

            if rt.runner.auth_profile.name == "unconfigured":
                print(c_yellow("No active model credentials found. Run `/auth login` first (or `gcloud auth application-default login`)."))
                continue

            try:
                _run_turn_interactive(
                    rt,
                    prompt_text,
                    extra_parts,
                    is_tty=is_tty,
                    stdin_claimed=stdin_claimed,
                    stdin_lock=stdin_lock,
                )
            except KeyboardInterrupt:
                rt.runner.cancel_event.set()
                print(c_yellow("\n[Turn interrupted cleanly — synthetic tool closure preserved]"))
            except Exception as exc:  # noqa: BLE001
                print(c_red(f"\n[Error]: {redact_secrets(str(exc))}"))
    finally:
        if hist_path is not None:
            _save_history_safe(hist_path)
        rt.runner.close()
    return 0


# ============================================================================
# Headless Unix Mode (`ush -p "..." --output-format text|json|stream-json`)
# ============================================================================


def run_headless(
    rt: HarnessRuntime, prompt: str, *, output_format: str = "text", max_turns: int = 30
) -> int:
    # Auto-approve workspace edits cleanly in headless `-p` ONLY if user passed `--mode auto_edit|yolo`;
    # read-only tools work out of the box in `default` and `plan` modes.
    if not sys.stdin.isatty():
        piped = sys.stdin.read()
        if piped.strip():
            prompt = f"{prompt}\n\n<stdin>\n{piped[:64_000]}\n</stdin>"

    extra_parts: list[CanonicalPart] = []
    if prompt.strip().startswith("/"):
        _, expanded, extra_parts = handle_slash_command(rt, prompt.strip(), headless=True)
        if not expanded:
            rt.runner.close()
            return 0
        prompt = expanded

    prompt, sk_blocks = rt.skill_manager.expand_inline_dollar_mentions(prompt, rt.runner.firewall)
    if sk_blocks:
        prompt = "\n\n".join(sk_blocks) + "\n\n" + prompt

    def _emit_jsonl(evt: StreamEvent) -> None:
        frame: dict[str, Any] = {"type": evt.kind, "ts": time.time(), "step": evt.step}
        if evt.elapsed_ms:
            frame["elapsed_ms"] = round(evt.elapsed_ms, 1)
        if evt.text:
            frame["text"] = evt.text
        if evt.part:
            if evt.kind == "tool_call":
                frame["tool_call"] = {
                    "id": evt.part.tool_call_id,
                    "name": evt.part.tool_name,
                    "args": evt.part.tool_args,
                }
            elif evt.kind == "tool_result":
                frame["tool_result"] = {
                    "id": evt.part.tool_call_id,
                    "name": evt.part.tool_name,
                    "is_error": evt.part.is_error,
                    "preview": (evt.part.text or "")[:400],
                }
        if evt.meta:
            frame["meta"] = evt.meta
        if evt.usage:
            frame["usage"] = evt.usage.__dict__
        print(json.dumps(frame, ensure_ascii=False), flush=True)

    live_text_trace = output_format == "text" and rt.trace_mode != TraceMode.OFF
    evt_cb: Any = None
    if output_format == "stream-json":
        evt_cb = _emit_jsonl
    elif live_text_trace:
        evt_cb = _stream_printer(rt)

    try:
        outcome = rt.runner.run_turn(
            prompt,
            extra_parts=extra_parts,
            max_steps=max_turns,
            on_event=evt_cb,
        )
    except Exception as exc:
        err_msg = redact_secrets(str(exc))
        hint = (
            "Check `crux auth status`, switch models with `/model` (or `-m gemini-3.1-pro-preview`), "
            "or run `crux auth login`."
        )
        if output_format in ("json", "stream-json"):
            print(
                json.dumps(
                    {
                        "error": err_msg,
                        "model": rt.runner.model_profile.model_id,
                        "auth_profile": rt.auth_profile.name if rt.auth_profile else None,
                        "hint": hint,
                    },
                    indent=2 if output_format == "json" else None,
                    ensure_ascii=False,
                )
            )
        else:
            print(f"[crux error] {err_msg}\nHint: {hint}", file=sys.stderr)
        return 1
    finally:
        rt.runner.close()

    if output_format == "text" and not live_text_trace:
        raw_reply = outcome.final_message.plain_text()
        if USE_COLOR and raw_reply:
            from ush.cli.markdown import render_markdown_ansi

            print(render_markdown_ansi(raw_reply, use_color=True))
        else:
            print(raw_reply)
    elif output_format == "json":
        ledger = rt.runner.trace_ledger
        t_blocks = ledger.thought_blocks(turn=ledger.turn_index) if ledger else []
        print(
            json.dumps(
                {
                    "session_id": rt.runner.session_id,
                    "trace_path": str(ledger.trace_path) if ledger else None,
                    "model": rt.runner.model_profile.model_id,
                    "tier": rt.runner.model_profile.tier.value,
                    "thinking_effort": rt.runner.thinking_effort.value,
                    "stop_reason": outcome.stop_reason.value,
                    "steps": outcome.steps_executed,
                    "thoughts": [
                        {
                            "step": b.get("step"),
                            "thought_ms": b.get("thought_ms"),
                            "reasoning_tokens": b.get("reasoning_tokens"),
                            "landmarks": b.get("landmarks"),
                            "thought": b.get("thought"),
                        }
                        for b in t_blocks
                    ],
                    "response": outcome.final_message.plain_text(),
                    "usage": outcome.usage.__dict__,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
    return 0


def _run_trace_cli(ush_home: pathlib.Path, workspace: pathlib.Path, args: argparse.Namespace) -> int:
    sessions_dir = ush_home / "sessions"
    action = args.action
    if action == "list":
        items = list_workspace_sessions(sessions_dir, workspace_filter=None if args.all else workspace)
        if not items:
            print("No saved trace sessions found.")
            return 0
        for s in items[:30]:
            m_lbl = ",".join(s.models[:2]) or "-"
            print(
                f"{s.session_id}  turns={s.turns:<2d} steps={s.steps:<3d} "
                f"thoughts={s.thought_blocks:<3d} ({s.reasoning_tokens:,}t)  [{m_lbl}]  {s.last_prompt[:64]}"
            )
        return 0

    sid = args.session
    if not sid or sid == "last":
        items = list_workspace_sessions(sessions_dir, workspace_filter=workspace) or list_workspace_sessions(sessions_dir)
        items = [s for s in items if s.trace_path.is_file()]
        if not items:
            print("No recorded `.trace.jsonl` sessions found yet.", file=sys.stderr)
            return 1
        sid = items[0].session_id

    tpath = sessions_dir / f"{sid}.trace.jsonl"
    if not tpath.is_file():
        print(f"Trace ledger not found: `{tpath}`", file=sys.stderr)
        return 1
    ledger = ThoughtTraceLedger(tpath, session_id=sid, workspace=str(workspace))

    if action == "path":
        print(str(tpath))
    elif action == "stats":
        print(ledger.stats_summary())
    elif action == "show":
        print(ledger.format_trace_view(args.target or "all"))
    elif action == "search":
        if not args.target:
            print("Usage: `crux trace search <query> [--session <id>]`", file=sys.stderr)
            return 2
        hits = ledger.search(args.target)
        for h in hits:
            print(f"[turn {h['turn']} step {h['step']} • {h['type']} • {h['reasoning_tokens']}t] {h['snippet']}")
        if not hits:
            print(f"No matches for `{args.target}` in `{sid}`.")
    elif action == "export":
        dest = pathlib.Path(args.target) if args.target else None
        out_f = ledger.export_markdown(dest)
        print(str(out_f))
    return 0


# ============================================================================
# CLI Argument Parser & Entrypoint (`ush`)
# ============================================================================


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="crux",
        description="Crucible (crux v0.1): Interactive long-form thought-tracing coding & 4-Gear Elastic Swarm CLI.",
    )
    p.add_argument("--version", action="version", version=f"crux {__version__} (Crucible)")
    p.add_argument("-C", "--cwd", default=".", help="Target workspace directory (default: current directory)")
    p.add_argument("-m", "--model", default=None, help="Model ID override (e.g. gemini-3.1-pro-preview, claude-opus-4-20250514, gpt-5, qwen3-coder)")
    p.add_argument("--profile", default=None, help="Named auth profile from `crux auth status`")
    p.add_argument(
        "--thinking",
        choices=["off", "low", "medium", "high", "max"],
        default="high",
        help="Cross-provider native thinking depth (default: high)",
    )
    p.add_argument(
        "--trace",
        choices=["full", "compact", "off"],
        default=None,
        help="Long-form thought-trace display mode (default: `full` interactive, `off` for `-p` unless set)",
    )
    p.add_argument("-c", "--continue", dest="continue_last", action="store_true", help="Resume the most recent session (prefers current workspace, falls back to most recent overall)")
    p.add_argument(
        "-r",
        "--resume",
        nargs="?",
        const="__pick__",
        default=None,
        metavar="SESSION_ID",
        help="Resume a `sess_...` ID, `last`, or pass bare `--resume` to open the Recent Sessions picker",
    )
    p.add_argument(
        "--mode",
        choices=["plan", "default", "supervised", "auto_edit", "yolo", "autonomous"],
        default="default",
        help="Execution permission mode",
    )
    p.add_argument(
        "--turbo",
        action="store_true",
        help="Hands-free autonomous turbo mode (160 steps/turn; security firewall stays enforced)",
    )
    p.add_argument(
        "--gear",
        choices=["auto", "g1", "g2", "g3", "g4"],
        default="auto",
        help="Elastic Swarm gearbox setting",
    )
    p.add_argument("-p", "--prompt", default=None, help="Run non-interactively on a prompt and exit (Unix/CI mode)")
    p.add_argument(
        "--output-format",
        choices=["text", "json", "stream-json"],
        default="text",
        help="Output format for headless `-p` mode",
    )
    p.add_argument(
        "--plain",
        action="store_true",
        help="Disable visual docked menus/threads UI and use plain scrollback line REPL",
    )
    p.add_argument("--max-turns", type=int, default=80, help="Maximum reasoning + tool-calling steps per turn (default: 80)")

    sub = p.add_subparsers(dest="subcommand")

    auth_p = sub.add_parser("auth", help="Manage multi-provider authentication & Keychain profiles")
    auth_p.add_argument("action", nargs="?", default="status", choices=["status", "login", "switch", "logout"])
    auth_p.add_argument("target", nargs="?", default=None, help="Provider slug (e.g. google-oauth, gemini-enterprise, anthropic, openai) or profile name")
    auth_p.add_argument("--no-browser", dest="no_browser", action="store_true", help="Print OAuth URL and accept manual copy-paste code instead of opening a browser")

    tr_p = sub.add_parser("trace", help="Inspect, search, or export durable long-form `.trace.jsonl` chains of thought")
    tr_p.add_argument("action", nargs="?", default="list", choices=["list", "show", "search", "stats", "export", "path"])
    tr_p.add_argument("target", nargs="?", default=None, help="Step (`all|last|3|t1:2`), search query, or export path")
    tr_p.add_argument("-s", "--session", default="last", help="Target `sess_...` ID (default: latest in workspace)")
    tr_p.add_argument("--all", action="store_true", help="List traces across all workspaces")

    sub.add_parser("doctor", help="Verify local auth, PTY, Shadow Git, skills, commands, and policies")

    serve_p = sub.add_parser("serve", help="Run ACP / JSON-RPC 2.0 daemon for IDEs and Swarm Peers")
    serve_p.add_argument("--acp", action="store_true", help="Serve over stdio (ACP/IDE mode)")
    serve_p.add_argument("--port", type=int, default=0, help="Serve strictly on 127.0.0.1:<port> TCP loopback")
    serve_p.add_argument(
        "--token",
        "--acp-token",
        dest="acp_token",
        default=None,
        help="Bearer token required on TCP loopback (defaults to USH_ACP_TOKEN or ephemeral 256-bit random token)",
    )

    ext_p = sub.add_parser("extensions", help="List or install USH / Gemini-CLI-compatible extensions")
    ext_p.add_argument("action", nargs="?", default="list", choices=["list", "install"])
    ext_p.add_argument("path", nargs="?", default=None)

    sw_p = sub.add_parser("swarm", help="Dispatch task to local 4-Gear Swarm, peer CLI, or remote container cluster")
    sw_p.add_argument("task", nargs="?", default="", help="Task prompt or message to solve")
    sw_p.add_argument(
        "--remote-cluster",
        "--argonclaw",
        dest="argonclaw",
        action="store_true",
        help="Dispatch directly to remote swarm cluster (`SWARM_HOST` / `SWARM_DISPATCHER_SCRIPT`)",
    )
    sw_p.add_argument(
        "--peer-cli",
        dest="peer_cli",
        action="store_true",
        help="Delegate task to external peer agent CLI (`USH_PEER_AGENT_BIN` / `agentapi`)",
    )
    sw_p.add_argument(
        "--peer-conversation",
        dest="peer_conversation",
        default=None,
        help="Existing peer conversation ID for `send-message` follow-up or `--peer-metadata`",
    )
    sw_p.add_argument(
        "--peer-metadata",
        dest="peer_metadata",
        action="store_true",
        help="Fetch JSON metadata for `--peer-conversation` via peer agent CLI",
    )
    sw_p.add_argument(
        "--gear",
        dest="swarm_gear",
        choices=["auto", "g1", "g2", "g3", "g4"],
        default=None,
        help="Swarm gearbox override (`auto|g1|g2|g3|g4`)",
    )
    sw_p.add_argument(
        "--output-format",
        dest="swarm_output_format",
        choices=["text", "json", "stream-json"],
        default=None,
        help="Output format (`text`, `json`, or `stream-json`)",
    )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    workspace = pathlib.Path(args.cwd).expanduser().resolve()
    ush_home = pathlib.Path(os.environ.get("USH_HOME", "~/.ush")).expanduser().resolve()

    if args.subcommand == "auth":
        vault = CredentialVault(ush_home, probe_ambient=True)
        if args.action == "status":
            print(render_auth_status(vault))
            return 0
        if args.action == "login":
            prof = interactive_login_wizard(vault, args.target, no_browser=args.no_browser)
            print(c_green(f"Logged in & activated `{prof.name}` ({prof.provider} via {prof.source})."))
            return 0
        if args.action == "switch":
            if not args.target:
                print("Usage: `ush auth switch <profile>`", file=sys.stderr)
                return 2
            prof = vault.switch_profile(args.target)
            print(c_green(f"Active profile -> `{prof.name}`"))
            return 0
        if args.action == "logout":
            if not args.target:
                print("Usage: `ush auth logout <profile>`", file=sys.stderr)
                return 2
            ok = vault.logout(args.target)
            print(f"Removed `{args.target}`: {ok}")
            return 0

    if args.subcommand == "trace":
        return _run_trace_cli(ush_home, workspace, args)

    open_sessions_picker = False
    all_recent = list_workspace_sessions(ush_home / "sessions", workspace_filter=None, include_empty=False)
    resume_sid: str | None = args.resume
    resumed_model_hint: str | None = None

    if resume_sid == "__pick__":
        if not args.plain and args.prompt is None and sys.stdin.isatty() and sys.stdout.isatty():
            open_sessions_picker = True
            resume_sid = None
        else:
            resume_sid = all_recent[0].session_id if all_recent else None
    elif resume_sid == "last":
        resume_sid = all_recent[0].session_id if all_recent else None
    elif not resume_sid and args.continue_last:
        local_recent = [s for s in all_recent if s.workspace == str(workspace)]
        chosen_recent = local_recent[0] if local_recent else (all_recent[0] if all_recent else None)
        if chosen_recent:
            resume_sid = chosen_recent.session_id

    if resume_sid and not args.model:
        matched_s = next((s for s in all_recent if s.session_id == resume_sid), None)
        if matched_s and matched_s.models:
            resumed_model_hint = matched_s.models[-1]

    default_trace = TraceMode.OFF if (args.prompt is not None or args.subcommand == "swarm") else TraceMode.FULL
    chosen_trace = TraceMode(args.trace) if args.trace else default_trace

    mode_alias = {"supervised": "default", "autonomous": "yolo"}.get(args.mode, args.mode)
    eff_mode = PermissionMode.YOLO if args.turbo else PermissionMode(mode_alias)
    eff_steps = max(args.max_turns, 160) if args.turbo else args.max_turns

    rt = HarnessRuntime(
        workspace,
        model_override=args.model or resumed_model_hint,
        profile_override=args.profile,
        mode=eff_mode,
        gear=Gear(args.gear),
        thinking_effort=ThinkingEffort(args.thinking),
        trace_mode=chosen_trace,
        max_steps=eff_steps,
        session_id=resume_sid,
    )

    if args.subcommand == "doctor":
        print(render_doctor_report(rt))
        rt.runner.close()
        return 0

    if args.subcommand == "extensions":
        if args.action == "install" and args.path:
            ext = rt.ext_manager.install_local(pathlib.Path(args.path))
            print(f"Installed extension `{ext.name}` v{ext.version} -> {ext.root}")
        else:
            for e in rt.ext_manager.extensions:
                print(f"{e.name}\tv{e.version}\t{e.root}")
            if not rt.ext_manager.extensions:
                print("No extensions installed.")
        rt.runner.close()
        return 0

    if args.subcommand == "serve":
        from ush.cli.acp_server import serve_loopback_tcp, serve_stdio

        try:
            if args.acp and not args.port:
                return serve_stdio(rt.runner)
            serve_loopback_tcp(rt.runner, port=args.port, auth_token=args.acp_token)
            return 0
        finally:
            rt.runner.close()

    if args.subcommand == "swarm":
        eff_fmt = getattr(args, "swarm_output_format", None) or args.output_format
        eff_gear = getattr(args, "swarm_gear", None) or args.gear
        try:
            if args.peer_cli or args.peer_metadata or args.peer_conversation:
                if not args.peer_metadata and not args.task.strip():
                    print("Usage: `crux swarm --peer-cli \"<task>\"`", file=sys.stderr)
                    return 2
                peer_res = rt.swarm.dispatch_to_peer_cli(
                    args.task,
                    conversation_id=args.peer_conversation,
                    metadata_only=args.peer_metadata,
                )
                if eff_fmt == "json":
                    print(
                        json.dumps(
                            {
                                "returncode": peer_res.returncode,
                                "conversation_id": peer_res.conversation_id,
                                "metadata": peer_res.metadata,
                                "output": peer_res.output,
                            },
                            indent=2,
                        )
                    )
                else:
                    if peer_res.output:
                        print(peer_res.output)
                return peer_res.returncode
            if not args.task.strip():
                print("Usage: `crux swarm \"<task>\"`", file=sys.stderr)
                return 2
            if args.argonclaw:
                rc, out = rt.swarm.dispatch_to_argonclaw_cluster(
                    task_id=new_id("ush"), title=args.task[:64], description=args.task
                )
                print(out)
                return rc
            rt.gear = select_initial_gear(args.task, Gear(eff_gear))
            print(c_cyan(f"[USH Swarm Gearbox -> {rt.gear.value.upper()}]"))
            return run_headless(rt, args.task, output_format=eff_fmt, max_turns=args.max_turns)
        finally:
            rt.runner.close()

    if args.prompt is not None:
        return run_headless(rt, args.prompt, output_format=args.output_format, max_turns=args.max_turns)

    if not args.plain and sys.stdin.isatty() and sys.stdout.isatty():
        from ush.cli.tui import run_visual_tui

        return run_visual_tui(
            rt,
            slash_handler=lambda r, line: handle_slash_command(r, line, headless=False),
            use_color=USE_COLOR,
            open_sessions_on_start=open_sessions_picker,
        )

    return run_interactive_tui(rt)


if __name__ == "__main__":
    raise SystemExit(main())
