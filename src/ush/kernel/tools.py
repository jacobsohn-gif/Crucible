"""The Essential 11 Static Tool Registry, RwLock Dispatcher & Drop-Safe Synthetic Orphan Repair.

Zero `eval()` / `exec()` — every tool dispatches through a typed schema-validated handler table.
"""

from __future__ import annotations

import ast
import base64
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import fnmatch
import mimetypes
import os
import pathlib
import re
import threading
import time
from typing import Any, Callable

from ush.core.types import CanonicalPart, PermissionMode, PolicyDecision
from ush.kernel.patch import StaleReadTracker, atomic_write_bytes, edit_file_safe
from ush.kernel.pty_manager import PersistentPtyManager
from ush.kernel.security import PolicyEngine, SecurityFirewall, redact_secrets

IMAGE_EXTS: frozenset[str] = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
SKIP_WALK_DIRS: frozenset[str] = frozenset(
    {".git", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "target", "dist"}
)
SYMBOL_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|fn|pub\s+fn|struct|pub\s+struct|enum|trait|interface|type|func)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]
    read_only: bool
    handler: Callable[[dict[str, Any], "ToolExecutionContext"], CanonicalPart]

    def to_wire_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass
class ToolExecutionContext:
    call_id: str
    firewall: SecurityFirewall
    stale_tracker: StaleReadTracker
    pty: PersistentPtyManager
    policy_engine: PolicyEngine
    mode: PermissionMode = PermissionMode.DEFAULT
    preview_bytes: int = 16_384
    cancel_event: threading.Event | None = None
    on_file_accessed: Callable[[pathlib.Path], None] | None = None
    on_file_modified: Callable[[pathlib.Path, str], None] | None = None
    approval_callback: Callable[[str, dict[str, Any], str], bool] | None = None
    ask_user_callback: Callable[[str, list[str]], str] | None = None
    swarm_callback: Callable[[dict[str, Any]], str] | None = None
    skill_activate_callback: Callable[[str], str] | None = None
    evaluator_consult_callback: Callable[[dict[str, Any]], dict[str, Any]] | None = None


class ReadWriteLock:
    """Fair read-parallel / write-exclusive synchronization lock guarding kernel tool batches."""

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            while self._writer or self._readers > 0:
                self._cond.wait()
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()


@dataclass
class ToolRegistry:
    """Static HashMap tool registry with RwLock batch dispatch and 100% synthetic orphan repair (CT-08)."""

    _specs: dict[str, ToolSpec] = field(default_factory=dict)
    _rwlock: ReadWriteLock = field(default_factory=ReadWriteLock)

    def register(self, spec: ToolSpec) -> None:
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self) -> set[str]:
        return set(self._specs.keys())

    def wire_schemas(
        self,
        *,
        allowed_names: set[str] | None = None,
        skill_enum: list[str] | None = None,
        evaluator_enum: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for name in sorted(self._specs):
            if allowed_names is not None and name not in allowed_names:
                continue
            d = self._specs[name].to_wire_dict()
            if name == "activate_skill":
                if not skill_enum:
                    continue
                d = {
                    "name": d["name"],
                    "description": d["description"],
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "enum": sorted(skill_enum),
                                "description": "Exact name of the available skill to activate.",
                            }
                        },
                        "required": ["name"],
                    },
                }
            elif name == "consult_evaluator":
                base_schema = d["input_schema"]
                props = dict(base_schema.get("properties") or {})
                wf_head = ["cold_eyes_review", "challenge_design", "break_loop"]
                extra_ids = [
                    str(e)
                    for e in (evaluator_enum or [])
                    if str(e) and str(e) not in (*wf_head, "all")
                ]
                eval_enum = [*wf_head, *extra_ids, "all"]
                ev_prop = dict(props.get("evaluator") or {"type": "string"})
                ev_prop["enum"] = eval_enum
                props["evaluator"] = ev_prop
                d = {
                    "name": d["name"],
                    "description": d["description"],
                    "input_schema": {**base_schema, "properties": props},
                }
            out.append(d)
        return out

    def dispatch_batch(
        self,
        calls: list[CanonicalPart],
        ctx_factory: Callable[[str], ToolExecutionContext],
        *,
        max_parallel_reads: int = 8,
        before_tool_hook: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any], str]] | None = None,
        after_tool_hook: Callable[[str, CanonicalPart], str] | None = None,
        on_tool_result: Callable[[CanonicalPart, float], None] | None = None,
    ) -> list[CanonicalPart]:
        """Executes tool calls preserving exact index alignment & guaranteeing synthetic `tool_result` closure."""
        # Pre-populate synthetic orphan repair slots BEFORE a single tool starts (CT-08)!
        results: dict[str, CanonicalPart] = {
            str(c.tool_call_id): CanonicalPart(
                kind="tool_result",
                tool_call_id=c.tool_call_id,
                tool_name=c.tool_name,
                text="[ABORTED: tool execution cancelled before completion]",
                is_error=True,
            )
            for c in calls
        }

        # Partition contiguous read calls for concurrent dispatch vs serialized write calls
        idx = 0
        n = len(calls)
        while idx < n:
            call = calls[idx]
            spec = self._specs.get(call.tool_name or "")
            if spec and spec.read_only and max_parallel_reads > 1:
                read_group: list[CanonicalPart] = []
                while idx < n:
                    c2 = calls[idx]
                    s2 = self._specs.get(c2.tool_name or "")
                    if s2 and s2.read_only and len(read_group) < max_parallel_reads:
                        read_group.append(c2)
                        idx += 1
                    else:
                        break
                if len(read_group) > 1:
                    with ThreadPoolExecutor(max_workers=min(max_parallel_reads, len(read_group))) as pool:
                        futs: list[tuple[str, Future[tuple[CanonicalPart, float]]]] = [
                            (
                                str(rc.tool_call_id),
                                pool.submit(
                                    self._run_single_timed,
                                    rc,
                                    ctx_factory(str(rc.tool_call_id)),
                                    before_tool_hook,
                                    after_tool_hook,
                                    on_tool_result,
                                ),
                            )
                            for rc in read_group
                        ]
                        for cid, fut in futs:
                            try:
                                part_res, _ = fut.result()
                                results[cid] = part_res
                            except Exception as exc:  # noqa: BLE001
                                results[cid].text = f"[TOOL_RUNTIME_ERROR: {exc}]"
                    continue
                # Single read falls through cleanly
                res, _ = self._run_single_timed(
                    read_group[0],
                    ctx_factory(str(read_group[0].tool_call_id)),
                    before_tool_hook,
                    after_tool_hook,
                    on_tool_result,
                )
                results[str(read_group[0].tool_call_id)] = res
            else:
                res, _ = self._run_single_timed(
                    call,
                    ctx_factory(str(call.tool_call_id)),
                    before_tool_hook,
                    after_tool_hook,
                    on_tool_result,
                )
                results[str(call.tool_call_id)] = res
                idx += 1

        return [results[str(c.tool_call_id)] for c in calls]

    def _run_single_timed(
        self,
        call: CanonicalPart,
        ctx: ToolExecutionContext,
        before_hook: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any], str]] | None,
        after_hook: Callable[[str, CanonicalPart], str] | None,
        on_tool_result: Callable[[CanonicalPart, float], None] | None,
    ) -> tuple[CanonicalPart, float]:
        t0 = time.monotonic()
        part = self._run_single(call, ctx, before_hook, after_hook)
        ms = (time.monotonic() - t0) * 1000.0
        if on_tool_result is not None:
            try:
                on_tool_result(part, ms)
            except Exception:  # noqa: BLE001
                pass
        return part, ms

    def _run_single(
        self,
        call: CanonicalPart,
        ctx: ToolExecutionContext,
        before_hook: Callable[[str, dict[str, Any]], tuple[bool, dict[str, Any], str]] | None,
        after_hook: Callable[[str, CanonicalPart], str] | None,
    ) -> CanonicalPart:
        cid = str(call.tool_call_id)
        tname = str(call.tool_name or "")
        args = dict(call.tool_args or {})

        try:
            if ctx.cancel_event and ctx.cancel_event.is_set():
                return CanonicalPart(
                    kind="tool_result",
                    tool_call_id=cid,
                    tool_name=tname,
                    text="[ABORTED_BY_USER: tool cancelled before start]",
                    is_error=True,
                )

            spec = self._specs.get(tname)
            if not spec:
                return CanonicalPart(
                    kind="tool_result",
                    tool_call_id=cid,
                    tool_name=tname,
                    text=f"Unknown tool `{tname}`. Available tools: {', '.join(sorted(self._specs))}",
                    is_error=True,
                )

            # 1. Lifecycle BeforeTool hook (may veto or rewrite args)
            if before_hook:
                allow, args, reason = before_hook(tname, args)
                if not allow:
                    return CanonicalPart(
                        kind="tool_result",
                        tool_call_id=cid,
                        tool_name=tname,
                        text=f"[BLOCKED_BY_HOOK]: {reason}",
                        is_error=True,
                    )

            # 2. PolicyEngine gating + interactive human approval
            cmd_str = str(args.get("command", "")) if tname == "exec_pty" else None
            decision, reason = ctx.policy_engine.check(tname, command=cmd_str, mode=ctx.mode)
            if decision == PolicyDecision.DENY:
                return CanonicalPart(
                    kind="tool_result",
                    tool_call_id=cid,
                    tool_name=tname,
                    text=f"[POLICY_DENIED]: {reason}",
                    is_error=True,
                )
            if decision == PolicyDecision.ASK_USER:
                if ctx.approval_callback is None or not ctx.approval_callback(tname, args, reason):
                    return CanonicalPart(
                        kind="tool_result",
                        tool_call_id=cid,
                        tool_name=tname,
                        text=f"[APPROVAL_REQUIRED_DENIED]: {reason}",
                        is_error=True,
                    )

            # 3. Execute under RwLock (Read-Parallel / Write-Exclusive)
            if spec.read_only:
                self._rwlock.acquire_read()
            else:
                self._rwlock.acquire_write()
            try:
                part = spec.handler(args, ctx)
                part.tool_call_id = cid
                part.tool_name = tname
            except Exception as exc:  # noqa: BLE001
                part = CanonicalPart(
                    kind="tool_result",
                    tool_call_id=cid,
                    tool_name=tname,
                    text=f"[{type(exc).__name__}]: {redact_secrets(str(exc))}",
                    is_error=True,
                )
            finally:
                if spec.read_only:
                    self._rwlock.release_read()
                else:
                    self._rwlock.release_write()

            if after_hook:
                try:
                    extra = after_hook(tname, part)
                    if extra:
                        part.text = f"{part.text}\n{extra}".strip()
                except Exception as exc:  # noqa: BLE001
                    part.text = f"{part.text}\n[AfterTool hook error: {redact_secrets(str(exc))}]".strip()
            return part
        except Exception as exc:  # noqa: BLE001
            return CanonicalPart(
                kind="tool_result",
                tool_call_id=cid,
                tool_name=tname,
                text=f"[TOOL_RUNTIME_ERROR ({type(exc).__name__})]: {redact_secrets(str(exc))}",
                is_error=True,
            )


# ============================================================================
# Essential 11 Built-In Tool Implementations
# ============================================================================


def _iter_safe_workspace_files(firewall: SecurityFirewall) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    root = firewall.workspace_root
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_WALK_DIRS and not d.startswith(".ush"))
        dp = pathlib.Path(dirpath)
        for fn in sorted(filenames):
            fp = dp / fn
            if firewall.is_ignored_secret_path(fp):
                continue
            out.append(fp)
    return out


def _extract_file_skeleton(path: pathlib.Path, text: str) -> str:
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            items: list[str] = []
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    args = [a.arg for a in node.args.args]
                    items.append(f"L{node.lineno}: def {node.name}({', '.join(args)})")
                elif isinstance(node, ast.ClassDef):
                    methods = [
                        n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ]
                    m_sfx = f" [{', '.join(methods[:8])}]" if methods else ""
                    items.append(f"L{node.lineno}: class {node.name}{m_sfx}")
            return "\n  ".join(items)
        except SyntaxError:
            pass
    hits: list[str] = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if SYMBOL_LINE_RE.match(line):
            hits.append(f"L{lineno}: {line.strip()[:100]}")
            if len(hits) >= 25:
                break
    return "\n  ".join(hits)


def build_default_tool_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def _read_file(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        target = ctx.firewall.resolve_and_validate(str(args["path"]), write=False)
        if not target.is_file():
            return CanonicalPart(kind="tool_result", text=f"File not found: `{target}`", is_error=True)
        raw = target.read_bytes()
        ctx.stale_tracker.record_read(target, raw)
        if ctx.on_file_accessed:
            ctx.on_file_accessed(target)

        if target.suffix.lower() in IMAGE_EXTS and len(raw) <= 8 * 1024 * 1024:
            mime = mimetypes.guess_type(target.name)[0] or "image/png"
            return CanonicalPart(
                kind="tool_result",
                text=f"Loaded image `{target.name}` ({len(raw):,} bytes).",
                mime_type=mime,
                base64_data=base64.b64encode(raw).decode("ascii"),
            )

        if b"\x00" in raw[:4096]:
            return CanonicalPart(
                kind="tool_result",
                text=f"Binary file `{target.name}` ({len(raw):,} bytes) skipped.",
                is_error=True,
            )

        lines = raw.decode("utf-8", errors="replace").splitlines()
        start = max(1, int(args.get("start_line") or 1))
        end = min(len(lines), int(args.get("end_line") or min(len(lines), start + 399)))
        numbered = [f"{ln:5d}: {redact_secrets(lines[ln - 1])}" for ln in range(start, end + 1)]
        header = f"[{target} | lines {start}-{end} of {len(lines)}]"
        return CanonicalPart(kind="tool_result", text=header + "\n" + "\n".join(numbered))

    def _edit_file(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        target = ctx.firewall.resolve_and_validate(str(args["path"]), write=True)
        res = edit_file_safe(
            target,
            target=str(args["target"]),
            replacement=str(args["replacement"]),
            stale_tracker=ctx.stale_tracker,
        )
        if not res.success:
            return CanonicalPart(kind="tool_result", text=res.error, is_error=True)
        if ctx.on_file_accessed:
            ctx.on_file_accessed(target)
        if ctx.on_file_modified:
            ctx.on_file_modified(target, res.diff_preview)
        return CanonicalPart(
            kind="tool_result",
            text=f"Edited `{target.name}` via pass=`{res.pass_used}` ({res.replacements} match(es)):\n{res.diff_preview[:3000]}",
        )

    def _write_file(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        target = ctx.firewall.resolve_and_validate(str(args["path"]), write=True)
        content = str(args.get("content", ""))
        raw = content.encode("utf-8")
        if target.is_file():
            conflict = ctx.stale_tracker.check_fresh(target, target.read_bytes())
            if conflict:
                return CanonicalPart(kind="tool_result", text=conflict, is_error=True)
        atomic_write_bytes(target, raw)
        ctx.stale_tracker.record_read(target, raw)
        if ctx.on_file_modified:
            ctx.on_file_modified(target, f"Wrote {len(raw):,} bytes to {target.name}")
        return CanonicalPart(kind="tool_result", text=f"Wrote `{target}` ({len(raw):,} bytes).")

    def _grep_files(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        pattern = str(args["pattern"])
        glob_filter = str(args.get("glob") or "*")
        case_ins = bool(args.get("case_insensitive", False))
        flags = re.IGNORECASE if case_ins else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), flags)

        matches: list[str] = []
        for fp in _iter_safe_workspace_files(ctx.firewall):
            rel = fp.relative_to(ctx.firewall.workspace_root).as_posix()
            if not (fnmatch.fnmatch(fp.name, glob_filter) or fnmatch.fnmatch(rel, glob_filter)):
                continue
            try:
                data = fp.read_bytes()
                if b"\x00" in data[:2048] or len(data) > 2_000_000:
                    continue
                for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                    if rx.search(line):
                        matches.append(f"{rel}:{lineno}: {redact_secrets(line.strip()[:200])}")
                        if len(matches) >= 120:
                            break
            except OSError:
                continue
            if len(matches) >= 120:
                break
        if not matches:
            return CanonicalPart(kind="tool_result", text=f"No matches found for `{pattern}`.")
        return CanonicalPart(kind="tool_result", text="\n".join(matches))

    def _glob_files(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        pat = str(args.get("pattern") or "*")
        skeleton = bool(args.get("skeleton", False))
        hits: list[str] = []
        for fp in _iter_safe_workspace_files(ctx.firewall):
            rel = fp.relative_to(ctx.firewall.workspace_root).as_posix()
            if pat in ("*", "**", "**/*") or fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(fp.name, pat):
                if skeleton and fp.suffix in (".py", ".ts", ".tsx", ".js", ".rs", ".go"):
                    try:
                        skel = _extract_file_skeleton(fp, fp.read_text(encoding="utf-8", errors="replace"))
                        hits.append(f"{rel}\n  {skel}" if skel else rel)
                    except OSError:
                        hits.append(rel)
                else:
                    hits.append(rel)
                if len(hits) >= 200:
                    break
        return CanonicalPart(kind="tool_result", text="\n".join(hits) if hits else "No matching files.")

    def _lsp_query(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        action = str(args.get("action", "definition"))
        symbol = str(args["symbol"]).strip()
        if not symbol:
            return CanonicalPart(kind="tool_result", text="Symbol must not be empty.", is_error=True)
        esc = re.escape(symbol)
        def_rx = re.compile(
            rf"^\s*(?:export\s+)?(?:async\s+)?(?:def|class|fn|pub\s+fn|struct|pub\s+struct|enum|trait|interface|type|func|const|let)\s+{esc}\b"
        )
        ref_rx = re.compile(rf"\b{esc}\b")

        defs: list[str] = []
        refs: list[str] = []
        for fp in _iter_safe_workspace_files(ctx.firewall):
            rel = fp.relative_to(ctx.firewall.workspace_root).as_posix()
            try:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for idx, ln in enumerate(lines, 1):
                if def_rx.search(ln):
                    doc_hint = lines[idx].strip()[:120] if idx < len(lines) else ""
                    defs.append(f"{rel}:{idx}: {ln.strip()[:160]}" + (f"  // {doc_hint}" if action == "hover" and doc_hint else ""))
                elif action in ("references", "incoming_calls") and ref_rx.search(ln):
                    refs.append(f"{rel}:{idx}: {ln.strip()[:140]}")
                    if len(refs) >= 80:
                        break
        out = defs if action in ("definition", "hover") else (defs + refs)[:80]
        return CanonicalPart(
            kind="tool_result",
            text=("\n".join(out)[:4000] if out else f"No `{action}` results found for symbol `{symbol}`."),
        )

    def _exec_pty(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        cmd = str(args["command"])
        wait_s = float(args.get("wait_seconds") or 15.0)
        is_daemon = bool(args.get("is_daemon", False))
        res = ctx.pty.execute(
            cmd,
            wait_seconds=max(0.2, min(wait_s, 300.0)),
            is_daemon=is_daemon,
            dedicated_pty=bool(args.get("dedicated_pty", False)),
            preview_bytes=ctx.preview_bytes,
            cancel_event=ctx.cancel_event,
        )
        meta = f"[pty_id={res.pty_id} status={res.status} exit_code={res.exit_code} cwd={res.cwd}]"
        is_err = res.status in ("timed_out", "killed", "crashed") or (res.exit_code not in (None, 0))
        return CanonicalPart(kind="tool_result", text=f"{meta}\n{res.output}".strip(), is_error=is_err)

    def _write_stdin(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        pty_id = str(args["pty_id"])
        if args.get("kill"):
            ok = ctx.pty.kill(pty_id)
            return CanonicalPart(kind="tool_result", text=f"Killed `{pty_id}`: {ok}")
        res = ctx.pty.write_stdin(
            pty_id,
            chars=str(args.get("input", "")),
            wait_seconds=float(args.get("wait_seconds") or 2.0),
            preview_bytes=ctx.preview_bytes,
        )
        return CanonicalPart(
            kind="tool_result",
            text=f"[pty_id={res.pty_id} status={res.status} exit_code={res.exit_code}]\n{res.output}".strip(),
            is_error=res.status in ("not_found", "timed_out", "killed", "crashed"),
        )

    def _ask_user(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        question = str(args["question"])
        options = [str(x) for x in (args.get("options") or [])]
        if ctx.ask_user_callback is None:
            return CanonicalPart(
                kind="tool_result",
                text="Autonomous headless mode active: proceed with your clearest conservative engineering judgment.",
            )
        ans = ctx.ask_user_callback(question, options)
        return CanonicalPart(kind="tool_result", text=f"User response: {ans}")

    def _swarm_dispatch(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        if ctx.swarm_callback is None:
            return CanonicalPart(kind="tool_result", text="Swarm blackboard callback not attached.", is_error=True)
        return CanonicalPart(kind="tool_result", text=ctx.swarm_callback(args))

    
    def _consult_evaluator(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        """Clean-room zero-history evaluator tool supporting workflow transitions (`challenge_design`, `cold_eyes_review`, `break_loop`) and custom plugins."""
        import json
        from dataclasses import replace
        from crux_evals.core.protocol import EvaluationFrame, split_targets_and_load_images
        from crux_evals.core.runner import build_default_cascade
        from crux_evals.core.state import capture_git_diff, discover_git_modified_files
        from crux_evals.core.workflows import WORKFLOW_MODES, run_workflow_evaluator

        ws = ctx.firewall.workspace_root
        chosen = str(args.get("evaluator") or args.get("lens") or "cold_eyes_review").strip()
        goal = str(args.get("goal") or args.get("objective") or "").strip()
        proposed_claims = str(args.get("proposed_claims", "")).strip() or None
        raw_files = args.get("files") if isinstance(args.get("files"), list) else args.get("target_paths")
        raw_list = [str(rp) for rp in raw_files] if isinstance(raw_files, list) else []

        text_paths, attached_images = split_targets_and_load_images(
            ws,
            raw_list,
            validate_path=lambda p: ctx.firewall.resolve_and_validate(p, write=False),
        )

        if ctx.evaluator_consult_callback is not None:
            payload = ctx.evaluator_consult_callback(
                {
                    "evaluator": chosen,
                    "lens": chosen,
                    "goal": goal,
                    "objective": goal,
                    "files": text_paths,
                    "target_paths": text_paths,
                    "attached_images": attached_images,
                    "proposed_claims": proposed_claims,
                }
            )
            return CanonicalPart(kind="tool_result", text=json.dumps(payload, indent=2, ensure_ascii=False))

        # Standalone fallback when ToolRegistry is invoked outside SessionRunner
        if not text_paths:
            text_paths = discover_git_modified_files(ws)
        diff = capture_git_diff(ws, text_paths if text_paths else None)
        cascade = build_default_cascade(enforce_read_before_edit=False, workspace_root=ws)
        frame = EvaluationFrame(
            session_id="native_tool",
            workspace_root=str(ws),
            event="MCPConsult",
            step_index=1,
            task_objective=goal or "Audit artifact quality and correctness",
            unified_git_diff=diff,
            modified_files=tuple(text_paths),
            attached_images=tuple(attached_images),
            actor_claim_text=proposed_claims,
        )
        if chosen in WORKFLOW_MODES:
            payload = run_workflow_evaluator(chosen, frame, cascade, focus_note=goal)
            return CanonicalPart(kind="tool_result", text=json.dumps(payload, indent=2, ensure_ascii=False))

        if chosen != "all":
            ev = cascade.get_evaluator(chosen)
            if ev is not None:
                verdict = ev.evaluate(frame.with_epistemic_firewall() if ev.manifest.requires_blind_context else frame)
                payload = verdict.to_mcp_payload()
                payload["epistemic_firewall_active"] = bool(ev.manifest.requires_blind_context)
                payload["usage_note"] = (
                    "Use `blind_spots`, `counterexamples`, and `suggested_next_probes` to improve the artifact directly. "
                    "NEVER render evaluator scorecards or harness metadata inside user deliverables."
                )
                return CanonicalPart(kind="tool_result", text=json.dumps(payload, indent=2, ensure_ascii=False))

        report = cascade.run(frame)
        payload = report.to_mcp_payload()
        payload["epistemic_firewall_active"] = True
        payload["usage_note"] = (
            "Use `blind_spots`, `counterexamples`, and `suggested_next_probes` to improve the artifact directly. "
            "NEVER render evaluator scorecards or harness metadata inside user deliverables."
        )
        return CanonicalPart(kind="tool_result", text=json.dumps(payload, indent=2, ensure_ascii=False))


    def _activate_skill(args: dict[str, Any], ctx: ToolExecutionContext) -> CanonicalPart:
        if ctx.skill_activate_callback is None:
            return CanonicalPart(kind="tool_result", text="Skill engine not initialized.", is_error=True)
        body = ctx.skill_activate_callback(str(args["name"]))
        return CanonicalPart(kind="tool_result", text=body, is_error=body.startswith("[SKILL_ERROR]"))

    specs = [
        ToolSpec(
            "read_file",
            "Read a text or image file with line numbers and record its SHA-256 for safe editing.",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["path"],
            },
            True,
            _read_file,
        ),
        ToolSpec(
            "edit_file",
            "Surgically replace `target` with `replacement` in a file (4-pass whitespace/indent/unicode tolerant + stale-read safe).",
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "target": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["path", "target", "replacement"],
            },
            False,
            _edit_file,
        ),
        ToolSpec(
            "write_file",
            "Create or atomically overwrite a file inside the workspace.",
            {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
            False,
            _write_file,
        ),
        ToolSpec(
            "grep_files",
            "Fast regex search across workspace source files honoring .ushignore.",
            {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string"},
                    "case_insensitive": {"type": "boolean"},
                },
                "required": ["pattern"],
            },
            True,
            _grep_files,
        ),
        ToolSpec(
            "glob_files",
            "List workspace files matching a glob; set `skeleton=true` for top-level class/function AST outlines.",
            {
                "type": "object",
                "properties": {"pattern": {"type": "string"}, "skeleton": {"type": "boolean"}},
            },
            True,
            _glob_files,
        ),
        ToolSpec(
            "lsp_query",
            "Lazy read-side symbol navigation (`definition`, `references`, `hover`, `incoming_calls`).",
            {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["definition", "references", "hover", "incoming_calls"],
                    },
                    "symbol": {"type": "string"},
                },
                "required": ["symbol"],
            },
            True,
            _lsp_query,
        ),
        ToolSpec(
            "exec_pty",
            "Execute a shell command inside the persistent stateful PTY (`cwd`/venv preserved; auto-detects interactive prompts).",
            {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "wait_seconds": {"type": "number"},
                    "is_daemon": {"type": "boolean"},
                    "dedicated_pty": {"type": "boolean"},
                },
                "required": ["command"],
            },
            False,
            _exec_pty,
        ),
        ToolSpec(
            "write_stdin",
            "Send input to, poll, or kill (`kill=true`) an interactive/background `pty_id`.",
            {
                "type": "object",
                "properties": {
                    "pty_id": {"type": "string"},
                    "input": {"type": "string"},
                    "wait_seconds": {"type": "number"},
                    "kill": {"type": "boolean"},
                },
                "required": ["pty_id"],
            },
            False,
            _write_stdin,
        ),
        ToolSpec(
            "ask_user",
            "Ask the user a brief multiple-choice clarification question when intent is genuinely ambiguous.",
            {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["question"],
            },
            False,
            _ask_user,
        ),
        ToolSpec(
            "swarm_dispatch",
            "Post a `CONTRACT_DELTA`, `SPIKE`, or `REQUEST_REVIEW` event to the Elastic Stigmergic Blackboard.",
            {
                "type": "object",
                "properties": {
                    "event_type": {
                        "type": "string",
                        "enum": ["CONTRACT_DELTA", "SPIKE", "REQUEST_REVIEW", "NOTE"],
                    },
                    "summary": {"type": "string"},
                    "scope": {"type": "string"},
                },
                "required": ["event_type", "summary"],
            },
            False,
            _swarm_dispatch,
        ),
        ToolSpec(
            "activate_skill",
            "Load full `SKILL.md` instructions and unlock bundled resource directory read access for a named skill.",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
            True,
            _activate_skill,
        ),
        ToolSpec(
            "consult_evaluator",
            (
                "Run a clean-room, zero-history critic in an isolated session unpolluted by your build history "
                "(inspecting ONLY raw workspace files, diffs, test output & screenshots — ZERO Actor reasoning). "
                "Just like a compiler or linter, it catches confirmation-bias bugs you can no longer see and is "
                "guaranteed to return `PASS` or at most 3–5 surgical file/element fixes (no essays). "
                "Use at workflow transitions (skip for trivial reads/Q&A): "
                "`challenge_design` before writing >250L files or multi-file rewrites; "
                "`cold_eyes_review` right before shipping non-trivial code, algorithms, or UIs (pass files + screenshots `.png`); "
                "`break_loop` after 2 failed attempts on the same bug/render issue; or any custom plugin ID."
            ),
            {
                "type": "object",
                "properties": {
                    "evaluator": {
                        "type": "string",
                        "enum": [
                            "cold_eyes_review",
                            "challenge_design",
                            "break_loop",
                            "trajectory_sentinel",
                            "clean_patch_and_contract",
                            "epistemic_grounding_auditor",
                            "blind_bars_judge",
                            "adversarial_falsifier",
                            "artifact_enhancer",
                            "all",
                        ],
                        "description": "Clean-room workflow phase (`cold_eyes_review` | `challenge_design` | `break_loop`) or custom evaluator ID (default: `cold_eyes_review`).",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Raw code/HTML files and screenshots (.png/.jpg) to inspect cold from disk (auto-hydrates modified files if omitted).",
                    },
                    "goal": {
                        "type": "string",
                        "description": "Optional 1-line objective (defaults to user prompt; no briefing paragraph needed).",
                    },
                    "proposed_claims": {
                        "type": "string",
                        "description": "Optional claims audited only by `epistemic_grounding_auditor` (stripped for all clean-room evaluators).",
                    },
                },
                "required": [],
            },
            True,
            _consult_evaluator,
        ),
    ]
    for s in specs:
        reg.register(s)
    return reg
