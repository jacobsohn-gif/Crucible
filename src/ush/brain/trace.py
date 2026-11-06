"""Durable Long-Form Cognitive Trace Ledger & Stream-Oriented Terminal Thought Tracer.

Designed for multi-step, multi-hour interactive problem-solving:
- Streams every native chain-of-thought token (`thinking_delta`) live with a clean scrollback gutter (`⋮ `)
  alongside interleaved tool calls, tool observation summaries, Popperian loop-guard pivots, live human
  steers, and epistemic compactions — with zero TUI screen-hijacking or cursor flicker.
- Persists 100% uncompacted verbatim chains of thought to `~/.ush/sessions/<session_id>.trace.jsonl` (`0600`,
  secret-redacted) so context compaction (`0.85`) or `/restore` branch rewinds NEVER lose a single historical thought.
- Supports `/trace show`, `/trace search`, `/trace stats`, `/trace export` (Markdown lab notebook), and `tail -f`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
import pathlib
import re
import sys
import threading
import time
from typing import Any, TextIO

from ush.core.types import StreamEvent, TraceMode
from ush.kernel.security import redact_secrets

# Heuristic cue extractor indexing key pivots/hypotheses inside long chains of thought
PIVOT_CUE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*[^*\n]{3,90}\*\*|(?:Wait|Actually|Hypothesis|Let's verify|Therefore|Root cause|Key insight|Falsif)[^\n]{0,120})",
    re.IGNORECASE,
)


def extract_thought_landmarks(thought_text: str, limit: int = 6) -> list[str]:
    """Extracts crisp bold headings & epistemic pivot lines from a long-form chain of thought."""
    landmarks: list[str] = []
    seen: set[str] = set()
    for m in PIVOT_CUE_RE.finditer(thought_text or ""):
        clean = " ".join(m.group(0).strip().strip("*").split())
        if len(clean) >= 6 and clean.lower() not in seen:
            seen.add(clean.lower())
            landmarks.append(clean[:120])
            if len(landmarks) >= limit:
                break
    if not landmarks and thought_text.strip():
        first = " ".join(thought_text.strip().splitlines()[0].strip("*# ").split())
        if first:
            landmarks.append(first[:120])
    return landmarks


@dataclass
class SessionTraceSummary:
    session_id: str
    trace_path: pathlib.Path
    dag_path: pathlib.Path
    workspace: str
    updated_at: float
    turns: int
    steps: int
    thought_blocks: int
    reasoning_tokens: int
    models: list[str]
    last_prompt: str


class ThoughtTraceLedger:
    """Append-only `0600` JSONL ledger storing uncompacted long-form reasoning traces."""

    def __init__(self, trace_path: pathlib.Path, *, session_id: str, workspace: str = "") -> None:
        self.trace_path = trace_path.expanduser().resolve()
        self.session_id = session_id
        self.workspace = workspace
        self._lock = threading.Lock()
        self.turn_index = self._recover_turn_index()

    def _recover_turn_index(self) -> int:
        if not self.trace_path.is_file():
            return 0
        max_turn = 0
        try:
            for line in self.trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = int(rec.get("turn", 0))
                if t > max_turn:
                    max_turn = t
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return max_turn

    def append(self, record_type: str, **fields: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "turn": self.turn_index,
            "type": record_type,
        }
        for k, v in fields.items():
            if isinstance(v, str):
                payload[k] = redact_secrets(v)
            else:
                payload[k] = v

        encoded = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.trace_path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, encoded.encode("utf-8"))
            finally:
                os.close(fd)
        return payload

    def record_stream_event(self, evt: StreamEvent) -> None:
        """Persists discrete lifecycle milestones (skips per-token deltas; full thoughts stored via record_step)."""
        if evt.kind == "turn_start":
            self.turn_index += 1
            self.append(
                "turn_start",
                workspace=self.workspace,
                prompt=evt.text,
                **evt.meta,
            )
        elif evt.kind == "step_start":
            self.append("step_start", step=evt.step, elapsed_ms=round(evt.elapsed_ms, 1), **evt.meta)
        elif evt.kind == "tool_result" and evt.part:
            self.append(
                "tool_result",
                step=evt.step,
                tool_call_id=evt.part.tool_call_id,
                tool_name=evt.part.tool_name,
                is_error=evt.part.is_error,
                elapsed_ms=round(evt.elapsed_ms, 1),
                preview=redact_secrets((evt.part.text or "")[:2000]),
                output_chars=len(evt.part.text or ""),
                **evt.meta,
            )
        elif evt.kind == "cognitive_event":
            self.append("cognitive_event", step=evt.step, detail=evt.text, **evt.meta)
        elif evt.kind == "done":
            u = evt.usage.__dict__ if evt.usage else {}
            self.append(
                "turn_done",
                step=evt.step,
                stop_reason=evt.stop_reason.value,
                elapsed_ms=round(evt.elapsed_ms, 1),
                usage=u,
                **evt.meta,
            )

    def record_assistant_step(
        self,
        *,
        step: int,
        model_id: str,
        thinking_effort: str,
        thought_text: str,
        response_text: str,
        tool_calls: list[dict[str, Any]],
        has_signature: bool,
        step_ms: float,
        thought_ms: float,
        usage: dict[str, int],
        stop_reason: str,
    ) -> None:
        if thought_text.strip():
            self.append(
                "thought_block",
                step=step,
                model=model_id,
                thinking_effort=thinking_effort,
                thought=thought_text,
                thought_chars=len(thought_text),
                landmarks=extract_thought_landmarks(thought_text),
                has_signature=has_signature,
                thought_ms=round(thought_ms, 1),
                reasoning_tokens=int(usage.get("reasoning_tokens", 0)),
            )
        self.append(
            "assistant_step",
            step=step,
            model=model_id,
            text=response_text,
            tool_calls=tool_calls,
            step_ms=round(step_ms, 1),
            stop_reason=stop_reason,
            usage=usage,
        )

    # ---- Inspection, Search & Markdown Notebook Export ----

    def read_records(self) -> list[dict[str, Any]]:
        if not self.trace_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        for ln in self.trace_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not ln.strip():
                continue
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out

    def thought_blocks(self, *, turn: int | None = None, step: int | None = None) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for r in self.read_records():
            if r.get("type") != "thought_block":
                continue
            if turn is not None and int(r.get("turn", 0)) != turn:
                continue
            if step is not None and int(r.get("step", 0)) != step:
                continue
            out.append(r)
        return out

    def search(self, query: str, *, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across every uncompacted thought block & cognitive event in the session."""
        q = query.strip().lower()
        if not q:
            return []
        hits: list[dict[str, Any]] = []
        for r in self.read_records():
            rtype = r.get("type")
            body = ""
            if rtype == "thought_block":
                body = str(r.get("thought", ""))
            elif rtype == "cognitive_event":
                body = str(r.get("detail", ""))
            elif rtype == "tool_result":
                body = str(r.get("preview", ""))
            if not body:
                continue
            pos = body.lower().find(q)
            if pos != -1:
                lo = max(0, pos - 120)
                hi = min(len(body), pos + len(q) + 180)
                snippet = ("..." if lo > 0 else "") + " ".join(body[lo:hi].split()) + ("..." if hi < len(body) else "")
                hits.append(
                    {
                        "turn": r.get("turn"),
                        "step": r.get("step"),
                        "type": rtype,
                        "reasoning_tokens": r.get("reasoning_tokens", 0),
                        "snippet": snippet,
                    }
                )
                if len(hits) >= limit:
                    break
        return hits

    def format_trace_view(self, target: str = "last") -> str:
        """Formats verbatim chain-of-thought blocks (`last`, `all`, or integer step `N` in latest turn)."""
        records = self.read_records()
        thoughts = [r for r in records if r.get("type") == "thought_block"]
        if not thoughts:
            return f"No recorded thought blocks yet in `{self.trace_path}`."

        selected: list[dict[str, Any]]
        arg = target.strip().lower() or "last"
        if arg == "last":
            selected = [thoughts[-1]]
        elif arg == "all":
            selected = thoughts
        elif arg.startswith("t") and ":" in arg:
            # e.g. t2:3 -> turn 2, step 3
            t_s, s_s = arg[1:].split(":", 1)
            selected = [
                r for r in thoughts if int(r.get("turn", 0)) == int(t_s) and int(r.get("step", 0)) == int(s_s)
            ]
        elif arg.isdigit():
            step_n = int(arg)
            cur_turn = thoughts[-1].get("turn")
            selected = [r for r in thoughts if r.get("turn") == cur_turn and int(r.get("step", 0)) == step_n]
            if not selected:
                selected = [r for r in thoughts if int(r.get("step", 0)) == step_n]
        else:
            return "Usage: `/trace show [last | all | <step> | t<turn>:<step>]`"

        if not selected:
            return f"No thought block matching `{target}`."

        lines: list[str] = []
        for b in selected:
            hdr = (
                f"=== Turn {b.get('turn')} • Step {b.get('step')} • {b.get('model')} "
                f"(effort={b.get('thinking_effort')} • {b.get('thought_ms', 0) / 1000.0:.2f}s "
                f"• {b.get('reasoning_tokens', 0)} thought tok • sig={'yes' if b.get('has_signature') else 'no'}) ==="
            )
            lines.append(hdr)
            if b.get("landmarks"):
                lines.append("Landmarks: " + "  ->  ".join(b["landmarks"]))
                lines.append("-" * min(88, len(hdr)))
            lines.append(str(b.get("thought", "")).rstrip())
            lines.append("")
        return "\n".join(lines).rstrip()

    def stats_summary(self) -> str:
        records = self.read_records()
        turns = sum(1 for r in records if r.get("type") == "turn_start")
        thoughts = [r for r in records if r.get("type") == "thought_block"]
        tools = [r for r in records if r.get("type") == "tool_result"]
        cog = [r for r in records if r.get("type") == "cognitive_event"]
        tot_tok = sum(int(r.get("reasoning_tokens", 0)) for r in thoughts)
        tot_chars = sum(int(r.get("thought_chars", 0)) for r in thoughts)
        tot_ms = sum(float(r.get("thought_ms", 0.0)) for r in thoughts)
        return (
            f"Session      : {self.session_id}\n"
            f"Trace ledger : {self.trace_path}\n"
            f"Turns / Steps: {turns} turns • {len(thoughts)} thought blocks • {len(tools)} tool executions • {len(cog)} cognitive events\n"
            f"Reasoning    : {tot_tok:,} thought tokens ({tot_chars:,} chars) across {tot_ms / 1000.0:.2f}s total think time"
        )

    def export_markdown(self, dest: pathlib.Path | None = None) -> pathlib.Path:
        """Exports the entire uncompacted multi-step trajectory to a readable Markdown lab notebook."""
        out_path = (dest or self.trace_path.with_suffix(".md")).expanduser().resolve()
        records = self.read_records()
        md: list[str] = [
            f"# Long-Form Cognitive Trace — `{self.session_id}`",
            "",
            f"- **Workspace**: `{self.workspace}`",
            f"- **Source JSONL**: `{self.trace_path}`",
            "",
        ]
        for r in records:
            rt = r.get("type")
            if rt == "turn_start":
                md.extend(
                    [
                        "---",
                        f"## Turn {r.get('turn')} (`{r.get('iso', '')}`)",
                        f"- **Model**: `{r.get('model', '')}` (Tier `{r.get('tier', '')}`, thinking=`{r.get('thinking_effort', '')}`)",
                        "",
                        "### Prompt",
                        "```text",
                        str(r.get("prompt", "")).rstrip(),
                        "```",
                        "",
                    ]
                )
            elif rt == "thought_block":
                md.extend(
                    [
                        f"### Turn {r.get('turn')} · Step {r.get('step')} — Chain of Thought "
                        f"(`{r.get('thought_ms', 0) / 1000.0:.2f}s`, `{r.get('reasoning_tokens', 0)}` tok)",
                        "",
                        "```text",
                        str(r.get("thought", "")).rstrip(),
                        "```",
                        "",
                    ]
                )
            elif rt == "assistant_step":
                for tc in r.get("tool_calls") or []:
                    md.append(f"- **Tool Call** `{tc.get('name')}`: `{json.dumps(tc.get('args', {}), ensure_ascii=False)[:400]}`")
                if r.get("text"):
                    md.extend(["", "#### Assistant Output", "", str(r.get("text", "")).rstrip(), ""])
            elif rt == "tool_result":
                status = "ERROR" if r.get("is_error") else "OK"
                prev = " ".join(str(r.get("preview", "")).split())[:320]
                md.append(
                    f"- **Observation** `{r.get('tool_name')}` [{status} · `{r.get('elapsed_ms', 0):.0f}ms` · `{r.get('output_chars', 0)}` chars]: `{prev}`"
                )
            elif rt == "cognitive_event":
                md.append(f"- **Cognitive Event (`{r.get('event', 'event')}`)**: {r.get('detail', '')}")
            elif rt == "turn_done":
                u = r.get("usage") or {}
                md.extend(
                    [
                        "",
                        f"> **Turn Summary**: `stop={r.get('stop_reason')}` · `{r.get('step')} steps` · "
                        f"`{r.get('elapsed_ms', 0) / 1000.0:.2f}s` · "
                        f"`in={u.get('input_tokens', 0)}` (`cached={u.get('cached_input_tokens', 0)}`) · "
                        f"`thought={u.get('reasoning_tokens', 0)}` · `out={u.get('output_tokens', 0)}`",
                        "",
                    ]
                )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(md) + "\n", encoding="utf-8")
        return out_path


_WRAPPER_BLOCK_RE = re.compile(
    r"<(?:attached_file|activated_skill|dynamic_instructions)\b[^>]*>.*?</(?:attached_file|activated_skill|dynamic_instructions)>",
    re.DOTALL,
)


def _clean_prompt_preview(raw: str, max_len: int = 96) -> str:
    cleaned = _WRAPPER_BLOCK_RE.sub(" ", raw or "")
    cleaned = " ".join(cleaned.split())
    return cleaned[:max_len]


def format_relative_age(ts: float, now: float | None = None) -> str:
    """Formats a Unix timestamp into a crisp human recency badge (`just now`, `4m ago`, `2h ago`, `Nov 04`)."""
    if ts <= 0:
        return "-"
    ref = time.time() if now is None else now
    delta = max(0.0, ref - ts)
    if delta < 45:
        return "just now"
    if delta < 3600:
        return f"{int(delta // 60)}m ago"
    if delta < 86400:
        return f"{int(delta // 3600)}h ago"
    if delta < 7 * 86400:
        return f"{int(delta // 86400)}d ago"
    return datetime.fromtimestamp(ts).strftime("%b %d")


def list_workspace_sessions(
    sessions_dir: pathlib.Path,
    workspace_filter: pathlib.Path | None = None,
    *,
    include_empty: bool = False,
) -> list[SessionTraceSummary]:
    """Discovers ALL resumable sessions strictly sorted most-recent-first (`updated_at` desc)."""
    sdir = sessions_dir.expanduser().resolve()
    if not sdir.is_dir():
        return []
    ws_norm = str(workspace_filter.expanduser().resolve()) if workspace_filter else None

    # Collect every session ID present as either `.jsonl` DAG or `.trace.jsonl` ledger
    session_ids: set[str] = set()
    for p in sdir.glob("sess_*.jsonl"):
        if p.name.endswith(".trace.jsonl"):
            session_ids.add(p.name[: -len(".trace.jsonl")])
        else:
            session_ids.add(p.stem)

    summaries: list[SessionTraceSummary] = []
    for sid in session_ids:
        p = sdir / f"{sid}.jsonl"
        tpath = sdir / f"{sid}.trace.jsonl"
        ws = ""
        turns = 0
        steps = 0
        tblocks = 0
        rtok = 0
        models: list[str] = []
        last_prompt = ""
        mtime = 0.0
        try:
            if p.is_file():
                mtime = max(mtime, p.stat().st_mtime)
            if tpath.is_file():
                mtime = max(mtime, tpath.stat().st_mtime)
        except OSError:
            continue

        if tpath.is_file():
            try:
                for ln in tpath.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not ln.strip():
                        continue
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    rec_ts = float(r.get("ts") or 0.0)
                    if rec_ts > mtime:
                        mtime = rec_ts
                    rt = r.get("type")
                    if rt == "turn_start":
                        turns += 1
                        ws = str(r.get("workspace") or ws)
                        cand_p = _clean_prompt_preview(str(r.get("prompt") or ""))
                        if cand_p:
                            last_prompt = cand_p
                        m = str(r.get("model") or "")
                        if m and m not in models:
                            models.append(m)
                    elif rt == "step_start":
                        steps += 1
                        m = str(r.get("model") or "")
                        if m and m not in models:
                            models.append(m)
                    elif rt == "thought_block":
                        tblocks += 1
                        rtok += int(r.get("reasoning_tokens", 0))
            except OSError:
                pass

        # Backfill or parse DAG JSONL when `.trace.jsonl` is absent or missing prompt/turns
        if p.is_file() and (turns == 0 or not last_prompt or (tblocks == 0 and steps == 0)):
            dag_turns = 0
            dag_steps = 0
            dag_tblocks = 0
            dag_rtok = 0
            try:
                for ln in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    if not ln.strip():
                        continue
                    try:
                        r = json.loads(ln)
                    except json.JSONDecodeError:
                        continue
                    role = r.get("role")
                    if role == "user":
                        parts = r.get("parts") or []
                        txts = [str(pt.get("text") or "") for pt in parts if pt.get("kind") in (None, "text")]
                        joined = "\n".join(t for t in txts if t)
                        if "[LIVE_USER_STEER]:" in joined:
                            continue
                        cand = _clean_prompt_preview(joined)
                        if cand:
                            dag_turns += 1
                            if not tpath.is_file():
                                last_prompt = cand
                            elif not last_prompt:
                                last_prompt = cand
                    elif role == "assistant":
                        dag_steps += 1
                        m = str(r.get("producer_model_id") or "")
                        if m and m not in models:
                            models.append(m)
                        for pt in r.get("parts") or []:
                            if pt.get("kind") == "thinking" and pt.get("text"):
                                dag_tblocks += 1
                        u = r.get("usage") or {}
                        dag_rtok += int(u.get("reasoning_tokens") or 0)
            except OSError:
                pass
            turns = max(turns, dag_turns)
            steps = max(steps, dag_steps)
            tblocks = max(tblocks, dag_tblocks)
            rtok = max(rtok, dag_rtok)

        if not include_empty and turns == 0 and steps == 0:
            continue
        if ws_norm and (not ws or ws != ws_norm):
            continue
        summaries.append(
            SessionTraceSummary(
                session_id=sid,
                trace_path=tpath,
                dag_path=p,
                workspace=ws,
                updated_at=mtime,
                turns=turns,
                steps=steps,
                thought_blocks=tblocks,
                reasoning_tokens=rtok,
                models=models,
                last_prompt=last_prompt,
            )
        )

    # Strict invariant: ALWAYS return most-recent-first across all discovered sessions
    summaries.sort(key=lambda s: (s.updated_at, s.session_id), reverse=True)
    return summaries


def summarize_tool_args(tool_name: str, args: dict[str, Any]) -> str:
    """Formats concise single-line tool call arguments for live long-form trace clarity."""
    if tool_name == "read_file":
        p = args.get("path", "")
        off = args.get("offset")
        lim = args.get("limit")
        rng = f", offset={off}" + (f", limit={lim}" if lim else "") if off or lim else ""
        return f'path="{p}"{rng}'
    if tool_name in ("write_file", "edit_file"):
        return f'path="{args.get("path", "")}"'
    if tool_name == "bash":
        cmd = " ".join(str(args.get("command", "")).split())
        if len(cmd) > 96:
            cmd = cmd[:93] + "..."
        d = ", daemon=true" if args.get("is_daemon") else ""
        return f'"{cmd}"{d}'
    if tool_name in ("grep", "glob"):
        pat = args.get("pattern", "")
        path = args.get("path", "")
        return f'pattern="{pat}"' + (f', path="{path}"' if path else "")
    raw = json.dumps(args, ensure_ascii=False)
    return raw[:110] + ("..." if len(raw) > 110 else "")


class StreamTracePrinter:
    """Plain, zero-flicker scrollback stream formatter for long-form multi-step chains of thought."""

    def __init__(
        self,
        *,
        mode: TraceMode = TraceMode.FULL,
        use_color: bool = False,
        out: TextIO | None = None,
        lock: threading.RLock | None = None,
    ) -> None:
        self.mode = mode
        self.use_color = use_color
        self.out = out or sys.stdout
        self.lock = lock or threading.RLock()
        self._in_think = False
        self._think_bol = True  # At beginning of a thought line -> emit `⋮ ` gutter
        self._wrote_text = False
        self._text_bol = True
        self._compact_buf: list[str] = []

    def _dim(self, s: str) -> str:
        return f"\033[2m{s}\033[0m" if self.use_color else s

    def _cyan(self, s: str) -> str:
        return f"\033[36m{s}\033[0m" if self.use_color else s

    def _green(self, s: str) -> str:
        return f"\033[32m{s}\033[0m" if self.use_color else s

    def _yellow(self, s: str) -> str:
        return f"\033[33m{s}\033[0m" if self.use_color else s

    def _red(self, s: str) -> str:
        return f"\033[31m{s}\033[0m" if self.use_color else s

    def _close_open_streams_unlocked(self) -> None:
        if self._in_think:
            if not self._think_bol:
                self.out.write("\n")
            self._in_think = False
            self._think_bol = True
        if self._wrote_text:
            if not self._text_bol:
                self.out.write("\n")
            self._wrote_text = False
            self._text_bol = True

    def emit_external_note(self, note: str) -> None:
        """Safely prints an interactive notice (e.g. live steer queued) without corrupting mid-line thought deltas."""
        with self.lock:
            if (self._in_think and not self._think_bol) or (self._wrote_text and not self._text_bol):
                self.out.write("\n")
                self._think_bol = True
                self._text_bol = True
            self.out.write(self._yellow(f"  » {note}") + "\n")
            self.out.flush()

    def __call__(self, evt: StreamEvent) -> None:
        with self.lock:
            kind = evt.kind

            if kind == "step_start":
                if self.mode == TraceMode.OFF:
                    return
                self._close_open_streams_unlocked()
                self._compact_buf.clear()
                m = evt.meta
                ctx_k = int(m.get("context_tokens", 0)) / 1000.0
                effort = m.get("thinking_effort", "high")
                elapsed = f"+{evt.elapsed_ms / 1000.0:.1f}s" if evt.step > 1 else "0.0s"
                hdr = (
                    f"\n── step {evt.step}/{m.get('max_steps', '?')} • {m.get('model', '')} "
                    f"(thinking={effort} • ctx={ctx_k:.1f}k • {elapsed}) ──"
                )
                self.out.write(self._dim(hdr) + "\n")
                self.out.flush()
                return

            if kind == "thinking_delta" and evt.text:
                if self.mode == TraceMode.OFF:
                    return
                if self.mode == TraceMode.COMPACT:
                    self._compact_buf.append(evt.text)
                    return
                # TraceMode.FULL -> live verbatim guttered stream!
                if not self._in_think:
                    self._in_think = True
                    self._think_bol = True
                chunks = evt.text.splitlines(keepends=True)
                rendered: list[str] = []
                for ch in chunks:
                    if self._think_bol:
                        rendered.append(self._dim("⋮ "))
                    rendered.append(self._dim(ch))
                    self._think_bol = ch.endswith("\n")
                self.out.write("".join(rendered))
                self.out.flush()
                return

            if kind == "thinking_end":
                if self.mode == TraceMode.OFF:
                    return
                chars = int(evt.meta.get("thought_chars", 0))
                rtok = int(evt.meta.get("reasoning_tokens", 0))
                sig = "✓" if evt.meta.get("has_signature") else "-"
                secs = evt.elapsed_ms / 1000.0
                if self.mode == TraceMode.COMPACT:
                    joined = "".join(self._compact_buf)
                    self._compact_buf.clear()
                    cues = extract_thought_landmarks(joined, limit=2)
                    summary = " -> ".join(cues) if cues else f"{chars} chars"
                    self.out.write(self._dim(f"  ◌ thought ({secs:.2f}s • ~{rtok} tok): {summary}\n"))
                    self.out.flush()
                    return
                if self._in_think and not self._think_bol:
                    self.out.write("\n")
                self._in_think = False
                self._think_bol = True
                foot = f"── /thought ({secs:.2f}s • ~{rtok} thought tok • {chars:,} chars • sig={sig}) ──\n"
                self.out.write(self._dim(foot))
                self.out.flush()
                return

            if kind == "text_delta" and evt.text:
                if self._in_think:
                    if not self._think_bol:
                        self.out.write("\n")
                    self._in_think = False
                    self._think_bol = True
                    self.out.write("\n")
                self.out.write(evt.text)
                self.out.flush()
                self._wrote_text = True
                self._text_bol = evt.text.endswith("\n")
                return

            if kind == "tool_call" and evt.part:
                if self.mode == TraceMode.OFF:
                    return
                self._close_open_streams_unlocked()
                arg_s = summarize_tool_args(evt.part.tool_name or "", evt.part.tool_args or {})
                self.out.write(self._cyan(f"  ▸ {evt.part.tool_name}({arg_s})") + "\n")
                self.out.flush()
                return

            if kind == "tool_result" and evt.part:
                if self.mode == TraceMode.OFF:
                    return
                self._close_open_streams_unlocked()
                txt = evt.part.text or ""
                lines_n = txt.count("\n") + (1 if txt and not txt.endswith("\n") else 0)
                first = ""
                for ln in txt.splitlines():
                    s = ln.strip()
                    if s:
                        first = s[:96] + ("..." if len(s) > 96 else "")
                        break
                tag = self._red("ERR") if evt.part.is_error else self._green("ok")
                meta_s = f"{evt.elapsed_ms:.0f}ms • {lines_n}L/{len(txt):,}B"
                tail = f" — {first}" if first else ""
                self.out.write(f"  ◂ {tag} {self._dim(f'{evt.part.tool_name} ({meta_s}){tail}')}\n")
                self.out.flush()
                return

            if kind == "cognitive_event":
                if self.mode == TraceMode.OFF:
                    return
                self._close_open_streams_unlocked()
                ev_name = evt.meta.get("event", "cognitive")
                self.out.write(self._yellow(f"  ◈ [{ev_name}] {evt.text}") + "\n")
                self.out.flush()
                return

            if kind == "done":
                self._close_open_streams_unlocked()
                if self.mode == TraceMode.OFF:
                    return
                u = evt.usage
                if u:
                    cache_pct = 100.0 * u.cache_hit_rate
                    t_path = evt.meta.get("trace_path", "")
                    tail_p = f" • trace={pathlib.Path(t_path).name}" if t_path else ""
                    summary = (
                        f"── done ({evt.step} step{'s' if evt.step != 1 else ''} • {evt.elapsed_ms / 1000.0:.2f}s "
                        f"• in={u.input_tokens:,} [cache {cache_pct:.0f}%] "
                        f"• thought={u.reasoning_tokens:,} tok • out={u.output_tokens:,} tok{tail_p}) ──\n"
                    )
                    self.out.write(self._dim(summary))
                    self.out.flush()
