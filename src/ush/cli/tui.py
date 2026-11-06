"""Visual Inline-Dock Terminal UI (`ush`): Navigable Menus, Live Thinking Threads & Concurrent Prompt Queue.

Zero-third-party-dependency Jetski-CLI / Gemini-CLI-style inline terminal compositor:
1. Preserves 100% native terminal scrollback & mouse text selection (no destructive alt-screen wipe).
2. Enforces `INV-TUI-1`: every line in the pinned bottom dock is strictly clamped to `<= cols - 1` visible
   Unicode columns (`vis_width`), preventing terminal line-wrap row-count drift and ghost lines.
3. Enforces `INV-TUI-2`: single-lock DEC Synchronized Output (`CSI ? 2026 h ... l`) diff flushing so
   streaming multi-step chains of thought scroll smoothly ABOVE your live composer and open menus.
4. Enforces `INV-TUI-3`: guaranteed `termios` + Bracketed Paste (`CSI ? 2004 l`) + cursor restoration.
5. Three first-class interactive capabilities:
   - Navigable Arrow-Key Menus: auto-popup `/` slash, `@file`, `$skill` fuzzy menus + modal pickers for
     `/model`, `/thinking`, `/turbo`, `/permissions`, `/trace`, `/sessions`, `/auth`, approvals & `ask_user`.
   - Visual Thinking Threads: structured `╭─ ◈ Step N/M` scrollback thread cards, live animated active
     thinking-thread ticker strip, and `Ctrl+O` / `/threads` navigable thread inspector & search overlay.
   - Always-Live Concurrent Composer: type & queue follow-up prompts (`Enter` -> Up-Next Queue, auto-runs
     when current turn finishes) OR press `Alt+S` / `/steer` to inject immediately mid-turn while watching
     live thinking threads stream above.
"""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
import os
import pathlib
import re
import select
import shutil
import signal
import sys
import termios
import threading
import time
import tty
from typing import TYPE_CHECKING, Any, Callable, TextIO
import unicodedata

from ush.brain.trace import extract_thought_landmarks, format_relative_age, list_workspace_sessions
from ush.core.types import (
    CanonicalPart,
    PermissionMode,
    StreamEvent,
    ThinkingEffort,
    TraceMode,
)
from ush.kernel.security import redact_secrets

if TYPE_CHECKING:
    from ush.cli.main import HarnessRuntime

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
SPINNER_FRAMES: tuple[str, ...] = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

# Curated frontier & local models surfaced in `/model` visual menu (plus live custom entry & Ollama probe)
RECOMMENDED_MODELS: tuple[tuple[str, str, str], ...] = (
    ("gemini-4-argon", "vertex/google", "Tier S · Frontier Gemini 4 Argon (1M ctx, deep native thinking)"),
    ("gemini-3.1-pro-preview", "vertex/google", "Tier S · Gemini 3.1 Pro Preview (1M ctx, LOW/MEDIUM/HIGH thinking)"),
    ("gemini-3-flash-preview", "vertex/google", "Tier A · Gemini 3 Flash Preview (ultra-fast multi-step agentic loop)"),
    ("gemini-2.5-pro", "vertex/google", "Tier S · Gemini 2.5 Pro GA (dynamic thinkingBudget, 1M ctx)"),
    ("gemini-2.5-flash", "vertex/google", "Tier A · Gemini 2.5 Flash GA (low-latency high throughput)"),
    ("claude-opus-4-20250514", "anthropic/vertex", "Tier S · Claude Opus 4 (interleaved thinking between tool calls)"),
    ("claude-sonnet-4@20250514", "vertex-anthropic", "Tier A · Claude Sonnet 4 on Vertex Model Garden"),
    ("claude-sonnet-4-20250514", "anthropic", "Tier A · Claude Sonnet 4 Direct API"),
    ("gpt-5", "openai", "Tier S · OpenAI GPT-5 (reasoning_effort + 400k ctx)"),
    ("o3", "openai", "Tier S · OpenAI o3 native reasoner"),
    ("deepseek-reasoner", "openrouter/openai", "Tier A · DeepSeek R1 (`reasoning_content` full CoT)"),
    ("qwen3-coder", "openrouter/local", "Tier A · Qwen3-Coder agentic coder"),
    ("__custom__", "custom", "Type any custom provider:model-id (Ollama, vLLM, OpenRouter, etc.)..."),
)


# ============================================================================
# Unicode & ANSI Display-Width Primitives (`INV-TUI-1`)
# ============================================================================


def strip_ansi(text: str) -> str:
    return ANSI_ESCAPE_RE.sub("", text)


def char_width(ch: str) -> int:
    if not ch or ch < " ":
        return 0
    if unicodedata.combining(ch) or unicodedata.category(ch) in ("Mn", "Me", "Cf"):
        return 0
    eaw = unicodedata.east_asian_width(ch)
    if eaw in ("W", "F"):
        return 2
    # Common wide symbols/emoji ranges that Python's UCD marks Neutral in some versions
    cp = ord(ch)
    if 0x1F300 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
        # Keep standard single-cell geometric glyphs (`▸`, `◂`, `◈`, `❯`, `✓`, `⚡`, `●`) right:
        # ⚡ (U+26A1) renders 2 cols on macOS/iTerm2/Ghostty/VSCode terminals!
        if cp in (0x26A1, 0x23ED, 0x23F3, 0x2705, 0x274C):
            return 2
        if 0x1F300 <= cp <= 0x1FAFF:
            return 2
    return 1


def vis_width(text: str) -> int:
    """Computes physical terminal column width ignoring ANSI SGR sequences."""
    plain = strip_ansi(text)
    return sum(char_width(ch) for ch in plain)


def clip_to_cols(text: str, max_cols: int, ellipsis: str = "…") -> str:
    """ANSI-preserving truncation guaranteeing `vis_width(result) <= max_cols` (`INV-TUI-1`)."""
    if max_cols <= 0:
        return ""
    if vis_width(text) <= max_cols:
        return text

    ell_w = vis_width(ellipsis)
    budget = max(0, max_cols - ell_w)
    out: list[str] = []
    used = 0
    i = 0
    had_ansi = False
    n = len(text)
    while i < n:
        if text[i] == "\x1b":
            m = ANSI_ESCAPE_RE.match(text, i)
            if m:
                out.append(m.group(0))
                had_ansi = True
                i = m.end()
                continue
        ch = text[i]
        if ch in ("\r", "\n", "\t"):
            ch = " "
        cw = char_width(ch)
        if used + cw > budget:
            break
        out.append(ch)
        used += cw
        i += 1
    if budget < max_cols and used + ell_w <= max_cols:
        out.append(ellipsis)
    if had_ansi:
        out.append("\x1b[0m")
    return "".join(out)


def pad_to_cols(text: str, target_cols: int) -> str:
    clipped = clip_to_cols(text, target_cols)
    rem = max(0, target_cols - vis_width(clipped))
    return clipped + (" " * rem)


class Styler:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def wrap(self, code: str, text: str) -> str:
        return f"\x1b[{code}m{text}\x1b[0m" if self.enabled else text

    def bold(self, t: str) -> str:
        return self.wrap("1", t)

    def dim(self, t: str) -> str:
        return self.wrap("2", t)

    def italic(self, t: str) -> str:
        return self.wrap("3", t)

    def cyan(self, t: str) -> str:
        return self.wrap("36", t)

    def bold_cyan(self, t: str) -> str:
        return self.wrap("1;36", t)

    def green(self, t: str) -> str:
        return self.wrap("32", t)

    def yellow(self, t: str) -> str:
        return self.wrap("33", t)

    def magenta(self, t: str) -> str:
        return self.wrap("35", t)

    def blue(self, t: str) -> str:
        return self.wrap("34", t)

    def red(self, t: str) -> str:
        return self.wrap("31", t)

    def sel_row(self, t: str) -> str:
        # High-contrast selected menu row (cyan accent bar + bold reverse/tinted bg)
        return f"\x1b[48;5;236;1;96m{t}\x1b[0m" if self.enabled else f"> {t}"

    def pill(self, fg_bg: str, t: str) -> str:
        return f"\x1b[{fg_bg}m {t} \x1b[0m" if self.enabled else f"[{t}]"


# ============================================================================
# Raw Key, Escape Sequence & Bracketed Paste Decoder (`INV-TUI-4`)
# ============================================================================


@dataclass(frozen=True)
class KeyEvent:
    kind: str
    # CHAR | PASTE | ENTER | ALT_ENTER | TAB | SHIFT_TAB | BACKSPACE | DELETE |
    # UP | DOWN | LEFT | RIGHT | ALT_UP | PGUP | PGDN | HOME | END |
    # ESC | CTRL_A | CTRL_E | CTRL_U | CTRL_W | CTRL_O | CTRL_T | CTRL_L | CTRL_C | CTRL_D | ALT_S
    text: str = ""


def decode_key_buffer(buf: str, *, esc_is_complete: bool = True) -> tuple[list[KeyEvent], str]:
    """Deterministically parses a stream of decoded terminal chars into `KeyEvent`s + incomplete tail."""
    events: list[KeyEvent] = []
    i = 0
    n = len(buf)

    while i < n:
        # 1. Bracketed paste `\x1b[200~ ... \x1b[201~`
        if buf.startswith("\x1b[200~", i):
            end_idx = buf.find("\x1b[201~", i + 6)
            if end_idx == -1:
                if esc_is_complete and n - i > 65_536:
                    # Safety bail if terminal dropped closing marker on gigantic paste
                    pasted = buf[i + 6 :].replace("\r\n", "\n").replace("\r", "\n")
                    events.append(KeyEvent("PASTE", pasted))
                    return events, ""
                return events, buf[i:]
            pasted = buf[i + 6 : end_idx].replace("\r\n", "\n").replace("\r", "\n")
            events.append(KeyEvent("PASTE", pasted))
            i = end_idx + 6
            continue

        ch = buf[i]

        # 2. Escape / CSI / SS3 / Alt+Key sequences
        if ch == "\x1b":
            if i + 1 >= n:
                if esc_is_complete:
                    events.append(KeyEvent("ESC"))
                    i += 1
                    continue
                return events, buf[i:]

            nxt = buf[i + 1]
            if nxt == "[":
                # CSI sequence: `\x1b[` + params `[0-9;?]*` + final `[@-~]`
                j = i + 2
                while j < n and (buf[j].isdigit() or buf[j] in ";?"):
                    j += 1
                if j >= n:
                    if esc_is_complete:
                        i = j
                        continue
                    return events, buf[i:]
                final = buf[j]
                seq = buf[i + 2 : j + 1]
                i = j + 1
                if seq == "A":
                    events.append(KeyEvent("UP"))
                elif seq == "B":
                    events.append(KeyEvent("DOWN"))
                elif seq == "C":
                    events.append(KeyEvent("RIGHT"))
                elif seq == "D":
                    events.append(KeyEvent("LEFT"))
                elif seq in ("H", "1~", "7~"):
                    events.append(KeyEvent("HOME"))
                elif seq in ("F", "4~", "8~"):
                    events.append(KeyEvent("END"))
                elif seq == "Z":
                    events.append(KeyEvent("SHIFT_TAB"))
                elif seq == "3~":
                    events.append(KeyEvent("DELETE"))
                elif seq == "5~":
                    events.append(KeyEvent("PGUP"))
                elif seq == "6~":
                    events.append(KeyEvent("PGDN"))
                elif seq in ("1;3A", "1;9A"):
                    events.append(KeyEvent("ALT_UP"))
                elif final == "u" and seq.startswith("13;"):
                    # Kitty/CSI-u Shift+Enter or Alt+Enter -> newline in composer!
                    events.append(KeyEvent("ALT_ENTER"))
                continue

            if nxt == "O":
                if i + 2 >= n:
                    if esc_is_complete:
                        i += 2
                        continue
                    return events, buf[i:]
                code = buf[i + 2]
                i += 3
                mapping = {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT", "H": "HOME", "F": "END"}
                if code in mapping:
                    events.append(KeyEvent(mapping[code]))
                continue

            # Alt + key (`\x1b` + recognized chord); otherwise emit `ESC` without swallowing next key!
            if nxt in ("\r", "\n"):
                i += 2
                events.append(KeyEvent("ALT_ENTER"))
            elif nxt in ("s", "S"):
                i += 2
                events.append(KeyEvent("ALT_S"))
            elif nxt == "\x1b":
                i += 2
                events.append(KeyEvent("ESC"))
            else:
                i += 1
                events.append(KeyEvent("ESC"))
            continue

        # 3. Control keys
        i += 1
        if ch in ("\r", "\n"):
            # Coalesce CRLF if terminal emits `\r\n`
            if ch == "\r" and i < n and buf[i] == "\n":
                i += 1
            events.append(KeyEvent("ENTER"))
        elif ch == "\t":
            events.append(KeyEvent("TAB"))
        elif ch in ("\x7f", "\x08"):
            events.append(KeyEvent("BACKSPACE"))
        elif ch == "\x03":
            events.append(KeyEvent("CTRL_C"))
        elif ch == "\x04":
            events.append(KeyEvent("CTRL_D"))
        elif ch == "\x01":
            events.append(KeyEvent("HOME"))
        elif ch == "\x05":
            events.append(KeyEvent("END"))
        elif ch == "\x15":
            events.append(KeyEvent("CTRL_U"))
        elif ch == "\x17":
            events.append(KeyEvent("CTRL_W"))
        elif ch == "\x0f":
            events.append(KeyEvent("CTRL_O"))
        elif ch == "\x14":
            events.append(KeyEvent("CTRL_T"))
        elif ch == "\x0c":
            events.append(KeyEvent("CTRL_L"))
        elif ch == "\x10":  # Ctrl+P -> UP in menus / history
            events.append(KeyEvent("UP"))
        elif ch == "\x0e":  # Ctrl+N -> DOWN in menus / history
            events.append(KeyEvent("DOWN"))
        elif ch >= " ":
            events.append(KeyEvent("CHAR", ch))

    return events, ""


# ============================================================================
# Thinking Threads State Tracker & Visual `Ctrl+O` Inspector
# ============================================================================


@dataclass
class ToolStepRecord:
    call_id: str
    name: str
    arg_summary: str
    ok: bool | None = None
    elapsed_ms: float = 0.0
    result_summary: str = ""


@dataclass
class ThinkingThreadNode:
    thread_id: str
    turn: int
    step: int
    max_steps: int
    model: str
    effort: str
    started_at: float
    status: str = "thinking"  # thinking | tools | done | error
    thought_chunks: list[str] = field(default_factory=list)
    headline: str = "Formulating initial plan..."
    thought_ms: float = 0.0
    reasoning_tokens: int = 0
    has_signature: bool = False
    tools: list[ToolStepRecord] = field(default_factory=list)
    cognitive_notes: list[str] = field(default_factory=list)
    response_preview: str = ""

    @property
    def thought_text(self) -> str:
        return "".join(self.thought_chunks)

    def update_headline(self) -> None:
        raw = self.thought_text
        marks = extract_thought_landmarks(raw, limit=6)
        if marks:
            self.headline = marks[-1][:84]
            return
        lines = [ln.strip(" #*-•\t") for ln in raw.splitlines() if ln.strip()]
        if lines:
            self.headline = lines[-1][:84]

    def live_tokens_est(self) -> int:
        if self.reasoning_tokens > 0:
            return self.reasoning_tokens
        chars = sum(len(c) for c in self.thought_chunks)
        return max(1, chars // 4) if chars else 0


class ThinkingThreadTracker:
    """Thread-safe live registry of multi-step thinking threads for the status strip & `Ctrl+O` navigator."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.turn_counter: int = 0
        self.turn_Active: bool = False
        self.threads: list[ThinkingThreadNode] = []
        self.active_thread: ThinkingThreadNode | None = None
        self._resp_buf: list[str] = []

    def load_historical_from_ledger(self, records: list[dict[str, Any]]) -> None:
        with self._lock:
            by_key: dict[tuple[int, int], ThinkingThreadNode] = {}
            max_turn = 0
            for r in records:
                t = int(r.get("turn") or 0)
                s = int(r.get("step") or 0)
                max_turn = max(max_turn, t)
                rtype = r.get("type")
                if rtype == "thought_block" and t > 0 and s > 0:
                    node = ThinkingThreadNode(
                        thread_id=f"T{t}·S{s}",
                        turn=t,
                        step=s,
                        max_steps=80,
                        model=str(r.get("model") or ""),
                        effort=str(r.get("thinking_effort") or "high"),
                        started_at=time.monotonic(),
                        status="done",
                        thought_chunks=[str(r.get("thought") or "")],
                        thought_ms=float(r.get("thought_ms") or 0.0),
                        reasoning_tokens=int(r.get("reasoning_tokens") or 0),
                        has_signature=bool(r.get("has_signature")),
                    )
                    node.update_headline()
                    by_key[(t, s)] = node
                    self.threads.append(node)
                elif rtype == "assistant_step" and (t, s) in by_key:
                    node = by_key[(t, s)]
                    node.response_preview = str(r.get("text") or "")[:160]
                    for tc in r.get("tool_calls") or []:
                        args = tc.get("args") or {}
                        arg_s = str(args.get("path") or args.get("command") or args.get("pattern") or "")[:56]
                        node.tools.append(
                            ToolStepRecord(
                                call_id=str(tc.get("id") or ""),
                                name=str(tc.get("name") or "tool"),
                                arg_summary=arg_s,
                                ok=True,
                            )
                        )
                elif rtype == "tool_result" and (t, s) in by_key:
                    node = by_key[(t, s)]
                    tname = str(r.get("tool") or "tool")
                    for tr in node.tools:
                        if tr.name == tname and tr.elapsed_ms == 0.0:
                            tr.ok = not bool(r.get("is_error"))
                            tr.elapsed_ms = float(r.get("elapsed_ms") or 0.0)
                            pv = str(r.get("preview") or "").splitlines()
                            tr.result_summary = pv[0][:80] if pv else ""
                            break
                elif rtype == "cognitive_event" and (t, s) in by_key:
                    by_key[(t, s)].cognitive_notes.append(f"[{r.get('event')}] {r.get('detail', '')[:120]}")
            self.turn_counter = max_turn

    def on_event(self, evt: StreamEvent) -> None:
        with self._lock:
            k = evt.kind
            if k == "turn_start":
                self.turn_counter += 1
                self.turn_Active = True
                self._resp_buf.clear()
                return

            if k == "step_start":
                if self.active_thread and self.active_thread.status in ("thinking", "tools"):
                    self.active_thread.status = "done"
                m = evt.meta
                node = ThinkingThreadNode(
                    thread_id=f"T{max(1, self.turn_counter)}·S{evt.step}",
                    turn=max(1, self.turn_counter),
                    step=evt.step,
                    max_steps=int(m.get("max_steps") or 80),
                    model=str(m.get("model") or ""),
                    effort=str(m.get("thinking_effort") or "high"),
                    started_at=time.monotonic(),
                    status="thinking",
                )
                self.threads.append(node)
                self.active_thread = node
                self._resp_buf.clear()
                return

            cur = self.active_thread
            if k == "thinking_delta" and evt.text and cur is not None:
                cur.status = "thinking"
                cur.thought_chunks.append(evt.text)
                cur.update_headline()
                return

            if k == "thinking_end" and cur is not None:
                cur.thought_ms = float(evt.elapsed_ms)
                cur.reasoning_tokens = int(evt.meta.get("reasoning_tokens") or cur.live_tokens_est())
                cur.has_signature = bool(evt.meta.get("has_signature"))
                cur.update_headline()
                if cur.status == "thinking":
                    cur.status = "done"
                return

            if k == "tool_call" and evt.part is not None and cur is not None:
                cur.status = "tools"
                args = evt.part.tool_args or {}
                arg_s = str(
                    args.get("path")
                    or args.get("command")
                    or args.get("pattern")
                    or args.get("query")
                    or args.get("skill_name")
                    or ""
                ).replace("\n", " ")[:60]
                cur.tools.append(
                    ToolStepRecord(
                        call_id=str(evt.part.tool_call_id or ""),
                        name=str(evt.part.tool_name or "tool"),
                        arg_summary=arg_s,
                    )
                )
                if cur.headline == "Formulating initial plan...":
                    cur.headline = f"Calling {evt.part.tool_name}({arg_s})"[:84]
                return

            if k == "tool_result" and evt.part is not None and cur is not None:
                cid = str(evt.part.tool_call_id or "")
                tname = str(evt.part.tool_name or "tool")
                first_ln = (evt.part.text or "").strip().splitlines()
                summary = first_ln[0][:88] if first_ln else ""
                matched = False
                for rec in cur.tools:
                    if (cid and rec.call_id == cid) or (rec.ok is None and rec.name == tname):
                        rec.ok = not evt.part.is_error
                        rec.elapsed_ms = float(evt.elapsed_ms)
                        rec.result_summary = summary
                        matched = True
                        break
                if not matched:
                    cur.tools.append(
                        ToolStepRecord(
                            call_id=cid,
                            name=tname,
                            arg_summary="",
                            ok=not evt.part.is_error,
                            elapsed_ms=float(evt.elapsed_ms),
                            result_summary=summary,
                        )
                    )
                if all(t.ok is not None for t in cur.tools):
                    cur.status = "done"
                return

            if k == "text_delta" and evt.text and cur is not None:
                if len(self._resp_buf) < 8:
                    self._resp_buf.append(evt.text)
                    cur.response_preview = "".join(self._resp_buf).strip()[:160]
                return

            if k == "cognitive_event" and cur is not None:
                ev_name = str(evt.meta.get("event") or "event")
                cur.cognitive_notes.append(f"[{ev_name}] {evt.text[:140]}")
                if ev_name == "provider_error":
                    cur.status = "error"
                return

            if k == "done":
                if cur is not None and cur.status in ("thinking", "tools"):
                    cur.status = "done"
                self.turn_Active = False

    def snapshot_live_status(self, spin_idx: int) -> str | None:
        with self._lock:
            if not self.turn_Active or self.active_thread is None:
                return None
            t = self.active_thread
            spin = SPINNER_FRAMES[spin_idx % len(SPINNER_FRAMES)]
            elapsed = time.monotonic() - t.started_at
            running_tools = [x for x in t.tools if x.ok is None]
            if running_tools:
                rt0 = running_tools[0]
                extra = f" +{len(running_tools) - 1} parallel" if len(running_tools) > 1 else ""
                arg = f"({rt0.arg_summary})" if rt0.arg_summary else ""
                return (
                    f"{spin} ◈ {t.thread_id}/{t.max_steps} · ▸ {rt0.name}{arg}{extra} "
                    f"· {elapsed:.1f}s · {t.live_tokens_est():,} tok"
                )
            return (
                f"{spin} ◈ {t.thread_id}/{t.max_steps} · {t.model} ({t.effort}) "
                f"· {elapsed:.1f}s · ~{t.live_tokens_est():,} tok — {t.headline}"
            )

    def list_threads(self, query: str = "") -> list[ThinkingThreadNode]:
        with self._lock:
            q = query.strip().lower()
            if not q:
                return list(self.threads)
            out: list[ThinkingThreadNode] = []
            for t in self.threads:
                hay = f"{t.thread_id} {t.model} {t.headline} {t.thought_text} {' '.join(x.name + ' ' + x.arg_summary for x in t.tools)}".lower()
                if q in hay:
                    out.append(t)
            return out


# ============================================================================
# Navigable Menus (`/` Slash, `@file`, `$skill`, Modal Pickers & `Ctrl+O` Threads)
# ============================================================================


@dataclass
class MenuItem:
    key: str
    label: str
    detail: str = ""
    badge: str = ""
    active: bool = False
    payload: Any = None


@dataclass
class MenuOverlay:
    kind: str
    # auto_slash | auto_file | auto_skill | picker | approval | ask_user | threads
    title: str
    items: list[MenuItem]
    selected: int = 0
    filter_query: str = ""
    allow_filter: bool = True
    max_visible: int = 8
    replace_span: tuple[int, int] | None = None  # char span in input_buf for auto-completion menus
    on_select: Callable[[MenuItem], None] | None = None
    on_cancel: Callable[[], None] | None = None
    # For `threads` explorer overlay:
    body_scroll: int = 0
    expanded_detail: bool = True

    def filtered_items(self) -> list[MenuItem]:
        q = self.filter_query.strip().lower()
        if not q:
            return self.items
        scored: list[tuple[int, int, MenuItem]] = []
        for idx, it in enumerate(self.items):
            kl = it.key.lower()
            ll = it.label.lower()
            dl = it.detail.lower()
            if kl == q:
                scored.append((0, idx, it))
            elif kl.startswith(q) or ll.startswith(q):
                scored.append((1, idx, it))
            elif q in kl or q in ll:
                scored.append((2, idx, it))
            elif q in dl:
                scored.append((3, idx, it))
            else:
                # Subsequence fuzzy match on key/label
                qi = 0
                for ch in ll:
                    if qi < len(q) and ch == q[qi]:
                        qi += 1
                if qi == len(q) and len(q) >= 2:
                    scored.append((4, idx, it))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [it for _, _, it in scored]

    def clamp_selection(self) -> list[MenuItem]:
        vis = self.filtered_items()
        if not vis:
            self.selected = 0
        else:
            self.selected = max(0, min(self.selected, len(vis) - 1))
        return vis

    def move(self, delta: int) -> None:
        vis = self.filtered_items()
        if not vis:
            self.selected = 0
            return
        self.selected = (self.selected + delta) % len(vis)
        self.body_scroll = 0

    def current(self) -> MenuItem | None:
        vis = self.clamp_selection()
        return vis[self.selected] if vis else None


class WorkspaceFileIndex:
    """Fast cached workspace file scanner for instant `@file` arrow-key completion."""

    SKIP_DIRS: frozenset[str] = frozenset(
        {".git", ".hg", ".venv", "venv", "node_modules", "__pycache__", ".mypy_cache", ".pytest_cache", "dist", "build"}
    )

    def __init__(self, workspace: pathlib.Path, max_files: int = 1_500) -> None:
        self.workspace = workspace
        self.max_files = max_files
        self._cache: list[str] = []
        self._built_at: float = 0.0

    def list_files(self) -> list[str]:
        now = time.monotonic()
        if self._cache and (now - self._built_at) < 20.0:
            return self._cache
        found: list[str] = []
        try:
            stack = [self.workspace]
            while stack and len(found) < self.max_files:
                cur = stack.pop()
                try:
                    entries = sorted(cur.iterdir(), key=lambda p: (not p.is_file(), p.name.lower()))
                except OSError:
                    continue
                dirs_to_push: list[pathlib.Path] = []
                for ent in entries:
                    name = ent.name
                    if name in self.SKIP_DIRS or name.startswith(".ush"):
                        continue
                    try:
                        rel = str(ent.relative_to(self.workspace))
                    except ValueError:
                        continue
                    if ent.is_dir() and not ent.is_symlink():
                        # Avoid deep crawl when started from huge home/Documents dir: keep depth <= 3
                        if rel.count(os.sep) < 3:
                            dirs_to_push.append(ent)
                    elif ent.is_file():
                        found.append(rel)
                        if len(found) >= self.max_files:
                            break
                for d in reversed(dirs_to_push):
                    stack.append(d)
        except OSError:
            pass
        self._cache = found
        self._built_at = now
        return self._cache


# ============================================================================
# Inline Pinned Dock Compositor (`INV-TUI-1`, `INV-TUI-2`)
# ============================================================================


class InlineDockCompositor:
    """Renders streaming permanent scrollback ABOVE a flicker-free, wrap-safe pinned bottom dock."""

    def __init__(
        self,
        *,
        out: TextIO | None = None,
        style: Styler | None = None,
        cols_fn: Callable[[], int] | None = None,
        use_sync_output: bool = True,
    ) -> None:
        self.out = out or sys.stdout
        self.style = style or Styler(True)
        self._cols_fn = cols_fn or (lambda: shutil.get_terminal_size((100, 30)).columns)
        self.use_sync_output = use_sync_output
        self.lock = threading.RLock()
        self._dock_rows_drawn: int = 0
        self._cursor_row_from_top: int = 0
        self._pending_scroll_lines: list[str] = []
        self._live_partial_line: str = ""
        self._dock_builder: Callable[[int], tuple[list[str], int, int]] | None = None
        self._closed = False

    def cols(self) -> int:
        return max(48, self._cols_fn())

    def rows(self) -> int:
        return max(12, shutil.get_terminal_size((100, 30)).lines)

    def set_dock_builder(self, fn: Callable[[int], tuple[list[str], int, int]]) -> None:
        """`fn(cols)` returns `(dock_lines_each_le_cols_minus_1, cursor_row_idx, cursor_col_1based)`."""
        with self.lock:
            self._dock_builder = fn

    def write_scroll(self, text: str) -> None:
        """Streams arbitrary text above the dock, committing `\n` lines permanently to native scrollback."""
        if not text:
            return
        with self.lock:
            combined = self._live_partial_line + text.replace("\r\n", "\n")
            parts = combined.split("\n")
            for complete in parts[:-1]:
                self._pending_scroll_lines.append(complete)
            self._live_partial_line = parts[-1]

    def println(self, line: str = "") -> None:
        with self.lock:
            if self._live_partial_line:
                self._pending_scroll_lines.append(self._live_partial_line)
                self._live_partial_line = ""
            for sub in line.replace("\r\n", "\n").split("\n"):
                self._pending_scroll_lines.append(sub)
        self.render()

    def clear_dock(self) -> None:
        with self.lock:
            buf: list[str] = []
            if self._dock_rows_drawn > 0:
                if self._cursor_row_from_top > 0:
                    buf.append(f"\x1b[{self._cursor_row_from_top}A")
                buf.append("\r\x1b[J")
                self._dock_rows_drawn = 0
                self._cursor_row_from_top = 0
            if self._live_partial_line:
                buf.append(self._live_partial_line + "\n")
                self._live_partial_line = ""
            for ln in self._pending_scroll_lines:
                buf.append(ln + "\n")
            self._pending_scroll_lines.clear()
            buf.append("\x1b[?25h")
            if buf:
                self.out.write("".join(buf))
                self.out.flush()

    def render(self) -> None:
        with self.lock:
            if self._closed or self._dock_builder is None:
                return
            width = self.cols()
            max_line_w = max(20, width - 1)
            dock_raw, cur_row, cur_col = self._dock_builder(width)

            # Prepend in-progress partial stream line (clamped safely) at row 0 of erasable zone
            erasable: list[str] = []
            row_offset = 0
            if self._live_partial_line:
                erasable.append(clip_to_cols(self._live_partial_line, max_line_w))
                row_offset = 1
            for dl in dock_raw:
                # Strict `INV-TUI-1`: NEVER allow a dock line to exceed `width - 1` physical columns!
                erasable.append(clip_to_cols(dl, max_line_w))

            if not erasable:
                erasable = [""]
            target_row = max(0, min(len(erasable) - 1, cur_row + row_offset))
            target_col = max(1, min(max_line_w, cur_col))

            # Vertical viewport guard (`INV-TUI-1`): never exceed `rows - 2` so `\x1b[nA` never hits top margin!
            max_dock_h = max(7, self.rows() - 2)
            if len(erasable) > max_dock_h:
                drop = len(erasable) - max_dock_h
                erasable = erasable[drop:]
                target_row = max(0, target_row - drop)

            buf: list[str] = []
            if self.use_sync_output:
                buf.append("\x1b[?2026h")

            # 1. Move to top of previous dock & erase to end of screen
            if self._dock_rows_drawn > 0:
                if self._cursor_row_from_top > 0:
                    buf.append(f"\x1b[{self._cursor_row_from_top}A")
                buf.append("\r\x1b[J")
            else:
                buf.append("\r\x1b[K")

            # 2. Emit newly completed permanent scrollback lines (never redrawn again!)
            if self._pending_scroll_lines:
                for sl in self._pending_scroll_lines:
                    buf.append(sl + "\n")
                self._pending_scroll_lines.clear()

            # 3. Draw fresh dock lines separated by `\n` (no trailing `\n` on final line!)
            buf.append("\n".join(erasable))

            # 4. Position cursor directly inside the live prompt input box!
            up_from_bottom = (len(erasable) - 1) - target_row
            if up_from_bottom > 0:
                buf.append(f"\x1b[{up_from_bottom}A")
            buf.append(f"\x1b[{target_col}G\x1b[?25h")

            if self.use_sync_output:
                buf.append("\x1b[?2026l")

            self._dock_rows_drawn = len(erasable)
            self._cursor_row_from_top = target_row

            self.out.write("".join(buf))
            self.out.flush()


# ============================================================================
# Visual Stream Event Formatter (Cards for Thinking Threads, Tools & Replies)
# ============================================================================


class VisualStreamRenderer:
    """Formats `StreamEvent`s into clean Jetski-style thinking thread cards + feeds `ThinkingThreadTracker`."""

    BOLD_MD_RE = re.compile(r"\*\*([^*\n]{1,90})\*\*")

    def __init__(
        self,
        compositor: InlineDockCompositor,
        tracker: ThinkingThreadTracker,
        *,
        mode_fn: Callable[[], TraceMode],
        on_dirty: Callable[[], None],
    ) -> None:
        self.comp = compositor
        self.s = compositor.style
        self.tracker = tracker
        self.mode_fn = mode_fn
        self.on_dirty = on_dirty
        self._in_think = False
        self._think_bol = True
        self._in_text = False
        self._last_flush = 0.0

    def _style_thought_chunk(self, chunk: str) -> str:
        # Highlight **Landmark Titles** cleanly inside dim thought cards
        def _repl(m: re.Match[str]) -> str:
            if not self.s.enabled:
                return f"**{m.group(1)}**"
            return f"\x1b[0;1;36m{m.group(1)}\x1b[0;2m"

        styled = self.BOLD_MD_RE.sub(_repl, chunk)
        return f"\x1b[2m{styled}\x1b[0m" if self.s.enabled else chunk

    def _close_open_blocks(self) -> None:
        if self._in_think:
            self.comp.write_scroll("\n")
            self._in_think = False
            self._think_bol = True
        if self._in_text:
            self.comp.write_scroll("\n")
            self._in_text = False

    def _maybe_flush(self, force: bool = False) -> None:
        now = time.monotonic()
        if force or (now - self._last_flush) >= 0.033:
            self._last_flush = now
            self.on_dirty()

    def emit_external_note(self, note: str) -> None:
        self._close_open_blocks()
        self.comp.write_scroll(f"  {self.s.yellow('» ' + note)}\n")
        self._maybe_flush(force=True)

    def __call__(self, evt: StreamEvent) -> None:
        self.tracker.on_event(evt)
        mode = self.mode_fn()
        k = evt.kind

        if k == "step_start":
            self._close_open_blocks()
            if mode != TraceMode.OFF:
                m = evt.meta
                ctx_k = int(m.get("context_tokens", 0)) / 1000.0
                effort = m.get("thinking_effort", "high")
                elapsed = f"+{evt.elapsed_ms / 1000.0:.1f}s" if evt.step > 1 else "0.0s"
                max_s = m.get("max_steps", "?")
                mdl = m.get("model", "")
                hdr = (
                    f"╭─ {self.s.bold_cyan(f'◈ Step {evt.step}/{max_s}')} "
                    f"{self.s.dim(f'· {mdl} · effort:{effort} · ctx:{ctx_k:.1f}k · {elapsed}')}"
                )
                self.comp.write_scroll("\n" + hdr + "\n")
            self._maybe_flush(force=True)
            return

        if k == "thinking_delta" and evt.text:
            if mode == TraceMode.FULL:
                self._in_think = True
                gutter = self.s.cyan("│ ")
                pieces = evt.text.splitlines(keepends=True)
                out: list[str] = []
                for pc in pieces:
                    if self._think_bol:
                        out.append(gutter)
                    body = pc[:-1] if pc.endswith("\n") else pc
                    if body:
                        out.append(self._style_thought_chunk(body))
                    if pc.endswith("\n"):
                        out.append("\n")
                        self._think_bol = True
                    else:
                        self._think_bol = False
                self.comp.write_scroll("".join(out))
            self._maybe_flush(force=False)
            return

        if k == "thinking_end":
            m = evt.meta
            secs = evt.elapsed_ms / 1000.0
            rtok = int(m.get("reasoning_tokens") or 0)
            sig = "sig ✓" if m.get("has_signature") else "unsigned"
            if mode == TraceMode.FULL:
                if self._in_think and not self._think_bol:
                    self.comp.write_scroll("\n")
                self._in_think = False
                self._think_bol = True
                self.comp.write_scroll(
                    self.s.dim(f"╰─ ◇ reasoned {secs:.2f}s · ~{rtok:,} tok · {sig}  (^O threads)\n")
                )
            elif mode == TraceMode.COMPACT:
                hl = self.tracker.active_thread.headline if self.tracker.active_thread else ""
                self.comp.write_scroll(
                    self.s.dim(f"╰─ ◇ reasoned {secs:.2f}s · ~{rtok:,} tok · {hl}  (^O expand)\n")
                )
            self._maybe_flush(force=True)
            return

        if k == "tool_call" and evt.part is not None:
            self._close_open_blocks()
            if mode != TraceMode.OFF:
                args = evt.part.tool_args or {}
                arg_preview = ", ".join(
                    f"{ak}={str(av).replace(chr(10), ' ')[:48]!r}" for ak, av in list(args.items())[:3]
                )
                self.comp.write_scroll(
                    f"  {self.s.cyan('├─ ▸')} {self.s.bold(str(evt.part.tool_name))}({self.s.dim(arg_preview)})\n"
                )
            self._maybe_flush(force=True)
            return

        if k == "tool_result" and evt.part is not None:
            self._close_open_blocks()
            if mode != TraceMode.OFF:
                raw = evt.part.text or ""
                nlines = raw.count("\n") + (1 if raw else 0)
                first = raw.strip().splitlines()[0][:88] if raw.strip() else ""
                ms = f"{evt.elapsed_ms:.0f}ms"
                if evt.part.is_error:
                    line = f"  {self.s.red(f'└─ ◂ ✗ {evt.part.tool_name}')} {self.s.dim(f'({ms})')} {self.s.red(first)}"
                else:
                    line = f"  {self.s.green(f'└─ ◂ ✓ {evt.part.tool_name}')} {self.s.dim(f'({ms} · {nlines}L) — {first}')}"
                self.comp.write_scroll(line + "\n")
            self._maybe_flush(force=True)
            return

        if k == "cognitive_event":
            self._close_open_blocks()
            ev = str(evt.meta.get("event") or "cognitive")
            self.comp.write_scroll(f"  {self.s.yellow(f'◈ [{ev}]')} {self.s.dim(evt.text)}\n")
            self._maybe_flush(force=True)
            return

        if k == "text_delta" and evt.text:
            if self._in_think:
                self._close_open_blocks()
            if not self._in_text:
                self.comp.write_scroll("\n")
                self._in_text = True
            self.comp.write_scroll(evt.text)
            self._maybe_flush(force=False)
            return

        if k == "done":
            self._close_open_blocks()
            u = evt.usage
            cache_pct = int(round(100.0 * u.cache_hit_rate)) if u else 0
            in_t = u.input_tokens if u else 0
            th_t = u.reasoning_tokens if u else 0
            out_t = u.output_tokens if u else 0
            secs = evt.elapsed_ms / 1000.0
            footer = (
                f"── ✓ {evt.step} step{'s' if evt.step != 1 else ''} · {secs:.2f}s "
                f"· in={in_t:,} (cache {cache_pct}%) · ◈ {th_t:,} tok · out={out_t:,} tok ──"
            )
            self.comp.write_scroll("\n" + self.s.dim(footer) + "\n")
            self._maybe_flush(force=True)


# ============================================================================
# Interactive Visual Session Controller (`ush`)
# ============================================================================


class VisualTuiController:
    """Drives the Jetski-style navigable menus, live thinking threads, and concurrent prompt queue."""

    def __init__(
        self,
        rt: HarnessRuntime,
        *,
        slash_handler: Callable[[HarnessRuntime, str], tuple[bool, str | None, list[CanonicalPart]]],
        out: TextIO | None = None,
        use_color: bool = True,
        cols_fn: Callable[[], int] | None = None,
    ) -> None:
        self.rt = rt
        self.slash_handler = slash_handler
        self.s = Styler(use_color)
        self.comp = InlineDockCompositor(out=out or sys.stdout, style=self.s, cols_fn=cols_fn)
        self.tracker = ThinkingThreadTracker()
        if rt.runner.trace_ledger is not None:
            self.tracker.load_historical_from_ledger(rt.runner.trace_ledger.read_records())
        self.file_index = WorkspaceFileIndex(rt.workspace)

        # Composer state (persists untouched across background turns, menus & approval dialogs!)
        self.input_buf: str = ""
        self.cursor: int = 0
        self.history: list[str] = []
        self.hist_idx: int = 0
        self._draft_before_hist: str = ""

        # Concurrent follow-up prompt queue ("prompt the next thing while thinking"!)
        # Reentrant lock (`RLock`) + strict rule: never invoke `render()`/`flash()` while holding `_q_lock`.
        self._q_lock = threading.RLock()
        self.next_queue: deque[tuple[str, list[CanonicalPart]]] = deque()

        # Active menu / modal overlay
        self.menu: MenuOverlay | None = None
        self._custom_input_cb: Callable[[str], None] | None = None
        self._custom_input_label: str = ""

        # Background worker synchronization
        self.turn_busy = threading.Event()
        self.should_exit = False
        self._spin_tick = 0
        self._flash_msg: tuple[str, float] | None = None
        self._stream_renderer = VisualStreamRenderer(
            self.comp,
            self.tracker,
            mode_fn=lambda: self.rt.trace_mode,
            on_dirty=self.comp.render,
        )
        self.comp.set_dock_builder(self._build_dock)

        # Wire visual approval & `ask_user` menus onto `SessionRunner`!
        self.rt.runner.approval_callback = self._request_tool_approval_menu
        self.rt.runner.ask_user_callback = self._request_ask_user_menu

    def flash(self, msg: str, duration: float = 4.5) -> None:
        self._flash_msg = (msg, time.monotonic() + duration)
        self.comp.render()

    # ---- Interactive Approval & Ask-User Overlays (Never clobber `self.input_buf`!) ----

    def _request_tool_approval_menu(self, tool_name: str, args: dict[str, Any], reason: str) -> bool:
        if self.rt.runner.mode == PermissionMode.YOLO:
            return True
        done = threading.Event()
        decision = [False]
        target = str(args.get("command") or args.get("path") or args)[:72]

        def _pick(item: MenuItem) -> None:
            if item.key == "allow":
                decision[0] = True
            elif item.key == "auto_edit":
                self.rt.runner.mode = PermissionMode.AUTO_EDIT
                decision[0] = True
                self.flash("Mode -> auto_edit (workspace edits auto-approved)")
            elif item.key == "turbo":
                self.rt.runner.mode = PermissionMode.YOLO
                self.rt.max_steps = max(self.rt.max_steps, 160)
                decision[0] = True
                self.flash("▸▸ Turbo (autonomous) enabled for session")
            else:
                decision[0] = False
            self.menu = None
            done.set()
            self.comp.render()

        def _cancel() -> None:
            decision[0] = False
            self.menu = None
            done.set()
            self.comp.render()

        with self.comp.lock:
            self.menu = MenuOverlay(
                kind="approval",
                title=f"? Approve `{tool_name}`: {target}  ({reason[:48]})",
                items=[
                    MenuItem("allow", "Allow once", "Execute this tool call and continue", badge="y"),
                    MenuItem("auto_edit", "Allow & enable auto-edit", "Auto-apply workspace edits this session", badge="e"),
                    MenuItem("turbo", "Allow & enable Turbo", "Autonomous mode · security firewall stays active", badge="t"),
                    MenuItem("deny", "Deny tool call", "Block and let agent adapt or wait for steer", badge="n"),
                ],
                allow_filter=False,
                on_select=_pick,
                on_cancel=_cancel,
            )
        self.comp.render()
        while not done.wait(timeout=0.1):
            if self.rt.runner.cancel_event.is_set() or self.should_exit:
                self.menu = None
                return False
        return decision[0]

    def _request_ask_user_menu(self, question: str, options: list[str]) -> str:
        done = threading.Event()
        answer = [""]

        items = [MenuItem(opt, opt, "Select option") for opt in options]
        items.append(MenuItem("__custom__", "Type custom response...", "Enter free-form answer in composer"))

        def _pick(item: MenuItem) -> None:
            self.menu = None
            if item.key == "__custom__":
                saved_draft = (self.input_buf, self.cursor)
                self.input_buf = ""
                self.cursor = 0
                self._custom_input_label = f"Answer `{question[:54]}`"

                def _finish_custom(val: str) -> None:
                    answer[0] = val
                    self._custom_input_label = ""
                    self.input_buf, self.cursor = saved_draft
                    done.set()
                    self.comp.render()

                self._custom_input_cb = _finish_custom
                self.comp.render()
                return
            answer[0] = item.key
            done.set()
            self.comp.render()

        def _cancel() -> None:
            self.menu = None
            done.set()
            self.comp.render()

        with self.comp.lock:
            self.menu = MenuOverlay(
                kind="ask_user",
                title=f"? Agent asks: {question}",
                items=items,
                allow_filter=False,
                on_select=_pick,
                on_cancel=_cancel,
            )
        self.comp.render()
        while not done.wait(timeout=0.1):
            if self.rt.runner.cancel_event.is_set() or self.should_exit:
                self.menu = None
                return ""
        return answer[0]

    # ---- Auto-Completion Popup Menus (`/`, `@`, `$`) ----

    def _refresh_auto_menu(self) -> None:
        if self._custom_input_cb is not None:
            return
        if self.menu and self.menu.kind in ("picker", "approval", "ask_user", "threads"):
            return

        before = self.input_buf[: self.cursor]

        # 1. Leading `/` slash command palette (when typing command token before first space)
        if self.input_buf.startswith("/") and "\n" not in self.input_buf:
            first_space = self.input_buf.find(" ")
            if first_space == -1 and self.cursor <= len(self.input_buf):
                query = self.input_buf[1:]
                items: list[MenuItem] = []
                busy = self.turn_busy.is_set()
                for sc in self.rt.slash_registry.list_commands():
                    badge = "live" if sc.available_during_task else sc.kind.value
                    if busy and not sc.available_during_task:
                        badge = "queues"
                    items.append(
                        MenuItem(
                            key=sc.name,
                            label=f"/{sc.name}",
                            detail=sc.description,
                            badge=badge,
                            payload=sc,
                        )
                    )
                prev_sel = self.menu.selected if (self.menu and self.menu.kind == "auto_slash" and self.menu.filter_query == query) else 0
                self.menu = MenuOverlay(
                    kind="auto_slash",
                    title="Slash Commands  (↑↓ navigate · Tab complete · Enter run/open · Esc close)",
                    items=items,
                    selected=prev_sel,
                    filter_query=query,
                    max_visible=8,
                    replace_span=(0, len(self.input_buf)),
                )
                self.menu.clamp_selection()
                return

        # 2. `@file` workspace picker or `$skill` picker on current token
        tok_match = re.search(r"(?:^|\s)([@$])([^\s]*)$", before)
        if tok_match:
            trig = tok_match.group(1)
            frag = tok_match.group(2)
            span_start = tok_match.start(1)
            span_end = self.cursor
            if trig == "@":
                files = self.file_index.list_files()
                items = [MenuItem(f, f"@{f}", "Attach workspace file") for f in files]
                self.menu = MenuOverlay(
                    kind="auto_file",
                    title="Attach @file  (↑↓ navigate · Tab/Enter insert · Esc close)",
                    items=items,
                    filter_query=frag,
                    max_visible=7,
                    replace_span=(span_start, span_end),
                )
                self.menu.clamp_selection()
                return
            if trig == "$":
                skills = self.rt.skill_manager.user_invocable_skills()
                items = [
                    MenuItem(sk.name, f"${sk.name}", sk.description[:80], badge=sk.tier)
                    for sk in skills
                ]
                self.menu = MenuOverlay(
                    kind="auto_skill",
                    title="Activate $skill  (↑↓ navigate · Tab/Enter insert · Esc close)",
                    items=items,
                    filter_query=frag,
                    max_visible=7,
                    replace_span=(span_start, span_end),
                )
                self.menu.clamp_selection()
                return

        if self.menu and self.menu.kind in ("auto_slash", "auto_file", "auto_skill"):
            self.menu = None

    # ---- Rich Interactive Modal Pickers (`/model`, `/thinking`, `/turbo`, `/sessions`, `/threads`, etc.) ----

    def open_model_menu(self) -> None:
        cur_m = self.rt.runner.model_profile.model_id
        cur_p = self.rt.runner.auth_profile.name
        items: list[MenuItem] = []
        seen: set[str] = set()

        # Include defaults from discovered auth profiles first (e.g. local Ollama models if running!)
        for prof in self.rt.vault.list_profiles():
            if prof.default_model and prof.default_model not in seen:
                act = prof.default_model == cur_m and prof.name == cur_p
                items.append(
                    MenuItem(
                        key=f"{prof.name}::{prof.default_model}",
                        label=prof.default_model,
                        detail=f"via profile `{prof.name}` ({prof.provider})",
                        badge="● active" if act else prof.provider,
                        active=act,
                        payload=(prof, prof.default_model),
                    )
                )
                seen.add(prof.default_model)

        for mid, prov_lbl, desc in RECOMMENDED_MODELS:
            if mid in seen and mid != "__custom__":
                continue
            act = mid == cur_m
            items.append(
                MenuItem(
                    key=mid,
                    label="+ Custom model ID..." if mid == "__custom__" else mid,
                    detail=desc,
                    badge="● active" if act else prov_lbl,
                    active=act,
                    payload=(None, mid),
                )
            )

        sel_idx = next((i for i, it in enumerate(items) if it.active), 0)

        def _pick(it: MenuItem) -> None:
            prof_obj, mid = it.payload
            self.menu = None
            if mid == "__custom__":
                saved = (self.input_buf, self.cursor)
                self.input_buf = ""
                self.cursor = 0
                self._custom_input_label = "Enter model ID (e.g. gemini-4-argon, openrouter:deepseek/deepseek-r1)"

                def _apply_custom(val: str) -> None:
                    self._custom_input_label = ""
                    self.input_buf, self.cursor = saved
                    if val.strip():
                        np = self.rt.runner.switch_model(val.strip())
                        self.flash(f"Model -> {np.model_id} (Tier {np.tier.value}, {np.family})")

                self._custom_input_cb = _apply_custom
                self.comp.render()
                return
            if prof_obj is not None and prof_obj.name != self.rt.runner.auth_profile.name:
                self.rt.vault.switch_profile(prof_obj.name)
                np = self.rt.runner.switch_model(mid, prof_obj)
            else:
                np = self.rt.runner.switch_model(mid)
            self.flash(f"Switched model -> {np.model_id} (Tier {np.tier.value} · {np.provider})")

        self.menu = MenuOverlay(
            kind="picker",
            title="Select Model  (↑↓ navigate · type to filter · Enter switch · Esc cancel)",
            items=items,
            selected=sel_idx,
            max_visible=9,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def open_thinking_menu(self) -> None:
        cur = self.rt.runner.thinking_effort
        specs = [
            (ThinkingEffort.MAX, "Maximum depth (Gemini HIGH / Claude 32k interleaved / OpenAI high + 200 steps)"),
            (ThinkingEffort.HIGH, "Deep multi-step reasoning (default — native CoT streamed on every step)"),
            (ThinkingEffort.MEDIUM, "Balanced thinking budget (Gemini MEDIUM / 8k thought budget)"),
            (ThinkingEffort.LOW, "Fast brief reasoning (Gemini LOW / 2k thought budget)"),
            (ThinkingEffort.OFF, "Disable extended thinking where model permits (lowest latency)"),
        ]
        items = [
            MenuItem(eff.value, eff.value.upper(), desc, badge="● active" if eff == cur else "", active=(eff == cur), payload=eff)
            for eff, desc in specs
        ]
        sel = next((i for i, it in enumerate(items) if it.active), 1)

        def _pick(it: MenuItem) -> None:
            eff: ThinkingEffort = it.payload
            self.rt.runner.set_thinking_effort(eff)
            self.menu = None
            self.flash(f"◈ Thinking effort -> {eff.value} (takes effect next model step)")

        self.menu = MenuOverlay(
            kind="picker",
            title="◈ Thinking Depth  (↑↓ navigate · Enter select · Ctrl+T cycles · Esc close)",
            items=items,
            selected=sel,
            allow_filter=False,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def _format_mode_label(self) -> str:
        mode = self.rt.runner.mode
        eff = self.rt.runner.thinking_effort
        if mode == PermissionMode.YOLO:
            if eff == ThinkingEffort.MAX:
                return "turbo:max"
            if eff == ThinkingEffort.LOW:
                return "turbo:fast"
            return "turbo"
        if mode == PermissionMode.AUTO_EDIT:
            return "auto-edit"
        if mode == PermissionMode.PLAN:
            return "plan"
        return "supervised"

    def open_turbo_permissions_menu(self) -> None:
        cur = self.rt.runner.mode
        eff = self.rt.runner.thinking_effort
        specs = [
            ("turbo_on", "▸▸ Turbo (Autonomous)", "Hands-free execution · 160 steps/turn · security firewall active", cur == PermissionMode.YOLO and eff not in (ThinkingEffort.MAX, ThinkingEffort.LOW)),
            ("turbo_max", "✦  Turbo Max (Deep)", "Autonomous + max reasoning depth · 200 steps/turn", cur == PermissionMode.YOLO and eff == ThinkingEffort.MAX),
            ("turbo_fast", "»  Turbo Fast (Rapid)", "Autonomous + low-latency reasoning (+ Flash on Gemini)", cur == PermissionMode.YOLO and eff == ThinkingEffort.LOW),
            ("auto_edit", "●  Auto-Edit", "Auto-apply workspace edits · confirm risky shell commands", cur == PermissionMode.AUTO_EDIT),
            ("default", "◐  Supervised", "Parallel read tools free · confirm edits & shell commands", cur == PermissionMode.DEFAULT),
            ("plan", "◇  Plan (Read-Only)", "Architecture & inspection only · zero mutations", cur == PermissionMode.PLAN),
        ]
        items = [
            MenuItem(k, lbl, desc, badge="● active" if act else "", active=act)
            for k, lbl, desc, act in specs
        ]
        sel = next((i for i, it in enumerate(items) if it.active), 0)

        def _pick(it: MenuItem) -> None:
            self.menu = None
            if it.key == "turbo_on":
                self._run_slash_captured("/turbo on", echo=False)
            elif it.key == "turbo_max":
                self._run_slash_captured("/turbo max", echo=False)
            elif it.key == "turbo_fast":
                self._run_slash_captured("/turbo fast", echo=False)
            else:
                self.rt.runner.mode = PermissionMode(it.key)
            self.flash(
                f"Mode -> {self._format_mode_label()} · ◈ {self.rt.runner.thinking_effort.value} · {self.rt.max_steps} steps"
            )

        self.menu = MenuOverlay(
            kind="picker",
            title="Execution Mode  (↑↓ navigate · Enter select · Shift+Tab cycles · Esc close)",
            items=items,
            selected=sel,
            allow_filter=False,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def open_trace_menu(self) -> None:
        cur = self.rt.trace_mode
        items = [
            MenuItem("threads", "◈ Open Thinking Threads Explorer (Ctrl+O)", "Navigate, search & scroll every thought thread interactively", badge="^O"),
            MenuItem("full", "Stream: FULL", "Stream verbatim chains of thought + tool cards live into scrollback", badge="● active" if cur == TraceMode.FULL else ""),
            MenuItem("compact", "Stream: COMPACT", "Show 1-line thought landmarks + live status ticker (full thoughts still saved)", badge="● active" if cur == TraceMode.COMPACT else ""),
            MenuItem("off", "Stream: OFF", "Show assistant replies only (100% of thoughts still saved in .trace.jsonl)", badge="● active" if cur == TraceMode.OFF else ""),
            MenuItem("export", "Export Session Thoughts -> Markdown", "Write full uncompacted research notebook (.trace.md)", badge=".md"),
            MenuItem("stats", "Show Reasoning Token & Timing Stats", "Print summary of session thought blocks & KV cache hits", badge="KPI"),
        ]

        def _pick(it: MenuItem) -> None:
            self.menu = None
            if it.key == "threads":
                self.open_threads_explorer()
                return
            if it.key in ("full", "compact", "off"):
                self.rt.trace_mode = TraceMode(it.key)
                self.flash(f"Trace stream -> {it.key} (.trace.jsonl still records 100% verbatim thoughts)")
                return
            if it.key == "export" and self.rt.runner.trace_ledger:
                out_f = self.rt.runner.trace_ledger.export_markdown()
                self.comp.println(self.s.green(f"✓ Exported uncompacted thinking trace -> {out_f}"))
                return
            if it.key == "stats" and self.rt.runner.trace_ledger:
                self.comp.println(self.rt.runner.trace_ledger.stats_summary())

        self.menu = MenuOverlay(
            kind="picker",
            title="◈ Thought Tracing & Threads  (↑↓ navigate · Enter select · Esc close)",
            items=items,
            allow_filter=False,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def open_sessions_menu(self) -> None:
        # Pull ALL recent non-empty sessions across every workspace, strictly sorted most-recent-first!
        slist = list_workspace_sessions(self.rt.ush_home / "sessions", workspace_filter=None, include_empty=False)
        if not slist:
            self.flash("No saved sessions found yet.")
            return
        cur_id = self.rt.runner.session_id
        cur_ws = str(self.rt.workspace)
        items: list[MenuItem] = []
        for s in slist[:80]:
            age = format_relative_age(s.updated_at)
            ws_tag = ""
            if s.workspace:
                ws_name = pathlib.Path(s.workspace).name or s.workspace
                ws_tag = f"{ws_name} · " if s.workspace != cur_ws else "here · "
            mdl = s.models[-1] if s.models else ""
            meta_badge = "● current" if s.session_id == cur_id else (f"{s.turns}t · {mdl}" if mdl else f"{s.turns}t")
            prompt_prev = s.last_prompt[:64] or "(no prompt text)"
            items.append(
                MenuItem(
                    key=s.session_id,
                    label=f"{age:<8s} {s.session_id}",
                    detail=f"{ws_tag}{prompt_prev}",
                    badge=meta_badge,
                    active=(s.session_id == cur_id),
                    payload=s,
                )
            )

        # Default cursor to index 0 (the single most recent session!) so Enter resumes immediately
        def _pick(it: MenuItem) -> None:
            self.menu = None
            if self.turn_busy.is_set():
                self.flash("Finish or Ctrl+C current turn before switching sessions.")
                return
            self._run_slash_captured(f"/resume {it.key}", echo=False)
            self._rebind_runner_and_threads()
            n_msg = len(self.rt.runner.dag.active_messages())
            self.flash(
                f"Resumed {it.key} ({n_msg} msgs · {len(self.tracker.threads)} ◈ threads · ^O inspect)"
            )

        self.menu = MenuOverlay(
            kind="picker",
            title=f"Recent Sessions ({len(items)} sorted newest-first · ↑↓ navigate · type search · Enter resume)",
            items=items,
            selected=0,
            max_visible=9,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def open_auth_menu(self) -> None:
        cur = self.rt.runner.auth_profile.name
        items = [
            MenuItem(
                key=p.name,
                label=p.name,
                detail=f"{p.provider} · model={p.default_model or '-'} · source={p.source}",
                badge="● active" if p.name == cur else p.auth_kind,
                active=(p.name == cur),
            )
            for p in self.rt.vault.list_profiles()
        ]
        if not items:
            self.flash("No auth profiles found — run `ush auth login`")
            return
        sel = next((i for i, it in enumerate(items) if it.active), 0)

        def _pick(it: MenuItem) -> None:
            self.menu = None
            prof = self.rt.vault.switch_profile(it.key)
            self.rt.runner.switch_model(prof.default_model or self.rt.runner.model_profile.model_id, prof)
            self.flash(f"Switched auth -> {prof.name} ({prof.provider} · {self.rt.runner.model_profile.model_id})")

        self.menu = MenuOverlay(
            kind="picker",
            title="Switch Auth Profile  (↑↓ navigate · Enter activate · Esc close)",
            items=items,
            selected=sel,
            allow_filter=False,
            on_select=_pick,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    def open_threads_explorer(self, initial_filter: str = "") -> None:
        nodes = self.tracker.list_threads()
        items = [
            MenuItem(
                key=n.thread_id,
                label=f"{n.thread_id} · {n.headline[:48]}",
                detail=f"{n.model} · {n.live_tokens_est():,}t · {len(n.tools)} tools",
                badge=n.status,
                payload=n,
            )
            for n in reversed(nodes)  # Newest live thread right at top!
        ]
        self.menu = MenuOverlay(
            kind="threads",
            title="◈ Thinking Threads  (↑↓ thread · ←/→ scroll · type search · ^E export · Esc)",
            items=items,
            selected=0,
            filter_query=initial_filter,
            max_visible=5,
            expanded_detail=True,
            on_cancel=lambda: setattr(self, "menu", None),
        )

    # ---- Dock Layout Builder (`INV-TUI-1`: every returned line is clipped smoothly) ----

    def _refresh_threads_items_live(self, m: MenuOverlay) -> None:
        cur_key = m.items[m.selected].key if (m.items and 0 <= m.selected < len(m.items)) else None
        nodes = list(reversed(self.tracker.list_threads()))
        m.items = [
            MenuItem(
                key=n.thread_id,
                label=f"{n.thread_id} · {n.headline[:46]}",
                detail=f"{n.model} · {n.live_tokens_est():,}t · {len(n.tools)} tools",
                badge="● live" if n.status in ("thinking", "tools") else f"{n.thought_ms/1000.0:.1f}s",
                payload=n,
            )
            for n in nodes
        ]
        if cur_key:
            for idx, it in enumerate(m.items):
                if it.key == cur_key:
                    m.selected = idx
                    break

    def _render_menu_lines(self, m: MenuOverlay, width: int) -> list[str]:
        inner = max(30, width - 5)
        lines: list[str] = []

        if m.kind == "threads":
            self._refresh_threads_items_live(m)

        vis = m.clamp_selection()
        q_lbl = f" · filter: `{m.filter_query}`" if (m.allow_filter and m.filter_query and m.kind not in ("auto_slash", "auto_file", "auto_skill")) else ""
        cnt_lbl = f" [{m.selected + 1}/{len(vis)}]" if vis else " [0 matches]"
        hdr_txt = clip_to_cols(f" {m.title}{q_lbl}{cnt_lbl} ", max(10, width - 5))
        bar_fill = max(0, (width - 4) - vis_width(hdr_txt))
        lines.append(self.s.cyan("╭─") + self.s.bold_cyan(hdr_txt) + self.s.cyan("─" * bar_fill))

        if not vis:
            lines.append(self.s.cyan("│ ") + self.s.dim("(no matching items — Backspace to widen filter or Esc to close)"))
        else:
            half = m.max_visible // 2
            start = max(0, min(m.selected - half, max(0, len(vis) - m.max_visible)))
            end = min(len(vis), start + m.max_visible)
            if start > 0:
                lines.append(self.s.cyan("│ ") + self.s.dim(f"  ↑ {start} above..."))
            for idx in range(start, end):
                it = vis[idx]
                is_sel = idx == m.selected
                ptr = "❯ " if is_sel else "  "
                badge = f" [{it.badge}]" if it.badge else ""
                lbl_w = min(34, max(16, inner // 3))
                left = pad_to_cols(f"{ptr}{it.label}", lbl_w)
                right_budget = max(8, inner - vis_width(left) - vis_width(badge) - 2)
                det = clip_to_cols(it.detail, right_budget)
                row_plain = f"{left}  {det}{badge}"
                row_padded = pad_to_cols(row_plain, inner)
                if is_sel:
                    lines.append(self.s.cyan("│") + self.s.sel_row(" " + row_padded))
                else:
                    styled_row = (
                        f" {self.s.bold(pad_to_cols(ptr + it.label, lbl_w))}  "
                        f"{self.s.dim(det)}{self.s.cyan(badge) if it.active else self.s.dim(badge)}"
                    )
                    lines.append(self.s.cyan("│") + clip_to_cols(styled_row, width - 2))
            if end < len(vis):
                lines.append(self.s.cyan("│ ") + self.s.dim(f"  ↓ {len(vis) - end} more..."))

            # For `Ctrl+O` Thinking Threads Explorer: live split-pane preview of selected thread's verbatim CoT & tools!
            if m.kind == "threads" and m.expanded_detail:
                cur_it = vis[m.selected]
                node: ThinkingThreadNode = cur_it.payload
                lines.append(self.s.cyan("├─ ") + self.s.bold(f"Thread {node.thread_id} ({node.model} · effort={node.effort} · {node.live_tokens_est():,} thought tok)"))
                body_lines: list[str] = []
                raw_t = node.thought_text.strip() or "(no explicit thought text emitted on this step)"
                wrap_w = max(24, width - 6)
                for para in raw_t.splitlines():
                    if not para.strip():
                        body_lines.append("")
                        continue
                    rem = para
                    while vis_width(rem) > wrap_w:
                        chunk = clip_to_cols(rem, wrap_w, ellipsis="")
                        body_lines.append(chunk)
                        rem = rem[len(strip_ansi(chunk)) :]
                        if not chunk:
                            break
                    if rem:
                        body_lines.append(rem)
                if node.tools:
                    body_lines.append("── Tool Calls in This Step ──")
                    for tr in node.tools:
                        st = "…" if tr.ok is None else ("✓" if tr.ok else "✗")
                        body_lines.append(f"  {st} {tr.name}({tr.arg_summary}) [{tr.elapsed_ms:.0f}ms] {tr.result_summary}")
                for cn in node.cognitive_notes:
                    body_lines.append(f"  ◈ {cn}")

                win_h = 8
                max_scr = max(0, len(body_lines) - win_h)
                m.body_scroll = max(0, min(m.body_scroll, max_scr))
                sl = body_lines[m.body_scroll : m.body_scroll + win_h]
                for bl in sl:
                    lines.append(self.s.cyan("│   ") + self.s.dim(clip_to_cols(bl, width - 6)))
                if len(body_lines) > win_h:
                    lines.append(
                        self.s.cyan("│   ")
                        + self.s.dim(f"[lines {m.body_scroll + 1}-{min(len(body_lines), m.body_scroll + win_h)} of {len(body_lines)} · ←/→ or PgUp/PgDn scroll]")
                    )

        lines.append(self.s.cyan("╰" + "─" * max(1, width - 3)))
        return lines

    def _build_dock(self, width: int) -> tuple[list[str], int, int]:
        lines: list[str] = []
        r = self.rt.runner

        # 1. Live Active Thinking Thread Ticker (visible whenever a turn is running!)
        live_status = self.tracker.snapshot_live_status(self._spin_tick)
        if live_status:
            lines.append(self.s.bold_cyan("  ") + self.s.cyan(clip_to_cols(live_status, width - 4)))
        elif self.turn_busy.is_set():
            spin = SPINNER_FRAMES[self._spin_tick % len(SPINNER_FRAMES)]
            lines.append(self.s.cyan(f"  {spin} Running turn on {r.model_profile.model_id}... (Ctrl+C cancel · Enter queues next · Alt+S steers live)"))

        # 2. Queued Follow-Up Prompts & Pending Mid-Turn Steers Strip!
        with self._q_lock:
            q_items = [q[0] for q in self.next_queue]
        steer_n = r.pending_steer_count()
        if q_items or steer_n:
            pills: list[str] = []
            if steer_n:
                pills.append(self.s.yellow(f"» {steer_n} live steer pending"))
            if q_items:
                prev = " · ".join(f"#{idx+1} {q.replace(chr(10), ' ')[:34]!r}" for idx, q in enumerate(q_items[:3]))
                more = f" +{len(q_items) - 3}" if len(q_items) > 3 else ""
                pills.append(self.s.green(f"↳ queue ({len(q_items)}): {prev}{more}") + self.s.dim("  [Alt+↑ edit · /steer now]"))
            lines.append("  " + clip_to_cols("  │  ".join(pills), width - 4))

        # 3. Transient notification flash
        if self._flash_msg is not None:
            msg, exp = self._flash_msg
            if time.monotonic() < exp:
                lines.append("  " + self.s.green("✓ " + clip_to_cols(msg, width - 6)))
            else:
                self._flash_msg = None

        # 4. Active Navigable Menu / Thinking Threads Explorer (renders neatly right above composer!)
        if self.menu is not None:
            lines.extend(self._render_menu_lines(self.menu, width))

        # 5. Always-Live Bordered Prompt Composer Box
        top_lbl = ""
        if self._custom_input_label:
            top_lbl = f" {self._custom_input_label} (Enter confirm · Esc cancel) "
        elif self.input_buf.startswith("/steer "):
            top_lbl = " » Live Steer: type guidance + Enter to inject into active turn "
        elif self.turn_busy.is_set():
            top_lbl = " Type next prompt (Enter queues · Alt+S steers live · / menu · ^O threads) "
        top_lbl_c = clip_to_cols(top_lbl, max(0, width - 6))
        top_dashes = max(0, (width - 4) - vis_width(top_lbl_c))
        lines.append(self.s.dim("╭─") + (self.s.yellow(top_lbl_c) if self._custom_input_label else self.s.dim(top_lbl_c)) + self.s.dim("─" * top_dashes))

        prompt_pfx = self.s.bold_cyan("❯ ") if not self.turn_busy.is_set() else self.s.green("↳ ")
        pfx_w = 2  # `❯ ` / `↳ `
        usable_w = max(16, width - pfx_w - 5)

        buf_lines = self.input_buf.split("\n")
        # Map flat `self.cursor` -> `(logical_line_idx, char_col_in_line)`
        rem_c = self.cursor
        cur_li = 0
        cur_ci = 0
        for li, bl in enumerate(buf_lines):
            if rem_c <= len(bl):
                cur_li = li
                cur_ci = rem_c
                break
            rem_c -= len(bl) + 1
        else:
            cur_li = len(buf_lines) - 1
            cur_ci = len(buf_lines[-1])

        max_box_lines = 5
        start_li = max(0, min(cur_li - (max_box_lines // 2), max(0, len(buf_lines) - max_box_lines)))
        end_li = min(len(buf_lines), start_li + max_box_lines)

        cursor_dock_row = len(lines)
        cursor_dock_col = 3 + pfx_w

        placeholder = ""
        if not self.input_buf and self.menu is None and not self._custom_input_label:
            placeholder = (
                "Prompt next task while model thinks…"
                if self.turn_busy.is_set()
                else "Ask anything · `/` menus · `@` files · `$` skills · `Alt+Enter` newline"
            )

        for li in range(start_li, end_li):
            bl = buf_lines[li]
            lead = prompt_pfx if li == 0 else self.s.dim("· ")
            if not self.input_buf and li == 0 and placeholder:
                row_str = self.s.dim("│ ") + lead + self.s.dim(clip_to_cols(placeholder, usable_w))
                cursor_dock_row = len(lines)
                cursor_dock_col = 3 + vis_width(lead)
                lines.append(row_str)
                continue

            # Horizontal viewport window so cursor stays comfortably visible on very long lines
            before_cur = bl[:cur_ci] if li == cur_li else bl
            h_shift = 0
            while li == cur_li and vis_width(bl[h_shift:cur_ci]) > usable_w - 2 and h_shift < cur_ci:
                h_shift += 1
            visible_slice = clip_to_cols(bl[h_shift:], usable_w, ellipsis="…")
            row_idx = len(lines)
            lines.append(self.s.dim("│ ") + lead + visible_slice)
            if li == cur_li:
                cursor_dock_row = row_idx
                cursor_dock_col = 3 + vis_width(lead) + vis_width(before_cur[h_shift:])

        extra_note = f" (+{len(buf_lines) - max_box_lines} lines)" if len(buf_lines) > max_box_lines else ""
        bot_note = clip_to_cols(extra_note, width - 6)
        lines.append(self.s.dim("╰" + "─" * max(1, (width - 3) - vis_width(bot_note)) + bot_note))

        # 6. Status & Visual Keybinding Bar (clean single-cell geometric glyphs — zero cartoon emojis)
        ctx_pct = 100.0 * r.dag.approx_context_tokens() / max(1, r.model_profile.context_window_tokens)
        m_lbl = self._format_mode_label()
        if r.mode == PermissionMode.YOLO:
            mode_styled = self.s.bold_cyan(f"▸▸ {m_lbl}") if r.thinking_effort != ThinkingEffort.MAX else self.s.magenta(f"✦ {m_lbl}")
        elif r.mode == PermissionMode.AUTO_EDIT:
            mode_styled = self.s.green(f"● {m_lbl}")
        elif r.mode == PermissionMode.PLAN:
            mode_styled = self.s.blue(f"◇ {m_lbl}")
        else:
            mode_styled = self.s.cyan(f"◐ {m_lbl}")

        think_glyph = "◇" if r.thinking_effort == ThinkingEffort.OFF else "◈"
        n_threads = len(self.tracker.threads)
        thought_tok = max(
            r.total_usage.reasoning_tokens,
            sum(t.live_tokens_est() for t in self.tracker.list_threads()),
        )
        stat = (
            f" {self.s.bold(r.model_profile.model_id)} · {self.s.cyan(f'{think_glyph} {r.thinking_effort.value}')}(^T) "
            f"· {mode_styled}(⇧Tab) · threads:{n_threads}(^O) "
            f"· {thought_tok:,} tok · ctx:{ctx_pct:.1f}% · `/` menu"
        )
        lines.append(clip_to_cols(stat, width - 2))

        return lines, cursor_dock_row, cursor_dock_col

    # ---- Background Turn Runner + Automatic Next-Prompt Queue Drain ----

    def submit_or_queue_prompt(
        self, prompt_text: str, extra_parts: list[CanonicalPart] | None = None, *, steer_now: bool = False
    ) -> None:
        text = prompt_text.strip()
        if not text:
            return
        pts = list(extra_parts or [])

        if self.turn_busy.is_set():
            if steer_now:
                self.rt.runner.enqueue_steer(text)
                cnt = self.rt.runner.pending_steer_count()
                self._stream_renderer.emit_external_note(f"Live mid-turn steer #{cnt}: {text[:88]}")
                return
            with self._q_lock:
                self.next_queue.append((text, pts))
                qlen = len(self.next_queue)
            self.flash(f"↳ Queued prompt #{qlen} (auto-runs after turn · Alt+S or `/steer` injects live)")
            return

        self._launch_turn_worker(text, pts)

    def _launch_turn_worker(self, first_prompt: str, first_parts: list[CanonicalPart]) -> None:
        self.turn_busy.set()
        self.comp.render()

        def _worker() -> None:
            cur_prompt: str | None = first_prompt
            cur_parts: list[CanonicalPart] = list(first_parts)
            try:
                while cur_prompt is not None and not self.should_exit:
                    preview = cur_prompt if len(cur_prompt) <= 240 else (cur_prompt[:237] + "...")
                    self.comp.write_scroll(f"\n{self.s.bold_cyan('❯')} {self.s.bold(preview)}\n")
                    self.comp.render()
                    try:
                        self.rt.runner.run_turn(
                            cur_prompt,
                            extra_parts=cur_parts,
                            max_steps=self.rt.max_steps,
                            on_event=self._stream_renderer,
                        )
                    except KeyboardInterrupt:
                        self.rt.runner.cancel_event.set()
                        self.comp.write_scroll(self.s.yellow("\n[Turn canceled cleanly]\n"))
                        break
                    except Exception as exc:  # noqa: BLE001
                        self.comp.write_scroll(self.s.red(f"\n[Error]: {redact_secrets(str(exc))}\n"))
                        break

                    if self.rt.runner.cancel_event.is_set():
                        # Pause auto-drain if user hit Ctrl+C so queued prompts never fire unexpectedly!
                        with self._q_lock:
                            paused = len(self.next_queue)
                        if paused:
                            self.flash(f"Turn stopped · {paused} queued prompt(s) paused (Alt+↑ to edit or /queue clear)")
                        break

                    with self._q_lock:
                        if self.next_queue:
                            cur_prompt, cur_parts = self.next_queue.popleft()
                        else:
                            cur_prompt = None
                            cur_parts = []
            finally:
                self.tracker.turn_Active = False
                self.turn_busy.clear()
                self.comp.render()

        threading.Thread(target=_worker, name="ush-visual-runner", daemon=True).start()

    # ---- Key Handling (Menus + Line Editor + Global Shortcuts) ----

    def _complete_auto_menu_item(self, m: MenuOverlay, item: MenuItem, *, execute_on_slash: bool) -> None:
        if m.kind == "auto_slash":
            interactive_pickers = {
                "model": self.open_model_menu,
                "thinking": self.open_thinking_menu,
                "turbo": self.open_turbo_permissions_menu,
                "permissions": self.open_turbo_permissions_menu,
                "trace": self.open_trace_menu,
                "sessions": self.open_sessions_menu,
                "resume": self.open_sessions_menu,
                "auth": self.open_auth_menu,
                "threads": self.open_threads_explorer,
            }
            if execute_on_slash and item.key in interactive_pickers:
                self.input_buf = ""
                self.cursor = 0
                self.menu = None
                interactive_pickers[item.key]()
                return
            # Argument-taking commands: Enter fills `/cmd ` in composer so user can type guidance seamlessly
            # (Exception: bare `/steer` with a queued prompt promotes that queued prompt immediately!)
            needs_arg = {"btw", "goal", "swarm", "arena", "learn", "steps"}
            if execute_on_slash and item.key == "steer":
                with self._q_lock:
                    can_promote = bool(self.next_queue) and self.turn_busy.is_set()
                if can_promote:
                    self.input_buf = ""
                    self.cursor = 0
                    self.menu = None
                    self._execute_submitted_line("/steer", steer_now=False)
                    return
                self.input_buf = "/steer "
                self.cursor = len(self.input_buf)
                self.menu = None
                return
            if execute_on_slash and item.key not in needs_arg:
                cmd_line = f"/{item.key}"
                self.input_buf = ""
                self.cursor = 0
                self.menu = None
                self._execute_submitted_line(cmd_line, steer_now=False)
                return
            self.input_buf = f"/{item.key} "
            self.cursor = len(self.input_buf)
            self.menu = None
            return

        if m.replace_span is not None:
            s0, s1 = m.replace_span
            prefix = "@" if m.kind == "auto_file" else "$"
            ins = f"{prefix}{item.key} "
            self.input_buf = self.input_buf[:s0] + ins + self.input_buf[s1:]
            self.cursor = s0 + len(ins)
            self.menu = None

    def _execute_submitted_line(self, raw: str, *, steer_now: bool) -> None:
        line = raw.strip()
        if not line:
            return
        if not self.history or self.history[-1] != raw:
            self.history.append(raw)
        self.hist_idx = len(self.history)

        # Open interactive visual pickers on bare `/model`, `/thinking`, `/turbo`, `/permissions`, `/trace`, `/sessions`, `/resume`, `/auth`, `/threads`
        if line.startswith("/") and " " not in line:
            bare = line[1:].lower()
            visual_dispatch = {
                "model": self.open_model_menu,
                "thinking": self.open_thinking_menu,
                "turbo": self.open_turbo_permissions_menu,
                "permissions": self.open_turbo_permissions_menu,
                "trace": self.open_trace_menu,
                "sessions": self.open_sessions_menu,
                "resume": self.open_sessions_menu,
                "auth": self.open_auth_menu,
                "threads": self.open_threads_explorer,
            }
            if bare in visual_dispatch:
                visual_dispatch[bare]()
                return

        if line.startswith("/threads"):
            q = line.split(None, 1)[1].strip() if " " in line else ""
            self.open_threads_explorer(q)
            return

        if line.startswith("/steer"):
            arg = line.split(None, 1)[1].strip() if " " in line else ""
            if not arg:
                # Promote oldest queued prompt into an immediate mid-turn steer (release `_q_lock` BEFORE rendering!)
                promoted: str | None = None
                promoted_parts: list[CanonicalPart] = []
                with self._q_lock:
                    if self.next_queue and self.turn_busy.is_set():
                        promoted, promoted_parts = self.next_queue.popleft()
                if promoted is not None:
                    self.submit_or_queue_prompt(promoted, promoted_parts, steer_now=True)
                    return
                # Keep `/steer ` ready in composer so typing guidance + Enter works smoothly
                self.input_buf = "/steer "
                self.cursor = len(self.input_buf)
                self.flash("Type guidance after `/steer ` + Enter (or press Alt+S anytime)")
                return
            from ush.cli.main import expand_user_prompt_pills

            try:
                final_steer, steer_parts = expand_user_prompt_pills(self.rt, arg)
            except Exception:  # noqa: BLE001
                final_steer, steer_parts = arg, []
            self.submit_or_queue_prompt(final_steer, steer_parts, steer_now=True)
            return

        if line.startswith("/queue"):
            arg = line.split(None, 1)[1].strip() if " " in line else "list"
            if arg == "clear":
                with self._q_lock:
                    n = len(self.next_queue)
                    self.next_queue.clear()
                self.flash(f"Cleared {n} queued prompt(s).")
                return
            if arg == "pop":
                popped_q: str | None = None
                with self._q_lock:
                    if self.next_queue:
                        popped_q, _ = self.next_queue.pop()
                if popped_q is not None:
                    self.input_buf = popped_q
                    self.cursor = len(popped_q)
                return
            if arg == "list":
                with self._q_lock:
                    snap_q = [q[0] for q in self.next_queue]
                if not snap_q:
                    self.flash("Prompt queue is empty.")
                else:
                    listing = "\n".join(f"  {i+1}. {q}" for i, q in enumerate(snap_q))
                    self.comp.println(self.s.bold("Queued Follow-Up Prompts:\n") + listing)
                return
            self.submit_or_queue_prompt(arg, steer_now=False)
            return

        if line.startswith("!"):
            sh_cmd = line[1:].strip()
            if not sh_cmd:
                return
            if self.turn_busy.is_set():
                self.flash("Shell `!cmd` waits for active turn PTY to finish (or queue a prompt).")
                return
            self.comp.write_scroll(f"\n{self.s.yellow('! ' + sh_cmd)}\n")
            res = self.rt.runner.pty.execute(sh_cmd, wait_seconds=30.0)
            if res.output:
                self.comp.write_scroll(res.output.rstrip("\n") + "\n")
            self.comp.println(self.s.dim(f"[exit={res.exit_code} · cwd={res.cwd}]"))
            return

        if line.startswith("/"):
            cname = line[1:].split(None, 1)[0].lower()
            cmd_spec = self.rt.slash_registry.get(cname)
            if self.turn_busy.is_set() and cmd_spec and not cmd_spec.available_during_task:
                # Queue prompt-producing slash commands (`/grill-me`, `/goal`, `$skill`) into next_queue!
                with self._q_lock:
                    self.next_queue.append((line, []))
                self.flash(f"Queued `{line[:40]}` to run after active turn (live commands work immediately)")
                return
            should_exit, prompt_or_none, extra_parts = self._run_slash_captured(line, echo=True)
            if cname in ("resume", "clear"):
                self._rebind_runner_and_threads()
            if should_exit:
                self.should_exit = True
                self.rt.runner.cancel_event.set()
                return
            if prompt_or_none:
                self.submit_or_queue_prompt(prompt_or_none, extra_parts, steer_now=steer_now)
            return

        # Expand inline `$skill-name` + `@file` pills via `expand_user_prompt_pills` if attached
        from ush.cli.main import expand_user_prompt_pills

        final_prompt, inline_parts = expand_user_prompt_pills(self.rt, line)
        self.submit_or_queue_prompt(final_prompt, inline_parts, steer_now=steer_now)

    def _rebind_runner_and_threads(self) -> None:
        self.rt.runner.approval_callback = self._request_tool_approval_menu
        self.rt.runner.ask_user_callback = self._request_ask_user_menu
        self.tracker = ThinkingThreadTracker()
        if self.rt.runner.trace_ledger is not None:
            self.tracker.load_historical_from_ledger(self.rt.runner.trace_ledger.read_records())
        self._stream_renderer.tracker = self.tracker

    def _run_slash_captured(
        self, line: str, *, echo: bool = True
    ) -> tuple[bool, str | None, list[CanonicalPart]]:
        import contextlib
        import io

        cap = io.StringIO()
        try:
            with contextlib.redirect_stdout(cap):
                res = self.slash_handler(self.rt, line)
        except Exception as exc:  # noqa: BLE001
            self.comp.println(self.s.red(f"Command error: {exc}"))
            return False, None, []
        printed = cap.getvalue()
        if echo and printed.strip():
            self.comp.println(printed.rstrip("\n"))
        return res

    def handle_key(self, ev: KeyEvent) -> None:
        k = ev.kind

        # 1. Global instant mode cyclers (`Shift+Tab` permissions/turbo, `Ctrl+T` thinking, `Ctrl+O` threads!)
        if k == "SHIFT_TAB":
            order = [PermissionMode.DEFAULT, PermissionMode.AUTO_EDIT, PermissionMode.YOLO, PermissionMode.PLAN]
            cur = self.rt.runner.mode
            nxt = order[(order.index(cur) + 1) % len(order)]
            self.rt.runner.mode = nxt
            if nxt == PermissionMode.YOLO:
                self.rt.max_steps = max(self.rt.max_steps, 160)
            self.flash(
                f"Mode -> {self._format_mode_label()}"
                + (" (hands-free · firewall enforced)" if nxt == PermissionMode.YOLO else "")
            )
            return

        if k == "CTRL_T":
            order_t = [ThinkingEffort.LOW, ThinkingEffort.MEDIUM, ThinkingEffort.HIGH, ThinkingEffort.MAX, ThinkingEffort.OFF]
            cur_t = self.rt.runner.thinking_effort
            nxt_t = order_t[(order_t.index(cur_t) + 1) % len(order_t)]
            self.rt.runner.set_thinking_effort(nxt_t)
            self.flash(f"◈ Thinking effort -> {nxt_t.value}")
            return

        if k == "CTRL_O":
            if self.menu and self.menu.kind == "threads":
                self.menu = None
            else:
                self.open_threads_explorer()
            self.comp.render()
            return

        if k == "CTRL_L":
            self.comp.render()
            return

        # 2. `Ctrl+C` & `Ctrl+D`
        if k == "CTRL_C":
            if self._custom_input_cb is not None:
                self._custom_input_cb("")
                return
            if self.menu is not None and self.menu.kind not in ("auto_slash", "auto_file", "auto_skill"):
                if self.menu.on_cancel:
                    self.menu.on_cancel()
                else:
                    self.menu = None
                self.comp.render()
                return
            if self.turn_busy.is_set():
                self.rt.runner.cancel_event.set()
                self.flash("Stopping active turn cleanly (preserving completed thoughts)...")
                return
            if self.input_buf:
                self.input_buf = ""
                self.cursor = 0
                self.menu = None
                self.comp.render()
                return
            self.flash("Press Ctrl+D or type `/quit` to exit `ush`")
            return

        if k == "CTRL_D":
            if not self.input_buf and not self.turn_busy.is_set():
                self.should_exit = True
            return

        # 3. Modal Menus (`picker`, `approval`, `ask_user`, `threads`) intercept navigation & filter keys
        if self.menu is not None and self.menu.kind in ("picker", "approval", "ask_user", "threads"):
            m = self.menu
            if k == "ESC":
                if m.on_cancel:
                    m.on_cancel()
                else:
                    self.menu = None
                self.comp.render()
                return
            if k == "UP":
                m.move(-1)
                self.comp.render()
                return
            if k == "DOWN" or k == "TAB":
                m.move(1)
                self.comp.render()
                return
            if m.kind == "threads" and k == "END":
                if self.rt.runner.trace_ledger:
                    out_f = self.rt.runner.trace_ledger.export_markdown()
                    self.flash(f"Exported thinking threads -> {out_f}")
                return
            if m.kind == "threads" and k in ("PGUP", "LEFT"):
                m.body_scroll = max(0, m.body_scroll - 6)
                self.comp.render()
                return
            if m.kind == "threads" and k in ("PGDN", "RIGHT"):
                m.body_scroll += 6
                self.comp.render()
                return
            if m.kind == "approval" and k == "CHAR" and ev.text.lower() in ("y", "e", "t", "n"):
                shortcut = {"y": 0, "e": 1, "t": 2, "n": 3}[ev.text.lower()]
                m.selected = shortcut
                cur_it = m.current()
                if cur_it and m.on_select:
                    m.on_select(cur_it)
                return
            if k == "ENTER":
                if m.kind == "threads":
                    m.expanded_detail = not m.expanded_detail
                    self.comp.render()
                    return
                cur_it = m.current()
                if cur_it and m.on_select:
                    m.on_select(cur_it)
                self.comp.render()
                return
            if m.allow_filter:
                if k == "BACKSPACE":
                    m.filter_query = m.filter_query[:-1]
                    m.selected = 0
                    self.comp.render()
                    return
                if k == "CTRL_U":
                    m.filter_query = ""
                    m.selected = 0
                    self.comp.render()
                    return
                if k == "CHAR":
                    m.filter_query += ev.text
                    m.selected = 0
                    self.comp.render()
                    return
            return

        # 4. Auto-popup completion menus (`auto_slash`, `auto_file`, `auto_skill`)
        if self.menu is not None and self.menu.kind in ("auto_slash", "auto_file", "auto_skill"):
            am = self.menu
            if k == "ESC":
                self.menu = None
                self.comp.render()
                return
            if k == "UP":
                am.move(-1)
                self.comp.render()
                return
            if k == "DOWN":
                am.move(1)
                self.comp.render()
                return
            if k == "TAB":
                cur_it = am.current()
                if cur_it:
                    self._complete_auto_menu_item(am, cur_it, execute_on_slash=False)
                    self._refresh_auto_menu()
                    self.comp.render()
                return
            if k == "ENTER":
                cur_it = am.current()
                if cur_it:
                    # In `auto_slash`, Enter executes exact command or opens its visual sub-menu!
                    self._complete_auto_menu_item(am, cur_it, execute_on_slash=(am.kind == "auto_slash"))
                    self._refresh_auto_menu()
                    self.comp.render()
                    return

        # 5. `Alt+Up` pops latest queued follow-up prompt back into composer for live editing!
        if k == "ALT_UP":
            popped_alt: str | None = None
            with self._q_lock:
                if self.next_queue:
                    popped_alt, _ = self.next_queue.pop()
            if popped_alt is not None:
                self.input_buf = popped_alt
                self.cursor = len(popped_alt)
                self.flash("Pulled queued prompt back into editor")
            return

        # 6. Composer Submit (`ENTER` queues/runs; `ALT_S` steers immediately mid-turn!)
        if k in ("ENTER", "ALT_S"):
            if self._custom_input_cb is not None:
                cb = self._custom_input_cb
                val = self.input_buf
                self._custom_input_cb = None
                cb(val)
                return
            # Support queued slash command typed directly with args (`handle_slash_command`)
            raw = self.input_buf
            self.input_buf = ""
            self.cursor = 0
            self.menu = None
            if raw.strip().startswith("/") and not raw.strip().startswith("/steer") and " " in raw.strip():
                self._execute_submitted_line(raw, steer_now=(k == "ALT_S"))
            else:
                self._execute_submitted_line(raw, steer_now=(k == "ALT_S"))
            self.comp.render()
            return

        if k == "ESC":
            if self._custom_input_cb is not None:
                cb = self._custom_input_cb
                self._custom_input_cb = None
                cb("")
            return

        if k == "ALT_ENTER":
            self.input_buf = self.input_buf[: self.cursor] + "\n" + self.input_buf[self.cursor :]
            self.cursor += 1
            self._refresh_auto_menu()
            self.comp.render()
            return

        # 7. Standard Composer Editing & History Navigation
        if k == "PASTE":
            self.input_buf = self.input_buf[: self.cursor] + ev.text + self.input_buf[self.cursor :]
            self.cursor += len(ev.text)
            self._refresh_auto_menu()
            self.comp.render()
            return

        if k == "CHAR":
            self.input_buf = self.input_buf[: self.cursor] + ev.text + self.input_buf[self.cursor :]
            self.cursor += len(ev.text)
            self._refresh_auto_menu()
            self.comp.render()
            return

        if k == "BACKSPACE":
            if self.cursor > 0:
                self.input_buf = self.input_buf[: self.cursor - 1] + self.input_buf[self.cursor :]
                self.cursor -= 1
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "DELETE":
            if self.cursor < len(self.input_buf):
                self.input_buf = self.input_buf[: self.cursor] + self.input_buf[self.cursor + 1 :]
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "LEFT":
            if self.cursor > 0:
                self.cursor -= 1
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "RIGHT":
            if self.cursor < len(self.input_buf):
                self.cursor += 1
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "HOME":
            nl = self.input_buf.rfind("\n", 0, self.cursor)
            self.cursor = 0 if nl == -1 else nl + 1
            self.comp.render()
            return

        if k == "END":
            nl = self.input_buf.find("\n", self.cursor)
            self.cursor = len(self.input_buf) if nl == -1 else nl
            self.comp.render()
            return

        if k == "CTRL_U":
            self.input_buf = ""
            self.cursor = 0
            self._refresh_auto_menu()
            self.comp.render()
            return

        if k == "CTRL_W":
            if self.cursor > 0:
                p = self.cursor
                while p > 0 and self.input_buf[p - 1].isspace():
                    p -= 1
                while p > 0 and not self.input_buf[p - 1].isspace():
                    p -= 1
                self.input_buf = self.input_buf[:p] + self.input_buf[self.cursor :]
                self.cursor = p
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "UP" and self.history:
            if "\n" in self.input_buf and self.input_buf.rfind("\n", 0, self.cursor) != -1:
                # Move cursor up inside multiline composer buffer
                cur_line_start = self.input_buf.rfind("\n", 0, self.cursor) + 1
                col = self.cursor - cur_line_start
                prev_end = cur_line_start - 1
                prev_start = self.input_buf.rfind("\n", 0, prev_end) + 1
                self.cursor = min(prev_end, prev_start + col)
                self.comp.render()
                return
            if self.hist_idx == len(self.history):
                self._draft_before_hist = self.input_buf
            if self.hist_idx > 0:
                self.hist_idx -= 1
                self.input_buf = self.history[self.hist_idx]
                self.cursor = len(self.input_buf)
                self._refresh_auto_menu()
                self.comp.render()
            return

        if k == "DOWN" and self.history:
            nxt_nl = self.input_buf.find("\n", self.cursor)
            if nxt_nl != -1:
                cur_line_start = self.input_buf.rfind("\n", 0, self.cursor) + 1
                col = self.cursor - cur_line_start
                next_end = self.input_buf.find("\n", nxt_nl + 1)
                if next_end == -1:
                    next_end = len(self.input_buf)
                self.cursor = min(next_end, nxt_nl + 1 + col)
                self.comp.render()
                return
            if self.hist_idx < len(self.history):
                self.hist_idx += 1
                self.input_buf = (
                    self._draft_before_hist if self.hist_idx == len(self.history) else self.history[self.hist_idx]
                )
                self.cursor = len(self.input_buf)
                self._refresh_auto_menu()
                self.comp.render()


def _load_plain_history(path: pathlib.Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        return [ln for ln in path.read_text(encoding="utf-8", errors="replace").splitlines()[-500:] if ln.strip()]
    except OSError:
        return []


def _save_plain_history(path: pathlib.Path, items: list[str]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for it in items[-500:]:
                if "\n" not in it:
                    fh.write(it + "\n")
        os.chmod(path, 0o600)
    except OSError:
        pass


def run_visual_tui(
    rt: HarnessRuntime,
    *,
    slash_handler: Callable[[HarnessRuntime, str], tuple[bool, str | None, list[CanonicalPart]]],
    use_color: bool = True,
    open_sessions_on_start: bool = False,
) -> int:
    """Runs the Jetski-style Visual Inline-Dock TUI on a real TTY with strict `termios` cleanup."""
    fd = sys.stdin.fileno()
    orig_attrs = termios.tcgetattr(fd)
    restored = [False]

    def _restore_tty() -> None:
        if not restored[0]:
            restored[0] = True
            try:
                sys.stdout.write("\x1b[?2004l\x1b[?25h")
                sys.stdout.flush()
                termios.tcsetattr(fd, termios.TCSADRAIN, orig_attrs)
            except Exception:  # noqa: BLE001
                pass

    atexit.register(_restore_tty)
    hist_file = rt.ush_home / "tui_history"
    ctrl = VisualTuiController(rt, slash_handler=slash_handler, use_color=use_color)
    ctrl.history = _load_plain_history(hist_file)
    ctrl.hist_idx = len(ctrl.history)

    prev_winch = signal.getsignal(signal.SIGWINCH)

    def _on_winch(_sig: int, _frm: Any) -> None:
        ctrl.comp.render()

    rt.hook_system.fire("SessionStart", {"workspace": str(rt.workspace)})

    try:
        tty.setcbreak(fd)
        # Keep `ISIG` off so Ctrl+C arrives cleanly as `\x03` into our raw decoder (zero torn redraws!)
        new_attrs = termios.tcgetattr(fd)
        new_attrs[3] &= ~(termios.ECHO | termios.ICANON | termios.ISIG)
        new_attrs[0] &= ~(termios.IXON | termios.ICRNL)
        termios.tcsetattr(fd, termios.TCSANOW, new_attrs)
        sys.stdout.write("\x1b[?2004h")  # Enable Bracketed Paste!
        sys.stdout.flush()
        signal.signal(signal.SIGWINCH, _on_winch)

        s = ctrl.s
        trace_p = rt.runner.trace_ledger.trace_path if rt.runner.trace_ledger else "-"
        ctrl.comp.println(
            f"{s.bold_cyan('╭─ crux')} {s.bold('v0.1')} {s.dim('(Crucible)')} · {s.dim(str(rt.workspace))}\n"
            f"{s.bold_cyan('│')}  session {s.cyan(rt.runner.session_id)} · trace {s.dim(str(trace_p))}\n"
            f"{s.bold_cyan('╰─')} {s.dim('`/` menus · `@` files · `$` skills · `/resume` recent · `^O` threads · `⇧Tab` turbo · `^T` thinking · `Enter` queue · `Alt+S` steer')}"
        )
        if open_sessions_on_start:
            ctrl.open_sessions_menu()
        ctrl.comp.render()

        pending_str = ""
        import codecs

        utf8_dec = codecs.getincrementaldecoder("utf-8")("replace")
        last_spin = time.monotonic()

        def _dispatch_safe(key_ev: KeyEvent) -> None:
            try:
                ctrl.handle_key(key_ev)
            except Exception as exc:  # noqa: BLE001
                ctrl.menu = None
                ctrl.comp.println(s.red(f"[UI error recovered]: {redact_secrets(str(exc))}"))

        while not ctrl.should_exit:
            try:
                ready, _, _ = select.select([fd], [], [], 0.08)
            except InterruptedError:
                continue

            now = time.monotonic()
            if ready:
                raw_bytes = os.read(fd, 4096)
                if not raw_bytes:
                    break
                pending_str += utf8_dec.decode(raw_bytes, final=False)
                # Peek briefly if buffer ends on incomplete `\x1b` / paste sequence
                has_more = False
                if pending_str.endswith("\x1b") or ("\x1b[200~" in pending_str and "\x1b[201~" not in pending_str):
                    try:
                        r2, _, _ = select.select([fd], [], [], 0.018)
                        if r2:
                            pending_str += utf8_dec.decode(os.read(fd, 4096), final=False)
                            r3, _, _ = select.select([fd], [], [], 0.004)
                            has_more = bool(r3)
                    except InterruptedError:
                        pass
                evs, pending_str = decode_key_buffer(pending_str, esc_is_complete=not has_more)
                for ev in evs:
                    _dispatch_safe(ev)
                    if ctrl.should_exit:
                        break
            elif pending_str:
                evs, pending_str = decode_key_buffer(pending_str, esc_is_complete=True)
                for ev in evs:
                    _dispatch_safe(ev)

            # Animate live thinking spinner & thread timer at ~10 FPS ONLY while turn is active!
            if (ctrl.turn_busy.is_set() or (ctrl._flash_msg and now >= ctrl._flash_msg[1])) and (now - last_spin) >= 0.10:
                last_spin = now
                ctrl._spin_tick += 1
                ctrl.comp.render()

        # Wait briefly if user exited right as turn cancelled
        ctrl.comp.clear_dock()
    finally:
        try:
            signal.signal(signal.SIGWINCH, prev_winch)
        except Exception:  # noqa: BLE001
            pass
        _restore_tty()
        _save_plain_history(hist_file, ctrl.history)
        rt.hook_system.fire("SessionEnd", {"session_id": rt.runner.session_id})
        rt.runner.close()
    return 0
