"""Sanctioned Agent Client Protocol (`agy acp`) Inference & Dynamic Model Discovery Bridge.

Embeds the official Antigravity CLI (`agy`) strictly as a zero-tool LLM completion and
runtime model-discovery pipe over documented ACP JSON-RPC 2.0 stdio, while Crucible (`crux`)
retains 100% ownership of the agent harness (system prompt, tool schemas, ReAct loop,
security firewall, permission policy engine, context compaction, and `.trace.jsonl` ledger).
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import hashlib
import json
import os
import pathlib
import re
import selectors
import shutil
import subprocess
import threading
import time
from typing import Any, Callable

from ush.brain.wire import DYNAMIC_TAG_RE, extract_fallback_tool_calls, parse_lenient_json_object
from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    ModelCapabilityProfile,
    StopReason,
    StreamEvent,
    ThinkingEffort,
    TokenUsage,
    canonical_json,
    new_id,
)

_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\r")
_EFFORT_SUFFIX_RE = re.compile(r"^(?P<base>.+)-(?P<effort>high|medium|low|xhigh|max)$", re.IGNORECASE)
_EFFORT_LABEL_RE = re.compile(r"\s*\((?:High|Medium|Low|XHigh|Max)\)\s*$", re.IGNORECASE)
_AGY_MALFORMED_BANNER_RE = re.compile(
    r"⚠\s*Your previous response contained an improperly formatted function call[\s\S]*?Retries remaining:\s*\d+\s*"
)
_AGY_MALFORMED_BANNER_STREAM_RE = re.compile(
    r"⚠\s*Your previous response contained an improperly formatted function call[\s\S]*?Retries remaining:\s*\d+[ \t]*\r?\n+"
)

# `agy acp` silently front-truncates any single `session/prompt` payload above ~195,000 chars (~54.8k tokens).
# Enforce a strict 135,000-char ceiling with tail-first preservation so the user's latest prompt is NEVER cut off.
MAX_ACP_PROMPT_CHARS = 135_000
_OLD_TOOL_OUTPUT_CAP_CHARS = 3_200
_RECENT_TOOL_OUTPUT_CAP_CHARS = 28_000
_DELTA_TOOL_OUTPUT_CAP_CHARS = 90_000

_CRUX_PIPE_AGENT_NAME = "crux-llm-pipe"
_CRUX_PIPE_AGENT_MD = """---
name: crux-llm-pipe
description: "Zero-tool LLM inference pipe for Crucible (crux) harness."
tools: []
excludeDefaultComponents: true
inheritCustomizations: false
inheritMcp: false
commandExecutionPolicy: "off"
hidden: true
---

# System Prompt

You are the core reasoning model inside the Crucible (`crux`) agent harness. Follow the system instructions and `<crux_tool_call>` protocol in the conversation accurately. NEVER invoke any host-injected function declaration (such as `default_api:manage_task` or `manage_task`); emit ONLY `<crux_tool_call>` XML blocks when calling tools. Always prioritize and answer the latest user request directly without repeating already-completed prior explanations.
"""


@dataclass(frozen=True)
class AgyModelOption:
    """Dynamically discovered model family & effort variants returned at runtime by `agy`."""

    model_id: str
    display_name: str
    variant_ids: tuple[str, ...] = ()


def resolve_agy_binary() -> str | None:
    """Locates the installed `agy` CLI binary using documented PATH / standard user bin paths."""
    if os.environ.get("USH_DISABLE_AGY_DISCOVERY") == "1":
        return None
    override = (os.environ.get("CRUX_AGY_BIN") or os.environ.get("USH_AGY_BIN") or "").strip()
    if override:
        cand = pathlib.Path(override).expanduser()
        return str(cand) if cand.is_file() and os.access(cand, os.X_OK) else shutil.which(override)
    found = shutil.which("agy")
    if found:
        return found
    local_bin = pathlib.Path("~/.local/bin/agy").expanduser()
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)
    return None


def parse_agy_models_cli_output(raw_output: str) -> list[AgyModelOption]:
    """Parses `agy models` stdout into clean base model options without effort clutter or provider tags."""
    cleaned = _ANSI_CSI_RE.sub("\n", raw_output)
    variants_by_base: dict[str, list[str]] = {}
    names_by_base: dict[str, str] = {}
    order: list[str] = []

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line or "Fetching available models" in line or line.lower().startswith(("usage:", "error:", "warning:")):
            continue
        parts = line.split(None, 1)
        raw_id = parts[0].strip()
        if not raw_id or not re.match(r"^[a-z0-9][a-z0-9._-]*$", raw_id, re.IGNORECASE):
            continue
        raw_name = parts[1].strip() if len(parts) > 1 else raw_id
        m = _EFFORT_SUFFIX_RE.match(raw_id)
        base_id = m.group("base") if m else raw_id
        base_name = _EFFORT_LABEL_RE.sub("", raw_name).strip() or base_id
        if base_id not in variants_by_base:
            variants_by_base[base_id] = []
            names_by_base[base_id] = base_name
            order.append(base_id)
        if raw_id not in variants_by_base[base_id]:
            variants_by_base[base_id].append(raw_id)

    return [
        AgyModelOption(
            model_id=bid,
            display_name=names_by_base[bid],
            variant_ids=tuple(variants_by_base[bid]),
        )
        for bid in order
    ]


def parse_acp_session_models(session_result: dict[str, Any]) -> tuple[str, list[AgyModelOption], set[str]]:
    """Extracts `(default_base_model, grouped_options, raw_variant_ids)` from ACP `session/new`."""
    models_obj = session_result.get("models") if isinstance(session_result.get("models"), dict) else {}
    avail_raw = models_obj.get("availableModels") if isinstance(models_obj, dict) else []
    cur_variant = str((models_obj or {}).get("currentModelId") or "").strip()

    raw_ids: list[str] = []
    variant_names: dict[str, str] = {}
    if isinstance(avail_raw, list):
        for item in avail_raw:
            if isinstance(item, dict) and item.get("modelId"):
                mid = str(item["modelId"]).strip()
                raw_ids.append(mid)
                variant_names[mid] = str(item.get("name") or mid).strip()

    # Prefer `configOptions[id="model"]` directly when provided by `agy acp` (exact UI order & display names!)
    cfg_opts = session_result.get("configOptions")
    grouped: list[AgyModelOption] = []
    default_base = ""
    if isinstance(cfg_opts, list):
        for opt in cfg_opts:
            if not isinstance(opt, dict) or opt.get("id") != "model":
                continue
            default_base = str(opt.get("currentValue") or "").strip()
            raw_choices = opt.get("options")
            if isinstance(raw_choices, list):
                for ch in raw_choices:
                    if not isinstance(ch, dict) or not ch.get("value"):
                        continue
                    bid = str(ch["value"]).strip()
                    bname = str(ch.get("name") or bid).strip()
                    v_ids = tuple(v for v in raw_ids if v == bid or v.startswith(f"{bid}-"))
                    grouped.append(AgyModelOption(model_id=bid, display_name=bname, variant_ids=v_ids))

    if not grouped and raw_ids:
        synth_lines = "\n".join(f"{vid}  {variant_names.get(vid, vid)}" for vid in raw_ids)
        grouped = parse_agy_models_cli_output(synth_lines)

    if not default_base and cur_variant:
        m = _EFFORT_SUFFIX_RE.match(cur_variant)
        default_base = m.group("base") if m else cur_variant
    if not default_base and grouped:
        default_base = grouped[0].model_id

    return default_base, grouped, set(raw_ids)


def resolve_acp_variant_id(
    requested_model: str,
    effort: ThinkingEffort,
    available_variant_ids: set[str],
    grouped_options: list[AgyModelOption] | None = None,
) -> str:
    """Maps a base model (`gemini-4-argon` or `Gemini 4 Argon`) + `ThinkingEffort` to a live ACP `modelId`."""
    req = requested_model.strip()
    if grouped_options:
        for opt in grouped_options:
            if req.lower() == opt.display_name.lower():
                req = opt.model_id
                break

    if req in available_variant_ids:
        return req

    m = _EFFORT_SUFFIX_RE.match(req)
    base = m.group("base") if m else req

    # Map `gemini-3.1-pro-preview` <-> `gemini-3.1-pro` transparently when switching across profiles
    if base.endswith("-preview") and not any(v == base or v.startswith(f"{base}-") for v in available_variant_ids):
        trimmed = base[: -len("-preview")]
        if any(v == trimmed or v.startswith(f"{trimmed}-") for v in available_variant_ids):
            base = trimmed

    if effort in (ThinkingEffort.MAX, ThinkingEffort.HIGH):
        prefs = ("high", "medium", "low")
    elif effort == ThinkingEffort.MEDIUM:
        prefs = ("medium", "high", "low")
    else:
        prefs = ("low", "medium", "high")

    for sfx in prefs:
        cand = f"{base}-{sfx}"
        if cand in available_variant_ids:
            return cand
    if base in available_variant_ids:
        return base
    return req


class ToolCallTagStreamSplitter:
    """Streams assistant prose with zero lag while withholding `<crux_tool_call>` / `<tool_call>` XML from TUI."""

    TAG_PAIRS: tuple[tuple[str, str], ...] = (
        ("<crux_tool_call>", "</crux_tool_call>"),
        ("<tool_call>", "</tool_call>"),
    )
    _BANNER_PREFIX = "⚠ Your previous response contained an improperly formatted function call"

    def __init__(self) -> None:
        self._active_close_tag: str | None = None
        self._buf = ""

    def feed(self, chunk: str) -> str:
        """Accepts a streamed text delta and returns only the visible prose safe to emit immediately."""
        if not chunk:
            return ""
        self._buf += chunk
        visible: list[str] = []
        while self._buf:
            if self._active_close_tag is None:
                self._buf = _AGY_MALFORMED_BANNER_STREAM_RE.sub("", self._buf)
                if not self._buf:
                    break

                best_idx = -1
                best_open = ""
                best_close = ""
                for otag, ctag in self.TAG_PAIRS:
                    idx = self._buf.find(otag)
                    if idx != -1 and (best_idx == -1 or idx < best_idx):
                        best_idx = idx
                        best_open = otag
                        best_close = ctag

                if best_idx != -1:
                    if best_idx > 0:
                        visible.append(self._buf[:best_idx])
                    self._buf = self._buf[best_idx + len(best_open) :]
                    self._active_close_tag = best_close
                    continue

                hold = max(self._partial_suffix_len(self._buf, otag) for otag, _ in self.TAG_PAIRS)
                warn_idx = self._buf.rfind("⚠")
                if warn_idx != -1 and len(self._buf) - warn_idx < 260:
                    tail = self._buf[warn_idx:]
                    if self._BANNER_PREFIX.startswith(tail) or tail.startswith("⚠ Your previous response"):
                        hold = max(hold, len(tail))

                emit_end = len(self._buf) - hold
                if emit_end > 0:
                    visible.append(self._buf[:emit_end])
                    self._buf = self._buf[emit_end:]
                break
            else:
                cidx = self._buf.find(self._active_close_tag)
                if cidx != -1:
                    self._buf = self._buf[cidx + len(self._active_close_tag) :]
                    self._active_close_tag = None
                    continue
                hold = self._partial_suffix_len(self._buf, self._active_close_tag)
                self._buf = self._buf[len(self._buf) - hold :] if hold else ""
                break
        return "".join(visible)

    def flush(self) -> str:
        rem = "" if self._active_close_tag is not None else _AGY_MALFORMED_BANNER_RE.sub("", self._buf)
        self._buf = ""
        self._active_close_tag = None
        return rem

    @staticmethod
    def _partial_suffix_len(text: str, tag: str) -> int:
        max_k = min(len(text), len(tag) - 1)
        for k in range(max_k, 0, -1):
            if text.endswith(tag[:k]):
                return k
        return 0


@dataclass(frozen=True)
class _AcpMsgRecord:
    """Normalization-invariant message record so dynamic tags & read tombstoning never drop live ACP sessions."""

    entry_id: str
    role: str
    msg_fp: str
    tool_part_fps: tuple[str, ...] = ()


@dataclass
class _TrackedAcpSession:
    acp_session_id: str
    current_variant_id: str
    sys_hash: str
    tools_hash: str
    msg_records: list[_AcpMsgRecord] = field(default_factory=list)


def _ensure_isolated_pipe_workspace(ush_home: pathlib.Path) -> pathlib.Path:
    """Creates `{ush_home}/agy_pipe` with a minimal `.git` root and zero-tool `crux-llm-pipe.md`."""
    pipe_dir = ush_home / "agy_pipe"
    agents_dir = pipe_dir / "_agents" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(pipe_dir, 0o700)
    except OSError:
        pass

    # Minimal `.git` marker so `agy` treats `pipe_dir` as an isolated customization root without shelling out
    git_dir = pipe_dir / ".git"
    if not git_dir.exists():
        (git_dir / "objects").mkdir(parents=True, exist_ok=True)
        (git_dir / "refs" / "heads").mkdir(parents=True, exist_ok=True)
        (git_dir / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")

    agent_md = agents_dir / f"{_CRUX_PIPE_AGENT_NAME}.md"
    if not agent_md.is_file() or agent_md.read_text(encoding="utf-8") != _CRUX_PIPE_AGENT_MD:
        agent_md.write_text(_CRUX_PIPE_AGENT_MD, encoding="utf-8")
    return pipe_dir


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:24]


def _is_pure_dynamic_msg(msg: CanonicalMessage) -> bool:
    """Returns True when a `user` frame carries only transient `<dynamic_instructions>` (erased on subsequent steps)."""
    if msg.role != "user":
        return False
    has_any = False
    for p in msg.parts:
        if p.kind == "image" and p.base64_data:
            return False
        if p.kind == "text":
            has_any = True
            if p.is_dynamic_instruction:
                continue
            if DYNAMIC_TAG_RE.sub("", p.text or "").strip():
                return False
    return has_any


def _build_acp_msg_record(msg: CanonicalMessage) -> _AcpMsgRecord:
    """Computes a stable prefix record invariant to `<dynamic_instructions>` erasure and model-switch thought scrubbing."""
    if msg.role == "tool":
        t_fps = tuple(
            _hash_text(f"{p.tool_name or ''}:{int(bool(p.is_error))}:{len(p.base64_data or '')}:{p.text or ''}")
            for p in msg.parts
            if p.kind == "tool_result"
        )
        return _AcpMsgRecord(
            entry_id=msg.entry_id or "",
            role="tool",
            msg_fp=_hash_text("tool:" + "|".join(t_fps)),
            tool_part_fps=t_fps,
        )

    if msg.role == "assistant":
        a_items: list[tuple[str, str, str]] = []
        for p in msg.parts:
            if p.kind == "thinking":
                continue
            if p.kind == "text":
                cleaned = _AGY_MALFORMED_BANNER_RE.sub("", p.text or "").strip()
                if cleaned:
                    a_items.append(("text", cleaned, ""))
            elif p.kind == "tool_call":
                a_items.append(("tool_call", p.tool_name or "", canonical_json(p.tool_args or {})))
        return _AcpMsgRecord(
            entry_id=msg.entry_id or "",
            role="assistant",
            msg_fp=_hash_text("assistant:" + json.dumps(a_items, ensure_ascii=False)),
        )

    u_items: list[tuple[str, str, int]] = []
    for p in msg.parts:
        if p.is_dynamic_instruction:
            continue
        if p.kind == "text":
            cleaned = DYNAMIC_TAG_RE.sub("", p.text or "").strip()
            if cleaned:
                u_items.append(("text", cleaned, 0))
        elif p.kind == "image":
            u_items.append(("image", p.mime_type or "", len(p.base64_data or "")))
    return _AcpMsgRecord(
        entry_id=msg.entry_id or "",
        role=msg.role,
        msg_fp=_hash_text(f"{msg.role}:" + json.dumps(u_items, ensure_ascii=False)),
    )


def _acp_record_matches(tr: _AcpMsgRecord, cur_msg: CanonicalMessage, cur_rec: _AcpMsgRecord) -> bool:
    """Checks whether `cur_msg` represents the same turn `agy` already holds in KV cache at this position."""
    if tr.role != cur_msg.role:
        return False
    if tr.entry_id and cur_msg.entry_id and tr.entry_id != cur_msg.entry_id:
        return False
    if tr.msg_fp == cur_rec.msg_fp:
        return True
    if tr.role == "tool":
        tool_parts = [p for p in cur_msg.parts if p.kind == "tool_result"]
        if len(tool_parts) != len(tr.tool_part_fps) or not tool_parts:
            return False
        for tp, tr_pfp, cur_pfp in zip(tool_parts, tr.tool_part_fps, cur_rec.tool_part_fps):
            if tr_pfp == cur_pfp:
                continue
            # `tombstone_superseded_reads` replaces older duplicate `read_file` outputs locally; `agy` already cached the read
            if not tp.is_error and (tp.text or "").startswith("[Superseded `read_file("):
                continue
            return False
        return True
    return False


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars < 320:
        return text[-max_chars:]
    head = int(max_chars * 0.60)
    tail = max(120, max_chars - head - 110)
    omitted = max(0, len(text) - head - tail)
    return f"{text[:head]}\n...[truncated {omitted:,} middle chars to fit ACP prompt window]...\n{text[-tail:]}"


def _render_tool_declarations(tools: list[dict[str, Any]]) -> str:
    if not tools:
        return ""
    lines = [
        "<crux_tool_protocol>",
        "You are executing inside the Crucible (`crux`) agent harness.",
        "ONLY the following tools exist. Whenever you need to call one or more tools, emit ONLY",
        '`<crux_tool_call>{"name": "<tool_name>", "arguments": {...}}</crux_tool_call>` XML blocks (one tag per call; '
        "you may emit multiple `<crux_tool_call>` tags in one turn for parallel reads) and then STOP and wait for "
        "`<tool_response>`. NEVER call host-injected function tools (e.g. `default_api:manage_task`); emit ONLY `<crux_tool_call>` XML blocks. Never fabricate tool outputs or claim a file was edited without emitting `<crux_tool_call>`.",
        "",
        "Available Crux Tools:",
    ]
    for t in sorted(tools, key=lambda item: str(item.get("name") or "")):
        schema_json = canonical_json(t.get("input_schema", {"type": "object", "properties": {}}))
        lines.append(f"- `{t['name']}`: {t.get('description', '')}\n  parameters: {schema_json}")
    lines.append("</crux_tool_protocol>")
    return "\n".join(lines)


def _message_to_acp_blocks(
    msg: CanonicalMessage,
    *,
    include_role_tag: bool = False,
    tool_output_cap: int | None = None,
    wrap_tag: str | None = None,
    text_cap: int | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if msg.role == "tool":
        res_lines: list[str] = []
        for p in msg.parts:
            if p.kind != "tool_result":
                continue
            status_attr = ' status="error"' if p.is_error else ""
            body_txt = p.text or ""
            if tool_output_cap is not None and len(body_txt) > tool_output_cap:
                body_txt = _truncate_middle(body_txt, tool_output_cap)
            res_lines.append(
                f'<tool_response name="{p.tool_name or "tool"}" call_id="{p.tool_call_id or ""}"{status_attr}>\n'
                f"{body_txt}\n"
                f"</tool_response>"
            )
            if p.base64_data:
                blocks.append(
                    {
                        "type": "image",
                        "data": p.base64_data,
                        "mimeType": p.mime_type or "image/png",
                    }
                )
        if res_lines:
            joined_res = "\n\n".join(res_lines)
            if text_cap is not None and len(joined_res) > text_cap:
                joined_res = _truncate_middle(joined_res, text_cap)
            blocks.insert(0, {"type": "text", "text": joined_res})
        return blocks

    if msg.role == "assistant":
        seg: list[str] = []
        for p in msg.parts:
            if p.kind == "text" and p.text:
                seg.append(_AGY_MALFORMED_BANNER_RE.sub("", p.text))
            elif p.kind == "tool_call":
                seg.append(
                    f'<crux_tool_call>{canonical_json({"name": p.tool_name, "arguments": p.tool_args or {}})}</crux_tool_call>'
                )
        joined = "\n".join(seg).strip()
        if text_cap is not None and len(joined) > text_cap:
            joined = _truncate_middle(joined, text_cap)
        if joined:
            tag = f"<prior_assistant_turn>\n{joined}\n</prior_assistant_turn>" if include_role_tag else joined
            blocks.append({"type": "text", "text": tag})
        return blocks

    role_lbl = wrap_tag or ("system_note" if msg.role == "system" else "user")
    text_segs: list[str] = []
    for p in msg.parts:
        if p.kind == "text" and p.text:
            text_segs.append(p.text)
        elif p.kind == "image" and p.base64_data:
            blocks.append(
                {
                    "type": "image",
                    "data": p.base64_data,
                    "mimeType": p.mime_type or "image/png",
                }
            )
    if text_segs:
        body = "\n".join(text_segs)
        if text_cap is not None and len(body) > text_cap:
            body = _truncate_middle(body, text_cap)
        if include_role_tag or wrap_tag:
            body = f"<{role_lbl}>\n{body}\n</{role_lbl}>"
        blocks.insert(0, {"type": "text", "text": body})
    return blocks


def _blocks_char_len(blocks: list[dict[str, Any]]) -> int:
    return sum(len(str(b.get("text") or "")) for b in blocks)


def _build_prior_conversation_digest(older_messages: list[CanonicalMessage], max_chars: int = 14_000) -> str:
    """Summarizes earlier turns chronologically when cold-resuming a large session so older goals & files stay visible."""
    user_turns: list[str] = []
    files_touched: list[str] = []
    prior_answers: list[str] = []

    for m in older_messages:
        if m.role == "user" and not _is_pure_dynamic_msg(m):
            cleaned = DYNAMIC_TAG_RE.sub("", m.plain_text() or "").strip()
            if cleaned:
                one_line = " ".join(cleaned.split())[:260]
                user_turns.append(one_line)
        elif m.role == "assistant":
            for p in m.parts:
                if p.kind == "tool_call" and isinstance(p.tool_args, dict):
                    fp = p.tool_args.get("path")
                    if isinstance(fp, str) and fp and fp not in files_touched:
                        files_touched.append(fp)
            ans_txt = _AGY_MALFORMED_BANNER_RE.sub("", m.plain_text() or "").strip()
            if ans_txt:
                first_ln = " ".join(ans_txt.splitlines()[:2])[:220]
                prior_answers.append(first_ln)

    if len(user_turns) <= 14:
        selected_users = [f"  {i + 1}. {u}" for i, u in enumerate(user_turns)]
    else:
        head_u = [f"  {i + 1}. {u}" for i, u in enumerate(user_turns[:4])]
        tail_start_num = len(user_turns) - 9
        tail_u = [f"  {tail_start_num + j}. {u}" for j, u in enumerate(user_turns[-10:])]
        selected_users = [*head_u, f"  ... ({len(user_turns) - 14} intermediate user turns) ...", *tail_u]

    lines = [
        "<prior_conversation_digest>",
        "Earlier turns in this session (already completed — DO NOT repeat prior explanations; use for background context only):",
    ]
    if selected_users:
        lines.append("Chronological earlier user prompts:")
        lines.extend(selected_users)
    if files_touched:
        lines.append("Files already inspected/modified earlier: " + ", ".join(files_touched[-30:]))
    if prior_answers:
        lines.append("Recent completed assistant conclusions (do not re-emit):")
        for pa in prior_answers[-6:]:
            lines.append(f"  - {pa}")
    lines.append("</prior_conversation_digest>")
    digest = "\n".join(lines)
    return _truncate_middle(digest, max_chars)


def _build_bounded_cold_prompt_blocks(
    *,
    system_prompt: str,
    tool_decl: str,
    messages: list[CanonicalMessage],
    max_total_chars: int = MAX_ACP_PROMPT_CHARS,
) -> list[dict[str, Any]]:
    """Builds a cold-hydration ACP prompt guaranteed `<= max_total_chars` while keeping the latest prompt 100% intact."""
    header_parts: list[str] = []
    if system_prompt.strip():
        header_parts.append(f"<crux_system_instructions>\n{system_prompt.strip()}\n</crux_system_instructions>")
    if tool_decl:
        header_parts.append(tool_decl)
    header_blocks: list[dict[str, Any]] = (
        [{"type": "text", "text": "\n\n".join(header_parts)}] if header_parts else []
    )
    header_chars = _blocks_char_len(header_blocks)
    avail_total = max(20_000, max_total_chars - header_chars - 512)

    if not messages:
        return [*header_blocks, {"type": "text", "text": "Hello."}]

    if len(messages) == 1:
        single_blocks = _message_to_acp_blocks(
            messages[0],
            include_role_tag=False,
            tool_output_cap=avail_total,
            text_cap=avail_total,
        )
        return [*header_blocks, *(single_blocks or [{"type": "text", "text": "Hello."}])]

    # Multi-turn cold hydration (e.g. `crux -c` resume): reserve headroom for `<prior_conversation_digest>` if needed
    digest_reserve = min(14_500, avail_total // 5)
    tail_budget = max(16_000, avail_total - digest_reserve)
    n = len(messages)
    last_is_user = messages[-1].role == "user"

    # Always render the final message first so its tail is guaranteed to fit verbatim
    final_cap = max(8_000, tail_budget - 2_000)
    final_blocks = _message_to_acp_blocks(
        messages[-1],
        include_role_tag=not last_is_user,
        wrap_tag="current_user_request" if last_is_user else None,
        tool_output_cap=final_cap,
        text_cap=final_cap,
    )
    used_chars = _blocks_char_len(final_blocks)

    # Walk backwards from `n - 2` down to `0` collecting as many recent turns verbatim as fit in `tail_budget`
    rendered_by_idx: dict[int, list[dict[str, Any]]] = {n - 1: final_blocks}
    tail_start = n - 1
    for idx in range(n - 2, -1, -1):
        t_cap = _RECENT_TOOL_OUTPUT_CAP_CHARS if idx >= n - 5 else _OLD_TOOL_OUTPUT_CAP_CHARS
        cand_blocks = _message_to_acp_blocks(
            messages[idx],
            include_role_tag=True,
            tool_output_cap=t_cap,
            text_cap=max(t_cap, 24_000),
        )
        cand_len = _blocks_char_len(cand_blocks)
        if used_chars + cand_len > tail_budget:
            break
        rendered_by_idx[idx] = cand_blocks
        used_chars += cand_len
        tail_start = idx

    # Snap `tail_start` forward off any orphan `role == "tool"` onto a clean `user`/`assistant` boundary
    while tail_start < n - 1 and messages[tail_start].role == "tool":
        used_chars -= _blocks_char_len(rendered_by_idx.pop(tail_start, []))
        tail_start += 1

    out_blocks: list[dict[str, Any]] = list(header_blocks)
    if tail_start > 0:
        digest_budget = max(2_000, avail_total - used_chars - 300)
        digest_txt = _build_prior_conversation_digest(messages[:tail_start], max_chars=min(14_000, digest_budget))
        out_blocks.append({"type": "text", "text": digest_txt})

    prior_tail_indices = [i for i in range(tail_start, n - 1) if i in rendered_by_idx]
    if last_is_user:
        if prior_tail_indices:
            out_blocks.append(
                {
                    "type": "text",
                    "text": (
                        "<recent_conversation_transcript>\n"
                        "[Completed prior turns in this session — reference for context, do NOT repeat their answers:]"
                    ),
                }
            )
            for i in prior_tail_indices:
                out_blocks.extend(rendered_by_idx[i])
            out_blocks.append({"type": "text", "text": "</recent_conversation_transcript>"})
        out_blocks.extend(final_blocks)
    else:
        out_blocks.append({"type": "text", "text": "<recent_conversation_transcript>"})
        for i in range(tail_start, n):
            if i in rendered_by_idx:
                out_blocks.extend(rendered_by_idx[i])
        out_blocks.append(
            {
                "type": "text",
                "text": "</recent_conversation_transcript>\nContinue the active turn directly from the latest `<tool_response>` above.",
            }
        )
    return out_blocks


def _build_bounded_delta_blocks(
    delta_msgs: list[CanonicalMessage],
    *,
    updated_tool_decl: str = "",
    max_total_chars: int = MAX_ACP_PROMPT_CHARS,
) -> list[dict[str, Any]]:
    """Renders incremental `delta_msgs` while guaranteeing single-step tool outputs never exceed `max_total_chars`."""
    prefix_blocks: list[dict[str, Any]] = []
    if updated_tool_decl:
        prefix_blocks.append(
            {"type": "text", "text": f"<updated_crux_tool_protocol>\n{updated_tool_decl}\n</updated_crux_tool_protocol>"}
        )
    avail = max(16_000, max_total_chars - _blocks_char_len(prefix_blocks) - 256)
    per_msg_cap = max(8_000, avail // max(1, len(delta_msgs)))
    t_cap = min(_DELTA_TOOL_OUTPUT_CAP_CHARS, per_msg_cap)

    blocks: list[dict[str, Any]] = list(prefix_blocks)
    for dm in delta_msgs:
        blocks.extend(
            _message_to_acp_blocks(
                dm,
                include_role_tag=False,
                tool_output_cap=t_cap,
                text_cap=per_msg_cap,
            )
        )
    return blocks or [{"type": "text", "text": "Continue."}]


class AgyAcpBridge:
    """Thread-safe persistent `agy acp` stdio worker supporting streaming thoughts, text, and models."""

    def __init__(self, ush_home: pathlib.Path, agy_bin: str) -> None:
        self.ush_home = ush_home
        self.agy_bin = agy_bin
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._rx_buf = bytearray()
        self._next_id = 1
        self._default_model: str = "gemini-4-argon"
        self._model_options: list[AgyModelOption] = []
        self._variant_ids: set[str] = set()
        self._sessions: dict[str, _TrackedAcpSession] = {}

    def close(self) -> None:
        with self._lock:
            self._shutdown_proc_locked()

    def _shutdown_proc_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._rx_buf.clear()
        self._sessions.clear()
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)
            else:
                proc.wait(timeout=0.5)
        except OSError:
            pass

    def _ensure_started_locked(self, timeout: float = 15.0) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._shutdown_proc_locked()
        pipe_dir = _ensure_isolated_pipe_workspace(self.ush_home)
        env = dict(os.environ)
        env.pop("ANTIGRAVITY_SOURCE_METADATA", None)
        proc = subprocess.Popen(
            [
                self.agy_bin,
                "acp",
                "--agent",
                _CRUX_PIPE_AGENT_NAME,
                "--include-default-customizations=false",
                "--disable-telemetry",
            ],
            cwd=str(pipe_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self._proc = proc
        try:
            self._rpc_locked("initialize", {"protocolVersion": 1, "clientCapabilities": {}}, timeout=timeout)
            boot_res = self._rpc_locked(
                "session/new",
                {"cwd": str(pipe_dir), "mcpServers": []},
                timeout=timeout,
            )
            def_m, opts, v_ids = parse_acp_session_models(boot_res)
            if def_m:
                self._default_model = def_m
            if opts:
                self._model_options = opts
            if v_ids:
                self._variant_ids = v_ids
            # Keep this warm bootstrap session ready for the first user session!
            boot_sid = str(boot_res.get("sessionId") or "")
            boot_cur = str(((boot_res.get("models") or {}).get("currentModelId")) or "")
            if boot_sid:
                self._sessions["__warm_spare__"] = _TrackedAcpSession(
                    acp_session_id=boot_sid,
                    current_variant_id=boot_cur,
                    sys_hash="",
                    tools_hash="",
                )
        except Exception:
            self._shutdown_proc_locked()
            raise

    def _write_json_locked(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise ConnectionError("agy acp stdin is closed")
        raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
        self._proc.stdin.write(raw)
        self._proc.stdin.flush()

    def _read_line_locked(
        self,
        *,
        deadline: float,
        cancel_event: threading.Event | None = None,
    ) -> tuple[str | None, bool]:
        """Returns `(json_line_or_None, was_canceled)` using `selectors` with zero leaked FDs."""
        while True:
            nl_idx = self._rx_buf.find(b"\n")
            if nl_idx != -1:
                line_bytes = bytes(self._rx_buf[:nl_idx])
                del self._rx_buf[: nl_idx + 1]
                return line_bytes.decode("utf-8", errors="replace"), False

            if cancel_event is not None and cancel_event.is_set():
                return None, True

            rem = deadline - time.monotonic()
            if rem <= 0:
                raise TimeoutError("Timed out waiting for response from `agy acp`.")

            if self._proc is None or self._proc.stdout is None:
                raise ConnectionError("`agy acp` process terminated unexpectedly.")

            sel = selectors.DefaultSelector()
            try:
                sel.register(self._proc.stdout, selectors.EVENT_READ)
                ready = sel.select(timeout=min(0.1, rem))
            finally:
                sel.close()

            if not ready:
                if self._proc.poll() is not None:
                    raise ConnectionError("`agy acp` exited unexpectedly.")
                continue

            chunk = os.read(self._proc.stdout.fileno(), 16_384)
            if not chunk:
                raise ConnectionError("EOF from `agy acp` stdout pipe.")
            self._rx_buf.extend(chunk)

    def _rpc_locked(
        self,
        method: str,
        params: dict[str, Any],
        *,
        timeout: float = 120.0,
        on_notification: Callable[[dict[str, Any]], None] | None = None,
        cancel_event: threading.Event | None = None,
        cancel_session_id: str = "",
    ) -> dict[str, Any]:
        req_id = self._next_id
        self._next_id += 1
        self._write_json_locked({"jsonrpc": "2.0", "id": req_id, "method": method, "params": params})
        deadline = time.monotonic() + timeout
        sent_cancel = False

        while True:
            line, canceled = self._read_line_locked(
                deadline=deadline,
                cancel_event=(None if sent_cancel else cancel_event),
            )
            if canceled and not sent_cancel:
                sent_cancel = True
                if cancel_session_id:
                    try:
                        self._write_json_locked(
                            {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": cancel_session_id}}
                        )
                    except OSError:
                        pass
                deadline = min(deadline, time.monotonic() + 2.0)
                continue

            if not line or not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            # Defense-in-depth: immediately cancel any unexpected server-to-client permission request!
            if "id" in obj and "method" in obj:
                srv_id = obj["id"]
                self._write_json_locked(
                    {"jsonrpc": "2.0", "id": srv_id, "result": {"outcome": {"outcome": "cancelled"}}}
                )
                continue

            if obj.get("id") == req_id:
                if "error" in obj and obj["error"]:
                    err_obj = obj["error"]
                    msg = err_obj.get("message") if isinstance(err_obj, dict) else str(err_obj)
                    raise RuntimeError(f"Antigravity ACP error: {msg}")
                res = obj.get("result")
                if sent_cancel:
                    return {"stopReason": "cancelled", "_canceled": True, **(res if isinstance(res, dict) else {})}
                return res if isinstance(res, dict) else {}

            if on_notification is not None and "method" in obj:
                on_notification(obj)

    def discover_models(self, timeout: float = 6.0) -> tuple[str, list[AgyModelOption]]:
        """Returns live `(default_model_id, available_model_options)` from the warm `agy acp` session."""
        with self._lock:
            if self._model_options:
                return self._default_model, list(self._model_options)
            self._ensure_started_locked(timeout=max(timeout, 4.0))
            return self._default_model, list(self._model_options)

    def stream_turn(
        self,
        *,
        profile: ModelCapabilityProfile,
        provider_name: str,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        thinking_effort: ThinkingEffort,
        on_event: Callable[[StreamEvent], None] | None = None,
        cancel_event: threading.Event | None = None,
        timeout: float = 300.0,
    ) -> CanonicalMessage:
        with self._lock:
            for attempt in range(2):
                try:
                    return self._stream_turn_once_locked(
                        profile=profile,
                        provider_name=provider_name,
                        system_prompt=system_prompt,
                        messages=messages,
                        tools=tools,
                        session_id=session_id,
                        thinking_effort=thinking_effort,
                        on_event=on_event,
                        cancel_event=cancel_event,
                        timeout=timeout,
                    )
                except (ConnectionError, BrokenPipeError):
                    self._shutdown_proc_locked()
                    if attempt == 1:
                        raise
            raise ConnectionError("Unable to establish `agy acp` session.")

    def _stream_turn_once_locked(
        self,
        *,
        profile: ModelCapabilityProfile,
        provider_name: str,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        thinking_effort: ThinkingEffort,
        on_event: Callable[[StreamEvent], None] | None,
        cancel_event: threading.Event | None,
        timeout: float,
    ) -> CanonicalMessage:
        self._ensure_started_locked(timeout=min(timeout, 20.0))
        pipe_dir = _ensure_isolated_pipe_workspace(self.ush_home)

        tool_decl = _render_tool_declarations(tools)
        sys_h = _hash_text(system_prompt)
        tools_h = _hash_text(tool_decl)
        is_isolated_eval = session_id.startswith("crux_eval_isolated:")
        sess_key = f"{session_id}::__notools__" if not tools else session_id
        durable_pairs = [
            (idx, m, _build_acp_msg_record(m))
            for idx, m in enumerate(messages)
            if not _is_pure_dynamic_msg(m)
        ]

        tracked = None if is_isolated_eval else self._sessions.get(sess_key)
        can_reuse = False
        delta_msgs: list[CanonicalMessage] = []
        if (
            not is_isolated_eval
            and tracked is not None
            and tracked.sys_hash == sys_h
            and len(durable_pairs) >= len(tracked.msg_records)
            and all(
                _acp_record_matches(tracked.msg_records[k], durable_pairs[k][1], durable_pairs[k][2])
                for k in range(len(tracked.msg_records))
            )
        ):
            last_matched_raw_idx = durable_pairs[len(tracked.msg_records) - 1][0] if tracked.msg_records else -1
            delta_msgs = messages[last_matched_raw_idx + 1 :]
            if delta_msgs and delta_msgs[0].role == "assistant":
                delta_msgs = delta_msgs[1:]
            if delta_msgs:
                can_reuse = True

        if can_reuse and tracked is not None:
            updated_decl = tool_decl if (tracked.tools_hash != tools_h and tool_decl) else ""
            prompt_blocks = _build_bounded_delta_blocks(delta_msgs, updated_tool_decl=updated_decl)
            tracked.tools_hash = tools_h
        else:
            spare = None if is_isolated_eval else self._sessions.pop("__warm_spare__", None)
            if spare is not None:
                tracked = _TrackedAcpSession(
                    acp_session_id=spare.acp_session_id,
                    current_variant_id=spare.current_variant_id,
                    sys_hash=sys_h,
                    tools_hash=tools_h,
                )
            else:
                new_res = self._rpc_locked(
                    "session/new",
                    {"cwd": str(pipe_dir), "mcpServers": []},
                    timeout=20.0,
                )
                tracked = _TrackedAcpSession(
                    acp_session_id=str(new_res.get("sessionId") or ""),
                    current_variant_id=str(((new_res.get("models") or {}).get("currentModelId")) or ""),
                    sys_hash=sys_h,
                    tools_hash=tools_h,
                )
            prompt_blocks = _build_bounded_cold_prompt_blocks(
                system_prompt=system_prompt,
                tool_decl=tool_decl,
                messages=messages,
            )

        assert tracked is not None
        target_variant = resolve_acp_variant_id(
            profile.model_id,
            thinking_effort,
            self._variant_ids,
            self._model_options,
        )
        if target_variant and target_variant != tracked.current_variant_id:
            self._rpc_locked(
                "session/set_model",
                {"sessionId": tracked.acp_session_id, "modelId": target_variant},
                timeout=10.0,
            )
            tracked.current_variant_id = target_variant

        think_buf: list[str] = []
        raw_text_buf: list[str] = []
        visible_text_buf: list[str] = []
        splitter = ToolCallTagStreamSplitter()
        think_t0: float | None = None
        think_closed = False

        def _close_thought() -> None:
            nonlocal think_closed
            if think_t0 is not None and not think_closed:
                think_closed = True
                if on_event:
                    chars = sum(len(x) for x in think_buf)
                    on_event(
                        StreamEvent(
                            kind="thinking_end",
                            elapsed_ms=(time.monotonic() - think_t0) * 1000.0,
                            meta={
                                "thought_chars": chars,
                                "reasoning_tokens": max(1, chars // 4),
                                "has_signature": True,
                            },
                        )
                    )

        def _on_notify(pkt: dict[str, Any]) -> None:
            nonlocal think_t0
            if pkt.get("method") != "session/update":
                return
            params = pkt.get("params")
            if not isinstance(params, dict) or params.get("sessionId") != tracked.acp_session_id:
                return
            upd = params.get("update")
            if not isinstance(upd, dict):
                return
            su = upd.get("sessionUpdate")
            content = upd.get("content")
            txt = str(content.get("text") or "") if isinstance(content, dict) else ""
            if not txt:
                return
            if su == "agent_thought_chunk":
                if think_t0 is None:
                    think_t0 = time.monotonic()
                think_buf.append(txt)
                if on_event:
                    on_event(StreamEvent(kind="thinking_delta", text=txt))
            elif su == "agent_message_chunk":
                _close_thought()
                raw_text_buf.append(txt)
                vis = splitter.feed(txt)
                if vis:
                    visible_text_buf.append(vis)
                    if on_event:
                        on_event(StreamEvent(kind="text_delta", text=vis))

        res = self._rpc_locked(
            "session/prompt",
            {"sessionId": tracked.acp_session_id, "prompt": prompt_blocks},
            timeout=timeout,
            on_notification=_on_notify,
            cancel_event=cancel_event,
            cancel_session_id=tracked.acp_session_id,
        )
        tail_vis = splitter.flush()
        if tail_vis:
            visible_text_buf.append(tail_vis)
            if on_event:
                on_event(StreamEvent(kind="text_delta", text=tail_vis))
        _close_thought()

        if res.get("_canceled") or res.get("stopReason") == "cancelled":
            self._sessions.pop(sess_key, None)
            return CanonicalMessage(role="assistant", parts=[], stop_reason=StopReason.CANCELED)

        usage_raw = res.get("usage") if isinstance(res.get("usage"), dict) else {}
        raw_in = int(usage_raw.get("inputTokens") or 0)
        cached_in = int(usage_raw.get("cachedReadTokens") or 0)
        usage = TokenUsage(
            input_tokens=raw_in + cached_in,
            output_tokens=int(usage_raw.get("outputTokens") or 0),
            cached_input_tokens=cached_in,
            reasoning_tokens=int(usage_raw.get("thoughtTokens") or 0),
        )
        if think_buf and usage.reasoning_tokens == 0:
            usage.reasoning_tokens = max(1, sum(len(x) for x in think_buf) // 4)

        full_raw_text = _AGY_MALFORMED_BANNER_RE.sub("", "".join(raw_text_buf))
        known_names = {str(t["name"]) for t in tools if t.get("name")} if tools else None
        cleaned_text, tool_calls = extract_fallback_tool_calls(full_raw_text, known_names)
        cleaned_text = cleaned_text.strip()

        # Also catch bare JSON tool invocation if model emitted a standalone JSON tool object without XML tags
        if not tool_calls and known_names and cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            bare_obj = parse_lenient_json_object(cleaned_text)
            if isinstance(bare_obj, dict):
                b_name = bare_obj.get("name") or bare_obj.get("tool")
                b_args = bare_obj.get("arguments") if "arguments" in bare_obj else bare_obj.get("args", {})
                if isinstance(b_name, str) and b_name in known_names and isinstance(b_args, dict):
                    tool_calls = [
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id=new_id("call"),
                            tool_name=b_name,
                            tool_args=b_args,
                        )
                    ]
                    cleaned_text = ""

        parts: list[CanonicalPart] = []
        if think_buf:
            parts.append(CanonicalPart(kind="thinking", text="".join(think_buf)))
        if cleaned_text:
            parts.append(CanonicalPart(kind="text", text=cleaned_text))
        for tc in tool_calls:
            parts.append(tc)
            if on_event:
                on_event(StreamEvent(kind="tool_call", part=tc))

        stop_r = StopReason.STOP
        if res.get("stopReason") == "max_tokens":
            stop_r = StopReason.MAX_TOKENS
        elif not parts:
            stop_r = StopReason.EMPTY_200_RESPONSE

        assistant_msg = CanonicalMessage(
            role="assistant",
            parts=parts,
            producer_provider=provider_name,
            producer_model_id=profile.model_id,
            stop_reason=stop_r,
            usage=usage,
        )

        # Record durable prefix state only for reusable non-isolated sessions
        if not is_isolated_eval:
            next_records = [rec for _, _, rec in durable_pairs]
            if assistant_msg.parts and stop_r != StopReason.EMPTY_200_RESPONSE:
                next_records.append(_build_acp_msg_record(assistant_msg))
            tracked.msg_records = next_records
            self._sessions[sess_key] = tracked
        return assistant_msg


_BRIDGES_LOCK = threading.Lock()
_BRIDGES: dict[tuple[str, str], AgyAcpBridge] = {}


def get_shared_agy_bridge(ush_home: pathlib.Path, agy_bin: str | None = None) -> AgyAcpBridge:
    """Returns a process-wide warm `AgyAcpBridge` for `(ush_home, agy_bin)`."""
    resolved = agy_bin or resolve_agy_binary()
    if not resolved:
        raise FileNotFoundError("Antigravity CLI (`agy`) was not found on PATH.")
    key = (str(ush_home.resolve()), resolved)
    with _BRIDGES_LOCK:
        bridge = _BRIDGES.get(key)
        if bridge is None:
            bridge = AgyAcpBridge(ush_home.resolve(), resolved)
            _BRIDGES[key] = bridge
        return bridge


def close_all_agy_bridges() -> None:
    with _BRIDGES_LOCK:
        bridges = list(_BRIDGES.values())
        _BRIDGES.clear()
    for b in bridges:
        b.close()


atexit.register(close_all_agy_bridges)
