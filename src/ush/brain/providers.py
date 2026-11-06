"""Multi-Provider Streaming Wire Client (Gemini, Vertex, Anthropic, Vertex-Claude, OpenAI, OpenRouter, Local).

Features:
- Deterministic `canonical_json` tool schema sorting and FNV-1a sticky cache routing.
- Real SSE (`text/event-stream`) line-buffered delta parsing across all 5 provider wire dialects.
- `EMPTY_200_RESPONSE` automatic retry and `MAX_TOKENS` truncated-tool-JSON discard (CT-02).
- Automatic HTTP `400` signature-strip retry and HTTP `429` multi-profile failover.
- Tier-B `<tool_call>` / fenced-JSON fallback extraction & 1-turn repair support (CT-04).
"""

from __future__ import annotations

from dataclasses import dataclass
import http.client
import json
import os
import re
import time
from typing import Any, Callable, Iterator, Protocol
import urllib.parse

from ush.auth.vault import AuthProfile, CredentialVault, secure_ssl_context
from ush.brain.wire import (
    ToolIdBijectionMap,
    extract_fallback_tool_calls,
    parse_lenient_json_object,
    transform_messages,
)
from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    ModelCapabilityProfile,
    StopReason,
    StreamEvent,
    ThinkingEffort,
    TokenUsage,
    canonical_json,
    fnv1a_64_hex,
    new_id,
)


class WireError(RuntimeError):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.detail = message


class ProviderTransport(Protocol):
    def stream_turn(
        self,
        *,
        profile: ModelCapabilityProfile,
        auth_profile: AuthProfile,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CanonicalMessage: ...


# ============================================================================
# Payload Builders (Exposed cleanly for deterministic CT-01 / CT-03 inspection)
# ============================================================================


def build_gemini_thinking_config(
    profile: ModelCapabilityProfile, effort: ThinkingEffort
) -> dict[str, Any] | None:
    """Builds exact Gemini 2.5 (`thinkingBudget`) vs Gemini 3/4 (`thinkingLevel`) wire config with `includeThoughts`."""
    low = profile.model_id.lower()
    if profile.reasoning_wire_mode != "gemini_thought_sig" and "gemini" not in low and "argon" not in low:
        return None
    if any(tag in low for tag in ("gemini-1.", "gemini-2.0")):
        return None

    is_gen3_plus = any(low.startswith(p) for p in ("gemini-3", "gemini-4")) or "argon" in low
    is_flash_or_lite = "flash" in low or "lite" in low

    if effort == ThinkingEffort.OFF:
        if is_gen3_plus:
            lvl = "MINIMAL" if is_flash_or_lite and "3.8" not in low else "LOW"
            return {"includeThoughts": False, "thinkingLevel": lvl}
        return {"includeThoughts": False, "thinkingBudget": 0 if is_flash_or_lite else 128}

    if is_gen3_plus:
        if effort == ThinkingEffort.LOW:
            lvl = "LOW"
        elif effort == ThinkingEffort.MEDIUM:
            # gemini-3-pro-preview supports LOW and HIGH only; 3.1-pro / 3.x-flash / 4-argon support MEDIUM
            lvl = "HIGH" if low == "gemini-3-pro-preview" else "MEDIUM"
        else:
            lvl = "HIGH"
        return {"includeThoughts": True, "thinkingLevel": lvl}

    # Gemini 2.5 series (`thinkingBudget` + `includeThoughts: True`)
    budget_map = {
        ThinkingEffort.LOW: 2048,
        ThinkingEffort.MEDIUM: 8192,
        ThinkingEffort.HIGH: 24576,
        ThinkingEffort.MAX: -1,  # Dynamic unbounded ceiling on Gemini 2.5
    }
    return {"includeThoughts": True, "thinkingBudget": budget_map.get(effort, 24576)}


class InlineThinkStreamSplitter:
    """Zero-lag streaming splitter converting inline `<think>...</think>` chunks into `thinking_delta` vs `text_delta`."""

    OPEN_TAG = "<think>"
    CLOSE_TAG = "</think>"

    def __init__(self) -> None:
        self._in_think = False
        self._pending = ""
        self._seen_any_text = False

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        if not chunk:
            return []
        self._pending += chunk
        out: list[tuple[str, str]] = []
        while self._pending:
            if not self._in_think:
                idx = self._pending.find(self.OPEN_TAG)
                if idx != -1:
                    before = self._pending[:idx]
                    if before:
                        out.append(("text_delta", before))
                        self._seen_any_text = True
                    self._pending = self._pending[idx + len(self.OPEN_TAG) :]
                    self._in_think = True
                    continue
                # Hold back possible partial `<think>` prefix at chunk tail
                keep = self._partial_suffix_len(self._pending, self.OPEN_TAG)
                emit = self._pending[: len(self._pending) - keep] if keep else self._pending
                if emit:
                    out.append(("text_delta", emit))
                    self._seen_any_text = True
                self._pending = self._pending[len(self._pending) - keep :] if keep else ""
                break
            else:
                idx = self._pending.find(self.CLOSE_TAG)
                if idx != -1:
                    thought = self._pending[:idx]
                    if thought:
                        out.append(("thinking_delta", thought))
                    self._pending = self._pending[idx + len(self.CLOSE_TAG) :].lstrip("\r\n")
                    self._in_think = False
                    continue
                keep = self._partial_suffix_len(self._pending, self.CLOSE_TAG)
                emit = self._pending[: len(self._pending) - keep] if keep else self._pending
                if emit:
                    out.append(("thinking_delta", emit))
                self._pending = self._pending[len(self._pending) - keep :] if keep else ""
                break
        return out

    def flush(self) -> list[tuple[str, str]]:
        if not self._pending:
            return []
        kind = "thinking_delta" if self._in_think else "text_delta"
        rem = self._pending
        self._pending = ""
        return [(kind, rem)]

    @staticmethod
    def _partial_suffix_len(buf: str, tag: str) -> int:
        max_k = min(len(buf), len(tag) - 1)
        for k in range(max_k, 0, -1):
            if buf.endswith(tag[:k]):
                return k
        return 0


def _sorted_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return json.loads(canonical_json(sorted(tools, key=lambda t: str(t.get("name", "")))))


_BASE64_SIG_RE = re.compile(r"^[A-Za-z0-9+/_-]+={0,2}$")
_GEMINI_MAJOR_RE = re.compile(r"^gemini(?:-live)?-(\d+)", re.IGNORECASE)
_GEMINI_INTERRUPTED_MODEL_BRIDGE = "[The previous response was interrupted before it completed.]"


def _is_valid_base64_signature(sig: str | None) -> bool:
    return bool(sig and len(sig) >= 4 and len(sig) % 4 == 0 and _BASE64_SIG_RE.match(sig))


def _gemini_major_version(model_id: str) -> int | None:
    m = _GEMINI_MAJOR_RE.match((model_id or "").strip())
    return int(m.group(1)) if m else None


def _gemini_requires_tool_call_id(model_id: str) -> bool:
    major = _gemini_major_version(model_id)
    return major is not None and major >= 3


def _gemini_supports_multimodal_fr(model_id: str) -> bool:
    major = _gemini_major_version(model_id)
    return major is not None and major >= 3


def _is_empty_gemini_part(p: dict[str, Any]) -> bool:
    if set(p.keys()) <= {"text"} and not str(p.get("text") or "").strip():
        return True
    return False


def _synthetic_gemini_fr(
    fn_name: str, fc_id: str | None = None, *, include_id: bool = False
) -> dict[str, Any]:
    fr: dict[str, Any] = {
        "name": fn_name or "tool",
        "response": {
            "error": "[SYNTHETIC_REPAIR: prior tool invocation interrupted before result was recorded]",
        },
    }
    if include_id and fc_id:
        fr["id"] = fc_id
    return {"functionResponse": fr}


def _enforce_gemini_tool_bijection(
    contents: list[dict[str, Any]], *, include_id: bool = False
) -> list[dict[str, Any]]:
    """Defense-in-depth wire invariant for Gemini 2.5/3/4:

    Guarantees EVERY `model` turn containing K >= 1 `functionCall` parts is IMMEDIATELY followed by a
    `user` turn containing ONLY K `functionResponse` parts with identical `name`s (and `id`s on Gemini 3+)
    in exact positional order, and strips any unanchored `functionResponse` turns.
    """
    healed: list[dict[str, Any]] = []
    idx = 0
    n = len(contents)

    while idx < n:
        cur = contents[idx]
        role = cur.get("role")
        parts = [p for p in (cur.get("parts") or []) if not _is_empty_gemini_part(p)]
        if not parts:
            idx += 1
            continue

        if role == "model":
            fc_specs = [
                (
                    str((p["functionCall"].get("name") or "tool")),
                    p["functionCall"].get("id"),
                )
                for p in parts
                if "functionCall" in p
            ]
            healed.append({"role": "model", "parts": parts})
            if not fc_specs:
                idx += 1
                continue

            pool: list[dict[str, Any]] = []
            consume_next = False
            if idx + 1 < n and contents[idx + 1].get("role") == "user":
                nxt_parts = contents[idx + 1].get("parts") or []
                if nxt_parts and all("functionResponse" in x for x in nxt_parts):
                    pool = list(nxt_parts)
                    consume_next = True

            # Fast path: exact count + exact positional names (& IDs when enabled) already match
            if len(pool) == len(fc_specs) and all(
                str((pool[k]["functionResponse"].get("name") or "tool")) == fc_specs[k][0]
                and (
                    not include_id
                    or not fc_specs[k][1]
                    or pool[k]["functionResponse"].get("id") == fc_specs[k][1]
                )
                for k in range(len(fc_specs))
            ):
                healed.append({"role": "user", "parts": pool})
            else:
                used = [False] * len(pool)
                aligned: list[dict[str, Any]] = []
                for fname, fcid in fc_specs:
                    picked: dict[str, Any] | None = None
                    if fcid:
                        for p_i, cand in enumerate(pool):
                            if not used[p_i] and cand["functionResponse"].get("id") == fcid:
                                used[p_i] = True
                                picked = cand
                                break
                    if picked is None:
                        for p_i, cand in enumerate(pool):
                            if not used[p_i] and str((cand["functionResponse"].get("name") or "")) == fname:
                                used[p_i] = True
                                picked = cand
                                break
                    if picked is not None:
                        fr_copy = dict(picked["functionResponse"])
                        fr_copy["name"] = fname
                        if include_id and fcid:
                            fr_copy["id"] = fcid
                        elif not include_id:
                            fr_copy.pop("id", None)
                        aligned.append({"functionResponse": fr_copy})
                    else:
                        aligned.append(_synthetic_gemini_fr(fname, fcid, include_id=include_id))
                healed.append({"role": "user", "parts": aligned})

            idx += 2 if consume_next else 1
            continue

        # `role == "user"`: strip any unanchored `functionResponse` parts without a preceding `functionCall` turn
        clean_user = [p for p in parts if "functionResponse" not in p]
        if clean_user:
            if (
                healed
                and healed[-1]["role"] == "user"
                and all("functionResponse" not in x for x in healed[-1]["parts"])
            ):
                healed[-1]["parts"].extend(clean_user)
            else:
                healed.append({"role": "user", "parts": clean_user})
        idx += 1

    return healed


def _enforce_gemini_role_alternation(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Enforces all 5 `historyHardening.ts` structural invariants simultaneously:

    1. Start Constraint: `contents[0]["role"] == "user"`.
    2. Strict Role Alternation: `contents[i]["role"] != contents[i+1]["role"]` for all `i`.
    3. Pure `functionResponse` Isolation: `functionResponse` turns NEVER merge with plain user text/media;
       instead, a deterministic closing `model` bridge separates `user(FR)` from an immediately following `user(text)`.
    4. Tool Bijection Preservation: never alters `model(FC) -> user(FR)` adjacency.
    5. Terminal Constraint: `contents[-1]["role"] == "user"` with non-empty parts.
    """
    alternating: list[dict[str, Any]] = []
    for cur in contents:
        role = cur.get("role")
        parts = [p for p in (cur.get("parts") or []) if not _is_empty_gemini_part(p)]
        if not parts:
            continue
        if not alternating:
            if role == "model":
                alternating.append({"role": "user", "parts": [{"text": "[Continuing conversation...]"}]})
            alternating.append({"role": role, "parts": parts})
            continue

        prev = alternating[-1]
        if prev["role"] == role:
            if role == "user":
                prev_has_fr = any("functionResponse" in x for x in prev["parts"])
                cur_has_fr = any("functionResponse" in x for x in parts)
                if not prev_has_fr and not cur_has_fr:
                    prev["parts"].extend(parts)
                else:
                    alternating.append(
                        {"role": "model", "parts": [{"text": _GEMINI_INTERRUPTED_MODEL_BRIDGE}]}
                    )
                    alternating.append({"role": "user", "parts": parts})
            else:
                prev_has_fc = any("functionCall" in x for x in prev["parts"])
                cur_has_fc = any("functionCall" in x for x in parts)
                if not prev_has_fc and not cur_has_fc:
                    prev["parts"].extend(parts)
                else:
                    alternating.append({"role": "user", "parts": [{"text": "Continue."}]})
                    alternating.append({"role": "model", "parts": parts})
        else:
            alternating.append({"role": role, "parts": parts})

    if not alternating or alternating[-1]["role"] == "model":
        alternating.append({"role": "user", "parts": [{"text": "Continue."}]})
    return alternating


def build_gemini_payload(
    system_prompt: str,
    messages: list[CanonicalMessage],
    tools: list[dict[str, Any]],
    profile: ModelCapabilityProfile,
    *,
    thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
    enable_thinking_config: bool = True,
    is_vertex: bool = False,
) -> dict[str, Any]:
    include_id = _gemini_requires_tool_call_id(profile.model_id)
    multimodal_fr = _gemini_supports_multimodal_fr(profile.model_id)
    contents: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_prompt = f"{system_prompt}\n\n{msg.plain_text()}".strip()
            continue
        g_role = "model" if msg.role == "assistant" else "user"
        parts: list[dict[str, Any]] = []
        deferred_g25_images: list[dict[str, Any]] = []
        turn_sig: str | None = None
        for p in msg.parts:
            if p.signature and not turn_sig:
                turn_sig = p.signature
            if p.kind == "thinking" and p.text:
                # Thought summary text is unscoped prose; never duplicate functionCall signature onto it.
                parts.append({"text": p.text, "thought": True})
            elif p.kind == "text":
                has_valid_sig = bool(p.signature) and (not is_vertex or _is_valid_base64_signature(p.signature))
                if (p.text and p.text.strip()) or has_valid_sig:
                    parts.append({"text": p.text})
            elif p.kind == "image" and p.base64_data:
                parts.append(
                    {
                        "inlineData": {
                            "mimeType": p.mime_type or "image/png",
                            "data": p.base64_data,
                        }
                    }
                )
            elif p.kind == "tool_call":
                fc_obj: dict[str, Any] = {
                    "name": p.tool_name,
                    "args": p.tool_args or {},
                }
                if include_id and p.tool_call_id:
                    fc_obj["id"] = p.tool_call_id
                fc_part: dict[str, Any] = {"functionCall": fc_obj}
                if p.signature:
                    fc_part["_raw_sig"] = p.signature
                parts.append(fc_part)
            elif p.kind == "tool_result":
                has_img = bool(p.base64_data and profile.supports_vision)
                resp_val = p.text if p.text else ("(see attached image)" if has_img else "")
                resp_obj: dict[str, Any] = (
                    {"error": resp_val or "Tool execution failed"}
                    if p.is_error
                    else {"output": resp_val}
                )
                fr_obj: dict[str, Any] = {
                    "name": p.tool_name or "tool",
                    "response": resp_obj,
                }
                if include_id and p.tool_call_id:
                    fr_obj["id"] = p.tool_call_id
                if has_img:
                    img_part = {
                        "inlineData": {
                            "mimeType": p.mime_type or "image/png",
                            "data": p.base64_data,
                        }
                    }
                    if multimodal_fr:
                        fr_obj["parts"] = [img_part]
                    else:
                        deferred_g25_images.append(img_part)
                parts.append({"functionResponse": fr_obj})
        if not parts:
            continue

        if g_role == "model":
            # Gemini 3/4 Law (`geminiChat.ts:2093-2112`, `historyHardening.ts:12, 37`):
            # On parallel `functionCall` turns, ONLY the FIRST `functionCall` carries `thoughtSignature`.
            # On Vertex AI, non-Base64 mock strings are replaced with the 32-char Base64URL-valid sentinel
            # `"skip_thought_signature_validator"` so mid-turn cross-model handoffs never fail with
            # `HTTP 400: Function call is missing a thought_signature in functionCall parts`.
            fc_indices = [idx for idx, pt in enumerate(parts) if "functionCall" in pt]
            if fc_indices:
                first_idx = fc_indices[0]
                raw_first = parts[first_idx].pop("_raw_sig", None) or turn_sig
                if is_vertex and raw_first and not _is_valid_base64_signature(raw_first):
                    raw_first = None
                primary_sig = raw_first or "skip_thought_signature_validator"
                parts[first_idx]["thoughtSignature"] = primary_sig
                for idx in fc_indices[1:]:
                    extra_sig = parts[idx].pop("_raw_sig", None)
                    if (
                        extra_sig
                        and extra_sig != primary_sig
                        and extra_sig != "skip_thought_signature_validator"
                        and (not is_vertex or _is_valid_base64_signature(extra_sig))
                    ):
                        parts[idx]["thoughtSignature"] = extra_sig
            elif turn_sig and (not is_vertex or _is_valid_base64_signature(turn_sig)):
                for idx in range(len(parts) - 1, -1, -1):
                    if not parts[idx].get("thought"):
                        parts[idx]["thoughtSignature"] = turn_sig
                        break
            if contents and contents[-1]["role"] == "model":
                prev_has_fc = any("functionCall" in x for x in contents[-1]["parts"])
                cur_has_fc = bool(fc_indices)
                if prev_has_fc or cur_has_fc:
                    if not prev_has_fc:
                        contents.append({"role": "user", "parts": [{"text": "Continue."}]})
                    contents.append({"role": "model", "parts": parts})
                else:
                    contents[-1]["parts"].extend(parts)
            else:
                contents.append({"role": "model", "parts": parts})
        else:
            # Gemini 3/4 Law: NEVER merge `functionResponse` parts and plain user `text`/`inlineData` parts
            # into the same `Content` object, and ONLY merge consecutive `functionResponse` messages when the
            # immediately preceding `model` turn still has unmatched `functionCall` slots open.
            fr_parts = [pt for pt in parts if "functionResponse" in pt]
            user_parts = [pt for pt in parts if "functionResponse" not in pt]
            if fr_parts:
                can_merge_fr = False
                if (
                    len(contents) >= 2
                    and contents[-1]["role"] == "user"
                    and all("functionResponse" in x for x in contents[-1]["parts"])
                    and contents[-2]["role"] == "model"
                ):
                    expected_n = sum(1 for x in contents[-2]["parts"] if "functionCall" in x)
                    if len(contents[-1]["parts"]) < expected_n:
                        can_merge_fr = True
                if can_merge_fr:
                    contents[-1]["parts"].extend(fr_parts)
                else:
                    contents.append({"role": "user", "parts": fr_parts})
            if deferred_g25_images:
                user_parts.extend([{"text": "Tool result image:"}, *deferred_g25_images])
            if user_parts:
                if (
                    contents
                    and contents[-1]["role"] == "user"
                    and all("functionResponse" not in x for x in contents[-1]["parts"])
                ):
                    contents[-1]["parts"].extend(user_parts)
                else:
                    contents.append({"role": "user", "parts": user_parts})

    contents = _enforce_gemini_tool_bijection(contents, include_id=include_id)
    contents = _enforce_gemini_role_alternation(contents)

    gen_cfg: dict[str, Any] = {
        "maxOutputTokens": profile.max_output_tokens,
    }
    if enable_thinking_config:
        t_cfg = build_gemini_thinking_config(profile, thinking_effort)
        if t_cfg:
            gen_cfg["thinkingConfig"] = t_cfg
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": gen_cfg,
    }
    if system_prompt:
        payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
    if tools:
        decls = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in _sorted_tools(tools)
        ]
        payload["tools"] = [{"functionDeclarations": decls}]
    return payload


def build_anthropic_payload(
    system_prompt: str,
    messages: list[CanonicalMessage],
    tools: list[dict[str, Any]],
    profile: ModelCapabilityProfile,
    *,
    is_vertex: bool = False,
    thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    anth_msgs: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            system_prompt = f"{system_prompt}\n\n{msg.plain_text()}".strip()
            continue
        a_role = "assistant" if msg.role == "assistant" else "user"
        blocks: list[dict[str, Any]] = []
        for p in msg.parts:
            if p.kind == "thinking":
                if p.redacted_data:
                    blocks.append({"type": "redacted_thinking", "data": p.redacted_data})
                elif p.signature:
                    blocks.append({"type": "thinking", "thinking": p.text, "signature": p.signature})
            elif p.kind == "text" and p.text and p.text.strip():
                blocks.append({"type": "text", "text": p.text})
            elif p.kind == "image" and p.base64_data:
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": p.mime_type or "image/png",
                            "data": p.base64_data,
                        },
                    }
                )
            elif p.kind == "tool_call":
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": p.tool_call_id,
                        "name": p.tool_name,
                        "input": p.tool_args or {},
                    }
                )
            elif p.kind == "tool_result":
                content_blocks: list[dict[str, Any]] = [{"type": "text", "text": p.text}]
                if p.base64_data and not p.is_error:
                    content_blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": p.mime_type or "image/png",
                                "data": p.base64_data,
                            },
                        }
                    )
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": p.tool_call_id,
                        "content": content_blocks,
                        "is_error": p.is_error,
                    }
                )
        if blocks:
            if anth_msgs and anth_msgs[-1]["role"] == a_role:
                anth_msgs[-1]["content"].extend(blocks)
            else:
                anth_msgs.append({"role": a_role, "content": blocks})

    # Anthropic structural invariants (`anthropic-messages.ts:1431-1521`):
    # 1. Start & end on `role == "user"`.
    # 2. Inside `assistant`: `thinking`/`redacted_thinking` blocks MUST precede `text`/`tool_use`.
    # 3. Inside `user`: `tool_result` blocks MUST strictly precede `text`/`image` blocks.
    if anth_msgs and anth_msgs[0]["role"] == "assistant":
        anth_msgs.insert(0, {"role": "user", "content": [{"type": "text", "text": "[Continuing conversation...]"}]})
    if not anth_msgs or anth_msgs[-1]["role"] == "assistant":
        anth_msgs.append({"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    for am in anth_msgs:
        c_blocks = am["content"]
        if am["role"] == "assistant":
            think_b = [b for b in c_blocks if b["type"] in ("thinking", "redacted_thinking")]
            rest_b = [b for b in c_blocks if b["type"] not in ("thinking", "redacted_thinking")]
            am["content"] = think_b + rest_b
        else:
            tr_b = [b for b in c_blocks if b["type"] == "tool_result"]
            rest_b = [b for b in c_blocks if b["type"] != "tool_result"]
            am["content"] = tr_b + rest_b

    # Place ephemeral KV-cache breakpoint on Turn-0 system prompt and trailing user block
    if anth_msgs and anth_msgs[-1]["content"]:
        anth_msgs[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}

    low_m = profile.model_id.lower()
    supports_ext_think = (
        enable_thinking
        and thinking_effort != ThinkingEffort.OFF
        and profile.reasoning_wire_mode == "anthropic_signed_thinking"
        and not any(tag in low_m for tag in ("claude-3-5", "claude-3-haiku", "claude-3-opus"))
    )
    anth_budgets = {
        ThinkingEffort.LOW: 2_048,
        ThinkingEffort.MEDIUM: 8_192,
        ThinkingEffort.HIGH: 20_480,
        ThinkingEffort.MAX: 32_000,
    }
    budget = anth_budgets.get(thinking_effort, 20_480)
    max_tok = max(profile.max_output_tokens, budget + 8_192) if supports_ext_think else profile.max_output_tokens

    payload: dict[str, Any] = {
        "max_tokens": min(64_000, max_tok),
        "stream": True,
        "messages": anth_msgs,
    }
    if supports_ext_think:
        payload["thinking"] = {"type": "enabled", "budget_tokens": budget}
    if is_vertex:
        payload["anthropic_version"] = "vertex-2023-10-16"
    else:
        payload["model"] = profile.model_id
    if system_prompt:
        payload["system"] = [
            {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
        ]
    if tools:
        payload["tools"] = [
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("input_schema", {"type": "object", "properties": {}}),
            }
            for t in _sorted_tools(tools)
        ]
    return payload


def build_openai_chat_payload(
    system_prompt: str,
    messages: list[CanonicalMessage],
    tools: list[dict[str, Any]],
    profile: ModelCapabilityProfile,
    session_id: str = "default",
    *,
    thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
    enable_reasoning_params: bool = True,
) -> dict[str, Any]:
    out_msgs: list[dict[str, Any]] = []
    if system_prompt:
        out_msgs.append({"role": "system", "content": system_prompt})

    for msg in messages:
        if msg.role == "system":
            out_msgs.append({"role": "system", "content": msg.plain_text()})
            continue
        if msg.role == "tool":
            for p in msg.parts:
                if p.kind == "tool_result":
                    out_msgs.append(
                        {
                            "role": "tool",
                            "tool_call_id": p.tool_call_id,
                            "content": p.text or "",
                        }
                    )
            continue
        if msg.role == "assistant":
            txt = "\n".join(p.text for p in msg.parts if p.kind == "text" and p.text)
            reasoning = "\n".join(p.text for p in msg.parts if p.kind == "thinking" and p.text)
            tcs = [
                {
                    "id": p.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": p.tool_name,
                        "arguments": canonical_json(p.tool_args or {}),
                    },
                }
                for p in msg.parts
                if p.kind == "tool_call"
            ]
            entry: dict[str, Any] = {"role": "assistant", "content": txt or ("" if tcs else " ")}
            if tcs:
                entry["tool_calls"] = tcs
            # DeepSeek / Kimi / GLM require reasoning_content field when reasoning mode is active
            if profile.reasoning_wire_mode == "reasoning_content_field":
                entry["reasoning_content"] = reasoning
            out_msgs.append(entry)
            continue

        # User turn (text + multimodal images)
        has_img = any(p.kind == "image" and p.base64_data for p in msg.parts)
        if not has_img:
            out_msgs.append({"role": "user", "content": msg.plain_text()})
        else:
            content_arr: list[dict[str, Any]] = []
            for p in msg.parts:
                if p.kind == "text" and p.text:
                    content_arr.append({"type": "text", "text": p.text})
                elif p.kind == "image" and p.base64_data:
                    mime = p.mime_type or "image/png"
                    content_arr.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{p.base64_data}"}}
                    )
            out_msgs.append({"role": "user", "content": content_arr})

    low_id = profile.model_id.lower()
    is_oai_reasoner = profile.family == "openai" and any(
        k in low_id for k in ("gpt-5", "o1", "o3", "o4", "codex")
    )
    payload: dict[str, Any] = {
        "model": profile.model_id,
        "messages": out_msgs,
        "stream": True,
    }
    if is_oai_reasoner and profile.provider == "openai":
        payload["max_completion_tokens"] = profile.max_output_tokens
        if enable_reasoning_params and thinking_effort != ThinkingEffort.OFF:
            payload["reasoning_effort"] = (
                "low"
                if thinking_effort == ThinkingEffort.LOW
                else ("medium" if thinking_effort == ThinkingEffort.MEDIUM else "high")
            )
    else:
        payload["max_tokens"] = profile.max_output_tokens
        if enable_reasoning_params and profile.provider == "openrouter" and thinking_effort != ThinkingEffort.OFF:
            payload["include_reasoning"] = True
    if profile.provider == "openai":
        payload["prompt_cache_key"] = fnv1a_64_hex(f"{session_id}:{profile.model_id}")
        payload["stream_options"] = {"include_usage": True}
    if tools:
        payload["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in _sorted_tools(tools)
        ]
    return payload


# ============================================================================
# Streaming HTTP + SSE Transport & Resilient Multi-Provider Client
# ============================================================================


def _iter_sse_json(resp: http.client.HTTPResponse) -> Iterator[dict[str, Any]]:
    """Line-buffered Server-Sent Events parser yielding JSON `data:` objects cleanly across TCP chunks."""
    data_lines: list[str] = []
    while True:
        raw_line = resp.readline()
        if not raw_line:
            break
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data_lines:
                joined = "\n".join(data_lines).strip()
                data_lines.clear()
                if joined and joined != "[DONE]":
                    try:
                        yield json.loads(joined)
                    except json.JSONDecodeError:
                        pass
            continue
        if line.startswith("data:"):
            val = line[5:].lstrip(" ")
            if val == "[DONE]":
                data_lines.clear()
                break
            data_lines.append(val)
    if data_lines:
        joined = "\n".join(data_lines).strip()
        if joined and joined != "[DONE]":
            try:
                yield json.loads(joined)
            except json.JSONDecodeError:
                pass


class HttpProviderTransport:
    """Zero-dependency TLS>=1.2 SSE streaming transport covering all 5 model ecosystems."""

    def __init__(
        self,
        vault: CredentialVault,
        timeout_seconds: float = 240.0,
        thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
    ) -> None:
        self.vault = vault
        self.timeout_seconds = timeout_seconds
        self.thinking_effort = thinking_effort

    def _open_sse(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port
        path = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        body = canonical_json(payload).encode("utf-8")

        if parsed.scheme == "https":
            conn: http.client.HTTPConnection = http.client.HTTPSConnection(
                host, port or 443, context=secure_ssl_context(), timeout=self.timeout_seconds
            )
        else:
            conn = http.client.HTTPConnection(host, port or 80, timeout=self.timeout_seconds)

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "crux-cli/0.1",
            **headers,
        }
        conn.request("POST", path, body=body, headers=req_headers)
        resp = conn.getresponse()
        if resp.status >= 400:
            err_body = resp.read(16_384).decode("utf-8", errors="replace")
            conn.close()
            raise WireError(resp.status, err_body)
        return conn, resp

    def stream_turn(
        self,
        *,
        profile: ModelCapabilityProfile,
        auth_profile: AuthProfile,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CanonicalMessage:
        token = self.vault.resolve_bearer_or_key(auth_profile)
        prov = auth_profile.provider
        if prov in ("google", "vertex"):
            return self._stream_gemini(profile, auth_profile, token, system_prompt, messages, tools, on_event)
        if prov in ("anthropic", "vertex-anthropic"):
            return self._stream_anthropic(profile, auth_profile, token, system_prompt, messages, tools, on_event)
        return self._stream_openai_compat(
            profile, auth_profile, token, system_prompt, messages, tools, session_id, on_event
        )

    def _stream_gemini(
        self,
        profile: ModelCapabilityProfile,
        auth: AuthProfile,
        token: str,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        on_event: Callable[[StreamEvent], None] | None,
    ) -> CanonicalMessage:
        is_vertex = auth.provider == "vertex"
        payload = build_gemini_payload(
            system_prompt,
            messages,
            tools,
            profile,
            thinking_effort=self.thinking_effort,
            is_vertex=is_vertex,
        )
        headers: dict[str, str] = {}

        def _open_gemini_req(body_payload: dict[str, Any]) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
            if is_vertex:
                is_g3_plus = any(profile.model_id.startswith(p) for p in ("gemini-3", "gemini-4"))
                reg = auth.region or ("global" if is_g3_plus else "us-central1")
                fallback_reg = "us-central1" if reg == "global" else "global"

                def _vertex_url(r: str) -> str:
                    host = (
                        "https://aiplatform.googleapis.com"
                        if r == "global"
                        else f"https://{r}-aiplatform.googleapis.com"
                    )
                    return (
                        f"{host}/v1/projects/{auth.project_id}/locations/{r}"
                        f"/publishers/google/models/{profile.model_id}:streamGenerateContent?alt=sse"
                    )

                headers["Authorization"] = f"Bearer {token}"
                try:
                    return self._open_sse(_vertex_url(reg), headers, body_payload)
                except WireError as exc:
                    if exc.status == 404 and not os.environ.get("GOOGLE_CLOUD_LOCATION"):
                        return self._open_sse(_vertex_url(fallback_reg), headers, body_payload)
                    raise
            url = f"{auth.base_url.rstrip('/')}/v1beta/models/{profile.model_id}:streamGenerateContent?alt=sse"
            headers["x-goog-api-key"] = token
            return self._open_sse(url, headers, body_payload)

        try:
            conn, resp = _open_gemini_req(payload)
        except WireError as exc:
            if exc.status == 400 and "thinking" in exc.detail.lower():
                fallback_payload = build_gemini_payload(
                    system_prompt,
                    messages,
                    tools,
                    profile,
                    enable_thinking_config=False,
                    is_vertex=is_vertex,
                )
                conn, resp = _open_gemini_req(fallback_payload)
            else:
                raise

        parts: list[CanonicalPart] = []
        seen_fc_ids: set[str] = set()
        text_buf: list[str] = []
        think_buf: list[str] = []
        think_sig: str | None = None
        last_sig: str | None = None
        think_started_at: float | None = None
        think_closed = False
        malformed_tool_stop = False
        usage = TokenUsage()
        stop_reason = StopReason.STOP

        def _close_thought_span() -> None:
            nonlocal think_closed
            if think_started_at is not None and not think_closed:
                think_closed = True
                if on_event:
                    ms = (time.monotonic() - think_started_at) * 1000.0
                    chars = sum(len(x) for x in think_buf)
                    on_event(
                        StreamEvent(
                            kind="thinking_end",
                            signature=think_sig or last_sig,
                            elapsed_ms=ms,
                            meta={
                                "thought_chars": chars,
                                "reasoning_tokens": usage.reasoning_tokens or max(1, chars // 4),
                                "has_signature": bool(think_sig or last_sig),
                            },
                        )
                    )

        try:
            for evt in _iter_sse_json(resp):
                um = evt.get("usageMetadata") or {}
                if um:
                    usage.input_tokens = int(um.get("promptTokenCount", usage.input_tokens))
                    usage.output_tokens = int(um.get("candidatesTokenCount", usage.output_tokens))
                    usage.cached_input_tokens = int(um.get("cachedContentTokenCount", usage.cached_input_tokens))
                    usage.reasoning_tokens = int(um.get("thoughtsTokenCount", usage.reasoning_tokens))
                for cand in evt.get("candidates") or []:
                    fr = cand.get("finishReason")
                    if fr == "MAX_TOKENS":
                        stop_reason = StopReason.MAX_TOKENS
                    elif fr in (
                        "SAFETY",
                        "RECITATION",
                        "BLOCKLIST",
                        "PROHIBITED_CONTENT",
                        "SPII",
                        "IMAGE_SAFETY",
                    ):
                        stop_reason = StopReason.CONTENT_FILTER
                    elif fr == "MISSING_THOUGHT_SIGNATURE":
                        raise WireError(400, "Gemini stream finishReason=MISSING_THOUGHT_SIGNATURE")
                    elif fr in (
                        "MALFORMED_FUNCTION_CALL",
                        "UNEXPECTED_TOOL_CALL",
                        "TOO_MANY_TOOL_CALLS",
                        "MALFORMED_RESPONSE",
                    ):
                        malformed_tool_stop = True
                    for p in (cand.get("content") or {}).get("parts") or []:
                        sig = p.get("thoughtSignature")
                        if sig:
                            last_sig = sig
                            if think_buf and not text_buf and not parts:
                                think_sig = sig
                        if p.get("thought"):
                            t_txt = str(p.get("text") or "")
                            if sig:
                                think_sig = sig
                            if t_txt:
                                if think_started_at is None:
                                    think_started_at = time.monotonic()
                                think_buf.append(t_txt)
                                if on_event:
                                    on_event(StreamEvent(kind="thinking_delta", text=t_txt, signature=sig))
                        elif "functionCall" in p:
                            _close_thought_span()
                            fc = p["functionCall"]
                            raw_fc_id = str(fc.get("id") or "").strip()
                            if not raw_fc_id or raw_fc_id in seen_fc_ids:
                                raw_fc_id = new_id("call")
                            seen_fc_ids.add(raw_fc_id)
                            tc = CanonicalPart(
                                kind="tool_call",
                                tool_call_id=raw_fc_id,
                                tool_name=str(fc.get("name", "")),
                                tool_args=dict(fc.get("args") or {}),
                                # Only attach fallback `last_sig` to the FIRST functionCall in a parallel batch!
                                signature=sig or (last_sig if not parts else None),
                            )
                            parts.append(tc)
                            if on_event:
                                on_event(StreamEvent(kind="tool_call", part=tc))
                        elif "text" in p and p["text"]:
                            _close_thought_span()
                            text_buf.append(p["text"])
                            if on_event:
                                on_event(StreamEvent(kind="text_delta", text=p["text"], signature=sig))
            _close_thought_span()
        finally:
            conn.close()

        assembled: list[CanonicalPart] = []
        if think_buf:
            assembled.append(
                CanonicalPart(kind="thinking", text="".join(think_buf), signature=think_sig or last_sig)
            )
        if text_buf:
            assembled.append(CanonicalPart(kind="text", text="".join(text_buf), signature=last_sig))
        assembled.extend(parts)
        if malformed_tool_stop and not parts:
            stop_reason = StopReason.IMPROPER_FORMAT
            if not assembled:
                assembled.append(
                    CanonicalPart(kind="text", text="[Malformed tool call discarded by Gemini stream validator]")
                )
        elif not assembled:
            stop_reason = StopReason.EMPTY_200_RESPONSE
        return CanonicalMessage(
            role="assistant",
            parts=assembled,
            producer_provider=auth.provider,
            producer_model_id=profile.model_id,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _stream_anthropic(
        self,
        profile: ModelCapabilityProfile,
        auth: AuthProfile,
        token: str,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        on_event: Callable[[StreamEvent], None] | None,
    ) -> CanonicalMessage:
        is_vertex = auth.provider == "vertex-anthropic"
        payload = build_anthropic_payload(
            system_prompt,
            messages,
            tools,
            profile,
            is_vertex=is_vertex,
            thinking_effort=self.thinking_effort,
        )
        if is_vertex:
            reg = auth.region or "us-east5"
            host = "https://aiplatform.googleapis.com" if reg == "global" else f"https://{reg}-aiplatform.googleapis.com"
            url = (
                f"{host}/v1/projects/{auth.project_id}/locations/{reg}"
                f"/publishers/anthropic/models/{profile.model_id}:streamRawPredict"
            )
            headers = {"Authorization": f"Bearer {token}"}
        else:
            url = f"{auth.base_url.rstrip('/')}/v1/messages"
            headers = {
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
            }
            if "thinking" in payload:
                headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        try:
            conn, resp = self._open_sse(url, headers, payload)
        except WireError as exc:
            if exc.status == 400 and ("thinking" in exc.detail.lower() or "beta" in exc.detail.lower()):
                headers.pop("anthropic-beta", None)
                plain_payload = build_anthropic_payload(
                    system_prompt, messages, tools, profile, is_vertex=is_vertex, enable_thinking=False
                )
                conn, resp = self._open_sse(url, headers, plain_payload)
            else:
                raise
        blocks: dict[int, dict[str, Any]] = {}
        think_t0: dict[int, float] = {}
        usage = TokenUsage()
        stop_reason = StopReason.STOP
        try:
            for evt in _iter_sse_json(resp):
                et = evt.get("type")
                if et == "message_start":
                    u = (evt.get("message") or {}).get("usage") or {}
                    usage.input_tokens += int(u.get("input_tokens", 0)) + int(u.get("cache_read_input_tokens", 0))
                    usage.cached_input_tokens += int(u.get("cache_read_input_tokens", 0))
                elif et == "content_block_start":
                    idx = int(evt.get("index", 0))
                    cb = evt.get("content_block") or {}
                    blocks[idx] = {
                        "type": cb.get("type"),
                        "id": cb.get("id"),
                        "name": cb.get("name"),
                        "buf": [],
                        "sig": "",
                        "data": cb.get("data"),
                    }
                    if cb.get("type") == "thinking":
                        think_t0[idx] = time.monotonic()
                elif et == "content_block_delta":
                    idx = int(evt.get("index", 0))
                    delta = evt.get("delta") or {}
                    b = blocks.setdefault(idx, {"type": "text", "buf": [], "sig": ""})
                    dt = delta.get("type")
                    if dt == "text_delta":
                        t = delta.get("text", "")
                        b["buf"].append(t)
                        if on_event and t:
                            on_event(StreamEvent(kind="text_delta", text=t))
                    elif dt == "thinking_delta":
                        t = delta.get("thinking", "")
                        if idx not in think_t0:
                            think_t0[idx] = time.monotonic()
                        b["buf"].append(t)
                        if on_event and t:
                            on_event(StreamEvent(kind="thinking_delta", text=t))
                    elif dt == "signature_delta":
                        b["sig"] += delta.get("signature", "")
                    elif dt == "input_json_delta":
                        b["buf"].append(delta.get("partial_json", ""))
                elif et == "content_block_stop":
                    idx = int(evt.get("index", 0))
                    b_stop = blocks.get(idx)
                    if b_stop and b_stop.get("type") == "thinking" and on_event:
                        chars = sum(len(x) for x in b_stop.get("buf", []))
                        ms = (time.monotonic() - think_t0.get(idx, time.monotonic())) * 1000.0
                        on_event(
                            StreamEvent(
                                kind="thinking_end",
                                signature=b_stop.get("sig") or None,
                                elapsed_ms=ms,
                                meta={
                                    "thought_chars": chars,
                                    "reasoning_tokens": max(1, chars // 4),
                                    "has_signature": bool(b_stop.get("sig")),
                                },
                            )
                        )
                elif et == "message_delta":
                    sr = (evt.get("delta") or {}).get("stop_reason")
                    if sr == "max_tokens":
                        stop_reason = StopReason.MAX_TOKENS
                    u = evt.get("usage") or {}
                    usage.output_tokens += int(u.get("output_tokens", 0))
        finally:
            conn.close()

        parts: list[CanonicalPart] = []
        total_think_chars = 0
        for idx in sorted(blocks):
            b = blocks[idx]
            bt = b.get("type")
            joined = "".join(b.get("buf", []))
            if bt == "thinking":
                total_think_chars += len(joined)
                parts.append(CanonicalPart(kind="thinking", text=joined, signature=b.get("sig") or None))
            elif bt == "redacted_thinking":
                parts.append(CanonicalPart(kind="thinking", redacted_data=b.get("data")))
            elif bt == "text" and joined:
                parts.append(CanonicalPart(kind="text", text=joined))
            elif bt == "tool_use":
                # Discard truncated tool call JSON on MAX_TOKENS (CT-02)
                parsed_args = parse_lenient_json_object(joined) if joined.strip() else {}
                if parsed_args is None:
                    if stop_reason == StopReason.MAX_TOKENS:
                        continue
                    parsed_args = {}
                tc = CanonicalPart(
                    kind="tool_call",
                    tool_call_id=b.get("id") or new_id("call"),
                    tool_name=b.get("name"),
                    tool_args=parsed_args,
                )
                parts.append(tc)
                if on_event:
                    on_event(StreamEvent(kind="tool_call", part=tc))

        if total_think_chars and usage.reasoning_tokens == 0:
            usage.reasoning_tokens = max(1, total_think_chars // 4)
        if not parts:
            stop_reason = StopReason.EMPTY_200_RESPONSE
        return CanonicalMessage(
            role="assistant",
            parts=parts,
            producer_provider=auth.provider,
            producer_model_id=profile.model_id,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _stream_openai_compat(
        self,
        profile: ModelCapabilityProfile,
        auth: AuthProfile,
        token: str,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        on_event: Callable[[StreamEvent], None] | None,
    ) -> CanonicalMessage:
        payload = build_openai_chat_payload(
            system_prompt, messages, tools, profile, session_id, thinking_effort=self.thinking_effort
        )
        url = f"{auth.base_url.rstrip('/')}/chat/completions"
        headers: dict[str, str] = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        if "openrouter.ai" in auth.base_url:
            headers["HTTP-Referer"] = "https://localhost/crux"
            headers["X-Title"] = "Crucible (crux)"

        try:
            conn, resp = self._open_sse(url, headers, payload)
        except WireError as exc:
            if exc.status == 400 and any(
                k in exc.detail.lower() for k in ("reasoning", "max_completion_tokens", "include_reasoning")
            ):
                plain_payload = build_openai_chat_payload(
                    system_prompt, messages, tools, profile, session_id, enable_reasoning_params=False
                )
                conn, resp = self._open_sse(url, headers, plain_payload)
            else:
                raise
        text_buf: list[str] = []
        think_buf: list[str] = []
        tc_accum: dict[int, dict[str, str]] = {}
        splitter = InlineThinkStreamSplitter()
        saw_native_reasoning = False
        think_started_at: float | None = None
        think_closed = False
        usage = TokenUsage()
        stop_reason = StopReason.STOP

        def _emit_think(chunk_txt: str) -> None:
            nonlocal think_started_at
            if not chunk_txt:
                return
            if think_started_at is None:
                think_started_at = time.monotonic()
            think_buf.append(chunk_txt)
            if on_event:
                on_event(StreamEvent(kind="thinking_delta", text=chunk_txt))

        def _close_oai_thought() -> None:
            nonlocal think_closed
            if think_started_at is not None and not think_closed:
                think_closed = True
                if on_event:
                    chars = sum(len(x) for x in think_buf)
                    on_event(
                        StreamEvent(
                            kind="thinking_end",
                            elapsed_ms=(time.monotonic() - think_started_at) * 1000.0,
                            meta={
                                "thought_chars": chars,
                                "reasoning_tokens": usage.reasoning_tokens or max(1, chars // 4),
                                "has_signature": False,
                            },
                        )
                    )

        try:
            for evt in _iter_sse_json(resp):
                u = evt.get("usage") or {}
                if u:
                    usage.input_tokens = int(u.get("prompt_tokens", usage.input_tokens))
                    usage.output_tokens = int(u.get("completion_tokens", usage.output_tokens))
                    details = u.get("prompt_tokens_details") or {}
                    usage.cached_input_tokens = int(details.get("cached_tokens", usage.cached_input_tokens))
                    c_details = u.get("completion_tokens_details") or {}
                    if "reasoning_tokens" in c_details:
                        usage.reasoning_tokens = int(c_details.get("reasoning_tokens", usage.reasoning_tokens))
                for choice in evt.get("choices") or []:
                    fr = choice.get("finish_reason")
                    if fr == "length":
                        stop_reason = StopReason.MAX_TOKENS
                    elif fr == "content_filter":
                        stop_reason = StopReason.CONTENT_FILTER
                    delta = choice.get("delta") or {}
                    rc = delta.get("reasoning_content") or delta.get("reasoning")
                    if rc:
                        saw_native_reasoning = True
                        _emit_think(str(rc))
                    txt = delta.get("content")
                    if txt:
                        if saw_native_reasoning:
                            _close_oai_thought()
                            text_buf.append(txt)
                            if on_event:
                                on_event(StreamEvent(kind="text_delta", text=txt))
                        else:
                            for ev_kind, seg in splitter.feed(str(txt)):
                                if ev_kind == "thinking_delta":
                                    _emit_think(seg)
                                else:
                                    _close_oai_thought()
                                    text_buf.append(seg)
                                    if on_event:
                                        on_event(StreamEvent(kind="text_delta", text=seg))
                    for tc in delta.get("tool_calls") or []:
                        _close_oai_thought()
                        idx = int(tc.get("index", 0))
                        slot = tc_accum.setdefault(idx, {"id": "", "name": "", "args": ""})
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
            for ev_kind, seg in splitter.flush():
                if ev_kind == "thinking_delta":
                    _emit_think(seg)
                elif seg:
                    _close_oai_thought()
                    text_buf.append(seg)
                    if on_event:
                        on_event(StreamEvent(kind="text_delta", text=seg))
            _close_oai_thought()
        finally:
            conn.close()

        if think_buf and usage.reasoning_tokens == 0:
            usage.reasoning_tokens = max(1, sum(len(x) for x in think_buf) // 4)

        parts: list[CanonicalPart] = []
        if think_buf:
            parts.append(CanonicalPart(kind="thinking", text="".join(think_buf)))
        full_text = "".join(text_buf)

        valid_tcs: list[CanonicalPart] = []
        for idx in sorted(tc_accum):
            slot = tc_accum[idx]
            raw_args = slot["args"].strip()
            parsed_args = parse_lenient_json_object(raw_args) if raw_args else {}
            if parsed_args is None:
                # Truncated tool call on MAX_TOKENS is NEVER dispatched (CT-02)
                if stop_reason == StopReason.MAX_TOKENS:
                    continue
                stop_reason = StopReason.IMPROPER_FORMAT
                continue
            tc_part = CanonicalPart(
                kind="tool_call",
                tool_call_id=slot["id"] or new_id("call"),
                tool_name=slot["name"],
                tool_args=parsed_args,
            )
            valid_tcs.append(tc_part)
            if on_event:
                on_event(StreamEvent(kind="tool_call", part=tc_part))

        # Tier-B fallback: extract `<tool_call>` XML or fenced JSON if model didn't use native tool_calls
        if not valid_tcs and full_text and stop_reason != StopReason.MAX_TOKENS:
            known = {t["name"] for t in tools} if tools else None
            cleaned_text, fallback_calls = extract_fallback_tool_calls(full_text, known)
            if fallback_calls:
                full_text = cleaned_text
                valid_tcs.extend(fallback_calls)
                for fc in fallback_calls:
                    if on_event:
                        on_event(StreamEvent(kind="tool_call", part=fc))

        if full_text:
            parts.append(CanonicalPart(kind="text", text=full_text))
        parts.extend(valid_tcs)
        if not parts:
            stop_reason = StopReason.EMPTY_200_RESPONSE
        return CanonicalMessage(
            role="assistant",
            parts=parts,
            producer_provider=auth.provider,
            producer_model_id=profile.model_id,
            stop_reason=stop_reason,
            usage=usage,
        )


@dataclass
class UniversalModelClient:
    """High-reliability wrapper handling transform_messages, Empty-200 retries, MAX_TOKENS, sig-400, & 429 failover."""

    vault: CredentialVault
    transport: ProviderTransport
    max_api_attempts: int = 3
    max_output_retries: int = 2

    def generate(
        self,
        *,
        profile: ModelCapabilityProfile,
        auth_profile: AuthProfile,
        system_prompt: str,
        history: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        id_map: ToolIdBijectionMap | None = None,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CanonicalMessage:
        strip_sigs = False
        cur_auth = auth_profile
        last_exc: Exception | None = None

        for attempt in range(self.max_api_attempts):
            wire_msgs = transform_messages(
                history, profile, id_map=id_map, strip_all_signatures=strip_sigs
            )
            try:
                msg = self.transport.stream_turn(
                    profile=profile,
                    auth_profile=cur_auth,
                    system_prompt=system_prompt,
                    messages=wire_msgs,
                    tools=tools,
                    session_id=session_id,
                    on_event=on_event,
                )
            except WireError as exc:
                last_exc = exc
                low_err = exc.detail.lower()
                if (
                    exc.status == 400
                    and (
                        "signature" in low_err
                        or "thinking" in low_err
                        or "base64" in low_err
                        or "invalid argument" in low_err
                        or "invalid_argument" in low_err
                    )
                    and not strip_sigs
                ):
                    strip_sigs = True
                    continue
                if exc.status == 429:
                    alt = self.vault.mark_rate_limited(cur_auth.name)
                    if alt and alt.name != cur_auth.name and alt.provider == cur_auth.provider:
                        cur_auth = alt
                        continue
                if exc.status in (429, 500, 502, 503, 504) and attempt + 1 < self.max_api_attempts:
                    time.sleep(min(2.0, 0.25 * (2**attempt)))
                    continue
                raise

            # Rule 7: Empty HTTP 200 stream -> automatic retry (CT-02)
            if msg.stop_reason == StopReason.EMPTY_200_RESPONSE and attempt + 1 < self.max_api_attempts:
                continue
            return msg

        if last_exc:
            raise last_exc
        return CanonicalMessage(
            role="assistant",
            parts=[CanonicalPart(kind="text", text="[Error: empty response after retries]")],
            stop_reason=StopReason.EMPTY_200_RESPONSE,
        )
