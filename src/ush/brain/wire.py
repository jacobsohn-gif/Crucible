"""Universal Multi-Provider Wire Normalizer (`transform_messages`) & Tier-B Tool Repair.

Implements all 8 cross-provider wire normalization laws (CT-01 .. CT-04):
1. Same-model cryptographic thinking / `thoughtSignature` preservation.
2. Foreign thinking & signature scrubbing on live model rotation / failover / swarm handoff.
3. Bijective <=64-char `^[a-zA-Z0-9_-]{1,64}$` tool-call ID normalization across all turns.
4. Multimodal vision downgrade to explicit notices + Anthropic error-image hoisting.
5. Aborted/errored assistant turn pruning & mid-tool-batch dynamic instruction holdback.
6. Tail-only `<dynamic_instructions>` erasure so historical prefix bytes remain 100% KV-cache stable.
7. Econometric TTL cache-warming calculator (`E[Savings] = P(cont)*C_miss - C_warm`).
8. Forgiving `<tool_call>` / fenced JSON / trailing-comma repair parser for Tier-B models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    ModelCapabilityProfile,
    StopReason,
    new_id,
)

SAFE_TOOL_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
DYNAMIC_TAG_RE = re.compile(r"<dynamic_instructions>.*?</dynamic_instructions>\s*", re.DOTALL)
XML_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
FENCED_JSON_RE = re.compile(r"```(?:json|tool_call)\s*(\{.*?\})\s*```", re.DOTALL)
TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


@dataclass
class ToolIdBijectionMap:
    """Deterministically maps arbitrary provider tool IDs (e.g. 450-char OpenAI pipe IDs) to <=64-char safe IDs."""

    raw_to_safe: dict[str, str] = field(default_factory=dict)
    safe_to_raw: dict[str, str] = field(default_factory=dict)

    def normalize(self, raw_id: str | None) -> str:
        raw = raw_id or new_id("call")
        if raw in self.raw_to_safe:
            return self.raw_to_safe[raw]
        if SAFE_TOOL_ID_RE.match(raw):
            safe = raw
        else:
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]
            prefix = re.sub(r"[^a-zA-Z0-9_-]", "", raw)[:16] or "call"
            safe = f"{prefix}_{digest}"[:64]
        self.raw_to_safe[raw] = safe
        self.safe_to_raw[safe] = raw
        return safe


def wrap_dynamic_instructions(body: str) -> CanonicalPart:
    return CanonicalPart(
        kind="text",
        text=f"<dynamic_instructions>\n{body.strip()}\n</dynamic_instructions>",
        is_dynamic_instruction=True,
    )


_SYNTHETIC_TOOL_REPAIR_TEXT = (
    "[SYNTHETIC_REPAIR: prior tool invocation interrupted before result was recorded]"
)


def _normalize_non_tool_part(
    p: CanonicalPart,
    *,
    is_final_user_frame: bool,
    same_model: bool,
    target_profile: ModelCapabilityProfile,
) -> CanonicalPart | None:
    if p.kind == "thinking":
        if same_model:
            return CanonicalPart(**p.to_dict())
        # Foreign thinking/signatures are strictly dropped (never converted to assistant text!)
        return None
    if p.kind == "text":
        if p.is_dynamic_instruction and not is_final_user_frame:
            return None
        txt = p.text if is_final_user_frame else DYNAMIC_TAG_RE.sub("", p.text)
        sig = p.signature if same_model else None
        if not txt.strip() and not sig:
            return None
        return CanonicalPart(kind="text", text=txt, signature=sig)
    if p.kind == "image":
        if not target_profile.supports_vision:
            label = p.alt_text or p.mime_type or "image"
            return CanonicalPart(
                kind="text",
                text=f"[Image omitted ({label}): target model `{target_profile.model_id}` does not support vision]",
            )
        return CanonicalPart(**p.to_dict())
    return None


def _normalize_tool_result_part(
    p: CanonicalPart,
    *,
    safe_id: str,
    expected_name: str,
    target_profile: ModelCapabilityProfile,
    hoisted_image_captions: list[CanonicalPart],
) -> CanonicalPart:
    resolved_name = expected_name or p.tool_name or "tool"
    # Rule 5: Anthropic/Bedrock forbid images inside `is_error=True` tool results
    if p.is_error and p.base64_data and target_profile.family == "anthropic":
        if target_profile.supports_vision:
            hoisted_image_captions.append(
                CanonicalPart(
                    kind="image",
                    mime_type=p.mime_type or "image/png",
                    base64_data=p.base64_data,
                    alt_text=f"Diagnostic image from errored tool {safe_id}",
                )
            )
        return CanonicalPart(
            kind="tool_result",
            tool_call_id=safe_id,
            tool_name=resolved_name,
            text=p.text or "[Tool failed; image moved to trailing user frame]",
            is_error=True,
        )
    return CanonicalPart(
        kind="tool_result",
        tool_call_id=safe_id,
        tool_name=resolved_name,
        text=p.text,
        mime_type=p.mime_type if target_profile.supports_vision else None,
        base64_data=p.base64_data if target_profile.supports_vision else None,
        is_error=p.is_error,
    )


def _clone_with_parts(msg: CanonicalMessage, parts: list[CanonicalPart], *, role: str | None = None) -> CanonicalMessage:
    return CanonicalMessage(
        role=role or msg.role,
        parts=parts,
        entry_id=msg.entry_id,
        parent_entry_id=msg.parent_entry_id,
        producer_provider=msg.producer_provider,
        producer_model_id=msg.producer_model_id,
        stop_reason=msg.stop_reason,
        created_at=msg.created_at,
    )


def transform_messages(
    history: list[CanonicalMessage],
    target_profile: ModelCapabilityProfile,
    id_map: ToolIdBijectionMap | None = None,
    strip_all_signatures: bool = False,
) -> list[CanonicalMessage]:
    """Produces a clean, provider-safe, KV-cache-stable message sequence enforcing `INV-WIRE-BIJECTION`.

    `INV-WIRE-BIJECTION`: Every `assistant` turn with K >= 1 `tool_call` parts `[c_1..c_K]` is IMMEDIATELY
    followed by a single `role="tool"` turn with EXACTLY K `tool_result` parts `[r_1..r_K]` in identical
    positional order (`r_k.tool_call_id == c_k.tool_call_id` and `r_k.tool_name == c_k.tool_name`), and
    stray/orphan/duplicate `tool_result` frames never leak to any provider wire.
    """
    id_map = id_map or ToolIdBijectionMap()

    # Step 1: Prune aborted/fatal incomplete assistant turns without matching tool results
    filtered: list[CanonicalMessage] = []
    for msg in history:
        if msg.role == "assistant" and msg.stop_reason in (StopReason.CANCELED, StopReason.FATAL_ERROR):
            if not msg.plain_text() and not msg.tool_calls():
                continue
            if msg.stop_reason == StopReason.CANCELED and msg.tool_calls():
                continue
        filtered.append(msg)

    # Identify index of the final user/tool frame allowed to carry `<dynamic_instructions>`
    last_user_like_idx = -1
    for idx in range(len(filtered) - 1, -1, -1):
        if filtered[idx].role in ("user", "tool"):
            last_user_like_idx = idx
            break

    # Step 2: Enforce strict per-batch 1-to-1 ordered `tool_call <-> tool_result` bijection,
    # scrub foreign thinking/signatures, remap tool IDs, downgrade/hoist images, and erase old dynamic tags.
    normalized: list[CanonicalMessage] = []
    n_msgs = len(filtered)
    idx = 0

    while idx < n_msgs:
        msg = filtered[idx]
        is_final_user_frame = idx == last_user_like_idx
        same_model = (
            not strip_all_signatures
            and msg.producer_model_id == target_profile.model_id
            and msg.producer_provider == target_profile.provider
        )

        # Stray/orphan `role="tool"` message without an immediately preceding open `assistant` tool batch:
        # drop orphan `tool_result` parts so Gemini/Anthropic/OpenAI never see an unanchored response turn.
        if msg.role == "tool":
            salvaged: list[CanonicalPart] = []
            for p in msg.parts:
                if p.kind != "tool_result":
                    np = _normalize_non_tool_part(
                        p,
                        is_final_user_frame=is_final_user_frame,
                        same_model=False,
                        target_profile=target_profile,
                    )
                    if np is not None:
                        salvaged.append(np)
            if salvaged:
                normalized.append(_clone_with_parts(msg, salvaged, role="user"))
            idx += 1
            continue

        if msg.role != "assistant":
            out_parts = [
                np
                for p in msg.parts
                if p.kind not in ("tool_call", "tool_result")
                and (
                    np := _normalize_non_tool_part(
                        p,
                        is_final_user_frame=is_final_user_frame,
                        same_model=same_model,
                        target_profile=target_profile,
                    )
                )
                is not None
            ]
            if out_parts:
                normalized.append(_clone_with_parts(msg, out_parts))
            idx += 1
            continue

        # `msg.role == "assistant"`
        out_parts = []
        batch_calls: list[tuple[str, str]] = []
        seen_in_batch: set[str] = set()
        raw_alias_in_batch: dict[str, list[str]] = {}

        for p in msg.parts:
            if p.kind == "tool_call":
                base_safe = id_map.normalize(p.tool_call_id)
                safe_id = base_safe
                if safe_id in seen_in_batch:
                    safe_id = f"{base_safe[:56]}_d{len(batch_calls)}"
                seen_in_batch.add(safe_id)
                if p.tool_call_id:
                    raw_alias_in_batch.setdefault(p.tool_call_id, []).append(safe_id)
                raw_alias_in_batch.setdefault(base_safe, []).append(safe_id)
                t_name = p.tool_name or "tool"
                batch_calls.append((safe_id, t_name))
                sig = p.signature if same_model else None
                out_parts.append(
                    CanonicalPart(
                        kind="tool_call",
                        tool_call_id=safe_id,
                        tool_name=t_name,
                        tool_args=dict(p.tool_args or {}),
                        signature=sig,
                    )
                )
            elif p.kind != "tool_result":
                np = _normalize_non_tool_part(
                    p,
                    is_final_user_frame=False,
                    same_model=same_model,
                    target_profile=target_profile,
                )
                if np is not None:
                    out_parts.append(np)

        if out_parts:
            normalized.append(_clone_with_parts(msg, out_parts))

        if not batch_calls:
            idx += 1
            continue

        # Look ahead strictly up to the next `assistant`/`system` turn to find tool results belonging to THIS batch.
        expected_ids = {cid for cid, _ in batch_calls}
        scan_end = idx + 1
        while scan_end < n_msgs and filtered[scan_end].role not in ("assistant", "system"):
            scan_end += 1

        def _match_safe_id(raw_tid: str | None, already_taken: set[str]) -> str | None:
            candidates: list[str] = []
            if raw_tid and raw_tid in raw_alias_in_batch:
                candidates.extend(raw_alias_in_batch[raw_tid])
            norm_tid = id_map.normalize(raw_tid) if raw_tid else ""
            if norm_tid in raw_alias_in_batch:
                for c in raw_alias_in_batch[norm_tid]:
                    if c not in candidates:
                        candidates.append(c)
            elif norm_tid in expected_ids:
                candidates.append(norm_tid)
            for cand in candidates:
                if cand in expected_ids and cand not in already_taken:
                    return cand
            return None

        last_matching_tool_j = -1
        probe_taken: set[str] = set()
        for j in range(idx + 1, scan_end):
            cand_msg = filtered[j]
            if cand_msg.role in ("tool", "user"):
                matched_here = False
                for p in cand_msg.parts:
                    if p.kind == "tool_result":
                        m_id = _match_safe_id(p.tool_call_id, probe_taken)
                        if m_id is not None:
                            probe_taken.add(m_id)
                            matched_here = True
                if matched_here:
                    last_matching_tool_j = j

        if last_matching_tool_j == -1:
            # Interrupted/orphan `tool_call` batch: close immediately in place with exact count & exact tool names!
            repairs = [
                CanonicalPart(
                    kind="tool_result",
                    tool_call_id=cid,
                    tool_name=cname,
                    text=_SYNTHETIC_TOOL_REPAIR_TEXT,
                    is_error=True,
                )
                for cid, cname in batch_calls
            ]
            normalized.append(CanonicalMessage(role="tool", parts=repairs))
            idx += 1
            continue

        matched_parts: dict[str, CanonicalPart] = {}
        taken_ids: set[str] = set()
        deferred_user_msgs: list[CanonicalMessage] = []
        hoisted_image_captions: list[CanonicalPart] = []
        rep_tool_msg: CanonicalMessage | None = None

        for j in range(idx + 1, last_matching_tool_j + 1):
            sub = filtered[j]
            sub_is_final = j == last_user_like_idx
            leftover_user_parts: list[CanonicalPart] = []
            for p in sub.parts:
                if p.kind == "tool_result":
                    m_id = _match_safe_id(p.tool_call_id, taken_ids)
                    if m_id is not None:
                        taken_ids.add(m_id)
                        matched_parts[m_id] = p
                        if rep_tool_msg is None and sub.role == "tool":
                            rep_tool_msg = sub
                elif p.kind != "tool_call":
                    np = _normalize_non_tool_part(
                        p,
                        is_final_user_frame=sub_is_final,
                        same_model=False,
                        target_profile=target_profile,
                    )
                    if np is not None:
                        leftover_user_parts.append(np)
            if leftover_user_parts:
                deferred_user_msgs.append(_clone_with_parts(sub, leftover_user_parts, role="user"))

        ordered_results: list[CanonicalPart] = []
        for cid, cname in batch_calls:
            raw_tr = matched_parts.get(cid)
            if raw_tr is not None:
                ordered_results.append(
                    _normalize_tool_result_part(
                        raw_tr,
                        safe_id=cid,
                        expected_name=cname,
                        target_profile=target_profile,
                        hoisted_image_captions=hoisted_image_captions,
                    )
                )
            else:
                ordered_results.append(
                    CanonicalPart(
                        kind="tool_result",
                        tool_call_id=cid,
                        tool_name=cname,
                        text=_SYNTHETIC_TOOL_REPAIR_TEXT,
                        is_error=True,
                    )
                )

        if rep_tool_msg is not None:
            normalized.append(_clone_with_parts(rep_tool_msg, ordered_results, role="tool"))
        else:
            normalized.append(CanonicalMessage(role="tool", parts=ordered_results))

        if hoisted_image_captions:
            normalized.append(
                CanonicalMessage(
                    role="user",
                    parts=[
                        CanonicalPart(kind="text", text="[Attached diagnostic screenshot from tool error:]"),
                        *hoisted_image_captions,
                    ],
                )
            )
        normalized.extend(deferred_user_msgs)
        idx = last_matching_tool_j + 1

    # Rule 9: Enforce Start & End Role Constraints across all providers:
    # - First non-system turn MUST be `role == "user"` (`historyHardening.ts:347-359`).
    # - Final turn MUST NOT be `role == "assistant"` (Gemini 3/4 & Anthropic reject model-ending requests with HTTP 400).
    first_conv_idx = next((i for i, m in enumerate(normalized) if m.role != "system"), -1)
    if first_conv_idx != -1 and normalized[first_conv_idx].role == "assistant":
        normalized.insert(
            first_conv_idx,
            CanonicalMessage(
                role="user",
                parts=[CanonicalPart(kind="text", text="[Continuing conversation...]")],
            ),
        )

    if normalized and normalized[-1].role == "assistant":
        normalized.append(
            CanonicalMessage(
                role="user",
                parts=[CanonicalPart(kind="text", text="Continue.")],
            )
        )

    return normalized


def parse_lenient_json_object(raw: str) -> dict[str, Any] | None:
    """Repairs common Tier-B model JSON quirks (trailing commas, outer stringification)."""
    cleaned = raw.strip()
    for candidate in (cleaned, TRAILING_COMMA_RE.sub(r"\1", cleaned)):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def extract_fallback_tool_calls(
    text: str, known_tool_names: set[str] | None = None
) -> tuple[str, list[CanonicalPart]]:
    """Extracts `<tool_call>{...}</tool_call>` or fenced JSON tool invocations for Tier-B models (CT-04)."""
    calls: list[CanonicalPart] = []
    spans_to_strip: list[tuple[int, int]] = []

    for regex in (XML_TOOL_CALL_RE, FENCED_JSON_RE):
        for m in regex.finditer(text):
            obj = parse_lenient_json_object(m.group(1))
            if not obj:
                continue
            name = obj.get("name") or obj.get("tool") or obj.get("function")
            args = obj.get("arguments") if "arguments" in obj else obj.get("args", obj.get("parameters", {}))
            if isinstance(args, str):
                args = parse_lenient_json_object(args) or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            if known_tool_names is not None and name not in known_tool_names:
                continue
            calls.append(
                CanonicalPart(
                    kind="tool_call",
                    tool_call_id=new_id("call"),
                    tool_name=name,
                    tool_args=args,
                )
            )
            spans_to_strip.append(m.span())
        if calls:
            break

    if not spans_to_strip:
        return text, []

    out: list[str] = []
    cursor = 0
    for start, end in sorted(spans_to_strip):
        out.append(text[cursor:start])
        cursor = end
    out.append(text[cursor:])
    return "".join(out).strip(), calls


def should_warm_prompt_cache(
    cached_prefix_tokens: int,
    continuation_prob: float,
    price_per_million_uncached: float = 3.0,
    cache_read_discount: float = 0.90,
    min_expected_savings_usd: float = 0.05,
) -> tuple[bool, float]:
    """Econometric KV-cache TTL keepalive rule: E[Savings] = P(cont)*C_miss - C_warm >= $0.05."""
    tokens_m = cached_prefix_tokens / 1_000_000.0
    cost_miss = tokens_m * price_per_million_uncached * cache_read_discount
    cost_warm = tokens_m * price_per_million_uncached * (1.0 - cache_read_discount)
    expected_savings = (continuation_prob * cost_miss) - cost_warm
    return expected_savings >= min_expected_savings_usd, expected_savings
