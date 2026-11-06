"""Core canonical schemas, enums, and deterministic serialization utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import time
import uuid
from typing import Any


class LoopPhase(str, Enum):
    PRE_INVOCATION = "PRE_INVOCATION"
    GENERATE = "GENERATE"
    TOOL_DISPATCH = "TOOL_DISPATCH"
    POST_INVOCATION = "POST_INVOCATION"
    STOP_HOOKS = "STOP_HOOKS"
    DONE = "DONE"


class StopReason(str, Enum):
    UNSPECIFIED = "UNSPECIFIED"
    STOP = "STOP"
    CANCELED = "CANCELED"
    MAX_TOKENS = "MAX_TOKENS"
    CONTENT_FILTER = "CONTENT_FILTER"
    IMPROPER_FORMAT = "IMPROPER_FORMAT"
    EMPTY_200_RESPONSE = "EMPTY_200_RESPONSE"
    STREAM_ERROR = "STREAM_ERROR"
    FATAL_ERROR = "FATAL_ERROR"


class ModelTier(str, Enum):
    S = "S"  # Frontier native reasoners (Gemini 4/3.x Pro, Claude Opus/Fable, GPT-5.x, o3)
    A = "A"  # Fast/open frontier coders (Gemini Flash, Sonnet, Qwen3-Coder, DeepSeek-V3/R1, Kimi-K2, GLM-5)
    B = "B"  # Compact / local / open-weight 7B-70B (Ollama, vLLM, SGLang, llama.cpp)


class Gear(str, Enum):
    AUTO = "auto"
    G1_SOLO = "g1"
    G2_STRIKE_PAIR = "g2"
    G3_SPIKE_WORKTREES = "g3"
    G4_CRUCIBLE = "g4"


class PermissionMode(str, Enum):
    PLAN = "plan"          # Read-only exploration & specification
    DEFAULT = "default"    # Parallel safe reads; prompt on edits & shell mutations
    AUTO_EDIT = "auto_edit"  # Auto-approve workspace edits; prompt on risky shell commands
    YOLO = "yolo"          # Explicit autonomous execution (hard-gated against catastrophic rules)


class PolicyDecision(str, Enum):
    ALLOW = "allow"
    ASK_USER = "ask_user"
    DENY = "deny"


class ThinkingEffort(str, Enum):
    """Universal cross-provider reasoning depth knob (`off | low | medium | high | max`)."""

    OFF = "off"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MAX = "max"


class TraceMode(str, Enum):
    """Interactive & headless thought-trace verbosity (`full | compact | off`)."""

    FULL = "full"        # Stream verbatim chain-of-thought + step/tool timings & observations
    COMPACT = "compact"  # Show 1-line thought summary/token count + tool timings (full thoughts still saved to disk)
    OFF = "off"          # Print final response text only (full thoughts still saved to disk)


@dataclass(frozen=True)
class ModelCapabilityProfile:
    model_id: str
    provider: str  # google | vertex | anthropic | vertex-anthropic | openai | openrouter | local
    family: str    # gemini | anthropic | openai | qwen | deepseek | kimi | glm | minimax | local_oss
    tier: ModelTier
    context_window_tokens: int = 200_000
    max_output_tokens: int = 16_384
    supports_vision: bool = True
    tool_calling_mode: str = "native_strict"  # native_strict | native_loose | xml_fenced_fallback
    reasoning_wire_mode: str = "none"
    # gemini_thought_sig | anthropic_signed_thinking | openai_encrypted | reasoning_content_field | inline_think_tags | none
    prompt_cache_ttl_seconds: int = 300
    max_parallel_read_tools: int = 8
    compaction_ratio: float = 0.85
    tool_preview_bytes: int = 16_384
    skill_catalog_budget_tokens: int = 2_500


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    def add(self, other: TokenUsage) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.reasoning_tokens += other.reasoning_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return min(1.0, self.cached_input_tokens / self.input_tokens)


@dataclass
class CanonicalPart:
    kind: str  # text | thinking | image | tool_call | tool_result
    text: str = ""
    # Thinking fields
    signature: str | None = None
    redacted_data: str | None = None
    # Image fields
    mime_type: str | None = None
    base64_data: str | None = None
    alt_text: str | None = None
    # Tool call / result fields
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None
    is_error: bool = False
    is_dynamic_instruction: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"kind": self.kind}
        if self.text:
            d["text"] = self.text
        if self.signature is not None:
            d["signature"] = self.signature
        if self.redacted_data is not None:
            d["redacted_data"] = self.redacted_data
        if self.mime_type is not None:
            d["mime_type"] = self.mime_type
        if self.base64_data is not None:
            d["base64_data"] = self.base64_data
        if self.alt_text is not None:
            d["alt_text"] = self.alt_text
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.tool_args is not None:
            d["tool_args"] = self.tool_args
        if self.is_error:
            d["is_error"] = True
        if self.is_dynamic_instruction:
            d["is_dynamic_instruction"] = True
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CanonicalPart:
        return cls(
            kind=str(d["kind"]),
            text=str(d.get("text", "")),
            signature=d.get("signature"),
            redacted_data=d.get("redacted_data"),
            mime_type=d.get("mime_type"),
            base64_data=d.get("base64_data"),
            alt_text=d.get("alt_text"),
            tool_call_id=d.get("tool_call_id"),
            tool_name=d.get("tool_name"),
            tool_args=d.get("tool_args"),
            is_error=bool(d.get("is_error", False)),
            is_dynamic_instruction=bool(d.get("is_dynamic_instruction", False)),
        )


def new_id(prefix: str = "msg") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class CanonicalMessage:
    role: str  # system | user | assistant | tool
    parts: list[CanonicalPart]
    entry_id: str = field(default_factory=lambda: new_id("node"))
    parent_entry_id: str | None = None
    producer_provider: str | None = None
    producer_model_id: str | None = None
    stop_reason: StopReason = StopReason.STOP
    created_at: float = field(default_factory=time.time)
    usage: TokenUsage | None = None
    modified_files: list[str] = field(default_factory=list)

    def plain_text(self) -> str:
        return "\n".join(p.text for p in self.parts if p.kind == "text" and p.text)

    def thinking_text(self) -> str:
        return "\n\n".join(p.text for p in self.parts if p.kind == "thinking" and p.text)

    def tool_calls(self) -> list[CanonicalPart]:
        return [p for p in self.parts if p.kind == "tool_call"]

    def approx_tokens(self) -> int:
        total_chars = 0
        for p in self.parts:
            total_chars += len(p.text)
            if p.tool_args is not None:
                total_chars += len(canonical_json(p.tool_args))
            if p.base64_data:
                total_chars += 1200  # ~300 image patch tokens
        return max(1, total_chars // 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "parent_entry_id": self.parent_entry_id,
            "role": self.role,
            "parts": [p.to_dict() for p in self.parts],
            "producer_provider": self.producer_provider,
            "producer_model_id": self.producer_model_id,
            "stop_reason": self.stop_reason.value,
            "created_at": self.created_at,
            "modified_files": list(self.modified_files),
            "usage": (
                {
                    "input_tokens": self.usage.input_tokens,
                    "output_tokens": self.usage.output_tokens,
                    "cached_input_tokens": self.usage.cached_input_tokens,
                    "reasoning_tokens": self.usage.reasoning_tokens,
                }
                if self.usage
                else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CanonicalMessage:
        u = d.get("usage")
        return cls(
            entry_id=str(d.get("entry_id") or new_id("node")),
            parent_entry_id=d.get("parent_entry_id"),
            role=str(d["role"]),
            parts=[CanonicalPart.from_dict(x) for x in d.get("parts", [])],
            producer_provider=d.get("producer_provider"),
            producer_model_id=d.get("producer_model_id"),
            stop_reason=StopReason(d.get("stop_reason", StopReason.STOP.value)),
            created_at=float(d.get("created_at", time.time())),
            modified_files=list(d.get("modified_files", [])),
            usage=TokenUsage(**u) if isinstance(u, dict) else None,
        )


@dataclass
class StreamEvent:
    """Real-time streaming & long-form cognitive trace event."""

    # Core kinds: turn_start | step_start | thinking_delta | thinking_end | text_delta |
    #             tool_call | tool_result | cognitive_event | step_end | usage | done
    kind: str
    text: str = ""
    signature: str | None = None
    part: CanonicalPart | None = None
    usage: TokenUsage | None = None
    stop_reason: StopReason = StopReason.STOP
    step: int = 0
    elapsed_ms: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


def canonical_json(obj: Any) -> str:
    """Deterministic sorted-key JSON serialization preserving prefix KV-cache bytes."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fnv1a_64_hex(text: str) -> str:
    """Deterministic FNV-1a 64-bit hash for sticky prompt-cache replica routing."""
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def sha256_hex(data: bytes | str) -> str:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()
