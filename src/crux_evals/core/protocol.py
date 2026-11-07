"""Immutable I/O Contracts, Complementary Navigation Schema & Epistemic Firewall Protocol.

Architectural Principles:
1. Zero Self-Grading (Epistemic Firewall):
   The Coding Agent (Actor) NEVER grades its own work. Ground-truth workspace state
   (unified git diffs, AST diagnostics, file mtimes, and raw command logs) is hydrated
   directly from disk by the MCP Server or Harness Telemetry Store.
2. Complementary Navigation by Default (Compass, Not Police):
   Evaluators act as independent external instruments that surface `blind_spots`,
   `counterexample` reproducers, `unverified_claims`, and `suggested_next_probes`
   to help the agent navigate complex problems without unnecessarily locking its tools.
3. Pluggable Extensibility:
   Any class implementing `EvaluatorProtocol` (or subclassing `BaseEvaluator`) with an
   `evaluator.yaml` manifest is auto-discovered by both the MCP Server and Cascade Runner.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import pathlib
import re
from typing import Any, Callable, Literal, Optional, Protocol, runtime_checkable

HookEvent = Literal["BeforeTool", "AfterTool", "Stop", "HypothesisCheckpoint", "MCPConsult"]
EvaluatorParadigm = Literal["deterministic", "llm_judge", "adversarial"]
Decision = Literal["ALLOW", "WARN", "VETO", "REWIND"]
EnforcementMode = Literal["complementary", "strict"]

# Pluggable LLM judge callable: (system_prompt, user_prompt, model_family_hint) -> raw_response
LLMJudgeCallable = Callable[[str, str, Optional[str]], str]


def sha256_short(text: str, length: int = 16) -> str:
    """Deterministic hex digest used for action, diff, and error fingerprinting."""
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:length]


@dataclass(frozen=True)
class ToolRecord:
    """Ground-truth record of a single tool invocation captured by the Harness/Server."""

    step_index: int
    tool_name: str
    tool_args: dict[str, Any]
    output_raw: str
    is_error: bool
    exit_code: int | None = None
    action_signature: str = ""
    error_signature: str = ""
    diff_state_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolRecord:
        return cls(
            step_index=int(data.get("step_index", 0)),
            tool_name=str(data.get("tool_name", "")),
            tool_args=dict(data.get("tool_args") or {}),
            output_raw=str(data.get("output_raw", "")),
            is_error=bool(data.get("is_error", False)),
            exit_code=data.get("exit_code"),
            action_signature=str(data.get("action_signature", "")),
            error_signature=str(data.get("error_signature", "")),
            diff_state_hash=str(data.get("diff_state_hash", "")),
        )


@dataclass(frozen=True)
class EvaluationFrame:
    """Immutable ground-truth frame hydrated from workspace state and passed to an Evaluator.

    Epistemic Firewall Rule:
        For Provenance-Blind Evaluators (`requires_blind_context=True`), `with_epistemic_firewall()`
        strips `actor_claim_text` (`None`), preventing confirmation bias and sycophancy.
    """

    session_id: str
    workspace_root: str
    event: HookEvent
    step_index: int

    # 1. Task Specification & Declared Invariants
    task_objective: str = ""
    declared_invariants: tuple[str, ...] = ()

    # 2. Current Tool Event (populated on BeforeTool / AfterTool)
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_output_raw: str = ""
    tool_is_error: bool = False
    tool_exit_code: int | None = None

    # 3. Ground-Truth Workspace & Git State (hydrated from disk, never Actor CoT)
    unified_git_diff: str = ""
    modified_files: tuple[str, ...] = ()
    read_files: tuple[str, ...] = ()

    # 4. Trajectory & Verification Telemetry
    tool_history: tuple[ToolRecord, ...] = ()
    last_edit_step: int = 0
    last_verification_step: int = 0
    last_verification_passed: bool | None = None
    last_verification_output: str = ""

    # 5. Quarantined Actor Output (Stripped for Blind Judges!)
    actor_claim_text: str | None = None
    actor_model_family: str | None = None

    @property
    def ran_verification_since_last_edit(self) -> bool:
        """True iff no code edits occurred, or a verification command passed after the latest edit."""
        if self.last_edit_step == 0:
            return True
        return (
            self.last_verification_step > self.last_edit_step
            and self.last_verification_passed is True
        )

    def with_epistemic_firewall(self) -> EvaluationFrame:
        """Returns a copy with Actor Chain-of-Thought and reasoning strictly stripped."""
        return replace(self, actor_claim_text=None)


@dataclass
class EvaluationVerdict:
    """Standardized Complementary Navigation & Verification Verdict emitted by any Evaluator."""

    evaluator_id: str
    paradigm: EvaluatorParadigm
    decision: Decision  # ALLOW | WARN | VETO | REWIND

    # 1. Navigation & BARS Calibration Scores
    bars_score: float = 1.0            # [0.0, 1.0] Behaviorally Anchored alignment score
    progress_advantage: float = 0.0    # [-1.0, +1.0] Step-level Process Advantage (PAV delta)
    uncertainty: float = 0.0           # [0.0, 1.0] Epistemic uncertainty

    # 2. Grounded Evidence & Rubric Breakdown
    criterion_scores: dict[str, float] = field(default_factory=dict)
    hard_vetoes_triggered: list[str] = field(default_factory=list)
    evidence_citations: list[str] = field(default_factory=list)
    localized_step_index: int | None = None
    counterexample: str | None = None
    remediation_directive: str = ""

    # 3. Complementary Navigation Insights (Value-Add for Agentic Problem Solving)
    blind_spots: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    suggested_next_probes: list[str] = field(default_factory=list)

    @property
    def alignment_score(self) -> float:
        return self.bars_score

    @property
    def progress_delta(self) -> float:
        return self.progress_advantage

    @property
    def passed(self) -> bool:
        return self.decision in ("ALLOW", "WARN") and not self.hard_vetoes_triggered

    def format_harness_feedback(self, *, mode: EnforcementMode = "complementary") -> str:
        """Formats the verdict for injection into `<dynamic_instructions>` or MCP tool output."""
        status_label = (
            "CLEAN"
            if self.decision == "ALLOW"
            else ("INSIGHTS_FOUND" if mode == "complementary" else self.decision)
        )
        header = (
            f"[CRUX-EVAL:{self.evaluator_id} ({self.paradigm.upper()}) -> {status_label} "
            f"| alignment={self.bars_score:.2f} progress_delta={self.progress_advantage:+.2f}]"
        )
        lines = [header]
        if self.hard_vetoes_triggered:
            lines.append(f"Critical Signals: {', '.join(self.hard_vetoes_triggered)}")
        if self.criterion_scores:
            scores_str = ", ".join(f"{k}={v:.2f}" for k, v in self.criterion_scores.items())
            lines.append(f"BARS Rubric: {scores_str} (aggregate={self.bars_score:.2f})")
        if self.blind_spots:
            lines.append("Blind Spots Identified:")
            for bs in self.blind_spots[:5]:
                lines.append(f"  - {bs}")
        if self.unverified_claims:
            lines.append("Unverified Claims / Assumptions:")
            for uc in self.unverified_claims[:5]:
                lines.append(f"  - {uc}")
        if self.evidence_citations:
            lines.append("Grounded Evidence:")
            for cit in self.evidence_citations[:5]:
                lines.append(f"  - {cit}")
        if self.counterexample:
            lines.append(f"Falsifying Counterexample:\n{self.counterexample}")
        if self.suggested_next_probes:
            lines.append("Suggested Next Probes:")
            for pr in self.suggested_next_probes[:5]:
                lines.append(f"  - {pr}")
        if self.remediation_directive:
            lines.append(f"Recommended Next Action: {self.remediation_directive}")
        return "\n".join(lines)

    def to_mcp_payload(self) -> dict[str, Any]:
        """Structured Complementary Feedback JSON returned by MCP `tools/call`."""
        status = "CLEAN" if self.decision == "ALLOW" else "INSIGHTS_FOUND"
        return {
            "evaluator": self.evaluator_id,
            "paradigm": self.paradigm,
            "status": status,
            "decision": self.decision,
            "alignment_score": round(self.bars_score, 4),
            "progress_delta": round(self.progress_advantage, 4),
            "uncertainty": round(self.uncertainty, 4),
            "criterion_scores": self.criterion_scores,
            "blind_spots": self.blind_spots or self.evidence_citations,
            "unverified_claims": self.unverified_claims,
            "counterexamples": [self.counterexample] if self.counterexample else [],
            "suggested_next_probes": (
                self.suggested_next_probes
                or ([self.remediation_directive] if self.remediation_directive else [])
            ),
            "hard_vetoes_triggered": self.hard_vetoes_triggered,
            "remediation_directive": self.remediation_directive,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CascadeReport:
    """Aggregated result across the multi-stage Evaluator Cascade."""

    event: HookEvent
    final_decision: Decision
    min_bars_score: float
    conservative_lcb_score: float
    cumulative_progress_advantage: float
    verdicts: list[EvaluationVerdict] = field(default_factory=list)
    short_circuited_at: str | None = None

    @property
    def blocked(self) -> bool:
        return self.final_decision in ("VETO", "REWIND")

    def combined_feedback(self, *, mode: EnforcementMode = "complementary") -> str:
        """Returns formatted feedback from all non-trivial verdicts."""
        relevant = [
            v.format_harness_feedback(mode=mode)
            for v in self.verdicts
            if v.decision != "ALLOW" or v.remediation_directive or v.blind_spots or v.suggested_next_probes
        ]
        return "\n\n".join(relevant)

    def to_mcp_payload(self) -> dict[str, Any]:
        """Serializes the full cascade into an open MCP JSON response."""
        all_blind_spots: list[str] = []
        all_unverified: list[str] = []
        all_counterexamples: list[str] = []
        all_probes: list[str] = []
        for v in self.verdicts:
            for bs in (v.blind_spots or v.evidence_citations):
                if bs not in all_blind_spots:
                    all_blind_spots.append(bs)
            for uc in v.unverified_claims:
                if uc not in all_unverified:
                    all_unverified.append(uc)
            if v.counterexample and v.counterexample not in all_counterexamples:
                all_counterexamples.append(v.counterexample)
            for pr in (v.suggested_next_probes or ([v.remediation_directive] if v.remediation_directive else [])):
                if pr and pr not in all_probes:
                    all_probes.append(pr)
        return {
            "status": "CLEAN" if self.final_decision == "ALLOW" else "INSIGHTS_FOUND",
            "final_decision": self.final_decision,
            "alignment_score": self.min_bars_score,
            "conservative_lcb_score": self.conservative_lcb_score,
            "progress_delta": self.cumulative_progress_advantage,
            "short_circuited_at": self.short_circuited_at,
            "blind_spots": all_blind_spots,
            "unverified_claims": all_unverified,
            "counterexamples": all_counterexamples,
            "suggested_next_probes": all_probes,
            "verdicts": [v.to_mcp_payload() for v in self.verdicts],
        }

    def to_hook_output(self, *, mode: EnforcementMode = "complementary") -> tuple[int, dict[str, Any]]:
        """Serializes into Universal CLI Hook exit code + stdout JSON (`crux` / Claude / Gemini).

        - In `complementary` mode (default): NEVER blocks the agent's tools (`exit 0`);
          injects navigational telemetry via `additionalContext` whenever insights exist.
        - In `strict` mode: Exit code 2 + `decision: block` triggers an automatic Harness Veto
          when `self.blocked` is True.
        """
        feedback = self.combined_feedback(mode=mode)
        if self.blocked and mode == "strict":
            return 2, {
                "decision": "block",
                "reason": feedback or f"Blocked by evaluator cascade ({self.final_decision})",
                "cascade": {
                    "final_decision": self.final_decision,
                    "min_bars_score": self.min_bars_score,
                    "conservative_lcb_score": self.conservative_lcb_score,
                    "short_circuited_at": self.short_circuited_at,
                },
            }
        if feedback:
            return 0, {
                "decision": "allow",
                "additionalContext": feedback,
                "systemMessage": feedback,
                "cascade": {
                    "final_decision": self.final_decision,
                    "min_bars_score": self.min_bars_score,
                    "conservative_lcb_score": self.conservative_lcb_score,
                },
            }
        return 0, {"decision": "allow"}


@dataclass(frozen=True)
class EvaluatorManifest:
    """Declarative metadata loaded from each module's `evaluator.yaml`."""

    evaluator_id: str
    name: str
    paradigm: EvaluatorParadigm
    stage: int  # 1 = Deterministic ($0, <50ms), 2 = LLM Judge, 3 = Adversarial Probe
    events: tuple[HookEvent, ...]
    tool_matcher: str = ".*"
    requires_blind_context: bool = True
    cross_family_preferred: bool = False
    mcp_tool_name: str = ""
    description: str = ""

    @property
    def resolved_mcp_tool_name(self) -> str:
        if self.mcp_tool_name:
            return self.mcp_tool_name
        return f"eval_{self.evaluator_id}"

    def matches(self, event: HookEvent, tool_name: str | None = None) -> bool:
        if event == "MCPConsult":
            return True
        if event not in self.events:
            return False
        if event in ("BeforeTool", "AfterTool") and tool_name:
            try:
                return bool(re.search(self.tool_matcher, tool_name))
            except re.error:
                return True
        return True

    @classmethod
    def from_yaml_file(cls, path: pathlib.Path) -> EvaluatorManifest:
        """Dependency-free parser for `evaluator.yaml` manifests."""
        raw = path.read_text(encoding="utf-8")
        data: dict[str, Any] = {}
        for line in raw.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or ":" not in stripped:
                continue
            key, val = stripped.split(":", 1)
            key = key.strip()
            val = val.strip()
            if val.startswith("[") and val.endswith("]"):
                items = [x.strip().strip("'\"") for x in val[1:-1].split(",") if x.strip()]
                data[key] = items
            elif val.lower() in ("true", "false"):
                data[key] = val.lower() == "true"
            elif val.isdigit():
                data[key] = int(val)
            else:
                data[key] = val.strip("'\"")
        eval_id = str(data.get("evaluator_id", path.parent.name))
        return cls(
            evaluator_id=eval_id,
            name=str(data.get("name", path.parent.name)),
            paradigm=data.get("paradigm", "deterministic"),
            stage=int(data.get("stage", 1)),
            events=tuple(data.get("events", ["AfterTool"])),
            tool_matcher=str(data.get("tool_matcher", ".*")),
            requires_blind_context=bool(data.get("requires_blind_context", True)),
            cross_family_preferred=bool(data.get("cross_family_preferred", False)),
            mcp_tool_name=str(data.get("mcp_tool_name", f"eval_{eval_id}")),
            description=str(data.get("description", "")),
        )


@runtime_checkable
class EvaluatorProtocol(Protocol):
    """Unified runtime interface implemented by all Deterministic, Judge, and Adversarial modules."""

    manifest: EvaluatorManifest

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        """Evaluates an immutable `EvaluationFrame` out-of-band and returns a verdict."""
        ...


class BaseEvaluator:
    """Convenience base class for authoring custom drop-in Evaluators.

    Subclasses automatically load `evaluator.yaml` from their own directory and
    can optionally define a custom `mcp_input_schema()` for MCP tool exposure.
    """

    manifest: EvaluatorManifest

    def __init__(self, manifest_path: pathlib.Path | None = None) -> None:
        if manifest_path is None:
            import inspect

            cls_file = pathlib.Path(inspect.getfile(self.__class__))
            manifest_path = cls_file.with_name("evaluator.yaml")
        self.manifest = EvaluatorManifest.from_yaml_file(manifest_path)

    def mcp_input_schema(self) -> dict[str, Any]:
        """Default MCP JSON Schema (Actor CoT excluded by design to prevent self-grading)."""
        props: dict[str, Any] = {
            "task_objective": {
                "type": "string",
                "description": "Objective task goal or requirement being solved (do not include Actor CoT).",
            },
            "target_paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of workspace-relative file paths to focus inspection on.",
            },
            "invariants": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional formal invariants (e.g. diff_must_contain:..., diff_must_not_contain:...).",
            },
        }
        if not self.manifest.requires_blind_context:
            props["proposed_claims"] = {
                "type": "string",
                "description": "Specific factual or verification claims to audit against raw tool history.",
            }
        return {
            "type": "object",
            "properties": props,
            "additionalProperties": False,
        }

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        raise NotImplementedError
