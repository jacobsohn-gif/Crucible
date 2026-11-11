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

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
import base64
import hashlib
import importlib.resources
import json
import mimetypes
import pathlib
import re
import threading
from typing import Any, Callable, Iterator, Literal, Optional, Protocol, Sequence, runtime_checkable

_IMAGE_MIME_BY_EXT: dict[str, str] = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}
_MAX_EVAL_IMAGE_BYTES = 4 * 1024 * 1024
_EVAL_IMAGE_TLS = threading.local()


@dataclass(frozen=True)
class ImageAttachment:
    """Screenshot or image artifact hydrated cleanly from workspace disk for multimodal cold-eyes inspection."""

    path: str
    mime_type: str
    base64_data: str


@contextmanager
def scoped_eval_images(images: Sequence[ImageAttachment]) -> Iterator[None]:
    """Attaches multimodal images in thread-local scope for an isolated `LLMJudgeCallable` call."""
    prev = getattr(_EVAL_IMAGE_TLS, "images", ())
    _EVAL_IMAGE_TLS.images = tuple(images)
    try:
        yield
    finally:
        _EVAL_IMAGE_TLS.images = prev


def get_current_eval_images() -> tuple[ImageAttachment, ...]:
    """Returns any screenshot attachments bound to the current evaluator call."""
    return tuple(getattr(_EVAL_IMAGE_TLS, "images", ()))


def _elide_middle_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 80:
        return text[:max_chars]
    half = (max_chars - 64) // 2
    omitted = len(text) - (2 * half)
    return f"{text[:half]}\n\n... [{omitted:,} middle chars elided] ...\n\n{text[-half:]}"


def split_targets_and_load_images(
    workspace_root: pathlib.Path,
    raw_paths: Sequence[str],
    *,
    validate_path: Callable[[str], pathlib.Path] | None = None,
) -> tuple[list[str], list[ImageAttachment]]:
    """Splits requested paths into non-image workspace files and base64-loaded image screenshots."""
    ws = workspace_root.expanduser().resolve()
    text_paths: list[str] = []
    images: list[ImageAttachment] = []

    for rp in raw_paths:
        raw = str(rp).strip()
        if not raw:
            continue
        fp = validate_path(raw) if validate_path is not None else (
            pathlib.Path(raw).resolve() if pathlib.Path(raw).is_absolute() else (ws / raw).resolve()
        )
        try:
            rel = fp.relative_to(ws).as_posix()
        except ValueError:
            rel = str(fp)

        ext = fp.suffix.lower()
        if ext in _IMAGE_MIME_BY_EXT and fp.is_file():
            try:
                raw_bytes = fp.read_bytes()
                if 0 < len(raw_bytes) <= _MAX_EVAL_IMAGE_BYTES:
                    mime = mimetypes.guess_type(str(fp))[0] or _IMAGE_MIME_BY_EXT[ext]
                    b64 = base64.b64encode(raw_bytes).decode("ascii")
                    if not any(img.path == rel for img in images):
                        images.append(ImageAttachment(path=rel, mime_type=mime, base64_data=b64))
                    continue
            except OSError:
                pass

        if rel not in text_paths:
            text_paths.append(rel)

    return text_paths, images


def read_bounded_target_excerpts(
    workspace_root: pathlib.Path,
    target_paths: Sequence[str],
    *,
    max_files: int = 6,
    max_chars_per_file: int = 12_000,
    total_chars: int = 36_000,
) -> str:
    """Hydrates raw files from disk using head+tail bounding so single-file HTML/JS apps expose both layout & logic."""
    ws = workspace_root.expanduser().resolve()
    excerpts: list[str] = []
    used = 0
    for rel in target_paths[:max_files]:
        fp = pathlib.Path(rel)
        if not fp.is_absolute():
            fp = (ws / fp).resolve()
        if fp.suffix.lower() in _IMAGE_MIME_BY_EXT or not fp.is_file():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        budget = min(max_chars_per_file, max(1_000, total_chars - used))
        bounded = _elide_middle_text(content, budget)
        block = f"--- {rel} ({len(content):,} chars) ---\n{bounded}"
        excerpts.append(block)
        used += len(bounded)
        if used >= total_chars:
            break
    return "\n\n".join(excerpts) if excerpts else "(none)"


def load_package_text_resource(
    anchor_path: pathlib.Path | str,
    package_name: str,
    resource_name: str,
) -> str:
    """Reads a text asset alongside a module cleanly across both filesystem and `.pyz` zipimport."""
    cand = pathlib.Path(anchor_path).with_name(resource_name)
    if cand.is_file():
        return cand.read_text(encoding="utf-8")
    return importlib.resources.files(package_name).joinpath(resource_name).read_text(encoding="utf-8")

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
    batch_has_mutation: bool = False

    # 3. Ground-Truth Workspace & Git State (hydrated from disk, never Actor CoT)
    unified_git_diff: str = ""
    modified_files: tuple[str, ...] = ()
    read_files: tuple[str, ...] = ()
    attached_images: tuple[ImageAttachment, ...] = ()

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

    # 4. Isolated Evaluator Thought & Analysis Metadata (separated from Actor CoT)
    evaluator_analysis: str = ""
    evaluator_thought: str = ""
    evaluator_model: str = ""
    isolated_context: bool = True

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
        payload: dict[str, Any] = {
            "evaluator": self.evaluator_id,
            "paradigm": self.paradigm,
            "status": status,
            "decision": self.decision,
            "isolated_context": self.isolated_context,
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
        if self.evaluator_analysis:
            payload["analysis"] = self.evaluator_analysis
        if self.evaluator_model:
            payload["evaluator_model"] = self.evaluator_model
        return payload

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


_FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)


def _parse_yaml_mapping(raw: str) -> dict[str, Any]:
    """Dependency-free colon-safe YAML key/value & list parser for `evaluator.yaml` and `EVALUATOR.md`."""
    data: dict[str, Any] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in raw.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if current_key and stripped.startswith("- ") and (raw_line.startswith(" ") or raw_line.startswith("\t")):
            item = stripped[2:].strip().strip("'\"")
            if current_list is None:
                current_list = []
                data[current_key] = current_list
            if item:
                current_list.append(item)
            continue
        if current_key and current_list is None and (raw_line.startswith("  ") or raw_line.startswith("\t")):
            prev = str(data.get(current_key, "")).strip()
            data[current_key] = f"{prev} {stripped}".strip()
            continue

        key_m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", raw_line)
        if not key_m:
            continue
        key, val = key_m.group(1).strip(), key_m.group(2).strip()
        current_key = key
        current_list = None
        if val in ("|", ">", ""):
            data[key] = ""
        elif val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            data[key] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()] if inner else []
        else:
            unq = val.strip("'\"")
            if unq.lower() in ("true", "false"):
                data[key] = unq.lower() == "true"
            elif unq.isdigit():
                data[key] = int(unq)
            else:
                data[key] = unq
    return data


@dataclass(frozen=True)
class EvaluatorManifest:
    """Declarative metadata loaded from `evaluator.yaml` or `EVALUATOR.md` frontmatter."""

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
    when_to_use: str = ""
    paths: tuple[str, ...] = ()

    @property
    def resolved_mcp_tool_name(self) -> str:
        if self.mcp_tool_name:
            return self.mcp_tool_name
        return f"eval_{self.evaluator_id}"

    @property
    def fingerprint(self) -> str:
        return sha256_short(f"{self.evaluator_id}|{self.description}|{self.when_to_use}|{self.paths}")

    def summary_line(self) -> str:
        wtu = f" (When useful: {self.when_to_use})" if self.when_to_use else ""
        blind = "blind" if self.requires_blind_context else "grounded"
        return f"- {self.evaluator_id} [{self.paradigm}, {blind}]: {self.description}{wtu}"

    def approx_catalog_tokens(self) -> int:
        return max(10, len(self.summary_line()) // 4)

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
    def from_yaml_text(cls, raw: str, *, default_id: str) -> EvaluatorManifest:
        """Parses an `evaluator.yaml` string into an `EvaluatorManifest`."""
        data = _parse_yaml_mapping(raw)
        eval_id = str(data.get("evaluator_id") or default_id).strip()
        raw_paths = data.get("paths") or []
        paths = tuple(str(x) for x in raw_paths) if isinstance(raw_paths, list) else ((str(raw_paths),) if raw_paths else ())
        raw_events = data.get("events") or ["AfterTool"]
        events = tuple(raw_events) if isinstance(raw_events, list) else (str(raw_events),)
        return cls(
            evaluator_id=eval_id,
            name=str(data.get("name", default_id)),
            paradigm=data.get("paradigm", "deterministic"),  # type: ignore[arg-type]
            stage=int(data.get("stage", 1)),
            events=events,  # type: ignore[arg-type]
            tool_matcher=str(data.get("tool_matcher", ".*")),
            requires_blind_context=bool(data.get("requires_blind_context", True)),
            cross_family_preferred=bool(data.get("cross_family_preferred", False)),
            mcp_tool_name=str(data.get("mcp_tool_name", f"eval_{eval_id}")),
            description=str(data.get("description", "")),
            when_to_use=str(data.get("when_to_use", data.get("whenToUse", ""))),
            paths=paths,
        )

    @classmethod
    def from_yaml_file(cls, path: pathlib.Path) -> EvaluatorManifest:
        """Dependency-free parser for `evaluator.yaml` manifests."""
        return cls.from_yaml_text(path.read_text(encoding="utf-8"), default_id=path.parent.name)

    @classmethod
    def from_markdown_text(cls, raw: str, *, default_id: str) -> tuple[EvaluatorManifest, str]:
        """Parses a skill-style `EVALUATOR.md` (`---` frontmatter + Markdown rubric body)."""
        m = _FRONTMATTER_RE.match(raw)
        if not m:
            manifest = cls(
                evaluator_id=default_id,
                name=" ".join(w.capitalize() for w in default_id.split("_")),
                paradigm="llm_judge",
                stage=2,
                events=("MCPConsult", "Stop"),
                requires_blind_context=True,
                cross_family_preferred=True,
                mcp_tool_name=f"eval_{default_id}",
                description=raw.strip().splitlines()[0].lstrip("# ").strip()[:240] if raw.strip() else default_id,
            )
            return manifest, raw.strip()

        fm_raw, body = m.group(1), m.group(2).strip()
        data = _parse_yaml_mapping(fm_raw)
        raw_id = str(data.get("evaluator_id") or data.get("name") or default_id).strip().lower()
        clean_id = re.sub(r"[^a-z0-9_]", "_", raw_id).strip("_") or default_id
        paradigm = str(data.get("paradigm") or "llm_judge")
        if paradigm not in ("deterministic", "llm_judge", "adversarial"):
            paradigm = "llm_judge"
        default_stage = 1 if paradigm == "deterministic" else (3 if paradigm == "adversarial" else 2)
        raw_events = data.get("events") or ["MCPConsult", "Stop"]
        events = tuple(raw_events) if isinstance(raw_events, list) else (str(raw_events),)
        raw_paths = data.get("paths") or []
        paths = tuple(str(x) for x in raw_paths) if isinstance(raw_paths, list) else ((str(raw_paths),) if raw_paths else ())
        desc = str(data.get("description") or "").strip()
        if not desc and body:
            desc = body.splitlines()[0].lstrip("# ").strip()[:240]

        manifest = cls(
            evaluator_id=clean_id,
            name=str(data.get("title") or data.get("name") or " ".join(w.capitalize() for w in clean_id.split("_"))),
            paradigm=paradigm,  # type: ignore[arg-type]
            stage=int(data.get("stage", default_stage)),
            events=events,  # type: ignore[arg-type]
            tool_matcher=str(data.get("tool_matcher", ".*")),
            requires_blind_context=bool(data.get("requires_blind_context", True)),
            cross_family_preferred=bool(data.get("cross_family_preferred", True)),
            mcp_tool_name=str(data.get("mcp_tool_name") or f"eval_{clean_id}"),
            description=desc,
            when_to_use=str(data.get("when_to_use", data.get("whenToUse", ""))),
            paths=paths,
        )
        return manifest, body

    @classmethod
    def from_markdown_file(cls, path: pathlib.Path) -> tuple[EvaluatorManifest, str]:
        return cls.from_markdown_text(path.read_text(encoding="utf-8", errors="replace"), default_id=path.parent.name)


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
        import inspect

        cls_file = pathlib.Path(inspect.getfile(self.__class__))
        pkg_name = self.__class__.__module__.rsplit(".", 1)[0]
        if manifest_path is not None and manifest_path.is_file():
            self.manifest = EvaluatorManifest.from_yaml_file(manifest_path)
            return
        yaml_text = load_package_text_resource(cls_file, pkg_name, "evaluator.yaml")
        default_id = pkg_name.rsplit(".", 1)[-1]
        self.manifest = EvaluatorManifest.from_yaml_text(yaml_text, default_id=default_id)

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


class MarkdownPromptEvaluator(BaseEvaluator):
    """Skill-like declarative evaluator loaded from `EVALUATOR.md` and run in strict model isolation.

    Enforces epistemic firewalling (`with_epistemic_firewall()`) whenever `requires_blind_context`
    is True so the Actor model's Chain-of-Thought never biases the evaluator's independent analysis.
    """

    def __init__(
        self,
        *,
        manifest: EvaluatorManifest,
        rubric_body: str,
        llm_judge: LLMJudgeCallable | None = None,
    ) -> None:
        self.manifest = manifest
        self.rubric_body = rubric_body.strip()
        self.llm_judge = llm_judge

    def build_isolated_prompt(self, frame: EvaluationFrame) -> tuple[str, str]:
        safe_frame = frame.with_epistemic_firewall() if self.manifest.requires_blind_context else frame
        ws = pathlib.Path(safe_frame.workspace_root).resolve()

        system_prompt = (
            f"You are an independent Crux-Evals lens (`{self.manifest.evaluator_id}`: {self.manifest.name}).\n"
            "You operate in STRICT ISOLATION from the coding agent: zero prior chat history, zero Actor reasoning, "
            "and ground-truth artifacts only.\n\n"
            f"Evaluator Instructions / Rubric:\n{self.rubric_body}\n\n"
            "Return STRICT JSON matching:\n"
            '{"analysis":"clear independent technical analysis","bars":0.0|0.5|1.0,'
            '"criterion_scores":{},"hard_vetoes":[],"blind_spots":[],'
            '"suggested_next_probes":[],"remediation":"","uncertainty":0.1}'
        )

        diff_block = safe_frame.unified_git_diff[:16_000] if safe_frame.unified_git_diff.strip() else "(no git diff)"
        verif_block = (
            safe_frame.last_verification_output[:4_000]
            if safe_frame.last_verification_output.strip()
            else "(no verification command output recorded)"
        )
        claims_block = (
            f"\n\n<actor_claims>\n{safe_frame.actor_claim_text[:4_000]}\n</actor_claims>"
            if not self.manifest.requires_blind_context and safe_frame.actor_claim_text
            else ""
        )
        excerpts_block = read_bounded_target_excerpts(ws, safe_frame.modified_files)
        imgs_block = (
            ", ".join(img.path for img in safe_frame.attached_images)
            if safe_frame.attached_images
            else "(none)"
        )

        user_payload = (
            f"<task_objective>\n{safe_frame.task_objective}\n</task_objective>\n\n"
            f"<declared_invariants>\n"
            + ("\n".join(f"- {inv}" for inv in safe_frame.declared_invariants) or "(none)")
            + "\n</declared_invariants>\n\n"
            f"<modified_files>\n{', '.join(safe_frame.modified_files) or '(none)'}\n</modified_files>\n\n"
            f"<attached_screenshots>\n{imgs_block}\n</attached_screenshots>\n\n"
            f"<anonymous_git_diff>\n{diff_block}\n</anonymous_git_diff>\n\n"
            f"<target_file_excerpts>\n{excerpts_block}\n</target_file_excerpts>\n\n"
            f"<raw_verification_output passed=\"{safe_frame.last_verification_passed}\">\n"
            f"{verif_block}\n</raw_verification_output>{claims_block}"
        )
        return system_prompt, user_payload

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        safe_frame = frame.with_epistemic_firewall() if self.manifest.requires_blind_context else frame
        if self.llm_judge is None:
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm=self.manifest.paradigm,
                decision="ALLOW",
                bars_score=1.0,
                progress_advantage=0.05,
                uncertainty=0.25,
                isolated_context=True,
                evaluator_analysis=(
                    f"Declarative lens `{self.manifest.evaluator_id}` inspected {len(safe_frame.modified_files)} "
                    f"target path(s) in deterministic mode (attach an isolated LLM judge for full semantic critique)."
                ),
            )

        sys_p, user_p = self.build_isolated_prompt(safe_frame)
        with scoped_eval_images(safe_frame.attached_images):
            raw = self.llm_judge(sys_p, user_p, safe_frame.actor_model_family)
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if not m:
            snippet = (raw or "").strip()[:500]
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm=self.manifest.paradigm,
                decision="ALLOW" if not snippet else "WARN",
                bars_score=0.75 if snippet else 1.0,
                uncertainty=0.3,
                isolated_context=True,
                evaluator_analysis=snippet or f"Lens `{self.manifest.evaluator_id}` completed.",
                blind_spots=[snippet] if snippet else [],
            )

        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            snippet = raw.strip()[:500]
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm=self.manifest.paradigm,
                decision="WARN",
                bars_score=0.5,
                uncertainty=0.4,
                isolated_context=True,
                evaluator_analysis=snippet,
                blind_spots=[snippet] if snippet else [],
            )

        crit: dict[str, float] = {}
        raw_crit = parsed.get("criterion_scores")
        if isinstance(raw_crit, dict):
            for k, v in raw_crit.items():
                try:
                    fv = float(v)
                    crit[str(k)] = 1.0 if fv >= 0.75 else (0.5 if fv >= 0.25 else 0.0)
                except (TypeError, ValueError):
                    pass

        try:
            raw_bars = float(parsed.get("bars", sum(crit.values()) / len(crit) if crit else 1.0))
        except (TypeError, ValueError):
            raw_bars = 1.0
        bars_score = 1.0 if raw_bars >= 0.75 else (0.5 if raw_bars >= 0.25 else 0.0)
        hard_vetoes = [str(v) for v in (parsed.get("hard_vetoes") or []) if str(v).strip()]
        blind_spots = [str(x) for x in (parsed.get("blind_spots") or []) if str(x).strip()]
        probes = [str(x) for x in (parsed.get("suggested_next_probes") or []) if str(x).strip()]
        remediation = str(parsed.get("remediation") or "").strip()
        analysis = str(parsed.get("analysis") or "").strip()
        try:
            uncertainty = max(0.0, min(1.0, float(parsed.get("uncertainty", 0.15))))
        except (TypeError, ValueError):
            uncertainty = 0.15

        if hard_vetoes:
            decision: Decision = "VETO"
            bars_score = 0.0
        elif bars_score < 0.75 or blind_spots:
            decision = "WARN"
        else:
            decision = "ALLOW"

        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm=self.manifest.paradigm,
            decision=decision,
            bars_score=bars_score,
            progress_advantage=round(bars_score - 0.5, 4),
            uncertainty=uncertainty,
            criterion_scores=crit,
            hard_vetoes_triggered=hard_vetoes,
            evidence_citations=blind_spots,
            remediation_directive=remediation,
            blind_spots=blind_spots,
            suggested_next_probes=probes,
            evaluator_analysis=analysis,
            isolated_context=True,
        )
