"""Multi-Stage Evaluator Cascade Runner (`EvaluatorCascadeRunner`)."""

from __future__ import annotations

import pathlib
from typing import Sequence

from crux_evals.core.protocol import (
    CascadeReport,
    Decision,
    EvaluationFrame,
    EvaluationVerdict,
    EvaluatorProtocol,
    LLMJudgeCallable,
)
from crux_evals.core.registry import EvaluatorRegistry

DECISION_SEVERITY: dict[Decision, int] = {
    "ALLOW": 0,
    "WARN": 1,
    "VETO": 2,
    "REWIND": 3,
}


class EvaluatorCascadeRunner:
    """Executes registered Evaluator modules in stage order with fast-fail short-circuiting."""

    def __init__(
        self,
        evaluators: Sequence[EvaluatorProtocol],
        *,
        short_circuit_on_stage_veto: bool = True,
        conservatism_lambda: float = 0.25,
    ) -> None:
        self.evaluators = sorted(evaluators, key=lambda ev: ev.manifest.stage)
        self.short_circuit_on_stage_veto = short_circuit_on_stage_veto
        self.conservatism_lambda = conservatism_lambda

    def get_evaluator(self, evaluator_id_or_tool: str) -> EvaluatorProtocol | None:
        """Looks up a registered evaluator by `evaluator_id` or `mcp_tool_name`."""
        for ev in self.evaluators:
            if ev.manifest.evaluator_id == evaluator_id_or_tool or ev.manifest.resolved_mcp_tool_name == evaluator_id_or_tool:
                return ev
        return None

    def run(self, frame: EvaluationFrame) -> CascadeReport:
        matching = [
            ev for ev in self.evaluators if ev.manifest.matches(frame.event, frame.tool_name)
        ]
        if not matching:
            return CascadeReport(
                event=frame.event,
                final_decision="ALLOW",
                min_bars_score=1.0,
                conservative_lcb_score=1.0,
                cumulative_progress_advantage=0.0,
                verdicts=[],
            )

        verdicts: list[EvaluationVerdict] = []
        worst_decision: Decision = "ALLOW"
        short_circuited_at: str | None = None

        stages = sorted({ev.manifest.stage for ev in matching})
        for stage_num in stages:
            stage_evaluators = [ev for ev in matching if ev.manifest.stage == stage_num]
            stage_vetoed = False
            for ev in stage_evaluators:
                input_frame = (
                    frame.with_epistemic_firewall() if ev.manifest.requires_blind_context else frame
                )
                verdict = ev.evaluate(input_frame)
                verdicts.append(verdict)
                if DECISION_SEVERITY[verdict.decision] > DECISION_SEVERITY[worst_decision]:
                    worst_decision = verdict.decision
                if verdict.decision in ("VETO", "REWIND"):
                    stage_vetoed = True
                    if short_circuited_at is None:
                        short_circuited_at = ev.manifest.evaluator_id

            if stage_vetoed and self.short_circuit_on_stage_veto:
                break

        any_hard_veto = any(bool(v.hard_vetoes_triggered) for v in verdicts)
        min_bars = min((v.bars_score for v in verdicts), default=1.0)
        if any_hard_veto:
            min_bars = 0.0

        lcb_scores = [
            max(0.0, v.bars_score - self.conservatism_lambda * v.uncertainty)
            * (0.0 if v.hard_vetoes_triggered else 1.0)
            for v in verdicts
        ]
        conservative_lcb = min(lcb_scores, default=1.0)
        avg_advantage = sum(v.progress_advantage for v in verdicts) / len(verdicts)

        return CascadeReport(
            event=frame.event,
            final_decision=worst_decision,
            min_bars_score=round(min_bars, 4),
            conservative_lcb_score=round(conservative_lcb, 4),
            cumulative_progress_advantage=round(avg_advantage, 4),
            verdicts=verdicts,
            short_circuited_at=short_circuited_at,
        )


def build_default_cascade(
    *,
    llm_judge: LLMJudgeCallable | None = None,
    llm_breaker: LLMJudgeCallable | None = None,
    enforce_read_before_edit: bool = True,
    enable_ast_property_fuzzer: bool = True,
    extra_plugin_dirs: list[pathlib.Path] | None = None,
    workspace_root: pathlib.Path | None = None,
    short_circuit_on_stage_veto: bool = True,
) -> EvaluatorCascadeRunner:
    """Constructs the Evaluator Cascade by auto-discovering all built-in + custom plugins."""
    registry = EvaluatorRegistry(
        extra_plugin_dirs=extra_plugin_dirs,
        workspace_root=workspace_root,
    )
    evaluators = registry.instantiate_all(
        llm_judge=llm_judge,
        llm_breaker=llm_breaker,
        kwargs_by_id={
            "epistemic_grounding_auditor": {"enforce_read_before_edit": enforce_read_before_edit},
            "adversarial_falsifier": {"enable_ast_property_fuzzer": enable_ast_property_fuzzer},
        },
    )
    return EvaluatorCascadeRunner(
        evaluators=evaluators,
        short_circuit_on_stage_veto=short_circuit_on_stage_veto,
    )
