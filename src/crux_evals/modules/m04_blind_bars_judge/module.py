"""Module 04: Provenance-Blind Cross-Family BARS Judge (`BlindBarsJudgeEvaluator`)."""

from __future__ import annotations

import json
import pathlib
import re
from typing import Callable

from crux_evals.core.protocol import (
    BaseEvaluator,
    Decision,
    EvaluationFrame,
    EvaluationVerdict,
    LLMJudgeCallable,
)
from crux_evals.core.registry import register_evaluator


@register_evaluator
class BlindBarsJudgeEvaluator(BaseEvaluator):
    """Confirmation-bias-resistant LLM-as-a-Judge with position-swap calibration."""

    def __init__(
        self,
        *,
        llm_judge: LLMJudgeCallable | None = None,
        pass_threshold: float = 0.75,
        conservatism_lambda: float = 0.25,
    ) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))
        self.rubric_prompt = pathlib.Path(__file__).with_name("rubric_prompt.md").read_text(encoding="utf-8")
        self.llm_judge = llm_judge
        self.pass_threshold = pass_threshold
        self.conservatism_lambda = conservatism_lambda

    @staticmethod
    def build_blind_prompt(frame: EvaluationFrame) -> str:
        """Constructs the user payload strictly from ground-truth artifacts (ZERO Actor CoT)."""
        blind_frame = frame.with_epistemic_firewall()
        assert blind_frame.actor_claim_text is None, "Epistemic firewall violation: actor_claim_text leaked!"

        invariants_block = "\n".join(f"- {inv}" for inv in blind_frame.declared_invariants) or "(none declared)"
        diff_block = blind_frame.unified_git_diff[:24_000] if blind_frame.unified_git_diff.strip() else "(no git diff)"
        verif_block = (
            blind_frame.last_verification_output[:6_000]
            if blind_frame.last_verification_output.strip()
            else "(no verification command output recorded)"
        )
        return (
            f"<task_objective>\n{blind_frame.task_objective}\n</task_objective>\n\n"
            f"<declared_invariants>\n{invariants_block}\n</declared_invariants>\n\n"
            f"<modified_files>\n{', '.join(blind_frame.modified_files) or '(none)'}\n</modified_files>\n\n"
            f"<anonymous_git_diff>\n{diff_block}\n</anonymous_git_diff>\n\n"
            f"<raw_verification_output passed=\"{blind_frame.last_verification_passed}\">\n"
            f"{verif_block}\n</raw_verification_output>"
        )

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        blind_frame = frame.with_epistemic_firewall()

        if self.llm_judge is None:
            return self._deterministic_bars_fallback(blind_frame)

        user_payload = self.build_blind_prompt(blind_frame)
        raw_reply = self.llm_judge(
            self.rubric_prompt,
            user_payload,
            blind_frame.actor_model_family,
        )
        m = re.search(r"\{.*\}", raw_reply, re.DOTALL)
        if not m:
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm="llm_judge",
                decision="WARN",
                bars_score=0.5,
                uncertainty=0.5,
                evidence_citations=["Blind BARS Judge returned malformed JSON; falling back to WARN."],
            )

        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm="llm_judge",
                decision="WARN",
                bars_score=0.5,
                uncertainty=0.5,
                evidence_citations=["Blind BARS Judge returned unparseable JSON."],
            )

        raw_bars = parsed.get("bars") or {}
        criterion_scores: dict[str, float] = {}
        for key in (
            "functional_completeness",
            "edge_case_resilience",
            "surgical_minimality",
            "verification_sufficiency",
        ):
            val = float(raw_bars.get(key, 0.5))
            if val < 0.25:
                criterion_scores[key] = 0.0
            elif val < 0.75:
                criterion_scores[key] = 0.5
            else:
                criterion_scores[key] = 1.0

        hard_vetoes = [str(v) for v in (parsed.get("hard_vetoes") or []) if str(v).strip()]
        uncertainty = max(0.0, min(1.0, float(parsed.get("uncertainty", 0.1))))

        mean_bars = sum(criterion_scores.values()) / len(criterion_scores)
        veto_multiplier = 0.0 if hard_vetoes else 1.0
        raw_score = mean_bars * veto_multiplier
        lcb_score = max(0.0, raw_score - self.conservatism_lambda * uncertainty) * veto_multiplier

        failed_checks = [str(x) for x in (parsed.get("failed_checks") or [])]
        remediation = str(parsed.get("remediation", "")).strip()

        decision: Decision = (
            "VETO" if (hard_vetoes or lcb_score < self.pass_threshold) else "ALLOW"
        )
        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm="llm_judge",
            decision=decision,
            bars_score=round(raw_score, 4),
            progress_advantage=round(lcb_score - 0.5, 4),
            uncertainty=uncertainty,
            criterion_scores=criterion_scores,
            hard_vetoes_triggered=hard_vetoes,
            evidence_citations=failed_checks,
            remediation_directive=remediation if decision != "ALLOW" else "",
            blind_spots=failed_checks,
            suggested_next_probes=[remediation] if remediation and decision != "ALLOW" else [],
        )

    def _deterministic_bars_fallback(self, blind_frame: EvaluationFrame) -> EvaluationVerdict:
        """Offline heuristic BARS scorer when no live LLM judge client is injected."""
        has_diff = bool(blind_frame.unified_git_diff.strip() or blind_frame.modified_files)
        verif_ok = blind_frame.ran_verification_since_last_edit
        scores = {
            "functional_completeness": 1.0 if has_diff or not blind_frame.last_edit_step else 0.5,
            "edge_case_resilience": 1.0 if verif_ok else 0.5,
            "surgical_minimality": 1.0,
            "verification_sufficiency": 1.0 if verif_ok else 0.0,
        }
        mean_bars = sum(scores.values()) / len(scores)
        decision: Decision = "ALLOW" if mean_bars >= self.pass_threshold else "VETO"
        rem = "" if decision == "ALLOW" else "Ensure changes are complete and verified by tests."
        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm="llm_judge",
            decision=decision,
            bars_score=mean_bars,
            progress_advantage=0.25 if decision == "ALLOW" else -0.5,
            uncertainty=0.1,
            criterion_scores=scores,
            remediation_directive=rem,
            suggested_next_probes=[rem] if rem else [],
        )

    @staticmethod
    def compare_candidates_position_swapped(
        objective: str,
        diff_a: str,
        diff_b: str,
        pairwise_fn: Callable[[str, str, str], str],
    ) -> str:
        """Evaluates BOTH (A, B) and (B, A) presentation orders to eliminate LLM slot bias."""
        res_ab = pairwise_fn(objective, diff_a, diff_b)
        res_ba = pairwise_fn(objective, diff_b, diff_a)
        vote_a = (1 if res_ab == "FIRST" else 0) + (1 if res_ba == "SECOND" else 0)
        vote_b = (1 if res_ab == "SECOND" else 0) + (1 if res_ba == "FIRST" else 0)
        if vote_a > vote_b:
            return "A"
        if vote_b > vote_a:
            return "B"
        return "TIE"


EVALUATOR_CLASS = BlindBarsJudgeEvaluator
