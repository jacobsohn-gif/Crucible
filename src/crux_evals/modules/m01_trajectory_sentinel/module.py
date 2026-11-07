"""Module 01: Trajectory Loop & Process Advantage Sentinel (`TrajectorySentinelEvaluator`)."""

from __future__ import annotations

import pathlib

from crux_evals.core.protocol import (
    BaseEvaluator,
    Decision,
    EvaluationFrame,
    EvaluationVerdict,
)
from crux_evals.core.registry import register_evaluator

READ_ONLY_TOOLS = frozenset({"read_file", "grep_files", "glob_files", "lsp_query"})


@register_evaluator
class TrajectorySentinelEvaluator(BaseEvaluator):
    """Deterministic O(1) trajectory & process-advantage compass."""

    def __init__(
        self,
        *,
        warn_error_streak: int = 3,
        rewind_error_streak: int = 5,
        max_read_spiral: int = 8,
    ) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))
        self.warn_error_streak = warn_error_streak
        self.rewind_error_streak = rewind_error_streak
        self.max_read_spiral = max_read_spiral

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        safe_frame = frame.with_epistemic_firewall()
        history = safe_frame.tool_history
        if not history:
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm="deterministic",
                decision="ALLOW",
                bars_score=1.0,
                progress_advantage=0.1,
            )

        latest = history[-1]
        citations: list[str] = []
        vetoes: list[str] = []
        blind_spots: list[str] = []
        suggested_probes: list[str] = []
        decision: Decision = "ALLOW"
        bars_score = 1.0
        progress_advantage = 0.25 if not latest.is_error else -0.25
        localized_step: int | None = None
        remediation = ""

        # 1. Exact Action Signature Streak
        if latest.action_signature and (latest.is_error or latest.tool_name not in READ_ONLY_TOOLS):
            exact_streak = 0
            for rec in reversed(history):
                if rec.action_signature == latest.action_signature:
                    exact_streak += 1
                else:
                    break
            if exact_streak >= self.rewind_error_streak:
                decision = "REWIND"
                bars_score = 0.0
                progress_advantage = -1.0
                vetoes.append("HARD_VETO_ACTION_LOOP_STALL")
                localized_step = history[-exact_streak].step_index
                msg = (
                    f"Identical tool invocation `{latest.tool_name}` repeated {exact_streak} times "
                    f"(steps {localized_step}..{latest.step_index})."
                )
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "STOP repeating the identical tool call. Rewind to the pre-loop state, "
                    "formulate a distinct root-cause hypothesis, and run a falsifying diagnostic."
                )
                suggested_probes.append("Run `git diff` or `grep_files` to inspect callers before retrying.")
            elif exact_streak >= self.warn_error_streak:
                decision = "WARN"
                bars_score = 0.5
                progress_advantage = -0.75
                localized_step = history[-exact_streak].step_index
                msg = f"Tool call `{latest.tool_name}` repeated {exact_streak} consecutive times."
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "You repeated the exact same action 3 times. State your active hypothesis "
                    "explicitly and run ONE diagnostic read/command to falsify it before retrying."
                )
                suggested_probes.append(remediation)

        # 2. Error-Signature Stagnation Across Different Edits/Commands
        if decision == "ALLOW" and latest.error_signature:
            err_matches = [
                rec.step_index
                for rec in history[-6:]
                if rec.error_signature == latest.error_signature
            ]
            streak = len(err_matches)
            if streak >= self.rewind_error_streak:
                decision = "REWIND"
                bars_score = 0.0
                progress_advantage = -1.0
                vetoes.append("HARD_VETO_ERROR_SIGNATURE_STAGNATION")
                localized_step = err_matches[0]
                msg = (
                    f"Same normalized error fingerprint `{latest.error_signature}` recurred {streak} times "
                    f"at steps {err_matches}."
                )
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "Your recent edits have failed with the exact same underlying exception 5 times. "
                    "Your current root-cause hypothesis is falsified; revert and inspect callers/types."
                )
                suggested_probes.append("Inspect upstream callers and input types with `grep_files`.")
            elif streak >= self.warn_error_streak:
                decision = "WARN"
                bars_score = 0.5
                progress_advantage = -0.8
                localized_step = err_matches[0]
                msg = (
                    f"Same normalized error fingerprint `{latest.error_signature}` recurred {streak} times "
                    f"at steps {err_matches}."
                )
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "STOP patching blindly: the last 3 attempts produced the identical error signature. "
                    "Isolate the minimal failing input in a one-liner before making another file edit."
                )
                suggested_probes.append(remediation)

        # 3. 2-Cycle Action Oscillation Detection (A -> B -> A -> B)
        if decision == "ALLOW" and len(history) >= 4:
            sigs = [r.action_signature for r in history[-4:]]
            if sigs[0] == sigs[2] and sigs[1] == sigs[3] and sigs[0] != sigs[1]:
                decision = "WARN"
                bars_score = 0.5
                progress_advantage = -0.7
                localized_step = history[-4].step_index
                msg = (
                    f"2-cycle oscillation detected between `{history[-2].tool_name}` and "
                    f"`{history[-1].tool_name}` across steps {history[-4].step_index}..{latest.step_index}."
                )
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "You are oscillating in an A -> B -> A -> B tool loop. Break the cycle by "
                    "recording what you learned and testing a third orthogonal path."
                )
                suggested_probes.append(remediation)

        # 4. Diff State Hash Oscillation
        if (
            decision == "ALLOW"
            and latest.tool_name in ("edit_file", "write_file")
            and latest.diff_state_hash
        ):
            prior_edit_hashes = [
                (r.step_index, r.diff_state_hash)
                for r in history[:-1]
                if r.tool_name in ("edit_file", "write_file") and r.diff_state_hash
            ]
            if prior_edit_hashes and prior_edit_hashes[-1][1] != latest.diff_state_hash:
                earlier_matches = [
                    step for step, h in prior_edit_hashes[:-1] if h == latest.diff_state_hash
                ]
                if earlier_matches:
                    decision = "WARN"
                    bars_score = 0.5
                    progress_advantage = -0.85
                    localized_step = earlier_matches[0]
                    msg = (
                        f"Workspace diff hash `{latest.diff_state_hash}` at step {latest.step_index} "
                        f"reverted to the exact patch state already tried at step {localized_step}."
                    )
                    citations.append(msg)
                    blind_spots.append(msg)
                    remediation = (
                        f"Patch oscillation detected: your latest edit recreated the exact code state "
                        f"from step {localized_step}. Do not toggle back and forth between two patches."
                    )
                    suggested_probes.append(remediation)

        # 5. Read-Without-Hypothesis Spiral
        if decision == "ALLOW" and len(history) >= self.max_read_spiral:
            tail = history[-self.max_read_spiral :]
            if all(r.tool_name in READ_ONLY_TOOLS for r in tail):
                decision = "WARN"
                bars_score = 0.5
                progress_advantage = -0.4
                localized_step = tail[0].step_index
                msg = (
                    f"{self.max_read_spiral} consecutive read/search calls executed without running "
                    f"an empirical reproduction or narrowing to an action."
                )
                citations.append(msg)
                blind_spots.append(msg)
                remediation = (
                    "Avoid unbounded read spirals. Synthesize your current findings into a concrete "
                    "hypothesis and execute a reproduction command or targeted edit."
                )
                suggested_probes.append(remediation)

        if (
            decision == "ALLOW"
            and safe_frame.ran_verification_since_last_edit
            and safe_frame.last_verification_step == latest.step_index
        ):
            progress_advantage = 1.0

        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm="deterministic",
            decision=decision,
            bars_score=bars_score,
            progress_advantage=progress_advantage,
            uncertainty=0.0,
            hard_vetoes_triggered=vetoes,
            evidence_citations=citations,
            localized_step_index=localized_step,
            remediation_directive=remediation,
            blind_spots=blind_spots,
            suggested_next_probes=suggested_probes,
        )


EVALUATOR_CLASS = TrajectorySentinelEvaluator
