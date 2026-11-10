"""3-Stage Cascading Judge Funnel (`clean_patch` + Blind Cross-Family `BARS` + Position-Swapped Bradley-Terry).

Implements CT-18:
- Stage A ($0 Deterministic Sieve): Audits `clean_patch` for deleted/weakened existing test assertions, secret leaks,
  and executes an optional verification shell command.
- Stage B (Provenance-Blind Cross-Family `BARS` Sentinel): Strips author thinking & identity (`ContextMode::DropAll`),
  routes to a distinct model family when available, and scores atomic `0 / 0.5 / 1` anchors x hard vetoes `prod(h_m)`.
- Stage C (Position-Swapped Pairwise Bradley-Terry MLE Tournament): Evaluates BOTH `(A, B)` and `(B, A)` orders to
  eliminate LLM slot bias and converges iterative Bradley-Terry skill estimates across candidate worktree patches!
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import subprocess
from typing import Callable, Mapping

from ush.auth.vault import AuthProfile, CredentialVault
from ush.brain.model_tiers import resolve_model_profile
from ush.brain.providers import UniversalModelClient
from ush.brain.wire import parse_lenient_json_object
from ush.core.types import CanonicalMessage, CanonicalPart, sha256_hex
from ush.kernel.security import SECRET_TOKEN_REGEXES, scrub_child_environment
from ush.kernel.shadow_git import CleanPatchAudit, ShadowGitStore


@dataclass
class CandidatePatch:
    candidate_id: str
    contributor_ids: tuple[str, ...]
    family: str
    unified_diff: str
    worktree_path: str | None = None
    summary: str = ""


@dataclass
class StageAResult:
    passed: bool
    audit: CleanPatchAudit
    test_exit_code: int | None = None
    reasons: list[str] = field(default_factory=list)
    crux_evals_report: dict[str, object] = field(default_factory=dict)


@dataclass
class BarsVerdict:
    passed: bool
    score: float  # [0.0, 1.0] after multiplicative hard vetoes
    uncertainty: float
    judge_id: str
    judge_family: str
    criterion_scores: dict[str, float]
    hard_vetoes_triggered: list[str]
    critique: str


PairwiseComparator = Callable[[CandidatePatch, CandidatePatch], str]  # returns "A" | "B" | "TIE"


class CascadingJudge:
    """Anti-Goodhart, provenance-blind, position-swap-calibrated evaluator."""

    def __init__(
        self,
        vault: CredentialVault | None = None,
        client: UniversalModelClient | None = None,
    ) -> None:
        self.vault = vault
        self.client = client

    # ---- Stage A: $0 Deterministic Sieve (`clean_patch` + Secret Scan + Test Cmd) ----

    @staticmethod
    def run_stage_a(
        candidate: CandidatePatch,
        *,
        test_command: str | None = None,
        allow_test_assertion_deletions: bool = False,
    ) -> StageAResult:
        reasons: list[str] = []
        audit = ShadowGitStore.audit_clean_patch(candidate.unified_diff)

        if not candidate.unified_diff.strip():
            reasons.append("Candidate produced an empty diff.")

        if not allow_test_assertion_deletions and audit.tampered_with_existing_tests:
            reasons.append(
                f"ANTI_GOODHART_VETO: Candidate deleted/weakened existing test assertion(s): "
                f"{'; '.join(audit.deleted_assertions[:3])}"
            )

        # Check added lines (`+...`) for hardcoded high-entropy credentials
        added_lines = "\n".join(
            ln[1:]
            for ln in candidate.unified_diff.splitlines()
            if ln.startswith("+") and not ln.startswith("+++")
        )
        for rx in SECRET_TOKEN_REGEXES:
            if rx.search(added_lines):
                reasons.append("HARD_VETO_SECRET_LEAK: Candidate diff introduces a high-entropy secret literal.")
                break

        rc: int | None = None
        if test_command and candidate.worktree_path and not reasons:
            res = subprocess.run(
                ["/bin/sh", "-c", test_command],
                cwd=candidate.worktree_path,
                env=scrub_child_environment(),
                capture_output=True,
                text=True,
                timeout=120,
            )
            rc = res.returncode
            if rc != 0:
                tail = (res.stdout + "\n" + res.stderr).strip()[-600:]
                reasons.append(f"Deterministic test command failed (exit {rc}): {tail}")

        crux_rep: dict[str, object] = {}
        try:
            from crux_evals.core.protocol import EvaluationFrame
            from crux_evals.core.runner import build_default_cascade

            ws_path = candidate.worktree_path or "."
            cascade = build_default_cascade(enforce_read_before_edit=False)
            frame = EvaluationFrame(
                session_id=candidate.candidate_id,
                workspace_root=str(ws_path),
                event="AfterTool",
                step_index=1,
                task_objective=candidate.summary or "Verify candidate patch",
                tool_name="edit_file",
                unified_git_diff=candidate.unified_diff,
            )
            rep = cascade.run(frame)
            crux_rep = rep.to_mcp_payload()
            if rep.final_decision in ("VETO", "REWIND") and not allow_test_assertion_deletions:
                for v in rep.verdicts:
                    for hv in v.hard_vetoes_triggered:
                        if hv not in str(reasons):
                            reasons.append(f"CRUX_EVALS_{hv}: {v.remediation_directive}")
        except Exception:
            pass

        return StageAResult(
            passed=(len(reasons) == 0),
            audit=audit,
            test_exit_code=rc,
            reasons=reasons,
            crux_evals_report=crux_rep,
        )

    # ---- Stage B: Provenance-Blind Cross-Family `BARS` Sentinel ----

    def run_stage_b_bars(
        self,
        objective: str,
        candidate: CandidatePatch,
        *,
        judge_auth: AuthProfile | None = None,
        judge_model_id: str | None = None,
    ) -> BarsVerdict:
        if not self.client or not self.vault:
            raise RuntimeError("Stage B requires an initialized UniversalModelClient and CredentialVault.")

        # Prefer a DIFFERENT model family than the candidate author to break correlated blind spots!
        chosen_auth = judge_auth or self.vault.select_cross_family_profile(candidate.family) or self.vault.get_active_profile()
        if not chosen_auth:
            raise RuntimeError("No authenticated judge profile available.")
        mid = judge_model_id or chosen_auth.default_model or "gemini-2.5-pro"
        j_prof = resolve_model_profile(mid, chosen_auth.provider)
        judge_id = f"judge:{chosen_auth.name}:{j_prof.model_id}"

        # Provenance-blind prompt: ONLY user objective + raw unified diff (zero builder CoT or model name!)
        sys_prompt = (
            "You are a Provenance-Blind Software Judge. Grade ONLY the anonymous unified diff against the objective.\n"
            "Return STRICT JSON matching:\n"
            '{"bars":{"correctness":0|0.5|1,"edge_cases":0|0.5|1,"minimality":0|0.5|1,"safety":0|0.5|1},'
            '"hard_vetoes":[],"uncertainty":0.1,"critique":"concise actionable defects"}'
        )
        user_msg = CanonicalMessage(
            role="user",
            parts=[
                CanonicalPart(
                    kind="text",
                    text=f"<objective>\n{objective}\n</objective>\n\n<anonymous_diff>\n{candidate.unified_diff[:28_000]}\n</anonymous_diff>",
                )
            ],
        )
        reply = self.client.generate(
            profile=j_prof,
            auth_profile=chosen_auth,
            system_prompt=sys_prompt,
            history=[user_msg],
            tools=[],
            session_id=f"bars_{sha256_hex(candidate.unified_diff)[:10]}",
        )
        raw = reply.plain_text()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = parse_lenient_json_object(m.group(0)) if m else None
        if not parsed:
            return BarsVerdict(
                passed=False,
                score=0.0,
                uncertainty=1.0,
                judge_id=judge_id,
                judge_family=j_prof.family,
                criterion_scores={},
                hard_vetoes_triggered=[],
                critique=f"Unparseable judge output: {raw[:300]}",
            )

        bars_raw = parsed.get("bars") if isinstance(parsed.get("bars"), dict) else {}
        crit: dict[str, float] = {}
        for k, v in bars_raw.items():
            try:
                # Snap strictly to BARS {0.0, 0.5, 1.0} anchors
                fv = float(v)
                crit[str(k)] = 1.0 if fv >= 0.75 else (0.5 if fv >= 0.25 else 0.0)
            except (TypeError, ValueError):
                crit[str(k)] = 0.0

        vetoes = [str(x) for x in (parsed.get("hard_vetoes") or []) if str(x).strip()]
        raw_mean = (sum(crit.values()) / len(crit)) if crit else 0.0
        veto_mult = 0.0 if vetoes else 1.0
        final_score = raw_mean * veto_mult
        unc = max(0.0, min(1.0, float(parsed.get("uncertainty", 0.2))))
        return BarsVerdict(
            passed=(final_score >= 0.75 and not vetoes),
            score=final_score,
            uncertainty=unc,
            judge_id=judge_id,
            judge_family=j_prof.family,
            criterion_scores=crit,
            hard_vetoes_triggered=vetoes,
            critique=str(parsed.get("critique", "")),
        )

    # ---- Stage C: Position-Swapped (A,B) + (B,A) Pairwise Bradley-Terry Ranking (CT-18) ----

    @staticmethod
    def swapped_pairwise_compare(
        cand_a: CandidatePatch,
        cand_b: CandidatePatch,
        raw_judge: PairwiseComparator,
    ) -> tuple[float, float]:
        """Evaluates both `(A, B)` and `(B, A)` slot orders so position-biased judges cancel out to 0.5-0.5 (CT-18)!"""
        # Pass 1: slot1 = A, slot2 = B
        v1 = raw_judge(cand_a, cand_b).upper()
        pts_a_1 = 1.0 if v1 == "A" else (0.0 if v1 == "B" else 0.5)

        # Pass 2 (SWAPPED): slot1 = B, slot2 = A -> "A" here means cand_b won slot 1!
        v2 = raw_judge(cand_b, cand_a).upper()
        pts_a_2 = 1.0 if v2 == "B" else (0.0 if v2 == "A" else 0.5)

        score_a = (pts_a_1 + pts_a_2) / 2.0
        return score_a, 1.0 - score_a

    @classmethod
    def rank_bradley_terry(
        cls,
        candidates: list[CandidatePatch],
        raw_judge: PairwiseComparator,
        iterations: int = 25,
    ) -> list[tuple[CandidatePatch, float]]:
        """Runs an all-pairs position-swapped tournament and fits Bradley-Terry MLE skill parameters `pi_i`."""
        n = len(candidates)
        if n == 0:
            return []
        if n == 1:
            return [(candidates[0], 1.0)]

        # W[i][j] = unbiased fractional wins of i over j (plus tiny 0.01 Dirichlet prior for convergence)
        wins = [[0.01 if i != j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                sa, sb = cls.swapped_pairwise_compare(candidates[i], candidates[j], raw_judge)
                wins[i][j] += sa
                wins[j][i] += sb

        # Zermelo iterative MLE update for Bradley-Terry strengths
        strengths = [1.0 for _ in range(n)]
        for _ in range(iterations):
            nxt: list[float] = []
            for i in range(n):
                w_i = sum(wins[i][j] for j in range(n) if i != j)
                denom = sum(
                    (wins[i][j] + wins[j][i]) / (strengths[i] + strengths[j])
                    for j in range(n)
                    if i != j
                )
                nxt.append(w_i / denom if denom > 0 else strengths[i])
            total = sum(nxt) or 1.0
            strengths = [x / total for x in nxt]

        ranked = sorted(zip(candidates, strengths), key=lambda pair: pair[1], reverse=True)
        return ranked

    def build_llm_pairwise_comparator(
        self, objective: str, judge_auth: AuthProfile | None = None
    ) -> PairwiseComparator:
        if not self.client or not self.vault:
            raise RuntimeError("LLM comparator requires initialized client and vault.")
        auth = judge_auth or self.vault.get_active_profile()
        if not auth:
            raise RuntimeError("No judge auth profile configured.")
        prof = resolve_model_profile(auth.default_model or "gemini-2.5-pro", auth.provider)

        def _cmp(slot_a: CandidatePatch, slot_b: CandidatePatch) -> str:
            prompt = (
                f"Objective:\n{objective}\n\n"
                f"--- PATCH A ---\n{slot_a.unified_diff[:14_000]}\n\n"
                f"--- PATCH B ---\n{slot_b.unified_diff[:14_000]}\n\n"
                "Compare correctness, edge-case coverage, security, and architectural simplicity. "
                "End your response with strictly `VERDICT: A`, `VERDICT: B`, or `VERDICT: TIE`."
            )
            msg = self.client.generate(
                profile=prof,
                auth_profile=auth,
                system_prompt="You are an impartial pairwise patch referee.",
                history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text=prompt)])],
                tools=[],
                session_id="bt_referee",
            )
            m = re.search(r"VERDICT:\s*(A|B|TIE)\b", msg.plain_text(), re.IGNORECASE)
            return m.group(1).upper() if m else "TIE"

        return _cmp
