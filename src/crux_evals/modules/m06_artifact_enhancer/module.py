"""Module 06: Generative Artifact & Pedagogy Enhancer Lens (`ArtifactEnhancerEvaluator`).

Designed to be invoked as a native tool (`consult_evaluator`) mid-task to elevate
the quality, interactivity, and mathematical depth of user deliverables without
acting as a punitive gatekeeper.
"""

from __future__ import annotations

import pathlib
import re

from crux_evals.core.protocol import BaseEvaluator, EvaluationFrame, EvaluationVerdict
from crux_evals.core.registry import register_evaluator

LEAKED_SCORECARD_RE = re.compile(
    r"(CRUX-EVALS\s+SCORECARD|crux_evals_scorecard|FINAL\s+DECISION:\s*ALLOW|conservative_lcb_score)",
    re.IGNORECASE,
)
STATIC_SLICE_ARRAY_RE = re.compile(r"const\s+V1_SLICE_HEX\s*=\s*\[", re.IGNORECASE)
FAKE_MOD_HASH_RE = re.compile(r"314159265\s*\+\s*idx\s*\*\s*7919")


@register_evaluator
class ArtifactEnhancerEvaluator(BaseEvaluator):
    """Surfaces high-leverage interactive, visual, and numerical enhancements for user deliverables."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        ws = pathlib.Path(frame.workspace_root).resolve()
        blind_spots: list[str] = []
        probes: list[str] = []
        scores: dict[str, float] = {
            "deliverable_purity": 1.0,
            "interactive_computation_depth": 1.0,
            "visual_pedagogy": 1.0,
        }

        candidates = list(frame.modified_files)
        if not candidates and ws.is_dir():
            for p in sorted(ws.iterdir()):
                if p.is_file() and p.suffix in (".html", ".py", ".js", ".ts", ".tsx"):
                    candidates.append(p.name)

        for rel in candidates:
            fp = ws / rel if not pathlib.Path(rel).is_absolute() else pathlib.Path(rel)
            if not fp.is_file():
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            # 1. Check for leaked evaluator/harness scorecards inside user-facing HTML/UI deliverables
            if fp.suffix in (".html", ".tsx", ".jsx", ".vue") and LEAKED_SCORECARD_RE.search(content):
                scores["deliverable_purity"] = 0.0
                blind_spots.append(
                    f"DELIVERABLE_POLLUTION in `{fp.name}`: Found internal `Crux-Evals Scorecard` / harness metadata "
                    f"rendered inside the user's visualization HTML. Users asked for a domain visualization, not harness telemetry."
                )
                probes.append(
                    f"Remove the `CRUX-EVALS SCORECARD` card from `{fp.name}` immediately so the UI stays 100% focused on the domain visualization."
                )

            # 2. Check HTML/JS visualizations for static canned values vs real interactive BigInt/FP64 computation
            if fp.suffix == ".html":
                if STATIC_SLICE_ARRAY_RE.search(content) or FAKE_MOD_HASH_RE.search(content):
                    scores["interactive_computation_depth"] = 0.5
                    blind_spots.append(
                        f"STATIC_PLACEHOLDER_DATA in `{fp.name}`: Stage 1 slice chips (`V1_SLICE_HEX`) and modulo residues "
                        f"(`314159265 + idx * 7919`) use hardcoded demo constants rather than live interactive computation."
                    )
                    probes.append(
                        f"Upgrade `{fp.name}` with a real in-browser `BigInt` + IEEE-754 FP64 Ozaki Scheme I & II kernel "
                        f"so users can edit/select input values `(a_il, b_lj)`, drag live sliders for slices `k in [2..8]` "
                        f"and moduli `s in [2..16]`, and watch real `conv_32i_2_8u`, `m_1=256 (+128 -> -128)` wraparound, "
                        f"and CRT reconstruction mod `M = prod m_t` (using JS `BigInt` since `M > 2^109` exceeds `Number.MAX_SAFE_INTEGER`)."
                    )
                if "BigInt" not in content:
                    scores["visual_pedagogy"] = 0.5
                    blind_spots.append(
                        f"JS_PRECISION_OPPORTUNITY in `{fp.name}`: Ozaki Scheme II modulus product `M = prod_(t=1..14) m_t` "
                        f"is ~109.7 bits (`> 2^53` JS `Number.MAX_SAFE_INTEGER`). Using native JS `BigInt` enables a live "
                        f"interactive CRT Dynamic-Range Cliff explorer (`2*c_max < M`) right inside the browser!"
                    )
                    probes.append(
                        f"Add a live interactive 'Single Dot-Product & CRT Reconstruction Microscope' + 'VMEM SRAM Occupancy Meter' "
                        f"powered by JS `BigInt` inside `{fp.name}` so the user can interactively see CRT overflow when `s` is too small "
                        f"and exact 53-bit FP64 lock-in as soon as `M > 2*c_max`."
                    )

        mean_score = sum(scores.values()) / len(scores)
        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm="deterministic",
            decision="WARN" if blind_spots else "ALLOW",
            bars_score=round(mean_score, 4),
            progress_advantage=0.35 if not blind_spots else 0.15,
            uncertainty=0.0,
            criterion_scores=scores,
            evidence_citations=blind_spots,
            remediation_directive=" ".join(probes),
            blind_spots=blind_spots,
            suggested_next_probes=probes,
        )


EVALUATOR_CLASS = ArtifactEnhancerEvaluator
