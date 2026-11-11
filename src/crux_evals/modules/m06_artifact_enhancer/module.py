"""Module 06: Generative Artifact & UX Enhancer Lens (`ArtifactEnhancerEvaluator`).

Domain-agnostic deliverable quality evaluator invoked mid-task via `consult_evaluator`
(without blocking tools) to catch internal harness scorecard leakage, unfinished
placeholders, missing interactive controls, and optional isolated UX review.
"""

from __future__ import annotations

import json
import pathlib
import re

from crux_evals.core.protocol import (
    BaseEvaluator,
    EvaluationFrame,
    EvaluationVerdict,
    LLMJudgeCallable,
    read_bounded_target_excerpts,
    scoped_eval_images,
)
from crux_evals.core.registry import register_evaluator

LEAKED_SCORECARD_RE = re.compile(
    r"(CRUX-EVALS\s+SCORECARD|crux_evals_scorecard|FINAL\s+DECISION:\s*ALLOW|"
    r"conservative_lcb_score|<dynamic_instructions>|<crux_tool_call>)",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(
    r"(Lorem\s+ipsum|\b(?:TODO|FIXME|XXX)\s*:|PLACEHOLDER_DATA|mockData\s*=\s*\[\s*\])",
    re.IGNORECASE,
)
NOOP_HANDLER_RE = re.compile(r"(?:onclick|onClick|onChange)\s*=\s*['\"{]\s*(?:\(\)\s*=>\s*\{\s*\}|void\s+0|return\s+false)", re.IGNORECASE)
VISUAL_SURFACE_RE = re.compile(r"(<canvas\b|<svg\b|Plotly|new\s+Chart\(|d3\.select|recharts)", re.IGNORECASE)
INTERACTIVE_CONTROL_RE = re.compile(r"(<input\b|<select\b|<button\b|addEventListener\s*\(|onClick\s*=|onChange\s*=)", re.IGNORECASE)
LARGE_INT_HINT_RE = re.compile(r"(MAX_SAFE_INTEGER|BigInt\b|\bmod\s+M\b|2\s*\*\*\s*5[3-9]|2\s*\*\*\s*[6-9]\d)", re.IGNORECASE)


@register_evaluator
class ArtifactEnhancerEvaluator(BaseEvaluator):
    """Domain-agnostic deliverable quality, purity, and interactive-depth evaluator."""

    def __init__(self, *, llm_judge: LLMJudgeCallable | None = None, **kwargs: object) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))
        self.llm_judge = llm_judge

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        safe_frame = frame.with_epistemic_firewall() if self.manifest.requires_blind_context else frame
        ws = pathlib.Path(safe_frame.workspace_root).resolve()
        blind_spots: list[str] = []
        probes: list[str] = []
        scores: dict[str, float] = {
            "deliverable_purity": 1.0,
            "placeholder_hygiene": 1.0,
            "interactive_depth": 1.0,
        }

        candidates = list(safe_frame.modified_files)
        if not candidates and ws.is_dir():
            for p in sorted(ws.iterdir()):
                if p.is_file() and p.suffix.lower() in (".html", ".htm", ".jsx", ".tsx", ".vue", ".svg"):
                    candidates.append(p.name)

        excerpts: list[str] = []
        for rel in candidates[:8]:
            fp = pathlib.Path(rel)
            if not fp.is_absolute():
                fp = (ws / fp).resolve()
            if not fp.is_file():
                continue
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            suffix = fp.suffix.lower()
            if suffix in (".html", ".htm", ".tsx", ".jsx", ".vue", ".svg", ".md"):
                excerpts.append(f"--- {rel} ---\n{content[:4_000]}")

            # 1. Forbid internal evaluator scorecards / harness XML inside user deliverables
            if suffix in (".html", ".htm", ".tsx", ".jsx", ".vue", ".svg", ".md") and LEAKED_SCORECARD_RE.search(content):
                scores["deliverable_purity"] = 0.0
                blind_spots.append(
                    f"DELIVERABLE_POLLUTION in `{fp.name}`: Internal harness/evaluator metadata "
                    f"(`Crux-Evals Scorecard` or harness tags) leaked into the user deliverable."
                )
                probes.append(
                    f"Remove all internal harness/evaluator scorecards and control tags from `{fp.name}` "
                    f"so the deliverable stays 100% focused on the user's domain."
                )

            # 2. Catch unfinished placeholders, Lorem ipsum, or empty no-op handlers
            if suffix in (".html", ".htm", ".tsx", ".jsx", ".vue", ".js", ".ts") and (
                PLACEHOLDER_RE.search(content) or NOOP_HANDLER_RE.search(content)
            ):
                scores["placeholder_hygiene"] = 0.5
                blind_spots.append(
                    f"STATIC_PLACEHOLDER_DATA in `{fp.name}`: Found placeholder markers (`TODO`/`Lorem ipsum`/`mockData`) "
                    f"or empty no-op UI handlers."
                )
                probes.append(
                    f"Replace static placeholder data and empty event handlers in `{fp.name}` with real computed values and live state updates."
                )

            # 3. Domain-appropriate UI / interactive visualization checks
            if suffix in (".html", ".htm"):
                if VISUAL_SURFACE_RE.search(content) and not INTERACTIVE_CONTROL_RE.search(content):
                    scores["interactive_depth"] = 0.5
                    blind_spots.append(
                        f"STATIC_VISUALIZATION in `{fp.name}`: Detected graphical canvas/chart elements without interactive parameter controls."
                    )
                    probes.append(
                        f"Consider adding interactive controls (e.g., sliders, selectors, or hover readouts) tailored to `{safe_frame.task_objective[:80] or fp.name}`."
                    )
                if "<html" in content.lower() and "viewport" not in content.lower():
                    scores["interactive_depth"] = min(scores["interactive_depth"], 0.75)
                    blind_spots.append(f"RESPONSIVE_VIEWPORT in `{fp.name}`: Missing `<meta name=\"viewport\" ...>` tag for responsive scaling.")
                if LARGE_INT_HINT_RE.search(content) and "BigInt" not in content:
                    scores["interactive_depth"] = min(scores["interactive_depth"], 0.5)
                    blind_spots.append(
                        f"JS_SAFE_INTEGER_HAZARD in `{fp.name}`: Detected >53-bit integer/modular arithmetic cues without `BigInt` precision."
                    )
                    probes.append(f"Use native JS `BigInt` in `{fp.name}` where values can exceed `Number.MAX_SAFE_INTEGER`.")

        analysis = ""
        if self.llm_judge is not None and (excerpts or safe_frame.attached_images):
            sys_p = (
                "You are a Cold-Eyes Deliverable & UX Quality Critic operating in a zero-history clean room. "
                "Inspect ONLY the raw artifact files, screenshots, and task objective (ZERO Actor CoT). "
                "Return STRICT JSON with AT MOST 5 surgical defects/improvements or empty lists if clean:\n"
                '{"analysis":"1-sentence assessment","blind_spots":["target: concrete defect"],"suggested_next_probes":["exact minimal fix"]}'
            )
            bounded_excerpts = (
                read_bounded_target_excerpts(ws, candidates, max_files=4, max_chars_per_file=10_000, total_chars=24_000)
                if candidates
                else "(none)"
            )
            usr_p = (
                f"<task_objective>\n{safe_frame.task_objective}\n</task_objective>\n\n"
                f"<attached_screenshots>\n{', '.join(i.path for i in safe_frame.attached_images) or '(none)'}\n</attached_screenshots>\n\n"
                f"<deliverable_excerpts>\n{bounded_excerpts}\n</deliverable_excerpts>"
            )
            with scoped_eval_images(safe_frame.attached_images):
                raw = self.llm_judge(sys_p, usr_p, safe_frame.actor_model_family)
            m = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    analysis = str(parsed.get("analysis") or "").strip()
                    for item in parsed.get("blind_spots") or []:
                        txt = str(item).strip()
                        if txt and txt not in blind_spots:
                            blind_spots.append(txt)
                    for pr in parsed.get("suggested_next_probes") or []:
                        txt = str(pr).strip()
                        if txt and txt not in probes:
                            probes.append(txt)
                except json.JSONDecodeError:
                    pass

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
            evaluator_analysis=analysis,
            isolated_context=True,
        )


EVALUATOR_CLASS = ArtifactEnhancerEvaluator
