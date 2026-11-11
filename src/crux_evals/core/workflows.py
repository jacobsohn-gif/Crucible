"""Workflow-Transition Clean-Room Evaluators (`challenge_design`, `cold_eyes_review`, `break_loop`).

Solves confirmation bias & briefing tax by anchoring evaluation to exact build-loop transitions:
1. `challenge_design` (Pre-Build): stress-tests architecture/assumptions before >250L edits or multi-file rewrites.
2. `cold_eyes_review` (Pre-Ship): zero-history linter pass over raw files, diffs, tests & screenshots (`<=5` surgical fixes or `PASS`).
3. `break_loop` (When Stuck): orthogonal clean-room root-cause hypotheses & 1-line probes after `>=2` failures (`<=3` items).
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

from crux_evals.core.protocol import (
    EvaluationFrame,
    LLMJudgeCallable,
    read_bounded_target_excerpts,
    scoped_eval_images,
)
from crux_evals.core.runner import EvaluatorCascadeRunner

WORKFLOW_MODES: tuple[str, ...] = ("cold_eyes_review", "challenge_design", "break_loop", "tournament")

_WORKFLOW_PHASE_MAP: dict[str, str] = {
    "challenge_design": "pre_build",
    "cold_eyes_review": "pre_ship",
    "break_loop": "when_stuck",
    "tournament": "contrastive_selection",
}

_PROMPT_BY_MODE: dict[str, str] = {
    "challenge_design": (
        "You are a Cold-Eyes Architecture & Design Critic operating in a ZERO-HISTORY clean room.\n"
        "The builder cannot erase its own planning assumptions. Inspect ONLY `<task_objective>`, any existing target files, "
        "`<proposed_focus>`, and attached screenshots/mockups.\n"
        "Hunt specifically for:\n"
        "1. State-model, schema, or API field mismatches before code is written.\n"
        "2. Over-engineered abstractions where a simpler mechanism works.\n"
        "3. Hidden edge cases (empty inputs, async/concurrency races, WebGL/DOM lifecycle, mobile/narrow viewport clipping).\n"
        "Return STRICT JSON (`status` MUST be `PASS` or `FINDINGS`; AT MOST 5 high-leverage items; NO long essays):\n"
        '{"status":"PASS"|"FINDINGS","summary":"1-sentence verdict","findings":[{"target":"file/component","issue":"concrete trap","fix":"minimal pre-build change"}]}'
    ),
    "cold_eyes_review": (
        "You are a Cold-Eyes Clean-Room Pre-Ship Critic operating in a ZERO-HISTORY context.\n"
        "The coding agent has 40k tokens of confirmation bias and can no longer see its own typos, mismatched state keys "
        "(`fooMode` written vs `foo` read), dead UI handlers, unverified edge crashes, or broken screenshots.\n"
        "Inspect ONLY raw workspace files, `<anonymous_git_diff>`, `<raw_verification_output>`, deterministic checks, and attached screenshots.\n"
        "Hunt for REAL defects only (no style nitpicks or scope-ballooning redesigns). Return STRICT JSON with AT MOST 5 surgical fixes or PASS:\n"
        '{"status":"PASS"|"FINDINGS","summary":"1-sentence verdict","findings":[{"target":"file:symbol or panel","issue":"concrete bug/gap","fix":"exact minimal fix"}]}'
    ),
    "break_loop": (
        "You are an Orthogonal Clean-Room Debugger operating in a ZERO-HISTORY context.\n"
        "An agent hit >=2 repeated failures and is likely trapped in a false mental model. Ignore how it tried to fix the bug; "
        "inspect ONLY the raw files, failing error/traceback, trajectory signals, and failing screenshots.\n"
        "Return STRICT JSON with AT MOST 3 distinct orthogonal root-cause hypotheses paired with concrete 1-line falsifying probes:\n"
        '{"status":"FINDINGS"|"PASS","summary":"1-sentence diagnosis","findings":[{"target":"file/symbol or runtime layer","issue":"orthogonal root cause","fix":"exact 1-line diagnostic probe or fix"}]}'
    ),
    "tournament": (
        "You are a Clean-Room Contrastive Architecture & Patch Tournament Judge operating in ZERO-HISTORY isolation.\n"
        "Single-artifact grading suffers from calibration drift; contrastive comparison between >=2 explicit candidates surfaces "
        "concrete algorithmic, state-model, and edge-case trade-offs.\n"
        "Inspect `<task_objective>`, `<candidate_proposals>`, any candidate files/diffs on disk, deterministic checks, and screenshots.\n"
        "Rank the candidates strictly on correctness, simplicity, failure modes, and invariant preservation. Return STRICT JSON (<=5 items):\n"
        '{"status":"PASS"|"FINDINGS","winner":"winning candidate ID/file","summary":"1-sentence winner + decisive reason","findings":[{"target":"candidate/file","issue":"concrete trade-off or flaw","fix":"adopt winner or merge specific safeguard"}]}'
    ),
}

_MAX_FINDINGS_BY_MODE: dict[str, int] = {
    "challenge_design": 5,
    "cold_eyes_review": 5,
    "break_loop": 3,
    "tournament": 5,
}


def _collect_deterministic_findings(
    mode: str,
    frame: EvaluationFrame,
    cascade: EvaluatorCascadeRunner,
) -> tuple[list[dict[str, str]], list[str]]:
    """Runs fast $0 deterministic / `/tmp` sandbox checks relevant to `mode`."""
    findings: list[dict[str, str]] = []
    checks_run: list[str] = []

    if mode == "challenge_design":
        checks_run.append("clean_room_design_context")
        return findings, checks_run
    if mode == "tournament" and not frame.modified_files:
        checks_run.append("clean_room_tournament_context")
        return findings, checks_run

    want_ids: list[str]
    if mode == "break_loop":
        want_ids = ["trajectory_sentinel"]
    else:
        want_ids = ["clean_patch_and_contract", "adversarial_falsifier"]
        has_ui = bool(frame.attached_images) or any(
            f.lower().endswith((".html", ".htm", ".jsx", ".tsx", ".vue", ".svg", ".md"))
            for f in frame.modified_files
        )
        if has_ui:
            want_ids.append("artifact_enhancer")
        # Include any path-scoped or custom evaluators (not the 6 built-in IDs)
        builtin_ids = {
            "trajectory_sentinel",
            "clean_patch_and_contract",
            "epistemic_grounding_auditor",
            "blind_bars_judge",
            "adversarial_falsifier",
            "artifact_enhancer",
        }
        for ev in cascade.evaluators:
            eid = ev.manifest.evaluator_id
            if eid not in builtin_ids and ev.manifest.paradigm == "deterministic" and eid not in want_ids:
                want_ids.append(eid)

    blind_frame = frame.with_epistemic_firewall()
    for eid in want_ids:
        ev = cascade.get_evaluator(eid)
        if ev is None:
            continue
        checks_run.append(eid)
        verdict = ev.evaluate(blind_frame)
        if verdict.counterexample:
            target = ", ".join(blind_frame.modified_files[:2]) or "workspace"
            first_line = verdict.counterexample.strip().splitlines()[-1][:220]
            findings.append(
                {
                    "target": f"{target} ({eid})",
                    "issue": first_line,
                    "fix": verdict.remediation_directive or "Handle the failing boundary input and add a regression test.",
                }
            )
            continue
        if verdict.decision == "ALLOW":
            continue
        for idx, bs in enumerate(verdict.blind_spots or verdict.unverified_claims or verdict.evidence_citations):
            probes = verdict.suggested_next_probes
            fix_hint = probes[idx] if idx < len(probes) else (verdict.remediation_directive or "Resolve finding and re-verify.")
            findings.append(
                {
                    "target": _guess_target_from_text(bs, blind_frame.modified_files, eid),
                    "issue": bs[:260],
                    "fix": fix_hint[:260],
                }
            )
        if not (verdict.blind_spots or verdict.unverified_claims or verdict.evidence_citations):
            findings.append(
                {
                    "target": ", ".join(blind_frame.modified_files[:2]) or eid,
                    "issue": f"{eid} reported {verdict.decision}",
                    "fix": verdict.remediation_directive[:260] or "Inspect deterministic check output.",
                }
            )

    return findings, checks_run


def _guess_target_from_text(text: str, files: tuple[str, ...], fallback: str) -> str:
    for f in files:
        base = pathlib.Path(f).name
        if base and base in text:
            return f
    return files[0] if files else fallback


def run_workflow_evaluator(
    mode: str,
    frame: EvaluationFrame,
    cascade: EvaluatorCascadeRunner,
    *,
    llm_judge: LLMJudgeCallable | None = None,
    focus_note: str = "",
) -> dict[str, Any]:
    """Executes `challenge_design`, `cold_eyes_review`, or `break_loop` in clean-room isolation with linter-style output."""
    norm_mode = mode if mode in WORKFLOW_MODES else "cold_eyes_review"
    cap = _MAX_FINDINGS_BY_MODE.get(norm_mode, 5)
    ws = pathlib.Path(frame.workspace_root).resolve()

    det_findings, checks_run = _collect_deterministic_findings(norm_mode, frame, cascade)

    llm_findings: list[dict[str, str]] = []
    llm_summary = ""
    llm_winner = ""
    if llm_judge is not None:
        checks_run.append(f"clean_room_llm:{norm_mode}")
        sys_prompt = _PROMPT_BY_MODE[norm_mode]
        blind_frame = frame.with_epistemic_firewall()
        target_excerpts = (
            read_bounded_target_excerpts(ws, blind_frame.modified_files, max_files=6, max_chars_per_file=12_000, total_chars=36_000)
            if blind_frame.modified_files
            else "(no files passed)"
        )
        diff_block = (
            blind_frame.unified_git_diff[:16_000]
            if blind_frame.unified_git_diff.strip()
            else "(no git diff — see <target_files_from_disk>)"
        )
        err_tail = ""
        if norm_mode == "break_loop":
            recent_errs = [
                f"Step {r.step_index} `{r.tool_name}` (exit={r.exit_code}): {r.output_raw[-900:]}"
                for r in blind_frame.tool_history[-8:]
                if r.is_error or "Traceback" in r.output_raw or "Error" in r.output_raw
            ]
            err_tail = "\n".join(recent_errs[-3:]) or blind_frame.last_verification_output[-1_500:]
        else:
            err_tail = blind_frame.last_verification_output[-3_000:]

        det_block = json.dumps(det_findings[:5], ensure_ascii=False) if det_findings else "[]"
        imgs_str = ", ".join(i.path for i in blind_frame.attached_images) or "(none)"
        focus_xml = (
            f"\n<proposed_focus>\n{focus_note.strip()[:1_600]}\n</proposed_focus>\n"
            if (focus_note.strip() and norm_mode in ("challenge_design", "tournament"))
            else ""
        )
        if norm_mode == "tournament" and frame.actor_claim_text:
            focus_xml += f"\n<candidate_proposals>\n{frame.actor_claim_text.strip()[:4_000]}\n</candidate_proposals>\n"

        user_payload = (
            f"<task_objective>\n{blind_frame.task_objective}\n</task_objective>\n"
            f"{focus_xml}"
            f"<files_inspected>\n{', '.join(blind_frame.modified_files) or '(none)'}\n</files_inspected>\n\n"
            f"<attached_screenshots>\n{imgs_str}\n</attached_screenshots>\n\n"
            f"<deterministic_findings>\n{det_block}\n</deterministic_findings>\n\n"
            f"<anonymous_git_diff>\n{diff_block}\n</anonymous_git_diff>\n\n"
            f"<target_files_from_disk>\n{target_excerpts}\n</target_files_from_disk>\n\n"
            f"<raw_error_or_verification passed=\"{blind_frame.last_verification_passed}\">\n"
            f"{err_tail or '(none)'}\n</raw_error_or_verification>"
        )
        with scoped_eval_images(blind_frame.attached_images):
            raw = llm_judge(sys_prompt, user_payload, blind_frame.actor_model_family)
        m = re.search(r"\{.*\}", raw or "", re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(0))
                llm_summary = str(parsed.get("summary") or "").strip()
                llm_winner = str(parsed.get("winner") or "").strip()
                raw_items = parsed.get("findings")
                if isinstance(raw_items, list):
                    for item in raw_items:
                        if isinstance(item, dict) and (item.get("issue") or item.get("target")):
                            llm_findings.append(
                                {
                                    "target": str(item.get("target") or "artifact")[:120],
                                    "issue": str(item.get("issue") or "")[:300],
                                    "fix": str(item.get("fix") or "")[:300],
                                }
                            )
            except json.JSONDecodeError:
                pass

    # Merge deterministic findings first + clean-room LLM findings, deduplicated and linter-capped (`<= 5` or `<= 3`)
    merged: list[dict[str, str]] = []
    seen_issues: set[str] = set()
    for item in [*det_findings, *llm_findings]:
        norm_key = re.sub(r"\s+", " ", item["issue"].strip().lower())[:90]
        if norm_key and norm_key not in seen_issues:
            seen_issues.add(norm_key)
            merged.append(item)
        if len(merged) >= cap:
            break

    status = "FINDINGS" if merged else "PASS"
    verdict = "FIX_REQUIRED" if merged else "PASS"
    default_summary = (
        f"PASS — clean-room {norm_mode} found no blocking defects across {len(frame.modified_files)} file(s)"
        + (f" and {len(frame.attached_images)} screenshot(s)." if frame.attached_images else ".")
        if not merged
        else f"Found {len(merged)} surgical item(s) during clean-room {norm_mode}."
    )
    out_payload: dict[str, Any] = {
        "evaluator": norm_mode,
        "workflow_phase": _WORKFLOW_PHASE_MAP.get(norm_mode, "pre_ship"),
        "status": status,
        "verdict": verdict,
        "clean_room": True,
        "isolated_context": True,
        "epistemic_firewall_active": True,
        "findings": merged,
        "summary": llm_summary or default_summary,
        "checks_run": checks_run,
        "files_inspected": list(frame.modified_files),
        "screenshots_inspected": [img.path for img in frame.attached_images],
        "usage_note": "Apply concrete fixes above silently or address blockers before shipping. Never expose internal evaluator tags to users.",
    }
    if norm_mode == "tournament":
        out_payload["winner"] = llm_winner or (frame.modified_files[0] if frame.modified_files and not merged else "")
    return out_payload
