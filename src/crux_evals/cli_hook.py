"""Universal JSON-stdio Lifecycle Hook Adapter (`python3 -m crux_evals.cli_hook`).

Supports two enforcement modes:
- `--mode complementary` (Default): Never blocks the agent (`exit 0`); injects
  navigational blind spots, counterexamples, and suggested next probes via `additionalContext`.
- `--mode strict`: Emits `exit 2` (`decision: block`) when a hard invariant veto triggers.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from typing import Any

from crux_evals.core.protocol import EnforcementMode, HookEvent
from crux_evals.core.runner import build_default_cascade
from crux_evals.core.state import HarnessTelemetryStore, capture_git_diff


def handle_hook_payload(
    payload: dict[str, Any],
    *,
    event_override: str | None = None,
    workspace_root: pathlib.Path | None = None,
    state_dir: pathlib.Path | None = None,
    mode: EnforcementMode = "complementary",
    extra_plugin_dirs: list[pathlib.Path] | None = None,
) -> tuple[int, dict[str, Any]]:
    """Processes a single hook invocation and returns `(exit_code, stdout_json)`."""
    raw_event = event_override or str(payload.get("event", "AfterTool"))
    event: HookEvent = (
        raw_event  # type: ignore[assignment]
        if raw_event in ("BeforeTool", "AfterTool", "Stop", "HypothesisCheckpoint", "MCPConsult")
        else "AfterTool"
    )

    ws = (workspace_root or pathlib.Path(payload.get("workspace_root") or os.getcwd())).resolve()
    s_dir = state_dir or (ws / ".ush" / "evaluator_state")
    store = HarnessTelemetryStore(s_dir)

    session_id = str(payload.get("session_id", os.environ.get("USH_SESSION_ID", "default")))
    telemetry = store.load(session_id=session_id, workspace_root=str(ws))

    if payload.get("task_objective"):
        telemetry.task_objective = str(payload["task_objective"])
    if isinstance(payload.get("declared_invariants"), list):
        telemetry.declared_invariants = [str(x) for x in payload["declared_invariants"]]

    tool_name = payload.get("tool_name")
    tool_input = payload.get("tool_input") or payload.get("tool_args") or {}
    if not isinstance(tool_input, dict):
        tool_input = {}

    current_diff = payload.get("unified_git_diff")
    if not isinstance(current_diff, str):
        current_diff = capture_git_diff(ws)

    if event == "AfterTool" and tool_name:
        output_raw = str(payload.get("preview") or payload.get("tool_output") or "")
        is_error = bool(payload.get("is_error", False))
        exit_code = payload.get("exit_code")
        telemetry.record_after_tool(
            tool_name=str(tool_name),
            tool_args=tool_input,
            output_raw=output_raw,
            is_error=is_error,
            exit_code=exit_code,
            current_diff=current_diff,
        )
        store.save(telemetry)

    frame = telemetry.to_evaluation_frame(
        event=event,
        tool_name=str(tool_name) if tool_name else None,
        tool_args=tool_input,
        tool_output_raw=str(payload.get("preview") or payload.get("tool_output") or ""),
        tool_is_error=bool(payload.get("is_error", False)),
        tool_exit_code=payload.get("exit_code"),
        unified_git_diff=current_diff,
        actor_claim_text=payload.get("final_text") or payload.get("actor_claim_text"),
        actor_model_family=payload.get("model_family"),
    )

    cascade = build_default_cascade(extra_plugin_dirs=extra_plugin_dirs, workspace_root=ws)
    report = cascade.run(frame)
    return report.to_hook_output(mode=mode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crux-Evals Universal CLI Hook")
    parser.add_argument("--event", default=None, help="Hook event override (BeforeTool, AfterTool, Stop)")
    parser.add_argument("--workspace", default=None, help="Workspace root directory")
    parser.add_argument(
        "--mode",
        choices=["complementary", "strict"],
        default=os.environ.get("CRUX_EVALS_MODE", "complementary"),
        help="Enforcement mode: complementary (advisory telemetry, exit 0) or strict (veto on block, exit 2)",
    )
    args = parser.parse_args(argv)

    raw_stdin = sys.stdin.read().strip() if not sys.stdin.isatty() else "{}"
    try:
        payload = json.loads(raw_stdin) if raw_stdin else {}
    except json.JSONDecodeError:
        payload = {}

    ws = pathlib.Path(args.workspace).resolve() if args.workspace else None
    exit_code, response_json = handle_hook_payload(
        payload,
        event_override=args.event,
        workspace_root=ws,
        mode=args.mode,  # type: ignore[arg-type]
    )
    sys.stdout.write(json.dumps(response_json, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
