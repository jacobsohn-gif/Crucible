"""Crux-Evals CLI & One-Command Evaluator Scaffolder (`python3 -m crux_evals.cli`).

Subcommands:
- `list`: Lists all discovered built-in and custom Evaluator plugins.
- `new <evaluator_id>`: Scaffolds a new plug-and-play Evaluator module (`evaluator.yaml` + `module.py`)
  that is immediately auto-discovered by both the MCP Server and Cascade Runner.
- `run`: Runs the evaluator suite once against the current workspace and prints JSON feedback.
- `mcp`: Launches the Open-Protocol JSON-RPC 2.0 MCP Server over stdio.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

from crux_evals.core.registry import EvaluatorRegistry, scaffold_new_evaluator
from crux_evals.core.runner import build_default_cascade
from crux_evals.core.state import HarnessTelemetryStore
from crux_evals.mcp_server import CruxEvalsMCPServer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crux-evals", description="Crux-Evals (Crucible Assay) CLI")
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List all discovered Evaluator modules and their MCP tool names")
    p_list.add_argument("--workspace", default=".", help="Workspace root directory")
    p_list.add_argument("--plugin-dir", action="append", default=[], help="Additional plugin directory")

    p_new = sub.add_parser("new", help="Scaffold a new plug-and-play Evaluator module (`EVALUATOR.md` or Python)")
    p_new.add_argument("evaluator_id", help="Snake_case identifier (e.g. sql_cost_guard)")
    p_new.add_argument("--name", default=None, help="Human-readable evaluator title")
    p_new.add_argument(
        "--style",
        choices=["markdown", "md", "python", "py"],
        default="markdown",
        help="Plugin format: `markdown` (single-file EVALUATOR.md lens, default) or `python` (evaluator.yaml + module.py)",
    )
    p_new.add_argument(
        "--paradigm",
        choices=["deterministic", "llm_judge", "adversarial"],
        default="llm_judge",
        help="Evaluator paradigm",
    )
    p_new.add_argument("--stage", type=int, default=None, help="Cascade stage (1, 2, or 3)")
    p_new.add_argument("--target-dir", default=None, help="Directory in which to create the plugin folder")
    p_new.add_argument("--description", default="Custom complementary evaluator plugin.", help="Description")
    p_new.add_argument("--when-to-use", default="", help="Usefulness hint explaining when the model should pick this evaluator")
    p_new.add_argument("--paths", action="append", default=[], help="Optional file glob trigger (e.g. --paths '*.sql')")

    p_run = sub.add_parser("run", help="Run a Crux-Evals workflow primitive, lens, or full cascade")
    p_run.add_argument(
        "evaluator",
        nargs="?",
        default=None,
        help="Optional workflow primitive (cold_eyes_review, challenge_design, break_loop) or evaluator ID",
    )
    p_run.add_argument("--workspace", default=".", help="Workspace root directory")
    p_run.add_argument("--objective", default="", help="Task objective")
    p_run.add_argument("--files", nargs="*", default=[], help="Target code files and/or screenshots (.png/.jpg/.webp)")
    p_run.add_argument("--plugin-dir", action="append", default=[], help="Additional plugin directory")

    p_mcp = sub.add_parser("mcp", help="Launch the Open-Protocol JSON-RPC 2.0 MCP Server over stdio")
    p_mcp.add_argument("--workspace", default=".", help="Workspace root directory")
    p_mcp.add_argument("--plugin-dir", action="append", default=[], help="Additional plugin directory")

    args = parser.parse_args(argv)

    if args.cmd == "new":
        target = pathlib.Path(args.target_dir).expanduser().resolve() if args.target_dir else None
        created = scaffold_new_evaluator(
            args.evaluator_id,
            name=args.name,
            paradigm=args.paradigm,
            stage=args.stage,
            target_root=target,
            description=args.description,
            when_to_use=args.when_to_use,
            paths=args.paths,
            style=args.style,
        )
        sys.stdout.write(json.dumps({"status": "CREATED", "plugin_dir": str(created), "style": args.style}, indent=2) + "\n")
        return 0

    if args.cmd == "mcp":
        ws = pathlib.Path(args.workspace).expanduser().resolve()
        pdirs = [pathlib.Path(x).expanduser().resolve() for x in args.plugin_dir]
        server = CruxEvalsMCPServer(workspace_root=ws, plugin_dirs=pdirs)
        return server.serve_stdio()

    if args.cmd == "run":
        from crux_evals.core.protocol import scoped_eval_images, split_targets_and_load_images
        from crux_evals.core.workflows import WORKFLOW_MODES, run_workflow_evaluator

        ws = pathlib.Path(args.workspace).expanduser().resolve()
        pdirs = [pathlib.Path(x).expanduser().resolve() for x in args.plugin_dir]
        store = HarnessTelemetryStore(ws / ".ush" / "evaluator_state")
        tel = store.load("default", str(ws))
        if args.objective:
            tel.task_objective = args.objective
        target_files: list[str] = []
        attached_images = []
        if args.files:
            target_files, attached_images = split_targets_and_load_images(ws, list(args.files))
        cascade = build_default_cascade(extra_plugin_dirs=pdirs, workspace_root=ws, short_circuit_on_stage_veto=False)
        frame = tel.to_evaluation_frame(
            "MCPConsult",
            target_paths=target_files,
            attached_images=attached_images,
        )
        ev_target = (args.evaluator or "").strip()
        with scoped_eval_images(attached_images):
            if ev_target in WORKFLOW_MODES:
                payload = run_workflow_evaluator(ev_target, frame, cascade)
            elif ev_target and ev_target != "all":
                ev = cascade.get_evaluator(ev_target)
                if ev is None:
                    sys.stderr.write(f"Unknown evaluator `{ev_target}`.\n")
                    return 2
                verdict = ev.evaluate(frame.with_epistemic_firewall() if ev.manifest.requires_blind_context else frame)
                payload = verdict.to_mcp_payload()
            else:
                payload = cascade.run(frame).to_mcp_payload()
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        return 0

    # Default (`list`)
    ws = pathlib.Path(getattr(args, "workspace", ".")).expanduser().resolve()
    pdirs = [pathlib.Path(x).expanduser().resolve() for x in getattr(args, "plugin_dir", [])]
    reg = EvaluatorRegistry(extra_plugin_dirs=pdirs, workspace_root=ws)
    catalog = [
        {
            "evaluator_id": m.evaluator_id,
            "mcp_tool_name": m.resolved_mcp_tool_name,
            "name": m.name,
            "paradigm": m.paradigm,
            "stage": m.stage,
            "events": list(m.events),
            "requires_blind_context": m.requires_blind_context,
            "cross_family_preferred": m.cross_family_preferred,
            "when_to_use": m.when_to_use,
            "paths": list(m.paths),
            "description": m.description,
        }
        for m in reg.list_manifests()
    ]
    sys.stdout.write(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
