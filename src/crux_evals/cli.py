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

    p_new = sub.add_parser("new", help="Scaffold a new plug-and-play Evaluator module")
    p_new.add_argument("evaluator_id", help="Snake_case identifier (e.g. sql_cost_guard)")
    p_new.add_argument("--name", default=None, help="Human-readable evaluator title")
    p_new.add_argument(
        "--paradigm",
        choices=["deterministic", "llm_judge", "adversarial"],
        default="deterministic",
        help="Evaluator paradigm",
    )
    p_new.add_argument("--stage", type=int, default=None, help="Cascade stage (1, 2, or 3)")
    p_new.add_argument("--target-dir", default=None, help="Directory in which to create the plugin folder")
    p_new.add_argument("--description", default="Custom complementary evaluator plugin.", help="Description")

    p_run = sub.add_parser("run", help="Run the Crux-Evals cascade against the workspace")
    p_run.add_argument("--workspace", default=".", help="Workspace root directory")
    p_run.add_argument("--objective", default="", help="Task objective")
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
        )
        sys.stdout.write(json.dumps({"status": "CREATED", "plugin_dir": str(created)}, indent=2) + "\n")
        return 0

    if args.cmd == "mcp":
        ws = pathlib.Path(args.workspace).expanduser().resolve()
        pdirs = [pathlib.Path(x).expanduser().resolve() for x in args.plugin_dir]
        server = CruxEvalsMCPServer(workspace_root=ws, plugin_dirs=pdirs)
        return server.serve_stdio()

    if args.cmd == "run":
        ws = pathlib.Path(args.workspace).expanduser().resolve()
        pdirs = [pathlib.Path(x).expanduser().resolve() for x in args.plugin_dir]
        store = HarnessTelemetryStore(ws / ".ush" / "evaluator_state")
        tel = store.load("default", str(ws))
        if args.objective:
            tel.task_objective = args.objective
        cascade = build_default_cascade(extra_plugin_dirs=pdirs, workspace_root=ws, short_circuit_on_stage_veto=False)
        report = cascade.run(tel.to_evaluation_frame("MCPConsult"))
        sys.stdout.write(json.dumps(report.to_mcp_payload(), indent=2, ensure_ascii=False) + "\n")
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
            "description": m.description,
        }
        for m, _ in reg.discover_classes()
    ]
    sys.stdout.write(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
