"""Open-Protocol MCP Server (`python3 -m crux_evals.mcp_server`).

Provides universal, zero-code compatibility with ANY coding harness that speaks
Model Context Protocol (Claude Code, Cursor, Gemini CLI, Codex CLI, Windsurf, Cline, Crux).

How it Enforces 'Zero Self-Grading' + 'Complementary Value':
1. Schema-Enforced Epistemic Firewall:
   Each evaluator MCP tool (`eval_trajectory_compass`, `eval_invariant_and_patch`,
   `eval_epistemic_grounding`, `eval_blind_judge`, `eval_adversarial_prober`, plus
   any custom drop-in plugin in `.crux_evals/plugins/` or `CRUX_EVALS_PLUGIN_PATH`)
   accepts ONLY objective pointers (`task_objective`, `target_paths`, `invariants`).
   There is no parameter for Actor Chain-of-Thought on blind/adversarial evaluators.
2. Direct Ground-Truth Hydration:
   The MCP server process reads `git diff HEAD`, `git status`, AST syntax trees, and
   executes adversarial probes in an isolated `/tmp` sandbox directly from disk.
3. Complementary Navigation Signal:
   Returns structured `alignment_score`, `progress_delta`, `blind_spots`,
   `counterexamples`, `unverified_claims`, and `suggested_next_probes`.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
from typing import Any

from crux_evals.core.protocol import BaseEvaluator, EvaluatorProtocol
from crux_evals.core.runner import build_default_cascade
from crux_evals.core.state import HarnessTelemetryStore, VERIFICATION_CMD_RE

MCP_PROTOCOL_VERSION = "2024-11-05"


class CruxEvalsMCPServer:
    """Zero-dependency JSON-RPC 2.0 MCP Server exposing all registered Evaluators."""

    def __init__(
        self,
        *,
        workspace_root: pathlib.Path | None = None,
        plugin_dirs: list[pathlib.Path] | None = None,
        state_dir: pathlib.Path | None = None,
    ) -> None:
        self.workspace_root = (workspace_root or pathlib.Path(os.getcwd())).expanduser().resolve()
        self.plugin_dirs = list(plugin_dirs or [])
        self.state_dir = state_dir or (self.workspace_root / ".ush" / "evaluator_state")
        self.store = HarnessTelemetryStore(self.state_dir)
        self.cascade = build_default_cascade(
            extra_plugin_dirs=self.plugin_dirs,
            workspace_root=self.workspace_root,
            short_circuit_on_stage_veto=False,
        )

    def refresh_plugins(self) -> None:
        """Re-scans built-in and custom plugin directories so newly added evaluators appear live."""
        self.cascade = build_default_cascade(
            extra_plugin_dirs=self.plugin_dirs,
            workspace_root=self.workspace_root,
            short_circuit_on_stage_veto=False,
        )

    def list_tools(self) -> list[dict[str, Any]]:
        """Returns MCP `tools/list` schemas for every discovered Evaluator + suite meta-tools."""
        self.refresh_plugins()
        tools: list[dict[str, Any]] = []
        for ev in self.cascade.evaluators:
            schema = (
                ev.mcp_input_schema()
                if isinstance(ev, BaseEvaluator) or hasattr(ev, "mcp_input_schema")
                else {
                    "type": "object",
                    "properties": {
                        "task_objective": {"type": "string"},
                        "target_paths": {"type": "array", "items": {"type": "string"}},
                        "invariants": {"type": "array", "items": {"type": "string"}},
                    },
                    "additionalProperties": False,
                }
            )
            tools.append(
                {
                    "name": ev.manifest.resolved_mcp_tool_name,
                    "description": (
                        f"[{ev.manifest.name} | Stage {ev.manifest.stage} {ev.manifest.paradigm}] "
                        f"{ev.manifest.description}"
                    ),
                    "inputSchema": schema,
                }
            )

        # Suite-wide meta-tool to run the entire multi-stage cascade in one call
        tools.append(
            {
                "name": "eval_run_suite",
                "description": (
                    "Runs the full multi-stage Crux-Evals cascade (Trajectory Compass, Clean Patch & "
                    "Contract, Epistemic Grounding, Blind BARS Judge, Adversarial Falsifier, plus any "
                    "custom plugins) hydrated directly from workspace ground truth."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_objective": {
                            "type": "string",
                            "description": "Objective task goal or requirement being solved.",
                        },
                        "target_paths": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional list of workspace-relative file paths to focus on.",
                        },
                        "invariants": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional formal invariants (diff_must_contain:..., etc.).",
                        },
                        "test_cmd": {
                            "type": "string",
                            "description": "Optional verification command (e.g. 'pytest') for the server to execute directly.",
                        },
                    },
                    "additionalProperties": False,
                },
            }
        )
        return tools

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Executes an MCP tool call with direct disk/git ground-truth hydration."""
        self.refresh_plugins()
        args = dict(arguments or {})
        session_id = str(args.get("session_id") or os.environ.get("USH_SESSION_ID", "default"))
        telemetry = self.store.load(session_id=session_id, workspace_root=str(self.workspace_root))

        if args.get("task_objective"):
            telemetry.task_objective = str(args["task_objective"])
        if isinstance(args.get("invariants"), list):
            telemetry.declared_invariants = [str(x) for x in args["invariants"]]

        target_paths = (
            [str(x) for x in args["target_paths"]] if isinstance(args.get("target_paths"), list) else None
        )

        # Optional server-side empirical test command execution (out-of-band from Actor)
        test_cmd = str(args.get("test_cmd", "")).strip()
        if test_cmd and VERIFICATION_CMD_RE.search(test_cmd):
            try:
                res = subprocess.run(
                    test_cmd,
                    shell=True,
                    cwd=str(self.workspace_root),
                    capture_output=True,
                    text=True,
                    timeout=15.0,
                )
                combined_out = (res.stdout + "\n" + res.stderr).strip()
                telemetry.record_after_tool(
                    tool_name="exec_pty",
                    tool_args={"command": test_cmd},
                    output_raw=combined_out,
                    is_error=(res.returncode != 0),
                    exit_code=res.returncode,
                )
            except (subprocess.SubprocessError, OSError) as exc:
                telemetry.record_after_tool(
                    tool_name="exec_pty",
                    tool_args={"command": test_cmd},
                    output_raw=str(exc),
                    is_error=True,
                    exit_code=1,
                )

        self.store.save(telemetry)

        frame = telemetry.to_evaluation_frame(
            event="MCPConsult",
            target_paths=target_paths,
            actor_claim_text=str(args["proposed_claims"]) if args.get("proposed_claims") else None,
        )

        if name == "eval_run_suite":
            report = self.cascade.run(frame)
            payload = report.to_mcp_payload()
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, indent=2, ensure_ascii=False),
                    }
                ],
                "structuredContent": payload,
                "isError": False,
            }

        ev: EvaluatorProtocol | None = self.cascade.get_evaluator(name)
        if ev is None:
            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps({"error": f"Unknown evaluator tool: {name}"}),
                    }
                ],
                "isError": True,
            }

        input_frame = frame.with_epistemic_firewall() if ev.manifest.requires_blind_context else frame
        verdict = ev.evaluate(input_frame)
        payload = verdict.to_mcp_payload()
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(payload, indent=2, ensure_ascii=False),
                }
            ],
            "structuredContent": payload,
            "isError": False,
        }

    def handle_jsonrpc(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Processes a single JSON-RPC 2.0 message and returns a response dict (or None for notifications)."""
        method = str(request.get("method", ""))
        req_id = request.get("id")
        params = request.get("params") or {}

        # Notifications (no `id`) do not receive a response
        if req_id is None and method.startswith("notifications/"):
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "resources": {"listChanged": False},
                        "prompts": {"listChanged": False},
                    },
                    "serverInfo": {
                        "name": "crux-evals",
                        "version": "0.2.0",
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": self.list_tools()},
            }

        if method == "tools/call":
            tool_name = str(params.get("name", ""))
            tool_args = params.get("arguments") or {}
            res = self.call_tool(tool_name, tool_args)
            return {"jsonrpc": "2.0", "id": req_id, "result": res}

        if method == "resources/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "resources": [
                        {
                            "uri": "eval://catalog/modules",
                            "name": "Registered Crux-Evals Modules",
                            "mimeType": "application/json",
                            "description": "List of all discovered deterministic, blind judge, and adversarial evaluators.",
                        },
                        {
                            "uri": "eval://telemetry/current",
                            "name": "Current Workspace Evaluation Telemetry",
                            "mimeType": "application/json",
                            "description": "Ground-truth read/edit/verification state for the active workspace session.",
                        },
                    ]
                },
            }

        if method == "resources/read":
            uri = str(params.get("uri", ""))
            if uri == "eval://catalog/modules":
                self.refresh_plugins()
                cat = [
                    {
                        "evaluator_id": ev.manifest.evaluator_id,
                        "mcp_tool_name": ev.manifest.resolved_mcp_tool_name,
                        "name": ev.manifest.name,
                        "paradigm": ev.manifest.paradigm,
                        "stage": ev.manifest.stage,
                        "requires_blind_context": ev.manifest.requires_blind_context,
                        "description": ev.manifest.description,
                    }
                    for ev in self.cascade.evaluators
                ]
                text = json.dumps(cat, indent=2)
            else:
                tel = self.store.load("default", str(self.workspace_root))
                text = json.dumps(tel.to_dict(), indent=2)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "contents": [
                        {"uri": uri, "mimeType": "application/json", "text": text}
                    ]
                },
            }

        if method == "prompts/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "prompts": [
                        {
                            "name": "audit_current_work",
                            "description": "Consult the Crux-Evals suite for independent navigation feedback on current changes.",
                            "arguments": [
                                {
                                    "name": "task_objective",
                                    "description": "The original task objective",
                                    "required": True,
                                }
                            ],
                        }
                    ]
                },
            }

        if method == "prompts/get":
            obj = str((params.get("arguments") or {}).get("task_objective", ""))
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "description": "Independent Crux-Evals Consultation Prompt",
                    "messages": [
                        {
                            "role": "user",
                            "content": {
                                "type": "text",
                                "text": (
                                    f"Call `eval_run_suite` with task_objective={obj!r} to obtain "
                                    f"independent blind-spot, invariant, and counterexample feedback."
                                ),
                            },
                        }
                    ],
                },
            }

        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    def serve_stdio(self) -> int:
        """Runs the standard newline-delimited JSON-RPC 2.0 MCP loop over stdin/stdout."""
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
            except json.JSONDecodeError:
                continue
            resp = self.handle_jsonrpc(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Crux-Evals Open-Protocol MCP Server")
    parser.add_argument("--workspace", default=".", help="Workspace root directory")
    parser.add_argument(
        "--plugin-dir",
        action="append",
        default=[],
        help="Additional directory to scan for custom Evaluator plugins",
    )
    args = parser.parse_args(argv)

    ws = pathlib.Path(args.workspace).expanduser().resolve()
    extra_dirs = [pathlib.Path(p).expanduser().resolve() for p in args.plugin_dir]
    server = CruxEvalsMCPServer(workspace_root=ws, plugin_dirs=extra_dirs)
    return server.serve_stdio()


if __name__ == "__main__":
    raise SystemExit(main())
