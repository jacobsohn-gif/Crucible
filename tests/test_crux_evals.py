"""Comprehensive Test Suite for Crux-Evals (Crucible Assay):
- 5 Core Evaluators (Deterministic, Hybrid, Blind BARS Judge, Adversarial Falsifier)
- Open-Protocol MCP Server (`initialize`, `tools/list`, `tools/call`, `resources/read`, Epistemic Firewall)
- Plug-and-Play Extensibility (`EvaluatorRegistry` auto-discovery + `scaffold_new_evaluator` CLI)
- Complementary vs. Strict Lifecycle Hook Adapter (`cli_hook.py`)
"""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from crux_evals import (
    AdversarialFalsifierEvaluator,
    BlindBarsJudgeEvaluator,
    CleanPatchAndContractEvaluator,
    CruxEvalsMCPServer,
    EpistemicGroundingAuditor,
    EvaluatorRegistry,
    SessionTelemetry,
    TrajectorySentinelEvaluator,
    build_default_cascade,
    scaffold_new_evaluator,
)
from crux_evals.cli_hook import handle_hook_payload


class TestCoreEvaluators(unittest.TestCase):
    def test_trajectory_sentinel_loops_and_oscillation(self) -> None:
        sentinel = TrajectorySentinelEvaluator(warn_error_streak=3, rewind_error_streak=5)
        tel = SessionTelemetry(session_id="s1", workspace_root="/tmp")
        for _ in range(3):
            tel.record_after_tool(
                "exec_pty",
                {"command": "pytest"},
                "FAILED test_foo.py::test_bar - AssertionError: 1 != 2",
                is_error=True,
                exit_code=1,
            )
        v3 = sentinel.evaluate(tel.to_evaluation_frame("AfterTool", tool_name="exec_pty"))
        self.assertEqual(v3.decision, "WARN")
        self.assertLess(v3.progress_delta, 0.0)
        self.assertTrue(len(v3.blind_spots) > 0)
        self.assertTrue(len(v3.suggested_next_probes) > 0)

    def test_clean_patch_anti_goodhart_and_secret_hygiene(self) -> None:
        gate = CleanPatchAndContractEvaluator()
        fake_key = "AKIA" + "1234567890ABCDEF"
        diff = (
            "diff --git a/tests/test_pricing.py b/tests/test_pricing.py\n"
            "--- a/tests/test_pricing.py\n"
            "+++ b/tests/test_pricing.py\n"
            "@@ -10,3 +10,3 @@\n"
            " def test_margin():\n"
            "-    assert calculate_margin(100) == 25\n"
            f"+    API_KEY = '{fake_key}'\n"
            "+    assert True\n"
        )
        tel = SessionTelemetry(session_id="c1", workspace_root="/tmp")
        verdict = gate.evaluate(tel.to_evaluation_frame("AfterTool", tool_name="edit_file", unified_git_diff=diff))
        self.assertEqual(verdict.decision, "VETO")
        self.assertIn("HARD_VETO_TEST_ASSERTION_TAMPERING", verdict.hard_vetoes_triggered)
        self.assertIn("HARD_VETO_TAUTOLOGICAL_ASSERTION", verdict.hard_vetoes_triggered)
        self.assertIn("HARD_VETO_SECRET_LEAK", verdict.hard_vetoes_triggered)

    def test_epistemic_grounding_read_before_edit_and_claim_audit(self) -> None:
        auditor = EpistemicGroundingAuditor()
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir)
            (ws / "service.py").write_text("x = 1\n", encoding="utf-8")
            tel = SessionTelemetry(session_id="e1", workspace_root=tmpdir)

            v_unread = auditor.evaluate(
                tel.to_evaluation_frame("BeforeTool", tool_name="edit_file", tool_args={"path": "service.py"})
            )
            self.assertEqual(v_unread.decision, "VETO")
            self.assertTrue(len(v_unread.suggested_next_probes) > 0)

            tel.record_after_tool("read_file", {"path": "service.py"}, "1: x = 1", False)
            v_read = auditor.evaluate(
                tel.to_evaluation_frame("BeforeTool", tool_name="edit_file", tool_args={"path": "service.py"})
            )
            self.assertEqual(v_read.decision, "ALLOW")

    def test_blind_bars_judge_strips_actor_cot(self) -> None:
        tel = SessionTelemetry(session_id="b1", workspace_root="/tmp", task_objective="Fix division bug")
        frame = tel.to_evaluation_frame(
            "Stop",
            unified_git_diff="+return a / b if b else 0.0",
            actor_claim_text="ACTOR_COT_MUST_NEVER_BE_SEEN_BY_BLIND_JUDGE",
        )
        prompt = BlindBarsJudgeEvaluator.build_blind_prompt(frame)
        self.assertNotIn("ACTOR_COT_MUST_NEVER_BE_SEEN_BY_BLIND_JUDGE", prompt)
        self.assertIn("+return a / b if b else 0.0", prompt)

    def test_adversarial_falsifier_catches_crash_and_passes_when_fixed(self) -> None:
        falsifier = AdversarialFalsifierEvaluator()
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir)
            (ws / "metrics.py").write_text(
                "def compute_mean(values: list[float]) -> float:\n"
                "    return sum(values) / len(values)\n",
                encoding="utf-8",
            )
            tel = SessionTelemetry(session_id="adv1", workspace_root=tmpdir, task_objective="Compute mean")
            tel.record_after_tool("write_file", {"path": "metrics.py"}, "ok", False)
            v_broken = falsifier.evaluate(tel.to_evaluation_frame("Stop"))
            self.assertEqual(v_broken.decision, "VETO")
            self.assertIn("ZeroDivisionError", v_broken.counterexample or "")

            (ws / "metrics.py").write_text(
                "def compute_mean(values: list[float]) -> float:\n"
                "    return sum(values) / len(values) if values else 0.0\n",
                encoding="utf-8",
            )
            v_fixed = falsifier.evaluate(tel.to_evaluation_frame("Stop"))
            self.assertEqual(v_fixed.decision, "ALLOW")


class TestOpenMCPServerAndDynamicExtensibility(unittest.TestCase):
    def test_mcp_server_jsonrpc_lifecycle_and_epistemic_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir)
            (ws / "calc.py").write_text(
                "def ratio(items: list[float]) -> float:\n    return sum(items) / len(items)\n",
                encoding="utf-8",
            )
            server = CruxEvalsMCPServer(workspace_root=ws)

            # 1. initialize
            init_resp = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            assert init_resp is not None
            self.assertEqual(init_resp["result"]["serverInfo"]["name"], "crux-evals")

            # 2. tools/list -> verify all 5 evaluators + eval_run_suite are exposed
            list_resp = server.handle_jsonrpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
            assert list_resp is not None
            tool_names = {t["name"] for t in list_resp["result"]["tools"]}
            self.assertIn("eval_trajectory_compass", tool_names)
            self.assertIn("eval_invariant_and_patch", tool_names)
            self.assertIn("eval_epistemic_grounding", tool_names)
            self.assertIn("eval_blind_judge", tool_names)
            self.assertIn("eval_adversarial_prober", tool_names)
            self.assertIn("eval_run_suite", tool_names)

            # Verify blind evaluators do NOT accept `proposed_claims` / Actor CoT in their MCP inputSchema
            blind_schema = next(
                t["inputSchema"] for t in list_resp["result"]["tools"] if t["name"] == "eval_blind_judge"
            )
            self.assertNotIn("proposed_claims", blind_schema["properties"])

            # 3. tools/call `eval_adversarial_prober` -> hydrates calc.py from disk and finds ZeroDivisionError!
            call_resp = server.handle_jsonrpc(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "eval_adversarial_prober",
                        "arguments": {
                            "task_objective": "Compute ratio safely",
                            "target_paths": ["calc.py"],
                        },
                    },
                }
            )
            assert call_resp is not None
            structured = call_resp["result"]["structuredContent"]
            self.assertEqual(structured["status"], "INSIGHTS_FOUND")
            self.assertEqual(len(structured["counterexamples"]), 1)
            self.assertIn("ZeroDivisionError", structured["counterexamples"][0])

    def test_plug_and_play_custom_evaluator_scaffolding_and_auto_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir)
            custom_plugins_dir = ws / ".crux_evals" / "plugins"

            # Use the one-command scaffolder to create a brand-new evaluator plugin
            created_dir = scaffold_new_evaluator(
                "sql_partition_guard",
                name="SQL Partition Filter Guard",
                paradigm="deterministic",
                target_root=custom_plugins_dir,
                description="Checks SQL files for missing WHERE partition filters.",
            )
            self.assertTrue((created_dir / "evaluator.yaml").is_file())
            self.assertTrue((created_dir / "module.py").is_file())

            # Verify EvaluatorRegistry and CruxEvalsMCPServer auto-discover it with ZERO manual wiring!
            reg = EvaluatorRegistry(workspace_root=ws)
            ids = [m.evaluator_id for m, _ in reg.discover_classes()]
            self.assertIn("sql_partition_guard", ids)

            server = CruxEvalsMCPServer(workspace_root=ws)
            tools = {t["name"] for t in server.list_tools()}
            self.assertIn("eval_sql_partition_guard", tools)

            # Call the newly scaffolded evaluator over MCP
            res = server.call_tool("eval_sql_partition_guard", {"task_objective": "Check SQL"})
            self.assertFalse(res["isError"])
            self.assertEqual(res["structuredContent"]["evaluator"], "sql_partition_guard")
            self.assertEqual(res["structuredContent"]["status"], "CLEAN")

    def test_complementary_vs_strict_hook_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir)
            (ws / "app.py").write_text("x = 1\n", encoding="utf-8")

            # Unread file edit in COMPLEMENTARY mode (default): returns exit_code 0 + additionalContext!
            rc_comp, out_comp = handle_hook_payload(
                {"session_id": "m1", "tool_name": "edit_file", "tool_input": {"path": "app.py"}},
                event_override="BeforeTool",
                workspace_root=ws,
                mode="complementary",
            )
            self.assertEqual(rc_comp, 0)
            self.assertEqual(out_comp["decision"], "allow")
            self.assertIn("Read-Before-Edit" if False else "HARD_VETO_UNREAD_FILE_EDIT", out_comp["additionalContext"])

            # Same event in STRICT mode: returns exit_code 2 + decision: block!
            rc_strict, out_strict = handle_hook_payload(
                {"session_id": "m1", "tool_name": "edit_file", "tool_input": {"path": "app.py"}},
                event_override="BeforeTool",
                workspace_root=ws,
                mode="strict",
            )
            self.assertEqual(rc_strict, 2)
            self.assertEqual(out_strict["decision"], "block")


    def test_live_session_runner_and_cascading_judge_crux_evals_wiring(self) -> None:
        """Verifies SessionRunner auto-wires HookSystem+Crux-Evals and CascadingJudge populates crux_evals_report."""
        from ush.auth.vault import AuthProfile, CredentialVault
        from ush.brain.providers import UniversalModelClient
        from ush.core.types import CanonicalMessage, CanonicalPart, StopReason, StreamEvent
        from ush.kernel.fsm import SessionRunner
        from ush.platform.hooks_memory_ext import HookSystem
        from ush.swarm.judge import CandidatePatch, CascadingJudge

        with tempfile.TemporaryDirectory() as tmpdir:
            ws = pathlib.Path(tmpdir) / "ws"
            ws.mkdir()
            home = pathlib.Path(tmpdir) / "home"
            vault = CredentialVault(home, probe_ambient=False)

            replies = [
                CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text="All tests passed and 0 failed!")],
                    stop_reason=StopReason.STOP,
                ),
                CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text="Acknowledged: verification command not yet executed.")],
                    stop_reason=StopReason.STOP,
                ),
            ]

            class _DummyTransport:
                def __init__(self) -> None:
                    self.idx = 0

                def stream_turn(self, **kwargs):
                    msg = replies[min(self.idx, len(replies) - 1)]
                    self.idx += 1
                    return msg

            auth = AuthProfile(
                name="test",
                provider="google",
                auth_kind="api_key",
                base_url="https://generativelanguage.googleapis.com",
                default_model="gemini-2.5-pro",
                source="none",
            )
            hs = HookSystem(ws, is_workspace_trusted=True, enable_crux_evals=True, crux_evals_mode="strict")
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(vault, _DummyTransport()),
                auth_profile=auth,
                ush_home=home,
                hook_runner=hs,
            )
            events: list[StreamEvent] = []
            try:
                outcome = runner.run_turn("Implement feature and verify", on_event=events.append)
                self.assertEqual(outcome.steps_executed, 2)
                self.assertIn("verification command not yet executed", outcome.final_message.plain_text())
                scorecards = [e for e in events if e.meta.get("event") == "crux_evals_scorecard"]
                self.assertGreaterEqual(len(scorecards), 2)
                self.assertEqual(scorecards[0].meta["decision"], "VETO")
                self.assertEqual(scorecards[-1].meta["decision"], "ALLOW")
            finally:
                runner.close()

            cand = CandidatePatch(
                candidate_id="c1",
                contributor_ids=("cell-0",),
                family="gemini",
                unified_diff="--- a/x.py\n+++ b/x.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n",
                worktree_path=str(ws),
                summary="Add y",
            )
            stage_a = CascadingJudge.run_stage_a(cand)
            self.assertTrue(stage_a.passed)
            self.assertEqual(stage_a.crux_evals_report.get("final_decision"), "ALLOW")



    def test_native_consult_evaluator_tool_and_artifact_enhancer_lens(self) -> None:
        """Verifies `consult_evaluator` is a native read-only tool in ToolRegistry and `artifact_enhancer` catches scorecard pollution vs clean interactive visualizations."""
        import json
        from ush.core.types import CanonicalPart, PermissionMode
        from ush.kernel.patch import StaleReadTracker
        from ush.kernel.pty_manager import PersistentPtyManager
        from ush.kernel.security import PolicyEngine, SecurityFirewall
        from ush.kernel.tools import ToolExecutionContext, build_default_tool_registry

        reg = build_default_tool_registry()
        spec = reg.get("consult_evaluator")
        self.assertIsNotNone(spec)
        self.assertTrue(spec.read_only)
        self.assertIn("consult_evaluator", PolicyEngine.READ_ONLY_TOOLS)

        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            polluted = ws / "viz.html"
            polluted.write_text("<html><body><h1>Crux-Evals Scorecard</h1></body></html>")
            pty = PersistentPtyManager(ws)
            try:
                res1 = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="c1", tool_name="consult_evaluator", tool_args={
                        "lens": "artifact_enhancer",
                        "objective": "Interactive Ozaki TPU visualization",
                        "target_paths": ["viz.html"],
                    })],
                    ctx_factory=lambda cid: ToolExecutionContext(call_id=cid, firewall=SecurityFirewall(ws), stale_tracker=StaleReadTracker(), pty=pty, policy_engine=PolicyEngine(), mode=PermissionMode.DEFAULT),
                )
                payload1 = json.loads(res1[0].text)
                self.assertEqual(payload1["evaluator"], "artifact_enhancer")
                self.assertEqual(payload1["status"], "INSIGHTS_FOUND")
                self.assertEqual(payload1["criterion_scores"]["deliverable_purity"], 0.0)
                self.assertTrue(any("DELIVERABLE_POLLUTION" in b for b in payload1["blind_spots"]))
            finally:
                pty.close_all()


if __name__ == "__main__":
    unittest.main()
