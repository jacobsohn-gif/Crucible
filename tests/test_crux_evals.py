"""Comprehensive Test Suite for Crux-Evals (Crucible Assay):
- 5 Core Evaluators (Deterministic, Hybrid, Blind BARS Judge, Adversarial Falsifier)
- Open-Protocol MCP Server (`initialize`, `tools/list`, `tools/call`, `resources/read`, Epistemic Firewall)
- Plug-and-Play Extensibility (`EvaluatorRegistry` auto-discovery + `scaffold_new_evaluator` CLI)
- Complementary vs. Strict Lifecycle Hook Adapter (`cli_hook.py`)
"""

from __future__ import annotations

import json
import os
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

    def test_skill_style_markdown_evaluator_authoring_isolation_and_trust_gate(self) -> None:
        """Verifies skill-like EVALUATOR.md plugins respect workspace trust, unlock by `paths:`, update wire enum, and enforce blind isolation."""
        from crux_evals.core.protocol import EvaluationFrame, MarkdownPromptEvaluator
        from ush.kernel.tools import build_default_tool_registry
        from ush.platform.hooks_memory_ext import HookSystem

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ws = root / "ws"
            home = root / "home"
            (ws / "queries").mkdir(parents=True)
            sql_file = ws / "queries" / "report.sql"
            sql_file.write_text("SELECT * FROM events;", encoding="utf-8")

            ws_plugins = ws / ".crux_evals" / "plugins"
            created = scaffold_new_evaluator(
                "sql_guard",
                name="SQL Query Audit",
                target_root=ws_plugins,
                description="Audits SQL queries for missing WHERE/LIMIT guards.",
                when_to_use="When creating or editing SQL query files.",
                paths=["*.sql"],
                style="markdown",
            )
            self.assertTrue((created / "EVALUATOR.md").is_file())
            self.assertFalse((created / "module.py").exists())

            # 1. Untrusted workspace MUST ignore workspace evaluators!
            hs_untrusted = HookSystem(ws, is_workspace_trusted=False, ush_home=home)
            self.assertNotIn("sql_guard", hs_untrusted.all_evaluator_ids())

            # 2. Trusting workspace + reloading discovers `sql_guard` just like a skill
            hs = HookSystem(ws, is_workspace_trusted=True, ush_home=home)
            self.assertIn("sql_guard", hs.all_evaluator_ids())

            # `sql_guard` has `paths: ["*.sql"]`, so it is excluded from Turn-0 `<available_evaluators>`
            t0_catalog = hs.initial_evaluator_catalog()
            self.assertIn("trajectory_sentinel", t0_catalog)
            self.assertNotIn("sql_guard", t0_catalog)

            # Touching `queries/report.sql` unlocks `<evaluators_delta>`
            delta = hs.on_file_accessed_evaluator_delta(sql_file)
            self.assertIn("<evaluators_delta>", delta)
            self.assertIn("sql_guard", delta)

            # Live ToolRegistry wire schema dynamically exposes custom plugin in `lens` enum!
            reg = build_default_tool_registry()
            schemas = reg.wire_schemas(evaluator_enum=hs.all_evaluator_ids())
            ce_schema = next(s for s in schemas if s["name"] == "consult_evaluator")
            self.assertIn("sql_guard", ce_schema["input_schema"]["properties"]["lens"]["enum"])

            # 3. Declarative MarkdownPromptEvaluator enforces blind isolation (`actor_claim_text` stripped!)
            captured_prompts: list[tuple[str, str]] = []

            def _mock_judge(sys_p: str, usr_p: str, _fam: str | None) -> str:
                captured_prompts.append((sys_p, usr_p))
                return json.dumps(
                    {
                        "analysis": "Unbounded full-table SELECT without WHERE or LIMIT.",
                        "bars": 0.5,
                        "criterion_scores": {"query_safety": 0.5},
                        "blind_spots": ["Missing partition filter on events table."],
                        "suggested_next_probes": ["Add WHERE event_date >= ... and LIMIT."],
                    }
                )

            manifest = next(m for m in hs.eval_registry.list_manifests() if m.evaluator_id == "sql_guard")
            from crux_evals.core.protocol import EvaluatorManifest

            _, body = EvaluatorManifest.from_markdown_file(created / "EVALUATOR.md")
            md_ev = MarkdownPromptEvaluator(manifest=manifest, rubric_body=body, llm_judge=_mock_judge)
            frame = EvaluationFrame(
                session_id="s_md",
                workspace_root=str(ws),
                event="MCPConsult",
                step_index=1,
                task_objective="Optimize weekly events query",
                modified_files=("queries/report.sql",),
                actor_claim_text="SECRET_ACTOR_COT_DO_NOT_LEAK",
            )
            verdict = md_ev.evaluate(frame)
            self.assertEqual(verdict.decision, "WARN")
            self.assertIn("Unbounded full-table SELECT", verdict.evaluator_analysis)
            self.assertEqual(len(captured_prompts), 1)
            self.assertNotIn("SECRET_ACTOR_COT_DO_NOT_LEAK", captured_prompts[0][1])
            self.assertIn("SELECT * FROM events;", captured_prompts[0][1])

    def test_model_vs_evaluator_hardline_isolation_live_telemetry_and_usefulness_nudges(self) -> None:
        """Verifies live SessionRunner locks human objective, routes isolated cross-family evaluator calls with tools=[] and separate thought/analysis, and injects usefulness reminders."""
        from ush.auth.vault import CredentialVault
        from ush.brain.model_tiers import resolve_model_profile
        from ush.brain.providers import UniversalModelClient
        from ush.core.types import CanonicalMessage, CanonicalPart, PermissionMode, StopReason, StreamEvent
        from ush.kernel.fsm import SessionRunner
        from ush.platform.hooks_memory_ext import HookSystem

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()
            (ws / "math_util.py").write_text("def avg(xs: list[float]) -> float:\n    return sum(xs) / len(xs)\n", encoding="utf-8")

            vault = CredentialVault(home, probe_ambient=False)
            g_auth = vault.register_api_key_profile("g-actor", "google", "AIzaMockGoogleKey123456789012345678", default_model="gemini-2.5-pro")
            c_auth = vault.register_api_key_profile("c-judge", "anthropic", "sk-ant-MockClaudeKey123456789012345", default_model="claude-sonnet-4")
            self.assertIsNotNone(c_auth)

            actor_calls: list[dict] = []
            eval_calls: list[dict] = []

            class _IsolatedSplitTransport:
                allow_isolated_evaluator = True

                def __init__(self) -> None:
                    self.actor_step = 0

                def stream_turn(self, *, profile, auth_profile, system_prompt, messages, tools, session_id, on_event=None, **_kw):
                    if session_id.startswith("crux_eval_isolated:"):
                        eval_calls.append(
                            {
                                "provider": auth_profile.provider,
                                "model": profile.model_id,
                                "tools": list(tools),
                                "session_id": session_id,
                                "msg_count": len(messages),
                                "user_text": messages[-1].plain_text(),
                                "system_prompt": system_prompt,
                            }
                        )
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=auth_profile.provider,
                            producer_model_id=profile.model_id,
                            parts=[
                                CanonicalPart(kind="thinking", text="ISOLATED_EVALUATOR_THOUGHT: checking division on empty xs=[]"),
                                CanonicalPart(
                                    kind="text",
                                    text=json.dumps(
                                        {
                                            "analysis": "avg([]) raises ZeroDivisionError when xs is empty.",
                                            "bars": {
                                                "functional_completeness": 0.5,
                                                "edge_case_resilience": 0.0,
                                                "surgical_minimality": 1.0,
                                                "verification_sufficiency": 0.0,
                                            },
                                            "hard_vetoes": [],
                                            "uncertainty": 0.1,
                                            "failed_checks": ["Empty list edge case unhandled in avg(xs)"],
                                            "remediation": "Guard len(xs) == 0 cleanly.",
                                        }
                                    ),
                                ),
                            ],
                            stop_reason=StopReason.STOP,
                        )

                    self.actor_step += 1
                    actor_calls.append(
                        {
                            "system_prompt": system_prompt,
                            "tools": list(tools),
                            "last_msg": "\n".join(p.text or "" for p in messages[-1].parts),
                        }
                    )
                    if self.actor_step == 1:
                        return CanonicalMessage(
                            role="assistant",
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="tc_eval_1",
                                    tool_name="consult_evaluator",
                                    tool_args={
                                        "lens": "blind_bars_judge",
                                        "objective": "ACTOR_SELF_SERVING_REWRITE",
                                        "target_paths": ["math_util.py"],
                                        "proposed_claims": "ACTOR_INTERNAL_COT_LEAK",
                                    },
                                )
                            ],
                            stop_reason=StopReason.STOP,
                        )
                    return CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="text", text="Guarded empty input cleanly after evaluator review.")],
                        stop_reason=StopReason.STOP,
                    )

            hs = HookSystem(ws, is_workspace_trusted=True, ush_home=home, enable_crux_evals=True, crux_evals_mode="complementary")
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(vault, _IsolatedSplitTransport()),
                auth_profile=g_auth,
                model_profile=resolve_model_profile("gemini-2.5-pro", "google"),
                mode=PermissionMode.AUTO_EDIT,
                ush_home=home,
                hook_runner=hs,
            )
            events: list[StreamEvent] = []
            try:
                outcome = runner.run_turn("HUMAN_VERBATIM_OBJECTIVE: Fix empty-input crash in math_util.py", on_event=events.append)
                self.assertEqual(outcome.steps_executed, 2)

                # Turn-0 system prompt teaches usefulness-based evaluator selection & lists available evaluators
                self.assertIn("consult_evaluator", actor_calls[0]["system_prompt"])
                self.assertIn("<available_evaluators>", actor_calls[0]["system_prompt"])

                # Isolated evaluator call used DIFFERENT model family (`anthropic`), tools=[], zero Actor CoT, and HUMAN objective!
                self.assertEqual(len(eval_calls), 1)
                self.assertEqual(eval_calls[0]["provider"], "anthropic")
                self.assertEqual(eval_calls[0]["tools"], [])
                self.assertEqual(eval_calls[0]["msg_count"], 1)
                self.assertIn("HUMAN_VERBATIM_OBJECTIVE", eval_calls[0]["user_text"])
                self.assertNotIn("ACTOR_SELF_SERVING_REWRITE", eval_calls[0]["user_text"])
                self.assertNotIn("ACTOR_INTERNAL_COT_LEAK", eval_calls[0]["user_text"])

                # Actor saw structured analysis returned from consult_evaluator without leaking into Turn-1 CoT
                self.assertIn("ZeroDivisionError", actor_calls[1]["last_msg"])
                self.assertIn("claude-sonnet-4", actor_calls[1]["last_msg"])

                # Evaluator thought & analysis were captured cleanly as a separate cognitive_event
                eval_evts = [e for e in events if e.meta.get("event") == "crux_evals_analysis"]
                self.assertEqual(len(eval_evts), 1)
                self.assertTrue(eval_evts[0].meta["cross_family"])
                self.assertIn("ISOLATED_EVALUATOR_THOUGHT", eval_evts[0].meta["evaluator_thought"])
            finally:
                runner.close()

    def test_generalized_artifact_enhancer_and_cli_eval_slash_commands(self) -> None:
        """Verifies m06_artifact_enhancer is domain-agnostic and `/evaluators` + `crux eval` + BUILTIN_SKILLS mirror skills."""
        from crux_evals.core.protocol import EvaluationFrame
        from crux_evals.modules.m06_artifact_enhancer.module import ArtifactEnhancerEvaluator
        from ush.cli.main import HarnessRuntime, handle_slash_command, main as crux_cli_main
        from ush.core.types import PermissionMode
        from ush.platform.skills import BUILTIN_SKILLS

        skill_names = {s.name for s in BUILTIN_SKILLS}
        self.assertIn("crux-evals", skill_names)
        self.assertIn("create-evaluator", skill_names)

        ev = ArtifactEnhancerEvaluator()
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()

            # Clean interactive dashboard with non-BigInt domain -> MUST PASS without domain-specific Ozaki advice!
            clean_dash = ws / "dashboard.html"
            clean_dash.write_text(
                '<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1"></head>'
                '<body><canvas id="c"></canvas><input type="range" id="s" oninput="draw()"><script>function draw(){}</script></body></html>',
                encoding="utf-8",
            )
            v_clean = ev.evaluate(
                EvaluationFrame(
                    session_id="g1",
                    workspace_root=str(ws),
                    event="MCPConsult",
                    step_index=1,
                    task_objective="Build interactive climate chart dashboard",
                    modified_files=("dashboard.html",),
                )
            )
            self.assertEqual(v_clean.decision, "ALLOW")
            self.assertNotIn("Ozaki", v_clean.remediation_directive)
            self.assertNotIn("BigInt", v_clean.remediation_directive)

            # Dashboard with placeholder text + non-interactive canvas -> caught cleanly
            stub_dash = ws / "stub.html"
            stub_dash.write_text("<html><body><p>TODO: replace Lorem ipsum chart</p><canvas></canvas></body></html>", encoding="utf-8")
            v_stub = ev.evaluate(
                EvaluationFrame(
                    session_id="g2",
                    workspace_root=str(ws),
                    event="MCPConsult",
                    step_index=1,
                    task_objective="Sales analytics page",
                    modified_files=("stub.html",),
                )
            )
            self.assertEqual(v_stub.decision, "WARN")
            self.assertTrue(any("STATIC_PLACEHOLDER_DATA" in b for b in v_stub.blind_spots))

            # `/evaluators new ...` + `crux eval list` work like `/skills` with USH_HOME isolation
            env_patch = {"USH_HOME": str(home), "USH_DISABLE_KEYCHAIN": "1", "GEMINI_API_KEY": "AIzaMock12345678901234567890"}
            old_env = {k: os.environ.get(k) for k in env_patch}
            os.environ.update(env_patch)
            try:
                rt = HarnessRuntime(ws, mode=PermissionMode.DEFAULT)
                try:
                    handle_slash_command(rt, "/evaluators new perf_lens --md")
                    self.assertIn("perf_lens", rt.hook_system.all_evaluator_ids())
                    self.assertTrue((home / "evaluators" / "perf_lens" / "EVALUATOR.md").is_file())
                    handle_slash_command(rt, "/reload")
                    self.assertIn("perf_lens", rt.hook_system.all_evaluator_ids())
                finally:
                    rt.runner.close()

                rc = crux_cli_main(["--cwd", str(ws), "eval", "list"])
                self.assertEqual(rc, 0)
            finally:
                for k, v in old_env.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

    def test_situational_usefulness_nudges_and_complementary_stop_crash_advisory(self) -> None:
        """Verifies thrash-streak evaluator hints and 1-shot complementary Stop advisory when `/tmp` crash is triggered."""
        from ush.auth.vault import CredentialVault
        from ush.brain.model_tiers import resolve_model_profile
        from ush.brain.providers import UniversalModelClient
        from ush.core.types import CanonicalMessage, CanonicalPart, PermissionMode, StopReason
        from ush.kernel.fsm import SessionRunner
        from ush.platform.hooks_memory_ext import HookSystem

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()

            vault = CredentialVault(home, probe_ambient=False)
            auth = vault.register_api_key_profile("g-test", "google", "AIzaMockGoogleKey123456789012345678", default_model="gemini-2.5-pro")
            seen_msgs: list[str] = []

            class _ThrashThenCrashTransport:
                def __init__(self) -> None:
                    self.step = 0

                def stream_turn(self, *, messages, **_kw):
                    self.step += 1
                    seen_msgs.append("\n".join(p.text or "" for p in messages[-1].parts))
                    if self.step <= 2:
                        return CanonicalMessage(
                            role="assistant",
                            parts=[CanonicalPart(kind="tool_call", tool_call_id=f"rf_{self.step}", tool_name="read_file", tool_args={"path": "missing_file.py"})],
                            stop_reason=StopReason.STOP,
                        )
                    if self.step == 3:
                        return CanonicalMessage(
                            role="assistant",
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="wf_1",
                                    tool_name="write_file",
                                    tool_args={"path": "ratio.py", "content": "def ratio(xs: list[float]) -> float:\n    return sum(xs) / len(xs)\n"},
                                )
                            ],
                            stop_reason=StopReason.STOP,
                        )
                    if self.step == 4:
                        return CanonicalMessage(
                            role="assistant",
                            parts=[CanonicalPart(kind="text", text="Implemented ratio.py!")],
                            stop_reason=StopReason.STOP,
                        )
                    return CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="text", text="Acknowledged advisory; will guard empty xs and run tests.")],
                        stop_reason=StopReason.STOP,
                    )

            hs = HookSystem(ws, is_workspace_trusted=True, ush_home=home, enable_crux_evals=True, crux_evals_mode="complementary")
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(vault, _ThrashThenCrashTransport()),
                auth_profile=auth,
                model_profile=resolve_model_profile("gemini-2.5-pro", "google"),
                mode=PermissionMode.AUTO_EDIT,
                ush_home=home,
                hook_runner=hs,
            )
            try:
                out = runner.run_turn("Create ratio.py cleanly")
                self.assertEqual(out.steps_executed, 5)
                self.assertIn("trajectory_sentinel", seen_msgs[2])
                self.assertTrue(any("[CRUX_EVALS_ADVISORY]" in m and "ZeroDivisionError" in m for m in seen_msgs))
                self.assertIn("Acknowledged advisory", out.final_message.plain_text())
            finally:
                runner.close()


if __name__ == "__main__":
    unittest.main()
