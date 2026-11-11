"""End-to-end verification of Crucible (`crux`) harness upgrades.

Covers:
1. Thread-safe `call_id` telemetry isolation across parallel `read_file` batches.
2. `batch_has_mutation` suppression of false read-spiral warnings on mixed `[read_file, edit_file]` turns.
3. Stateless `run_command` (`stdin=DEVNULL`, clean process-group timeout, `/plan` & catastrophic command gates).
4. Chunked `write_file(mode='append')` authoring with `StaleReadTracker` conflict protection.
5. First-class `render_ui` (`validate_render_target` SSRF firewall, ephemeral loopback server, WebGL/rAF probe shim, PNG + console capture).
6. Contrastive `tournament` clean-room workflow mode (native tool + MCP server).
7. Complexity-triggered Auto Pre-Ship Cold-Eyes Gate in `SessionRunner` (`STOP_HOOKS`).
8. Multi-generation `<verified_environment_facts>` persistence across `SessionDAG` compactions.
"""

from __future__ import annotations

import json
import pathlib
import struct
import tempfile
import unittest
import urllib.error
import urllib.request
import zlib

from crux_evals.mcp_server import CruxEvalsMCPServer
from ush.auth.vault import AuthProfile, CredentialVault
from ush.brain.agy_acp import _CRUX_PIPE_AGENT_MD, _render_tool_declarations
from ush.brain.context_dag import SessionDAG, extract_verified_environment_facts
from ush.brain.model_tiers import resolve_model_profile
from ush.brain.providers import ProviderTransport, UniversalModelClient
from ush.core.types import CanonicalMessage, CanonicalPart, PermissionMode, StopReason, StreamEvent
from ush.kernel.fsm import SessionRunner
from ush.kernel.patch import StaleReadTracker
from ush.kernel.pty_manager import PersistentPtyManager
from ush.kernel.security import PolicyEngine, SecurityFirewall
from ush.kernel.tools import ToolExecutionContext, build_default_tool_registry
from ush.kernel.ui_renderer import inject_render_probe_html, render_ui_capture, validate_render_target
from ush.platform.hooks_memory_ext import HookSystem


def _tiny_png_bytes() -> bytes:
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)
    raw_scanlines = b"\x00\x10\x20\x30\x40\x50\x60\x00\x70\x80\x90\xa0\xb0\xc0"
    idat = zlib.compress(raw_scanlines)
    return b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", idat) + _chunk(b"IEND", b"")


class TestHarnessUpgrades(unittest.TestCase):
    def test_parallel_read_call_id_isolation_prevents_clobbered_telemetry_and_false_loop_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            for idx, name in enumerate(("a.py", "b.py", "c.py", "d.py")):
                (ws / name).write_text(f"VAL_{idx} = {idx}\n", encoding="utf-8")

            hs = HookSystem(ws, is_workspace_trusted=True, enable_crux_evals=True)
            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            fw = SecurityFirewall(ws)
            tracker = StaleReadTracker()
            pol = PolicyEngine()
            try:
                calls = [
                    CanonicalPart(kind="tool_call", tool_call_id=f"r{i}", tool_name="read_file", tool_args={"path": fn})
                    for i, fn in enumerate(("a.py", "b.py", "c.py", "d.py"))
                ]
                results = reg.dispatch_batch(
                    calls,
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol
                    ),
                    max_parallel_reads=4,
                    before_tool_hook=hs.before_tool,
                    after_tool_hook=hs.after_tool,
                )
                self.assertEqual(len(results), 4)
                self.assertTrue(all(not r.is_error for r in results))
                self.assertEqual(set(hs.session_telemetry.read_files), {"a.py", "b.py", "c.py", "d.py"})
                sigs = {rec.action_signature for rec in hs.session_telemetry.tool_history}
                self.assertEqual(len(sigs), 4)

                # 4 distinct missing files in parallel MUST produce 4 distinct action_signatures and ZERO loop warnings!
                err_calls = [
                    CanonicalPart(kind="tool_call", tool_call_id=f"e{i}", tool_name="read_file", tool_args={"path": f"missing_{i}.py"})
                    for i in range(4)
                ]
                err_results = reg.dispatch_batch(
                    err_calls,
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol
                    ),
                    max_parallel_reads=4,
                    before_tool_hook=hs.before_tool,
                    after_tool_hook=hs.after_tool,
                )
                for r in err_results:
                    self.assertTrue(r.is_error)
                    self.assertNotIn("repeated 3 consecutive times", r.text)
            finally:
                pty.close_all()

    def test_mixed_read_and_edit_batch_suppresses_false_read_spiral_warning(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            for i in range(8):
                (ws / f"f{i}.py").write_text(f"x = {i}\n", encoding="utf-8")

            hs = HookSystem(ws, is_workspace_trusted=True, enable_crux_evals=True)
            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            fw = SecurityFirewall(ws)
            tracker = StaleReadTracker()
            pol = PolicyEngine()
            try:
                # Run 7 sequential reads first
                for i in range(7):
                    reg.dispatch_batch(
                        [CanonicalPart(kind="tool_call", tool_call_id=f"s{i}", tool_name="read_file", tool_args={"path": f"f{i}.py"})],
                        ctx_factory=lambda cid: ToolExecutionContext(
                            call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol
                        ),
                        before_tool_hook=hs.before_tool,
                        after_tool_hook=hs.after_tool,
                    )
                # Step 8+9 in ONE batch: [read_file(f7.py), edit_file(f0.py)] — must NOT warn on read_file!
                mixed_res = reg.dispatch_batch(
                    [
                        CanonicalPart(kind="tool_call", tool_call_id="s7", tool_name="read_file", tool_args={"path": "f7.py"}),
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="s8",
                            tool_name="edit_file",
                            tool_args={"path": "f0.py", "target": "x = 0", "replacement": "x = 42"},
                        ),
                    ],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.AUTO_EDIT
                    ),
                    before_tool_hook=hs.before_tool,
                    after_tool_hook=hs.after_tool,
                )
                self.assertNotIn("consecutive read/search calls", mixed_res[0].text)
                self.assertFalse(mixed_res[1].is_error)
            finally:
                pty.close_all()

    def test_run_command_stateless_closed_stdin_timeout_plan_and_policy_gates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            (ws / "sub").mkdir()
            (ws / "sub" / "hello.js").write_text("const x = 1 + 2;\n", encoding="utf-8")
            hs = HookSystem(ws, is_workspace_trusted=True, enable_crux_evals=True)
            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            fw = SecurityFirewall(ws)
            tracker = StaleReadTracker()
            pol = PolicyEngine()
            try:
                # 1. `cat` exits immediately with rc=0 because stdin=DEVNULL (never hangs waiting for EOF!)
                r_cat = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="c1", tool_name="run_command", tool_args={"command": "cat && echo OK_EOF"})],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.YOLO
                    ),
                    before_tool_hook=hs.before_tool,
                    after_tool_hook=hs.after_tool,
                )[0]
                self.assertFalse(r_cat.is_error)
                self.assertIn("OK_EOF", r_cat.text)

                # 2. Clean timeout kills process group without leaking pipes (verified under -W error::ResourceWarning)
                r_to = reg.dispatch_batch(
                    [
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="c2",
                            tool_name="run_command",
                            tool_args={"command": "sleep 20", "timeout_seconds": 0.6},
                        )
                    ],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.YOLO
                    ),
                )[0]
                self.assertTrue(r_to.is_error)
                self.assertIn("status=timed_out", r_to.text)

                # 3. Verification regex recognizes `python3 -m unittest` / `node --check` via `run_command`
                r_ver = reg.dispatch_batch(
                    [
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="c3",
                            tool_name="run_command",
                            tool_args={"command": "python3 -c 'assert 2 + 2 == 4'", "cwd": "sub"},
                        )
                    ],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.YOLO
                    ),
                    before_tool_hook=hs.before_tool,
                    after_tool_hook=hs.after_tool,
                )[0]
                self.assertFalse(r_ver.is_error)
                self.assertTrue(hs.session_telemetry.last_verification_passed)

                # 4. Plan mode denies `run_command` & YOLO still blocks catastrophic `rm -rf /`
                r_plan = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="c4", tool_name="run_command", tool_args={"command": "echo hi"})],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.PLAN
                    ),
                )[0]
                self.assertTrue(r_plan.is_error)
                self.assertIn("[POLICY_DENIED]", r_plan.text)

                r_catas = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="c5", tool_name="run_command", tool_args={"command": "rm -rf /"})],
                    ctx_factory=lambda cid: ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.YOLO
                    ),
                )[0]
                self.assertTrue(r_catas.is_error)
                self.assertIn("[POLICY_DENIED]", r_catas.text)
            finally:
                pty.close_all()

    def test_write_file_append_mode_and_stale_read_safety(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            fw = SecurityFirewall(ws)
            tracker = StaleReadTracker()
            pol = PolicyEngine()
            try:
                def _ctx(cid: str) -> ToolExecutionContext:
                    return ToolExecutionContext(
                        call_id=cid, firewall=fw, stale_tracker=tracker, pty=pty, policy_engine=pol, mode=PermissionMode.AUTO_EDIT
                    )

                r1 = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="w1", tool_name="write_file", tool_args={"path": "doc.html", "content": "<html>\n", "mode": "overwrite"})],
                    ctx_factory=_ctx,
                )[0]
                r2 = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="w2", tool_name="write_file", tool_args={"path": "doc.html", "content": "<body>Hello</body>\n", "mode": "append"})],
                    ctx_factory=_ctx,
                )[0]
                r3 = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="w3", tool_name="write_file", tool_args={"path": "doc.html", "content": "</html>\n", "mode": "append"})],
                    ctx_factory=_ctx,
                )[0]
                self.assertFalse(r1.is_error)
                self.assertFalse(r2.is_error)
                self.assertFalse(r3.is_error)
                self.assertIn("Appended", r3.text)
                self.assertEqual((ws / "doc.html").read_text(encoding="utf-8"), "<html>\n<body>Hello</body>\n</html>\n")

                # Out-of-band external mutation MUST trigger stale-read guard on append!
                (ws / "doc.html").write_text("EXTERNAL_OVERWRITE", encoding="utf-8")
                r_stale = reg.dispatch_batch(
                    [CanonicalPart(kind="tool_call", tool_call_id="w4", tool_name="write_file", tool_args={"path": "doc.html", "content": "more", "mode": "append"})],
                    ctx_factory=_ctx,
                )[0]
                self.assertTrue(r_stale.is_error)
                self.assertIn("modified externally", r_stale.text.lower())
            finally:
                pty.close_all()

    def test_render_ui_security_ssrf_ephemeral_probe_and_multimodal_png_capture(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            html_file = ws / "scene.html"
            orig_html = "<!doctype html><html><head><title>Scene</title></head><body><canvas id='gl'></canvas></body></html>"
            html_file.write_text(orig_html, encoding="utf-8")
            fw = SecurityFirewall(ws)

            # 1. SSRF & sandbox traversal blocks
            for bad in (
                "http://169.254.169.254/latest/meta-data/",
                "https://example.com/",
                "http://192.168.1.10:8080/",
                "http://127.0.0.1/no_port",
                "../../etc/passwd",
            ):
                with self.assertRaises(PermissionError):
                    validate_render_target(bad, fw)

            kind, resolved, tail = validate_render_target("scene.html?cam=overview", fw)
            self.assertEqual(kind, "file")
            self.assertEqual(resolved, html_file.resolve())
            self.assertEqual(tail, "?cam=overview")

            # 2. In-memory probe injection preserves WebGL buffer & caps rAF
            injected = inject_render_probe_html(orig_html, max_raf_frames=10, settle_ms=200)
            self.assertIn("preserveDrawingBuffer:true", injected)
            self.assertIn("data-crux-gl-mirror", injected)
            self.assertEqual(html_file.read_text(encoding="utf-8"), orig_html)

            # 3. Live ephemeral loopback HTTP server + mock browser runner producing PNG + console log
            png_data = _tiny_png_bytes()
            fetched_bodies: list[str] = []
            traversal_codes: list[int] = []

            def _mock_browser(cmd: list[str], _env: dict[str, str], _timeout: float) -> tuple[int, str, str]:
                url = cmd[-1]
                shot_arg = next(a.split("=", 1)[1] for a in cmd if a.startswith("--screenshot="))
                with urllib.request.urlopen(url, timeout=5) as resp:
                    fetched_bodies.append(resp.read().decode("utf-8"))
                base_origin = url.split("/scene.html", 1)[0]
                try:
                    urllib.request.urlopen(f"{base_origin}/..%2f..%2fetc%2fpasswd", timeout=5)
                except urllib.error.HTTPError as e:
                    traversal_codes.append(e.code)
                pathlib.Path(shot_arg).write_bytes(png_data)
                stderr = '[0101/000000:INFO:CONSOLE(12)] "Uncaught ReferenceError: fooMode is not defined", source: http://127.0.0.1/scene.html (12)\n'
                return 0, "", stderr

            ui_res = render_ui_capture(
                "scene.html?cam=overview",
                fw,
                output_path="renders/shot.png",
                width=1280,
                height=720,
                browser_runner=_mock_browser,
            )
            self.assertTrue(ui_res.success)
            self.assertEqual(ui_res.png_bytes, png_data)
            self.assertIn("preserveDrawingBuffer:true", fetched_bodies[0])
            self.assertIn(traversal_codes[0], (403, 404))
            self.assertTrue(any("fooMode is not defined" in m for m in ui_res.console_messages))
            # Disk HTML remains 100% untouched
            self.assertEqual(html_file.read_text(encoding="utf-8"), orig_html)
            self.assertIn("render_ui", PolicyEngine.READ_ONLY_TOOLS)

    def test_tournament_workflow_mode_native_and_mcp_contrastive_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            ws = pathlib.Path(td)
            (ws / "opt_a.py").write_text("def solve(xs):\n    return sum(xs) / len(xs)\n", encoding="utf-8")
            (ws / "opt_b.py").write_text("def solve(xs):\n    return sum(xs) / len(xs) if xs else 0.0\n", encoding="utf-8")

            hs = HookSystem(ws, is_workspace_trusted=True, enable_crux_evals=True)
            seen_prompts: list[tuple[str, str]] = []

            def _judge(sys_p: str, usr_p: str, _fam: str | None) -> str:
                seen_prompts.append((sys_p, usr_p))
                return json.dumps(
                    {
                        "status": "FINDINGS",
                        "winner": "opt_b.py",
                        "summary": "opt_b.py wins by guarding empty input without extra state.",
                        "findings": [
                            {
                                "target": "opt_a.py",
                                "issue": "Raises ZeroDivisionError on empty list `[]`.",
                                "fix": "Adopt `opt_b.py` empty-sequence guard.",
                            }
                        ],
                    }
                )

            res = hs.consult_evaluator(
                lens="tournament",
                objective="Choose safer mean implementation",
                target_paths=["opt_a.py", "opt_b.py"],
                proposed_claims="Option A: direct division. Option B: conditional empty guard.",
                llm_judge=_judge,
            )
            self.assertEqual(res["evaluator"], "tournament")
            self.assertEqual(res["workflow_phase"], "contrastive_selection")
            self.assertEqual(res["winner"], "opt_b.py")
            self.assertIn("<candidate_proposals>", seen_prompts[0][1])
            self.assertIn("Option B", seen_prompts[0][1])

            # Also verify MCP server lists & executes `tournament`
            mcp = CruxEvalsMCPServer(workspace_root=ws)
            tools = {t["name"] for t in mcp.list_tools()}
            self.assertIn("tournament", tools)
            rpc = mcp.handle_jsonrpc(
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "tournament", "arguments": {"files": ["opt_b.py"], "goal": "Verify candidate B"}},
                }
            )
            self.assertFalse(rpc["result"]["isError"])
            self.assertEqual(rpc["result"]["structuredContent"]["evaluator"], "tournament")

    def test_auto_pre_ship_cold_eyes_gate_catches_polluted_ui_without_manual_consult(self) -> None:
        class _AutoGateTransport(ProviderTransport):
            def __init__(self) -> None:
                self.calls = 0
                self.seen_last_texts: list[str] = []

            def stream_turn(self, **kwargs):  # type: ignore[no-untyped-def]
                self.calls += 1
                hist = kwargs.get("messages") or kwargs.get("history") or []
                self.seen_last_texts.append(hist[-1].plain_text())
                if self.calls == 1:
                    # Actor writes polluted HTML with leaked scorecard & missing viewport
                    return CanonicalMessage(
                        role="assistant",
                        parts=[
                            CanonicalPart(
                                kind="tool_call",
                                tool_call_id="tc1",
                                tool_name="write_file",
                                tool_args={
                                    "path": "viz.html",
                                    "content": "<html><body><h1>Crux-Evals Scorecard</h1><p>TODO: finish chart</p></body></html>",
                                },
                            )
                        ],
                        stop_reason=StopReason.STOP,
                    )
                if self.calls == 2:
                    # Actor tries to STOP prematurely WITHOUT calling consult_evaluator!
                    return CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="text", text="Shipped viz.html!")],
                        stop_reason=StopReason.STOP,
                    )
                if self.calls == 3:
                    # Auto Pre-Ship Gate caught it! Actor fixes viz.html.
                    return CanonicalMessage(
                        role="assistant",
                        parts=[
                            CanonicalPart(
                                kind="tool_call",
                                tool_call_id="tc2",
                                tool_name="write_file",
                                tool_args={
                                    "path": "viz.html",
                                    "content": (
                                        "<!doctype html><html><head><meta name='viewport' content='width=device-width,initial-scale=1'></head>"
                                        "<body><canvas id='c'></canvas><button onclick='step()'>Step</button></body></html>"
                                    ),
                                },
                            )
                        ],
                        stop_reason=StopReason.STOP,
                    )
                return CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text="Clean interactive viz.html delivered.")],
                    stop_reason=StopReason.STOP,
                )

        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()
            vault = CredentialVault(home / "creds.json")
            auth = AuthProfile(name="g", provider="google", auth_kind="api_key", base_url="https://generativelanguage.googleapis.com", default_model="gemini-2.5-pro", source="none")
            hs = HookSystem(ws, is_workspace_trusted=True, ush_home=home, enable_crux_evals=True)
            tr = _AutoGateTransport()
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(vault, tr),
                auth_profile=auth,
                model_profile=resolve_model_profile("gemini-2.5-pro", "google"),
                mode=PermissionMode.AUTO_EDIT,
                ush_home=home,
                hook_runner=hs,
            )
            events: list[StreamEvent] = []
            try:
                out = runner.run_turn("Build clean interactive viz.html", on_event=events.append)
                self.assertEqual(out.steps_executed, 4)
                self.assertTrue(any(e.meta.get("event") == "auto_pre_ship_cold_eyes" for e in events))
                self.assertTrue(any("[AUTO_PRE_SHIP_COLD_EYES]" in t and "DELIVERABLE_POLLUTION" in t for t in tr.seen_last_texts))
                self.assertNotIn("Scorecard", (ws / "viz.html").read_text(encoding="utf-8"))
            finally:
                runner.close()

    def test_verified_environment_facts_survive_multi_generation_compaction_and_acp_host_guard(self) -> None:
        # Verify ACP pipe prompt explicitly forbids host-injected manage_task calls
        self.assertIn("default_api:manage_task", _CRUX_PIPE_AGENT_MD)
        decl = _render_tool_declarations([{"name": "read_file", "description": "Read", "input_schema": {"type": "object"}}])
        self.assertIn("default_api:manage_task", decl)

        # Verify multi-generation compaction preserves <verified_environment_facts>
        dag = SessionDAG()
        dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Run playwright screenshot and tests")]))
        dag.append(
            CanonicalMessage(
                role="assistant",
                parts=[
                    CanonicalPart(kind="tool_call", tool_call_id="k1", tool_name="run_command", tool_args={"command": "python3 -c 'import playwright'"}),
                    CanonicalPart(kind="tool_call", tool_call_id="k2", tool_name="read_file", tool_args={"path": "../outside.txt"}),
                    CanonicalPart(kind="tool_call", tool_call_id="k3", tool_name="run_command", tool_args={"command": "pytest -q"}),
                ],
            )
        )
        dag.append(
            CanonicalMessage(
                role="tool",
                parts=[
                    CanonicalPart(kind="tool_result", tool_call_id="k1", tool_name="run_command", text="Traceback:\nModuleNotFoundError: No module named 'playwright'", is_error=True),
                    CanonicalPart(kind="tool_result", tool_call_id="k2", tool_name="read_file", text="[PermissionError]: Sandbox boundary violation: ../outside.txt", is_error=True),
                    CanonicalPart(kind="tool_result", tool_call_id="k3", tool_name="run_command", text="[status=completed exit_code=0]\n12 passed in 0.4s", is_error=False),
                ],
            )
        )
        dag.append(CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Playwright missing; pytest passed.")]))
        dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Continue phase 2")]))
        dag.append(CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Working on phase 2.")]))

        self.assertTrue(dag.maybe_compact(context_limit_tokens=100, force=True))
        gen1_text = "\n".join(m.plain_text() for m in dag.active_messages())
        self.assertIn("<verified_environment_facts>", gen1_text)
        self.assertIn("No module named 'playwright'", gen1_text)
        self.assertIn("Sandbox boundary violation", gen1_text)
        self.assertIn("pytest -q", gen1_text)

        # Append more turns and compact a SECOND time using a custom summarizer that omits facts —
        # `maybe_compact` MUST carry `<verified_environment_facts>` forward lossless into Generation 2!
        for step in range(4):
            dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text=f"Follow-up {step}")]))
            dag.append(CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text=f"Done {step}")]))

        self.assertTrue(dag.maybe_compact(context_limit_tokens=100, summarizer=lambda _msgs: "Custom high-level summary.", force=True))
        gen2_msgs = dag.active_messages()
        gen2_facts = extract_verified_environment_facts(gen2_msgs)
        joined_facts = " | ".join(gen2_facts)
        self.assertIn("No module named 'playwright'", joined_facts)
        self.assertIn("Sandbox boundary violation", joined_facts)
        self.assertIn("pytest -q", joined_facts)


if __name__ == "__main__":
    unittest.main()
