"""Complete 20-Gate Deterministic Conformance, Chaos & Security Suite (`CT-01` .. `CT-20`).

Exercises every subsystem in Universal Swarm Harness (`ush`) against real filesystems,
real Unix PTYs, real out-of-tree Shadow Git & Git Worktrees, real SQLite WAL ledgers,
and deterministic mock wire transports in <10 seconds with 0 flaky network calls.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import pathlib
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable
import unittest

from ush.auth.vault import AuthProfile, CredentialVault
from ush.brain.context_dag import PopperianLoopGuard, SessionDAG
from ush.brain.model_tiers import resolve_model_profile
from ush.brain.providers import (
    UniversalModelClient,
    build_anthropic_payload,
    build_gemini_payload,
    build_openai_chat_payload,
)
from ush.brain.wire import (
    ToolIdBijectionMap,
    extract_fallback_tool_calls,
    transform_messages,
    wrap_dynamic_instructions,
)
from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    ModelCapabilityProfile,
    PermissionMode,
    PolicyDecision,
    StopReason,
    StreamEvent,
    TokenUsage,
    canonical_json,
)
from ush.kernel.fsm import SessionRunner
from ush.kernel.patch import StaleReadTracker, edit_file_safe
from ush.kernel.pty_manager import PersistentPtyManager
from ush.kernel.security import PolicyEngine, RepoTrustStore, SecurityFirewall, scrub_child_environment
from ush.kernel.shadow_git import ShadowGitStore
from ush.kernel.tools import ToolExecutionContext, build_default_tool_registry
from ush.platform.commands import SlashCommandRegistry, expand_command_template
from ush.platform.hooks_memory_ext import HookSystem, MemoryManager
from ush.platform.skills import SkillManager
from ush.swarm.engine import ElasticSwarmCoordinator, StigmergicBlackboard, SwarmInvariantError
from ush.swarm.judge import CandidatePatch, CascadingJudge


class ScriptedTransport:
    """Deterministic programmable ProviderTransport recording exact wire payloads per step."""

    def __init__(self, handler: Callable[[int, ModelCapabilityProfile, list[CanonicalMessage], list[dict[str, Any]]], CanonicalMessage]) -> None:
        self.handler = handler
        self.calls: list[dict[str, Any]] = []

    def stream_turn(
        self,
        *,
        profile: ModelCapabilityProfile,
        auth_profile: AuthProfile,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]],
        session_id: str,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CanonicalMessage:
        idx = len(self.calls)
        self.calls.append(
            {
                "profile": profile,
                "auth": auth_profile,
                "system": system_prompt,
                "messages": messages,
                "tools": tools,
            }
        )
        return self.handler(idx, profile, messages, tools)


def _dummy_auth(provider: str = "vertex") -> AuthProfile:
    return AuthProfile(
        name=f"mock-{provider}",
        provider=provider,
        auth_kind="none",
        base_url="http://127.0.0.1",
        project_id="mock-proj",
    )


class ConformanceSuiteCT01ToCT20(unittest.TestCase):
    # ------------------------------------------------------------------
    # CT-01: Cross-Provider Mid-Session Rotation (12-turn multimodal+tool)
    # ------------------------------------------------------------------
    def test_ct01_cross_provider_mid_session_rotation(self) -> None:
        long_oai_id = "call_" + ("x" * 120) + "|item_" + ("y" * 120)
        history: list[CanonicalMessage] = [
            CanonicalMessage(
                role="user",
                parts=[
                    CanonicalPart(kind="text", text="Inspect diagram and patch config"),
                    CanonicalPart(kind="image", mime_type="image/png", base64_data="iVBORw0KGgo="),
                ],
            ),
            CanonicalMessage(
                role="assistant",
                producer_provider="vertex",
                producer_model_id="gemini-2.5-pro",
                parts=[
                    CanonicalPart(kind="thinking", text="Gemini internal thought", signature="gem_sig_123"),
                    CanonicalPart(kind="tool_call", tool_call_id=long_oai_id, tool_name="read_file", tool_args={"path": "a.py"}, signature="gem_sig_123"),
                ],
            ),
            CanonicalMessage(
                role="tool",
                parts=[
                    CanonicalPart(
                        kind="tool_result",
                        tool_call_id=long_oai_id,
                        tool_name="read_file",
                        text="Error screenshot attached",
                        mime_type="image/png",
                        base64_data="iVBORw0KGgo=",
                        is_error=True,
                    )
                ],
            ),
            CanonicalMessage(
                role="assistant",
                producer_provider="anthropic",
                producer_model_id="claude-opus-4-20250514",
                parts=[
                    CanonicalPart(kind="thinking", text="Claude signed thought", signature="anth_sig_999"),
                    CanonicalPart(kind="text", text="Applied analysis."),
                ],
            ),
        ]
        # Pad to 12 turns cleanly
        for i in range(4):
            history.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text=f"Followup {i}")]))
            history.append(
                CanonicalMessage(
                    role="assistant",
                    producer_provider="openai",
                    producer_model_id="gpt-5",
                    parts=[CanonicalPart(kind="text", text=f"Ack {i}")],
                )
            )
        self.assertEqual(len(history), 12)

        id_map = ToolIdBijectionMap()
        # 1. Rotate to Anthropic: foreign Gemini thinking scrubbed, >64-char ID shortened, error image hoisted!
        anth_prof = resolve_model_profile("claude-opus-4-20250514", "anthropic")
        anth_msgs = transform_messages(history, anth_prof, id_map)
        anth_wire = build_anthropic_payload("sys", anth_msgs, [], anth_prof)
        serialized_anth = json.dumps(anth_wire)
        self.assertNotIn("gem_sig_123", serialized_anth)
        self.assertIn("anth_sig_999", serialized_anth)  # Same-model Claude signature preserved!
        for m in anth_wire["messages"]:
            for b in m["content"]:
                if b["type"] in ("tool_use", "tool_result"):
                    tid = b.get("id") or b.get("tool_use_id")
                    self.assertLessEqual(len(tid), 64)
                if b["type"] == "tool_result" and b.get("is_error"):
                    self.assertTrue(all(cb["type"] != "image" for cb in b["content"]))

        # 2. Rotate to Gemini -> OpenAI -> non-vision Local OSS (`deepseek-coder:7b`)
        gem_prof = resolve_model_profile("gemini-2.5-pro", "vertex")
        gem_wire = build_gemini_payload("sys", transform_messages(history, gem_prof, id_map), [], gem_prof)
        self.assertNotIn("anth_sig_999", json.dumps(gem_wire))
        self.assertIn("gem_sig_123", json.dumps(gem_wire))

        oai_prof = resolve_model_profile("deepseek-r1", "openrouter")
        oai_wire = build_openai_chat_payload("sys", transform_messages(history, oai_prof, id_map), [], oai_prof)
        self.assertNotIn("gem_sig_123", json.dumps(oai_wire))
        self.assertTrue(any("does not support vision" in json.dumps(m) for m in oai_wire["messages"]))

    # ------------------------------------------------------------------
    # CT-02: EMPTY_200_RESPONSE & MAX_TOKENS Truncated-Tool Auto-Continue
    # ------------------------------------------------------------------
    def test_ct02_empty_200_and_max_tokens_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            ws.mkdir()
            home = pathlib.Path(tmp) / "home"
            vault = CredentialVault(home, probe_ambient=False)

            seq = [
                # Attempt 0: Empty HTTP 200 -> retried transparently inside UniversalModelClient
                CanonicalMessage(role="assistant", parts=[], stop_reason=StopReason.EMPTY_200_RESPONSE),
                # Attempt 1: Hits MAX_TOKENS (partial tool call already dropped) -> SessionRunner auto-continues
                CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text="Part 1...")],
                    stop_reason=StopReason.MAX_TOKENS,
                ),
                # Step 2 continuation: finishes cleanly
                CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text="Part 2 complete.")],
                    stop_reason=StopReason.STOP,
                ),
            ]
            transport = ScriptedTransport(lambda i, _p, _m, _t: seq[i])
            client = UniversalModelClient(vault, transport)
            runner = SessionRunner(workspace_root=ws, client=client, auth_profile=_dummy_auth(), ush_home=home)
            try:
                out = runner.run_turn("Explain design")
                self.assertEqual(out.final_message.plain_text(), "Part 2 complete.")
                self.assertEqual(len(transport.calls), 3)
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-03: Prefix KV-Cache Byte Stability & Tail-Only `<dynamic_instructions>`
    # ------------------------------------------------------------------
    def test_ct03_prefix_kv_cache_byte_stability(self) -> None:
        prof = resolve_model_profile("gpt-5", "openai")
        history: list[CanonicalMessage] = []
        prev_prefix_bytes: list[str] = []

        for step in range(1, 21):
            u = CanonicalMessage(
                role="user",
                parts=[
                    CanonicalPart(kind="text", text=f"User turn {step}"),
                    wrap_dynamic_instructions(f"ephemeral_clock_or_skill_delta_{step}"),
                ],
            )
            a = CanonicalMessage(
                role="assistant",
                producer_provider="openai",
                producer_model_id="gpt-5",
                parts=[CanonicalPart(kind="text", text=f"Assistant reply {step}")],
            )
            history.append(u)
            wire = build_openai_chat_payload("Turn0 System", transform_messages(history, prof), [], prof)
            msgs = wire["messages"]
            # Only final user turn carries dynamic instructions; all historical turns 0..k-1 are byte-identical!
            serialized_items = [canonical_json(m) for m in msgs]
            if prev_prefix_bytes:
                self.assertEqual(serialized_items[: len(prev_prefix_bytes)], prev_prefix_bytes)
            self.assertIn(f"ephemeral_clock_or_skill_delta_{step}", serialized_items[-1])
            for older in serialized_items[:-1]:
                self.assertNotIn("dynamic_instructions", older)
            history.append(a)
            # Recompute stable prefix after dynamic tag fades into history
            post_wire = build_openai_chat_payload(
                "Turn0 System",
                transform_messages(history + [CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="probe")])], prof),
                [],
                prof,
            )
            prev_prefix_bytes = [canonical_json(m) for m in post_wire["messages"][:-1]]

    # ------------------------------------------------------------------
    # CT-04: Tier-B Malformed Tool-Call Fallback & 1-Turn Repair
    # ------------------------------------------------------------------
    def test_ct04_tier_b_malformed_tool_call_repair(self) -> None:
        # 1. XML + trailing comma extraction
        raw_text = (
            "<scratchpad>Reading file</scratchpad>\n"
            '<tool_call>{"name": "read_file", "arguments": {"path": "hello.py",}}</tool_call>'
        )
        cleaned, calls = extract_fallback_tool_calls(raw_text, {"read_file", "edit_file"})
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].tool_name, "read_file")
        self.assertEqual(calls[0].tool_args, {"path": "hello.py"})
        self.assertNotIn("<tool_call>", cleaned)

        # 2. IMPROPER_FORMAT recovery loop inside SessionRunner
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            ws.mkdir()
            home = pathlib.Path(tmp) / "home"
            seq = [
                CanonicalMessage(role="assistant", parts=[], stop_reason=StopReason.IMPROPER_FORMAT),
                CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Recovered cleanly!")]),
            ]
            transport = ScriptedTransport(lambda i, _p, _m, _t: seq[i])
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), transport),
                auth_profile=_dummy_auth("local"),
                model_profile=resolve_model_profile("qwen2.5-coder:14b", "local"),
                ush_home=home,
            )
            try:
                out = runner.run_turn("Test repair")
                self.assertEqual(out.final_message.plain_text(), "Recovered cleanly!")
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-05: 4-Pass Fuzzy Patch, UTF-8 BOM & CRLF Preservation Torture
    # ------------------------------------------------------------------
    def test_ct05_four_pass_fuzzy_patch_torture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fp = pathlib.Path(tmp) / "win_unicode.py"
            # UTF-8 BOM + CRLF + typographic quotes + em-dash + 4-space indent
            original = (
                b"\xef\xbb\xbf"
                + 'def greet():\r\n    msg = \u201cHello \u2014 World\u201d   \r\n    return msg\r\n'.encode("utf-8")
            )
            fp.write_bytes(original)
            tracker = StaleReadTracker()
            tracker.record_read(fp, original)

            # Model emits ASCII quotes, ASCII hyphen, and unindented block!
            res = edit_file_safe(
                fp,
                target='msg = "Hello - World"\nreturn msg',
                replacement='msg = "Patched Cleanly"\nreturn msg.upper()',
                stale_tracker=tracker,
            )
            self.assertTrue(res.success, res.error)
            self.assertEqual(res.pass_used, "unicode_fold")
            after = fp.read_bytes()
            self.assertTrue(after.startswith(b"\xef\xbb\xbf"))  # BOM preserved!
            decoded = after[3:].decode("utf-8")
            self.assertIn("\r\n", decoded)  # CRLF preserved!
            self.assertIn('    msg = "Patched Cleanly"\r\n    return msg.upper()\r\n', decoded)  # Re-indented!

    # ------------------------------------------------------------------
    # CT-06: Deadlock-Free Sorted File Locks + SHA-256 Stale-Read Guard
    # ------------------------------------------------------------------
    def test_ct06_deadlock_free_locks_and_stale_read_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a = pathlib.Path(tmp) / "a.py"
            b = pathlib.Path(tmp) / "b.py"
            a.write_text("x = 1\n", encoding="utf-8")
            b.write_text("y = 1\n", encoding="utf-8")

            from ush.kernel.patch import GLOBAL_FILE_LOCKS

            def _worker(order: list[pathlib.Path]) -> bool:
                for _ in range(25):
                    with GLOBAL_FILE_LOCKS.acquire_sorted(order):
                        time.sleep(0.001)
                return True

            with ThreadPoolExecutor(max_workers=2) as pool:
                f1 = pool.submit(_worker, [a, b])
                f2 = pool.submit(_worker, [b, a])
                self.assertTrue(f1.result(timeout=5))
                self.assertTrue(f2.result(timeout=5))

            # Stale-read guard returns diff without clobbering external edit
            tracker = StaleReadTracker()
            tracker.record_read(a, a.read_bytes())
            a.write_text("x = 999  # external developer edit\n", encoding="utf-8")
            res = edit_file_safe(a, "x = 1", "x = 2", stale_tracker=tracker)
            self.assertFalse(res.success)
            self.assertIn("STALE_READ_CONFLICT", res.error)
            self.assertIn("external developer edit", res.error)
            self.assertIn("999", a.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # CT-07: Persistent PTY OSC 633, Interactive `[y/N]`, Flood & `is_daemon`
    # ------------------------------------------------------------------
    def test_ct07_persistent_pty_prompt_flood_and_daemon(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            sub = ws / "nested"
            sub.mkdir()
            pty = PersistentPtyManager(ws)
            try:
                # 1. `cd` + env export persist across separate calls!
                r1 = pty.execute("cd nested && export MY_FLAG=persisted_42 && pwd")
                self.assertEqual(r1.status, "completed")
                self.assertEqual(r1.exit_code, 0)
                r2 = pty.execute("echo $MY_FLAG:$(basename $PWD)")
                self.assertIn("persisted_42:nested", r2.output)

                # 2. Interactive `[y/N]` prompt yields via strong heuristic in <1.5s without hanging!
                t0 = time.monotonic()
                r_prompt = pty.execute(
                    'python3 -c "import sys; sys.stdout.write(\'Proceed [y/N] \'); sys.stdout.flush(); print(\'GOT:\'+sys.stdin.readline().strip())"',
                    wait_seconds=8.0,
                    dedicated_pty=True,
                )
                elapsed = time.monotonic() - t0
                self.assertEqual(r_prompt.status, "interactive_prompt")
                self.assertLess(elapsed, 2.5)
                r_cont = pty.write_stdin(r_prompt.pty_id, "y\n", wait_seconds=3.0)
                self.assertIn("GOT:y", r_cont.output)

                # 3. Multi-megabyte stdout flood caps cleanly at 1 MiB + disk spillover!
                r_flood = pty.execute(
                    'python3 -c "print(\'HEAD_MARK\'); print(\'X\' * 2_500_000); print(\'TAIL_MARK\')"',
                    wait_seconds=15.0,
                    preview_bytes=4096,
                )
                self.assertEqual(r_flood.status, "completed")
                self.assertIn("HEAD_MARK", r_flood.output)
                self.assertIn("TAIL_MARK", r_flood.output)
                self.assertIsNotNone(r_flood.spillover_path)
                self.assertTrue(pathlib.Path(r_flood.spillover_path or "").is_file())

                # 4. `is_daemon=True` background task never blocks quiescence barrier!
                r_bg = pty.execute("sleep 30", is_daemon=True)
                self.assertEqual(r_bg.status, "running_background")
                self.assertEqual(pty.blocking_background_count(), 0)
            finally:
                pty.close_all()

    # ------------------------------------------------------------------
    # CT-08: Drop-Safe Cancellation & Synthetic `tool_result` Orphan Repair
    # ------------------------------------------------------------------
    def test_ct08_cancellation_and_orphan_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp)
            pty = PersistentPtyManager(ws)
            reg = build_default_tool_registry()
            cancel = threading.Event()
            fw = SecurityFirewall(ws)
            pol = PolicyEngine()
            tracker = StaleReadTracker()

            calls = [
                CanonicalPart(kind="tool_call", tool_call_id="c1", tool_name="exec_pty", tool_args={"command": "sleep 20", "dedicated_pty": True}),
                CanonicalPart(kind="tool_call", tool_call_id="c2", tool_name="read_file", tool_args={"path": "missing.txt"}),
            ]

            def _trigger_cancel() -> None:
                time.sleep(0.25)
                cancel.set()

            threading.Thread(target=_trigger_cancel, daemon=True).start()
            try:
                results = reg.dispatch_batch(
                    calls,
                    lambda cid: ToolExecutionContext(
                        call_id=cid,
                        firewall=fw,
                        stale_tracker=tracker,
                        pty=pty,
                        policy_engine=pol,
                        mode=PermissionMode.YOLO,
                        cancel_event=cancel,
                    ),
                )
                self.assertEqual([r.tool_call_id for r in results], ["c1", "c2"])
                self.assertTrue(all(r.is_error for r in results))
            finally:
                pty.close_all()

    # ------------------------------------------------------------------
    # CT-09: Out-of-Tree Shadow Git Zero-Pollution Capture & Restore
    # ------------------------------------------------------------------
    def test_ct09_shadow_git_zero_pollution_restore(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            home = pathlib.Path(tmp) / "home"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "app.py").write_text("v = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "app.py"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init"],
                cwd=repo,
                check=True,
            )
            idx_before = (repo / ".git" / "index").read_bytes()
            status_before = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout

            shadow = ShadowGitStore(repo, ush_home=home)
            snap = shadow.capture("before-edits")
            # Invasive edits + extra created file
            (repo / "app.py").write_text("v = 999\n", encoding="utf-8")
            (repo / "rogue_new.py").write_text("bad = True\n", encoding="utf-8")
            shadow.restore(snap)

            self.assertEqual((repo / "app.py").read_text(encoding="utf-8"), "v = 1\n")
            self.assertFalse((repo / "rogue_new.py").exists())
            self.assertEqual((repo / ".git" / "index").read_bytes(), idx_before)
            status_after = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True).stdout
            self.assertEqual(status_after, status_before)

    # ------------------------------------------------------------------
    # CT-10 & CT-11: Popperian Loop Guard, `0.85` Compaction & DAG `LCA` Rewind
    # ------------------------------------------------------------------
    def test_ct10_and_ct11_loop_guard_compaction_and_dag_lca(self) -> None:
        guard = PopperianLoopGuard()
        self.assertIsNone(guard.record("edit_file", '{"path":"a.py"}', True, "Target not found"))
        self.assertIsNone(guard.record("edit_file", '{"path":"a.py"}', True, "Target not found"))
        nudge = guard.record("edit_file", '{"path":"a.py"}', True, "Target not found")
        self.assertIsNotNone(nudge)
        self.assertIn("streak=3", nudge or "")

        with tempfile.TemporaryDirectory() as tmp:
            dag = SessionDAG(pathlib.Path(tmp) / "sess.jsonl")
            anchor = dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Build parser")]))
            # Version mismatch rejected cleanly on stale compaction attempt
            ver, _, _ = dag.snapshot_for_compaction()
            for idx in range(10):
                dag.append(
                    CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="text", text=f"Dead-end step {idx} " + ("z" * 400))],
                        modified_files=["src/bad_branch.py"] if idx == 3 else [],
                    )
                )
            self.assertFalse(dag.commit_compaction(ver, anchor.entry_id, "stale"))
            self.assertTrue(dag.maybe_compact(context_limit_tokens=400, ratio=0.85))
            self.assertIn("<epistemic_compaction_handoff>", dag.active_messages()[0].plain_text())

            # CT-11: Rewind 10 dead-end steps back to anchor -> restores clean lineage + `<branch_lessons>`!
            lesson = dag.rewind_to(anchor.entry_id, failure_note="Regex parser choked on nested parens")
            self.assertIsNotNone(lesson)
            active = dag.active_messages()
            self.assertEqual(len(active), 2)
            self.assertIn("<branch_lessons>", active[-1].plain_text())
            self.assertIn("src/bad_branch.py", active[-1].plain_text())
            self.assertIn("Regex parser choked", active[-1].plain_text())

    # ------------------------------------------------------------------
    # CT-12: 3-Layer Security Firewall (Env Scrub + .ushignore + Trust Quarantine)
    # ------------------------------------------------------------------
    def test_ct12_three_layer_security_firewall(self) -> None:
        clean_env = scrub_child_environment(
            {"PATH": "/usr/bin", "AWS_SECRET_ACCESS_KEY": "TOP_SECRET", "OPENAI_API_KEY": "sk-1234", "SAFE_FLAG": "1"}
        )
        self.assertNotIn("AWS_SECRET_ACCESS_KEY", clean_env)
        self.assertNotIn("OPENAI_API_KEY", clean_env)
        self.assertEqual(clean_env["SAFE_FLAG"], "1")

        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "repo"
            evil_sibling = pathlib.Path(tmp) / "repo-evil"
            ws.mkdir()
            evil_sibling.mkdir()
            (ws / ".env").write_text("DB_PASS=secret\n", encoding="utf-8")
            (ws / ".env.example").write_text("DB_PASS=\n", encoding="utf-8")
            (evil_sibling / "steal.txt").write_text("nope", encoding="utf-8")

            fw = SecurityFirewall(ws)
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate(".env")
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate("../repo-evil/steal.txt")  # Partial-prefix bypass blocked!
            self.assertEqual(fw.resolve_and_validate(".env.example").name, ".env.example")

            trust = RepoTrustStore(pathlib.Path(tmp) / "home")
            (ws / ".mcp.json").write_text('{"mcpServers":{"bad":{"command":"evil"}}}', encoding="utf-8")
            self.assertFalse(trust.is_trusted(ws))
            trust.trust(ws)
            self.assertTrue(trust.is_trusted(ws))
            # Mutating `.mcp.json` automatically invalidates SHA-256 trust!
            (ws / ".mcp.json").write_text('{"mcpServers":{"bad":{"command":"evil2"}}}', encoding="utf-8")
            self.assertFalse(trust.is_trusted(ws))

    # ------------------------------------------------------------------
    # CT-13: Mid-Turn `TurnSteerQueue` Boundary Safety
    # ------------------------------------------------------------------
    def test_ct13_mid_turn_steer_boundary_safety(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            ws.mkdir()
            (ws / "a.txt").write_text("alpha\n", encoding="utf-8")
            home = pathlib.Path(tmp) / "home"
            runner_ref: dict[str, SessionRunner] = {}

            def _script(step: int, _p: Any, msgs: list[CanonicalMessage], _t: Any) -> CanonicalMessage:
                if step == 0:
                    return CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="tool_call", tool_call_id="t1", tool_name="read_file", tool_args={"path": "a.txt"})],
                    )
                # Verify steer drained STRICTLY after `[assistant(tool_call), tool(tool_result)]`!
                roles = [m.role for m in msgs]
                self.assertEqual(roles[:4], ["user", "assistant", "tool", "user"])
                self.assertIn("[LIVE_USER_STEER]: Also check edge case!", msgs[3].plain_text())
                return CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Handled both.")])

            transport = ScriptedTransport(_script)
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), transport),
                auth_profile=_dummy_auth(),
                ush_home=home,
            )
            runner_ref["r"] = runner
            # Inject steer mid-tool execution via BeforeTool hook simulation
            hs = HookSystem(ws, is_workspace_trusted=True)
            orig_bt = hs.before_tool

            def _inject_steer(tname: str, args: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
                runner_ref["r"].enqueue_steer("Also check edge case!")
                return orig_bt(tname, args)

            hs.before_tool = _inject_steer  # type: ignore[method-assign]
            runner.hook_runner = hs
            try:
                out = runner.run_turn("Read a.txt")
                self.assertEqual(out.final_message.plain_text(), "Handled both.")
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-14: Skills Trust Gate, Colon YAML, `paths:` `<skills_delta>` & Sandbox Grant
    # ------------------------------------------------------------------
    def test_ct14_skills_trust_colon_paths_delta_and_sandbox_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            (ws / "src" / "auth").mkdir(parents=True)
            (ws / "src" / "auth" / "jwt.py").write_text("tok = 1\n", encoding="utf-8")

            # External user skill (outside workspace!) with bundled reference file + colon in description + paths: glob
            sk_dir = home / "skills" / "jwt-hardener"
            (sk_dir / "references").mkdir(parents=True)
            (sk_dir / "references" / "checklist.md").write_text("Reject alg:none!", encoding="utf-8")
            (sk_dir / "SKILL.md").write_text(
                '---\nname: jwt-hardener\ndescription: Security rule: validate JWT exp and alg: never trust none\npaths: ["**/auth/**"]\n---\nRead references/checklist.md first.\n',
                encoding="utf-8",
            )

            # Untrusted workspace skill MUST be ignored until trusted!
            ws_sk = ws / ".ush" / "skills" / "untrusted-evil"
            ws_sk.mkdir(parents=True)
            (ws_sk / "SKILL.md").write_text("---\nname: untrusted-evil\ndescription: evil\n---\nbad\n", encoding="utf-8")

            sm = SkillManager(ws, is_workspace_trusted=False, ush_home=home, include_user_ecosystem_dirs=False)
            self.assertIsNone(sm.get("untrusted-evil"))
            self.assertIsNotNone(sm.get("jwt-hardener"))
            self.assertIn("alg: never", sm.get("jwt-hardener").description)  # type: ignore[union-attr]

            # `jwt-hardener` has `paths: ["**/auth/**"]`, so it stays OUT of Turn 0 catalog!
            turn0 = sm.initial_catalog_block(budget_tokens=2000)
            self.assertNotIn("jwt-hardener", turn0)

            # Accessing `src/auth/jwt.py` emits a tail `<skills_delta>` without mutating Turn 0!
            delta = sm.on_file_accessed(ws / "src" / "auth" / "jwt.py")
            self.assertIn("<skills_delta>", delta)
            self.assertIn("jwt-hardener", delta)

            # Out-of-repo `references/checklist.md` is blocked BEFORE activation, readable AFTER `activate_skill`!
            fw = SecurityFirewall(ws)
            ref_file = sk_dir / "references" / "checklist.md"
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate(ref_file)
            activated_xml = sm.activate("jwt-hardener", fw)
            self.assertIn("references/checklist.md", activated_xml)
            self.assertEqual(fw.resolve_and_validate(ref_file).read_text(encoding="utf-8"), "Reject alg:none!")

    # ------------------------------------------------------------------
    # CT-15: 4-Loader Slash Commands, `!{}` Injection Shield & `/btw` Side Fork
    # ------------------------------------------------------------------
    def test_ct15_slash_no_shadow_macro_escape_and_btw_fork(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            cmd_dir = ws / ".ush" / "commands"
            cmd_dir.mkdir(parents=True)
            # Attempt to hijack built-in `/help` + custom `!{echo {{args}}}` command
            (cmd_dir / "help.toml").write_text('prompt = "hijacked"\n', encoding="utf-8")
            (cmd_dir / "echoer.toml").write_text('prompt = "Out: !{echo {{args}}}"\n', encoding="utf-8")

            sm = SkillManager(ws, is_workspace_trusted=True, ush_home=home, include_user_ecosystem_dirs=False)
            reg = SlashCommandRegistry(ws, skill_manager=sm, ush_home=home, is_workspace_trusted=True)

            # Built-in `/help` stays builtin; repo collision moved to `/workspace.help`!
            self.assertEqual(reg.get("help").source, "builtin")  # type: ignore[union-attr]
            self.assertIsNotNone(reg.get("workspace.help"))

            # Shell injection `$(touch pwned)` inside `{{args}}` is quoted literally & checked by PolicyEngine!
            pol = PolicyEngine()
            pol.load_toml_text('[[rule]]\ntoolName="exec_pty"\ncommandPrefix=["echo"]\ndecision="allow"\n', tier="user")
            fw = SecurityFirewall(ws)
            pwn_target = ws / "pwned_marker"
            expanded, _ = expand_command_template(
                reg.get("echoer").template,  # type: ignore[union-attr]
                f"hello $(touch {pwn_target})",
                firewall=fw,
                policy_engine=pol,
            )
            self.assertFalse(pwn_target.exists())
            self.assertIn("$(touch", expanded)

            # `/btw` executes with `tools=[]` and causes ZERO mutations to SessionDAG!
            transport = ScriptedTransport(
                lambda _i, _p, _m, tools: (
                    self.assertEqual(tools, [])
                    or CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Side answer")])
                )
            )
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), transport),
                auth_profile=_dummy_auth(),
                ush_home=home,
            )
            try:
                runner.dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Main task")]))
                ver_before = runner.dag.history_version
                reply = runner.ask_btw("Quick question?")
                self.assertEqual(reply.plain_text(), "Side answer")
                self.assertEqual(runner.dag.history_version, ver_before)
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-16: 9-Event Hooks (`BeforeToolSelection`, `Stop` Veto) & Clamped Policies
    # ------------------------------------------------------------------
    def test_ct16_hooks_and_privilege_clamped_extension_policies(self) -> None:
        pol = PolicyEngine()
        # Extension tries to sneak in `allow` and `yolo` -> rejected; its `deny` IS honored!
        loaded = pol.load_toml_text(
            """
            [[rule]]
            toolName = "exec_pty"
            decision = "allow"
            [[rule]]
            toolName = "exec_pty"
            commandPrefix = ["rm"]
            decision = "deny"
            reason = "ext blocks rm"
            """,
            tier="extension",
        )
        self.assertEqual(loaded, 1)
        self.assertEqual(len(pol.clamped_warnings), 1)
        # Wrapper + chain peeling (`sudo env FOO=1 echo hi && rm -rf build`) catches `rm`!
        dec, rsn = pol.check("exec_pty", command="echo ok && sudo -u root env X=1 rm -rf build", mode=PermissionMode.YOLO)
        self.assertEqual(dec, PolicyDecision.DENY)
        self.assertIn("ext blocks rm", rsn)

        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            ws.mkdir()
            hooks_file = ws / "hooks.json"
            flag_file = ws / "vetoed_once"
            hooks_file.write_text(
                json.dumps(
                    {
                        "hooks": {
                            "BeforeToolSelection": [
                                {"command": 'echo \'{"allowedTools":["read_file","grep_files"]}\''}
                            ],
                            "Stop": [
                                {
                                    "command": f'if [ ! -f "{flag_file}" ]; then touch "{flag_file}"; echo "Run tests before stopping!" >&2; exit 2; fi'
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            hs = HookSystem(ws, is_workspace_trusted=True)
            hs.load_hooks_json(hooks_file, source="workspace")

            seen_tool_counts: list[int] = []
            replies = [
                CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Premature done")]),
                CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="Verified done!")]),
            ]

            def _step(i: int, _p: Any, _m: Any, tools: list[dict[str, Any]]) -> CanonicalMessage:
                seen_tool_counts.append(len(tools))
                return replies[i]

            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), ScriptedTransport(_step)),
                auth_profile=_dummy_auth(),
                ush_home=home,
                hook_runner=hs,
            )
            try:
                out = runner.run_turn("Implement feature")
                self.assertEqual(seen_tool_counts[0], 2)  # Slimmed by BeforeToolSelection!
                self.assertEqual(out.steps_executed, 2)   # Re-opened once by `Stop` exit 2 veto!
                self.assertEqual(out.final_message.plain_text(), "Verified done!")
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-17: 4-Scope + JIT Subdirectory Memory (Inode Dedup) & `.inbox` Redaction
    # ------------------------------------------------------------------
    def test_ct17_jit_memory_dedup_and_inbox_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            pkg = ws / "packages" / "billing"
            pkg.mkdir(parents=True)
            (ws / "AGENTS.md").write_text("Root rule: strict typing.", encoding="utf-8")
            os.symlink(ws / "AGENTS.md", ws / "GEMINI.md")  # Symlink duplicate!
            (pkg / "AGENTS.md").write_text("Billing rule: integer cents only.", encoding="utf-8")
            (pkg / "charge.py").write_text("cents = 100\n", encoding="utf-8")

            mm = MemoryManager(ws, ush_home=home)
            init_mem = mm.initial_memory_block()
            self.assertEqual(init_mem.count("Root rule: strict typing."), 1)  # Deduped by inode!
            self.assertNotIn("integer cents", init_mem)  # Subdir NOT loaded eagerly!

            jit1 = mm.on_file_accessed(pkg / "charge.py")
            self.assertIn("integer cents only", jit1)
            jit2 = mm.on_file_accessed(pkg / "charge.py")
            self.assertEqual(jit2, "")  # Loaded strictly ONCE!

            # `/learn` redacts leaked bearer/API keys and stages into `.inbox` without touching MEMORY.md
            fake_secret = "sk-ant-" + ("A1b2C3d4" * 4)
            prop = mm.stage_learning_patch("Auth Note", [f"Never use key {fake_secret} in tests"])
            self.assertFalse(mm.memory_index_path.exists())
            self.assertNotIn(fake_secret, prop.diff_text)
            self.assertIn("[REDACTED_SECRET]", prop.diff_text)
            self.assertTrue(mm.apply_inbox_patch(prop.patch_id))
            self.assertTrue(mm.memory_index_path.is_file())

    # ------------------------------------------------------------------
    # CT-18: `clean_patch` Anti-Goodhart Sieve & Position-Swapped Bradley-Terry
    # ------------------------------------------------------------------
    def test_ct18_clean_patch_and_position_swapped_judge(self) -> None:
        cheating_diff = (
            "diff --git a/tests/test_math.py b/tests/test_math.py\n"
            "--- a/tests/test_math.py\n+++ b/tests/test_math.py\n"
            "@@ -1,2 +1,1 @@\n def test_add():\n-    assert add(2, 2) == 4\n+    pass\n"
        )
        res = CascadingJudge.run_stage_a(CandidatePatch("c_cheat", ("w1",), "gemini", cheating_diff))
        self.assertFalse(res.passed)
        self.assertTrue(any("ANTI_GOODHART_VETO" in r for r in res.reasons))

        # Slot-biased judge that ALWAYS says "A" (prefers first slot) is neutralized to exact 0.5/0.5 tie!
        c1 = CandidatePatch("m1", ("w1",), "gemini", "diff --git a/x.py b/x.py\n+x=1\n")
        c2 = CandidatePatch("m2", ("w2",), "anthropic", "diff --git a/x.py b/x.py\n+x=2\n")
        ranked_bias = CascadingJudge.rank_bradley_terry([c1, c2], lambda _a, _b: "A")
        self.assertAlmostEqual(ranked_bias[0][1], 0.5, places=3)
        self.assertAlmostEqual(ranked_bias[1][1], 0.5, places=3)

        # Content-consistent judge preferring `+x=2` regardless of slot ranks `m2` cleanly #1!
        ranked_real = CascadingJudge.rank_bradley_terry(
            [c1, c2], lambda a, _b: "A" if "+x=2" in a.unified_diff else "B"
        )
        self.assertEqual(ranked_real[0][0].candidate_id, "m2")
        self.assertGreater(ranked_real[0][1], 0.85)

    # ------------------------------------------------------------------
    # CT-19 & CT-20: Gear 2->3 Worktree Upshift + `CONTRACT_DELTA` + `INV-SWARM-1`
    # ------------------------------------------------------------------
    def test_ct19_and_ct20_swarm_upshift_contract_delta_and_quorum_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = pathlib.Path(tmp) / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "api.py").write_text("def calc():\n    return 0\n", encoding="utf-8")
            (repo / "cli.py").write_text("from api import calc\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "base"],
                cwd=repo,
                check=True,
            )

            coord = ElasticSwarmCoordinator(repo)
            ready = threading.Event()

            def _shard_api(wt: Any, bb: StigmergicBlackboard, wid: str) -> None:
                (wt.path / "api.py").write_text("def calc(scale: int = 1) -> int:\n    return 42 * scale\n", encoding="utf-8")
                bb.post("CONTRACT_DELTA", wid, "calc(scale: int = 1) -> int", scope="api.py")
                ready.set()

            def _shard_cli(wt: Any, bb: StigmergicBlackboard, _wid: str) -> None:
                ready.wait(timeout=5)
                deltas = bb.read_since(0, event_types=("CONTRACT_DELTA",))
                self.assertEqual(len(deltas), 1)
                (wt.path / "cli.py").write_text(
                    "from api import calc\n\ndef run() -> int:\n    return calc(scale=2)\n",
                    encoding="utf-8",
                )

            summary = coord.execute_with_stall_upshift(
                "Implement scaled calculator across api.py and cli.py",
                solo_attempt=lambda: False,  # Simulate Gear-2 stall -> auto-upshift to Gear 3!
                parallel_shards=[("worker_api", _shard_api), ("worker_cli", _shard_cli)],
                judge_id="judge:claude-sentinel",
                judge_family="anthropic",
            )
            self.assertTrue(summary.upshifted)
            self.assertEqual(summary.merged_workers, ["worker_api", "worker_cli"])
            self.assertEqual(len(summary.contract_deltas), 1)
            self.assertIn("calc(scale=2)", (repo / "cli.py").read_text(encoding="utf-8"))
            self.assertIn("42 * scale", (repo / "api.py").read_text(encoding="utf-8"))

            # CT-20: Self-certification (`judge_id in contributor_ids`) & SQLite hash tampering MUST raise!
            with self.assertRaises(SwarmInvariantError):
                coord.blackboard.record_quorum_verdict(
                    candidate_id="self_cert",
                    contributor_ids=("worker_api",),
                    judge_id="worker_api",
                    judge_family="gemini",
                    bars_score=1.0,
                    approved=True,
                )

            # Tamper directly with bars_score in SQLite -> SHA-256 chain check rejects deterministically!
            import sqlite3

            conn = sqlite3.connect(coord.blackboard.db_path)
            try:
                with conn:
                    conn.execute("UPDATE consensus_ledger SET bars_score = 0.123 WHERE seq = 1")
            finally:
                conn.close()
            with self.assertRaises(SwarmInvariantError):
                coord.blackboard.verify_ledger_integrity()

    # ------------------------------------------------------------------
    # CT-21: Cross-Provider Native Thinking Wire + Uncompacted Long-Form Trace Ledger
    # ------------------------------------------------------------------
    def test_ct21_long_form_thought_tracing_wire_and_ledger(self) -> None:
        from io import StringIO

        from ush.brain.providers import InlineThinkStreamSplitter, build_gemini_thinking_config
        from ush.brain.trace import StreamTracePrinter, list_workspace_sessions
        from ush.core.types import ThinkingEffort, TraceMode

        # 1. Provider-native extended thinking wire parameters
        g3 = resolve_model_profile("gemini-3.1-pro-preview", "vertex")
        g25 = resolve_model_profile("gemini-2.5-pro", "vertex")
        cl = resolve_model_profile("claude-opus-4-20250514", "anthropic")
        oai = resolve_model_profile("gpt-5", "openai")
        orr = resolve_model_profile("openrouter:deepseek/deepseek-r1", "openrouter")

        p_g3 = build_gemini_payload("sys", [], [], g3, thinking_effort=ThinkingEffort.HIGH)
        self.assertEqual(
            p_g3["generationConfig"]["thinkingConfig"],
            {"includeThoughts": True, "thinkingLevel": "HIGH"},
        )
        self.assertEqual(
            build_gemini_thinking_config(g25, ThinkingEffort.MAX),
            {"includeThoughts": True, "thinkingBudget": -1},
        )
        p_cl = build_anthropic_payload("sys", [], [], cl, thinking_effort=ThinkingEffort.HIGH)
        self.assertEqual(p_cl["thinking"]["type"], "enabled")
        self.assertGreaterEqual(p_cl["thinking"]["budget_tokens"], 16_000)
        self.assertGreater(p_cl["max_tokens"], p_cl["thinking"]["budget_tokens"])

        p_oai = build_openai_chat_payload("sys", [], [], oai, thinking_effort=ThinkingEffort.HIGH)
        self.assertEqual(p_oai["reasoning_effort"], "high")
        p_orr = build_openai_chat_payload("sys", [], [], orr, thinking_effort=ThinkingEffort.HIGH)
        self.assertTrue(p_orr["include_reasoning"])

        # 2. Byte-sliced `<think>...</think>` live splitter (3-char frames across tag boundaries)
        raw_stream = "<think>Hypothesis A: lock order inversion.\nWait — check RwLock!</think>\nFixed lock order."
        splitter = InlineThinkStreamSplitter()
        emitted: list[tuple[str, str]] = []
        for i in range(0, len(raw_stream), 3):
            emitted.extend(splitter.feed(raw_stream[i : i + 3]))
        emitted.extend(splitter.flush())
        thought_joined = "".join(s for k, s in emitted if k == "thinking_delta")
        text_joined = "".join(s for k, s in emitted if k == "text_delta")
        self.assertEqual(thought_joined, "Hypothesis A: lock order inversion.\nWait — check RwLock!")
        self.assertEqual(text_joined, "Fixed lock order.")

        # 3. Multi-step long-form thought trace surviving compaction + search + Markdown export
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            ws.mkdir()
            (ws / "mod.py").write_text("x = 1\n", encoding="utf-8")

            def _multi(step: int, _p: Any, _msgs: list[CanonicalMessage], _t: Any) -> CanonicalMessage:
                if step == 0:
                    return CanonicalMessage(
                        role="assistant",
                        parts=[
                            CanonicalPart(
                                kind="thinking",
                                text="**Inspecting module**\nHypothesis alpha_invariant_77: read mod.py before concluding.",
                                signature="sig-step-1",
                            ),
                            CanonicalPart(
                                kind="tool_call",
                                tool_call_id="c1",
                                tool_name="read_file",
                                tool_args={"path": "mod.py"},
                                signature="sig-step-1",
                            ),
                        ],
                        usage=TokenUsage(input_tokens=400, output_tokens=30, reasoning_tokens=64),
                    )
                return CanonicalMessage(
                    role="assistant",
                    parts=[
                        CanonicalPart(
                            kind="thinking",
                            text="**Verifying mod.py**\nObserved x = 1; Wait — hypothesis confirmed cleanly.",
                            signature="sig-step-2",
                        ),
                        CanonicalPart(kind="text", text="Confirmed x = 1."),
                    ],
                    usage=TokenUsage(input_tokens=520, output_tokens=12, cached_input_tokens=380, reasoning_tokens=48),
                )

            journal = home / "sessions" / "sess_longtrace01.jsonl"
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), ScriptedTransport(_multi)),
                auth_profile=_dummy_auth(),
                session_id="sess_longtrace01",
                journal_path=journal,
                ush_home=home,
            )
            buf = StringIO()
            printer = StreamTracePrinter(mode=TraceMode.FULL, use_color=False, out=buf)
            try:
                runner.run_turn("Trace mod.py deeply", on_event=printer)
                # Force aggressive DAG compaction — raw thoughts MUST still remain intact in `.trace.jsonl`!
                runner.dag.maybe_compact(context_limit_tokens=1, ratio=0.01)
                assert runner.trace_ledger is not None
                hits = runner.trace_ledger.search("alpha_invariant_77")
                self.assertEqual(len(hits), 1)
                self.assertEqual(hits[0]["step"], 1)
                md_out = runner.trace_ledger.export_markdown()
                md_txt = md_out.read_text(encoding="utf-8")
                self.assertIn("alpha_invariant_77", md_txt)
                self.assertIn("Step 2 — Chain of Thought", md_txt)
                rendered = buf.getvalue()
                self.assertIn("⋮ **Inspecting module**", rendered)
                self.assertIn("▸ read_file", rendered)
                self.assertIn("◂ ok read_file", rendered)
                s_list = list_workspace_sessions(home / "sessions", workspace_filter=ws)
                self.assertEqual(len(s_list), 1)
                self.assertEqual(s_list[0].thought_blocks, 2)
                self.assertEqual(s_list[0].reasoning_tokens, 112)
            finally:
                runner.close()

    def test_ct22_gemini_parallel_signature_and_trailing_turn_safety(self) -> None:
        g4 = resolve_model_profile("gemini-4-argon", "vertex")
        hist = [
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Audit main.py")]),
            CanonicalMessage(
                role="assistant",
                producer_provider="vertex",
                producer_model_id="gemini-4-argon",
                parts=[
                    CanonicalPart(kind="thinking", text="Reading two slices.", signature="sig_dup_296"),
                    CanonicalPart(kind="tool_call", tool_call_id="c1", tool_name="read_file", tool_args={"path": "a.py"}, signature="sig_dup_296"),
                    CanonicalPart(kind="tool_call", tool_call_id="c2", tool_name="read_file", tool_args={"path": "b.py"}, signature="sig_dup_296"),
                ],
            ),
            CanonicalMessage(
                role="tool",
                parts=[
                    CanonicalPart(kind="tool_result", tool_call_id="c1", tool_name="read_file", text="a=1"),
                    CanonicalPart(kind="tool_result", tool_call_id="c2", tool_name="read_file", text="b=2"),
                ],
            ),
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Did you complete?")]),
        ]
        wire = transform_messages(hist, g4)
        pay = build_gemini_payload("sys", wire, [], g4)
        # 1. Parallel functionCall signature law: ONLY first functionCall gets signature; `thought` and `c2` do NOT duplicate it!
        model_turn = pay["contents"][1]
        self.assertEqual(model_turn["role"], "model")
        self.assertNotIn("thoughtSignature", model_turn["parts"][0])  # thought: True
        self.assertEqual(model_turn["parts"][1].get("thoughtSignature"), "sig_dup_296")  # c1
        self.assertNotIn("thoughtSignature", model_turn["parts"][2])  # c2 MUST NOT duplicate!
        # 2. functionResponse + trailing user text stay in SEPARATE `Content` frames bridged by a `model` turn
        #    so BOTH pure `functionResponse` isolation AND strict `user <-> model` alternation hold!
        self.assertEqual(len(pay["contents"]), 5)
        self.assertTrue(all("functionResponse" in p for p in pay["contents"][2]["parts"]))
        self.assertEqual(pay["contents"][3]["role"], "model")
        self.assertEqual(pay["contents"][4], {"role": "user", "parts": [{"text": "Did you complete?"}]})
        # 3. Trailing assistant turn guard: never emit a request ending on `model`/`assistant`
        trailing_hist = [
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Hi")]),
            CanonicalMessage(role="assistant", producer_provider="vertex", producer_model_id="gemini-4-argon", parts=[CanonicalPart(kind="text", text="Partial")]),
        ]
        w_tail = transform_messages(trailing_hist, g4)
        self.assertEqual(w_tail[-1].role, "user")
        p_tail = build_gemini_payload("sys", w_tail, [], g4)
        self.assertEqual(p_tail["contents"][-1]["role"], "user")

    def test_ct23_visual_tui_menus_thinking_threads_and_prompt_queue(self) -> None:
        from io import StringIO
        from ush.core.types import ThinkingEffort
        from ush.cli.main import HarnessRuntime, handle_slash_command
        from ush.cli.tui import (
            VisualTuiController,
            clip_to_cols,
            decode_key_buffer,
            strip_ansi,
            vis_width,
        )

        # 1. `INV-TUI-1`: ANSI/Emoji width clamping never overflows column budget
        wide = "\x1b[1;36m🧠 Step 3/160 · ⚡ turbo · " + ("long_token_" * 12) + "\x1b[0m"
        clipped = clip_to_cols(wide, 59)
        self.assertLessEqual(vis_width(clipped), 59)
        self.assertTrue(clipped.endswith("\x1b[0m"))

        # 2. `INV-TUI-4`: Escape sequences + Shift+Tab + Alt+S + Ctrl+O + Bracketed Paste decoding
        evs, rem = decode_key_buffer(
            "/mo\x1b[B\t\x1b[Z\x0f\x1bs\x1b[200~line1\r\nline2\x1b[201~\r",
            esc_is_complete=True,
        )
        self.assertEqual(rem, "")
        self.assertEqual(
            [e.kind for e in evs],
            ["CHAR", "CHAR", "CHAR", "DOWN", "TAB", "SHIFT_TAB", "CTRL_O", "ALT_S", "PASTE", "ENTER"],
        )
        paste_ev = [e for e in evs if e.kind == "PASTE"][0]
        self.assertEqual(paste_ev.text, "line1\nline2")

        # 3. End-to-end navigable menus + live thinking threads + concurrent next-prompt queue
        with tempfile.TemporaryDirectory() as raw_dir:
            ws = pathlib.Path(raw_dir) / "ws"
            home = pathlib.Path(raw_dir) / "home"
            ws.mkdir()
            (ws / "alpha.py").write_text("VALUE = 42\n", encoding="utf-8")
            old_home = os.environ.get("USH_HOME")
            os.environ["USH_HOME"] = str(home)
            gate_turn1 = threading.Event()
            seen_prompts: list[str] = []

            class _SlowStreamingTransport:
                thinking_effort = ThinkingEffort.HIGH

                def stream_turn(self, profile, auth_profile, system_prompt, messages, tools, session_id="", on_event=None):
                    user_txts = [m.plain_text() for m in messages if m.role == "user"]
                    cur_q = user_txts[-1] if user_txts else ""
                    seen_prompts.append(cur_q)
                    if len(seen_prompts) == 1:
                        if on_event:
                            on_event(StreamEvent(kind="thinking_delta", text="**Analyzing alpha.py**\nChecking invariants."))
                            on_event(
                                StreamEvent(
                                    kind="thinking_end",
                                    elapsed_ms=120.0,
                                    meta={"thought_chars": 43, "reasoning_tokens": 22, "has_signature": True},
                                )
                            )
                        gate_turn1.wait(timeout=3.0)
                        if on_event:
                            on_event(StreamEvent(kind="text_delta", text="Turn 1 done."))
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=auth_profile.provider,
                            producer_model_id=profile.model_id,
                            parts=[
                                CanonicalPart(kind="thinking", text="**Analyzing alpha.py**\nChecking invariants.", signature="s1"),
                                CanonicalPart(kind="text", text="Turn 1 done."),
                            ],
                            usage=TokenUsage(input_tokens=200, output_tokens=8, reasoning_tokens=22),
                        )
                    if on_event:
                        on_event(StreamEvent(kind="thinking_delta", text="**Follow-up Thread**\nVerified queued prompt."))
                        on_event(
                            StreamEvent(
                                kind="thinking_end",
                                elapsed_ms=80.0,
                                meta={"thought_chars": 44, "reasoning_tokens": 18, "has_signature": True},
                            )
                        )
                        on_event(StreamEvent(kind="text_delta", text="Turn 2 queued reply."))
                    return CanonicalMessage(
                        role="assistant",
                        producer_provider=auth_profile.provider,
                        producer_model_id=profile.model_id,
                        parts=[
                            CanonicalPart(kind="thinking", text="**Follow-up Thread**\nVerified queued prompt.", signature="s2"),
                            CanonicalPart(kind="text", text="Turn 2 queued reply."),
                        ],
                        usage=TokenUsage(input_tokens=300, output_tokens=8, reasoning_tokens=18),
                    )

            rt = HarnessRuntime(ws, mode=PermissionMode.AUTO_EDIT)
            rt.runner.client.transport = _SlowStreamingTransport()
            screen = StringIO()
            ctrl = VisualTuiController(
                rt,
                slash_handler=lambda r, line: handle_slash_command(r, line, headless=False),
                out=screen,
                use_color=True,
                cols_fn=lambda: 88,
            )
            try:
                # A. Live `/` auto-menu + `/model` interactive arrow-key picker
                for ke in decode_key_buffer("/mod")[0]:
                    ctrl.handle_key(ke)
                self.assertIsNotNone(ctrl.menu)
                self.assertEqual(ctrl.menu.kind, "auto_slash")
                self.assertEqual(ctrl.menu.current().key, "model")
                # Press Enter -> opens interactive Model Picker menu!
                ctrl.handle_key(decode_key_buffer("\r")[0][0])
                self.assertIsNotNone(ctrl.menu)
                self.assertEqual(ctrl.menu.kind, "picker")
                # Strictly verify zero provider labels (`antigravity`, `vertex`, `openai`, etc.) on `/model` items!
                for m_item in ctrl.menu.items:
                    if not m_item.active:
                        self.assertEqual(m_item.badge, "")
                    self.assertNotIn(":", m_item.label)
                # Filter to `gemini-3.8-flash` & select with Enter
                for ke in decode_key_buffer("3.8-flash\r")[0]:
                    ctrl.handle_key(ke)
                self.assertIsNone(ctrl.menu)
                self.assertEqual(rt.runner.model_profile.model_id, "gemini-3.8-flash")

                # B. `@` file auto-popup menu completes `@alpha.py `
                for ke in decode_key_buffer("Inspect @alp\t")[0]:
                    ctrl.handle_key(ke)
                self.assertIn("@alpha.py ", ctrl.input_buf)

                # Verify every dock line honors `INV-TUI-1` (`<= cols - 1 == 87` visible cols!)
                dock_lines, _, _ = ctrl._build_dock(88)
                for dl in dock_lines:
                    self.assertLessEqual(vis_width(dl), 87, strip_ansi(dl))

                # C. Submit Turn 1; WHILE Turn 1 is actively running, queue Turn 2 AND open `Ctrl+O` Thinking Threads!
                ctrl.handle_key(decode_key_buffer("\r")[0][0])
                deadline = time.monotonic() + 2.0
                while len(ctrl.tracker.threads) == 0 and time.monotonic() < deadline:
                    time.sleep(0.02)

                # Queue next prompt while model is still in Turn 1!
                for ke in decode_key_buffer("Now verify follow-up task\r")[0]:
                    ctrl.handle_key(ke)
                self.assertEqual(len(ctrl.next_queue), 1)

                # Open `Ctrl+O` Thinking Threads explorer live mid-turn and check narrow 64-col clamp!
                ctrl.handle_key(decode_key_buffer("\x0f")[0][0])
                self.assertIsNotNone(ctrl.menu)
                self.assertEqual(ctrl.menu.kind, "threads")
                narrow_lines, _, _ = ctrl._build_dock(64)
                for nl in narrow_lines:
                    self.assertLessEqual(vis_width(clip_to_cols(nl, 63)), 63)
                ctrl.handle_key(decode_key_buffer("\x1b")[0][0])  # Esc closes threads drawer

                # Release Turn 1 -> queued Turn 2 MUST drain & run automatically!
                gate_turn1.set()
                deadline = time.monotonic() + 3.0
                while ctrl.turn_busy.is_set() and time.monotonic() < deadline:
                    time.sleep(0.02)

                self.assertEqual(len(seen_prompts), 2)
                self.assertIn("VALUE = 42", seen_prompts[0])  # `@alpha.py` expanded!
                self.assertIn("Now verify follow-up task", seen_prompts[1])
                self.assertEqual(len(ctrl.tracker.threads), 2)
                rendered_plain = strip_ansi(screen.getvalue())
                self.assertIn("Analyzing alpha.py", rendered_plain)
                self.assertIn("Turn 2 queued reply.", rendered_plain)
            finally:
                gate_turn1.set()
                rt.runner.close()
                if old_home is None:
                    os.environ.pop("USH_HOME", None)
                else:
                    os.environ["USH_HOME"] = old_home

    def test_ct24_steer_deadlock_free_modern_glyphs_and_recent_resume(self) -> None:
        from io import StringIO
        from ush.brain.trace import list_workspace_sessions
        from ush.core.types import ThinkingEffort
        from ush.cli.main import HarnessRuntime, handle_slash_command
        from ush.cli.tui import VisualTuiController, decode_key_buffer, strip_ansi

        old_emojis = ("🧠", "⚡", "🔥", "🚀", "🛡", "📐", "⏭")

        with tempfile.TemporaryDirectory() as raw_dir:
            base = pathlib.Path(raw_dir)
            home = base / "home"
            ws_a = base / "proj_alpha"
            ws_b = base / "proj_beta"
            ws_c = base / "proj_gamma"
            for d in (ws_a, ws_b, ws_c, home / "sessions"):
                d.mkdir(parents=True, exist_ok=True)

            old_home = os.environ.get("USH_HOME")
            os.environ["USH_HOME"] = str(home)
            try:
                # 1. Seed 3 sessions across DIFFERENT workspaces with distinct mtimes + 1 empty stub
                s_old_dag = home / "sessions" / "sess_older1111111.jsonl"
                s_old_tr = home / "sessions" / "sess_older1111111.trace.jsonl"
                s_old_dag.write_text(
                    json.dumps({"entry_id": "n1", "role": "user", "parts": [{"kind": "text", "text": "Old alpha work"}]}) + "\n",
                    encoding="utf-8",
                )
                s_old_tr.write_text(
                    json.dumps({"ts": 1_700_000_100.0, "type": "turn_start", "workspace": str(ws_a), "prompt": "Old alpha work", "model": "gemini-2.5-pro"}) + "\n",
                    encoding="utf-8",
                )
                os.utime(s_old_dag, (1_700_000_100.0, 1_700_000_100.0))
                os.utime(s_old_tr, (1_700_000_100.0, 1_700_000_100.0))

                s_mid_dag = home / "sessions" / "sess_mid222222222.jsonl"
                s_mid_dag.write_text(
                    json.dumps({"entry_id": "m1", "role": "user", "parts": [{"kind": "text", "text": "<attached_file path=\"x.py\">code</attached_file>\nClean mid prompt"}]})
                    + "\n"
                    + json.dumps({"entry_id": "m2", "parent_entry_id": "m1", "role": "assistant", "producer_model_id": "claude-opus-4-20250514", "parts": [{"kind": "thinking", "text": "Deep plan"}, {"kind": "text", "text": "Done"}], "usage": {"reasoning_tokens": 95}})
                    + "\n",
                    encoding="utf-8",
                )
                os.utime(s_mid_dag, (1_700_000_500.0, 1_700_000_500.0))

                s_new_dag = home / "sessions" / "sess_newest333333.jsonl"
                s_new_tr = home / "sessions" / "sess_newest333333.trace.jsonl"
                s_new_dag.write_text(
                    json.dumps({"entry_id": "k1", "role": "user", "parts": [{"kind": "text", "text": "Newest beta task"}]}) + "\n",
                    encoding="utf-8",
                )
                s_new_tr.write_text(
                    json.dumps({"ts": 1_700_000_900.0, "type": "turn_start", "workspace": str(ws_b), "prompt": "Newest beta task", "model": "gemini-4-argon"}) + "\n"
                    + json.dumps({"ts": 1_700_000_905.0, "type": "thought_block", "turn": 1, "step": 1, "model": "gemini-4-argon", "thought": "**Plan**\nCheck.", "reasoning_tokens": 140}) + "\n",
                    encoding="utf-8",
                )
                os.utime(s_new_dag, (1_700_000_900.0, 1_700_000_900.0))
                os.utime(s_new_tr, (1_700_000_905.0, 1_700_000_905.0))

                # Empty stub must be ignored
                (home / "sessions" / "sess_empty0000000.jsonl").write_text("", encoding="utf-8")

                # Verify ALL 3 non-empty sessions across ALL workspaces come back strictly newest-first!
                recent = list_workspace_sessions(home / "sessions", workspace_filter=None, include_empty=False)
                self.assertEqual([s.session_id for s in recent], ["sess_newest333333", "sess_mid222222222", "sess_older1111111"])
                self.assertEqual(recent[1].last_prompt, "Clean mid prompt")
                self.assertEqual(recent[1].thought_blocks, 1)
                self.assertEqual(recent[1].reasoning_tokens, 95)

                # 2. Launch TUI in `ws_c` (a 3rd workspace!) -> `/resume` menu STILL pulls all 3 sorted newest-first!
                entered_step1 = threading.Event()
                release_step1 = threading.Event()
                saw_steers: list[str] = []

                class _SteerableTransport:
                    thinking_effort = ThinkingEffort.HIGH

                    def stream_turn(self, profile, auth_profile, system_prompt, messages, tools, session_id="", on_event=None):
                        txts = [m.plain_text() for m in messages if m.role == "user"]
                        joined = "\n".join(txts)
                        if "[LIVE_USER_STEER]:" in joined:
                            saw_steers.append(joined)
                            return CanonicalMessage(
                                role="assistant",
                                producer_provider=auth_profile.provider,
                                producer_model_id=profile.model_id,
                                parts=[CanonicalPart(kind="text", text="Applied steer cleanly.")],
                            )
                        entered_step1.set()
                        release_step1.wait(timeout=3.0)
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=auth_profile.provider,
                            producer_model_id=profile.model_id,
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="c_list",
                                    tool_name="glob_files",
                                    tool_args={"pattern": "*"},
                                    signature="sig_ok",
                                )
                            ],
                        )

                rt = HarnessRuntime(ws_c, mode=PermissionMode.YOLO)
                rt.runner.client.transport = _SteerableTransport()
                scr = StringIO()
                ctrl = VisualTuiController(
                    rt,
                    slash_handler=lambda r, ln: handle_slash_command(r, ln, headless=False),
                    out=scr,
                    use_color=True,
                    cols_fn=lambda: 96,
                )

                # Verify `/resume` picker shows all 3 cross-workspace sessions newest-first, and Enter resumes #1!
                for ke in decode_key_buffer("/resume\r")[0]:
                    ctrl.handle_key(ke)
                self.assertIsNotNone(ctrl.menu)
                self.assertEqual([it.key for it in ctrl.menu.items], ["sess_newest333333", "sess_mid222222222", "sess_older1111111"])
                ctrl.handle_key(decode_key_buffer("\r")[0][0])  # Resume top-most newest session!
                self.assertEqual(rt.runner.session_id, "sess_newest333333")
                self.assertEqual(rt.runner.model_profile.model_id, "gemini-4-argon")
                self.assertEqual(len(ctrl.tracker.threads), 1)
                rt.runner.client.transport = _SteerableTransport()

                # Verify modern geometric styling: zero cartoon emojis & zero `yolo` badge in status/menus!
                ctrl.open_turbo_permissions_menu()
                dock_plain = "\n".join(strip_ansi(x) for x in ctrl._build_dock(96)[0])
                for em in old_emojis:
                    self.assertNotIn(em, dock_plain)
                self.assertNotIn("yolo", dock_plain.lower())
                self.assertIn("▸▸ turbo", dock_plain)
                self.assertIn("◈ high", dock_plain)
                ctrl.handle_key(decode_key_buffer("\x1b")[0][0])

                # 3. Regression test for `/steer` deadlock: queue a prompt mid-turn, then invoke bare `/steer`
                #    via slash menu Enter (exact sequence that previously self-deadlocked `_q_lock`!) + `/steer <msg>`!
                for ke in decode_key_buffer("Start turn\r")[0]:
                    ctrl.handle_key(ke)
                self.assertTrue(entered_step1.wait(timeout=2.0))

                # Queue a follow-up prompt, then promote it via `/st` + Enter in `< 0.2s` (zero deadlock!)
                t0 = time.monotonic()
                for ke in decode_key_buffer("Focus on release binary\r/steer\r")[0]:
                    ctrl.handle_key(ke)
                # Also send explicit `/steer <arg>` AND empty `/st\r` (which now fills `/steer ` cleanly!)
                for ke in decode_key_buffer("/steer Keep zero deps\r/steer\r")[0]:
                    ctrl.handle_key(ke)
                self.assertEqual(ctrl.input_buf, "/steer ")
                ctrl.handle_key(decode_key_buffer("\x15")[0][0])  # Ctrl+U clears `/steer `
                self.assertLess(time.monotonic() - t0, 0.5, "Main UI thread blocked or deadlocked on /steer!")
                self.assertEqual(rt.runner.pending_steer_count(), 2)

                release_step1.set()
                deadline = time.monotonic() + 3.0
                while ctrl.turn_busy.is_set() and time.monotonic() < deadline:
                    time.sleep(0.02)

                self.assertFalse(ctrl.turn_busy.is_set())
                self.assertEqual(len(saw_steers), 1)
                self.assertIn("Focus on release binary", saw_steers[0])
                self.assertIn("Keep zero deps", saw_steers[0])
                full_out = strip_ansi(scr.getvalue())
                for em in old_emojis:
                    self.assertNotIn(em, full_out)
            finally:
                release_step1.set()
                rt.runner.close()
                if old_home is None:
                    os.environ.pop("USH_HOME", None)
                else:
                    os.environ["USH_HOME"] = old_home

    # ------------------------------------------------------------------
    # CT-25: Exact 1-to-1 `functionCall` / `functionResponse` Bijection
    #        Across Resumed Interrupted Sessions, Multi-Step Loops & Partial Batches
    # ------------------------------------------------------------------
    def test_ct25_tool_call_response_bijection_across_mid_history_orphans(self) -> None:
        gem = resolve_model_profile("gemini-4-argon", "google")
        anth = resolve_model_profile("claude-opus-4-20250514", "anthropic")
        oai = resolve_model_profile("gpt-5", "openai")

        def _assert_gemini_bijection(contents: list[dict[str, Any]]) -> None:
            self.assertTrue(contents)
            self.assertEqual(contents[0]["role"], "user")
            self.assertEqual(contents[-1]["role"], "user")
            for idx, c in enumerate(contents):
                if idx + 1 < len(contents):
                    self.assertNotEqual(
                        c["role"],
                        contents[idx + 1]["role"],
                        f"Role alternation violated at {idx}->{idx + 1}: {c['role']} followed by {contents[idx + 1]['role']}",
                    )
                fcs = [pt["functionCall"]["name"] for pt in c["parts"] if "functionCall" in pt]
                frs = [pt["functionResponse"]["name"] for pt in c["parts"] if "functionResponse" in pt]
                if fcs:
                    self.assertEqual(c["role"], "model")
                    self.assertLess(idx + 1, len(contents))
                    nxt = contents[idx + 1]
                    self.assertEqual(nxt["role"], "user")
                    nxt_frs = [pt["functionResponse"]["name"] for pt in nxt["parts"] if "functionResponse" in pt]
                    self.assertEqual(len(nxt_frs), len(nxt["parts"]), "Mixed functionResponse + text in one turn!")
                    self.assertEqual(nxt_frs, fcs, f"HTTP 400 mismatch at {idx}: FC={fcs} vs FR={nxt_frs}")
                if frs:
                    self.assertEqual(c["role"], "user")
                    self.assertGreater(idx, 0)
                    prev_fcs = [
                        pt["functionCall"]["name"] for pt in contents[idx - 1]["parts"] if "functionCall" in pt
                    ]
                    self.assertEqual(frs, prev_fcs)

        # 1. Exact reproduction of resumed crashed session (`[orphan grep_files] -> [user push] -> Step 1 -> Step 2 -> Step 3`)
        base_hist: list[CanonicalMessage] = [
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Package Crucible v0.1")]),
            CanonicalMessage(
                role="assistant",
                producer_provider="google",
                producer_model_id="gemini-4-argon",
                parts=[
                    CanonicalPart(kind="thinking", text="Grep main.py", signature="sig_65"),
                    CanonicalPart(
                        kind="tool_call",
                        tool_call_id="call_orphan_grep",
                        tool_name="grep_files",
                        tool_args={"pattern": "Health"},
                        signature="sig_65",
                    ),
                ],
            ),
            # Session crashed before `tool_result`; user resumed and prompted:
            CanonicalMessage(
                role="user",
                parts=[
                    CanonicalPart(kind="text", text="Can you push the code to my git repo please"),
                    wrap_dynamic_instructions("<skills_delta>git</skills_delta>"),
                ],
            ),
        ]
        step1_pair = [
            CanonicalMessage(
                role="assistant",
                producer_provider="google",
                producer_model_id="gemini-4-argon",
                parts=[
                    CanonicalPart(kind="tool_call", tool_call_id="call_git_1", tool_name="exec_pty", tool_args={"command": "git status"}, signature="sig_67")
                ],
            ),
            CanonicalMessage(
                role="tool",
                parts=[CanonicalPart(kind="tool_result", tool_call_id="call_git_1", tool_name="exec_pty", text="ok")],
            ),
        ]
        step2_pair = [
            CanonicalMessage(
                role="assistant",
                producer_provider="google",
                producer_model_id="gemini-4-argon",
                parts=[
                    CanonicalPart(kind="tool_call", tool_call_id="call_git_2", tool_name="exec_pty", tool_args={"command": "git push"}, signature="sig_69")
                ],
            ),
            CanonicalMessage(
                role="tool",
                parts=[CanonicalPart(kind="tool_result", tool_call_id="call_git_2", tool_name="exec_pty", text="timed_out", is_error=True)],
            ),
        ]

        w_s1 = build_gemini_payload("sys", transform_messages(base_hist, gem), [], gem)["contents"]
        w_s2 = build_gemini_payload("sys", transform_messages(base_hist + step1_pair, gem), [], gem)["contents"]
        w_s3 = build_gemini_payload("sys", transform_messages(base_hist + step1_pair + step2_pair, gem), [], gem)["contents"]
        for w in (w_s1, w_s2, w_s3):
            _assert_gemini_bijection(w)

        # Verify `grep_files` synthetic closure sits at [2], closing `model` bridge at [3], and user prompt at [4]!
        self.assertEqual(w_s3[1]["parts"][-1]["functionCall"]["name"], "grep_files")
        self.assertEqual(w_s3[2]["parts"][0]["functionResponse"]["name"], "grep_files")
        self.assertEqual(w_s3[3]["role"], "model")
        self.assertIn("Can you push the code", w_s3[4]["parts"][0]["text"])
        self.assertEqual(w_s3[5]["parts"][0]["functionCall"]["name"], "exec_pty")
        self.assertEqual(w_s3[6]["parts"][0]["functionResponse"]["name"], "exec_pty")
        self.assertEqual(w_s3[7]["parts"][0]["functionCall"]["name"], "exec_pty")
        self.assertEqual(w_s3[8]["parts"][0]["functionResponse"]["name"], "exec_pty")
        # Prefix [0..len(w_s2)-1] is byte-for-byte KV-cache identical between Step 2 and Step 3!
        self.assertEqual([canonical_json(c) for c in w_s2], [canonical_json(c) for c in w_s3[: len(w_s2)]])
        # Defense-in-depth: even raw untransformed `build_gemini_payload` enforces exact bijection & alternation!
        _assert_gemini_bijection(build_gemini_payload("sys", base_hist + step1_pair + step2_pair, [], gem)["contents"])

        # 2. Partial + out-of-order + duplicate + stray orphan tool results across ALL 3 provider families
        chaos_hist: list[CanonicalMessage] = [
            CanonicalMessage(role="tool", parts=[CanonicalPart(kind="tool_result", tool_call_id="stray_0", tool_name="read_file", text="orphan")]),
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Inspect 3 files")]),
            CanonicalMessage(
                role="assistant",
                parts=[
                    CanonicalPart(kind="tool_call", tool_call_id="dup_id", tool_name="read_file", tool_args={"path": "a.py"}),
                    CanonicalPart(kind="tool_call", tool_call_id="c_mid", tool_name="grep_files", tool_args={"pattern": "x"}),
                    CanonicalPart(kind="tool_call", tool_call_id="dup_id", tool_name="glob_files", tool_args={"pattern": "*.py"}),
                ],
            ),
            # Mid-batch dynamic user note + out-of-order partial tool result + duplicate result + foreign ID
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Mid-batch note")]),
            CanonicalMessage(
                role="tool",
                parts=[
                    CanonicalPart(kind="tool_result", tool_call_id="c_mid", tool_name="grep_files", text="hit"),
                    CanonicalPart(kind="tool_result", tool_call_id="c_mid", tool_name="grep_files", text="duplicate ignored"),
                    CanonicalPart(kind="tool_result", tool_call_id="foreign_99", tool_name="exec_pty", text="stray"),
                ],
            ),
        ]
        g_chaos = build_gemini_payload("sys", transform_messages(chaos_hist, gem), [], gem)["contents"]
        _assert_gemini_bijection(g_chaos)
        self.assertEqual(
            [p["functionResponse"]["name"] for p in g_chaos[2]["parts"]],
            ["read_file", "grep_files", "glob_files"],
        )
        self.assertEqual(g_chaos[3]["role"], "model")
        self.assertEqual(g_chaos[4]["parts"][0]["text"], "Mid-batch note")

        a_chaos = build_anthropic_payload("sys", transform_messages(chaos_hist, anth), [], anth)["messages"]
        tu_ids = [b["id"] for b in a_chaos[1]["content"] if b["type"] == "tool_use"]
        tr_ids = [b["tool_use_id"] for b in a_chaos[2]["content"] if b["type"] == "tool_result"]
        self.assertEqual(len(set(tu_ids)), 3)
        self.assertEqual(tr_ids, tu_ids)

        o_chaos = build_openai_chat_payload("sys", transform_messages(chaos_hist, oai), [], oai)["messages"]
        o_tc_ids = [t["id"] for t in o_chaos[2]["tool_calls"]]
        o_tr_ids = [m["tool_call_id"] for m in o_chaos[3:6]]
        self.assertEqual(o_tr_ids, o_tc_ids)

        # 3. `SessionRunner` mid-TOOL_DISPATCH crash persists exact synthetic closure in `.jsonl`
        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            ws.mkdir()
            home = pathlib.Path(tmp) / "home"
            jpath = home / "sessions" / "sess_crash_guard.jsonl"
            transport = ScriptedTransport(
                lambda _i, _p, _m, _t: CanonicalMessage(
                    role="assistant",
                    parts=[
                        CanonicalPart(kind="tool_call", tool_call_id="c1", tool_name="read_file", tool_args={"path": "x"}),
                        CanonicalPart(kind="tool_call", tool_call_id="c2", tool_name="grep_files", tool_args={"pattern": "y"}),
                    ],
                )
            )
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(CredentialVault(home, probe_ambient=False), transport),
                auth_profile=_dummy_auth("google"),
                journal_path=jpath,
                ush_home=home,
            )

            def _boom(*_a: Any, **_kw: Any) -> list[CanonicalPart]:
                raise RuntimeError("simulated mid-dispatch kill")

            runner.tools.dispatch_batch = _boom  # type: ignore[method-assign]
            try:
                with self.assertRaises(RuntimeError):
                    runner.run_turn("Trigger tool crash")
                reloaded = SessionDAG(jpath).active_messages()
                self.assertEqual([m.role for m in reloaded], ["user", "assistant", "tool"])
                self.assertEqual([p.tool_name for p in reloaded[2].parts], ["read_file", "grep_files"])
                _assert_gemini_bijection(build_gemini_payload("sys", transform_messages(reloaded, gem), [], gem)["contents"])
            finally:
                runner.close()

    # ------------------------------------------------------------------
    # CT-26: Cross-Harness Turn State Machine Hardening
    #        (Step-1 Rollback, Durable Pre-Turn Boundary Closure, Gemini 3+/4+ vs 2.5
    #         Multimodal/ID/Vertex Wire Laws, Compaction Alternation, & Stream Guards)
    # ------------------------------------------------------------------
    def test_ct26_turn_state_machine_rollback_alternation_compaction_and_stream_guards(self) -> None:
        from ush.brain.providers import WireError

        g4 = resolve_model_profile("gemini-4-argon", "google")
        g25 = resolve_model_profile("gemini-2.5-pro", "vertex")
        anth = resolve_model_profile("claude-opus-4-20250514", "anthropic")

        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            ws.mkdir()
            (ws / "hello.txt").write_text("alpha\n", encoding="utf-8")
            home = pathlib.Path(tmp) / "home"
            jpath = home / "sessions" / "sess_turn_laws.jsonl"

            # 1. Step-1 API failure rolls back uncommitted user prompt cleanly on Turn 1 AND Turn 2!
            step_mode = {"mode": "fail_step1"}

            def _transport_fn(
                call_idx: int,
                profile: Any,
                messages: list[CanonicalMessage],
                _tools: list[dict[str, Any]],
            ) -> CanonicalMessage:
                m = step_mode["mode"]
                if m == "fail_step1":
                    raise WireError(500, "transient upstream failure")
                if m == "ok_turn":
                    return CanonicalMessage(
                        role="assistant",
                        producer_provider="google",
                        producer_model_id=profile.model_id,
                        parts=[CanonicalPart(kind="text", text="Turn succeeded cleanly.")],
                    )
                if m == "tool_then_fail_step2":
                    if messages[-1].role != "tool":
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider="google",
                            producer_model_id=profile.model_id,
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="c_read_1",
                                    tool_name="read_file",
                                    tool_args={"path": "hello.txt"},
                                )
                            ],
                        )
                    raise WireError(503, "step-2 outage after tool committed")
                if m == "thinking_only_then_answer":
                    # First attempt emits internal thought ONLY; runner must auto-continue!
                    has_nudge = any("internal reasoning only" in m_i.plain_text() for m_i in messages)
                    if not has_nudge:
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider="google",
                            producer_model_id=profile.model_id,
                            parts=[CanonicalPart(kind="thinking", text="Silent chain of thought...", signature="QUJDREVGRw==")],
                            stop_reason=StopReason.STOP,
                        )
                    return CanonicalMessage(
                        role="assistant",
                        producer_provider="google",
                        producer_model_id=profile.model_id,
                        parts=[CanonicalPart(kind="text", text="Concrete visible answer.")],
                        stop_reason=StopReason.STOP,
                    )
                if m == "missing_sig_retry":
                    # First attempt raises MISSING_THOUGHT_SIGNATURE; client retries with strip_all_signatures=True
                    has_any_sig = any(
                        bool(pt.signature) for msg_i in messages for pt in msg_i.parts
                    )
                    if has_any_sig:
                        raise WireError(400, "Gemini stream finishReason=MISSING_THOUGHT_SIGNATURE")
                    return CanonicalMessage(
                        role="assistant",
                        producer_provider="google",
                        producer_model_id=profile.model_id,
                        parts=[CanonicalPart(kind="text", text="Recovered after signature scrub.")],
                    )
                raise AssertionError(f"Unexpected mode {m} at {call_idx}")

            client = UniversalModelClient(
                CredentialVault(home, probe_ambient=False),
                ScriptedTransport(_transport_fn),
                max_api_attempts=2,
            )
            runner = SessionRunner(
                workspace_root=ws,
                client=client,
                auth_profile=_dummy_auth("google"),
                model_profile=g4,
                journal_path=jpath,
                ush_home=home,
            )
            try:
                # 1a. Turn-1 failure leaves 0 active messages and 0 checkpoints (both in memory and on disk reload)
                with self.assertRaises(WireError):
                    runner.run_turn("Orphaned prompt on turn 1?")
                self.assertEqual(runner.dag.active_messages(), [])
                self.assertEqual(SessionDAG(jpath).active_messages(), [])
                self.assertEqual(len(runner.checkpoints), 0)

                # 1b. Successful turn commits cleanly; subsequent Step-1 failure preserves Turn 1 without orphan user tail!
                step_mode["mode"] = "ok_turn"
                out_ok = runner.run_turn("First real turn")
                self.assertEqual(out_ok.final_message.plain_text(), "Turn succeeded cleanly.")
                self.assertEqual([m.role for m in SessionDAG(jpath).active_messages()], ["user", "assistant"])

                step_mode["mode"] = "fail_step1"
                with self.assertRaises(WireError):
                    runner.run_turn("Should roll back cleanly")
                self.assertEqual([m.role for m in SessionDAG(jpath).active_messages()], ["user", "assistant"])

                # 1c. Step-2 failure AFTER tool execution NEVER rolls back committed tool work, AND the next
                #     `run_turn` durably closes the unanswered `role="tool"` boundary with a closing `assistant` turn!
                step_mode["mode"] = "tool_then_fail_step2"
                with self.assertRaises(WireError):
                    runner.run_turn("Read hello.txt then fail")
                self.assertEqual(
                    [m.role for m in SessionDAG(jpath).active_messages()],
                    ["user", "assistant", "user", "assistant", "tool"],
                )

                # 2. `THINKING_ONLY_RESPONSE` auto-continuation + durable pre-turn `closeUnansweredToolResponseTurn`!
                step_mode["mode"] = "thinking_only_then_answer"
                events_seen: list[str] = []
                out_think = runner.run_turn(
                    "Resume after unanswered tool turn",
                    on_event=lambda e: events_seen.append(str((e.meta or {}).get("event", ""))),
                )
                self.assertEqual(out_think.final_message.plain_text(), "Concrete visible answer.")
                self.assertIn("thinking_only_continue", events_seen)
                roles_after = [m.role for m in runner.dag.active_messages()]
                # Notice index 4 (`tool`) is followed by index 5 (`assistant` closure) BEFORE index 6 (`user`)!
                self.assertEqual(roles_after[4:7], ["tool", "assistant", "user"])

                # 3. `MISSING_THOUGHT_SIGNATURE` auto-scrub retry via `UniversalModelClient`
                step_mode["mode"] = "missing_sig_retry"
                out_sig = runner.run_turn("Trigger signature scrub retry")
                self.assertEqual(out_sig.final_message.plain_text(), "Recovered after signature scrub.")
            finally:
                runner.close()

            # 4. Gemini 3+/4+ vs Gemini 2.5 Wire Laws (`id`, `{output}`/`{error}`, multimodal `parts`, Vertex `#4032`)
            mm_hist = [
                CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="Capture UI")]),
                CanonicalMessage(
                    role="assistant",
                    producer_provider="vertex",
                    producer_model_id="gemini-4-argon",
                    parts=[
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="call_shot_1",
                            tool_name="read_file",
                            tool_args={"path": "ui.png"},
                            signature="not_base64_mock",
                        ),
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="call_shot_2",
                            tool_name="exec_pty",
                            tool_args={"command": "false"},
                        ),
                    ],
                ),
                CanonicalMessage(
                    role="tool",
                    parts=[
                        CanonicalPart(
                            kind="tool_result",
                            tool_call_id="call_shot_1",
                            tool_name="read_file",
                            text="rendered",
                            mime_type="image/png",
                            base64_data="iVBORw0KGgo=",
                        ),
                        CanonicalPart(
                            kind="tool_result",
                            tool_call_id="call_shot_2",
                            tool_name="exec_pty",
                            text="exit 1",
                            is_error=True,
                        ),
                    ],
                ),
            ]
            # Gemini 4 on Vertex: `id` present, multimodal `parts` nested inside `functionResponse`,
            # and invalid non-b64 mock sig replaced with 32-char `skip_thought_signature_validator`!
            g4_vertex_pay = build_gemini_payload("sys", mm_hist, [], g4, is_vertex=True)["contents"]
            self.assertEqual(g4_vertex_pay[1]["parts"][0]["functionCall"]["id"], "call_shot_1")
            self.assertEqual(
                g4_vertex_pay[1]["parts"][0].get("thoughtSignature"),
                "skip_thought_signature_validator",
            )
            self.assertNotIn("thoughtSignature", g4_vertex_pay[1]["parts"][1])
            fr0 = g4_vertex_pay[2]["parts"][0]["functionResponse"]
            fr1 = g4_vertex_pay[2]["parts"][1]["functionResponse"]
            self.assertEqual(fr0["id"], "call_shot_1")
            self.assertEqual(fr0["response"], {"output": "rendered"})
            self.assertIn("inlineData", fr0["parts"][0])
            self.assertEqual(fr1["id"], "call_shot_2")
            self.assertEqual(fr1["response"], {"error": "exit 1"})

            # Valid Base64 signature IS preserved on Vertex!
            mm_hist[1].parts[0].signature = "QUJDREVGR0hJSktMTU5PUA=="
            g4_valid_sig = build_gemini_payload("sys", mm_hist, [], g4, is_vertex=True)["contents"]
            self.assertEqual(g4_valid_sig[1]["parts"][0].get("thoughtSignature"), "QUJDREVGR0hJSktMTU5PUA==")

            # Gemini 2.5: `id` omitted on `functionCall`/`functionResponse`, and image expands into alternating turns!
            g25_pay = build_gemini_payload("sys", mm_hist, [], g25, is_vertex=True)["contents"]
            self.assertNotIn("id", g25_pay[1]["parts"][0]["functionCall"])
            self.assertNotIn("id", g25_pay[2]["parts"][0]["functionResponse"])
            self.assertNotIn("parts", g25_pay[2]["parts"][0]["functionResponse"])
            self.assertEqual([c["role"] for c in g25_pay], ["user", "model", "user", "model", "user"])
            self.assertTrue(any("inlineData" in p for p in g25_pay[4]["parts"]))

            # 5. Multi-turn Compaction snaps to `user` boundary & preserves `user -> assistant -> user` alternation!
            cdag = SessionDAG(home / "sessions" / "sess_compact_alt.jsonl")
            for t_i in range(4):
                cdag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text=f"User turn {t_i} " + ("u" * 200))]))
                cdag.append(
                    CanonicalMessage(
                        role="assistant",
                        parts=[CanonicalPart(kind="tool_call", tool_call_id=f"tc_{t_i}", tool_name="read_file", tool_args={"path": f"f{t_i}.py"})],
                    )
                )
                cdag.append(
                    CanonicalMessage(
                        role="tool",
                        parts=[CanonicalPart(kind="tool_result", tool_call_id=f"tc_{t_i}", tool_name="read_file", text="content " + ("x" * 240))],
                    )
                )
                cdag.append(CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text=f"Done {t_i}")]))
            self.assertTrue(cdag.maybe_compact(context_limit_tokens=400, ratio=0.5))
            c_active = cdag.active_messages()
            self.assertEqual([m.role for m in c_active[:3]], ["user", "assistant", "user"])
            g4_post_compact = build_gemini_payload("sys", transform_messages(c_active, g4), [], g4)["contents"]
            for i in range(len(g4_post_compact) - 1):
                self.assertNotEqual(g4_post_compact[i]["role"], g4_post_compact[i + 1]["role"])

            # 6. Anthropic strict block ordering (`thinking` first in `assistant`, `tool_result` first in `user`, start/end `user`)
            anth_dirty = [
                CanonicalMessage(
                    role="assistant",
                    producer_provider="anthropic",
                    producer_model_id=anth.model_id,
                    parts=[
                        CanonicalPart(kind="text", text="Leading text before thinking"),
                        CanonicalPart(kind="thinking", text="Signed thought", signature="sig_anth_1"),
                        CanonicalPart(kind="tool_call", tool_call_id="tu_1", tool_name="read_file", tool_args={"path": "a.py"}),
                    ],
                ),
                CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="User note before tool_result")]),
                CanonicalMessage(
                    role="tool",
                    parts=[CanonicalPart(kind="tool_result", tool_call_id="tu_1", tool_name="read_file", text="file_body")],
                ),
            ]
            anth_pay = build_anthropic_payload("sys", anth_dirty, [], anth)["messages"]
            self.assertEqual(anth_pay[0]["role"], "user")
            self.assertEqual(anth_pay[-1]["role"], "user")
            self.assertEqual(anth_pay[1]["content"][0]["type"], "thinking")
            self.assertEqual(anth_pay[2]["content"][0]["type"], "tool_result")
            self.assertEqual(anth_pay[2]["content"][1]["type"], "text")

    def test_ct27_enterprise_antigravity_claude_openai_litellm_and_429_process_recovery(self) -> None:
        """CT-27: Gemini Enterprise, Antigravity, Claude, OpenAI `/v1/responses`, LiteLLM, 429 & Dead-PTY Recovery."""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import signal
        from unittest.mock import patch

        from ush.auth.vault import fetch_openai_compat_models
        from ush.brain.providers import (
            HttpProviderTransport,
            WireError,
            build_openai_responses_payload,
            parse_retry_delay_seconds,
        )
        from ush.kernel.tools import ToolSpec

        # 0. Verify retry-delay parser across RFC/ms headers and Google RetryInfo JSON
        self.assertAlmostEqual(parse_retry_delay_seconds(429, {"retry-after-ms": "250"}, "") or 0.0, 0.25, places=2)
        self.assertAlmostEqual(parse_retry_delay_seconds(429, {"retry-after": "3"}, "") or 0.0, 3.0, places=2)
        google_429_body = json.dumps(
            {
                "error": {
                    "code": 429,
                    "message": "Resource exhausted",
                    "details": [{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "1.5s"}],
                }
            }
        )
        self.assertAlmostEqual(parse_retry_delay_seconds(429, {}, google_429_body) or 0.0, 1.5, places=2)

        recorded_reqs: list[dict[str, Any]] = []
        route_counters: dict[str, int] = {}

        class _MultiEndpointMockHandler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def _send_sse(self, frames: list[str], status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
                payload_bytes = "".join(frames).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(payload_bytes)))
                self.send_header("Connection", "close")
                for hk, hv in (extra_headers or {}).items():
                    self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(payload_bytes)
                self.wfile.flush()

            def _send_json(self, obj: dict[str, Any], status: int = 200, extra_headers: dict[str, str] | None = None) -> None:
                raw = json.dumps(obj).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                for hk, hv in (extra_headers or {}).items():
                    self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(raw)
                self.wfile.flush()

            def do_GET(self) -> None:  # noqa: N802
                if self.path.endswith("/models"):
                    self._send_json(
                        {
                            "data": [
                                {"id": "anthropic/claude-sonnet-4-20250514"},
                                {"id": "vertex_ai/gemini-2.5-pro"},
                            ]
                        }
                    )
                    return
                self._send_json({"error": "not found"}, status=404)

            def do_POST(self) -> None:  # noqa: N802
                nbytes = int(self.headers.get("Content-Length") or "0")
                raw_body = self.rfile.read(nbytes).decode("utf-8") if nbytes > 0 else "{}"
                body = json.loads(raw_body) if raw_body.strip() else {}
                hdrs = {k.lower(): v for k, v in self.headers.items()}
                recorded_reqs.append({"path": self.path, "headers": hdrs, "body": body})
                cnt = route_counters.get(self.path, 0) + 1
                route_counters[self.path] = cnt

                # 1a. Gemini Enterprise Express Mode (`/v1/publishers/google/models/...`)
                if "/v1/publishers/google/models/" in self.path:
                    chunk = {
                        "candidates": [
                            {
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {"thought": True, "text": "Enterprise Express thought.", "thoughtSignature": "c2lnX2V4cA=="},
                                        {"text": "Enterprise Express OK"},
                                    ],
                                },
                                "finishReason": "STOP",
                            }
                        ],
                        "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 6, "thoughtsTokenCount": 5},
                    }
                    self._send_sse([f"data: {json.dumps(chunk)}\n\n"])
                    return

                # 1b. Gemini Enterprise Multi-Region Cascade (`global` -> 429, `us-central1` -> 200)
                if "/locations/global/publishers/google/models/" in self.path:
                    self._send_json({"error": {"code": 429, "message": "Global region saturated"}}, status=429)
                    return
                if "/locations/us-central1/publishers/google/models/" in self.path:
                    chunk = {
                        "candidates": [{"content": {"role": "model", "parts": [{"text": "Failover us-central1 OK"}]}, "finishReason": "STOP"}]
                    }
                    self._send_sse([f"data: {json.dumps(chunk)}\n\n"])
                    return

                # 1c. Gemini Enterprise Code Assist (`v1internal:streamGenerateContent?alt=sse`)
                if "v1internal:streamGenerateContent" in self.path:
                    wrapped_evt = {
                        "response": {
                            "candidates": [
                                {"content": {"role": "model", "parts": [{"text": f"CodeAssist project={body.get('project')} OK"}]}, "finishReason": "STOP"}
                            ]
                        }
                    }
                    self._send_sse([f"data: {json.dumps(wrapped_evt)}\n\n"])
                    return

                # 2a. Antigravity Hosted Gemini Endpoint (`/antigravity/v1beta/models/...`)
                if self.path.startswith("/antigravity/v1beta/models/"):
                    chunk = {
                        "candidates": [
                            {
                                "content": {"role": "model", "parts": [{"text": f"Antigravity tier={body.get('serviceTier')} OK"}]},
                                "finishReason": "STOP",
                            }
                        ]
                    }
                    self._send_sse([f"data: {json.dumps(chunk)}\n\n"])
                    return

                # 3. Claude Default Endpoint (`/v1/messages`) with mid-stream 529 overloaded recovery on 1st attempt
                if self.path == "/v1/messages":
                    if cnt == 1:
                        err_frame = 'event: error\ndata: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}\n\n'
                        self._send_sse([err_frame], status=200)
                        return
                    frames = [
                        'data: {"type":"message_start","message":{"usage":{"input_tokens":15}}}\n\n',
                        'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}\n\n',
                        'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Claude deep thought"}}\n\n',
                        'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig_claude_ok"}}\n\n',
                        'data: {"type":"content_block_stop","index":0}\n\n',
                        'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}\n\n',
                        'data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Claude Bearer OK"}}\n\n',
                        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}\n\n',
                    ]
                    self._send_sse(frames)
                    return

                # 4a. OpenAI `/v1/responses` Endpoint
                if self.path == "/openai/v1/responses":
                    frames = [
                        'data: {"type":"response.reasoning_summary_text.delta","delta":"Responses CoT step"}\n\n',
                        'data: {"type":"response.output_text.delta","delta":"OpenAI Responses OK"}\n\n',
                        (
                            'data: {"type":"response.completed","response":{"status":"completed",'
                            '"output":[{"type":"reasoning","id":"rs_1","encrypted_content":"enc_blob_xyz","summary":[{"type":"summary_text","text":"Responses CoT step"}]},'
                            '{"type":"message","role":"assistant","content":[{"type":"output_text","text":"OpenAI Responses OK"}]}],'
                            '"usage":{"input_tokens":20,"output_tokens":10,"input_tokens_details":{"cached_tokens":8},"output_tokens_details":{"reasoning_tokens":6}}}}\n\n'
                        ),
                    ]
                    self._send_sse(frames)
                    return

                # 4b. Legacy OpenAI proxy returning 404 on `/responses` -> automatic fallback to `/chat/completions`
                if self.path == "/legacy-oai/v1/responses":
                    self._send_json({"error": {"message": "Responses endpoint not implemented"}}, status=404)
                    return
                if self.path == "/legacy-oai/v1/chat/completions":
                    frames = [
                        'data: {"choices":[{"delta":{"content":"Fallback ChatCompletions OK"},"finish_reason":"stop"}]}\n\n',
                        "data: [DONE]\n\n",
                    ]
                    self._send_sse(frames)
                    return

                # 5. LiteLLM Proxy (`/litellm/v1/chat/completions`):
                #    - Rejects `stream_options` on 1st call -> client auto-retries without `stream_options`
                #    - Rejects oversized prompt (`>900` chars) with `ContextWindowExceededError` -> SessionRunner auto-compacts & retries!
                if self.path == "/litellm/v1/chat/completions":
                    if "stream_options" in body:
                        self._send_json(
                            {"error": {"message": "Unsupported parameter: stream_options"}},
                            status=400,
                        )
                        return
                    serialized_msgs = json.dumps(body.get("messages", []))
                    if len(serialized_msgs) > 1200 and "<epistemic_compaction_handoff>" not in serialized_msgs:
                        self._send_json(
                            {"error": {"message": "litellm.ContextWindowExceededError: prompt is too long (exceeds maximum context length)"}},
                            status=400,
                        )
                        return
                    frames = [
                        'data: {"choices":[{"delta":{"reasoning_content":"LiteLLM R1 thought","content":"LiteLLM OK"},"finish_reason":"stop"}]}\n\n',
                        "data: [DONE]\n\n",
                    ]
                    self._send_sse(frames)
                    return

                # 6a. Persistent 429 endpoint (fails 3x with Retry-After, succeeds on 4th attempt)
                if self.path == "/ratelimit/v1/chat/completions":
                    if cnt <= 3:
                        self._send_json(
                            {"error": {"message": "Rate limit reached, try again in 15ms"}},
                            status=429,
                            extra_headers={"Retry-After-Ms": "15"},
                        )
                        return
                    self._send_sse(
                        [
                            'data: {"choices":[{"delta":{"content":"Recovered after 3x 429"},"finish_reason":"stop"}]}\n\n',
                            "data: [DONE]\n\n",
                        ]
                    )
                    return

                # 6b. Multi-profile 429 failover (`primary` always 429, `backup` 200)
                if self.path == "/failover-primary/v1/chat/completions":
                    self._send_json({"error": {"message": "Primary quota exhausted"}}, status=429, extra_headers={"Retry-After": "1"})
                    return
                if self.path == "/failover-backup/v1/chat/completions":
                    self._send_sse(
                        [
                            'data: {"choices":[{"delta":{"content":"Backup profile rescued turn"},"finish_reason":"stop"}]}\n\n',
                            "data: [DONE]\n\n",
                        ]
                    )
                    return

                self._send_json({"error": f"unhandled {self.path}"}, status=404)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _MultiEndpointMockHandler)
        port = srv.server_address[1]
        base = f"http://127.0.0.1:{port}"
        srv_thread = threading.Thread(target=srv.serve_forever, daemon=True)
        srv_thread.start()

        try:
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = pathlib.Path(tmp)
                home = tmp_path / "ush_home"
                ws = tmp_path / "ws"
                ws.mkdir()
                codex_dir = tmp_path / "codex_home"
                codex_dir.mkdir()
                claude_dir = tmp_path / "claude_home"
                claude_dir.mkdir()
                (codex_dir / "auth.json").write_text(
                    json.dumps({"tokens": {"access_token": "eyJ_codex_oauth_tok", "account_id": "acct_99"}}),
                    encoding="utf-8",
                )
                (claude_dir / ".credentials.json").write_text(
                    json.dumps({"claudeAiOauth": {"accessToken": "sk-ant-oat01-ambient-claude"}}),
                    encoding="utf-8",
                )

                sleep_calls: list[float] = []
                with patch.dict(os.environ, {"USH_DISABLE_KEYCHAIN": "1"}, clear=False):
                    vault = CredentialVault(home, probe_ambient=False)
                    transport = HttpProviderTransport(vault)
                    client = UniversalModelClient(
                        vault,
                        transport,
                        max_api_attempts=6,
                        sleep_fn=lambda secs: sleep_calls.append(secs),
                    )

                    # --- 1. Gemini Enterprise: Express Mode, Multi-Region Vertex Failover, & Code Assist `v1internal` ---
                    ent_express = vault.register_api_key_profile(
                        name="ent-express",
                        provider="gemini-enterprise",
                        api_key="AQ.enterprise-express-key",
                        base_url=base,
                        default_model="gemini-2.5-pro",
                        api_mode="express",
                        service_tier="priority",
                    )
                    p_ent = resolve_model_profile("gemini-enterprise:gemini-2.5-pro", "gemini-enterprise")
                    m_exp = client.generate(
                        profile=p_ent,
                        auth_profile=ent_express,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="ping")])],
                        tools=[],
                        session_id="sess_ent",
                    )
                    self.assertEqual(m_exp.plain_text(), "Enterprise Express OK")
                    self.assertEqual(m_exp.thinking_text(), "Enterprise Express thought.")
                    exp_req = next(r for r in recorded_reqs if "/v1/publishers/google/models/" in r["path"])
                    self.assertEqual(exp_req["headers"].get("x-goog-api-key"), "AQ.enterprise-express-key")
                    self.assertEqual(exp_req["body"].get("serviceTier"), "priority")

                    ent_vertex = vault.register_api_key_profile(
                        name="ent-vertex-global",
                        provider="gemini-enterprise",
                        api_key="ya29.mock-adc-token",
                        base_url=base,
                        auth_kind="bearer",
                        project_id="ent-proj-1",
                        region="global",
                        default_model="gemini-2.5-pro",
                        api_mode="vertex",
                    )
                    m_reg = client.generate(
                        profile=p_ent,
                        auth_profile=ent_vertex,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="cascade")])],
                        tools=[],
                        session_id="sess_ent",
                    )
                    self.assertEqual(m_reg.plain_text(), "Failover us-central1 OK")

                    ent_ca = vault.register_api_key_profile(
                        name="ent-code-assist",
                        provider="code-assist",
                        api_key="ya29.ca-token",
                        base_url=base,
                        auth_kind="bearer",
                        project_id="ca-ent-proj",
                        default_model="gemini-2.5-pro",
                        api_mode="code_assist",
                    )
                    p_ca = resolve_model_profile("code-assist:gemini-2.5-pro", "code-assist")
                    m_ca = client.generate(
                        profile=p_ca,
                        auth_profile=ent_ca,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="ca")])],
                        tools=[],
                        session_id="sess_ca",
                    )
                    self.assertEqual(m_ca.plain_text(), "CodeAssist project=ca-ent-proj OK")

                    # --- 2. Antigravity Hosted Gemini (`priority`) & Local LiteRT OpenAI-Compatible Endpoint ---
                    ag_hosted = vault.register_api_key_profile(
                        name="ag-hosted",
                        provider="antigravity",
                        api_key="ag-secret-key",
                        base_url=f"{base}/antigravity",
                        default_model="gemini-2.5-pro",
                        api_mode="gemini",
                        service_tier="priority",
                    )
                    p_ag = resolve_model_profile("antigravity:gemini-2.5-pro", "antigravity")
                    m_ag = client.generate(
                        profile=p_ag,
                        auth_profile=ag_hosted,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="ag")])],
                        tools=[],
                        session_id="sess_ag",
                    )
                    self.assertEqual(m_ag.plain_text(), "Antigravity tier=priority OK")

                    ag_litert = vault.register_api_key_profile(
                        name="ag-litert",
                        provider="antigravity",
                        api_key="",
                        auth_kind="none",
                        base_url=f"{base}/legacy-oai/v1",
                        default_model="litert-gemma-3",
                        api_mode="chat_completions",
                    )
                    p_ag_local = resolve_model_profile("antigravity:litert-gemma-3", "antigravity")
                    m_ag_local = client.generate(
                        profile=p_ag_local,
                        auth_profile=ag_litert,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="local")])],
                        tools=[],
                        session_id="sess_ag",
                    )
                    self.assertEqual(m_ag_local.plain_text(), "Fallback ChatCompletions OK")

                    # --- 3. Claude `/v1` Base-URL Normalization, Ambient Bearer Discovery & 529 Overloaded Recovery ---
                    with patch.dict(
                        os.environ,
                        {
                            "USH_DISABLE_KEYCHAIN": "1",
                            "HOME": str(tmp_path),
                            "CODEX_HOME": str(codex_dir),
                            "CLAUDE_CONFIG_DIR": str(claude_dir),
                            "ANTHROPIC_BASE_URL": f"{base}/v1",
                            "OPENAI_BASE_URL": f"{base}/openai/v1",
                            "OPENAI_ORG_ID": "org_crux_test",
                            "OPENAI_PROJECT": "proj_crux_test",
                        },
                        clear=True,
                    ):
                        amb_vault = CredentialVault(tmp_path / "amb_home", probe_ambient=False)
                        amb_vault.discover_ambient_credentials()
                        cli_claude = amb_vault._profiles["claude-oauth"]
                        cli_codex = amb_vault._profiles["codex-oauth"]
                        self.assertEqual(cli_claude.auth_kind, "oauth")
                        self.assertEqual(cli_codex.extra_headers.get("ChatGPT-Account-Id"), "acct_99")
                        self.assertEqual(cli_codex.extra_headers.get("OpenAI-Organization"), "org_crux_test")

                        amb_client = UniversalModelClient(
                            amb_vault,
                            HttpProviderTransport(amb_vault),
                            max_api_attempts=4,
                            sleep_fn=lambda secs: sleep_calls.append(secs),
                        )
                        p_claude = resolve_model_profile("claude-sonnet-4-20250514", "anthropic")
                        m_claude = amb_client.generate(
                            profile=p_claude,
                            auth_profile=cli_claude,
                            system_prompt="sys",
                            history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi claude")])],
                            tools=[],
                            session_id="sess_claude",
                        )
                        self.assertEqual(m_claude.plain_text(), "Claude Bearer OK")
                        self.assertEqual(m_claude.thinking_text(), "Claude deep thought")
                        claude_reqs = [r for r in recorded_reqs if r["path"] == "/v1/messages"]
                        self.assertEqual(len(claude_reqs), 2)  # 1st hit mid-stream 529 overloaded_error, 2nd succeeded!
                        self.assertEqual(claude_reqs[-1]["headers"].get("authorization"), "Bearer sk-ant-oat01-ambient-claude")

                        # --- 4. OpenAI `/v1/responses` Streaming, ChatGPT OAuth Rules, `encrypted_content` Round-Trip & 404 Fallback ---
                        p_gpt5 = resolve_model_profile("openai:gpt-5", "openai")
                        m_resp = amb_client.generate(
                            profile=p_gpt5,
                            auth_profile=cli_codex,
                            system_prompt="sys",
                            history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi gpt5")])],
                            tools=[],
                            session_id="sess_oai_resp",
                        )
                        self.assertEqual(m_resp.plain_text(), "OpenAI Responses OK")
                        self.assertEqual(m_resp.thinking_text(), "Responses CoT step")
                        resp_req = next(r for r in recorded_reqs if r["path"] == "/openai/v1/responses")
                        self.assertIs(resp_req["body"].get("store"), False)
                        self.assertNotIn("max_output_tokens", resp_req["body"])  # Omitted for ChatGPT OAuth non-sk- token!
                        self.assertEqual(resp_req["headers"].get("session_id"), "sess_oai_resp")
                        self.assertEqual(resp_req["headers"].get("openai-organization"), "org_crux_test")

                        # Verify Turn-2 `reasoning.encrypted_content` round-trip in `build_openai_responses_payload`
                        turn2_pay = build_openai_responses_payload(
                            "sys",
                            [
                                CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi gpt5")]),
                                m_resp,
                                CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="follow up")]),
                            ],
                            [],
                            p_gpt5,
                            session_id="sess_oai_resp",
                        )
                        enc_items = [it for it in turn2_pay["input"] if it.get("type") == "reasoning"]
                        self.assertEqual(len(enc_items), 1)
                        self.assertEqual(enc_items[0].get("encrypted_content"), "enc_blob_xyz")

                    # Verify automatic `/v1/responses` -> `/v1/chat/completions` fallback on 404
                    oai_legacy = vault.register_api_key_profile(
                        name="oai-legacy",
                        provider="openai",
                        api_key="sk-proj-legacy123",
                        base_url=f"{base}/legacy-oai/v1",
                        default_model="gpt-5",
                        api_mode="responses",
                    )
                    m_oai_fb = client.generate(
                        profile=resolve_model_profile("openai:gpt-5", "openai"),
                        auth_profile=oai_legacy,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="fallback")])],
                        tools=[],
                        session_id="sess_fb",
                    )
                    self.assertEqual(m_oai_fb.plain_text(), "Fallback ChatCompletions OK")

                    # --- 5. LiteLLM Verbatim Model Prefix, Dual Headers, Live `/models`, & Context-Overflow Auto-Compaction ---
                    litellm_prof = vault.register_api_key_profile(
                        name="litellm-gw",
                        provider="litellm",
                        api_key="sk-litellm-master",
                        base_url=f"{base}/litellm/v1",
                        default_model="anthropic/claude-sonnet-4-20250514",
                        api_mode="chat_completions",
                    )
                    live_models = fetch_openai_compat_models(litellm_prof.base_url, api_key="sk-litellm-master")
                    self.assertIn("anthropic/claude-sonnet-4-20250514", live_models)

                    p_lite = resolve_model_profile("litellm:anthropic/claude-sonnet-4-20250514", "litellm")
                    self.assertEqual(p_lite.provider, "litellm")
                    self.assertEqual(p_lite.model_id, "anthropic/claude-sonnet-4-20250514")

                    lite_runner = SessionRunner(
                        workspace_root=ws,
                        client=client,
                        auth_profile=litellm_prof,
                        model_profile=p_lite,
                        mode=PermissionMode.YOLO,
                        ush_home=home,
                    )
                    try:
                        # Seed prior messages with >1200 chars so the next turn triggers LiteLLM ContextWindowExceededError (400),
                        # verifying `SessionRunner` emergency-compacts the DAG and completes without failing!
                        lite_runner.dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="u1 " + ("A" * 700))]))
                        lite_runner.dag.append(CanonicalMessage(role="assistant", parts=[CanonicalPart(kind="text", text="a1 " + ("B" * 700))]))
                        overflow_evts: list[str] = []
                        out_lite = lite_runner.run_turn(
                            "Summarize via LiteLLM",
                            on_event=lambda e: overflow_evts.append(str(e.meta.get("event", ""))) if e.kind == "cognitive_event" else None,
                        )
                        self.assertEqual(out_lite.final_message.plain_text(), "LiteLLM OK")
                        self.assertIn("context_overflow_recovery", overflow_evts)
                        lite_reqs = [r for r in recorded_reqs if r["path"] == "/litellm/v1/chat/completions"]
                        self.assertEqual(lite_reqs[-1]["body"]["model"], "anthropic/claude-sonnet-4-20250514")
                        self.assertEqual(lite_reqs[-1]["headers"].get("x-litellm-api-key"), "sk-litellm-master")
                    finally:
                        lite_runner.close()

                    # --- 6a. Persistent `429` Backoff Recovery + Multi-Profile `429` Failover + Instant Cancel ---
                    rl_vault = CredentialVault(tmp_path / "rl_home", probe_ambient=False)
                    rl_prof = rl_vault.register_api_key_profile(
                        name="rl-single",
                        provider="openai",
                        api_key="sk-rl",
                        base_url=f"{base}/ratelimit/v1",
                        default_model="gpt-4.1",
                        api_mode="chat_completions",
                    )
                    rl_client = UniversalModelClient(
                        rl_vault,
                        HttpProviderTransport(rl_vault),
                        max_api_attempts=6,
                        sleep_fn=lambda secs: sleep_calls.append(secs),
                    )
                    before_sleeps = len(sleep_calls)
                    m_rl = rl_client.generate(
                        profile=resolve_model_profile("gpt-4.1", "openai"),
                        auth_profile=rl_prof,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="retry 429")])],
                        tools=[],
                        session_id="sess_rl",
                    )
                    self.assertEqual(m_rl.plain_text(), "Recovered after 3x 429")
                    self.assertGreaterEqual(len(sleep_calls) - before_sleeps, 3)

                    # Multi-profile 429 failover (`fo-primary` -> `fo-backup`)
                    fo_vault = CredentialVault(tmp_path / "fo_home", probe_ambient=False)
                    fo_prim = fo_vault.register_api_key_profile(
                        name="fo-primary",
                        provider="litellm",
                        api_key="sk-1",
                        base_url=f"{base}/failover-primary/v1",
                        default_model="gpt-4.1",
                        api_mode="chat_completions",
                    )
                    fo_vault.register_api_key_profile(
                        name="fo-backup",
                        provider="litellm",
                        api_key="sk-2",
                        base_url=f"{base}/failover-backup/v1",
                        default_model="gpt-4.1",
                        api_mode="chat_completions",
                    )
                    fo_client = UniversalModelClient(
                        fo_vault,
                        HttpProviderTransport(fo_vault),
                        max_api_attempts=4,
                        sleep_fn=lambda secs: sleep_calls.append(secs),
                    )
                    failover_events: list[str] = []
                    m_fo = fo_client.generate(
                        profile=resolve_model_profile("litellm:gpt-4.1", "litellm"),
                        auth_profile=fo_prim,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="failover now")])],
                        tools=[],
                        session_id="sess_fo",
                        on_event=lambda e: failover_events.append(str(e.meta.get("event", ""))) if e.kind == "cognitive_event" else None,
                    )
                    self.assertEqual(m_fo.plain_text(), "Backup profile rescued turn")
                    self.assertIn("api_failover", failover_events)

                    # Instant cancellation during 429 backoff
                    pre_cancel = threading.Event()
                    pre_cancel.set()
                    m_can = fo_client.generate(
                        profile=resolve_model_profile("litellm:gpt-4.1", "litellm"),
                        auth_profile=fo_prim,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="cancel")])],
                        tools=[],
                        session_id="sess_can",
                        cancel_event=pre_cancel,
                    )
                    self.assertEqual(m_can.stop_reason, StopReason.CANCELED)

                # --- 6b. Dead-PTY `SIGKILL` Auto-Respawn + Shell `exit` + Crashed Tool/Hook Process Containment ---
                pty_step = {"n": 0}

                def _pty_resilience_script(
                    _idx: int,
                    prof: ModelCapabilityProfile,
                    msgs: list[CanonicalMessage],
                    _tools: list[dict[str, Any]],
                ) -> CanonicalMessage:
                    pty_step["n"] += 1
                    step_n = pty_step["n"]
                    if step_n == 1:
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=prof.provider,
                            producer_model_id=prof.model_id,
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="tc_pty_1",
                                    tool_name="exec_pty",
                                    tool_args={"command": "echo ALIVE_AFTER_RESPAWN"},
                                ),
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="tc_crash_tool",
                                    tool_name="flaky_subproc",
                                    tool_args={},
                                ),
                            ],
                        )
                    if step_n == 2:
                        # Subshell `exit 9` kills the persistent shell mid-command; PTY manager must report rc=9 & respawn!
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=prof.provider,
                            producer_model_id=prof.model_id,
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="tc_pty_exit",
                                    tool_name="exec_pty",
                                    tool_args={"command": "exit 9"},
                                )
                            ],
                        )
                    if step_n == 3:
                        # Prove the persistent PTY still runs commands cleanly right after `exit 9`!
                        return CanonicalMessage(
                            role="assistant",
                            producer_provider=prof.provider,
                            producer_model_id=prof.model_id,
                            parts=[
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id="tc_pty_post_exit",
                                    tool_name="exec_pty",
                                    tool_args={"command": "echo RECOVERED_POST_EXIT"},
                                )
                            ],
                        )
                    tool_texts = "\n".join(p.text for m in msgs if m.role == "tool" for p in m.parts)
                    return CanonicalMessage(
                        role="assistant",
                        producer_provider=prof.provider,
                        producer_model_id=prof.model_id,
                        parts=[CanonicalPart(kind="text", text=f"All recovered:\n{tool_texts}")],
                    )

                reg = build_default_tool_registry()

                def _boom_handler(_args: dict[str, Any], _ctx: ToolExecutionContext) -> CanonicalPart:
                    raise OSError(32, "Broken child pipe in external tool")

                reg.register(
                    ToolSpec(
                        "flaky_subproc",
                        "Simulates a crashing child process",
                        {"type": "object", "properties": {}},
                        False,
                        _boom_handler,
                    )
                )
                hooks = HookSystem(ws, is_workspace_trusted=True)
                hooks_file = ws / "hooks.json"
                hooks_file.write_text(
                    json.dumps(
                        {
                            "hooks": {
                                "BeforeTool": [{"matcher": "exec_pty", "hooks": [{"command": "echo 'hook err' >&2; exit 1"}]}]
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                hooks.load_hooks_json(hooks_file, source="workspace")

                proc_runner = SessionRunner(
                    workspace_root=ws,
                    client=UniversalModelClient(vault, ScriptedTransport(_pty_resilience_script)),
                    auth_profile=_dummy_auth("google"),
                    model_profile=resolve_model_profile("gemini-2.5-pro", "google"),
                    mode=PermissionMode.YOLO,
                    tools=reg,
                    hook_runner=hooks,
                    ush_home=home,
                )
                try:
                    # Spawn the primary PTY first, then externally `SIGKILL` its bash process behind `PersistentPtyManager`'s back!
                    init_res = proc_runner.pty.execute("echo BOOT_OK")
                    self.assertIn("BOOT_OK", init_res.output)
                    prim_handle = proc_runner.pty._handles[proc_runner.pty.PRIMARY_ID]
                    os.kill(prim_handle.pid, signal.SIGKILL)
                    prim_handle.proc.wait(timeout=2.0)

                    # Now run a full multi-step turn: must auto-respawn dead PTY, contain `flaky_subproc` OSError & failing hook,
                    # survive `exit 9`, and complete!
                    proc_outcome = proc_runner.run_turn("Run resilient PTY & tool sequence")
                    final_txt = proc_outcome.final_message.plain_text()
                    self.assertIn("ALIVE_AFTER_RESPAWN", final_txt)
                    self.assertIn("BrokenPipeError", final_txt)
                    self.assertIn("RECOVERED_POST_EXIT", final_txt)

                    # Also verify `write_stdin` on an unknown/dead PTY returns structured `not_found` instead of raising KeyError
                    dead_stdin = proc_runner.pty.write_stdin("pty_nonexistent_999", "y\n")
                    self.assertEqual(dead_stdin.status, "not_found")
                finally:
                    proc_runner.close()
        finally:
            srv.shutdown()
            srv.server_close()

    # ------------------------------------------------------------------
    # CT-28: Zero Approval-Prompt Spam on Multi-Line Commands, 10-FPS Spinner & Parallel Calls
    # ------------------------------------------------------------------
    def test_ct28_approval_menu_zero_spam_and_parallel_serialization(self) -> None:
        from io import StringIO

        from ush.cli.main import HarnessRuntime, handle_slash_command
        from ush.cli.tui import VisualTuiController, clip_to_cols, decode_key_buffer, strip_ansi

        # 1. `clip_to_cols` fast-path (`vis_width <= max_cols`) MUST strip `\n`, `\r`, `\t`, `\v`, `\f`
        short_multiline = "cd /tmp &&\necho hello\t\tworld\r\ndone\x0bnext"
        clipped_short = clip_to_cols(short_multiline, 120)
        for bad_ch in ("\n", "\r", "\t", "\x0b", "\x0c"):
            self.assertNotIn(bad_ch, clipped_short)
        self.assertIn("cd /tmp && echo hello world done next", clipped_short)

        with tempfile.TemporaryDirectory() as tmp:
            ws = pathlib.Path(tmp) / "ws"
            home = pathlib.Path(tmp) / "home"
            ws.mkdir()
            home.mkdir()
            old_home = os.environ.get("USH_HOME")
            os.environ["USH_HOME"] = str(home)
            try:
                rt = HarnessRuntime(ws, mode=PermissionMode.DEFAULT)
                scr = StringIO()
                ctrl = VisualTuiController(
                    rt,
                    slash_handler=lambda r, ln: handle_slash_command(r, ln, headless=False),
                    out=scr,
                    use_color=True,
                    cols_fn=lambda: 92,
                )
                ctrl.turn_busy.set()
                ctrl.tracker.on_event(
                    StreamEvent(
                        kind="step_start",
                        step=1,
                        meta={"model": "gemini-4-argon", "thinking_effort": "high", "max_steps": 80, "context_tokens": 1200},
                    )
                )
                ctrl.tracker.on_event(
                    StreamEvent(
                        kind="tool_call",
                        step=1,
                        part=CanonicalPart(
                            kind="tool_call",
                            tool_call_id="call_ml_1",
                            tool_name="exec_pty",
                            tool_args={"command": "cd /tmp &&\necho line2\t\tok"},
                        ),
                    )
                )
                ctrl.comp.render()

                # 2. Trigger interactive approval on a multi-line `exec_pty` command from worker thread
                approved_box: list[bool] = []
                worker_done = threading.Event()

                def _ask_worker() -> None:
                    res = ctrl._request_tool_approval_menu(
                        "exec_pty",
                        {"command": "cd /tmp &&\necho line2\t\tok"},
                        "shell execution\nrequires acceptance",
                    )
                    approved_box.append(res)
                    worker_done.set()

                t = threading.Thread(target=_ask_worker, daemon=True)
                t.start()

                deadline = time.monotonic() + 2.0
                while not ctrl.is_awaiting_user and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ctrl.is_awaiting_user)

                # Snapshot exact output buffer right after initial approval box render
                initial_buf = scr.getvalue()
                self.assertEqual(strip_ansi(initial_buf).count("Approve `exec_pty`"), 1)

                # Simulate 25 timer ticks + redundant `comp.render()` calls while waiting for user acceptance:
                # MUST write 0 extra bytes and NEVER repeat or scroll the approval prompt!
                for _ in range(25):
                    if ctrl.turn_busy.is_set() and not ctrl.is_awaiting_user:
                        ctrl._spin_tick += 1
                    ctrl.comp.render()
                    time.sleep(0.004)

                self.assertEqual(
                    len(scr.getvalue()),
                    len(initial_buf),
                    "Approval prompt wrote extra bytes / spammed redraws while awaiting acceptance!",
                )
                self.assertEqual(strip_ansi(scr.getvalue()).count("Approve `exec_pty`"), 1)

                # Press `y` to accept once
                ctrl.handle_key(decode_key_buffer("y")[0][0])
                self.assertTrue(worker_done.wait(timeout=2.0))
                self.assertEqual(approved_box, [True])
                self.assertFalse(ctrl.is_awaiting_user)

                # 3. Concurrent `write_file` approvals serialize cleanly, and `e` (`auto_edit`) auto-approves #2!
                scr_parallel = StringIO()
                ctrl.comp.out = scr_parallel
                ctrl.comp._last_frame_sig = None
                results_par: list[bool] = []
                par_lock = threading.Lock()

                def _write_approval(idx: int) -> None:
                    ok = ctrl._request_tool_approval_menu(
                        "write_file",
                        {"path": f"src/mod_{idx}.py"},
                        "workspace write",
                    )
                    with par_lock:
                        results_par.append(ok)

                t1 = threading.Thread(target=_write_approval, args=(1,), daemon=True)
                t2 = threading.Thread(target=_write_approval, args=(2,), daemon=True)
                t1.start()
                t2.start()

                deadline = time.monotonic() + 2.0
                while not ctrl.is_awaiting_user and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ctrl.is_awaiting_user)

                # Choose `e` ("Allow & enable auto-edit") -> approves #1 AND auto-approves queued #2 without a second menu!
                ctrl.handle_key(decode_key_buffer("e")[0][0])
                t1.join(timeout=2.0)
                t2.join(timeout=2.0)
                self.assertEqual(results_par, [True, True])
                self.assertEqual(rt.runner.mode, PermissionMode.AUTO_EDIT)
                self.assertEqual(strip_ansi(scr_parallel.getvalue()).count("Approve `write_file`"), 1)
            finally:
                ctrl.should_exit = True
                rt.runner.cancel_event.set()
                rt.runner.close()
                if old_home is None:
                    os.environ.pop("USH_HOME", None)
                else:
                    os.environ["USH_HOME"] = old_home


if __name__ == "__main__":
    unittest.main(verbosity=2)
