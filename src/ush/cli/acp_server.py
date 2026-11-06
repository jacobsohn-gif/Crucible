"""Authenticated Loopback (`127.0.0.1` ONLY) & Stdio ACP / NDJSON JSON-RPC 2.0 Daemon (`crux serve`)."""

from __future__ import annotations

import hmac
import inspect
import json
import os
import secrets
import socketserver
import sys
from typing import Any, Callable

from ush.kernel.fsm import SessionRunner
from ush.kernel.security import redact_secrets

MAX_RPC_LINE_BYTES = 2 * 1024 * 1024  # 2 MiB hard frame cap preventing memory DoS


def handle_rpc_request(runner: SessionRunner, req: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(req, dict):
        return {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
    req_id = req.get("id")
    method = str(req.get("method", ""))
    params = req.get("params") if isinstance(req.get("params"), dict) else {}

    try:
        if method in ("initialize", "acp/initialize"):
            result: Any = {
                "protocolVersion": "1.0",
                "serverInfo": {"name": "crux", "title": "Crucible", "version": "0.1"},
                "capabilities": {
                    "streaming": True,
                    "btwSideQuestion": True,
                    "liveTurnSteer": True,
                    "shadowGitRestore": True,
                    "elasticSwarmGears": ["g1", "g2", "g3", "g4"],
                },
            }
        elif method in ("session/status", "thread/status"):
            result = {
                "session_id": runner.session_id,
                "phase": runner.phase.value,
                "model": runner.model_profile.model_id,
                "tier": runner.model_profile.tier.value,
                "auth_profile": runner.auth_profile.name,
                "mode": runner.mode.value,
                "approx_context_tokens": runner.dag.approx_context_tokens(),
                "cache_hit_rate": round(runner.total_usage.cache_hit_rate, 4),
            }
        elif method in ("session/prompt", "turn/start"):
            prompt = str(params.get("prompt", ""))
            max_steps = max(1, min(200, int(params.get("max_steps", 30))))
            outcome = runner.run_turn(prompt, max_steps=max_steps)
            result = {
                "text": outcome.final_message.plain_text(),
                "steps": outcome.steps_executed,
                "stop_reason": outcome.stop_reason.value,
                "input_tokens": outcome.usage.input_tokens,
                "output_tokens": outcome.usage.output_tokens,
                "cached_input_tokens": outcome.usage.cached_input_tokens,
            }
        elif method == "session/btw":
            reply = runner.ask_btw(str(params.get("question", "")))
            result = {"text": reply.plain_text(), "stop_reason": reply.stop_reason.value}
        elif method == "session/steer":
            runner.enqueue_steer(str(params.get("text", "")))
            result = {"queued": True}
        elif method == "session/cancel":
            runner.cancel_event.set()
            result = {"cancelled": True}
        elif method == "session/switch_model":
            np = runner.switch_model(str(params.get("model", "")))
            result = {"model": np.model_id, "provider": np.provider, "tier": np.tier.value}
        else:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        return {"jsonrpc": "2.0", "id": req_id, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32000, "message": redact_secrets(str(exc))},
        }


def _verify_token_constant_time(candidate: str, expected: str) -> bool:
    if not candidate or not expected:
        return False
    return hmac.compare_digest(candidate.encode("utf-8"), expected.encode("utf-8"))


def serve_stdio(runner: SessionRunner, input_stream: Any | None = None, output_stream: Any | None = None) -> int:
    in_s = input_stream if input_stream is not None else sys.stdin
    out_s = output_stream if output_stream is not None else sys.stdout
    while True:
        raw = in_s.readline(MAX_RPC_LINE_BYTES + 1)
        if not raw:
            break
        if len(raw.encode("utf-8") if isinstance(raw, str) else raw) > MAX_RPC_LINE_BYTES:
            # Drain remainder of oversized line
            while raw and not str(raw).endswith("\n"):
                raw = in_s.readline(MAX_RPC_LINE_BYTES)
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Frame exceeds 2 MiB limit"}}
            out_s.write(json.dumps(resp) + "\n")
            out_s.flush()
            continue
        line = (raw if isinstance(raw, str) else raw.decode("utf-8", errors="replace")).strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
        else:
            resp = handle_rpc_request(runner, req)
        out_s.write(json.dumps(resp) + "\n")
        out_s.flush()
    return 0


def serve_loopback_tcp(
    runner: SessionRunner,
    port: int = 0,
    on_bound: Callable[..., None] | None = None,
    *,
    auth_token: str | None = None,
    host: str = "127.0.0.1",
) -> None:
    """Strictly binds `127.0.0.1` ONLY and requires constant-time bearer token authentication."""
    if host != "127.0.0.1":
        raise PermissionError(f"Security violation: ACP TCP daemon refuses non-loopback bind `{host}`.")
    effective_token = (auth_token or os.environ.get("USH_ACP_TOKEN") or secrets.token_urlsafe(32)).strip()

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            session_authed = False
            while True:
                raw = self.rfile.readline(MAX_RPC_LINE_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_RPC_LINE_BYTES:
                    while raw and not raw.endswith(b"\n"):
                        raw = self.rfile.readline(MAX_RPC_LINE_BYTES)
                    err_big = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Frame exceeds 2 MiB limit"}}
                    self.wfile.write((json.dumps(err_big) + "\n").encode("utf-8"))
                    self.wfile.flush()
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    resp: dict[str, Any] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
                else:
                    if not isinstance(req, dict):
                        resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
                    else:
                        req_id = req.get("id")
                        method = str(req.get("method", ""))
                        params = req.get("params") if isinstance(req.get("params"), dict) else {}
                        supplied_tok = str(
                            req.get("auth_token")
                            or params.get("_auth_token")
                            or params.get("token")
                            or ""
                        )
                        if method in ("authenticate", "acp/authenticate"):
                            if _verify_token_constant_time(supplied_tok, effective_token):
                                session_authed = True
                                resp = {"jsonrpc": "2.0", "id": req_id, "result": {"authenticated": True}}
                            else:
                                resp = {
                                    "jsonrpc": "2.0",
                                    "id": req_id,
                                    "error": {"code": -32001, "message": "Unauthorized: invalid ACP token"},
                                }
                        elif session_authed or _verify_token_constant_time(supplied_tok, effective_token):
                            session_authed = True
                            resp = handle_rpc_request(runner, req)
                        else:
                            resp = {
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "error": {"code": -32001, "message": "Unauthorized: missing or invalid ACP token"},
                            }
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()

    class _ReusableServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with _ReusableServer(("127.0.0.1", port), _Handler) as srv:
        actual_port = int(srv.server_address[1])
        if on_bound:
            n_params = len(inspect.signature(on_bound).parameters)
            if n_params >= 3:
                on_bound(actual_port, effective_token, srv)
            elif n_params >= 2:
                on_bound(actual_port, effective_token)
            else:
                on_bound(actual_port)
        else:
            print(
                f"Crux JSON-RPC/ACP server listening strictly on 127.0.0.1:{actual_port} "
                f"(auth_token required via USH_ACP_TOKEN / acp/authenticate)",
                flush=True,
            )
        srv.serve_forever()
