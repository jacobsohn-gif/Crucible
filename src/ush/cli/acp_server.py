"""Loopback (`127.0.0.1` ONLY) & Stdio ACP / NDJSON JSON-RPC 2.0 Daemon (`ush serve --acp | --jsonrpc`)."""

from __future__ import annotations

import json
import socketserver
import sys
from typing import Any, Callable

from ush.kernel.fsm import SessionRunner


def handle_rpc_request(runner: SessionRunner, req: dict[str, Any]) -> dict[str, Any]:
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
            max_steps = int(params.get("max_steps", 30))
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
            "error": {"code": -32000, "message": str(exc)},
        }


def serve_stdio(runner: SessionRunner) -> int:
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
        else:
            resp = handle_rpc_request(runner, req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()
    return 0


def serve_loopback_tcp(runner: SessionRunner, port: int = 0, on_bound: Callable[[int], None] | None = None) -> None:
    """Strictly binds `127.0.0.1` (NEVER `0.0.0.0`!)."""

    class _Handler(socketserver.StreamRequestHandler):
        def handle(self) -> None:
            for raw in self.rfile:
                line = raw.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    req = json.loads(line)
                except json.JSONDecodeError:
                    resp = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Invalid JSON"}}
                else:
                    resp = handle_rpc_request(runner, req)
                self.wfile.write((json.dumps(resp) + "\n").encode("utf-8"))
                self.wfile.flush()

    class _ReusableServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    with _ReusableServer(("127.0.0.1", port), _Handler) as srv:
        actual_port = int(srv.server_address[1])
        if on_bound:
            on_bound(actual_port)
        else:
            print(f"USH JSON-RPC/ACP server listening strictly on 127.0.0.1:{actual_port}", flush=True)
        srv.serve_forever()
