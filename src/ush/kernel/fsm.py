"""Total 6-Phase Session Actor (`SessionRunner`), `TurnSteerQueue` & 100%-KV-Cached `/btw` Side-Chat.

Phases: PRE_INVOCATION -> GENERATE -> TOOL_DISPATCH -> POST_INVOCATION -> STOP_HOOKS -> DONE
Implements CT-02, CT-04, CT-08, CT-10, CT-13, CT-15, and CT-16.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import os
import pathlib
import subprocess
import threading
import time
from typing import Any, Callable

from ush.auth.vault import AuthProfile
from ush.brain.context_dag import PopperianLoopGuard, SessionDAG
from ush.brain.evaluator_isolation import IsolatedEvaluatorRunner
from ush.brain.model_tiers import build_adaptive_system_prompt, resolve_model_profile
from ush.brain.providers import UniversalModelClient, WireError, is_context_overflow_error
from ush.brain.trace import ThoughtTraceLedger
from ush.brain.wire import ToolIdBijectionMap, wrap_dynamic_instructions
from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    LoopPhase,
    ModelCapabilityProfile,
    ModelTier,
    PermissionMode,
    StopReason,
    StreamEvent,
    ThinkingEffort,
    TokenUsage,
    canonical_json,
    new_id,
)
from ush.kernel.patch import StaleReadTracker
from ush.kernel.pty_manager import PersistentPtyManager
from ush.kernel.security import PolicyEngine, SecurityFirewall
from ush.kernel.shadow_git import ShadowGitStore, ShadowSnapshot
from ush.kernel.tools import ToolExecutionContext, ToolRegistry, build_default_tool_registry


@dataclass
class TurnCheckpoint:
    entry_id: str
    shadow_snapshot: ShadowSnapshot | None
    prompt_preview: str


@dataclass
class TurnOutcome:
    final_message: CanonicalMessage
    steps_executed: int
    stop_reason: StopReason
    usage: TokenUsage
    thrash_upshift_requested: bool = False


class SessionRunner:
    """Single-writer 6-phase execution actor powering interactive TUI, headless `-p`, ACP, & Swarm islands."""

    def __init__(
        self,
        *,
        workspace_root: pathlib.Path,
        client: UniversalModelClient,
        auth_profile: AuthProfile,
        model_profile: ModelCapabilityProfile | None = None,
        mode: PermissionMode = PermissionMode.DEFAULT,
        session_id: str | None = None,
        journal_path: pathlib.Path | None = None,
        ush_home: pathlib.Path | None = None,
        tools: ToolRegistry | None = None,
        policy_engine: PolicyEngine | None = None,
        hook_runner: Any | None = None,
        skill_manager: Any | None = None,
        memory_manager: Any | None = None,
        thinking_effort: ThinkingEffort = ThinkingEffort.HIGH,
        trace_ledger: ThoughtTraceLedger | None = None,
        auto_pre_ship_review: bool = True,
        auto_cold_eyes_llm: bool = False,
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.session_id = session_id or new_id("sess")
        self.client = client
        self.auth_profile = auth_profile
        default_mid = auth_profile.default_model or "gemini-2.5-pro"
        self.model_profile = model_profile or resolve_model_profile(default_mid, auth_profile.provider)
        self.mode = mode

        self.firewall = SecurityFirewall(self.workspace_root)
        self.stale_tracker = StaleReadTracker()
        self.pty = PersistentPtyManager(self.workspace_root)
        self.shadow_git = ShadowGitStore(self.workspace_root, ush_home=ush_home)
        self.tools = tools or build_default_tool_registry()
        self.policy_engine = policy_engine or PolicyEngine()
        self.dag = SessionDAG(journal_path)
        self.thinking_effort = thinking_effort
        if hasattr(self.client.transport, "thinking_effort"):
            self.client.transport.thinking_effort = thinking_effort  # type: ignore[attr-defined]
        if trace_ledger is not None:
            self.trace_ledger: ThoughtTraceLedger | None = trace_ledger
        elif journal_path is not None:
            t_path = journal_path.with_name(f"{journal_path.stem}.trace.jsonl")
            self.trace_ledger = ThoughtTraceLedger(
                t_path, session_id=self.session_id, workspace=str(self.workspace_root)
            )
        else:
            self.trace_ledger = None
        self.id_map = ToolIdBijectionMap()
        self.loop_guard = PopperianLoopGuard()

        if hook_runner is None:
            from ush.platform.hooks_memory_ext import HookSystem

            self.hook_runner = HookSystem(
                self.workspace_root,
                is_workspace_trusted=False,
                ush_home=ush_home,
                enable_crux_evals=True,
                crux_evals_mode="complementary",
            )
        else:
            self.hook_runner = hook_runner
        self.skill_manager = skill_manager
        self.memory_manager = memory_manager

        self._active_turn_emit: Callable[[StreamEvent], None] | None = None
        self.isolated_evaluator = IsolatedEvaluatorRunner(
            vault=getattr(self.client, "vault", None),
            client=self.client,
            get_actor_auth=lambda: self.auth_profile,
            get_actor_profile=lambda: self.model_profile,
            thinking_effort=ThinkingEffort.MEDIUM,
            on_trace_event=self._emit_isolated_eval_event,
        )

        self.phase: LoopPhase = LoopPhase.DONE
        self.writer_lease_epoch: int = 0
        self.total_usage = TokenUsage()
        self.checkpoints: list[TurnCheckpoint] = []
        self.edits_attempted = 0
        self.edits_failed = 0
        self.auto_pre_ship_review = auto_pre_ship_review
        self.auto_cold_eyes_llm = auto_cold_eyes_llm

        self._steer_lock = threading.Lock()
        self._pending_steers: deque[CanonicalMessage] = deque()
        self.cancel_event = threading.Event()
        self._cached_turn0_system_prompt: str | None = None
        self._pending_jit_deltas: list[str] = []

        # Interactive UI callbacks (attachable by TUI / ACP)
        self.approval_callback: Callable[[str, dict[str, Any], str], bool] | None = None
        self.ask_user_callback: Callable[[str, list[str]], str] | None = None
        self.swarm_callback: Callable[[dict[str, Any]], str] | None = None

    # ---- Live Model Rotation & Mid-Turn Steering (CT-01, CT-13) ----

    def switch_model(self, model_id: str, auth_profile: AuthProfile | None = None) -> ModelCapabilityProfile:
        if auth_profile is not None:
            self.auth_profile = auth_profile
            self.model_profile = resolve_model_profile(model_id, self.auth_profile.provider)
            return self.model_profile

        resolved = resolve_model_profile(model_id, self.auth_profile.provider)
        if resolved.provider != self.auth_profile.provider and self.client.vault is not None:
            now = time.time()
            candidates = [
                p
                for p in self.client.vault.list_profiles()
                if p.provider == resolved.provider and p.is_available(now)
            ]
            if not candidates and resolved.provider in ("google", "vertex", "gemini-enterprise", "antigravity", "code-assist"):
                active_name = self.client.vault.active_profile_name
                candidates = sorted(
                    [
                        p
                        for p in self.client.vault.list_profiles()
                        if p.provider in ("google", "vertex", "gemini-enterprise", "antigravity", "code-assist")
                        and p.is_available(now)
                    ],
                    key=lambda p: (0 if p.name == active_name else 1, 0 if p.provider == "antigravity" else 1),
                )
            if candidates:
                self.auth_profile = candidates[0]
                resolved = resolve_model_profile(model_id, self.auth_profile.provider)
        self.model_profile = resolved
        return self.model_profile

    def set_thinking_effort(self, effort: ThinkingEffort) -> ThinkingEffort:
        self.thinking_effort = effort
        if hasattr(self.client.transport, "thinking_effort"):
            self.client.transport.thinking_effort = effort  # type: ignore[attr-defined]
        return self.thinking_effort

    def pending_steer_count(self) -> int:
        with self._steer_lock:
            return len(self._pending_steers)

    def enqueue_steer(self, text: str, image_part: CanonicalPart | None = None) -> None:
        """Queues a live human steer safely drained ONLY at POST_INVOCATION after tool_result closure (CT-13)."""
        parts = [CanonicalPart(kind="text", text=f"[LIVE_USER_STEER]: {text}")]
        if image_part is not None:
            parts.append(image_part)
        with self._steer_lock:
            self._pending_steers.append(CanonicalMessage(role="user", parts=parts))

    def _drain_steers(self) -> int:
        drained = 0
        with self._steer_lock:
            while self._pending_steers:
                self.dag.append(self._pending_steers.popleft())
                drained += 1
        return drained

    def _emit_isolated_eval_event(self, evt: StreamEvent) -> None:
        if self.trace_ledger is not None:
            self.trace_ledger.record_stream_event(evt)
        emit = self._active_turn_emit
        if emit is not None:
            emit(evt)

    def _build_consult_evaluator_callback(self) -> Callable[[dict[str, Any]], dict[str, Any]]:
        def _consult(req: dict[str, Any]) -> dict[str, Any]:
            chosen = str(req.get("evaluator") or req.get("lens") or "cold_eyes_review")

            def _last_meta() -> dict[str, Any]:
                tr = self.isolated_evaluator.last_trace
                if tr is None:
                    return {}
                return {
                    "evaluator_model": tr.evaluator_model,
                    "evaluator_thought": tr.evaluator_thought,
                }

            return self.hook_runner.consult_evaluator(
                lens=chosen,
                objective=str(req.get("goal") or req.get("objective") or ""),
                target_paths=list(req.get("files") or req.get("target_paths") or []),
                attached_images=list(req.get("attached_images") or []),
                proposed_claims=req.get("proposed_claims"),
                actor_model_family=self.model_profile.family,
                llm_judge=self.isolated_evaluator.make_llm_callable(chosen),
                llm_breaker=self.isolated_evaluator.make_llm_callable("adversarial_falsifier"),
                get_last_eval_meta=_last_meta,
            )

        return _consult

    def get_system_prompt(self) -> str:
        """Turn-0 system prompt is generated ONCE and kept immutable to preserve >90% prefix KV-cache."""
        if self._cached_turn0_system_prompt is None:
            mem = self.memory_manager.initial_memory_block() if self.memory_manager else ""
            sk = (
                self.skill_manager.initial_catalog_block(self.model_profile.skill_catalog_budget_tokens)
                if self.skill_manager
                else ""
            )
            ev = (
                self.hook_runner.initial_evaluator_catalog()
                if (self.hook_runner and hasattr(self.hook_runner, "initial_evaluator_catalog"))
                else ""
            )
            self._cached_turn0_system_prompt = build_adaptive_system_prompt(
                self.model_profile,
                str(self.workspace_root),
                self.mode,
                memory_block=mem,
                initial_skills_block=sk,
                initial_evaluators_block=ev,
            )
        return self._cached_turn0_system_prompt

    # ---- `/btw` Cache-Safe Zero-Mutation Side Question (CT-15) ----

    def ask_btw(
        self,
        question: str,
        *,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> CanonicalMessage:
        """Answers a side question mid-task reusing 100% of the exact prefix KV cache with `tools=[]` and 0 DAG writes."""
        snapshot = list(self.dag.active_messages(apply_hygiene=True))
        # Ensure we never send an unclosed tool_call batch to the side fork
        if snapshot and snapshot[-1].role == "assistant" and snapshot[-1].tool_calls():
            snapshot = snapshot[:-1]
        btw_user = CanonicalMessage(
            role="user",
            parts=[
                CanonicalPart(
                    kind="text",
                    text=(
                        f"[SIDE_QUESTION /btw — answer concisely using current session context; "
                        f"no tools are enabled and this exchange will not alter the main task]:\n{question}"
                    ),
                )
            ],
        )
        fork_history = snapshot + [btw_user]
        reply = self.client.generate(
            profile=self.model_profile,
            auth_profile=self.auth_profile,
            system_prompt=self.get_system_prompt(),
            history=fork_history,
            tools=[],  # Strictly NO_TOOLS!
            session_id=self.session_id,
            id_map=ToolIdBijectionMap(
                raw_to_safe=dict(self.id_map.raw_to_safe),
                safe_to_raw=dict(self.id_map.safe_to_raw),
            ),
            on_event=on_event,
            cancel_event=self.cancel_event,
        )
        if reply.usage:
            self.total_usage.add(reply.usage)
        return reply

    # ---- Total 6-Phase Execution Loop ----

    def _close_unanswered_turn_boundaries(self) -> None:
        """Durable pre-prompt boundary repair (`geminiChat.ts:534-537, 856-936`):

        1. If durable history ended on `assistant` with unanswered `tool_call`s, append synthetic `role="tool"` error results.
        2. If durable history ended on `role="tool"` (an unanswered tool response turn), append a closing `assistant` turn
           so role alternation (`user -> assistant -> tool -> assistant -> user`) holds across all providers.
        """
        active = self.dag.active_messages(apply_hygiene=False)
        if not active:
            return
        last = active[-1]
        if last.role == "assistant":
            pending_tcs = last.tool_calls()
            if pending_tcs:
                synth = [
                    CanonicalPart(
                        kind="tool_result",
                        tool_call_id=tc.tool_call_id,
                        tool_name=tc.tool_name or "tool",
                        text="[SYNTHETIC_REPAIR: prior tool invocation interrupted before result was recorded]",
                        is_error=True,
                    )
                    for tc in pending_tcs
                ]
                self.dag.append(CanonicalMessage(role="tool", parts=synth))
                active = self.dag.active_messages(apply_hygiene=False)
                last = active[-1]
        if last.role == "tool":
            self.dag.append(
                CanonicalMessage(
                    role="assistant",
                    producer_provider=self.auth_profile.provider,
                    producer_model_id=self.model_profile.model_id,
                    parts=[
                        CanonicalPart(
                            kind="text",
                            text="[The previous response was interrupted before it completed.]",
                        )
                    ],
                )
            )

    def run_turn(
        self,
        user_prompt: str,
        *,
        extra_parts: list[CanonicalPart] | None = None,
        max_steps: int = 80,
        upshift_on_thrash_streak: int = 0,
        on_event: Callable[[StreamEvent], None] | None = None,
    ) -> TurnOutcome:
        self.writer_lease_epoch += 1
        self.cancel_event.clear()
        turn_t0 = time.monotonic()
        turn_usage = TokenUsage()
        steps = 0
        output_retries = 0
        stop_vetoes = 0
        complementary_stop_nudges = 0
        context_overflow_recoveries = 0
        modified_in_step: list[str] = []
        turn_modified_files: list[str] = []
        did_consult_eval_in_turn = False
        turn_edits_since_eval = 0
        eval_wave_nudge_sent = False
        thrash_eval_nudge_sent = False

        def _emit(evt: StreamEvent) -> None:
            if self.trace_ledger is not None:
                self.trace_ledger.record_stream_event(evt)
            if on_event is not None:
                on_event(evt)

        prev_turn_emit = self._active_turn_emit
        self._active_turn_emit = _emit

        # Lazy pre-mutation Shadow Git checkpoint (0ms startup on read-only turns; captures strictly BEFORE first write tool!)
        snap: ShadowSnapshot | None = None

        if self.hook_runner:
            ok, extra_ctx = self.hook_runner.fire("UserPromptSubmit", {"prompt": user_prompt})
            if not ok:
                aborted = CanonicalMessage(
                    role="assistant",
                    parts=[CanonicalPart(kind="text", text=f"[Blocked by UserPromptSubmit hook]: {extra_ctx}")],
                    stop_reason=StopReason.CANCELED,
                )
                return TurnOutcome(aborted, 0, StopReason.CANCELED, turn_usage)
            if extra_ctx:
                self._pending_jit_deltas.append(extra_ctx)

        # Ensure Turn-0 system prompt catalogs (skills + evaluators) populate `_shown_fingerprints` BEFORE delta computation
        self.get_system_prompt()

        # Compute BM25/path skill & evaluator catalog deltas for tail `<dynamic_instructions>` (0 Turn-0 drift!)
        if self.skill_manager:
            delta = self.skill_manager.compute_turn_delta(
                query=user_prompt,
                budget_tokens=self.model_profile.skill_catalog_budget_tokens,
            )
            if delta:
                self._pending_jit_deltas.append(delta)
        if self.hook_runner and hasattr(self.hook_runner, "compute_evaluator_turn_delta"):
            ev_delta = self.hook_runner.compute_evaluator_turn_delta(user_prompt)
            if ev_delta:
                self._pending_jit_deltas.append(ev_delta)

        # Durable pre-turn boundary repair (`geminiChat.ts:534-537`):
        # Close any unanswered trailing `assistant(tool_call)` and/or unanswered trailing `tool(tool_result)`
        # BEFORE recording a genuinely new user prompt so tool pairing and role alternation stay intact.
        self._close_unanswered_turn_boundaries()

        user_parts = [CanonicalPart(kind="text", text=user_prompt)]
        if extra_parts:
            user_parts.extend(extra_parts)
        if self._pending_jit_deltas:
            user_parts.append(wrap_dynamic_instructions("\n".join(self._pending_jit_deltas)))
            self._pending_jit_deltas.clear()

        pre_turn_leaf = self.dag.leaf_id
        pre_cp_len = len(self.checkpoints)
        committed_steps = 0

        u_msg = self.dag.append(CanonicalMessage(role="user", parts=user_parts))
        self.checkpoints.append(
            TurnCheckpoint(entry_id=u_msg.entry_id, shadow_snapshot=snap, prompt_preview=user_prompt[:80])
        )
        _emit(
            StreamEvent(
                kind="turn_start",
                text=user_prompt,
                meta={
                    "model": self.model_profile.model_id,
                    "tier": self.model_profile.tier.value,
                    "thinking_effort": self.thinking_effort.value,
                    "max_steps": max_steps,
                    "trace_path": str(self.trace_ledger.trace_path) if self.trace_ledger else "",
                },
            )
        )

        last_assistant = CanonicalMessage(role="assistant", parts=[])
        self.phase = LoopPhase.PRE_INVOCATION

        try:
            while steps < max_steps:
                # --- Phase 1: PRE_INVOCATION ---
                if self.cancel_event.is_set():
                    last_assistant.stop_reason = StopReason.CANCELED
                    if committed_steps == 0 and not last_assistant.parts:
                        self.dag.rollback_uncommitted(pre_turn_leaf)
                        del self.checkpoints[pre_cp_len:]
                        self.phase = LoopPhase.DONE
                        break
                    self.phase = LoopPhase.STOP_HOOKS
                elif self.phase == LoopPhase.PRE_INVOCATION:
                    # 0.85 Context Compaction check (CT-10)
                    pre_ctx = self.dag.approx_context_tokens()
                    if self.hook_runner and pre_ctx >= int(
                        self.model_profile.context_window_tokens * self.model_profile.compaction_ratio
                    ):
                        self.hook_runner.fire("PreCompress", {"session_id": self.session_id})
                    compacted = self.dag.maybe_compact(
                        self.model_profile.context_window_tokens,
                        ratio=self.model_profile.compaction_ratio,
                    )
                    if compacted:
                        post_ctx = self.dag.approx_context_tokens()
                        _emit(
                            StreamEvent(
                                kind="cognitive_event",
                                step=steps + 1,
                                text=(
                                    f"Distilled history into <epistemic_state> ({pre_ctx:,} -> {post_ctx:,} ctx tok; "
                                    f"100% raw chains of thought remain preserved in trace ledger)"
                                ),
                                meta={"event": "epistemic_compaction", "pre_tokens": pre_ctx, "post_tokens": post_ctx},
                            )
                        )
                    self.phase = LoopPhase.GENERATE

                # --- Phase 2: GENERATE ---
                if self.phase == LoopPhase.GENERATE:
                    steps += 1
                    step_t0 = time.monotonic()
                    step_state: dict[str, Any] = {
                        "saw_think_delta": False,
                        "saw_text_delta": False,
                        "saw_tool_call": False,
                        "thought_ms": 0.0,
                    }
                    if hasattr(self.client.transport, "thinking_effort"):
                        self.client.transport.thinking_effort = self.thinking_effort  # type: ignore[attr-defined]
                    _emit(
                        StreamEvent(
                            kind="step_start",
                            step=steps,
                            elapsed_ms=(step_t0 - turn_t0) * 1000.0,
                            meta={
                                "model": self.model_profile.model_id,
                                "thinking_effort": self.thinking_effort.value,
                                "context_tokens": self.dag.approx_context_tokens(),
                                "max_steps": max_steps,
                            },
                        )
                    )
                    allowed_tools: set[str] | None = (
                        set(self.policy_engine.READ_ONLY_TOOLS | {"ask_user"})
                        if self.mode == PermissionMode.PLAN
                        else None
                    )
                    if self.hook_runner:
                        allowed_tools = self.hook_runner.filter_tools(
                            allowed_tools, self.tools.names(), self.model_profile
                        )
                        self.hook_runner.fire("BeforeModel", {"step": steps, "model": self.model_profile.model_id})

                    skill_enum = self.skill_manager.model_invocable_names() if self.skill_manager else []
                    eval_enum = (
                        self.hook_runner.all_evaluator_ids()
                        if (self.hook_runner and hasattr(self.hook_runner, "all_evaluator_ids"))
                        else None
                    )
                    wire_tools = self.tools.wire_schemas(
                        allowed_names=allowed_tools,
                        skill_enum=skill_enum,
                        evaluator_enum=eval_enum,
                    )

                    hist = self.dag.active_messages(apply_hygiene=True)
                    if self._pending_jit_deltas and hist:
                        dyn_part = wrap_dynamic_instructions("\n".join(self._pending_jit_deltas))
                        self._pending_jit_deltas.clear()
                        # Keep `role="tool"` strictly pure `tool_result` parts across Gemini 4 / OpenAI / Anthropic;
                        # only merge inline when the tail message is already a `user` turn.
                        if hist[-1].role == "user":
                            last_copy = CanonicalMessage.from_dict(hist[-1].to_dict())
                            last_copy.parts.append(dyn_part)
                            hist = hist[:-1] + [last_copy]
                        else:
                            hist = hist + [CanonicalMessage(role="user", parts=[dyn_part])]

                    def _step_evt(e: StreamEvent) -> None:
                        e.step = steps
                        if e.kind == "thinking_delta":
                            step_state["saw_think_delta"] = True
                        elif e.kind == "text_delta":
                            step_state["saw_text_delta"] = True
                        elif e.kind == "tool_call":
                            step_state["saw_tool_call"] = True
                        elif e.kind == "thinking_end":
                            step_state["thought_ms"] = float(e.elapsed_ms)
                        if on_event is not None:
                            on_event(e)

                    try:
                        last_assistant = self.client.generate(
                            profile=self.model_profile,
                            auth_profile=self.auth_profile,
                            system_prompt=self.get_system_prompt(),
                            history=hist,
                            tools=wire_tools,
                            session_id=self.session_id,
                            id_map=self.id_map,
                            on_event=_step_evt,
                            cancel_event=self.cancel_event,
                        )
                    except WireError as exc:
                        if (
                            context_overflow_recoveries < 2
                            and is_context_overflow_error(exc.status, exc.detail)
                            and self.dag.maybe_compact(self.model_profile.context_window_tokens, ratio=0.0, force=True)
                        ):
                            context_overflow_recoveries += 1
                            steps = max(0, steps - 1)
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps + 1,
                                    text=(
                                        f"Endpoint context limit reached (HTTP {exc.status}) — "
                                        f"emergency-compacted session history and retrying step"
                                    ),
                                    meta={"event": "context_overflow_recovery", "status": exc.status},
                                )
                            )
                            self.phase = LoopPhase.GENERATE
                            continue
                        if self.trace_ledger is not None:
                            self.trace_ledger.record_stream_event(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=str(exc),
                                    meta={"event": "provider_error"},
                                )
                            )
                        raise
                    except Exception as exc:
                        if self.trace_ledger is not None:
                            self.trace_ledger.record_stream_event(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=str(exc),
                                    meta={"event": "provider_error"},
                                )
                            )
                        raise
                    step_ms = (time.monotonic() - step_t0) * 1000.0
                    if last_assistant.usage:
                        turn_usage.add(last_assistant.usage)
                        self.total_usage.add(last_assistant.usage)

                    thought_txt = last_assistant.thinking_text()
                    has_sig = any(bool(p.signature) for p in last_assistant.parts if p.kind in ("thinking", "tool_call"))
                    if thought_txt and not step_state["saw_think_delta"] and on_event is not None:
                        on_event(StreamEvent(kind="thinking_delta", text=thought_txt, step=steps))
                        on_event(
                            StreamEvent(
                                kind="thinking_end",
                                step=steps,
                                elapsed_ms=step_ms,
                                meta={
                                    "thought_chars": len(thought_txt),
                                    "reasoning_tokens": (
                                        last_assistant.usage.reasoning_tokens
                                        if last_assistant.usage and last_assistant.usage.reasoning_tokens
                                        else max(1, len(thought_txt) // 4)
                                    ),
                                    "has_signature": has_sig,
                                },
                            )
                        )
                    plain_txt = last_assistant.plain_text()
                    if plain_txt and not step_state["saw_text_delta"] and on_event is not None:
                        on_event(StreamEvent(kind="text_delta", text=plain_txt, step=steps))
                    if not step_state["saw_tool_call"] and on_event is not None:
                        for tcp in last_assistant.tool_calls():
                            on_event(StreamEvent(kind="tool_call", part=tcp, step=steps))
                    if self.trace_ledger is not None:
                        u_dict = last_assistant.usage.__dict__ if last_assistant.usage else {}
                        self.trace_ledger.record_assistant_step(
                            step=steps,
                            model_id=self.model_profile.model_id,
                            thinking_effort=self.thinking_effort.value,
                            thought_text=thought_txt,
                            response_text=last_assistant.plain_text(),
                            tool_calls=[
                                {"id": p.tool_call_id, "name": p.tool_name, "args": p.tool_args or {}}
                                for p in last_assistant.tool_calls()
                            ],
                            has_signature=has_sig,
                            step_ms=step_ms,
                            thought_ms=float(step_state["thought_ms"] or step_ms),
                            usage=dict(u_dict),
                            stop_reason=last_assistant.stop_reason.value,
                        )
                    if self.hook_runner:
                        self.hook_runner.fire("AfterModel", {"stop_reason": last_assistant.stop_reason.value})

                    sr = last_assistant.stop_reason
                    tcalls = last_assistant.tool_calls()

                    # Deterministic ModelStopReason transition table
                    if sr in (StopReason.CANCELED, StopReason.CONTENT_FILTER, StopReason.FATAL_ERROR):
                        if sr == StopReason.CANCELED and committed_steps == 0 and not plain_txt.strip() and not tcalls:
                            self.dag.rollback_uncommitted(pre_turn_leaf)
                            del self.checkpoints[pre_cp_len:]
                            self.phase = LoopPhase.DONE
                            break
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        self.phase = LoopPhase.STOP_HOOKS
                    elif sr == StopReason.MAX_TOKENS and not tcalls:
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        if output_retries < self.client.max_output_retries:
                            output_retries += 1
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=f"Hit MAX_TOKENS; discarded truncated JSON & auto-continuing (retry {output_retries}/{self.client.max_output_retries})",
                                    meta={"event": "max_tokens_continue"},
                                )
                            )
                            self.dag.append(
                                CanonicalMessage(
                                    role="user",
                                    parts=[
                                        wrap_dynamic_instructions(
                                            "Previous response hit MAX_TOKENS (any partial tool call was discarded). "
                                            "Continue directly from where you stopped."
                                        )
                                    ],
                                )
                            )
                            self.phase = LoopPhase.GENERATE
                            continue
                        self.phase = LoopPhase.STOP_HOOKS
                    elif sr == StopReason.IMPROPER_FORMAT and not tcalls:
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        if output_retries < self.client.max_output_retries:
                            output_retries += 1
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=f"Dropped malformed tool JSON & injected 1-turn repair prompt (retry {output_retries}/{self.client.max_output_retries})",
                                    meta={"event": "tool_json_repair"},
                                )
                            )
                            self.dag.append(
                                CanonicalMessage(
                                    role="user",
                                    parts=[
                                        wrap_dynamic_instructions(
                                            "Malformed tool call JSON detected and dropped. Re-issue ONE valid JSON tool call matching the schema."
                                        )
                                    ],
                                )
                            )
                            self.phase = LoopPhase.GENERATE
                            continue
                        self.phase = LoopPhase.STOP_HOOKS
                    elif sr == StopReason.STOP and not tcalls and not plain_txt.strip() and bool(thought_txt):
                        # `THINKING_ONLY_RESPONSE` guard (`geminiChat.ts:1744-1752`):
                        # Model emitted internal thoughts only with zero visible output or tool calls.
                        if output_retries < self.client.max_output_retries:
                            output_retries += 1
                            self.dag.append(last_assistant)
                            committed_steps += 1
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=f"Thinking-only response with no visible text or tool call; auto-continuing (retry {output_retries}/{self.client.max_output_retries})",
                                    meta={"event": "thinking_only_continue"},
                                )
                            )
                            self.dag.append(
                                CanonicalMessage(
                                    role="user",
                                    parts=[
                                        wrap_dynamic_instructions(
                                            "Your previous turn contained internal reasoning only without a visible response or tool call. "
                                            "Emit your concrete tool call or user-facing response now."
                                        )
                                    ],
                                )
                            )
                            self.phase = LoopPhase.GENERATE
                            continue
                        last_assistant.parts.append(CanonicalPart(kind="text", text="[Completed internal reasoning.]"))
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        self.phase = LoopPhase.POST_INVOCATION
                    elif tcalls:
                        # Tier-B stability: clamp to 1 tool call per step if compact local model emitted multiple
                        if self.model_profile.tier == ModelTier.B and len(tcalls) > 1:
                            keep_first = tcalls[0]
                            last_assistant.parts = [
                                p for p in last_assistant.parts if p.kind != "tool_call"
                            ] + [keep_first]
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        self.phase = LoopPhase.TOOL_DISPATCH
                    else:
                        self.dag.append(last_assistant)
                        committed_steps += 1
                        self.phase = LoopPhase.POST_INVOCATION

                # --- Phase 3: TOOL_DISPATCH ---
                if self.phase == LoopPhase.TOOL_DISPATCH:
                    modified_in_step = []
                    tcalls = last_assistant.tool_calls()

                    def _on_access(fp: pathlib.Path) -> None:
                        if self.memory_manager:
                            jit_mem = self.memory_manager.on_file_accessed(fp)
                            if jit_mem:
                                self._pending_jit_deltas.append(jit_mem)
                        if self.skill_manager:
                            sk_delta = self.skill_manager.on_file_accessed(
                                fp, self.model_profile.skill_catalog_budget_tokens
                            )
                            if sk_delta:
                                self._pending_jit_deltas.append(sk_delta)
                        if self.hook_runner and hasattr(self.hook_runner, "on_file_accessed_evaluator_delta"):
                            ev_delta = self.hook_runner.on_file_accessed_evaluator_delta(fp)
                            if ev_delta:
                                self._pending_jit_deltas.append(ev_delta)

                    def _on_mod(fp: pathlib.Path, _diff: str) -> None:
                        rel = str(fp)
                        if rel not in modified_in_step:
                            modified_in_step.append(rel)

                    def _ctx_factory(cid: str) -> ToolExecutionContext:
                        return ToolExecutionContext(
                            call_id=cid,
                            firewall=self.firewall,
                            stale_tracker=self.stale_tracker,
                            pty=self.pty,
                            policy_engine=self.policy_engine,
                            mode=self.mode,
                            preview_bytes=self.model_profile.tool_preview_bytes,
                            cancel_event=self.cancel_event,
                            on_file_accessed=_on_access,
                            on_file_modified=_on_mod,
                            approval_callback=self.approval_callback,
                            ask_user_callback=self.ask_user_callback,
                            swarm_callback=self.swarm_callback,
                            skill_activate_callback=(
                                (lambda name: self.skill_manager.activate(name, self.firewall))
                                if self.skill_manager
                                else None
                            ),
                            evaluator_consult_callback=(
                                self._build_consult_evaluator_callback()
                                if (self.hook_runner and hasattr(self.hook_runner, "consult_evaluator"))
                                else None
                            ),
                        )

                    # Capture pre-mutation Shadow Git snapshot strictly BEFORE the first mutating tool (CT-09, 0ms read overhead!)
                    cur_cp = self.checkpoints[-1]
                    if cur_cp.shadow_snapshot is None and any(
                        (tc.tool_name or "") not in self.policy_engine.READ_ONLY_TOOLS for tc in tcalls
                    ):
                        try:
                            cur_cp.shadow_snapshot = self.shadow_git.capture(label=f"turn:{user_prompt[:48]}")
                        except (subprocess.TimeoutExpired, RuntimeError):
                            cur_cp.shadow_snapshot = None

                    before_cb = self.hook_runner.before_tool if self.hook_runner else None
                    after_cb = self.hook_runner.after_tool if self.hook_runner else None

                    def _on_tr(tr_part: CanonicalPart, tr_ms: float) -> None:
                        _emit(StreamEvent(kind="tool_result", part=tr_part, step=steps, elapsed_ms=tr_ms))

                    try:
                        results = self.tools.dispatch_batch(
                            tcalls,
                            _ctx_factory,
                            max_parallel_reads=self.model_profile.max_parallel_read_tools,
                            before_tool_hook=before_cb,
                            after_tool_hook=after_cb,
                            on_tool_result=_on_tr,
                        )
                    except BaseException:
                        # CT-16 / INV-WIRE-BIJECTION: Never leave a persisted `assistant(tool_call)` without its
                        # exact 1-to-1 `role="tool"` closure in `.jsonl` if interrupted/killed mid-dispatch!
                        synth_results = [
                            CanonicalPart(
                                kind="tool_result",
                                tool_call_id=tc.tool_call_id,
                                tool_name=tc.tool_name or "tool",
                                text="[Interrupted/canceled before tool result completed]",
                                is_error=True,
                            )
                            for tc in tcalls
                        ]
                        self.dag.append(
                            CanonicalMessage(role="tool", parts=synth_results, modified_files=list(modified_in_step))
                        )
                        committed_steps += 1
                        raise

                    # Track KPI edit metrics, Popperian falsification streaks & utility-based evaluator hints
                    guard_notes: list[str] = []
                    did_consult_eval = False
                    ui_edited_in_step = False
                    for mf in modified_in_step:
                        if mf not in turn_modified_files:
                            turn_modified_files.append(mf)
                    for tc, tr in zip(tcalls, results):
                        if tc.tool_name == "consult_evaluator":
                            did_consult_eval = True
                            if not tr.is_error:
                                did_consult_eval_in_turn = True
                            turn_edits_since_eval = 0
                            eval_wave_nudge_sent = False
                            thrash_eval_nudge_sent = False
                        if tc.tool_name in ("edit_file", "write_file"):
                            if tc.tool_name == "edit_file":
                                self.edits_attempted += 1
                                if tr.is_error:
                                    self.edits_failed += 1
                            if not tr.is_error:
                                turn_edits_since_eval += 1
                                tpath = str((tc.tool_args or {}).get("path", "")).lower()
                                if tpath.endswith((".html", ".htm", ".jsx", ".tsx", ".vue", ".svg")):
                                    ui_edited_in_step = True
                        nudge = self.loop_guard.record(
                            str(tc.tool_name),
                            canonical_json(tc.tool_args or {})[:240],
                            tr.is_error,
                            tr.text,
                        )
                        if nudge:
                            guard_notes.append(nudge)

                    if not did_consult_eval:
                        if self.loop_guard.thrash_streak >= 2 and not thrash_eval_nudge_sent:
                            thrash_eval_nudge_sent = True
                            guard_notes.append(
                                "Clean-room debugger hint: 2+ repeated failures detected — call "
                                "`consult_evaluator(evaluator='break_loop', files=[...])` to get <=3 orthogonal "
                                "root-cause hypotheses & probes unpolluted by failed retries."
                            )
                        elif (ui_edited_in_step or turn_edits_since_eval >= 2) and not eval_wave_nudge_sent:
                            eval_wave_nudge_sent = True
                            if ui_edited_in_step:
                                guard_notes.append(
                                    "Pre-ship cold-eyes hint: UI/visualization updated — capture a screenshot via `render_ui(target=...)`, "
                                    "then call `consult_evaluator(evaluator='cold_eyes_review', files=[..., 'shot.png'])` (PASS or <=5 fixes)."
                                )
                            else:
                                guard_notes.append(
                                    "Pre-ship cold-eyes hint: non-trivial code updated — after running tests via `run_command` / `exec_pty`, "
                                    "call `consult_evaluator(evaluator='cold_eyes_review', files=[...])` before finalizing (PASS or <=5 fixes)."
                                )

                    tool_msg = CanonicalMessage(
                        role="tool",
                        parts=results,
                        modified_files=list(modified_in_step),
                    )
                    self.dag.append(tool_msg)
                    committed_steps += 1
                    if guard_notes:
                        self._pending_jit_deltas.extend(guard_notes)
                        for gn in guard_notes:
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=gn[:220],
                                    meta={"event": "popperian_guard", "thrash_streak": self.loop_guard.thrash_streak},
                                )
                            )

                    if upshift_on_thrash_streak > 0 and self.loop_guard.thrash_streak >= upshift_on_thrash_streak:
                        self.phase = LoopPhase.DONE
                        return TurnOutcome(
                            last_assistant,
                            steps,
                            StopReason.STOP,
                            turn_usage,
                            thrash_upshift_requested=True,
                        )

                    self.phase = LoopPhase.STOP_HOOKS if self.cancel_event.is_set() else LoopPhase.POST_INVOCATION

                # --- Phase 4: POST_INVOCATION (Protocol-Safe Steer Drain + Quiescence Check) ---
                if self.phase == LoopPhase.POST_INVOCATION:
                    drained = self._drain_steers()
                    if drained > 0:
                        _emit(
                            StreamEvent(
                                kind="cognitive_event",
                                step=steps,
                                text=f"Merged {drained} live human steer message(s) cleanly at safe tool boundary",
                                meta={"event": "live_steer_injected", "count": drained},
                            )
                        )
                    had_tools = bool(last_assistant.tool_calls())
                    if drained > 0 or had_tools:
                        self.phase = LoopPhase.PRE_INVOCATION
                        continue
                    # 5-Condition Quiescence check (is_daemon servers do NOT block!)
                    if self.pty.blocking_background_count() == 0:
                        self.phase = LoopPhase.STOP_HOOKS
                    else:
                        self.phase = LoopPhase.STOP_HOOKS

                # --- Phase 5: STOP_HOOKS ---
                if self.phase == LoopPhase.STOP_HOOKS:
                    if (
                        self.hook_runner
                        and not self.cancel_event.is_set()
                        and last_assistant.stop_reason == StopReason.STOP
                        and stop_vetoes < 3
                    ):
                        allow_stop, feedback = self.hook_runner.fire(
                            "Stop", {"final_text": last_assistant.plain_text()[:1000]}
                        )
                        eval_rep = getattr(self.hook_runner, "last_eval_report", None)
                        if eval_rep is not None:
                            ev_ids = [v.evaluator_id for v in getattr(eval_rep, "verdicts", [])]
                            _emit(
                                StreamEvent(
                                    kind="cognitive_event",
                                    step=steps,
                                    text=(
                                        f"Crux-Evals Scorecard: decision={eval_rep.final_decision} "
                                        f"min_bars={eval_rep.min_bars_score:.2f} "
                                        f"lcb={eval_rep.conservative_lcb_score:.2f} "
                                        f"evaluators={ev_ids}"
                                    ),
                                    meta={
                                        "event": "crux_evals_scorecard",
                                        "decision": eval_rep.final_decision,
                                        "min_bars_score": eval_rep.min_bars_score,
                                        "conservative_lcb_score": eval_rep.conservative_lcb_score,
                                        "evaluators": ev_ids,
                                    },
                                )
                            )
                        if not allow_stop:
                            stop_vetoes += 1
                            self.dag.append(
                                CanonicalMessage(
                                    role="user",
                                    parts=[wrap_dynamic_instructions(f"[STOP_HOOK_VETO]: {feedback}")],
                                )
                            )
                            self.phase = LoopPhase.PRE_INVOCATION
                            continue
                        if (
                            eval_rep is not None
                            and complementary_stop_nudges == 0
                            and steps < max_steps
                            and getattr(eval_rep, "final_decision", "ALLOW") in ("VETO", "REWIND")
                            and any(getattr(v, "hard_vetoes_triggered", ()) for v in getattr(eval_rep, "verdicts", []))
                        ):
                            advisory = eval_rep.combined_feedback(mode="complementary").strip()
                            if advisory:
                                complementary_stop_nudges += 1
                                self.dag.append(
                                    CanonicalMessage(
                                        role="user",
                                        parts=[
                                            wrap_dynamic_instructions(
                                                f"[CRUX_EVALS_ADVISORY]: {advisory}\n"
                                                "Resolve or empirically verify the findings above before finishing."
                                            )
                                        ],
                                    )
                                )
                                self.phase = LoopPhase.PRE_INVOCATION
                                continue
                        if (
                            self.auto_pre_ship_review
                            and getattr(self.hook_runner, "enable_crux_evals", False)
                            and hasattr(self.hook_runner, "consult_evaluator")
                            and complementary_stop_nudges == 0
                            and steps < max_steps
                            and not did_consult_eval_in_turn
                            and (
                                len(turn_modified_files) >= 2
                                or any(
                                    fp.lower().endswith((".html", ".htm", ".jsx", ".tsx", ".vue", ".svg"))
                                    for fp in turn_modified_files
                                )
                            )
                        ):
                            use_llm = self.auto_cold_eyes_llm or (os.environ.get("CRUX_AUTO_COLD_EYES_LLM") == "1")
                            auto_res = self.hook_runner.consult_evaluator(
                                lens="cold_eyes_review",
                                target_paths=list(turn_modified_files),
                                actor_model_family=self.model_profile.family,
                                llm_judge=(
                                    self.isolated_evaluator.make_llm_callable("cold_eyes_review")
                                    if use_llm
                                    else None
                                ),
                                llm_breaker=None,
                                record_consult=False,
                            )
                            findings = auto_res.get("findings") if isinstance(auto_res, dict) else None
                            if isinstance(findings, list) and findings and auto_res.get("status") == "FINDINGS":
                                did_consult_eval_in_turn = True
                                complementary_stop_nudges += 1
                                bullet_lines = [
                                    f"- [{it.get('target', 'artifact')}] {it.get('issue', '')} -> Fix: {it.get('fix', '')}"
                                    for it in findings[:5]
                                    if isinstance(it, dict)
                                ]
                                _emit(
                                    StreamEvent(
                                        kind="cognitive_event",
                                        step=steps,
                                        text=f"Auto Pre-Ship Cold-Eyes Gate caught {len(bullet_lines)} defect(s) across {len(turn_modified_files)} modified file(s)",
                                        meta={
                                            "event": "auto_pre_ship_cold_eyes",
                                            "findings_count": len(bullet_lines),
                                            "files": list(turn_modified_files),
                                        },
                                    )
                                )
                                self.dag.append(
                                    CanonicalMessage(
                                        role="user",
                                        parts=[
                                            wrap_dynamic_instructions(
                                                "[AUTO_PRE_SHIP_COLD_EYES]: Clean-room pre-ship checks found concrete issues in modified deliverables:\n"
                                                + "\n".join(bullet_lines)
                                                + "\nResolve these surgical items before delivering the final answer (never expose internal evaluator tags to the user)."
                                            )
                                        ],
                                    )
                                )
                                self.phase = LoopPhase.PRE_INVOCATION
                                continue
                    self.phase = LoopPhase.DONE
                    break
        except BaseException:
            # Step-1 Rollback Law (`geminiChat.ts:506-518, 815-842`, `qwen_code/llm-chat.ts:6800-6867`):
            # Never leave an unanswered orphan `user` prompt in `.jsonl` when Step 1 throws before any model/tool commit!
            if committed_steps == 0:
                self.dag.rollback_uncommitted(pre_turn_leaf)
                del self.checkpoints[pre_cp_len:]
            self.phase = LoopPhase.DONE
            raise
        finally:
            self._active_turn_emit = prev_turn_emit

        self.phase = LoopPhase.DONE
        _emit(
            StreamEvent(
                kind="done",
                step=steps,
                elapsed_ms=(time.monotonic() - turn_t0) * 1000.0,
                stop_reason=last_assistant.stop_reason,
                usage=turn_usage,
                meta={"trace_path": str(self.trace_ledger.trace_path) if self.trace_ledger else ""},
            )
        )
        return TurnOutcome(last_assistant, steps, last_assistant.stop_reason, turn_usage)

    def restore_checkpoint(
        self, index: int = -1, *, mode: str = "code+chat", reason: str = ""
    ) -> tuple[list[str], CanonicalMessage | None]:
        """Time-travels code via Shadow Git and/or rewinds DAG with `<branch_lessons>` (`code+chat|code-only|chat-only`)."""
        if not self.checkpoints:
            return [], None
        cp = self.checkpoints[index]
        restored_files: list[str] = []
        lesson: CanonicalMessage | None = None
        if mode in ("code+chat", "code-only") and cp.shadow_snapshot is not None:
            restored_files = self.shadow_git.restore(cp.shadow_snapshot)
        if mode in ("code+chat", "chat-only"):
            lesson = self.dag.rewind_to(cp.entry_id, inject_branch_lessons=True, failure_note=reason)
        if self.trace_ledger is not None:
            self.trace_ledger.append(
                "cognitive_event",
                step=0,
                event="branch_rewind",
                detail=f"Rewound session ({mode}) to `{cp.entry_id}`; restored {len(restored_files)} file(s). Reason: {reason or 'user /restore'}",
            )
        return restored_files, lesson

    def close(self) -> None:
        self.pty.close_all()
