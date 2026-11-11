"""Strict Model-vs-Evaluator Isolation Bridge (`IsolatedEvaluatorRunner`).

Enforces the hard architectural boundary between the coding Actor model and Evaluators:
1. Evaluators ALWAYS execute in a fresh, zero-history, tool-free context (`tools=[]`,
   `session_id="crux_eval_isolated:<tag>:<nonce>"`) — NEVER sharing or mutating the
   Actor's `SessionDAG` or warm `agy acp` conversation state.
2. Prefers a distinct cross-family model (`vault.select_cross_family_profile`), or a
   configured distinct `CRUX_EVAL_MODEL`, while still keeping independent analysis +
   thinking enabled on the evaluator call itself.
3. Captures the Evaluator's own thought and structured analysis separately from the
   Actor's Chain-of-Thought so neither side contaminates the other.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import threading
from typing import Callable

from ush.auth.vault import AuthProfile, CredentialVault, PROVIDER_FAMILY_MAP
from ush.brain.model_tiers import resolve_model_profile
from ush.brain.providers import HttpProviderTransport, UniversalModelClient
from ush.core.types import (
    CanonicalMessage,
    CanonicalPart,
    ModelCapabilityProfile,
    StreamEvent,
    ThinkingEffort,
    new_id,
)


@dataclass(frozen=True)
class IsolatedEvalTrace:
    """Captured metadata for one isolated evaluator LLM invocation."""

    evaluator_tag: str
    evaluator_model: str
    evaluator_family: str
    actor_family: str
    cross_family: bool
    evaluator_thought: str
    evaluator_analysis: str


class IsolatedEvaluatorRunner:
    """Runs LLM-backed evaluators in complete separation from the Actor model's context."""

    def __init__(
        self,
        *,
        vault: CredentialVault | None,
        client: UniversalModelClient | None,
        get_actor_auth: Callable[[], AuthProfile | None],
        get_actor_profile: Callable[[], ModelCapabilityProfile | None],
        thinking_effort: ThinkingEffort = ThinkingEffort.MEDIUM,
        on_trace_event: Callable[[StreamEvent], None] | None = None,
    ) -> None:
        self.vault = vault
        self.client = client
        self._get_actor_auth = get_actor_auth
        self._get_actor_profile = get_actor_profile
        self.thinking_effort = thinking_effort
        self._on_trace_event = on_trace_event
        self._lock = threading.Lock()
        self._failed_profile_names: set[str] = set()
        self.last_trace: IsolatedEvalTrace | None = None

    @staticmethod
    def _is_usable_auth(auth: AuthProfile | None) -> bool:
        if auth is None:
            return False
        return bool(auth.auth_kind != "none" or getattr(auth, "api_mode", "") == "agy_acp")

    def is_live_llm_available(self) -> bool:
        """Returns True iff a real isolated LLM call can be issued without hijacking test doubles."""
        if self.client is None or self.vault is None:
            return False
        transport = getattr(self.client, "transport", None)
        if transport is None:
            return False
        if not (isinstance(transport, HttpProviderTransport) or bool(getattr(transport, "allow_isolated_evaluator", False))):
            return False
        actor_auth = self._get_actor_auth()
        eval_auth, _, _ = self.select_evaluator_target()
        if eval_auth is None:
            eval_auth = actor_auth
        return self._is_usable_auth(eval_auth)

    def select_evaluator_target(
        self,
        actor_family_override: str | None = None,
    ) -> tuple[AuthProfile | None, ModelCapabilityProfile | None, bool]:
        """Selects an independent evaluator profile/model (cross-family preferred)."""
        actor_auth = self._get_actor_auth()
        actor_prof = self._get_actor_profile()
        actor_fam = (
            actor_family_override
            or (actor_prof.family if actor_prof else "")
            or (PROVIDER_FAMILY_MAP.get(actor_auth.provider, actor_auth.provider) if actor_auth else "")
        )
        actor_name = actor_auth.name if actor_auth else None

        chosen_auth: AuthProfile | None = None
        if self.vault is not None and actor_fam:
            cand = self.vault.select_cross_family_profile(actor_fam, exclude_profile=actor_name)
            if cand is not None and cand.name not in self._failed_profile_names:
                chosen_auth = cand
        if chosen_auth is None or not self._is_usable_auth(chosen_auth):
            chosen_auth = actor_auth
        if (chosen_auth is None or not self._is_usable_auth(chosen_auth)) and self.vault is not None:
            cand_active = self.vault.get_active_profile()
            if cand_active is not None and cand_active.name not in self._failed_profile_names:
                chosen_auth = cand_active
        if chosen_auth is None or not self._is_usable_auth(chosen_auth):
            return None, None, False

        env_model = os.environ.get("CRUX_EVAL_MODEL", "").strip()
        if env_model:
            eval_prof = resolve_model_profile(env_model, chosen_auth.provider)
        elif chosen_auth.name == actor_name and actor_prof is not None:
            eval_prof = actor_prof
        else:
            eval_mid = chosen_auth.default_model or (actor_prof.model_id if actor_prof else "gemini-2.5-pro")
            eval_prof = resolve_model_profile(eval_mid, chosen_auth.provider)

        cross_family = bool(actor_fam and eval_prof.family != actor_fam)
        return chosen_auth, eval_prof, cross_family

    def _build_isolated_client(self) -> UniversalModelClient | None:
        if self.client is None or self.vault is None:
            return None
        transport = self.client.transport
        if isinstance(transport, HttpProviderTransport):
            eval_transport: object = HttpProviderTransport(
                self.vault,
                timeout_seconds=min(float(transport.timeout_seconds), 90.0),
                thinking_effort=self.thinking_effort,
            )
            with transport._ca_lock:
                eval_transport._code_assist_projects = dict(transport._code_assist_projects)  # type: ignore[attr-defined]
            return UniversalModelClient(
                self.vault,
                eval_transport,  # type: ignore[arg-type]
                max_api_attempts=2,
                max_output_retries=1,
            )
        if getattr(transport, "allow_isolated_evaluator", False):
            return UniversalModelClient(
                self.vault,
                transport,
                max_api_attempts=2,
                max_output_retries=1,
            )
        return None

    def invoke(
        self,
        *,
        evaluator_tag: str,
        system_prompt: str,
        user_payload: str,
        actor_model_family: str | None = None,
    ) -> str:
        """Executes a single isolated, tool-free, zero-history evaluator model call (including screenshots)."""
        if not self.is_live_llm_available():
            return ""
        eval_auth, eval_prof, is_cross = self.select_evaluator_target(actor_model_family)
        eval_client = self._build_isolated_client()
        if eval_auth is None or eval_prof is None or eval_client is None:
            return ""

        from crux_evals.core.protocol import get_current_eval_images

        eval_images = get_current_eval_images()
        parts: list[CanonicalPart] = [CanonicalPart(kind="text", text=user_payload)]
        for img in eval_images:
            if img.base64_data:
                parts.append(
                    CanonicalPart(
                        kind="image",
                        mime_type=img.mime_type or "image/png",
                        base64_data=img.base64_data,
                        alt_text=img.path,
                    )
                )

        isolated_sid = f"crux_eval_isolated:{evaluator_tag}:{new_id('ev')}"
        user_msg = CanonicalMessage(role="user", parts=parts)
        try:
            # Hardline separation: tools=[], empty history, ZERO Actor streaming callback leaks
            reply = eval_client.generate(
                profile=eval_prof,
                auth_profile=eval_auth,
                system_prompt=system_prompt,
                history=[user_msg],
                tools=[],
                session_id=isolated_sid,
                on_event=None,
            )
        except Exception:
            self._failed_profile_names.add(eval_auth.name)
            actor_auth = self._get_actor_auth()
            actor_prof = self._get_actor_profile()
            if (
                actor_auth is not None
                and actor_auth.name != eval_auth.name
                and self._is_usable_auth(actor_auth)
                and actor_prof is not None
            ):
                eval_auth = actor_auth
                eval_prof = actor_prof
                is_cross = False
                try:
                    reply = eval_client.generate(
                        profile=eval_prof,
                        auth_profile=eval_auth,
                        system_prompt=system_prompt,
                        history=[user_msg],
                        tools=[],
                        session_id=isolated_sid,
                        on_event=None,
                    )
                except Exception:
                    return ""
            else:
                return ""

        raw_text = reply.plain_text().strip()
        thought_text = reply.thinking_text().strip()
        analysis_text = _extract_analysis_summary(raw_text)

        actor_fam = actor_model_family or (
            self._get_actor_profile().family if self._get_actor_profile() else ""
        )
        trace = IsolatedEvalTrace(
            evaluator_tag=evaluator_tag,
            evaluator_model=eval_prof.model_id,
            evaluator_family=eval_prof.family,
            actor_family=actor_fam,
            cross_family=is_cross,
            evaluator_thought=thought_text,
            evaluator_analysis=analysis_text,
        )
        with self._lock:
            self.last_trace = trace

        if self._on_trace_event is not None:
            try:
                mode_lbl = "cross-family" if is_cross else "isolated-session"
                summary_snippet = analysis_text[:180] or f"completed ({len(thought_text)} eval thought chars)"
                self._on_trace_event(
                    StreamEvent(
                        kind="cognitive_event",
                        text=(
                            f"Crux-Evals [{evaluator_tag}] via `{eval_prof.model_id}` ({mode_lbl}): "
                            f"{summary_snippet}"
                        ),
                        meta={
                            "event": "crux_evals_analysis",
                            "evaluator": evaluator_tag,
                            "evaluator_model": eval_prof.model_id,
                            "evaluator_family": eval_prof.family,
                            "actor_family": actor_fam,
                            "cross_family": is_cross,
                            "evaluator_thought": thought_text[:2_400],
                            "evaluator_analysis": analysis_text[:1_200],
                            "screenshots": len(eval_images),
                        },
                    )
                )
            except Exception:
                pass
        return raw_text

    def make_llm_callable(self, evaluator_tag: str = "evaluator") -> Callable[[str, str, str | None], str] | None:
        """Returns a `(system_prompt, user_payload, actor_family) -> str` callable if live LLM is available."""
        if not self.is_live_llm_available():
            return None

        def _call(system_prompt: str, user_payload: str, actor_family: str | None) -> str:
            return self.invoke(
                evaluator_tag=evaluator_tag,
                system_prompt=system_prompt,
                user_payload=user_payload,
                actor_model_family=actor_family,
            )

        return _call


def _extract_analysis_summary(raw_text: str) -> str:
    m = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                for key in ("summary", "analysis", "remediation", "critique"):
                    val = str(obj.get(key) or "").strip()
                    if val:
                        return val
        except (ValueError, json.JSONDecodeError):
            pass
    return " ".join(raw_text.splitlines()[:3]).strip()[:400]
