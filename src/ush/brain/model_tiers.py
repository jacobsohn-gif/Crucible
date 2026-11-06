"""Model Capability Registry & 3-Tier Adaptive Cognitive Scaffolding (`Tier S / A / B`)."""

from __future__ import annotations

from ush.core.types import ModelCapabilityProfile, ModelTier, PermissionMode


BUILTIN_PROFILES: dict[str, ModelCapabilityProfile] = {
    # --- Google Gemini / Vertex AI ---
    "gemini-2.5-pro": ModelCapabilityProfile(
        model_id="gemini-2.5-pro",
        provider="vertex",
        family="gemini",
        tier=ModelTier.S,
        context_window_tokens=1_000_000,
        max_output_tokens=65_536,
        supports_vision=True,
        reasoning_wire_mode="gemini_thought_sig",
        prompt_cache_ttl_seconds=300,
        max_parallel_read_tools=8,
        skill_catalog_budget_tokens=3_000,
    ),
    "gemini-2.5-flash": ModelCapabilityProfile(
        model_id="gemini-2.5-flash",
        provider="vertex",
        family="gemini",
        tier=ModelTier.A,
        context_window_tokens=1_000_000,
        max_output_tokens=32_768,
        supports_vision=True,
        reasoning_wire_mode="gemini_thought_sig",
        max_parallel_read_tools=4,
        compaction_ratio=0.80,
        tool_preview_bytes=10_240,
        skill_catalog_budget_tokens=1_500,
    ),
    # --- Anthropic Direct & Vertex Model Garden ---
    "claude-opus-4-20250514": ModelCapabilityProfile(
        model_id="claude-opus-4-20250514",
        provider="anthropic",
        family="anthropic",
        tier=ModelTier.S,
        context_window_tokens=200_000,
        max_output_tokens=32_000,
        supports_vision=True,
        reasoning_wire_mode="anthropic_signed_thinking",
        max_parallel_read_tools=8,
        skill_catalog_budget_tokens=3_000,
    ),
    "claude-sonnet-4@20250514": ModelCapabilityProfile(
        model_id="claude-sonnet-4@20250514",
        provider="vertex-anthropic",
        family="anthropic",
        tier=ModelTier.A,
        context_window_tokens=200_000,
        max_output_tokens=16_384,
        supports_vision=True,
        reasoning_wire_mode="anthropic_signed_thinking",
        max_parallel_read_tools=4,
        compaction_ratio=0.82,
        tool_preview_bytes=12_288,
        skill_catalog_budget_tokens=2_000,
    ),
    # --- OpenAI ---
    "gpt-5": ModelCapabilityProfile(
        model_id="gpt-5",
        provider="openai",
        family="openai",
        tier=ModelTier.S,
        context_window_tokens=400_000,
        max_output_tokens=32_768,
        supports_vision=True,
        reasoning_wire_mode="openai_encrypted",
        max_parallel_read_tools=8,
        skill_catalog_budget_tokens=3_000,
    ),
}


def resolve_model_profile(model_id: str, default_provider: str = "vertex") -> ModelCapabilityProfile:
    """Resolves exact or heuristic capability tier for any frontier, open-cloud, or local model."""
    provider = default_provider
    clean_id = model_id.strip()
    if ":" in clean_id and clean_id.split(":", 1)[0] in (
        "google",
        "vertex",
        "vertex-anthropic",
        "anthropic",
        "openai",
        "openrouter",
        "local",
    ):
        provider, clean_id = clean_id.split(":", 1)

    if clean_id in BUILTIN_PROFILES:
        base = BUILTIN_PROFILES[clean_id]
        if provider == base.provider:
            return base
        return ModelCapabilityProfile(
            model_id=clean_id,
            provider=provider,
            family=base.family,
            tier=base.tier,
            context_window_tokens=base.context_window_tokens,
            max_output_tokens=base.max_output_tokens,
            supports_vision=base.supports_vision,
            tool_calling_mode=base.tool_calling_mode,
            reasoning_wire_mode=base.reasoning_wire_mode,
            max_parallel_read_tools=base.max_parallel_read_tools,
            compaction_ratio=base.compaction_ratio,
            tool_preview_bytes=base.tool_preview_bytes,
            skill_catalog_budget_tokens=base.skill_catalog_budget_tokens,
        )

    low = clean_id.lower()
    # Family classification
    if "gemini" in low or "argon" in low:
        family = "gemini"
        tier = ModelTier.A if ("flash" in low or "lite" in low) else ModelTier.S
        return ModelCapabilityProfile(
            model_id=clean_id,
            provider=provider if provider in ("google", "vertex") else "google",
            family=family,
            tier=tier,
            context_window_tokens=1_048_576,
            max_output_tokens=65_536,
            supports_vision=True,
            reasoning_wire_mode="gemini_thought_sig",
            max_parallel_read_tools=8 if tier == ModelTier.S else 4,
            compaction_ratio=0.85 if tier == ModelTier.S else 0.80,
            skill_catalog_budget_tokens=3_000 if tier == ModelTier.S else 1_500,
        )
    if "claude" in low or "opus" in low or "sonnet" in low or "haiku" in low or "fable" in low:
        tier = ModelTier.S if ("opus" in low or "fable" in low) else ModelTier.A
        return ModelCapabilityProfile(
            model_id=clean_id,
            provider=provider if provider in ("anthropic", "vertex-anthropic", "openrouter") else "anthropic",
            family="anthropic",
            tier=tier,
            context_window_tokens=200_000,
            max_output_tokens=32_000,
            supports_vision=True,
            reasoning_wire_mode="anthropic_signed_thinking",
            max_parallel_read_tools=8 if tier == ModelTier.S else 4,
            skill_catalog_budget_tokens=3_000 if tier == ModelTier.S else 1_800,
        )
    if any(k in low for k in ("gpt-5", "gpt-4", "o3", "o4", "codex")):
        tier = ModelTier.S if any(k in low for k in ("gpt-5", "o3", "pro")) and "mini" not in low else ModelTier.A
        return ModelCapabilityProfile(
            model_id=clean_id,
            provider=provider if provider in ("openai", "openrouter") else "openai",
            family="openai",
            tier=tier,
            context_window_tokens=200_000,
            max_output_tokens=32_000,
            supports_vision=True,
            reasoning_wire_mode="openai_encrypted",
            max_parallel_read_tools=8 if tier == ModelTier.S else 4,
            skill_catalog_budget_tokens=3_000 if tier == ModelTier.S else 1_500,
        )
    if any(k in low for k in ("deepseek", "qwen", "kimi", "glm", "minimax")) and provider != "local":
        fam = next(k for k in ("deepseek", "qwen", "kimi", "glm", "minimax") if k in low)
        has_r = any(k in low for k in ("r1", "reasoner", "thinking", "qwq", "k2"))
        is_small = any(sz in low for sz in ("-1.5b", "-3b", "-7b", "-8b", "-14b", ":7b", ":14b"))
        tier = ModelTier.B if is_small else ModelTier.A
        return ModelCapabilityProfile(
            model_id=clean_id,
            provider=provider if provider in ("openrouter", "local") else "openrouter",
            family=fam,
            tier=tier,
            context_window_tokens=128_000 if tier == ModelTier.A else 32_768,
            max_output_tokens=16_384 if tier == ModelTier.A else 8_192,
            supports_vision=("vl" in low),
            tool_calling_mode="native_loose" if tier == ModelTier.A else "xml_fenced_fallback",
            reasoning_wire_mode="reasoning_content_field" if has_r else "none",
            max_parallel_read_tools=4 if tier == ModelTier.A else 1,
            compaction_ratio=0.80 if tier == ModelTier.A else 0.70,
            tool_preview_bytes=10_240 if tier == ModelTier.A else 4_096,
            skill_catalog_budget_tokens=1_500 if tier == ModelTier.A else 600,
        )

    # Default: Local / Open-Weight Tier-B adaptive profile (forgiving XML/JSON, 1-tool cap, 70% compact)
    return ModelCapabilityProfile(
        model_id=clean_id,
        provider=provider,
        family="local_oss",
        tier=ModelTier.B,
        context_window_tokens=32_768,
        max_output_tokens=8_192,
        supports_vision=("vision" in low or "-vl" in low or "llava" in low),
        tool_calling_mode="xml_fenced_fallback",
        reasoning_wire_mode="inline_think_tags" if ("r1" in low or "qwq" in low) else "none",
        max_parallel_read_tools=1,
        compaction_ratio=0.70,
        tool_preview_bytes=4_096,
        skill_catalog_budget_tokens=600,
    )


def build_adaptive_system_prompt(
    profile: ModelCapabilityProfile,
    workspace_root: str,
    mode: PermissionMode,
    memory_block: str = "",
    initial_skills_block: str = "",
) -> str:
    """Constructs a cache-invariant Turn-0 system prompt scaled to Tier S / A / B."""
    mode_note = (
        "MODE: PLAN (Read-Only). Inspect code, clarify edge cases, and synthesize a concrete implementation plan. "
        "Do not modify files until switched out of plan mode."
        if mode == PermissionMode.PLAN
        else "MODE: EXECUTE. Reproduce issues before editing, make surgical changes, and verify via tests/commands before declaring completion."
    )

    if profile.tier == ModelTier.S:
        core = (
            f"You are Crucible (crux), a rigorous software engineering and scientific problem-solving agent.\n"
            f"Workspace root: {workspace_root}\n"
            f"{mode_note}\n"
            "Invariants:\n"
            "1. Read target files before editing; preserve unrelated comments and formatting.\n"
            "2. Verify hypotheses empirically with `exec_pty`/`grep_files` before concluding.\n"
            "3. Never weaken or delete test assertions to make a suite pass; never leak secrets.\n"
            f"4. Batch independent read tools in parallel (up to {profile.max_parallel_read_tools})."
        )
    elif profile.tier == ModelTier.A:
        core = (
            f"You are Crucible (crux), an autonomous software engineering agent operating in `{workspace_root}`.\n"
            f"{mode_note}\n"
            "Workflow:\n"
            "1. ORIENT: Use `glob_files` / `grep_files` / `read_file` before making edits.\n"
            "2. REPRODUCE: Run failing test or reproduction command in `exec_pty` first when fixing bugs.\n"
            "3. EDIT: Use `edit_file` with exact surrounding lines copied from `read_file`.\n"
            "4. VERIFY: Re-run tests/lints and read full assertion output before stopping.\n"
            f"5. You may issue up to {profile.max_parallel_read_tools} parallel read-only tool calls per step."
        )
    else:
        # Tier B (7B-32B local/open-weight): compact, rigid single-step contract + XML fallback format
        core = (
            f"You are Crucible (crux) coding agent in `{workspace_root}`.\n"
            f"{mode_note}\n"
            "STRICT STEP PROTOCOL (1 tool call per turn):\n"
            "- Briefly write `<scratchpad>Fact: ... | Hypothesis: ... | Next: tool_name</scratchpad>` (max 4 lines).\n"
            "- Call AT MOST ONE tool per response. Always `read_file` before `edit_file`.\n"
            "- After editing code, ALWAYS run tests via `exec_pty` before saying done.\n"
            "- Prefer native tool calls; if unavailable, emit `<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>`."
        )

    sections = [core]
    if memory_block.strip():
        sections.append(f"<workspace_memory>\n{memory_block.strip()}\n</workspace_memory>")
    if initial_skills_block.strip():
        sections.append(initial_skills_block.strip())
    return "\n\n".join(sections)
