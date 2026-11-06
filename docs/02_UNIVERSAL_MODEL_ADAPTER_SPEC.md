# `02` — Universal Model Adapter Specification ("Almost Any Model at Peak IQ")

> Companion to [MASTER_IMPLEMENTATION_PLAN.md](../MASTER_IMPLEMENTATION_PLAN.md). Grounded in empirical harness teardowns (`04_ESSENTIAL_HARNESS_FORMAL_SPECIFICATION.md`, `05_CORE_PROBLEM_SOLVING_AND_REASONING_ARCHITECTURE.md`, `07_INNER_REASONING_LOOP_AND_LONG_FORM_EXECUTION_DEEP_DIVE.md`, and `13_THE_FINAL_SEVEN_PRODUCTION_HARNESS_PILLARS.md`).

---

## 1. Why "One Prompt & Wire Format Fits All" Fails

Most coding harnesses are secretly hyper-tuned to *one* vendor's flagship model:
- Put a **smaller/open-weight model** (`Qwen-14B/32B`, `Llama`, `DeepSeek-Distill`, `Devstral`) inside `codex-rs` or `Claude Code` and it drowns in a `15k`-token system prompt, mangles 6 simultaneous parallel JSON tool calls, forgets to run tests before saying *"Fixed!"*, or fails exact-match `edit_file` targets 40% of the time.
- Conversely, put a **Tier-S frontier reasoner** (`Gemini 4 Argon`, `Claude Opus 5.5`, `GPT-5.2`) inside a rigid micro-managed prompt formulated for a 2024 model and you cripple its native latent reasoning and burn `3x` unnecessary turns.
- Worse, in a **multi-model swarm or quota failover**, switching a live session from Provider A to Provider B crashes immediately with `400 Bad Request` on invalid cryptographic thinking signatures or incompatible tool-call IDs.

`USH` solves all three via **(1) a Canonical Internal IR + 6-Rule Wire Normalizer**, **(2) a Declarative `ModelCapabilityProfile` Registry**, and **(3) 3-Tier Adaptive Cognitive Scaffolding**.

---

## 2. Declarative `ModelCapabilityProfile` & The 3 Adaptive Tiers

Every configured endpoint maps (via explicit registry + auto-detected heuristics) to a `ModelCapabilityProfile`:

```json
{
  "model_id": "provider/model-name",
  "family": "gemini | anthropic | openai | qwen | deepseek | glm | kimi | minimax | local_oss",
  "tier": "S | A | B",
  "context_window_tokens": 200000,
  "max_output_tokens": 32768,
  "supports_vision": true,
  "tool_calling_mode": "native_strict | native_loose | xml_fenced_fallback",
  "reasoning_wire_mode": "gemini_thought_sig | anthropic_signed_thinking | openai_encrypted | reasoning_content_field | inline_think_tags | none",
  "prompt_cache_ttl_seconds": 300,
  "max_parallel_read_tools": 8
}
```

### How the Harness Adapts Automatically by Tier

| Harness Mechanism | **Tier S (`Frontier Native Reasoners`)**<br>*Gemini 4 Argon, Claude Opus/Fable, GPT-5.2* | **Tier A (`Fast / Open Frontier Workhorses`)**<br>*Gemini Flash, Sonnet, Qwen3-Coder, DeepSeek-V3/R1, Kimi-K2, GLM-5* | **Tier B (`Compact / Local / Open-Weight 7B–32B`)**<br>*Ollama, vLLM, SGLang, quantized local coders* |
| :--- | :--- | :--- | :--- |
| **System Prompt Footprint** | **Minimalist (`~1,200` tokens)**: Crisp objective invariants + environment facts; trusts latent CoT instead of lecturing. | **Balanced (`~2,000` tokens)**: Adds concise search-before-edit & patch formatting examples. | **Compact Structured (`~1,500` tokens)**: Numbered `OBSERVE -> HYPOTHESIS -> 1 ACTION -> VERIFY` template; zero prose fluff. |
| **Latent vs. Explicit Thinking** | Native high-effort thinking pinned invariant across turn (`cache-safe`). | Native `thinking` / `reasoning_content` circulated verbatim within family. | Injects lightweight `<scratchpad>` (`<=6` lines: *Fact, Hypothesis, Exact Next Tool*) when `reasoning_wire_mode == "none"`. |
| **Parallel Tool Batch Cap** | Up to **`8`** parallel read calls per step + speculative streaming read prefetch. | Up to **`4`** parallel read calls per step. | Strictly **`1` tool call per step** (eliminates multi-JSON syntax corruption by >90%). |
| **Tool Call Parsing & Repair** | Strict native function calling + schema validation. | Native function calling + tolerant trailing-comma/stringified-JSON unwrapping. | Native tools **plus** deterministic `<tool_call>` XML / fenced-JSON fallback parser + 1-shot schema-diff repair feedback. |
| **Context Compaction Trigger** | `85%` of `200k–1M` window; `16 KB` per-tool output preview before disk spill. | `80%` of `128k–256k` window; `10 KB` per-tool preview. | **`70%` of `32k–64k` window**; aggressive `4 KB` middle-out tool previews + immediate stale-read tombstoning. |
| **Completion Verification** | Layer-1 deterministic checks + fast self-verify (`Gear 1`) or cross-family Critic (`Gear 2`). | Mandatory Layer-1 command execution check before `DONE` allowed (`anti-hallucination gate`). | **Strict Harness-Driven Repro & Post-Edit Test Hook**: Harness auto-runs configured lint/test cmd after edits and feeds exact errors back! |
| **Best Role in a Swarm** | Chief Architect, Hard-Bug Island Builder, Blind `BARS` Judge, Red-Team Prosecutor, Crossover Synthesizer. | Sharded Module Builder,Executable Spike Runner, Dedicated Test/Repro Author, Fast Triage. | Zero-cost parallel File/Log Scouts, RepoMap Summarizers, Candidate Patch Pre-Filterers, Offline Dream Consolidators. |

---

## 3. Universal Multi-Provider Wire Translator (`transformMessages`)

All session history is persisted on disk in **one provider-neutral `CanonicalTranscriptRecord` JSONL format**, storing raw reasoning items tagged with `producer_provider`, `producer_api`, and `producer_model_id`. Right before each outbound HTTP stream, `transformMessages(history, target_profile)` enforces **8 lossless cross-model rules**:

1. **Same-Model Cryptographic Signature Fidelity**:
   - When `record.producer_model_id == target_profile.model_id`: replay `thoughtSignature` (Gemini), `{type: "thinking", thinking, signature}` / `redacted_thinking` (Anthropic), `encrypted_content` (OpenAI Responses), or `reasoning_content` (DeepSeek/Kimi/GLM, including `reasoning_content: ""` on empty assistant tool calls) byte-for-byte.
2. **Cross-Model Foreign Thinking Scrubbing**:
   - When switching models (failover, user model switch, or swarm handoff): **drop foreign `thinking` blocks and delete foreign `thoughtSignature` fields** (converting Gemini's missing historical signature via `skip_thought_signature_validator` where required). Never downgrade foreign `<thinking>` into visible assistant text, which poisons downstream models into leaking internal reasoning to the user.
3. **Automatic Signature-400 Fallback Retry**:
   - If an OpenAI-compat proxy or upgraded model checkpoint returns HTTP `400` complaining about an invalid thinking signature, automatically strip historical signatures once and retry transparently.
4. **Universal Tool-Call ID Bijection**:
   - Maintain a deterministic session map normalizing 450-char OpenAI pipe IDs (`call_abc|item_xyz`) into 64-char `^[a-zA-Z0-9_-]{1,64}$` IDs across both `assistant` tool calls and `tool_result` frames so Anthropic/Gemini/Bedrock never reject replayed turns.
5. **Multimodal Downgrade & Error-Image Splitting**:
   - If `!target_profile.supports_vision`, replace image blocks with explicit text placeholders (`[Image omitted: current model lacks vision; switch to vision model or inspect DOM text]`).
   - For Anthropic/Bedrock targets, automatically hoist images out of `is_error: true` tool results into an immediately following user caption block (`splitErrorToolResultImages`).
6. **Aborted/Errored Turn Pruning & System-Message Holdback**:
   - Skip incomplete `stop_reason in {"error", "aborted"}` assistant records on replay; hold back any `<system-reminder>` or blackboard delta arriving mid-tool-batch until all matching `tool_result` records have been emitted.
7. **Stream Integrity Guards (`Empty-200` + `MAX_TOKENS`)**:
   - Treat HTTP `200` streams with `0` text/tool deltas as retryable `EMPTY_200_RESPONSE` errors; treat `finish_reason == MAX_TOKENS` with partial/truncated tool JSON by discarding the broken tool call and auto-continuing generation with an explicit truncation notice.
8. **Prefix KV-Cache Preservation & Econometric Cache Warmer**:
   - Sort all JSON Schema keys deterministically (`canonical_json`); never mutate Turn-0 system prompts mid-session (append mode/skill/blackboard deltas at tail wrapped in `<dynamic_instructions>` stripped from older turns); compute FNV-1a `session_affinity_hash` for sticky replica routing; and run the econometric TTL cache warmer ($\mathbb{E}[\text{Savings}] = P(\text{cont}) \cdot C_{\text{miss}} - C_{\text{warm}} \ge \$0.05$).

---

## 4. Complete Supported Model Matrix & Multi-Account Authentication (`ush auth`)

### 4.1 Five Native Wire Adapters $\to$ 300+ Cloud, Cloud-Enterprise & Local Models
Every model family can be used solo (`/model`) **or mixed simultaneously inside one Swarm** (e.g., `gcloud` Vertex Gemini + Vertex Claude + local `vLLM` scouts):

| Adapter | Supported Models | Official Supported Login / Auth Flows |
| :--- | :--- | :--- |
| **`google-genai` & `vertex-ai`** | **Gemini 4 / 3.x Pro, Flash, Flash-Lite** + **Vertex Model Garden** (**Claude Opus/Sonnet**, Llama, Mistral, Qwen, DeepSeek on GCP) | 1. **Zero-Config GCP ADC** (`gcloud auth application-default login` / GCE & GKE workload service accounts)<br>2. **Browser OAuth 2.0 + PKCE** (`ush auth login google`, headless copy-URL / SSH fallback)<br>3. `GEMINI_API_KEY` (AI Studio) |
| **`anthropic` (+ Vertex / Bedrock)** | **Claude Opus, Sonnet, Haiku** | 1. `ANTHROPIC_API_KEY` (interactive masked paste via `ush auth login anthropic`)<br>2. **GCP Vertex AI ADC** (`vertex-anthropic:@project/region` — zero extra key if already logged into `gcloud`!)<br>3. **AWS Bedrock SSO / IAM** (`aws sso login`) |
| **`openai-responses` (+ Azure)** | **GPT-5.x, o3 / o4-mini, Codex models** | 1. `OPENAI_API_KEY` / **Official OAuth PKCE + Headless SSH Device-Code Flow** (`--device-code`, matching `codex-rs` `device_code_auth.rs`)<br>2. **Azure Entra ID (`az login`) / Azure API Key** |
| **`openai-compat-cloud`** | **OpenRouter (`300+` models with 1 key)**, **Qwen3-Coder**, **DeepSeek-V3/R1**, **Kimi-K2**, **GLM-5**, **MiniMax**, **Groq / Together / Fireworks / xAI** | Per-provider API key or OAuth Device Flow (`ush auth login openrouter\|qwen\|deepseek\|kimi`) + multi-account quota failover pools (`work`, `research`, `backup`) |
| **`local-inference`** | **Ollama, vLLM, SGLang, llama.cpp, LM Studio** (`7B`–`671B` open weights, local Mac Metal or remote GPU clusters) | **Zero login required** on `localhost` (auto-discovers `/v1/models` on startup!); optional Bearer token or mTLS client cert for remote/STIG GPU endpoints |

### 4.2 Hybrid Credential Vault & Multi-Profile Quota Rotation
Modeled on Gemini CLI's `OAuthCredentialStorage` (`oauth-credential-storage.ts`) and `oauth2.ts`:
1. **OS Keychain Primary + `0600` Headless Fallback**: On macOS/Desktop Linux/Windows, tokens and API keys store inside the native OS Keychain (`macOS Keychain` / `libsecret`); on headless SSH servers & STIG containers, `USH` falls back to `~/.ush/credentials.vault.json` (`chmod 0600`)—**never** committed to repos and **always** stripped from child `exec_pty` environments.
2. **First-Run Onboarding Wizard (`ush` / `/auth`)**: On first launch, if `gcloud` ADC or `Ollama`/`vLLM` or standard env vars are present, `ush` detects them automatically in `<100ms` so you can code immediately with zero prompts; otherwise `/auth login` presents a clean interactive picker (Browser OAuth PKCE, SSH Device/Copy-URL flow, GCP Vertex Project+Region bind, or masked API key entry).
3. **Named Profiles & Auto-Failover (`auth1:model,auth2:model`)**: Bind multiple accounts/projects concurrently (e.g., `prod-vertex`, `backup-vertex`, `cluster-vllm`, `anthropic-direct`), auto-refresh OAuth tokens 5 minutes before expiry, and automatically fail over on HTTP `429` quota exhaustion without dropping the turn (via `transformMessages`).
