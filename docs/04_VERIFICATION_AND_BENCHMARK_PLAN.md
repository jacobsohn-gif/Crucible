# `04` — Conformance, Chaos & 6-KPI Trajectory Benchmark Plan

> Companion to [MASTER_IMPLEMENTATION_PLAN.md](../MASTER_IMPLEMENTATION_PLAN.md). Grounded in empirical harness teardowns (`04_ESSENTIAL_HARNESS_FORMAL_SPECIFICATION.md`, `09_THE_EIGHT_SILENT_MULTIPLIERS_OF_ELITE_HARNESSES.md`, and `14_CLI_ARCHITECTURE_SKILLS_SLASH_COMMANDS_AND_EXTENSIBILITY.md`).

---

## 1. Automated Conformance & Chaos Test Suite (`CT-01` – `CT-20`)

Every milestone PR must pass deterministic mock-provider + real-filesystem conformance gates in `<30s` with zero external API flakiness:

| ID | Milestone | Invariant Tested | Pass Criterion |
| :--- | :---: | :--- | :--- |
| `CT-01` | `M0` | **Cross-Provider Mid-Session Rotation** | Replay 12-turn multimodal + tool transcript across Gemini $\to$ Anthropic $\to$ OpenAI $\to$ Local mock wire validators with `0` schema/signature/ID errors. |
| `CT-02` | `M0` | **`EMPTY_200_RESPONSE` & `MAX_TOKENS` Truncated-Tool Recovery** | Truncated tool JSON discarded (never executed); stream auto-continues up to `max_output_retries`. |
| `CT-03` | `M0` | **Prefix KV-Cache Byte Stability** | Steps `1..20` keep byte-identical serialized prefix tokens for all historical items `0..k-1`; `<dynamic_instructions>` appears solely on final turn. |
| `CT-04` | `M0` | **Tier-B Malformed Tool-Call Repair** | Simulated 14B model emitting fenced XML/trailing-comma JSON recovers within 1 repair cycle or fallback parser with `0` loop crashes. |
| `CT-05` | `M1` | **4-Pass Fuzzy Patch & Line-Ending Torture** | Smart quotes, Unicode em-dashes, drifted indentation, UTF-8 BOM, and mixed CRLF/LF files patch cleanly with exact untouched-line bytes preserved. |
| `CT-06` | `M1` | **Deadlock-Free Concurrent File Locks + Stale-Read Guard** | Two parallel patches on `{a.py, b.py}` vs `{b.py, a.py}` never deadlock; external edit between `read_file` and `edit_file` returns unified diff without clobbering. |
| `CT-07` | `M1` | **Persistent PTY `OSC 633`, Interactive Prompt & Flood Safety** | `cd` + `venv` persist across calls; `[y/N]` prompt yields within `<1.5s` via heuristic; `500 MB` stdout flood caps cleanly at `1 MiB` ring + disk spillover; `is_daemon` server doesn't block turn completion. |
| `CT-08` | `M1` | **Drop-Safe Cancellation & Orphan Repair** | Mid-batch `cancel` kills child process group (`os.killpg`) and injects synthetic `ABORTED` `tool_result` for every open `tool_call.id`. |
| `CT-09` | `M1` | **Shadow Git Zero-Pollution Restore** | Invasive 5-file edit + `restore()` leaves real `.git/status` and `.git/index` mtime/hash 100% untouched. |
| `CT-10` | `M2` | **Popperian Loop-Breaker & `0.85` Async Compaction** | 3 identical failing edits trigger falsification nudge; `0.85` background compaction commits atomically only when `history_version` matches. |
| `CT-11` | `M2` | **DAG Session Rewind with `LCA` Branch Summary** | Rewinding 10 dead-end steps restores clean ancestor context *plus* injected `<branch_lessons>` listing falsified cause and `modified_files`. |
| `CT-12` | `M3` | **3-Layer Security Firewall** | `AWS_SECRET_ACCESS_KEY` absent from child `env`; `.ushignore` blocks `read_file`/`grep_files` on `.env`; unapproved repo `.mcp.json` never spawned. |
| `CT-13` | `M3` | **Mid-Turn `TurnSteerQueue` Boundary Safety** | Steer injected while 3 tools run never splits `[assistant(tool_call), tool_result]`; drains immediately before next `GENERATE`. |
| `CT-14` | `M4` | **Skills Trust Gate, `paths:` Globs, BM25 `<skills_delta>` & Sandbox Grant** | Untrusted workspace `.ush/skills` ignored; `paths:` skill surfaces only after matching file read via tail `<skills_delta>` (`0` Turn-0 prefix drift); `activate_skill` unlocks `dirname(skill.path)/references/` read access; colon-containing frontmatter parses cleanly. |
| `CT-15` | `M4` | **4-Loader Slash Commands, `!{}` Injection Shield & `/btw` Cache Fork** | Repo `.toml` command `/help` cannot shadow built-in `/help`; `/greet $(touch pwned)` inside `!{echo {{args}}}` shell-escapes safely and honors `PolicyEngine`; `/btw` fires while tools run with `tools=[]` and `0` mutations to parent session DAG. |
| `CT-16` | `M4` | **9-Event Hooks & Privilege-Clamped Extension `policies.toml`** | `BeforeToolSelection` slims tool schemas dynamically; `Stop` hook `exit 2` re-opens turn; Extension `policies.toml` with `decision = "allow"`/`"yolo"` is hard-clamped and rejected while `deny`/`ask_user` + `sudo`/`&&` unwrapping enforce cleanly. |
| `CT-17` | `M4` | **4-Scope + JIT Subdirectory Memory & `.inbox/*.patch` Redaction** | Symlinked `AGENTS.md`/`GEMINI.md` inode-deduplicated; reading `packages/foo/a.ts` JIT-loads `packages/foo/AGENTS.md` once; `/learn` redacts seeded bearer tokens and stages `.inbox/*.patch` without direct file mutation. |
| `CT-18` | `M5` | **`clean_patch` Anti-Goodhart & Position-Swap Judge** | Candidate patch that deletes a failing assertion is caught 100% of the time in Stage A; swapped $(A,B)$/$(B,A)$ evaluator neutralizes slot bias. |
| `CT-19` | `M6` | **Live Gear 2 $\to$ Gear 3 Stall Upshift & `CONTRACT_DELTA`** | Simulated thrashing Builder auto-forks into parallel worktrees; shared signature edit in Worktree 1 propagates `CONTRACT_DELTA` to Worktree 2 and merges cleanly via `git merge-tree`. |
| `CT-20` | `M6` | **`INV-SWARM-1` Cross-Family Quorum Ledger** | Self-review (`judge_id in contributor_ids`) or tampered SHA-256 ledger hash is deterministically rejected before merge. |

---

## 2. The 6-KPI Live Model & Swarm Benchmark Suite (`ush bench`)

Every prompt, tool-schema, compactor, or swarm-routing change is evaluated across **30 curated repository tasks** (10 routine Gear-1/2 bugs, 10 multi-module refactors, 5 `<25%` pass-rate algorithmic bugs, and 5 fuzzy non-deterministic architecture/UI tasks) across **Tier S**, **Tier A**, and **Tier B** models, tracking six telemetry KPIs computed directly from JSONL rollout logs:

1. **End-to-End Verified Solve Rate (`%`)** — Evaluated against held-out tests + cross-family blind rubric (`Tier B` solo lift target: **`+15–25 pp`** over naive ReAct loop).
2. **`edit_file` Mechanical Failure Rate (`%`)** — `failed_edits / total_edits` (target: **`<2.0%`** on Tier S/A, **`<5.0%`** on Tier B).
3. **Prefix KV-Cache Hit Rate (`%`)** — `cached_input_tokens / total_input_tokens` (target: **`>90%`** on multi-step runs).
4. **Thrash & Superseded Re-Read Ratio** — Fraction of turns tripping Popperian streak guards or re-reading unchanged files.
5. **False-Victory Catch & Conversion Rate (`%`)** — Premature completion claims intercepted by Stage A/B Judges and converted into verified passes.
6. **Pareto Cost-Latency Curve (`$/solved task` & `p50/p95 wall time`)** — Proving that the **4-Gear Elastic Swarm** achieves Gear-1/2 speed/cost on the easy 75% while unlocking Gear-4 swarm solves on the hardest 25%.

---

## 3. Security Verification Plan

- **Input Validation, Path Traversal & Sandbox Boundary Enforcement**: Every file/skill/macro path resolves via `Path.resolve()` and checks `resolved.is_relative_to(allowed_root)` (preventing `../` traversal and partial-prefix `/repo-evil` bypasses). SQLite blackboard queries exclusively use `?` parameterized statements (`0` SQL string concatenation).
- **Secrets Hygiene & Auth Vault Hardening**: Zero hardcoded API keys or fallback secrets anywhere in code; credentials store in OS Keychain (`macOS security`) or `0600` vault files; all `*KEY*`/`*SECRET*`/`*TOKEN*`/`*PASSWORD*`/`*CREDENTIAL*` env vars are scrubbed from child PTYs; high-entropy tokens are redacted from logs and `/learn` memory patches; HTTPS enforces `TLS >= 1.2`.
- **Command Injection & Localhost Binding**: Slash-command `!{...}` interpolations escape arguments via `shlex.quote()` and validate through the 4-tier `PolicyEngine` (where Extension rules are hard-clamped against `allow`/`yolo`); repo-local hooks/skills/MCP require explicit SHA-256 workspace trust; all local OAuth callback and `ush serve --acp|--jsonrpc` servers bind strictly to `127.0.0.1` (never `0.0.0.0`).
- **Automated Security Verification**: Run `CT-12`, `CT-14`, `CT-15`, `CT-16`, `CT-17`, and `CT-20` plus static secret/AST checks across all newly created source files before completion.
