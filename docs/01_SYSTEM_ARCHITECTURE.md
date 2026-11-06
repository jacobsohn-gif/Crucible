# `01` — Core System Architecture (`Kernel`, `Tools`, `PTY`, `DAG Context` & `Security`)

> Companion to [MASTER_IMPLEMENTATION_PLAN.md](../MASTER_IMPLEMENTATION_PLAN.md). Grounded in empirical harness teardowns (`04_ESSENTIAL_HARNESS_FORMAL_SPECIFICATION.md`, `07_INNER_REASONING_LOOP_AND_LONG_FORM_EXECUTION_DEEP_DIVE.md`, `09_THE_EIGHT_SILENT_MULTIPLIERS_OF_ELITE_HARNESSES.md`, and `13_THE_FINAL_SEVEN_PRODUCTION_HARNESS_PILLARS.md`).

---

## 1. Total 6-Phase Session Actor & Concurrency Model

Every harness worker runs a deterministic, single-writer 6-phase state machine per `thread_id` guarded by a monotonic `writer_lease_epoch`:

`PRE_INVOCATION` $\to$ `GENERATE` $\to$ `TOOL_DISPATCH` $\to$ `POST_INVOCATION` $\to$ `STOP_HOOKS` $\to$ `DONE`

### Core Execution Invariants
1. **`RwLock` Read-Parallel / Write-Exclusive Tool Gate**:
   - Pure read tools (`read_file`, `grep_files`, `glob_files`, `lsp_query`) acquire a **shared reader lock** and execute concurrently (capped by the model's `max_parallel_tools` profile).
   - Mutating or stateful tools (`edit_file`, `write_file`, `exec_pty`, `write_stdin`) acquire an **exclusive writer lock**; multi-file patches canonicalize and **lexicographically sort** target paths before acquiring per-file locks so concurrent workers can never deadlock.
2. **Drop-Safe Synthetic Orphan Repair**:
   - Every dispatched `tool_call.id` is registered in a pending frame ledger *before* execution starts. If a tool times out, throws, or is interrupted by user cancel, the kernel guarantees a synthetic error `tool_result` (`TOOL_TIMEOUT` / `ABORTED_BY_USER`) is written before closing the batch—making dangling-tool-call `400` replay errors structurally impossible.
3. **Protocol-Safe `TurnSteerQueue`**:
   - Live human messages (text + images) arriving mid-turn enter `pending_steer_queue` (`turn_steer_queued`) without killing in-flight tools, and drain strictly at `POST_INVOCATION` *after* all `tool_result` records close and *before* the next `GENERATE` step.
4. **5-Condition Quiescence Barrier**:
   - A turn completes iff `planner_idle && blocking_bg_ptys == 0 && active_children == 0 && !compaction_committing && steer_queue.is_empty()`. Processes spawned with `is_daemon = true` (dev servers, watchers) never block turn settlement.

---

## 2. The "Essential 11" Tool Surface (Static Dispatch, Zero `eval()`)

All tools register in a static typed `HashMap<ToolName, ToolHandler>` validated against JSON Schema *before* execution—small enough (`~2.7k` schema tokens) for 14B local models to master without tool confusion, yet rich enough for 1M-LoC production monorepos and 500+ on-demand domain skills ([05_CLI_SKILLS_COMMANDS_HOOKS_AND_EXTENSIONS_SPEC.md](05_CLI_SKILLS_COMMANDS_HOOKS_AND_EXTENSIONS_SPEC.md)):

| # | Tool | Lock | Key Forgiving / High-Leverage Mechanics |
| :-: | :--- | :---: | :--- |
| 1 | `read_file` | Read | Line-numbered windows (`start_line`, `end_line`), binary/image multimodal auto-detection, records `last_read_sha256[path]`. |
| 2 | `edit_file` | Write | **4-Pass Fuzzy Seeker** (exact $\to$ `trim_end` $\to$ `trim` $\to$ Unicode quote/dash/space fold), CRLF-preserving LCS line endings, UTF-8 BOM preservation, **SHA-256 stale-read check** (returns external diff on mismatch). |
| 3 | `write_file` | Write | Atomic temp-file + `fsync` + `rename`; requires prior `read_file` if overwriting an existing tracked file. |
| 4 | `grep_files` | Read | Fast `ripgrep`-backed regex/literal search honoring `.gitignore` + `.ushignore`, bounded results sorted by recent modification / path proximity. |
| 5 | `glob_files` | Read | Fast file-tree & path discovery + optional `--skeleton` mode returning Tree-sitter top-level signatures. |
| 6 | `lsp_query` | Read | **Lazy read-side only**: `definition`, `references`, `hover`, `incoming_calls`, `outgoing_calls` (zero eager post-edit blocking; 1,000-char bounded error detail). |
| 7 | `exec_pty` | Write | Executes inside a **Persistent Stateful PTY** (`cwd`/`venv` preserved), returns early with `pty_id` on `is_daemon=true`, soft wait timeout, or detected interactive prompt. |
| 8 | `write_stdin` | Write | Sends keystrokes/input to or polls output/kills an active `pty_id` (enables REPLs, interactive CLIs, `tuistory`-style TUI driving). |
| 9 | `ask_user` | Write | Structured 1–4 multiple-choice clarification card; crash-resumable on session reload; disabled automatically in headless autonomous `/goal` islands. |
| 10 | `swarm_dispatch` | R/W | Posts structured events (`SPIKE`, `SHARD`, `CONTRACT_DELTA`, `REQUEST_REVIEW`) to the Stigmergic Blackboard or spawns bounded sub-workers (`flash`/local scouts). |
| 11 | `activate_skill` | Read+Grant | JSON Schema `enum`-constrained progressive skill loader: confirms external resource tree, dynamically grants read-sandbox access to `dirname(skill.path)`, injects `<activated_skill>`, or forks an isolated subagent when `context: fork`. |

*(Optional lazy extension channel: host-trusted `mcp_call` schemas loaded via tail `<system-reminder>` on demand so Turn-0 prefix caches stay untouched.)*

---

## 3. Persistent Stateful PTY & Anti-Hang Execution Engine

1. **Startup Environment Normalization**:
   - Probe `$SHELL -l -c /usr/bin/env` once (`5s` cap) to capture missing toolchain `PATH` entries (`cargo`, `uv`, `nvm`, `homebrew`).
   - Inject non-interactive guards: `GIT_TERMINAL_PROMPT=0`, `GIT_EDITOR=true`, `PAGER=cat`, `CI=true`, `DEBIAN_FRONTEND=noninteractive`.
2. **Dual-Mode Completion & Stall Detection**:
   - **Primary (`OSC 633` Shell Integration)**: Stateful cross-chunk parser tracks `633;C` (start), `633;D;<exit_code>` (finish), and `633;P;Cwd=<dir>` (`64 KB` max pending sequence cap).
   - **Fallback (`Markerless Prompt Heuristics`)**: For nested `ssh`/Docker shells lacking `OSC 633`, evaluate carriage-return-resolved cursor line: yield after `400ms` idle on **`strong` prompts** (`$`, `#`, `❯`, `>>>`, `[y/N]`, `Password:`) and after `5,000ms` quiet window on **`weak` prompts** (`>`, `%`).
3. **Bounded `1 MiB` Middle-Out Output + Disk Spillover**:
   - Capture streams into a `HeadTailBuffer` keeping top header lines + bottom assertion tracebacks; whenever output exceeds `16 KB`, spill the full raw log to `<scratch_dir>/tool_outputs/<call_id>.log` and cite its path + original line/token counts.

---

## 4. Out-of-Tree Shadow Git + DAG Context Architecture

1. **Shadow Git (`~/.ush/snapshots/<repo_hash>/`)**:
   - Bare Git repo outside the workspace pointing `--work-tree` at the active directory. Captures lightweight Git tree hashes concurrently before mutating turns, enabling instant 0-risk per-step undo, clean diffs for Judges, and fork points for Gear-3/4 swarm upshifts without touching the user's real `.git/index`.
2. **3-Stage Memory & Context Hygiene**:
   - **Every Turn (Zero-LLM Hygiene)**: Replace older duplicate full `read_file(path)` outputs with 1-line tombstones (`[Superseded by step #N]`); erase `<dynamic_instructions>` from historical user messages (`i < last_user`).
   - **At `85%` Context Watermark (`AsyncCompactor`)**: Snapshot `(messages, history_version)` off the hot path and generate an **Epistemic State Handoff** storing: *(1) Exact Objective & Constraints, (2) Verified Facts & File:Line Map, (3) Falsified Hypotheses (DO NOT REPEAT), (4) Active Failing Assertion, (5) Next Concrete Step* + pointer to raw JSONL transcript.
   - **On Dead-End Rewind (`Session DAG` + `LCA` Summarizer)**: Conversations store `parent_entry_id` pointers forming a tree. Jumping from dead-end `leaf_A` back to `node_k` walks to `LCA(leaf_A, node_k)` and injects a compact `<branch_lessons>` summary (`touched_files` + why `leaf_A` failed).

---

## 5. Three-Layer Security & Workspace Trust Boundary

1. **Child-Process Environment Scrubbing**: Strip internal harness tokens plus case-insensitive `*KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*CREDENTIAL*` patterns before every `exec_pty` spawn (unless explicitly allowlisted in host operator config).
2. **Live `.ushignore` / `.clineignore` Context Firewall**: Block `read_file`, `grep_files`, and `glob_files` from exposing matched `.env*`, `*.pem`, `id_*`, or proprietary paths to any LLM prompt; additionally scrub high-entropy bearer/API key regexes from PTY stdout before context ingestion.
3. **Repo Trust Quarantine & Untrusted Data Framing**: Never auto-execute repo-local `.mcp.json`, git hooks (`core.hooksPath=/dev/null` on untrusted clones), or project scripts until the exact config SHA-256 is approved in host storage; wrap external web/issue payloads in `<untrusted_external_data>` tags and enforce structural argv command policies (`execpolicy`) against destructive/exfiltration commands.
