# Crucible (`crux` `v0.1`)

[![Version](https://img.shields.io/badge/version-v0.1-00b4d8?style=flat-square)](https://github.com/jacobsohn-gif/Crucible/releases/tag/v0.1)
[![Dependencies](https://img.shields.io/badge/PyPI_dependencies-0_(stdlib_only)-2ec4b6?style=flat-square)](pyproject.toml)
[![Distributions](https://img.shields.io/badge/distributions-Source_%2B_Compiled_(Mach--O_%26_.pyz)-ff9f1c?style=flat-square)](releases/v0.1/)
[![Conformance](https://img.shields.io/badge/conformance_gates-23%2F23_passing_(CT--01..CT--23)-38b000?style=flat-square)](tests/test_conformance_ct01_to_ct20.py)
[![License](https://img.shields.io/badge/license-Apache--2.0-6c757d?style=flat-square)](pyproject.toml)

**Crucible (`crux`)** is a standalone, zero-third-party-dependency **interactive coding & 4-Gear Elastic Swarm CLI** with **uncompacted long-form thought tracing**, distilled from empirical architectural teardowns of **24 global coding-agent codebases**.

Every hypothesis is tested in the fire: `crux` pairs a **Popperian falsification loop** (reproduce $\to$ hypothesize $\to$ patch $\to$ verify) with a scrollback-native **Visual Inline-Dock TUI**, lossless **mid-session cross-provider model rotation**, an **out-of-tree Shadow Git time machine**, and a **4-Gear Stigmergic Swarm** governed by a **3-Stage Blind Cascading Judge**.

---

## Table of Contents

1. [Why Crucible (`crux`)?](#1-why-crucible-crux)
2. [Visual Inline-Dock TUI Preview](#2-visual-inline-dock-tui-preview)
3. [Installation: Compiled & Source Distributions (`v0.1`)](#3-installation-compiled--source-distributions-v01)
4. [Quickstart & Common Workflows](#4-quickstart--common-workflows)
5. [System Architecture](#5-system-architecture)
6. [Interactive TUI Controls, Hotkeys & Slash Commands](#6-interactive-tui-controls-hotkeys--slash-commands)
7. [Universal Model Brain & Cross-Provider Wire Normalizer](#7-universal-model-brain--cross-provider-wire-normalizer)
8. [4-Gear Elastic Swarm & 3-Stage Blind Cascading Judge](#8-4-gear-elastic-swarm--3-stage-blind-cascading-judge)
9. [3-Layer Security Firewall & Out-of-Tree Shadow Git](#9-3-layer-security-firewall--out-of-tree-shadow-git)
10. [Verification Matrix (`CT-01` – `CT-23`)](#10-verification-matrix-ct-01--ct-23)
11. [Codebase Map & Architecture Specifications](#11-codebase-map--architecture-specifications)

---

## 1. Why Crucible (`crux`)?

| Capability | Typical Coding CLI | Crucible (`crux` `v0.1`) |
| :--- | :--- | :--- |
| **Runtime Footprint** | 40–150+ PyPI/npm packages, fragile venvs | **`0` third-party packages** (100% Python 3.11+ stdlib + native C launcher, `<50ms` cold startup) |
| **Distribution Formats** | Source-only or heavy container | **Both Source & Compiled**: Native `arm64` Mach-O binary (`crux`), portable `.pyc` zipapp (`crux.pyz`), and source tarball |
| **Chain-of-Thought Visibility** | Hidden, truncated, or lost on context compaction | **100% uncompacted `.trace.jsonl` ledger** + live `Ctrl+O` **Thinking Threads Explorer** + Markdown research notebook export |
| **Concurrent Prompting** | Blocks keyboard input while model is thinking | **Always-live prompt composer**: press `Enter` mid-turn to queue into `⏭ Up next (N)` or `Alt+S` to steer the active turn live |
| **Terminal Scrollback** | Full-screen alt-buffer (`tput smcup`) breaks scrollback & copy-paste | **Scrollback-native Inline Dock** (`DEC ?2026h` synchronized output, `<= cols - 1` wrapping, mouse copy works natively over SSH/`tmux`) |
| **Cross-Provider Hot-Swap** | Locked to one vendor per session | **8-Rule `transform_messages` normalizer** swaps Gemini $\leftrightarrow$ Claude $\leftrightarrow$ OpenAI $\leftrightarrow$ Local mid-session without losing thought signatures |
| **Multi-Agent Swarm** | Uncoordinated parallel sub-agents | **4-Gear Elastic Gearbox (`G1`–`G4`)** + SQLite WAL Stigmergic Blackboard + **3-Stage Blind Cascading Judge** (`clean_patch` + `BARS` + Bradley-Terry) |

---

## 2. Visual Inline-Dock TUI Preview

On any interactive TTY, `crux` renders a flicker-free **Inline-Dock TUI** ([`src/ush/cli/tui.py`](src/ush/cli/tui.py)) that streams structured reasoning cards into native terminal scrollback while keeping your prompt editor live at the bottom:

```text
╭─ crux v0.1 (Crucible) · /workspace/project
│  session sess_9f3a1c82b4e0 · trace ~/.ush/sessions/sess_9f3a1c82b4e0.trace.jsonl
╰─ `/` menus · `@` files · `$` skills · `Ctrl+O` thinking threads · `⇧Tab` turbo · `Ctrl+T` thinking · `Enter` queues next · `Alt+S` steers live

╭─ 🧠 Step 1/160 · 1.2s · 418 tok · [hypothesis: verify stale-read lock ordering]
│  ⋮ Inspecting RwLock acquisition order in patch.py before reproducing concurrency test...
├─ ▸ read_file {"path": "src/ush/kernel/patch.py", "start_line": 1, "end_line": 90}
└─ ◂ ✓ ok (14ms)

⠋ Thinking [T1·S2/160] 2.4s · Evaluating 4-pass fuzzy patch invariants...  (Ctrl+O inspect threads)
⏭ Up next (1):
  1. Run the CT-05 and CT-06 conformance gates and export the trace notebook
╭──────────────────────────────────────────────────────────────────────────────────────╮
│ ❯ @src/ush/kernel/security.py Check if .ushignore blocks symlink traversal           │
╰──────────────────────────────────────────────────────────────────────────────────────╯
  gemini-3.1-pro-preview (Tier S) · ⚡ turbo (yolo · 160 steps) · 🧠 high · ⚙ auto · ^O threads (2)
```

Pressing **`Ctrl+O`** (while idle *or* mid-turn) opens the interactive split-pane **Thinking Threads Explorer** to browse every turn/step, scroll full uncompacted chains of thought (`←`/`→`/`PgUp`/`PgDn`), filter by keyword, or press **`Ctrl+E`** to export a complete Markdown research notebook.

---

## 3. Installation: Compiled & Source Distributions (`v0.1`)

Crucible `v0.1` ships with **both compiled executables and clean source distributions** inside [`releases/v0.1/`](releases/v0.1/) and on the [GitHub Releases page](https://github.com/jacobsohn-gif/Crucible/releases/tag/v0.1):

### Option A: Compiled Native Binary (macOS Apple Silicon `arm64`)
Compiled with `clang -O3` into a native Mach-O 64-bit executable with optimized `.pyc` bytecode embedded directly in `.rodata`:

```bash
# Run directly from the repository or download from GitHub Releases
./releases/v0.1/crux --version
# -> crux 0.1 (Crucible)

# Copy anywhere on your $PATH
cp ./releases/v0.1/crux-0.1-darwin-arm64 ~/.local/bin/crux
```

### Option B: Compiled Portable Single-File Bytecode (`crux.pyz`)
A platform-independent, single-file executable ZIP archive containing pre-compiled `.pyc` bytecode (`optimize=2`) that runs on **any macOS or Linux host** with Python 3.11+:

```bash
chmod +x ./releases/v0.1/crux.pyz
./releases/v0.1/crux.pyz --version
```

### Option C: Source Installation & Editable Development
```bash
git clone git@github.com:jacobsohn-gif/Crucible.git
cd Crucible

# Install `crux` (and backwards-compatible `ush` alias) onto ~/.local/bin + run all tests & builds
make install
make all
```

### Verifying Release Checksums (`SHA256SUMS.txt`)
```bash
cd releases/v0.1 && shasum -a 256 -c SHA256SUMS.txt
```

---

## 4. Quickstart & Common Workflows

```bash
# 1. Authenticate (auto-discovers ambient Vertex AI gcloud ADC, API keys, or local Ollama/vLLM)
gcloud auth application-default login
crux auth status
crux doctor

# 2. Launch Interactive Visual Inline-Dock TUI (or add --turbo for hands-free 160-step execution)
crux
crux --turbo -m gemini-3.1-pro-preview

# 3. Resume your latest session + full thought threads in the current workspace
crux -c --turbo
crux --resume sess_xxxxxxxxxxxx

# 4. Inspect, search, or export uncompacted Chains of Thought offline
crux trace list
crux trace stats
crux trace show last                 # or: crux trace show all | crux trace show t2:4
crux trace search "deadlock"
crux trace export ./audit_notebook.md

# 5. Headless Unix / CI Mode (`text`, `json` with verbatim `thoughts[]`, or `stream-json` SSE events)
crux -p "Audit src/ush/kernel/fsm.py invariants" --trace full
crux -p "Verify pyproject.toml dependencies" --output-format json
crux -p "Trace step-by-step execution" --output-format stream-json

# 6. Dispatch the 4-Gear Elastic Swarm, Peer Agent CLI, or start the loopback ACP JSON-RPC 2.0 daemon
crux swarm --gear auto "Implement feature X and verify unit tests"
crux swarm --peer-cli "Review auth module invariants and return findings"
crux swarm --argonclaw "Run parallel 6-stage architecture falsification review"
crux serve --acp
```

---

## 5. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                    Layer 1: CLI, Visual Inline-Dock TUI & ACP Daemon                    │
│  bin/crux · tui.py (DEC ?2026h, Menus, ^O Threads, Prompt Queue) · main.py · acp_server │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ StreamEvent (thinking_delta, tool_*, status)
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│             Layer 2: Universal Model Brain, Wire Normalizer & Thought Ledger            │
│  providers.py (Vertex/Gemini, Claude, OpenAI, OpenRouter, Local) · wire.py (8-Rule      │
│  transform_messages + ToolIdBijectionMap) · context_dag.py (LCA + 0.85 Compaction)      │
│  trace.py (Append-Only 0600 Secret-Redacted .trace.jsonl + Markdown Notebook Export)    │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ CanonicalMessage / PopperianLoopGuard
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│                  Layer 3: 6-Phase Execution Kernel & Security Firewall                  │
│  fsm.py (SessionRunner + Safe-Boundary Live Steer + /btw Zero-History Fork)             │
│  tools.py (Essential-11 RwLock Tools) · patch.py (4-Pass Fuzzy Edit + Stale-Read Guard) │
│  pty_manager.py (Persistent OSC 633 PTY) · shadow_git.py (Out-of-Tree Time Travel)      │
│  security.py (3-Layer Firewall: Env Scrubbing + .ushignore + Untrusted Quarantine)      │
└──────────────────────┬───────────────────────────────────────────┬──────────────────────┘
                       │                                           │
┌──────────────────────▼──────────────────────┐ ┌──────────────────▼──────────────────────┐
│    Layer 4: Skills, Commands, Hooks & Mem   │ │  Layer 5: 4-Gear Swarm & Cascading Judge│
│  skills.py (4-Tier SKILL.md Progressive)    │ │  engine.py (G1..G4 Gearbox + SQLite WAL │
│  commands.py (Unshadowable Slash Registry)  │ │  Stigmergic Blackboard + ContractDelta) │
│  hooks_memory_ext.py (9-Event Lifecycle     │ │  judge.py (Stage-A clean_patch ->       │
│  Hooks + Inbox-Gated Secret-Redacted Memory)│ │  Stage-B Swapped BARS -> Bradley-Terry) │
└─────────────────────────────────────────────┘ └─────────────────────────────────────────┘
```

---

## 6. Interactive TUI Controls, Hotkeys & Slash Commands

### Global Keyboard Shortcuts
| Shortcut | Action |
| :--- | :--- |
| **`/`**, **`@`**, **`$`** | Open interactive arrow-key **Slash Command Palette** (`/`), **File Attachment Picker** (`@`), or **Skill Activator** (`$`) |
| **`Ctrl+O`** | Toggle split-pane **Thinking Threads Explorer** (`↑`/`↓` select step, `←`/`→`/`PgUp`/`PgDn` scroll CoT, type to search, `Ctrl+E` export) |
| **`Shift+Tab`** | Cycle execution mode live (`default` $\to$ `auto_edit` $\to$ `turbo (yolo)` $\to$ `plan`) |
| **`Ctrl+T`** | Cycle native reasoning depth (`low` $\to$ `medium` $\to$ `high` $\to$ `max` $\to$ `off`) |
| **`Enter` (mid-turn)** | Queue the current draft into **`⏭ Up next (N)`** to auto-run as soon as the active turn completes |
| **`Alt+↑`** | Pop the most recently queued `Up next` prompt back into the composer for editing |
| **`Alt+S` (mid-turn)** | **Live-steer** the running turn immediately at the next safe `POST_INVOCATION` tool boundary (`CT-13`) |
| **`Alt+Enter`** | Insert a literal newline inside the multi-line prompt composer (Bracketed Paste also supported) |
| **`!<command>`** | Execute `<command>` directly inside the persistent stateful `OSC 633` PTY shell (preserving `cwd` and virtualenv) |

### Core Slash Commands
| Command | Description |
| :--- | :--- |
| `/threads [query]` | Open the interactive Thinking Threads Explorer (optionally pre-filtered by `query`) |
| `/turbo [on\|fast\|max\|off]` | Open the Turbo picker or switch directly into hands-free `yolo` mode (`160–200` steps/turn, firewall still active) |
| `/thinking [off\|low\|medium\|high\|max]` | Adjust native reasoning budget across Gemini, Claude, OpenAI, OpenRouter, and local `<think>` models |
| `/trace [full\|compact\|off]` | Set live thought display mode (`full` verbatim gutter, `compact` landmarks + timings, `off` final answer only) |
| `/trace show [last\|all\|<step>\|t<turn>:<step>]` | Replay verbatim uncompacted chains of thought from disk even after `0.85` context window compaction |
| `/trace search <query>` / `/trace export [path.md]` | Full-text search across every historical thought block or export a complete Markdown research notebook |
| `/queue [list\|pop\|clear]` / `/steer <msg>` | Manage queued follow-up prompts or inject immediate steering guidance mid-turn |
| `/btw <question>` | **100% prefix-KV-cached zero-history side question** — answers using current context without mutating session history (`CT-15`) |
| `/goal <objective>` / `/plan` / `/grill-me` | Run autonomous Popperian falsification loop, switch to read-only specification mode, or Socratically stress-test a design |
| `/restore [code+chat\|code-only\|chat-only]` | Rewind workspace files via out-of-tree Shadow Git and/or rewind the conversation DAG to its Lowest Common Ancestor (`LCA`) with synthesized `<branch_lessons>` |
| `/model [id]` / `/profile` / `/auth` | Hot-swap models or auth profiles mid-session via the 8-rule cross-provider normalizer (`CT-01`) |
| `/swarm <task>` / `/arena <task>` / `/review` | Launch the 4-Gear Stigmergic Swarm, run a parallel Git-worktree model race, or invoke the blind 2-stage `BARS` diff reviewer |
| `/skills` / `/learn <fact>` / `/trust [status\|grant]` | Inspect 4-tier `SKILL.md` skills, record a secret-redacted workspace memory note, or manage repository trust quarantine |

---

## 7. Universal Model Brain & Cross-Provider Wire Normalizer

Crucible normalizes streaming SSE wire protocols, native extended thinking blocks, and tool-call lifecycles across **all major frontier and local model families** ([`src/ush/brain/providers.py`](src/ush/brain/providers.py), [`src/ush/brain/wire.py`](src/ush/brain/wire.py)):

| Provider Family | Native Thinking Wire Control | Reliability & Failover Safeguards |
| :--- | :--- | :--- |
| **Google Vertex AI & Gemini API** (`gemini-3.1-pro-preview`, `gemini-4-argon`, `gemini-2.5-pro/flash`) | `thinkingConfig.thinkingLevel` (`LOW`/`MEDIUM`/`HIGH`) or `thinkingBudget` + `includeThoughts=true` | Automatic Vertex `global` $\leftrightarrow$ `us-central1` failover; preserves parallel tool-call `thoughtSignature` bytes (`CT-22`) |
| **Anthropic Claude (Direct & Vertex)** (`claude-opus-4`, `claude-sonnet-4`) | `thinking: {type: "enabled", budget_tokens: N}` + beta header `interleaved-thinking-2025-05-14` | Preserves cryptographic thinking `signature` blocks on same-provider turns; strips cleanly on cross-provider handoff |
| **OpenAI & Reasoning Models** (`gpt-5`, `o3`, `o4-mini`, `gpt-4.1`) | `reasoning_effort` (`low`/`medium`/`high`) + `reasoning_content` delta extraction | Automatic fallback if an endpoint rejects reasoning parameters (`HTTP 400` retry) |
| **OpenRouter / DeepSeek / Qwen / Kimi** (`deepseek-r1`, `qwen3-coder`, `kimi-k2`) | `include_reasoning: true` + chunk-safe `<think>...</think>` `InlineThinkStreamSplitter` | Repairs malformed Tier-B XML/JSON tool calls (`CT-04`) & enforces strict single-step scratchpad protocol on 7B–32B models |
| **Local GPU Endpoints** (`Ollama`, `vLLM`, `SGLang`, `llama.cpp`) | Auto-discovers `localhost:11434` / `localhost:8000` + `<think>` stream splitting | Zero-config offline operation with automatic Tier-B prompt scaling |

### 8-Rule `transform_messages` Invariants (`CT-01` – `CT-03`)
1. **System Prompt Byte-Stability (`CT-03`)**: Turn-0 system prompt and sorted tool schemas remain byte-identical across turns; dynamic skills and memory deltas are injected as user-role context pills to preserve **100% prefix KV-cache hit rates**.
2. **Bijective Tool-Call ID Normalization**: `ToolIdBijectionMap` translates provider-specific tool IDs into strict `^[a-zA-Z0-9_-]{1,64}$` tokens and back.
3. **Orphan Tool-Call Closure (`CT-08`)**: Any interrupted or canceled tool call receives an immediate deterministic `[SYNTHETIC_REPAIR]` result before the next model request so strict wire APIs never reject the transcript.
4. **Empty `HTTP 200` & `MAX_TOKENS` Recovery (`CT-02`)**: Transparently retries empty 200 streams and automatically splices truncated `MAX_TOKENS` responses via seamless continuation prompts.

---

## 8. 4-Gear Elastic Swarm & 3-Stage Blind Cascading Judge

Crucible scales compute elastically to match task complexity ([`src/ush/swarm/engine.py`](src/ush/swarm/engine.py), [`src/ush/swarm/judge.py`](src/ush/swarm/judge.py)):

- **Gear 1 (`G1` — Direct Fast Path)**: Single-agent execution in the active workspace for localized edits and queries.
- **Gear 2 (`G2` — Falsification Triad)**: Sequential **Planner $\to$ Executor $\to$ Adversarial Verifier** pipeline with automatic upshift on test failure.
- **Gear 3 (`G3` — Parallel Worktree Arena)**: Spawns isolated Git worktrees across diverse models/strategies in parallel and selects the winning patch via the Cascading Judge.
- **Gear 4 (`G4` — Stigmergic Blackboard Swarm & ArgonClaw Bridge)**: Full multi-agent DAG coordination over a SQLite WAL `StigmergicBlackboard` (`CONTRACT_DELTA`, `SPIKE`, `REQUEST_REVIEW`) with optional remote dispatch to the 12-agent **ArgonClaw Swarm**.

### 3-Stage Blind Cascading Judge (`CT-18` – `CT-20`)
Every candidate patch in `G3`/`G4` is stripped of model identity (`clean_patch`) and evaluated through three stages:
1. **Stage A (Deterministic Execution Gate)**: Rejects patches that fail git apply, syntax compilation, security firewall checks, or mandatory test suites.
2. **Stage B (Position-Swapped `BARS` Rubric)**: Evaluates surviving patches twice `(A, B)` and `(B, A)` across a 1–5 Behaviorally Anchored Rating Scale (`correctness`, `minimality`, `invariants`, `verification`) to eliminate positional bias.
3. **Stage C (Bradley-Terry Pairwise Solver & Quorum Ledger)**: Computes maximum-likelihood skill ratings and records the winning commit into an append-only, SHA-256 hash-chained SQLite `ConsensusLedger` (`CT-20`).

---

## 9. 3-Layer Security Firewall & Out-of-Tree Shadow Git

- **Layer 1 — Child Environment Scrubbing (`CT-12`)**: Strips cloud tokens, API keys, and SSH/GPG secrets from untrusted subprocesses before spawning PTY commands.
- **Layer 2 — `.ushignore` & Sensitive File Denylist (`CT-12`)**: Hard-blocks read/write/grep access to `.env*`, `*.pem`, `*.key`, credentials, and user-defined `.ushignore` patterns across all tools.
- **Layer 3 — Untrusted Repository Quarantine (`CT-12`, `CT-14`, `CT-16`)**: Blocks workspace-defined lifecycle hooks, MCP servers, and auto-activated skills in untrusted repos until explicitly authorized via `/trust grant`, and clamps extension policies so they can never weaken user security rules.
- **Zero-Pollution Shadow Git Time Travel (`CT-09`)**: Checkpoints workspace state into an out-of-tree bare repository (`~/.ush/snapshots/<workspace_hash>/repo.git`) before every mutating turn—never touching your project's `.git/index` or reflog.

---

## 10. Verification Matrix (`CT-01` – `CT-23`)

Run the complete zero-dependency test suite in `<3 seconds`:

```bash
.venv/bin/python -W error::ResourceWarning -m unittest discover -s tests -v
```

| Gate | Subsystem Verified | Status |
| :--- | :--- | :--- |
| **`CT-01`** | Mid-session cross-provider rotation (Gemini $\to$ Claude $\to$ OpenAI $\to$ Local) | ✅ Pass |
| **`CT-02`** | Empty `HTTP 200` stream retry & `MAX_TOKENS` continuation splicing | ✅ Pass |
| **`CT-03`** | Turn-0 prefix KV-cache byte stability across dynamic skill & memory injections | ✅ Pass |
| **`CT-04`** | Tier-B malformed JSON/XML tool-call repair & retry feedback loop | ✅ Pass |
| **`CT-05`** | 4-pass fuzzy `edit_file` engine (exact, CRLF/whitespace, indent-shifted, Unicode punctuation) | ✅ Pass |
| **`CT-06`** | Deadlock-free lexicographical `RwLock` acquisition & SHA-256 stale-read rejection | ✅ Pass |
| **`CT-07`** | Persistent `OSC 633` PTY state preservation (`cwd`/`venv`), interactive prompt detection & ring-buffer flood cap | ✅ Pass |
| **`CT-08`** | Mid-stream `SIGINT` cancellation & deterministic orphan tool-call closure | ✅ Pass |
| **`CT-09`** | Out-of-tree Shadow Git snapshot & restore with zero `.git/status` pollution | ✅ Pass |
| **`CT-10`** | Popperian loop guard (3-strike identical tool-call halt) & `0.85` `<epistemic_state>` compaction | ✅ Pass |
| **`CT-11`** | Conversation DAG `LCA` rewind with synthesized `<branch_lessons>` injection | ✅ Pass |
| **`CT-12`** | 3-Layer Security Firewall (env scrub, `.ushignore` block, untrusted repo quarantine) | ✅ Pass |
| **`CT-13`** | Mid-turn live steering (`enqueue_steer`) injected strictly at `POST_INVOCATION` boundary | ✅ Pass |
| **`CT-14`** | 4-Tier `SKILL.md` engine, colon-delimited `USH_SKILLS_PATH`, delta injection & sandbox path grant | ✅ Pass |
| **`CT-15`** | Unshadowable builtin slash commands, `!cmd` macro injection & zero-history `/btw` side fork | ✅ Pass |
| **`CT-16`** | 9-event lifecycle hooks (`exit 2` block) & privilege-clamped extension policy loading | ✅ Pass |
| **`CT-17`** | JIT workspace memory deduplication & secret-redacted `.inbox` quarantine | ✅ Pass |
| **`CT-18`** | `clean_patch` identity stripping & position-swapped Stage-B `BARS` Judge invariance | ✅ Pass |
| **`CT-19`** | Elastic Swarm `G1` $\to$ `G2` automatic upshift & SQLite WAL `CONTRACT_DELTA` propagation | ✅ Pass |
| **`CT-20`** | SHA-256 hash-chained `ConsensusLedger` tamper detection | ✅ Pass |
| **`CT-21`** | Uncompacted `.trace.jsonl` thought ledger, `<think>` chunk splitter, full-text search & Markdown export | ✅ Pass |
| **`CT-22`** | Gemini parallel tool-call `thoughtSignature` preservation & trailing user-turn enforcement | ✅ Pass |
| **`CT-23`** | Visual Inline-Dock TUI menus (`/`, `@`, `$`), `Ctrl+O` Thinking Threads Explorer & concurrent prompt queue | ✅ Pass |

---

## 11. Codebase Map & Architecture Specifications

### Source Modules (`29` Files, `0` External Dependencies)
- **CLI, Visual Inline-Dock TUI & ACP Server (`src/ush/cli/`)**
  - [`bin/crux`](bin/crux) & [`pyproject.toml`](pyproject.toml) — Symlink-safe source launcher & PEP 621 package config.
  - [`src/ush/cli/tui.py`](src/ush/cli/tui.py) — Visual Inline-Dock TUI (`InlineDockCompositor`, `VisualTuiController`, `ThinkingThreadTracker`, `MenuOverlay`, DEC `?2026h` sync output, `Ctrl+O` Threads Explorer, Concurrent Prompt Queue).
  - [`src/ush/cli/main.py`](src/ush/cli/main.py) — CLI entrypoint, slash command handlers, headless `-p` runner, `crux trace`, `crux auth`, `crux swarm`, and `crux doctor`.
  - [`src/ush/cli/acp_server.py`](src/ush/cli/acp_server.py) — Loopback-guarded (`127.0.0.1`) & `stdio` JSON-RPC 2.0 Agent Client Protocol (ACP) server.
- **Universal Model Brain & Long-Form Thought Ledger (`src/ush/brain/`)**
  - [`src/ush/brain/providers.py`](src/ush/brain/providers.py) — Native extended/interleaved thinking SSE clients for Vertex/Gemini, Claude, OpenAI, OpenRouter, and local endpoints.
  - [`src/ush/brain/wire.py`](src/ush/brain/wire.py) — 8-rule `transform_messages` cross-provider normalizer & `ToolIdBijectionMap`.
  - [`src/ush/brain/trace.py`](src/ush/brain/trace.py) — Append-only `0600` secret-redacted `ThoughtTraceLedger` (`.trace.jsonl`), landmark extractor, full-text search, and Markdown notebook exporter.
  - [`src/ush/brain/context_dag.py`](src/ush/brain/context_dag.py) — Branching `SessionDAG`, `LCA` rewind, observation masking, `0.85` compaction, and `PopperianLoopGuard`.
  - [`src/ush/brain/model_tiers.py`](src/ush/brain/model_tiers.py) — `Tier S / A / B` capability profiles and cache-invariant system prompt builder.
- **Multi-Account Auth & Keychain Vault (`src/ush/auth/`)**
  - [`src/ush/auth/vault.py`](src/ush/auth/vault.py) — Ambient credential discovery (`gcloud` Vertex ADC, ENV, localhost), OAuth 2.0 + PKCE (`S256`), and macOS Keychain / `0600` atomic file vault.
- **Execution Kernel & Security Firewall (`src/ush/kernel/`)**
  - [`src/ush/kernel/fsm.py`](src/ush/kernel/fsm.py) — 6-phase `SessionRunner`, `StreamEvent` dispatcher, safe-boundary `enqueue_steer`, and `/btw` side-chat fork.
  - [`src/ush/kernel/tools.py`](src/ush/kernel/tools.py) & [`src/ush/kernel/patch.py`](src/ush/kernel/patch.py) — Essential-11 `RwLock` tools, 4-pass fuzzy `edit_file`, and `StaleReadTracker`.
  - [`src/ush/kernel/pty_manager.py`](src/ush/kernel/pty_manager.py) — Persistent stateful `OSC 633` PTY engine with prompt auto-detection.
  - [`src/ush/kernel/shadow_git.py`](src/ush/kernel/shadow_git.py) & [`src/ush/kernel/security.py`](src/ush/kernel/security.py) — Out-of-tree Shadow Git store and 3-layer security firewall.
- **Extensibility Platform & 4-Gear Swarm (`src/ush/platform/`, `src/ush/swarm/`)**
  - [`src/ush/platform/skills.py`](src/ush/platform/skills.py), [`src/ush/platform/commands.py`](src/ush/platform/commands.py), [`src/ush/platform/hooks_memory_ext.py`](src/ush/platform/hooks_memory_ext.py) — 4-tier `SKILL.md` loader, slash command registry, 9-event hooks, and JIT workspace memory.
  - [`src/ush/swarm/engine.py`](src/ush/swarm/engine.py) & [`src/ush/swarm/judge.py`](src/ush/swarm/judge.py) — 4-Gear Elastic Swarm, SQLite WAL `StigmergicBlackboard`, 3-Stage Cascading Judge, and SHA-256 `ConsensusLedger`.
- **Build & Release Automation (`scripts/`, `releases/v0.1/`)**
  - [`scripts/build_release.py`](scripts/build_release.py) & [`Makefile`](Makefile) — Reproducible compiler pipeline generating native `arm64` Mach-O binary (`crux`), portable `.pyz` bytecode (`crux.pyz`), source tarball (`crucible-0.1-src.tar.gz`), and `SHA256SUMS.txt`.

### Deep-Dive Design Specifications (`docs/`)
- [`MASTER_IMPLEMENTATION_PLAN.md`](MASTER_IMPLEMENTATION_PLAN.md) — End-to-end architectural blueprint and distillation synthesis.
- [`docs/01_SYSTEM_ARCHITECTURE.md`](docs/01_SYSTEM_ARCHITECTURE.md) — 6-Phase Kernel FSM, Essential-11 Tools, Persistent PTY, Shadow Git, and Security Firewall.
- [`docs/02_UNIVERSAL_MODEL_ADAPTER_SPEC.md`](docs/02_UNIVERSAL_MODEL_ADAPTER_SPEC.md) — Cross-Provider Wire Normalizer, Extended Thinking Streams, and Tier S/A/B Prompt Scaling.
- [`docs/03_ELASTIC_SWARM_AND_JUDGE_SPEC.md`](docs/03_ELASTIC_SWARM_AND_JUDGE_SPEC.md) — 4-Gear Stigmergic Swarm, SQLite WAL Blackboard, and 3-Stage Blind Cascading Judge.
- [`docs/04_VERIFICATION_AND_BENCHMARK_PLAN.md`](docs/04_VERIFICATION_AND_BENCHMARK_PLAN.md) — Formal Specification of Conformance & Chaos Gates (`CT-01` – `CT-23`).
- [`docs/05_CLI_SKILLS_COMMANDS_HOOKS_AND_EXTENSIONS_SPEC.md`](docs/05_CLI_SKILLS_COMMANDS_HOOKS_AND_EXTENSIONS_SPEC.md) — 4-Tier `SKILL.md` Progressive Disclosure, Slash Commands, Lifecycle Hooks, and JIT Memory.

---

## License

Apache-2.0. Built and maintained by Stephen Jacobsohn ([@jacobsohn-gif](https://github.com/jacobsohn-gif)).
