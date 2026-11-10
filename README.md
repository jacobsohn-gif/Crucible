# Crucible (`crux` `v0.1`)

[![Version](https://img.shields.io/badge/version-v0.1-00b4d8?style=flat-square)](https://github.com/jacobsohn-gif/Crucible/releases/tag/v0.1)
[![Dependencies](https://img.shields.io/badge/PyPI_dependencies-0_(stdlib_only)-2ec4b6?style=flat-square)](pyproject.toml)
[![Distributions](https://img.shields.io/badge/distributions-Source_%2B_Compiled_(Mach--O_%26_.pyz)-ff9f1c?style=flat-square)](releases/v0.1/)
[![Verification](https://img.shields.io/badge/tests-55%2F55_passing_(CT--01..CT--28_%2B_Crux--Evals)-38b000?style=flat-square)](tests/)
[![License](https://img.shields.io/badge/license-Apache--2.0-6c757d?style=flat-square)](pyproject.toml)

**Crucible (`crux`)** is a standalone, zero-third-party-dependency **interactive coding, reasoning-assay & 4-Gear Elastic Swarm CLI** with **uncompacted long-form thought tracing**, **streaming Terminal Markdown + LaTeX math rendering**, and **live runtime model discovery**.

Every hypothesis is tested in the fire: `crux` pairs a **Popperian falsification loop** (reproduce $\to$ hypothesize $\to$ patch $\to$ verify) with a scrollback-native **Visual Inline-Dock TUI**, dynamic **Antigravity ACP (`agy acp`) & multi-provider enterprise auth**, lossless **mid-session cross-provider model rotation**, an **out-of-tree Shadow Git time machine**, the native **`Crux-Evals` (Crucible Assay)** open-protocol MCP evaluator bank, and a **4-Gear Stigmergic Swarm** governed by a **3-Stage Blind Cascading Judge**.

### The Crucible Ecosystem
1. **[`Crucible` (`crux`)](https://github.com/jacobsohn-gif/Crucible)** *(this repo)* — Interactive thought-tracing coding & swarm CLI, bundled natively with `crux_evals`.
2. **[`Crux-Evals` (`crux-evals`)](https://github.com/jacobsohn-gif/Crux-Evals)** — Standalone Open-Protocol MCP & Hook Evaluator Suite (Deterministic Sentinels, Blind `BARS` Judge, Adversarial `/tmp` Breaker) compatible with Claude Code, Cursor, Gemini CLI, Codex, and Crucible.
3. **[`Crux-Swarm` (`crux-swarm`)](https://github.com/jacobsohn-gif/Crux-Swarm)** — Symmetric `Crux-Cell` container runtime, socket-activated OS/Browser supervisor, 4-Gear Stigmergic Spine, and Mobile PWA Cockpit.

---

## Table of Contents

1. [Why Crucible (`crux`)?](#1-why-crucible-crux)
2. [Recent Bug Fixes & Feature Updates](#2-recent-bug-fixes--feature-updates)
3. [Visual Inline-Dock TUI, Streaming Markdown/LaTeX & Interactive Cards](#3-visual-inline-dock-tui-streaming-markdownlatex--interactive-cards)
4. [Installation: Compiled & Source Distributions (`v0.1`)](#4-installation-compiled--source-distributions-v01)
5. [Quickstart & Common Workflows](#5-quickstart--common-workflows)
6. [System Architecture](#6-system-architecture)
7. [Interactive TUI Controls, Hotkeys & Slash Commands](#7-interactive-tui-controls-hotkeys--slash-commands)
8. [Universal Model Brain, Live Antigravity ACP & Cross-Provider Normalizer](#8-universal-model-brain-live-antigravity-acp--cross-provider-normalizer)
9. [Native Crux-Evals (Crucible Assay) MCP & Hook Evaluator Suite](#9-native-crux-evals-crucible-assay-mcp--hook-evaluator-suite)
10. [4-Gear Elastic Swarm & 3-Stage Blind Cascading Judge](#10-4-gear-elastic-swarm--3-stage-blind-cascading-judge)
11. [3-Layer Security Firewall & Out-of-Tree Shadow Git](#11-3-layer-security-firewall--out-of-tree-shadow-git)
12. [Verification Matrix (`CT-01` – `CT-28` & Adversarial Suites)](#12-verification-matrix-ct-01--ct-28--adversarial-suites)
13. [Codebase Map](#13-codebase-map)

---

## 1. Why Crucible (`crux`)?

| Capability | Typical Coding CLI | Crucible (`crux` `v0.1`) |
| :--- | :--- | :--- |
| **Runtime Footprint** | 40–150+ PyPI/npm packages, fragile venvs | **`0` third-party packages** (100% Python 3.11+ `stdlib` + native C launcher, `<50ms` cold startup) |
| **Distribution Formats** | Source-only or heavy container | **Both Source & Compiled**: Native `arm64` Mach-O binary (`crux`), portable `.pyc` zipapp (`crux.pyz`), and source tarball |
| **Model Discovery & UX** | Hardcoded static model IDs & cluttered vendor prefixes | **Live runtime discovery** (`agy acp`, `:loadCodeAssist`, `/models`) showing clean, login-scoped model names (including `Gemini 4 Argon` & `Gemini 3.8/3.7/3.6 Flash`) |
| **Terminal Readability** | Raw unstyled Markdown or broken LaTeX/Unicode tofu | **Streaming Terminal Markdown & font-safe LaTeX math renderer** (box tables, syntax highlighting, `$...$`/`$$...$$`/`\begin{aligned}` $\to$ Menlo-safe Unicode) |
| **Chain-of-Thought Visibility** | Hidden, truncated, or lost on context compaction | **100% uncompacted `.trace.jsonl` ledger** + live `Ctrl+O` **Thinking Threads Explorer** + Markdown research notebook export |
| **Concurrent Prompting & Menus** | Blocks keyboard while thinking; clips long options | **Always-live composer** (`Enter` queues `⏭ Up next`, `Alt+S` live-steers) + **full-width multi-line `ask_user` cards** (`Space` expand all, `1`–`9` quick-pick) |
| **Terminal Scrollback** | Full-screen alt-buffer (`tput smcup`) breaks copy-paste | **Scrollback-native Inline Dock** (`DEC ?2026h` synchronized output, `<= cols - 1` wrapping, mouse copy works natively over SSH/`tmux`) |
| **Independent Self-Verification** | Model grades its own output | **Native `Crux-Evals` (Crucible Assay)** 5-module MCP/Hook evaluator bank + **3-Stage Blind Cascading Judge** (`clean_patch` + `BARS` + Bradley-Terry) |

---

## 2. Recent Bug Fixes & Feature Updates

### Bug Fixes & Reliability Hardening
- **Fixed Cut-Off `ask_user` Options & Clipped Approval Targets** ([`src/ush/cli/tui.py`](src/ush/cli/tui.py), [`src/ush/cli/main.py`](src/ush/cli/main.py)):
  - Removed the 34-column label truncation on interactive `ask_user` question cards. Questions and all multiple-choice options now word-wrap across full terminal width (`_render_ask_user_menu_lines`).
  - Automatically expands every option in full when they fit inside the dock, always expands the currently highlighted option across multiple lines, and adds **`Space`** to toggle full scrollable expansion of all options (`←`/`→`/`PgUp`/`PgDn`), **`1`–`9`** & **`c`** instant quick-select keys, persistent wrapped question display while typing a custom reply, and wrapped detail footers for long tool-approval targets and `/`, `@`, `$` picker items.
- **Fixed Multi-Turn Context Loss & Repeated Output on Antigravity ACP** ([`src/ush/brain/agy_acp.py`](src/ush/brain/agy_acp.py), [`src/ush/brain/wire.py`](src/ush/brain/wire.py), [`src/ush/brain/context_dag.py`](src/ush/brain/context_dag.py)):
  - Made `AgyAcpBridge` warm-session prefix tracking (`_AcpMsgRecord`) invariant to `<dynamic_instructions>` tail erasure, superseded `read_file` observation tombstoning, assistant replies quoting `<dynamic_instructions>`, zero-history `/btw` side forks, `/plan` toggles, and live `/model` switches—eliminating spurious session resets and repeated answers across long sessions.
  - Capped single `session/prompt` cold-hydration payloads at `135,000` chars with tail-first retention (`<prior_conversation_digest>` + verbatim `<current_user_request>`) so large resumed transcripts never hit `agy acp`'s `~195 KB` front-truncation window.
  - Isolated Crucible's XML tool-call tags as `<crux_tool_call>` (retaining backward-compatible `<tool_call>` parsing) so bridged tool calls never collide with `agy`'s internal stream validator, preserved both initial and latest user goals across `0.85` compaction, and fixed `TokenUsage.input_tokens` accounting (`raw_in + cached_in`).
- **Fixed Unrendered LaTeX / Special Math Characters & Tofu Boxes** ([`src/ush/cli/markdown.py`](src/ush/cli/markdown.py), [`src/ush/cli/tui.py`](src/ush/cli/tui.py)):
  - Added full inline/display math rendering (`$...$`, `$$...$$`, `\(...\)`, `\[...\]`, `\frac`, `\sqrt`, `\sum`, `\int`, `\mathbb`, `\mathcal`, `\text`, Greek letters, arrows, operators, sub/superscripts, and `\begin{aligned|cases|bmatrix|pmatrix}` environments) across streaming assistant replies, reasoning summaries, and `Ctrl+O` threads while rigorously leaving fenced/inline code untouched.
  - Mapped Supplementary Multilingual Plane math codepoints onto Basic Multilingual Plane glyphs verified against macOS Menlo / Apple Terminal so mathematical symbols render cleanly with zero missing-glyph tofu boxes (`□`/`?`).
- **Fixed Repeated Tool-Approval Prompt Spam (`CT-28`)** ([`src/ush/cli/tui.py`](src/ush/cli/tui.py)):
  - Eliminated approval-box redraw loops while paused waiting for user acceptance by halting background spinner ticks whenever an interactive modal menu is open, deduplicating dock frame digests, and serializing parallel `RwLock` write approvals behind a dedicated `_interactive_prompt_lock`.
- **Fixed Static Model Lists, Missing `Gemini 4 Argon` & Cluttered Provider Prefixes (`CT-27`)** ([`src/ush/brain/agy_acp.py`](src/ush/brain/agy_acp.py), [`src/ush/auth/vault.py`](src/ush/auth/vault.py), [`src/ush/cli/tui.py`](src/ush/cli/tui.py)):
  - Replaced static model assumptions and `HTTP 404` cloud endpoint guesses with live runtime model loading straight from `agy acp` (`Gemini 4 Argon`, `Gemini 3.8 Flash`, `Gemini 3.7 Flash`, `Gemini 3.6 Flash`, `Gemini 3.1 Pro`), Google Code Assist `:loadCodeAssist`, and OpenAI/LiteLLM `/models`.
  - Removed noisy `gemini/` and `antigravity/` prefixes from `/model` and status bars, prioritizing the user's active authenticated session automatically.
- **Hardened Rate-Limit, PTY, Wire Bijection & Firewall Edge Cases (`CT-24` – `CT-27`)** ([`src/ush/brain/providers.py`](src/ush/brain/providers.py), [`src/ush/kernel/pty_manager.py`](src/ush/kernel/pty_manager.py), [`src/ush/kernel/security.py`](src/ush/kernel/security.py)):
  - Added interruptible `429`/`503`/`529` exponential backoff, `HTTP 401` ADC token auto-refresh, context-overflow auto-compaction, dead-PTY shell auto-respawn, strict `functionCall` $\leftrightarrow$ `functionResponse` bijection across mid-history orphan repairs, deadlock-free `/steer`, shell-wrapper unwrapping (`sudo`/`env`/`sh -c`), `.ushignore` negation hardening, and constant-time bearer auth on `crux serve --acp`.

### Major New Capabilities
- **Native `Crux-Evals` (Crucible Assay) Suite & Hardline Model Isolation** ([`src/crux_evals/`](src/crux_evals/), [`src/ush/brain/evaluator_isolation.py`](src/ush/brain/evaluator_isolation.py)): 6 plug-and-play evaluators (`trajectory_sentinel`, `clean_patch_and_contract`, `epistemic_grounding_auditor`, `blind_bars_judge`, `adversarial_falsifier`, generalized `artifact_enhancer`) + `consult_evaluator` tool with live `SessionTelemetry` hydration, cross-family isolated LLM inference (`tools=[]`, zero Actor history/CoT), `EVALUATOR.md` skill-like authoring (`/evaluators` & `crux eval`), and utility-driven selection hints.
- **Streaming Terminal Markdown Engine** ([`src/ush/cli/markdown.py`](src/ush/cli/markdown.py)): Zero-dependency incremental Markdown stream formatter with syntax-highlighted fenced blocks, Unicode box-drawn GFM tables, callout blocks, semantic headings, lists, and clickable file links.
- **Enterprise Multi-Provider Auth & Peer CLI Delegation** ([`src/ush/auth/vault.py`](src/ush/auth/vault.py), [`src/ush/swarm/engine.py`](src/ush/swarm/engine.py)): Support for Gemini Enterprise (VPC-SC, Priority Tier, Express), OpenAI `/v1/responses` + Device/PKCE OAuth, Claude OAuth/Vertex, LiteLLM proxy discovery, workstation command-auth helpers, and `crux swarm --peer-cli` external agent delegation.

---

## 3. Visual Inline-Dock TUI, Streaming Markdown/LaTeX & Interactive Cards

On any interactive TTY, `crux` renders a flicker-free **Inline-Dock TUI** ([`src/ush/cli/tui.py`](src/ush/cli/tui.py), [`src/ush/cli/markdown.py`](src/ush/cli/markdown.py)) that streams structured reasoning cards, formatted Markdown tables/code/math, and interactive decision cards into native terminal scrollback while keeping your prompt editor live at the bottom:

```text
╭─ crux v0.1 (Crucible) · /workspace/project
│  session sess_9f3a1c82b4e0 · trace ~/.ush/sessions/sess_9f3a1c82b4e0.trace.jsonl
╰─ `/` menus · `@` files · `$` skills · `Ctrl+O` thinking threads · `⇧Tab` turbo · `Ctrl+T` thinking · `Enter` queues next · `Alt+S` steers live

╭─ 🧠 Step 1/160 · 1.2s · 418 tok · [hypothesis: verify stale-read lock ordering]
│  ⋮ Inspecting RwLock acquisition order L₁ < L₂ → deadlock-free before reproducing test...
├─ ▸ read_file {"path": "src/ush/kernel/patch.py", "start_line": 1, "end_line": 90}
└─ ◂ ✓ ok (14ms)

╭─ ❓ Crux Question (↑/↓ select · Space expand all · Enter confirm · 1-9/c pick)
│  Which concurrency strategy should we apply to the session snapshot writer?
│  ❯ 1. Atomic temp-file + fsync + os.replace under lexicographical RwLock (Recommended —
│       preserves crash consistency without blocking parallel read_file callers)
│    2. SQLite WAL single-writer journal with busy_timeout=5000ms
│    ✎ Type custom answer… (c)
╰──────────────────────────────────────────────────────────────────────────────────────
  Gemini 4 Argon (Tier S) · ⚡ default · 🧠 high · ⚙ auto · ^O threads (2)
```

Pressing **`Ctrl+O`** (while idle *or* mid-turn) opens the interactive split-pane **Thinking Threads Explorer** to browse every turn/step, scroll full uncompacted chains of thought (`←`/`→`/`PgUp`/`PgDn`), filter by keyword, or press **`Ctrl+E`** to export a complete Markdown research notebook.

---

## 4. Installation: Compiled & Source Distributions (`v0.1`)

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
python3.11 ./releases/v0.1/crux.pyz --version
```

### Option C: Source Installation & Editable Development
```bash
git clone https://github.com/jacobsohn-gif/Crucible.git
cd Crucible

# Build release artifacts, verify tests, and install `crux` + `ush` onto ~/.local/bin
make all
make install
```

### Verifying Release Checksums (`SHA256SUMS.txt`)
```bash
cd releases/v0.1 && shasum -a 256 -c SHA256SUMS.txt
```

---

## 5. Quickstart & Common Workflows

```bash
# 1. Authenticate (Auto-detects local Antigravity `agy` login + live models, Vertex ADC, API keys, or local Ollama/vLLM)
crux auth login antigravity          # Native Antigravity login / runtime `agy acp` discovery (zero GCP project needed)
gcloud auth application-default login
crux auth status
crux doctor

# 2. Launch Interactive Visual Inline-Dock TUI (or add --turbo for hands-free 160-step execution)
crux
crux --turbo

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

# 6. Run Native Crux-Evals (Crucible Assay) Evaluators or Open MCP Server
PYTHONPATH=src python3 -m crux_evals.cli list
PYTHONPATH=src python3 -m crux_evals.cli run --objective "Verify patch invariants"
PYTHONPATH=src python3 -m crux_evals.cli mcp --workspace .

# 7. Dispatch the 4-Gear Elastic Swarm, Peer Agent CLI, or Authenticated Loopback ACP Server
crux swarm --gear auto "Implement feature X and verify unit tests"
crux swarm --peer-cli "Review auth module invariants and return findings"
crux swarm --argonclaw "Run parallel 6-stage architecture falsification review"
crux serve --acp
```

---

## 6. System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│             Layer 1: CLI, Visual Inline-Dock TUI, Markdown/LaTeX & ACP Daemon           │
│  bin/crux · tui.py (DEC ?2026h, Full-Width Menus, ^O Threads, Zero-Spam Approval Dock)  │
│  markdown.py (Streaming GFM Tables, Code & Font-Safe LaTeX Math) · main.py · acp_server │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ StreamEvent (thinking_delta, tool_*, status)
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│        Layer 2: Universal Model Brain, Live ACP Bridge, Wire Normalizer & Traces        │
│  agy_acp.py (Dynamic Runtime Models + Warm Session Reuse + <=135KB Tail-Safe Hydration) │
│  providers.py (Antigravity, Code Assist, Gemini/Vertex, Claude, OpenAI, LiteLLM, Local) │
│  wire.py (8-Rule transform_messages + <crux_tool_call> + Bijective ToolIdBijectionMap)  │
│  context_dag.py (LCA + 0.85 Compaction) · trace.py (0600 Secret-Redacted .trace.jsonl)  │
└───────────────────────────────────────────┬─────────────────────────────────────────────┘
                                            │ CanonicalMessage / PopperianLoopGuard
┌───────────────────────────────────────────▼─────────────────────────────────────────────┐
│                  Layer 3: 6-Phase Execution Kernel & Security Firewall                  │
│  fsm.py (SessionRunner + Safe-Boundary Live Steer + /btw Zero-History Fork)             │
│  tools.py (Essential-11 RwLock Tools) · patch.py (4-Pass Fuzzy Edit + Stale-Read Guard) │
│  pty_manager.py (Persistent OSC 633 PTY + Auto-Respawn) · shadow_git.py (Out-of-Tree)   │
│  security.py (3-Layer Firewall: Shell Unwrap + .ushignore + Repo Trust Fingerprinting)  │
└──────────────────────┬───────────────────────────┬──────────────────────────────────────┘
                       │                           │
┌──────────────────────▼───────────────┐ ┌─────────▼─────────────────┐ ┌──────────────────▼──────────────────┐
│  Layer 4: Skills, Cmds, Hooks & Mem  │ │ Layer 5: 4-Gear Swarm     │ │ Layer 6: Crux-Evals (Assay)         │
│  skills.py (4-Tier SKILL.md)         │ │ engine.py (G1..G4 Gearbox │ │ crux_evals/ (5-Module Cascade,      │
│  commands.py (Unshadowable Slash)    │ │ + SQLite WAL Blackboard)  │ │ Open MCP Server, CLI Hook &         │
│  hooks_memory_ext.py (9-Event Hooks) │ │ judge.py (3-Stage BARS)   │ │ Auto-Discovery EvaluatorRegistry)   │
└──────────────────────────────────────┘ └───────────────────────────┘ └─────────────────────────────────────┘
```

---

## 7. Interactive TUI Controls, Hotkeys & Slash Commands

### Global & Menu Keyboard Shortcuts
| Shortcut | Action |
| :--- | :--- |
| **`/`**, **`@`**, **`$`** | Open interactive arrow-key **Slash Command Palette** (`/`), **File Attachment Picker** (`@`), or **Skill Activator** (`$`) |
| **`Space` (in menus / `ask_user`)** | Expand or collapse **all** full-width wrapped options/targets with `←`/`→`/`PgUp`/`PgDn` scrolling |
| **`1`–`9` / `c` (in `ask_user`)** | Immediately pick numbered option `1`–`9` or press `c` to type a custom free-form answer |
| **`y` / `e` / `t` / `n` (in approvals)** | Approve tool once (`y`), switch to `auto_edit` (`e`), switch to `turbo` (`t`), or deny (`n`) |
| **`Ctrl+O`** | Toggle split-pane **Thinking Threads Explorer** (`↑`/`↓` select step, `←`/`→`/`PgUp`/`PgDn` scroll CoT, type to search, `Ctrl+E` export) |
| **`Shift+Tab`** | Cycle execution mode live (`default` $\to$ `auto_edit` $\to$ `turbo (yolo)` $\to$ `plan`) |
| **`Ctrl+T`** | Cycle native reasoning depth (`low` $\to$ `medium` $\to$ `high` $\to$ `max` $\to$ `off`) |
| **`Enter` (mid-turn)** | Queue the current draft into **`⏭ Up next (N)`** to auto-run as soon as the active turn completes |
| **`Alt+↑`** | Pop the most recently queued `Up next` prompt back into the composer for editing |
| **`Alt+S` (mid-turn)** | **Live-steer** the running turn immediately at the next safe `POST_INVOCATION` tool boundary (`CT-13`, `CT-24`) |
| **`Alt+Enter`** | Insert a literal newline inside the multi-line prompt composer (Bracketed Paste also supported) |
| **`!<command>`** | Execute `<command>` directly inside the persistent stateful `OSC 633` PTY shell (preserving `cwd` and virtualenv) |

### Core Slash Commands
| Command | Description |
| :--- | :--- |
| `/model [id]` / `/profile` / `/auth` | Browse clean, dynamically discovered models for your active login or hot-swap providers mid-session (`CT-01`, `CT-27`) |
| `/threads [query]` | Open the interactive Thinking Threads Explorer (optionally pre-filtered by `query`) |
| `/turbo [on\|fast\|max\|off]` | Open the Turbo picker or switch directly into hands-free `yolo` mode (`160–200` steps/turn, firewall still active) |
| `/thinking [off\|low\|medium\|high\|max]` | Adjust native reasoning budget across Antigravity, Gemini, Claude, OpenAI, OpenRouter, and local `<think>` models |
| `/trace [full\|compact\|off]` | Set live thought display mode (`full` verbatim gutter, `compact` landmarks + timings, `off` final answer only) |
| `/trace show [last\|all\|<step>\|t<turn>:<step>]` | Replay verbatim uncompacted chains of thought from disk even after `0.85` context window compaction |
| `/trace search <query>` / `/trace export [path.md]` | Full-text search across every historical thought block or export a complete Markdown research notebook |
| `/queue [list\|pop\|clear]` / `/steer <msg>` | Manage queued follow-up prompts or inject immediate steering guidance mid-turn |
| `/btw <question>` | **100% prefix-KV-cached zero-history side question** — answers using current context without mutating session history or ACP warm cache (`CT-15`) |
| `/goal <objective>` / `/plan` / `/grill-me` | Run autonomous Popperian falsification loop, switch to read-only specification mode, or Socratically stress-test a design |
| `/restore [code+chat\|code-only\|chat-only]` | Rewind workspace files via out-of-tree Shadow Git and/or rewind the conversation DAG to its Lowest Common Ancestor (`LCA`) with synthesized `<branch_lessons>` |
| `/swarm <task>` / `/arena <task>` / `/review` | Launch the 4-Gear Stigmergic Swarm, run a parallel Git-worktree model race, or invoke the blind 2-stage `BARS` diff reviewer |
| `/skills` / `/evaluators` / `/learn <fact>` / `/trust` | Inspect 4-tier `SKILL.md` skills or `EVALUATOR.md`/Python evaluators (`/evaluators new <id>`), record workspace memory, or manage repo trust quarantine |

---

## 8. Universal Model Brain, Live Antigravity ACP & Cross-Provider Normalizer

Crucible normalizes runtime model discovery, streaming wire protocols, extended thinking blocks, and tool-call lifecycles across **all major frontier, enterprise, and local model families** ([`src/ush/brain/agy_acp.py`](src/ush/brain/agy_acp.py), [`src/ush/brain/providers.py`](src/ush/brain/providers.py), [`src/ush/brain/wire.py`](src/ush/brain/wire.py)):

| Provider Family | Runtime Discovery & Wire Protocol | Reliability & Failover Safeguards |
| :--- | :--- | :--- |
| **Google Antigravity (`agy acp` & Code Assist)** (`Gemini 4 Argon`, `Gemini 3.8/3.7/3.6 Flash`, `Gemini 3.1 Pro`) | Live JSON-RPC `session/new` dynamic model table via `agy acp` + `:loadCodeAssist` project/tier handshake | Normalization-invariant warm multi-turn ACP session reuse, `<=135 KB` tail-safe cold hydration, `<crux_tool_call>` isolation, and clean vendor-free display names |
| **Gemini API, Gemini Enterprise & Vertex AI** (`gemini-3.1-pro-preview`, `gemini-2.5-pro/flash`) | `thinkingConfig.thinkingLevel` / `thinkingBudget` + `includeThoughts=true`; VPC-SC custom hosts, Priority Tier & Express | Vertex `global` $\leftrightarrow$ `us-central1` failover, `HTTP 401` ADC refresh, interruptible `429`/`503` backoff, parallel `thoughtSignature` preservation (`CT-22`, `CT-25`) |
| **Anthropic Claude (Direct, OAuth & Vertex)** (`claude-opus-4`, `claude-sonnet-4`) | `thinking: {type: "enabled", budget_tokens: N}` + `interleaved-thinking-2025-05-14` | Preserves cryptographic thinking `signature` blocks on same-provider turns; strips cleanly on cross-provider handoff |
| **OpenAI (`/v1/responses` & Chat)** (`gpt-5`, `o3`, `o4-mini`, `gpt-4.1`) | `/v1/responses` + `/v1/chat/completions`, PKCE/Device OAuth, `reasoning_effort` + `reasoning_content` extraction | Automatic fallback if an endpoint rejects reasoning parameters (`HTTP 400` retry) |
| **LiteLLM, OpenRouter, DeepSeek, Qwen & Kimi** (`deepseek-r1`, `qwen3-coder`, `kimi-k2`) | Live `/models` discovery, dual auth headers, `include_reasoning: true` + chunk-safe `<think>` `InlineThinkStreamSplitter` | Repairs malformed Tier-B XML/JSON tool calls (`CT-04`) & enforces single-step scratchpad discipline on 7B–32B models |
| **Local GPU & On-Device Endpoints** (`Ollama`, `vLLM`, `SGLang`, `llama.cpp`, `LiteRT`) | Auto-discovers `localhost:11434` / `localhost:8000` + `<think>` stream splitting | Zero-config offline operation with automatic Tier-B prompt scaling |

### Core Wire Invariants (`CT-01` – `CT-03`, `CT-25` – `CT-27`)
1. **System Prompt Byte-Stability (`CT-03`)**: Turn-0 system prompt and sorted tool schemas remain byte-identical across turns; dynamic skills and memory deltas are injected as tail-only `<dynamic_instructions>` user pills to preserve **prefix KV-cache hits** and warm ACP sessions.
2. **Bijective Tool-Call ID & `functionResponse` Closure (`CT-08`, `CT-25`)**: `ToolIdBijectionMap` translates provider-specific tool IDs into strict `^[a-zA-Z0-9_-]{1,64}$` tokens, and interrupted/mid-history orphan tool calls receive deterministic `[SYNTHETIC_REPAIR]` closures aligned 1-to-1 with `functionCall` turns.
3. **Empty `HTTP 200`, `429` & `MAX_TOKENS` Recovery (`CT-02`, `CT-27`)**: Retries empty 200 streams and transient `429`/`503`/`529` throttling with interruptible backoff, auto-compacts on context overflow, and seamlessly splices `MAX_TOKENS` continuations.

---

## 9. Native Crux-Evals (Crucible Assay) MCP & Hook Evaluator Suite

Crucible ships with **[`Crux-Evals`](src/crux_evals/)** built directly into `src/crux_evals/` (and exposed at `ush.evaluators` and `crux-evals`) so coding agents receive independent, epistemic-firewalled navigational feedback **without grading their own work**:

| Stage | Evaluator ID | MCP Tool Name | Paradigm | Independent Signal Hydrated From Disk |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | `trajectory_sentinel` | `eval_trajectory_compass` | **Deterministic** (`<2ms`) | Action/error/diff hashes $\to$ detects exact loops, error-signature stagnation, A$\leftrightarrow$B patch oscillation & read spirals |
| **Stage 1** | `clean_patch_and_contract` | `eval_invariant_and_patch` | **Deterministic** (`<20ms`) | `git diff HEAD` + Python AST + secret regexes $\to$ blocks test-assertion deletion (Anti-Goodhart), tautological asserts, secrets & unverified edits |
| **Stage 2** | `epistemic_grounding_auditor` | `eval_epistemic_grounding` | **Hybrid** | Read/edit ledger + raw command stdout vs. claims $\to$ enforces read-before-edit provenance & catches unverified completion claims |
| **Stage 2** | `blind_bars_judge` | `eval_blind_judge` | **Blind LLM Judge** | Anonymous `git diff` + raw test output (**Actor CoT stripped**) $\to$ position-swapped `0.0 \| 0.5 \| 1.0` `BARS` score with Conservative Bayesian LCB |
| **Stage 3** | `adversarial_falsifier` | `eval_adversarial_prober` | **Adversarial Breaker** | Touched `.py` ASTs + isolated `/tmp` sandbox $\to$ synthesizes boundary inputs (`[]`, `0`, `-1`, `{}`, `''`) and reports only reproducible crashes |

### Using `Crux-Evals` via Built-In Tools, `/evaluators`, `crux eval`, or Open MCP
```bash
# Inside crux interactive TUI: list, scaffold skill-style EVALUATOR.md lenses, or reload live
/evaluators list
/evaluators new sql_guard --md          # creates single-file ~/.ush/evaluators/sql_guard/EVALUATOR.md
/evaluators reload

# From any shell (works identically in compiled `crux`, `crux.pyz`, and source):
crux eval list
crux eval new sql_cost_guard --style md --when-to-use "When editing .sql queries"
crux eval run --lens blind_bars_judge
crux eval mcp                           # Open-Protocol JSON-RPC 2.0 MCP Server
```

---

## 10. 4-Gear Elastic Swarm & 3-Stage Blind Cascading Judge

Crucible scales compute elastically to match task complexity ([`src/ush/swarm/engine.py`](src/ush/swarm/engine.py), [`src/ush/swarm/judge.py`](src/ush/swarm/judge.py)) and integrates with [`Crux-Swarm`](https://github.com/jacobsohn-gif/Crux-Swarm):

- **Gear 1 (`G1` — Direct Fast Path)**: Single-agent execution in the active workspace for localized edits and queries.
- **Gear 2 (`G2` — Falsification Triad)**: Sequential **Planner $\to$ Executor $\to$ Adversarial Verifier** pipeline with automatic upshift on test failure.
- **Gear 3 (`G3` — Parallel Worktree Arena)**: Spawns isolated Git worktrees across diverse models/strategies in parallel and selects the winning patch via the Cascading Judge.
- **Gear 4 (`G4` — Stigmergic Blackboard Swarm, Peer CLI & ArgonClaw Bridge)**: Multi-agent DAG coordination over a SQLite WAL `StigmergicBlackboard` (`CONTRACT_DELTA`, `SPIKE`, `REQUEST_REVIEW`) with external peer-CLI delegation (`--peer-cli`) and optional remote cluster dispatch (`--argonclaw`).

### 3-Stage Blind Cascading Judge (`CT-18` – `CT-20`)
Every candidate patch in `G3`/`G4` is stripped of model identity (`clean_patch`) and evaluated through three stages:
1. **Stage A (Deterministic Execution Gate)**: Rejects patches that fail git apply, syntax compilation, security firewall checks, or mandatory test suites.
2. **Stage B (Position-Swapped `BARS` Rubric)**: Evaluates surviving patches twice `(A, B)` and `(B, A)` across a Behaviorally Anchored Rating Scale (`correctness`, `minimality`, `invariants`, `verification`) to eliminate positional bias.
3. **Stage C (Bradley-Terry Pairwise Solver & Quorum Ledger)**: Computes maximum-likelihood skill ratings and records the winning commit into an append-only, SHA-256 hash-chained SQLite `ConsensusLedger` (`CT-20`).

---

## 11. 3-Layer Security Firewall & Out-of-Tree Shadow Git

- **Layer 1 — Child Environment Scrubbing & Shell-Chain Unwrapping (`CT-12`)**: Strips cloud tokens, API keys, and SSH/GPG secrets from untrusted subprocesses and recursively unwraps `sudo`, `env`, `timeout`, `nohup`, and `sh -c` wrappers before policy evaluation.
- **Layer 2 — `.ushignore` (Negation-Safe) & Protected Control-Path Denylist (`CT-12`)**: Hard-blocks read/write/grep access to `.env*`, `*.pem`, `*.key`, credentials, user `.ushignore` patterns, and protected control files (`.git/hooks`, `.ush/policies.toml`, `.mcp.json`).
- **Layer 3 — Fingerprinted Untrusted Repository Quarantine (`CT-12`, `CT-14`, `CT-16`)**: Quarantines workspace-defined hooks, MCP servers, and auto-activated skills in untrusted repos (invalidating trust automatically if hook/policy fingerprints mutate) until authorized via `/trust grant`, while clamping extension rules so they never weaken user security policies.
- **Zero-Pollution Shadow Git Time Travel (`CT-09`)**: Checkpoints workspace state into an out-of-tree bare repository (`~/.ush/snapshots/<workspace_hash>/repo.git`) before every mutating turn—never touching your project's `.git/index` or reflog.

---

## 12. Verification Matrix (`CT-01` – `CT-28` & Adversarial Suites)

Run the complete `55`-test zero-dependency verification suite (`28` conformance gates in `26` test cases + `15` adversarial security/TUI/ACP replay invariants + `12` `Crux-Evals` MCP, isolation & skill-parity tests):

```bash
.venv/bin/python -W error::SyntaxWarning -W error::ResourceWarning -m unittest discover -s tests -v
```

| Gate / Suite | Subsystem Verified | Status |
| :--- | :--- | :---: |
| **`CT-01`** | Mid-session cross-provider rotation (Gemini $\to$ Claude $\to$ OpenAI $\to$ Local) | ✅ Pass |
| **`CT-02`** | Empty `HTTP 200` stream retry & `MAX_TOKENS` continuation splicing | ✅ Pass |
| **`CT-03`** | Turn-0 prefix KV-cache byte stability across dynamic skill & memory injections | ✅ Pass |
| **`CT-04`** | Tier-B malformed JSON/XML (`<crux_tool_call>` / `<tool_call>`) repair & retry loop | ✅ Pass |
| **`CT-05`** | 4-pass fuzzy `edit_file` engine (exact, CRLF/whitespace, indent-shifted, Unicode punctuation) | ✅ Pass |
| **`CT-06`** | Deadlock-free lexicographical `RwLock` acquisition & SHA-256 stale-read rejection | ✅ Pass |
| **`CT-07`** | Persistent `OSC 633` PTY state preservation (`cwd`/`venv`), prompt detection & flood cap | ✅ Pass |
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
| **`CT-21`** | Uncompacted `.trace.jsonl` thought ledger, `<think>` chunk splitter, search & Markdown export | ✅ Pass |
| **`CT-22`** | Gemini parallel tool-call `thoughtSignature` preservation & trailing user-turn enforcement | ✅ Pass |
| **`CT-23`** | Visual Inline-Dock TUI menus (`/`, `@`, `$`), `Ctrl+O` Threads Explorer & concurrent prompt queue | ✅ Pass |
| **`CT-24`** | Deadlock-free live `/steer`, geometric single-cell TUI glyphs & workspace-scoped `/resume` | ✅ Pass |
| **`CT-25`** | Strict `functionCall` $\leftrightarrow$ `functionResponse` bijection across mid-history orphan repairs | ✅ Pass |
| **`CT-26`** | Turn FSM rollback safety, strict role alternation, compaction & streaming loop guards | ✅ Pass |
| **`CT-27`** | Antigravity ACP dynamic models, Gemini Enterprise, Claude, OpenAI `/v1/responses`, LiteLLM & `429` recovery | ✅ Pass |
| **`CT-28`** | Zero-spam interactive tool approval dock deduplication & parallel `RwLock` prompt serialization | ✅ Pass |
| **Security & UX Suite** | Warm multi-turn ACP session reuse, `<=135 KB` tail-safe cold hydration, LaTeX/Markdown rendering, full-width `ask_user` wrapping/`Space` expand, PTY respawn, authenticated ACP & stateful fuzzing ([`tests/test_security_and_invariants.py`](tests/test_security_and_invariants.py)) | ✅ `15/15` |
| **Crux-Evals Suite** | Trajectory Sentinel, Anti-Goodhart `clean_patch`, Epistemic Grounding, Blind `BARS` Judge, `/tmp` Adversarial Falsifier, `artifact_enhancer`, `EVALUATOR.md` Skill Parity, Hardline Model Isolation, Utility Nudges & Open MCP ([`tests/test_crux_evals.py`](tests/test_crux_evals.py)) | ✅ `12/12` |

---

## 13. Codebase Map

### Core `Crucible` Runtime (`src/ush/` — `0` External Dependencies)
- **CLI, Visual Inline-Dock TUI, Markdown/LaTeX Renderer & ACP Server (`src/ush/cli/`)**
  - [`bin/crux`](bin/crux), [`bin/ush`](bin/ush) & [`pyproject.toml`](pyproject.toml) — Symlink-safe source launchers & PEP 621 package metadata (`crux`, `ush`, `crux-evals`).
  - [`src/ush/cli/tui.py`](src/ush/cli/tui.py) — Visual Inline-Dock TUI (`InlineDockCompositor`, `VisualTuiController`, `ThinkingThreadTracker`, `MenuOverlay`, DEC `?2026h` sync output, full-width `ask_user` & zero-spam approval cards, `Ctrl+O` Threads Explorer, Concurrent Prompt Queue).
  - [`src/ush/cli/markdown.py`](src/ush/cli/markdown.py) — Zero-dependency streaming & block Terminal Markdown, GFM box-table, syntax-highlighting, and font-verified LaTeX/Unicode math renderer.
  - [`src/ush/cli/main.py`](src/ush/cli/main.py) — CLI entrypoint, slash command handlers, headless `-p` runner, `crux trace`, `crux auth`, `crux swarm`, and `crux doctor`.
  - [`src/ush/cli/acp_server.py`](src/ush/cli/acp_server.py) — Authenticated loopback (`127.0.0.1`) & `stdio` JSON-RPC 2.0 Agent Client Protocol (ACP) server.
- **Universal Model Brain, Antigravity ACP Bridge & Thought Ledger (`src/ush/brain/`)**
  - [`src/ush/brain/agy_acp.py`](src/ush/brain/agy_acp.py) — Native Antigravity ACP (`agy acp`) live model discovery, warm multi-turn session reuse, `<=135 KB` tail-safe hydration, ephemeral evaluator session isolation, and `<crux_tool_call>` bidirectional tool transport.
  - [`src/ush/brain/evaluator_isolation.py`](src/ush/brain/evaluator_isolation.py) — Hardline Actor-vs-Evaluator isolation bridge (`tools=[]`, zero Actor history/CoT, cross-family preference, separate evaluator thought/analysis events).
  - [`src/ush/brain/providers.py`](src/ush/brain/providers.py) — Extended/interleaved thinking SSE & HTTP clients for Antigravity, Code Assist, Gemini Enterprise, Vertex AI, Claude, OpenAI (`/v1/responses` + Chat), LiteLLM, OpenRouter, and local endpoints.
  - [`src/ush/brain/wire.py`](src/ush/brain/wire.py) — 8-rule `transform_messages` cross-provider normalizer, XML/JSON tool repair, and `ToolIdBijectionMap`.
  - [`src/ush/brain/trace.py`](src/ush/brain/trace.py) — Append-only `0600` secret-redacted `ThoughtTraceLedger` (`.trace.jsonl`), landmark extractor, full-text search, and Markdown notebook exporter.
  - [`src/ush/brain/context_dag.py`](src/ush/brain/context_dag.py) — Branching `SessionDAG`, `LCA` rewind, observation tombstoning, dual-anchor `0.85` compaction, and `PopperianLoopGuard`.
  - [`src/ush/brain/model_tiers.py`](src/ush/brain/model_tiers.py) — `Tier S / A / B` capability profiles and cache-invariant system prompt builder.
- **Multi-Account Auth & Keychain Vault (`src/ush/auth/`)**
  - [`src/ush/auth/vault.py`](src/ush/auth/vault.py) — Ambient credential & `agy` discovery, Google Code Assist, Vertex ADC, OAuth 2.0 + PKCE/Device flows, workstation command-auth, and macOS Keychain / `0600` file vault.
- **Execution Kernel & Security Firewall (`src/ush/kernel/`)**
  - [`src/ush/kernel/fsm.py`](src/ush/kernel/fsm.py) — 6-phase `SessionRunner`, `StreamEvent` dispatcher, safe-boundary `enqueue_steer`, and `/btw` side-chat fork.
  - [`src/ush/kernel/tools.py`](src/ush/kernel/tools.py) & [`src/ush/kernel/patch.py`](src/ush/kernel/patch.py) — Essential-11 `RwLock` tools, 4-pass fuzzy `edit_file`, and `StaleReadTracker`.
  - [`src/ush/kernel/pty_manager.py`](src/ush/kernel/pty_manager.py) — Persistent stateful `OSC 633` PTY engine with prompt auto-detection and dead-shell auto-respawn.
  - [`src/ush/kernel/shadow_git.py`](src/ush/kernel/shadow_git.py) & [`src/ush/kernel/security.py`](src/ush/kernel/security.py) — Out-of-tree Shadow Git store, shell-unwrapping 3-layer security firewall, and fingerprinted `RepoTrustStore`.
- **Extensibility Platform, 4-Gear Swarm & Native Evaluators (`src/ush/platform/`, `src/ush/swarm/`, `src/crux_evals/`)**
  - [`src/ush/platform/skills.py`](src/ush/platform/skills.py), [`src/ush/platform/commands.py`](src/ush/platform/commands.py), [`src/ush/platform/hooks_memory_ext.py`](src/ush/platform/hooks_memory_ext.py) — 4-tier `SKILL.md` loader, slash command registry, 9-event hooks, and JIT workspace memory.
  - [`src/ush/swarm/engine.py`](src/ush/swarm/engine.py) & [`src/ush/swarm/judge.py`](src/ush/swarm/judge.py) — 4-Gear Elastic Swarm, peer CLI bridge, SQLite WAL `StigmergicBlackboard`, 3-Stage Cascading Judge, and SHA-256 `ConsensusLedger`.
  - [`src/crux_evals/`](src/crux_evals/) & [`src/ush/evaluators/`](src/ush/evaluators/) — Native `Crux-Evals` (Crucible Assay) 5-module cascade (`m01`–`m05`), auto-discovery `EvaluatorRegistry`, `cli.py` scaffolder, `cli_hook.py`, and `mcp_server.py`.
- **Build, Release & Test Suites (`scripts/`, `releases/v0.1/`, `tests/`)**
  - [`scripts/build_release.py`](scripts/build_release.py) & [`Makefile`](Makefile) — Reproducible compiler pipeline producing native `arm64` Mach-O `crux`, portable `crux.pyz`, clean `crucible-0.1-src.tar.gz`, and `SHA256SUMS.txt`.
  - [`tests/test_conformance_ct01_to_ct20.py`](tests/test_conformance_ct01_to_ct20.py), [`tests/test_security_and_invariants.py`](tests/test_security_and_invariants.py), [`tests/test_crux_evals.py`](tests/test_crux_evals.py) — `55` deterministic conformance, security, TUI, ACP, isolation, and evaluator tests.

---

## License

Apache-2.0. Built and maintained by Stephen Jacobsohn ([@jacobsohn-gif](https://github.com/jacobsohn-gif)).
