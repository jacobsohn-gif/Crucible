# Crucible (`crux`) Harness Upgrades: Architectural Rationale (What & Why)

This document records the engineering upgrades implemented across the Crucible (`crux` / `ush` + `crux_evals`) harness, pairing **what** changed in code with **why** each change was necessary—grounded in concrete failure modes observed during long-horizon multi-file, WebGL/UI, and multi-turn engineering sessions.

---

## Executive Summary Matrix

| # | Upgrade | Primary Modules | Failure Mode Eliminated ("Why") |
| :--- | :--- | :--- | :--- |
| **1** | **First-Class `render_ui` Web/WebGL Renderer** | `src/ush/kernel/ui_renderer.py`, `tools.py`, `security.py` | Eliminates black WebGL screenshots (`SharedImageManager::ProduceSkia` / cleared drawing buffers), infinite `requestAnimationFrame` hangs, `file://` CORS blocks, and ad-hoc shell Chrome flags. |
| **2** | **Stateless `run_command` + Chunked `write_file(mode='append')`** | `src/ush/kernel/tools.py`, `security.py`, `state.py` | Eliminates macOS interactive PTY `MAX_CANON` (`1,024`-byte) heredoc input truncation, open-`stdin` hangs (`cat`, `git`, test runners), and oversized single-JSON tool payloads on large artifacts. |
| **3** | **Complexity-Triggered Auto Pre-Ship Cold-Eyes Gate** | `src/ush/kernel/fsm.py` | Solves the **Confirmation-Bias Catch-22**: an actor convinced its artifact is finished often forgets to invoke `consult_evaluator` voluntarily. `STOP_HOOKS` now automatically runs a clean-room pre-ship pass on UI deliverables or multi-file (`>=2`) edits. |
| **4** | **Contrastive `tournament` Clean-Room Evaluator** | `src/crux_evals/core/workflows.py`, `mcp_server.py`, `registry.py`, `tools.py` | Single-artifact LLM grading suffers from calibration drift; side-by-side contrastive ranking of `>=2` candidate designs/patches in zero-history isolation surfaces concrete algorithmic and edge-case trade-offs. |
| **5** | **Thread-Safe `call_id` Telemetry Isolation** | `src/ush/platform/hooks_memory_ext.py`, `tools.py` | Fixes a `ThreadPoolExecutor` race where parallel `read_file` calls clobbered `HookSystem._last_tool_args['read_file']`, corrupting `read_files` provenance and firing false `trajectory_sentinel` loop warnings. |
| **6** | **Batch-Aware `TrajectorySentinel` (`batch_has_mutation`)** | `src/crux_evals/modules/m01_trajectory_sentinel/module.py`, `protocol.py`, `state.py`, `tools.py` | Suppresses false `"8 consecutive read/search calls"` warnings when the actor has *already* emitted an `edit_file` / `write_file` / `run_command` in the exact same tool batch behind a leading `read_file`. |
| **7** | **Multi-Generation `<verified_environment_facts>` Compaction Ledger** | `src/ush/brain/context_dag.py` | Prevents post-compaction amnesia where an agent re-discovers missing packages (`No module named ...`), sandbox boundaries, or verified test commands after `0.85` context compaction. |
| **8** | **ACP Host-Injected Tool Suppression (`default_api:manage_task`)** | `src/ush/brain/agy_acp.py` | Prevents bridged models inside `agy acp` from co-emitting host-injected native function declarations alongside `<crux_tool_call>`, avoiding wasted round-trips. |

---

## Detailed Engineering Breakdown: What Changed & Why

### 1. First-Class `render_ui` Tool (`src/ush/kernel/ui_renderer.py`, `src/ush/kernel/tools.py`)

#### Why
When agents build interactive HTML/CSS/SVG dashboards, D3 charts, or Three.js/WebGL 3D scenes and attempt to visually verify them via raw shell commands (`google-chrome --headless --screenshot ...`), four failure modes repeatedly occur:
1. **Black WebGL Canvases on Headless Chromium:** Modern macOS Chromium (`SkiaGraphite` / `SharedImageManager::ProduceSkia`) composites hardware-less WebGL canvases as solid black rectangles unless ANGLE SwiftShader (`--use-gl=angle --use-angle=swiftshader --enable-unsafe-swiftshader`) is paired with `preserveDrawingBuffer: true` and a settled frame snapshot.
2. **Infinite `requestAnimationFrame` Loops:** Standard 60fps `requestAnimationFrame(animate)` loops prevent headless Chrome's page-load quiescence detector from ever firing, hanging the shell tool until timeout.
3. **`file://` Origin & ES Module CORS Errors:** Opening local `.html` files over `file://` blocks ES module imports, `fetch()`, and canvas `toDataURL()` taint rules.
4. **Silent JS Runtime Exceptions:** A static HTML check cannot see `Uncaught ReferenceError: fooMode is not defined` thrown inside an animation frame or button callback.

#### What Changed
- Added **`src/ush/kernel/ui_renderer.py`** and registered **`render_ui`** as a first-class read-only tool (`PolicyEngine.READ_ONLY_TOOLS` — safe even in `/plan` mode, where screenshots write strictly to `.ush/renders/` outside the source tree).
- **Ephemeral Loopback Server + Non-Mutating In-Memory Probe Shim:** `render_ui` serves workspace files over a random `127.0.0.1:<ephemeral_port>` HTTP server gated by `SecurityFirewall.resolve_and_validate(..., write=False)` (returning `HTTP 403` on any `../` or `.ushignore` traversal). For `.html`/`.htm` responses, it injects a zero-disk-mutation `<head>` shim that:
  - Forces `{ preserveDrawingBuffer: true }` on `webgl` / `webgl2` contexts,
  - Caps `requestAnimationFrame` at a deterministic frame budget (`max_raf_frames=12`) so animations render initial geometry and then settle cleanly,
  - Mirrors active WebGL canvases to crisp `<img>` overlays via `toDataURL('image/png')` after `wait_ms`,
  - Captures `window.onerror`, `unhandledrejection`, and `console.error` diagnostics from Chromium stderr (`_CONSOLE_ERR_RE`).
- **Strict SSRF Firewall (`validate_render_target`):** Only workspace-validated local files (with optional `?query` parameters such as `?cam=overview`) or explicit loopback URLs (`http://127.0.0.1:<port>` / `http://localhost:<port>`) are permitted. External domains, `https://`, RFC1918 LANs, and cloud metadata IPs (`169.254.169.254`) raise `PermissionError` immediately.
- **Direct Multimodal Return:** Returns `CanonicalPart` with `mime_type="image/png"` and `base64_data` alongside console error counts so both the Actor and `consult_evaluator(evaluator="cold_eyes_review", files=[..., ".ush/renders/..."])` inspect the real rendered pixels.

---

### 2. Stateless `run_command` & Chunked `write_file(mode="append")` (`src/ush/kernel/tools.py`)

#### Why
- **Interactive PTY Limits vs. One-Shot Commands:** `exec_pty` maintains a persistent stateful pseudoterminal (`cwd`, shell vars, interactive REPLs, background daemons). However, macOS/POSIX PTYs in canonical mode enforce a **`1,024`-byte line input buffer (`MAX_CANON`)**—meaning multi-KB heredocs (`python - << 'PY' ...`) silently truncate and hang waiting for `PY`. Furthermore, commands that read `stdin` if attached to a TTY (`cat`, `git`, `ssh`, certain test runners) can hang waiting for `EOF`.
- **Single-Call JSON Escaping Overhead on Large Files:** Authoring a `45–80 KB` self-contained `.html` or simulation module inside a single JSON-encoded `write_file` call doubles escaping overhead and risks provider output-token truncation.

#### What Changed
- Added **`run_command(command, cwd?, timeout_seconds?)`**:
  - Spawns `["/bin/bash", "--noprofile", "--norc", "-c", command]` (fallback `/bin/sh`) directly with `stdin=subprocess.DEVNULL`, `stdout=PIPE`, `stderr=PIPE`, `start_new_session=True`, and `firewall.scrubbed_env()`.
  - Enforces the **exact same** `PolicyEngine` catastrophic-command & 4-tier TOML rules as `exec_pty` (`cmd_str` gated identically in `ToolRegistry._run_single`, denied in `/plan` mode).
  - Kills the entire child process group (`os.killpg(..., SIGKILL)`) on `timeout_seconds` or `cancel_event`, draining and closing pipes deterministically (`0` `ResourceWarning`s under `python -W error::ResourceWarning`).
  - Added `node --check` to `VERIFICATION_CMD_RE` in `src/crux_evals/core/state.py` so JS syntax checks via `run_command` count toward empirical verification.
- Extended **`write_file(path, content, mode="overwrite" | "append")`**:
  - `mode="append"` verifies `StaleReadTracker.check_fresh` against current disk bytes (blocking silent corruption if the file was modified externally), concatenates `existing + raw`, writes atomically via `atomic_write_bytes`, and updates the SHA-256 read ledger—allowing clean 15–20 KB multi-chunk file authoring without temporary patch scripts.

---

### 3. Complexity-Triggered Auto Pre-Ship Cold-Eyes Gate (`src/ush/kernel/fsm.py`)

#### Why
Prompt-level instructions (`"call consult_evaluator before shipping"`) suffer from a fundamental **Confirmation-Bias Catch-22**: the exact moment an actor has 40k tokens of sunk-cost reasoning convincing itself an implementation is complete is the moment it feels *least* uncertain—and therefore skips calling `consult_evaluator`.

At the same time, forcing an LLM critic call on every trivial 1-line edit or Q&A turn would waste latency and tokens.

#### What Changed
- In `SessionRunner.run_turn` (`Phase 5: STOP_HOOKS`), `crux` tracks `turn_modified_files` and `did_consult_eval_in_turn`.
- When the actor attempts to `STOP` **without** having called `consult_evaluator` during the turn, and **either**:
  1. `len(turn_modified_files) >= 2` (multi-file change), **or**
  2. Any modified file is a UI/visual deliverable (`.html`, `.htm`, `.jsx`, `.tsx`, `.vue`, `.svg`),
  `SessionRunner` automatically executes a **clean-room `cold_eyes_review`** (`clean_patch_and_contract` + `adversarial_falsifier` + `artifact_enhancer` + any path-scoped deterministic plugins; `<15ms`, `$0` token cost by default, or full cross-family LLM judge when `auto_cold_eyes_llm=True` / `CRUX_AUTO_COLD_EYES_LLM=1`).
- If `status == "FINDINGS"` (e.g., leaked harness scorecards, `TODO` placeholders, dead UI handlers, missing responsive viewport meta, AST/contract violations, or `/tmp` boundary crashes), `SessionRunner` emits `StreamEvent(meta={"event": "auto_pre_ship_cold_eyes"})`, injects `[AUTO_PRE_SHIP_COLD_EYES]` once (`complementary_stop_nudges += 1`), and loops back to `PRE_INVOCATION` so the actor fixes the defects before the user ever sees them. If `PASS`, the turn finishes immediately with zero extra LLM calls.

---

### 4. Contrastive `tournament` Clean-Room Workflow Mode (`src/crux_evals/core/workflows.py`, `mcp_server.py`)

#### Why
Evaluating a single design or patch in isolation (`0.0–1.0` scalar grading) is vulnerable to rubric leniency: any plausible-looking solution often scores "good enough" until an edge case breaks it. Empirically (AlphaCode / AlphaEvolve / pairwise RLHF), **contrastive comparison** between `>=2` explicit candidates forces the judge to identify concrete invariants where Option A fails and Option B holds.

#### What Changed
- Added **`tournament`** (`workflow_phase: "contrastive_selection"`) to `WORKFLOW_MODES`, `consult_evaluator`, `EvaluatorRegistry.initial_catalog_block`, and `CruxEvalsMCPServer`.
- Accepts candidate files (`files=["opt_a.py", "opt_b.py"]`) and/or architectural candidate descriptions (`goal` / `proposed_claims` hydrated into `<candidate_proposals>`), runs Stage-1 deterministic & `/tmp` falsification checks across any on-disk candidate files first, and runs a zero-history contrastive judge returning strict JSON (`winner`, `summary`, and `<=5` falsifying trade-offs in `findings`).

---

### 5. Thread-Safe `call_id` Telemetry Isolation (`src/ush/platform/hooks_memory_ext.py`, `src/ush/kernel/tools.py`)

#### Why
`ToolRegistry.dispatch_batch` executes contiguous read-only tools (`read_file`, `grep_files`, `glob_files`) concurrently via `ThreadPoolExecutor`. Previously, `HookSystem.before_tool(tool_name, args)` stored `self._last_tool_args[tool_name] = dict(args)` keyed only by `tool_name`!

When 4 parallel `read_file` calls (`a.py`, `b.py`, `c.py`, `d.py`) entered `before_tool` concurrently:
1. Thread 4 overwrote `self._last_tool_args["read_file"] = {"path": "d.py"}` before Threads 1–3 reached `after_tool`.
2. `after_tool` recorded `"d.py"` **4 times** in `SessionTelemetry`, losing `a.py`, `b.py`, and `c.py` from `session_telemetry.read_files` (breaking read-before-edit provenance).
3. All 4 calls received the **exact same `action_signature`**, so if those 4 parallel reads hit errors (e.g., probing 4 distinct missing files), `TrajectorySentinel` falsely fired `"Tool call read_file repeated 3 consecutive times"`!

#### What Changed
- Added `self._in_flight_tool_args: dict[str, dict[str, Any]]` keyed by unique `call_id` inside `HookSystem` (protected by `self._telemetry_lock`).
- `ToolRegistry._run_single` passes `call_id=cid` when `getattr(before_hook, "_accepts_call_id", False)` is `True` (preserving 100% backward compatibility with 2-arg custom hooks in conformance tests), and `HookSystem.after_tool` pops `self._in_flight_tool_args.pop(result_part.tool_call_id)` cleanly.

---

### 6. Batch-Aware `TrajectorySentinel` (`src/crux_evals/modules/m01_trajectory_sentinel/module.py`)

#### Why
Models frequently emit `[read_file("foo.py"), edit_file("foo.py", ...)]` in a single assistant turn. Because `dispatch_batch` executes index `0` (`read_file`) before index `1` (`edit_file`), `after_tool` on `read_file` saw an 8th consecutive read in `tool_history` and injected a false `"8 consecutive read/search calls executed without narrowing to an action"` warning—even though `edit_file` was already queued and executing milliseconds later in the same batch!

#### What Changed
- `ToolRegistry.dispatch_batch` computes `batch_has_mutation` across the full batch up front and passes it to `HookSystem.after_tool` -> `EvaluationFrame.batch_has_mutation`.
- `TrajectorySentinelEvaluator` (Check 5) suppresses the read-spiral warning whenever `safe_frame.batch_has_mutation` is `True`, and recognizes `render_ui` and `activate_skill` in `READ_ONLY_TOOLS`.

---

### 7. Multi-Generation `<verified_environment_facts>` Compaction Ledger (`src/ush/brain/context_dag.py`)

#### Why
During long sessions that cross the `0.85` context compaction threshold (`SessionDAG.maybe_compact`) once or multiple times, prose summaries often compress away negative environment constraints (`ModuleNotFoundError: No module named 'playwright'`, `Sandbox boundary violation`, missing CLI binaries) or exact passing test commands—causing the post-compaction model to waste turns re-running commands that already failed 30 steps ago.

#### What Changed
- Added **`extract_verified_environment_facts(messages)`** in `src/ush/brain/context_dag.py`:
  - Harvests empirical environment constraints (`ModuleNotFoundError`, `ImportError`, `command not found`, `Sandbox boundary violation`, `[POLICY_DENIED]`, missing headless browser binaries) and verified passing test/check commands (`pytest`, `unittest`, `cargo test`, `npm test`, `node --check`, `ruff check`, `mypy`, `tsc`, `go test`).
  - Re-hydrates any existing `<verified_environment_facts>` block from prior `<epistemic_compaction_handoff>` summaries so pinned facts survive **arbitrarily many** compaction generations—even when a custom LLM `summarizer` callable is used.

---

### 8. Antigravity ACP Host-Tool Suppression (`src/ush/brain/agy_acp.py`)

#### Why
Even when `crux-llm-pipe.md` declares `tools: []`, the `agy acp` host layer exposes internal task-management function declarations (such as `default_api:manage_task`) to the underlying model. Occasionally, models co-emitted native `manage_task` calls alongside `<crux_tool_call>` XML tags, triggering an extra internal ACP round-trip.

#### What Changed
- Hardened both `_CRUX_PIPE_AGENT_MD` and `_render_tool_declarations` (`<crux_tool_protocol>`) in `src/ush/brain/agy_acp.py` to explicitly forbid calling host-injected function declarations (`default_api:manage_task` / `manage_task`) and emit strictly `<crux_tool_call>` XML blocks.

---

## Verification

All 8 upgrades are deterministically tested in [`tests/test_harness_upgrades.py`](../tests/test_harness_upgrades.py) alongside the full conformance, security, and evaluator suites (`64` tests total, zero warnings under `-W error::ResourceWarning`):

```bash
.venv/bin/python -W error::ResourceWarning -m unittest discover -s tests -v
```
