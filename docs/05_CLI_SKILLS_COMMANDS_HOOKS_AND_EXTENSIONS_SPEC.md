# `05` — Gemini-CLI-Class Developer Platform: Skills, Slash Commands, Hooks, Memory, Policies & Extensions

> Companion to [MASTER_IMPLEMENTATION_PLAN.md](../MASTER_IMPLEMENTATION_PLAN.md). Distilled directly from our source-level teardown of **Gemini CLI**, **Codex CLI**, **Qwen Code**, **OpenCode v2**, **Kimi Code**, and **Pi** (`14_CLI_ARCHITECTURE_SKILLS_SLASH_COMMANDS_AND_EXTENSIBILITY.md`).

---

## 1. Platform Design Goal: Tri-Modal CLI + Drop-In Ecosystem Compatibility

`USH` is not a library or bare headless loop—it ships as a **real standalone CLI executable (`ush`) installed straight onto your `$PATH`** (just like `gemini`, `claude`, `codex`, or `opencode`), running one unified engine across **three entry modes**:
1. **Interactive Rich Terminal TUI (`ush` in any repo folder)**: Full terminal coding interface with streaming markdown + syntax-highlighted unified diffs, fuzzy `/command` + `@file` + `$skill` autocomplete popups, `!cmd` instant shell toggle, `Ctrl+V` clipboard image paste, vim/multiline composer, inline tool-approval cards, `/auth` account switcher, and a live status bar (`model | profile | tier | ctx% | cache% | gear G1..G4 | cost`).
2. **Headless Unix / CI Executable (`ush -p "..." --output-format text|json|stream-json`)**: Composable stdin/stdout pipelines (`git diff | ush -p "/review"`), `ush auth login|status|switch`, `ush bench`, deterministic exit codes, and bounded `--max-turns` / `--max-budget-usd`.
3. **ACP / NDJSON Daemon (`ush serve --acp | --jsonrpc`)**: Plugs `ush` directly into **VS Code, Zed, Neovim, or JetBrains** as an IDE agent panel *and* serves distributed Swarm Peer Nodes over the exact same protocol.

**Zero-Migration Ecosystem Compatibility**: Out of the box, `ush` discovers existing `.agents/skills/`, `.gemini/skills/`, `.claude/skills/`, `AGENTS.md`, `GEMINI.md`, and `CLAUDE.md` files so users bring their existing skills, rules, and extensions on Day 1 with zero rewriting!

---

## 2. The 3-Stage Universal Skills Engine (`5` to `500+` Skills, Any Model Size)

### 2.1 Discovery Scopes & Trust Precedence (Lowest $\to$ Highest)
1. **Built-in Skills** (`builtin:` — `/init`, `/review`, `/learn`, `/grill-me`, `/dissect`, `/swarm-spike`)
2. **Active Extensions** (`ext:<extName>:<skillName>`)
3. **User Global Skills** (`~/.ush/skills/`, `~/.agents/skills/`, optional compat `~/.gemini/skills/`, `~/.claude/skills/`)
4. **Workspace Skills** (`.ush/skills/`, `.agents/skills/`, `.gemini/skills/`, `.claude/skills/`) — **loaded ONLY when `workspace.is_trusted == true`**.

### 2.2 `SKILL.md` Frontmatter Schema (Colon-Safe Parser)
Uses standard YAML parsing with a Gemini-CLI-style indentation-aware line fallback so unquoted colons in `description:` never break loading:

```yaml
---
name: security-migration               # Validated slug: ^[a-z0-9-]+$, <=64 chars
description: Audit and migrate auth... # Concise summary (<=1024 chars)
when_to_use: Trigger when touching OAuth, JWT, session cookies, or RBAC
argument-hint: "<target-dir-or-pr>"
paths: ["**/auth/**", "**/*.jwt.*"]    # Optional: lazy path-glob activation (Qwen pattern)
context: inline                        # 'inline' (default) | 'fork' (clean subagent context!)
model: inherit                         # 'inherit' | 'fast' | '<provider>:<model>'
allowed-tools: ["exec_pty(pytest *)"]  # Additive session permission grants while active
disable-model-invocation: false        # true = manual /security-migration only (deploy/release)
user-invocable: true                   # false = hidden from '/' menu; model-only knowledge
---
```

### 2.3 Three-Layer Progressive Disclosure (Crucial for KV-Cache & Small Tier-B Models!)
1. **Path-Gated Filtering**: Skills with `paths:` stay completely out of the model's catalog until `read_file`/`edit_file`/`grep_files` touches a symlink-resolved matching path.
2. **BM25 + LRU Budgeted Delta Selector**:
   - Enforces a tier-scaled `SkillCatalogTokenBudget` (`Tier S`: `3,000` tok; `Tier A`: `1,500` tok; `Tier B`: `600` tok / top-8 BM25 matches to user prompt!).
   - Tracks `CatalogSnapshot { name -> (sha256, shown) }`: newly relevant skills mid-session inject *only* a tiny `<skills_delta>` block inside `<dynamic_instructions>`, keeping Turn 0's KV cache 100% immutable!
3. **Sandboxed `activate_skill` Tool + Dual User Triggers (`/skill` & `$skill`)**:
   - `activate_skill(name)` uses a dynamic JSON Schema `enum` of visible skill names so even 14B local models never hallucinate skill IDs.
   - For non-builtin skills, prompts/logs the resource tree first, **dynamically adds `dirname(skill.path)` to the read-sandbox allowlist** (so bundled `references/`, `examples/`, and `scripts/` outside the repo become readable!), sets `USH_SKILL_ROOT`, and returns `<activated_skill><instructions>...</instructions><available_resources>...</available_resources></activated_skill>`—or executes in an isolated subagent if `context: fork`!
   - Users can invoke any `user-invocable` skill via `/skill-name [args]` OR inline `$skill-name` mentions (`"Run $dissect then $security-migration on src/auth"`).

---

## 3. Unified 4-Loader Slash Command System & Safe Macro DSL

### 3.1 Four Concurrent Command Loaders + `SlashCommandResolver`
1. **`BuiltinCommandLoader`** — Always retains primary bare `/name` (untrusted repos/extensions can **never** shadow built-ins!).
2. **`FileCommandLoader`** — Loads both `.toml` (Gemini CLI style) and `.md` (Claude/OpenCode style) macro files from `~/.ush/commands/` and trusted `.ush/commands/`, mapping nested folders to namespaces (`commands/git/fixup.toml` $\to$ `/git:fixup`).
3. **`SkillCommandLoader`** — Auto-exposes every `user-invocable` skill as `/<skill-name> [args]`.
4. **`McpPromptLoader`** — Auto-exposes connected MCP server prompts as `/<server>:<prompt> --arg=val` with generated `help` subcommands.
*(Collisions between non-builtins deterministically prefix both to `<source>.<name>` and surface a non-fatal notice in `/commands`.)*

### 3.2 Complete Built-In Slash Command Roster

| Category | Built-In Commands | Execution Model |
| :--- | :--- | :--- |
| **Modes, Planning & Autonomy** | `/plan` (read-only/spike Plan Mode), `/grill-me` (Socratic spec stress-test), `/goal <objective>` (bounded multi-hour loop), `/effort <low\|med\|high>` | FSM state transition (`0` tokens to toggle) / Epistemic prompt turn |
| **Swarm, Judge & Arena** | `/swarm [auto\|g1..g4] <task>` (4-Gear Elastic Swarm), `/arena --models m1,m2 <task>` (parallel worktree race + Bradley-Terry judge), `/review [base]` (role-stripped cross-family blind diff critic) | Layer-3 Swarm & Stage B/C Judge dispatch |
| **Session Tree, Rewind & Side-Chat** | **`/btw <question>`** (100%-KV-cached zero-history side Q&A while agent works!), `/restore` / `/rewind` (`code+chat`, `code-only`, `chat-only` via Shadow Git + DAG), `/tree` & `/fork`, `/resume`, `/compact [focus]`, `/clear`, `/export` | Instant client/DAG/Shadow-Git action or forked `NO_TOOLS` cache read |
| **Models & Routing** | `/model [provider:id]` (live mid-session hot-swap via `transformMessages`!), `/profile` (Tier S/A/B inspection & swarm role map), `/stats` (tokens, cache `%`, cost, 6-KPI counters) | Instant client state update (`0` LLM tokens) |
| **Extensibility & Memory** | `/init` (generate `AGENTS.md`/`USH.md`), `/memory [show\|add\|inbox\|refresh]`, `/learn` (extract reusable skill/memory patch from current session), `/skills`, `/commands`, `/hooks`, `/mcp`, `/extensions`, `/reload` | Client UI / `skill-extraction-agent` inbox patcher |
| **Safety & Diagnostics** | `/permissions` (`plan` / `default` / `auto-edit` / `yolo`), `/policies`, `/trust`, `/diff`, `/doctor` (toolchain, PTY, auth, LSP & sandbox health check) | Instant deterministic inspection |

### 3.3 Injection-Safe 3-Stage Macro Interpolation Pipeline
Every `.toml` / `.md` custom command passes through:
1. **`AtFileProcessor` (`@{path}`)**: Expands workspace text, images, and PDFs into native multimodal message parts while blocking `.gitignore`/`.ushignore` secret files.
2. **`ShellProcessor` (`!{cmd {{args}}}`)**: Substitutes **shell-escaped** `{{args}}` inside `!{...}` (blocking shell injection), evaluates the resolved command through `PolicyEngine` (`deny` aborts; `ask_user` prompts confirmation; `allow` runs in bounded read-mostly shell), and splices stdout.
3. **`ArgumentProcessor` (`{{args}}` / `$ARGUMENTS` / `$1..$9`)**: Substitutes raw prose arguments outside shell blocks (appending trailing user input automatically if no placeholder is written).

---

## 4. 9-Event Lifecycle Hooks, `policies.toml`, Memory Inbox & Extensions

1. **9-Event Hook Bus (`hooks.json` — Host-Trusted Only for Workspace Scope)**:
   - Events: `SessionStart`, `UserPromptSubmit`, **`BeforeToolSelection`** *(lets Tier-B adapters & skills filter visible tool schemas per step)*, **`BeforeModel`**, **`AfterModel`**, `BeforeTool` *(can `allow` / `deny` / `ask` / rewrite args)*, `AfterTool` *(can append lint/test `additionalContext`)*, `PreCompress`, and `Stop` *(can veto premature completion with `exit 2`)*.
2. **4-Scope Memory + JIT Subdirectory Load + `/memory inbox` Learning**:
   - Auto-discovers global $\to$ extension $\to$ project-root $\to$ **JIT package-subdirectory** `AGENTS.md`/`GEMINI.md`/`CLAUDE.md`/`USH.md` files (inode-deduplicated + `@relative/import.md` include support).
   - Offline `/dream` and `/learn` extractors treat transcripts strictly as **untrusted read-only data**, redact secrets, maintain `MEMORY.md` as a compact index pointing to on-demand topic files, and stage unified `.patch` files in `.inbox/` for 1-click `/memory inbox` human approval.
3. **Privilege-Clamped Policy Engine (`policies.toml`)**:
   - Evaluates priority-sorted rules across 4 precedence tiers (`Enterprise Admin` > `User` > `Workspace` > `Extension`), parsing shell chains (`&&`, `|`, `sudo`, `env`) and MCP wildcards (`mcp_server_*`). **Extension rules are hard-clamped**: they may emit `deny` or `ask_user`, never `allow` or `yolo`.
4. **One-File Extension Manifest (`ush-extension.json` + `gemini-extension.json` importer)**:
   - Bundles `skills/`, `commands/`, `agents/`, `hooks/hooks.json`, `policies/policies.toml`, `mcpServers`, and `USH.md` into a single `ush extensions install <git-url-or-path>` package!
