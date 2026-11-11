"""3-Stage Universal Skills Engine (`SkillManager`).

Implements CT-14:
1. 4-Tier Trust-Gated Discovery (`builtin < extension < user < trusted_workspace`) across `.ush/skills`,
   `.agents/skills`, `.gemini/skills`, and `.claude/skills`.
2. Colon-safe YAML frontmatter parser (unquoted colons in `description:` never break loading!) + strict slug validator.
3. Qwen-style `paths:` glob lazy gating + Codex-style BM25/LRU token-budgeted `<skills_delta>` tail updates
   (keeps Turn-0 KV cache 100% immutable even with 500+ skills on Tier-B 32k models!).
4. `activate_skill` dynamic read-sandbox grant (`firewall.grant_read_directory(skill.root_dir)`) + `$skill` triggers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import fnmatch
import math
import pathlib
import re
from typing import Iterable

from ush.core.types import sha256_hex
from ush.kernel.security import SecurityFirewall

SKILL_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
FRONTMATTER_RE = re.compile(r"\A---\r?\n(.*?)\r?\n---\r?\n?(.*)\Z", re.DOTALL)
WORD_RE = re.compile(r"[a-z0-9]{2,}")
DOLLAR_SKILL_RE = re.compile(r"(?<![A-Za-z0-9_/])\$([a-z0-9][a-z0-9-]{1,63})\b")


@dataclass
class SkillDefinition:
    name: str
    description: str
    body: str
    tier: str  # builtin | extension | user | workspace
    location: pathlib.Path | None = None
    when_to_use: str = ""
    argument_hint: str = ""
    paths: tuple[str, ...] = ()
    context_mode: str = "inline"  # inline | fork
    model_override: str = "inherit"
    allowed_tools: tuple[str, ...] = ()
    disable_model_invocation: bool = False
    user_invocable: bool = True

    @property
    def fingerprint(self) -> str:
        return sha256_hex(f"{self.name}|{self.description}|{self.when_to_use}|{self.paths}")[:16]

    def summary_line(self) -> str:
        wtu = f" (When: {self.when_to_use})" if self.when_to_use else ""
        return f"- {self.name}: {self.description}{wtu}"

    def approx_catalog_tokens(self) -> int:
        return max(8, len(self.summary_line()) // 4)


BUILTIN_SKILLS: tuple[SkillDefinition, ...] = (
    SkillDefinition(
        name="dissect",
        description="Surgically deconstruct a complex problem into objectives, hidden assumptions, invariants, and failure modes.",
        when_to_use="Before implementing complex algorithms or unclear architectural specs.",
        tier="builtin",
        body=(
            "1. List explicit functional objectives.\n"
            "2. Surface unstated assumptions & edge boundaries (concurrency, empty inputs, auth, scale).\n"
            "3. Define strict testable invariants (`INV-1..k`).\n"
            "4. Propose the smallest empirical reproduction/test before touching implementation code."
        ),
    ),
    SkillDefinition(
        name="grill-me",
        description="Stress-test a design or plan Socratically using concrete code facts and multiple-choice tradeoffs.",
        when_to_use="When user asks to refine requirements or challenge architectural decisions.",
        tier="builtin",
        body=(
            "Inspect relevant repository files first, identify top 2-3 high-consequence design tradeoffs, "
            "and resolve them succinctly with `ask_user` (putting your recommended option first) before coding."
        ),
    ),
    SkillDefinition(
        name="swarm-spike",
        description="Execute fast throwaway prototype spikes in isolated worktrees before freezing shared interfaces.",
        when_to_use="On multi-module refactors or unfamiliar APIs where static speculation risks interface mismatches.",
        tier="builtin",
        context_mode="fork",
        body=(
            "Prototype minimal vertical slices, record empirical signature facts, and emit a 3-line `CONTRACT_DELTA` "
            "via `swarm_dispatch` so sibling workers code against proven reality."
        ),
    ),
    SkillDefinition(
        name="crux-evals",
        description="Run zero-history clean-room evaluators (`consult_evaluator`) at Pre-Build (`challenge_design`), Pre-Ship (`cold_eyes_review`), and When-Stuck (`break_loop`) with zero briefing tax.",
        when_to_use="Before >250L builds (`challenge_design`), right before shipping non-trivial code/UI/screenshots (`cold_eyes_review`), or after 2 failed attempts (`break_loop`).",
        tier="builtin",
        body=(
            "`consult_evaluator(evaluator, files, goal)` executes in an isolated zero-history clean-room unpolluted by your build context "
            "and returns `PASS` or at most 3–5 surgical file/element fixes (zero essays):\n"
            "- `challenge_design` (Pre-Build): pass target files + 1-line `goal` right before large writes/refactors to catch schema/state/layout traps.\n"
            "- `cold_eyes_review` (Pre-Ship): pass raw source/HTML files + verification screenshots (`.png`) right before your final answer.\n"
            "- `break_loop` (When Stuck): call after 2 failed attempts on the same bug/render issue for <=3 orthogonal root-cause hypotheses & probes.\n"
            "- Custom plugins from `<available_evaluators>` can also be passed by exact ID. Act silently without exposing evaluator metadata."
        ),
    ),
    SkillDefinition(
        name="create-evaluator",
        description="Author custom plug-and-play evaluators just like skills (`EVALUATOR.md` or `evaluator.yaml` + `module.py`) with strict model isolation.",
        when_to_use="When the user asks to add a new evaluator, audit lens, or domain-specific rubric.",
        tier="builtin",
        body=(
            "Create custom evaluators just like skills in `~/.ush/evaluators/<id>/` (user-global) or trusted `<workspace>/.crux_evals/plugins/<id>/`:\n"
            "1. Declarative Rubric Lens (zero Python — create `EVALUATOR.md`):\n"
            "   Frontmatter keys: `evaluator_id`, `name`, `description`, `when_to_use`, `paths: [\"*.ext\"]`, `paradigm: llm_judge`, "
            "`requires_blind_context: true`, `cross_family_preferred: true`.\n"
            "   Body: crisp rubric instructions evaluated in strict isolation (no Actor CoT).\n"
            "2. Or run `/evaluators new <id>` / `crux eval new <id> [--style md|py]` to scaffold either format.\n"
            "3. Run `/reload` or `/evaluators reload` so it immediately appears in `consult_evaluator(lens=...)` and MCP `tools/list`."
        ),
    ),
)


def parse_colon_safe_frontmatter(markdown_text: str) -> tuple[dict[str, object], str]:
    """Colon-safe frontmatter parser supporting scalars, inline JSON/YAML lists, and multiline indented values (CT-14)."""
    m = FRONTMATTER_RE.match(markdown_text)
    if not m:
        return {}, markdown_text.strip()
    fm_raw, body = m.group(1), m.group(2).strip()
    meta: dict[str, object] = {}
    current_key: str | None = None
    current_list: list[str] | None = None

    for raw_line in fm_raw.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        # List item continuation: `  - "*.py"`
        stripped = raw_line.strip()
        if current_key and stripped.startswith("- ") and (raw_line.startswith(" ") or raw_line.startswith("\t")):
            item = stripped[2:].strip().strip("\"'")
            if current_list is None:
                current_list = []
                meta[current_key] = current_list
            current_list.append(item)
            continue
        # Indented multiline string continuation
        if current_key and current_list is None and (raw_line.startswith("  ") or raw_line.startswith("\t")):
            prev = str(meta.get(current_key, "")).strip()
            meta[current_key] = f"{prev} {stripped}".strip()
            continue

        key_m = re.match(r"^([A-Za-z0-9_-]+)\s*:\s*(.*)$", raw_line)
        if not key_m:
            continue
        key, val = key_m.group(1).strip(), key_m.group(2).strip()
        current_key = key
        current_list = None

        if val in ("|", ">", ""):
            meta[key] = ""
            continue
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1].strip()
            meta[key] = (
                [x.strip().strip("\"'") for x in inner.split(",") if x.strip()]
                if inner
                else []
            )
            continue
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]
        if val.lower() in ("true", "false"):
            meta[key] = val.lower() == "true"
        else:
            meta[key] = val

    return meta, body


def load_skill_file(skill_md: pathlib.Path, tier: str) -> SkillDefinition | None:
    try:
        raw = skill_md.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    meta, body = parse_colon_safe_frontmatter(raw)
    raw_name = str(meta.get("name") or skill_md.parent.name).strip().lower().replace("_", "-")
    if not SKILL_NAME_RE.match(raw_name):
        return None
    desc = str(meta.get("description") or "").strip()[:1024]
    if not desc and body:
        desc = body.splitlines()[0].lstrip("# ").strip()[:240]
    if not desc:
        return None

    paths_val = meta.get("paths", [])
    paths = tuple(paths_val) if isinstance(paths_val, list) else ((str(paths_val),) if paths_val else ())
    tools_val = meta.get("allowed-tools", meta.get("allowedTools", []))
    allowed_tools = tuple(tools_val) if isinstance(tools_val, list) else ((str(tools_val),) if tools_val else ())

    return SkillDefinition(
        name=raw_name,
        description=desc,
        body=body,
        tier=tier,
        location=skill_md.resolve(),
        when_to_use=str(meta.get("when_to_use", meta.get("whenToUse", ""))),
        argument_hint=str(meta.get("argument-hint", meta.get("argumentHint", ""))),
        paths=paths,
        context_mode=str(meta.get("context", "inline")),
        model_override=str(meta.get("model", "inherit")),
        allowed_tools=allowed_tools,
        disable_model_invocation=bool(meta.get("disable-model-invocation", meta.get("disableModelInvocation", False))),
        user_invocable=bool(meta.get("user-invocable", meta.get("userInvocable", True))),
    )


class SkillManager:
    """Discovers skills across 4 trust tiers, computes BM25/path `<skills_delta>`s, and activates resources safely."""

    def __init__(
        self,
        workspace_root: pathlib.Path,
        *,
        is_workspace_trusted: bool = False,
        ush_home: pathlib.Path | None = None,
        extension_skill_dirs: Iterable[pathlib.Path] = (),
        include_user_ecosystem_dirs: bool = True,
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.is_workspace_trusted = is_workspace_trusted
        self.ush_home = (ush_home or pathlib.Path("~/.ush")).expanduser().resolve()
        self.extension_skill_dirs = list(extension_skill_dirs)
        self.include_user_ecosystem_dirs = include_user_ecosystem_dirs

        self._skills: dict[str, SkillDefinition] = {}
        self.override_warnings: list[str] = []
        self._unlocked_path_skills: set[str] = set()
        self._shown_fingerprints: dict[str, str] = {}
        self._activated: set[str] = set()
        self.reload()

    def reload(self) -> int:
        self._skills.clear()
        self.override_warnings.clear()

        # Tier 1: Built-in
        for b in BUILTIN_SKILLS:
            self._skills[b.name] = b

        # Tier 2: Extensions
        for ed in self.extension_skill_dirs:
            self._scan_dir(ed, "extension")

        # Tier 3: User Global (`.ush/skills` + optional drop-in `.agents`, `.gemini`, `.claude` compatibility)
        user_dirs = [self.ush_home / "skills"]
        if self.include_user_ecosystem_dirs:
            home = pathlib.Path.home()
            user_dirs.extend(
                [
                    home / ".agents" / "skills",
                    home / ".gemini" / "config" / "skills",
                    home / ".gemini" / "skills",
                    home / ".claude" / "skills",
                ]
            )
        for ud in user_dirs:
            self._scan_dir(ud, "user")

        # Tier 4: Workspace (LOADED STRICTLY WHEN `is_workspace_trusted == True`! CT-14)
        if self.is_workspace_trusted:
            for rel in (".ush/skills", ".agents/skills", ".gemini/skills", ".claude/skills"):
                self._scan_dir(self.workspace_root / rel, "workspace")

        return len(self._skills)

    def _scan_dir(self, root: pathlib.Path, tier: str) -> None:
        if not root.is_dir():
            return
        for child in sorted(root.iterdir()):
            skill_md = child / "SKILL.md" if child.is_dir() else (child if child.name == "SKILL.md" else None)
            if not skill_md or not skill_md.is_file():
                continue
            sk = load_skill_file(skill_md, tier)
            if not sk:
                continue
            if sk.name in self._skills and self._skills[sk.name].tier != "builtin":
                self.override_warnings.append(
                    f"Skill `{sk.name}` ({tier}) overrides earlier `{self._skills[sk.name].tier}` definition."
                )
            self._skills[sk.name] = sk

    def all_skills(self) -> list[SkillDefinition]:
        return [self._skills[k] for k in sorted(self._skills)]

    def get(self, name: str) -> SkillDefinition | None:
        return self._skills.get(name)

    def user_invocable_skills(self) -> list[SkillDefinition]:
        return [s for s in self.all_skills() if s.user_invocable]

    def model_invocable_names(self) -> list[str]:
        return [
            s.name
            for s in self.all_skills()
            if not s.disable_model_invocation and (not s.paths or s.name in self._unlocked_path_skills)
        ]

    # ---- BM25 Ranking + Token-Budgeted `<skills_delta>` Catalog (CT-14) ----

    def _eligible_model_skills(self) -> list[SkillDefinition]:
        return [
            s
            for s in self.all_skills()
            if not s.disable_model_invocation and (not s.paths or s.name in self._unlocked_path_skills)
        ]

    def _rank_bm25(self, query: str, candidates: list[SkillDefinition]) -> list[SkillDefinition]:
        q_terms = set(WORD_RE.findall(query.lower()))
        if not q_terms or len(candidates) <= 1:
            return candidates

        docs = [WORD_RE.findall(f"{s.name} {s.description} {s.when_to_use}".lower()) for s in candidates]
        n_docs = len(docs)
        avgdl = max(1.0, sum(len(d) for d in docs) / n_docs)
        df: dict[str, int] = {t: sum(1 for d in docs if t in d) for t in q_terms}

        scored: list[tuple[float, str, SkillDefinition]] = []
        for sk, doc in zip(candidates, docs):
            score = 0.0
            dl = len(doc)
            # Activated skills stay pinned at top
            if sk.name in self._activated:
                score += 100.0
            for t in q_terms:
                tf = doc.count(t)
                if tf == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df[t] + 0.5) / (df[t] + 0.5))
                score += idf * ((tf * 2.2) / (tf + 1.2 * (1.0 - 0.75 + 0.75 * dl / avgdl)))
            scored.append((-score, sk.name, sk))
        scored.sort(key=lambda x: (x[0], x[1]))
        return [x[2] for x in scored]

    def _select_within_budget(self, ranked: list[SkillDefinition], budget_tokens: int) -> list[SkillDefinition]:
        chosen: list[SkillDefinition] = []
        used = 0
        for s in ranked:
            cost = s.approx_catalog_tokens()
            if chosen and used + cost > budget_tokens:
                continue
            chosen.append(s)
            used += cost
        return chosen

    def initial_catalog_block(self, budget_tokens: int = 2_000) -> str:
        """Emits Turn-0 static skills catalog (only unconditional model-invocable skills fitting budget)."""
        ranked = self._eligible_model_skills()
        chosen = self._select_within_budget(ranked, budget_tokens)
        if not chosen:
            return ""
        for s in chosen:
            self._shown_fingerprints[s.name] = s.fingerprint
        lines = ["<available_skills>", "Call `activate_skill(name)` when a skill applies:"]
        lines.extend(s.summary_line() for s in chosen)
        lines.append("</available_skills>")
        return "\n".join(lines)

    def compute_turn_delta(self, query: str, budget_tokens: int = 1_500) -> str:
        """Returns ONLY newly surfaced/BM25-matched skills not yet shown in Turn 0 (`0` prefix KV drift! CT-14)."""
        # Also auto-include any explicit `$skill-name` user mentions
        mentioned = set(DOLLAR_SKILL_RE.findall(query))
        ranked = self._rank_bm25(query, self._eligible_model_skills())
        selected = self._select_within_budget(ranked, budget_tokens)

        fresh: list[SkillDefinition] = []
        for mname in sorted(mentioned):
            sk = self._skills.get(mname)
            if sk and sk not in fresh:
                fresh.append(sk)
        for sk in selected:
            if self._shown_fingerprints.get(sk.name) != sk.fingerprint and sk not in fresh:
                fresh.append(sk)

        if not fresh:
            return ""
        for sk in fresh:
            self._shown_fingerprints[sk.name] = sk.fingerprint
        lines = ["<skills_delta>", "Newly relevant skills unlocked for `activate_skill`:"]
        lines.extend(sk.summary_line() for sk in fresh)
        lines.append("</skills_delta>")
        return "\n".join(lines)

    def on_file_accessed(self, path: pathlib.Path, budget_tokens: int = 1_500) -> str:
        """Qwen-style symlink-aware `paths:` glob matcher triggering a tail `<skills_delta>` on match (CT-14)."""
        resolved = path.resolve()
        try:
            rel = resolved.relative_to(self.workspace_root).as_posix()
        except ValueError:
            rel = resolved.name

        newly_unlocked: list[SkillDefinition] = []
        for sk in self.all_skills():
            if not sk.paths or sk.name in self._unlocked_path_skills:
                continue
            for pat in sk.paths:
                if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(resolved.name, pat) or (
                    pat.startswith("**/") and fnmatch.fnmatch(rel, pat[3:])
                ):
                    self._unlocked_path_skills.add(sk.name)
                    newly_unlocked.append(sk)
                    break

        if not newly_unlocked:
            return ""
        chosen = self._select_within_budget(newly_unlocked, budget_tokens)
        for sk in chosen:
            self._shown_fingerprints[sk.name] = sk.fingerprint
        return (
            "<skills_delta>\n"
            + "\n".join(sk.summary_line() for sk in chosen)
            + "\n</skills_delta>"
        )

    def expand_inline_dollar_mentions(self, prompt: str, firewall: SecurityFirewall) -> tuple[str, list[str]]:
        """Detects `$skill-name` pills in user input and appends activated skill instructions cleanly."""
        activated_blocks: list[str] = []
        for m in DOLLAR_SKILL_RE.finditer(prompt):
            sname = m.group(1)
            sk = self._skills.get(sname)
            if sk and sk.user_invocable:
                activated_blocks.append(self.activate(sname, firewall))
        return prompt, activated_blocks

    def activate(self, name: str, firewall: SecurityFirewall) -> str:
        """Activates a skill and dynamically grants read-sandbox access to its bundled folder (CT-14)."""
        sk = self._skills.get(name)
        if not sk:
            return f"[SKILL_ERROR] Unknown skill `{name}`."
        self._activated.add(name)

        resources: list[str] = []
        root_attr = ""
        if sk.location is not None:
            skill_dir = sk.location.parent.resolve()
            # Gemini CLI pattern: dynamically grant read sandbox access to external `~/.ush/skills/<name>/`!
            firewall.grant_read_directory(skill_dir)
            root_attr = f' root="{skill_dir}"'
            for fp in sorted(skill_dir.rglob("*")):
                if fp.is_file() and fp.name != "SKILL.md" and len(resources) < 40:
                    resources.append(fp.relative_to(skill_dir).as_posix())

        res_xml = (
            "\n<available_resources>\n" + "\n".join(resources) + "\n</available_resources>"
            if resources
            else ""
        )
        return (
            f'<activated_skill name="{sk.name}" context="{sk.context_mode}"{root_attr}>\n'
            f"<instructions>\n{sk.body}\n</instructions>{res_xml}\n"
            f"</activated_skill>"
        )
