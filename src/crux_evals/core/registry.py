"""Plug-and-Play Evaluator Auto-Discovery Registry (`EvaluatorRegistry`).

Supports dual-format extensibility matching Crucible's skill architecture:
1. Declarative Markdown Lenses (`EVALUATOR.md` with YAML frontmatter + rubric body,
   executed in strict model-isolated context with zero Python boilerplate).
2. Python Plugin Modules (`evaluator.yaml` + `module.py` with `EVALUATOR_CLASS` / `@register_evaluator`).

4-Tier Trust-Gated Discovery Order (`builtin < user < env < trusted_workspace`):
  - Built-in (`crux_evals/modules/`, with `importlib.resources` zipapp fallback for compiled `crux`)
  - User Global (`~/.ush/evaluators/`, `~/.crux_evals/plugins/`)
  - Environment (`CRUX_EVALS_PLUGIN_PATH`)
  - Trusted Workspace (`<ws>/.crux_evals/plugins/`, `<ws>/.ush/evaluators/` when trusted)
"""

from __future__ import annotations

import fnmatch
import importlib
import importlib.resources
import importlib.util
import inspect
import os
import pathlib
import re
import sys
from typing import Any, Callable, TypeVar

from crux_evals.core.protocol import (
    EvaluatorManifest,
    EvaluatorProtocol,
    MarkdownPromptEvaluator,
)

T = TypeVar("T", bound=type)

_REGISTERED_CLASSES: dict[str, type] = {}
_WORD_RE = re.compile(r"[a-z0-9]{2,}")


def register_evaluator(cls: T) -> T:
    """Class decorator that registers an Evaluator class in the global plugin catalog."""
    _REGISTERED_CLASSES[cls.__name__] = cls
    return cls


class EvaluatorRegistry:
    """Discovers, instantiates, and generates usefulness catalogs for built-in and custom Evaluator plugins."""

    def __init__(
        self,
        *,
        extra_plugin_dirs: list[pathlib.Path] | None = None,
        workspace_root: pathlib.Path | None = None,
        ush_home: pathlib.Path | None = None,
        is_workspace_trusted: bool = True,
        include_user_dirs: bool = True,
    ) -> None:
        self.builtin_modules_dir = pathlib.Path(__file__).resolve().parent.parent / "modules"
        self.workspace_root = workspace_root.expanduser().resolve() if workspace_root is not None else None
        self.ush_home = (
            ush_home.expanduser().resolve()
            if ush_home is not None
            else pathlib.Path(os.environ.get("USH_HOME", "~/.ush")).expanduser().resolve()
        )
        self.is_workspace_trusted = is_workspace_trusted
        self.include_user_dirs = include_user_dirs
        self.extra_plugin_dirs: list[pathlib.Path] = list(extra_plugin_dirs or [])

        self._unlocked_path_evals: set[str] = set()
        self._shown_fingerprints: dict[str, str] = {}

    def _search_roots(self) -> list[pathlib.Path]:
        roots: list[pathlib.Path] = []
        if self.builtin_modules_dir.is_dir():
            roots.append(self.builtin_modules_dir)

        if self.include_user_dirs:
            for user_dir in (
                self.ush_home / "evaluators",
                pathlib.Path.home() / ".crux_evals" / "plugins",
            ):
                if user_dir.is_dir() and user_dir not in roots:
                    roots.append(user_dir)

        env_paths = os.environ.get("CRUX_EVALS_PLUGIN_PATH", "").strip()
        if env_paths:
            for raw_p in env_paths.split(os.pathsep):
                if raw_p.strip():
                    p = pathlib.Path(raw_p.strip()).expanduser().resolve()
                    if p.is_dir() and p not in roots:
                        roots.append(p)

        for ep in self.extra_plugin_dirs:
            if ep.is_dir() and ep not in roots:
                roots.append(ep)

        if self.workspace_root is not None and self.is_workspace_trusted:
            for ws_dir in (
                self.workspace_root / ".crux_evals" / "plugins",
                self.workspace_root / ".ush" / "evaluators",
            ):
                if ws_dir.is_dir() and ws_dir not in roots:
                    roots.append(ws_dir)
        return roots

    def discover_entries(self) -> list[tuple[EvaluatorManifest, Callable[..., EvaluatorProtocol]]]:
        """Discovers built-in (including `.pyz` zipimport) and custom evaluators as `(manifest, factory)` pairs."""
        discovered: dict[str, tuple[EvaluatorManifest, Callable[..., EvaluatorProtocol]]] = {}

        # Zipapp / compiled binary fallback when `crux_evals/modules` is inside `.pyz`
        if not self.builtin_modules_dir.is_dir():
            self._discover_packaged_builtins(discovered)

        for root in self._search_roots():
            for child in sorted(root.iterdir()):
                if not child.is_dir() or child.name.startswith(("__", ".")):
                    continue
                yaml_path = child / "evaluator.yaml"
                mod_path = child / "module.py"
                md_path = child / "EVALUATOR.md"

                if yaml_path.is_file() and mod_path.is_file():
                    try:
                        manifest = EvaluatorManifest.from_yaml_file(yaml_path)
                        cls = self._load_evaluator_class_from_file(mod_path, manifest.evaluator_id)
                        if cls is not None:
                            discovered[manifest.evaluator_id] = (manifest, cls)
                    except Exception:
                        continue
                elif md_path.is_file():
                    try:
                        manifest, body = EvaluatorManifest.from_markdown_file(md_path)
                        discovered[manifest.evaluator_id] = (
                            manifest,
                            self._make_markdown_factory(manifest, body),
                        )
                    except Exception:
                        continue

        return sorted(discovered.values(), key=lambda item: (item[0].stage, item[0].evaluator_id))

    def _discover_packaged_builtins(
        self,
        discovered: dict[str, tuple[EvaluatorManifest, Callable[..., EvaluatorProtocol]]],
    ) -> None:
        """Loads built-in evaluator modules via `importlib.resources` when packaged inside `crux.pyz`."""
        try:
            mod_root = importlib.resources.files("crux_evals.modules")
            entries = sorted(mod_root.iterdir(), key=lambda item: item.name)
        except Exception:
            return

        for child in entries:
            if not child.is_dir() or child.name.startswith(("__", ".")):
                continue
            try:
                yaml_res = child.joinpath("evaluator.yaml")
                md_res = child.joinpath("EVALUATOR.md")
                if yaml_res.is_file():
                    manifest = EvaluatorManifest.from_yaml_text(
                        yaml_res.read_text(encoding="utf-8"),
                        default_id=child.name,
                    )
                    pkg = importlib.import_module(f"crux_evals.modules.{child.name}.module")
                    cls = getattr(pkg, "EVALUATOR_CLASS", None)
                    if not inspect.isclass(cls):
                        for _, obj in inspect.getmembers(pkg, inspect.isclass):
                            if obj.__module__ == pkg.__name__ and hasattr(obj, "evaluate") and obj.__name__ != "BaseEvaluator":
                                cls = obj
                                break
                    if inspect.isclass(cls):
                        discovered[manifest.evaluator_id] = (manifest, cls)
                elif md_res.is_file():
                    manifest, body = EvaluatorManifest.from_markdown_text(
                        md_res.read_text(encoding="utf-8"),
                        default_id=child.name,
                    )
                    discovered[manifest.evaluator_id] = (
                        manifest,
                        self._make_markdown_factory(manifest, body),
                    )
            except Exception:
                continue

    @staticmethod
    def _make_markdown_factory(
        manifest: EvaluatorManifest,
        body: str,
    ) -> Callable[..., EvaluatorProtocol]:
        def _factory(*, llm_judge: Any = None, **_ignored: Any) -> MarkdownPromptEvaluator:
            return MarkdownPromptEvaluator(
                manifest=manifest,
                rubric_body=body,
                llm_judge=llm_judge,
            )

        return _factory

    def discover_classes(self) -> list[tuple[EvaluatorManifest, Any]]:
        """Backward-compatible alias returning `(EvaluatorManifest, cls_or_factory)` pairs."""
        return self.discover_entries()

    def list_manifests(self) -> list[EvaluatorManifest]:
        return [m for m, _ in self.discover_entries()]

    def all_evaluator_ids(self) -> list[str]:
        return [m.evaluator_id for m, _ in self.discover_entries()]

    def instantiate_all(
        self,
        *,
        kwargs_by_id: dict[str, dict[str, Any]] | None = None,
        llm_judge: Callable[..., str] | None = None,
        llm_breaker: Callable[..., str] | None = None,
    ) -> list[EvaluatorProtocol]:
        """Instantiates all discovered evaluators with optional dependency injection."""
        kwargs_map = dict(kwargs_by_id or {})
        instances: list[EvaluatorProtocol] = []
        for manifest, factory in self.discover_entries():
            target_fn = factory.__init__ if inspect.isclass(factory) else factory
            init_sig = inspect.signature(target_fn)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in init_sig.parameters.values())
            call_kwargs: dict[str, Any] = dict(kwargs_map.get(manifest.evaluator_id, {}))
            if (accepts_var_kw or "llm_judge" in init_sig.parameters) and "llm_judge" not in call_kwargs and llm_judge is not None:
                call_kwargs["llm_judge"] = llm_judge
            if (accepts_var_kw or "llm_breaker" in init_sig.parameters) and "llm_breaker" not in call_kwargs and llm_breaker is not None:
                call_kwargs["llm_breaker"] = llm_breaker
            instance = factory(**call_kwargs)
            instances.append(instance)
        return instances

    # ---- Skill-Parity Turn-0 Catalog & Path/Situational Delta Discovery ----

    def initial_catalog_block(self, budget_tokens: int = 1_200) -> str:
        """Emits cache-stable Turn-0 catalog highlighting clean-room workflow primitives + unconditional evaluators."""
        candidates = [m for m in self.list_manifests() if not m.paths]
        lines = [
            "<available_evaluators>",
            "Clean-room zero-history evaluators (`consult_evaluator` — pass `evaluator`, `files` [code/HTML/screenshots.png], optional 1-line `goal`):",
            "- cold_eyes_review [pre_ship, clean-room]: Right before final answer on non-trivial code/UI/viz — inspects raw files, diffs, tests & screenshots cold; returns PASS or <=5 surgical fixes.",
            "- challenge_design [pre_build, clean-room]: Right before writing >250L files or multi-file rewrites — catches schema/state/layout traps early (<=5 items or PASS).",
            "- break_loop [when_stuck, clean-room]: After 2 failed attempts at a bug/render issue — ignores failed chat history and returns <=3 orthogonal hypotheses + probes.",
            "- tournament [contrastive_selection, clean-room]: Compare 2+ candidate architectures, algorithms, or files side-by-side and return the winner + falsification trade-offs.",
        ]
        used = sum(max(12, len(ln) // 4) for ln in lines)
        for m in candidates:
            # Mark all unconditional builtins as shown in Turn-0 so they aren't re-injected in Turn-1 delta
            self._shown_fingerprints[m.evaluator_id] = m.fingerprint
            if m.evaluator_id in (
                "trajectory_sentinel",
                "clean_patch_and_contract",
                "epistemic_grounding_auditor",
                "blind_bars_judge",
                "adversarial_falsifier",
            ):
                continue
            # Custom user/workspace unconditional plugins still surface clearly in Turn-0 catalog!
            cost = m.approx_catalog_tokens()
            if used + cost <= budget_tokens:
                lines.append(m.summary_line())
                used += cost
        lines.append("</available_evaluators>")
        return "\n".join(lines)

    def on_file_accessed(self, path: pathlib.Path, budget_tokens: int = 800) -> str:
        """Unlocks path-scoped evaluators matching touched files via tail `<evaluators_delta>`."""
        resolved = path.resolve()
        if self.workspace_root is not None:
            try:
                rel = resolved.relative_to(self.workspace_root).as_posix()
            except ValueError:
                rel = resolved.name
        else:
            rel = resolved.name

        newly_unlocked: list[EvaluatorManifest] = []
        for m in self.list_manifests():
            if not m.paths or m.evaluator_id in self._unlocked_path_evals:
                continue
            for pat in m.paths:
                if (
                    fnmatch.fnmatch(rel, pat)
                    or fnmatch.fnmatch(resolved.name, pat)
                    or (pat.startswith("**/") and fnmatch.fnmatch(rel, pat[3:]))
                ):
                    self._unlocked_path_evals.add(m.evaluator_id)
                    if self._shown_fingerprints.get(m.evaluator_id) != m.fingerprint:
                        newly_unlocked.append(m)
                    break

        if not newly_unlocked:
            return ""
        used = 0
        chosen: list[EvaluatorManifest] = []
        for m in newly_unlocked:
            cost = m.approx_catalog_tokens()
            if chosen and used + cost > budget_tokens:
                continue
            chosen.append(m)
            used += cost
            self._shown_fingerprints[m.evaluator_id] = m.fingerprint
        return (
            "<evaluators_delta>\n"
            "Specialized domain evaluator(s) relevant to touched files (`consult_evaluator`):\n"
            + "\n".join(m.summary_line() for m in chosen)
            + "\n</evaluators_delta>"
        )

    def compute_turn_delta(self, query: str, budget_tokens: int = 800) -> str:
        """Emits newly added or query-matched evaluators without mutating Turn-0 system prompt."""
        q_terms = set(_WORD_RE.findall((query or "").lower()))
        fresh: list[EvaluatorManifest] = []
        for m in self.list_manifests():
            if m.paths and m.evaluator_id not in self._unlocked_path_evals:
                # Unlock on explicit mention of evaluator id or strong keyword overlap in query
                id_words = set(m.evaluator_id.split("_"))
                if m.evaluator_id not in query.lower() and len(q_terms & id_words) < max(1, len(id_words)):
                    continue
                self._unlocked_path_evals.add(m.evaluator_id)
            if self._shown_fingerprints.get(m.evaluator_id) != m.fingerprint:
                fresh.append(m)
        if not fresh:
            return ""
        used = 0
        chosen: list[EvaluatorManifest] = []
        for m in fresh:
            cost = m.approx_catalog_tokens()
            if chosen and used + cost > budget_tokens:
                continue
            chosen.append(m)
            used += cost
            self._shown_fingerprints[m.evaluator_id] = m.fingerprint
        return (
            "<evaluators_delta>\n"
            "Additional evaluator lens(es) available for `consult_evaluator`:\n"
            + "\n".join(m.summary_line() for m in chosen)
            + "\n</evaluators_delta>"
        )

    @staticmethod
    def _load_evaluator_class_from_file(mod_path: pathlib.Path, evaluator_id: str) -> type | None:
        module_name = f"crux_evals_plugin_{evaluator_id}_{abs(hash(str(mod_path)))}"
        spec = importlib.util.spec_from_file_location(module_name, str(mod_path))
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        explicit = getattr(module, "EVALUATOR_CLASS", None)
        if inspect.isclass(explicit):
            return explicit

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and hasattr(obj, "evaluate") and obj.__name__ != "BaseEvaluator":
                return obj
        return None


def scaffold_new_evaluator(
    evaluator_id: str,
    *,
    name: str | None = None,
    paradigm: str = "deterministic",
    stage: int | None = None,
    target_root: pathlib.Path | None = None,
    mcp_tool_name: str | None = None,
    description: str = "Custom complementary evaluator plugin.",
    when_to_use: str = "",
    paths: list[str] | tuple[str, ...] | None = None,
    style: str = "python",
) -> pathlib.Path:
    """Creates a ready-to-run Evaluator plugin directory (`EVALUATOR.md` or `evaluator.yaml` + `module.py`)."""
    clean_id = re.sub(r"[^a-z0-9_]", "_", evaluator_id.strip().lower()).strip("_")
    if not clean_id:
        raise ValueError("evaluator_id must contain alphanumeric characters")

    resolved_style = style.strip().lower()
    if resolved_style not in ("markdown", "md", "python", "py"):
        resolved_style = "python"
    is_markdown = resolved_style in ("markdown", "md")
    eff_paradigm = "llm_judge" if is_markdown and paradigm == "deterministic" else paradigm

    resolved_stage = stage if stage is not None else (
        1 if eff_paradigm == "deterministic" else (2 if eff_paradigm == "llm_judge" else 3)
    )
    resolved_name = name or " ".join(w.capitalize() for w in clean_id.split("_"))
    resolved_mcp = mcp_tool_name or f"eval_{clean_id}"
    class_name = "".join(w.capitalize() for w in clean_id.split("_")) + "Evaluator"
    resolved_when = when_to_use.strip() or f"When auditing tasks involving {resolved_name.lower()}."
    clean_paths = [str(p).strip() for p in (paths or ()) if str(p).strip()]

    if target_root is None:
        builtin_dir = pathlib.Path(__file__).resolve().parent.parent / "modules"
        # Compiled `crux.pyz` lives inside a zip file (non-writable); fall back cleanly to `~/.ush/evaluators`
        if builtin_dir.is_dir() and os.access(builtin_dir, os.W_OK):
            target_root = builtin_dir
        else:
            target_root = pathlib.Path(os.environ.get("USH_HOME", "~/.ush")).expanduser().resolve() / "evaluators"
    target_root = target_root.expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    folder_name = clean_id
    if target_root.name == "modules" and not re.match(r"^m\d+_", clean_id):
        existing_nums = [
            int(m.group(1))
            for p in target_root.iterdir()
            if p.is_dir() and (m := re.match(r"^m(\d+)_", p.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 7
        folder_name = f"m{next_num:02d}_{clean_id}"

    plugin_dir = target_root / folder_name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    paths_yaml = f"paths: [{', '.join(clean_paths)}]\n" if clean_paths else ""
    requires_blind = "false" if eff_paradigm == "llm_judge" and "grounding" in clean_id else "true"

    if is_markdown:
        md_content = (
            "---\n"
            f"evaluator_id: {clean_id}\n"
            f"name: {resolved_name}\n"
            f"paradigm: {eff_paradigm}\n"
            f"stage: {resolved_stage}\n"
            "events: [MCPConsult, Stop]\n"
            f"requires_blind_context: {requires_blind}\n"
            "cross_family_preferred: true\n"
            f"mcp_tool_name: {resolved_mcp}\n"
            f"description: {description}\n"
            f"when_to_use: {resolved_when}\n"
            f"{paths_yaml}"
            "---\n\n"
            f"# {resolved_name} (`{clean_id}`)\n\n"
            "Evaluate ONLY the ground-truth artifacts (`<task_objective>`, `<anonymous_git_diff>`, "
            "`<target_file_excerpts>`, and `<raw_verification_output>`).\n\n"
            "## Rubric Checks\n"
            "1. **Correctness & Edge Cases**: Check for boundary conditions, error handling, and missing edge validation.\n"
            "2. **Invariants & Quality**: Verify adherence to `<task_objective>` and `<declared_invariants>`.\n"
            "3. **Actionable Next Probes**: Surface concrete `blind_spots` and `suggested_next_probes` the agent can act on immediately.\n"
        )
        (plugin_dir / "EVALUATOR.md").write_text(md_content, encoding="utf-8")
        return plugin_dir

    yaml_content = (
        f"evaluator_id: {clean_id}\n"
        f"name: {resolved_name}\n"
        f"paradigm: {eff_paradigm}\n"
        f"stage: {resolved_stage}\n"
        f"events: [AfterTool, Stop]\n"
        f"tool_matcher: .*\n"
        f"requires_blind_context: {requires_blind}\n"
        f"cross_family_preferred: {'true' if eff_paradigm != 'deterministic' else 'false'}\n"
        f"mcp_tool_name: {resolved_mcp}\n"
        f"description: {description}\n"
        f"when_to_use: {resolved_when}\n"
        f"{paths_yaml}"
    )
    (plugin_dir / "evaluator.yaml").write_text(yaml_content, encoding="utf-8")

    module_content = f'''"""Custom Evaluator Plugin: {resolved_name} (`{class_name}`)."""

from __future__ import annotations

import pathlib

from crux_evals.core.protocol import (
    BaseEvaluator,
    EvaluationFrame,
    EvaluationVerdict,
)
from crux_evals.core.registry import register_evaluator


@register_evaluator
class {class_name}(BaseEvaluator):
    """{description}"""

    def __init__(self) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        safe_frame = frame.with_epistemic_firewall() if self.manifest.requires_blind_context else frame
        blind_spots: list[str] = []
        suggested_probes: list[str] = []

        _ = safe_frame

        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm=self.manifest.paradigm,
            decision="ALLOW",
            bars_score=1.0,
            progress_advantage=0.1,
            uncertainty=0.0,
            blind_spots=blind_spots,
            suggested_next_probes=suggested_probes,
        )


EVALUATOR_CLASS = {class_name}
'''
    (plugin_dir / "module.py").write_text(module_content, encoding="utf-8")
    (plugin_dir / "__init__.py").write_text(
        f"from .module import {class_name}\n\n__all__ = [{class_name!r}]\n",
        encoding="utf-8",
    )
    return plugin_dir
