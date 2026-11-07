"""Plug-and-Play Evaluator Auto-Discovery Registry (`EvaluatorRegistry`).

Makes adding future evaluators effortless:
1. Drop-In Folder Discovery:
   Create any folder under `crux_evals/modules/<name>/` or `<workspace>/.crux_evals/plugins/<name>/`
   (or any path in `CRUX_EVALS_PLUGIN_PATH`) containing `evaluator.yaml` + `module.py`.
   `EvaluatorRegistry.discover()` automatically loads it and exposes it to both the
   MCP Server (`tools/list`, `tools/call`) and the Cascade Runner (`EvaluatorCascadeRunner`).
2. Decorator Registration:
   Use `@register_evaluator` on any Python class implementing `EvaluatorProtocol`.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import pathlib
import re
import sys
from typing import Any, Callable, TypeVar

from crux_evals.core.protocol import EvaluatorManifest, EvaluatorProtocol

T = TypeVar("T", bound=type)

_REGISTERED_CLASSES: dict[str, type] = {}


def register_evaluator(cls: T) -> T:
    """Class decorator that registers an Evaluator class in the global plugin catalog."""
    _REGISTERED_CLASSES[cls.__name__] = cls
    return cls


class EvaluatorRegistry:
    """Discovers, instantiates, and manages built-in and external Evaluator plugins."""

    def __init__(
        self,
        *,
        extra_plugin_dirs: list[pathlib.Path] | None = None,
        workspace_root: pathlib.Path | None = None,
    ) -> None:
        self.builtin_modules_dir = pathlib.Path(__file__).resolve().parent.parent / "modules"
        self.extra_plugin_dirs: list[pathlib.Path] = list(extra_plugin_dirs or [])

        # Automatically include `<workspace>/.crux_evals/plugins` if present
        if workspace_root is not None:
            ws_plugins = workspace_root.expanduser().resolve() / ".crux_evals" / "plugins"
            if ws_plugins.is_dir() and ws_plugins not in self.extra_plugin_dirs:
                self.extra_plugin_dirs.append(ws_plugins)

        # Automatically include colon-separated paths from `CRUX_EVALS_PLUGIN_PATH`
        env_paths = os.environ.get("CRUX_EVALS_PLUGIN_PATH", "").strip()
        if env_paths:
            for raw_p in env_paths.split(os.pathsep):
                if raw_p.strip():
                    p = pathlib.Path(raw_p.strip()).expanduser().resolve()
                    if p.is_dir() and p not in self.extra_plugin_dirs:
                        self.extra_plugin_dirs.append(p)

    def discover_classes(self) -> list[tuple[EvaluatorManifest, type]]:
        """Scans built-in and external plugin directories for `(EvaluatorManifest, cls)` pairs."""
        discovered: dict[str, tuple[EvaluatorManifest, type]] = {}

        search_roots = [self.builtin_modules_dir] + self.extra_plugin_dirs
        for root in search_roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not child.is_dir() or child.name.startswith(("__", ".")):
                    continue
                yaml_path = child / "evaluator.yaml"
                mod_path = child / "module.py"
                if not yaml_path.is_file() or not mod_path.is_file():
                    continue
                manifest = EvaluatorManifest.from_yaml_file(yaml_path)
                cls = self._load_evaluator_class_from_file(mod_path, manifest.evaluator_id)
                if cls is not None:
                    discovered[manifest.evaluator_id] = (manifest, cls)

        return sorted(discovered.values(), key=lambda item: (item[0].stage, item[0].evaluator_id))

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
        for manifest, cls in self.discover_classes():
            init_sig = inspect.signature(cls.__init__)
            call_kwargs: dict[str, Any] = dict(kwargs_map.get(manifest.evaluator_id, {}))
            if "llm_judge" in init_sig.parameters and "llm_judge" not in call_kwargs and llm_judge is not None:
                call_kwargs["llm_judge"] = llm_judge
            if "llm_breaker" in init_sig.parameters and "llm_breaker" not in call_kwargs and llm_breaker is not None:
                call_kwargs["llm_breaker"] = llm_breaker
            instance = cls(**call_kwargs)
            instances.append(instance)
        return instances

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
) -> pathlib.Path:
    """Creates a ready-to-run Evaluator plugin directory with `evaluator.yaml` and `module.py`."""
    clean_id = re.sub(r"[^a-z0-9_]", "_", evaluator_id.strip().lower()).strip("_")
    if not clean_id:
        raise ValueError("evaluator_id must contain alphanumeric characters")

    resolved_stage = stage if stage is not None else (1 if paradigm == "deterministic" else (2 if paradigm == "llm_judge" else 3))
    resolved_name = name or " ".join(w.capitalize() for w in clean_id.split("_"))
    resolved_mcp = mcp_tool_name or f"eval_{clean_id}"
    class_name = "".join(w.capitalize() for w in clean_id.split("_")) + "Evaluator"

    if target_root is None:
        target_root = pathlib.Path(__file__).resolve().parent.parent / "modules"
    target_root = target_root.expanduser().resolve()
    target_root.mkdir(parents=True, exist_ok=True)

    folder_name = clean_id
    if target_root.name == "modules" and not re.match(r"^m\d+_", clean_id):
        existing_nums = [
            int(m.group(1))
            for p in target_root.iterdir()
            if p.is_dir() and (m := re.match(r"^m(\d+)_", p.name))
        ]
        next_num = (max(existing_nums) + 1) if existing_nums else 6
        folder_name = f"m{next_num:02d}_{clean_id}"

    plugin_dir = target_root / folder_name
    plugin_dir.mkdir(parents=True, exist_ok=True)

    yaml_content = (
        f"evaluator_id: {clean_id}\n"
        f"name: {resolved_name}\n"
        f"paradigm: {paradigm}\n"
        f"stage: {resolved_stage}\n"
        f"events: [AfterTool, Stop]\n"
        f"tool_matcher: .*\n"
        f"requires_blind_context: {'false' if paradigm == 'llm_judge' and 'grounding' in clean_id else 'true'}\n"
        f"cross_family_preferred: {'true' if paradigm != 'deterministic' else 'false'}\n"
        f"mcp_tool_name: {resolved_mcp}\n"
        f"description: {description}\n"
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

