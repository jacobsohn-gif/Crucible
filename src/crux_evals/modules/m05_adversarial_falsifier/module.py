"""Module 05: Popperian Adversarial Counterexample Falsifier (`AdversarialFalsifierEvaluator`)."""

from __future__ import annotations

import ast
import os
import pathlib
import re
import subprocess
import sys
import tempfile

from crux_evals.core.protocol import (
    BaseEvaluator,
    EvaluationFrame,
    EvaluationVerdict,
    LLMJudgeCallable,
)
from crux_evals.core.registry import register_evaluator

PYTHON_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


@register_evaluator
class AdversarialFalsifierEvaluator(BaseEvaluator):
    """Forked Red-Team Breaker that synthesizes & runs edge-case counterexample probes."""

    def __init__(
        self,
        *,
        llm_breaker: LLMJudgeCallable | None = None,
        sandbox_timeout_seconds: float = 5.0,
        enable_ast_property_fuzzer: bool = True,
    ) -> None:
        super().__init__(pathlib.Path(__file__).with_name("evaluator.yaml"))
        self.breaker_prompt = pathlib.Path(__file__).with_name("breaker_prompt.md").read_text(encoding="utf-8")
        self.llm_breaker = llm_breaker
        self.sandbox_timeout_seconds = sandbox_timeout_seconds
        self.enable_ast_property_fuzzer = enable_ast_property_fuzzer

    def evaluate(self, frame: EvaluationFrame) -> EvaluationVerdict:
        blind_frame = frame.with_epistemic_firewall()
        ws = pathlib.Path(blind_frame.workspace_root).resolve()
        py_files = [
            rel for rel in blind_frame.modified_files
            if rel.endswith(".py") and not os.path.basename(rel).startswith("test_") and not rel.endswith("_test.py")
        ]

        probes: list[tuple[str, str]] = []

        if self.llm_breaker is not None and (blind_frame.unified_git_diff.strip() or py_files):
            user_payload = (
                f"<task_objective>\n{blind_frame.task_objective}\n</task_objective>\n\n"
                f"<declared_invariants>\n{chr(10).join(blind_frame.declared_invariants)}\n</declared_invariants>\n\n"
                f"<modified_files>\n{', '.join(blind_frame.modified_files)}\n</modified_files>\n\n"
                f"<anonymous_git_diff>\n{blind_frame.unified_git_diff[:20_000]}\n</anonymous_git_diff>"
            )
            raw_script = self.llm_breaker(
                self.breaker_prompt,
                user_payload,
                blind_frame.actor_model_family,
            )
            m = PYTHON_BLOCK_RE.search(raw_script)
            script_code = m.group(1).strip() if m else raw_script.strip()
            if script_code and ("assert" in script_code or "import" in script_code):
                probes.append(("llm_adversarial_probe", script_code))

        if self.enable_ast_property_fuzzer and py_files:
            synth_probes = self._synthesize_ast_boundary_probes(ws, py_files)
            probes.extend(synth_probes)

        if not probes:
            return EvaluationVerdict(
                evaluator_id=self.manifest.evaluator_id,
                paradigm="adversarial",
                decision="ALLOW",
                bars_score=1.0,
                progress_advantage=0.1,
                criterion_scores={"adversarial_resilience": 1.0},
            )

        for probe_name, probe_code in probes:
            failed, counterexample_report = self._run_probe_in_sandbox(ws, probe_code)
            if failed:
                msg = f"Adversarial probe `{probe_name}` triggered a reproducible runtime failure."
                rem = (
                    "Reproduce the adversarial counterexample above, add a unit test covering "
                    "that boundary input, and fix the implementation so it handles the edge case safely."
                )
                return EvaluationVerdict(
                    evaluator_id=self.manifest.evaluator_id,
                    paradigm="adversarial",
                    decision="VETO",
                    bars_score=0.0,
                    progress_advantage=-0.9,
                    uncertainty=0.0,
                    criterion_scores={"adversarial_resilience": 0.0},
                    hard_vetoes_triggered=["HARD_VETO_ADVERSARIAL_COUNTEREXAMPLE"],
                    evidence_citations=[msg],
                    counterexample=counterexample_report,
                    remediation_directive=rem,
                    blind_spots=[msg],
                    suggested_next_probes=[rem],
                )

        return EvaluationVerdict(
            evaluator_id=self.manifest.evaluator_id,
            paradigm="adversarial",
            decision="ALLOW",
            bars_score=1.0,
            progress_advantage=0.3,
            uncertainty=0.0,
            criterion_scores={"adversarial_resilience": 1.0},
        )

    def _run_probe_in_sandbox(self, workspace_root: pathlib.Path, probe_code: str) -> tuple[bool, str]:
        """Runs a Python falsification script in a temporary directory with `PYTHONPATH=workspace_root`."""
        clean_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": str(workspace_root),
            "HOME": tempfile.gettempdir(),
            "LANG": "C.UTF-8",
        }
        with tempfile.TemporaryDirectory(prefix="crux_eval_adv_") as tmpdir:
            probe_file = pathlib.Path(tmpdir) / "adversarial_probe.py"
            probe_file.write_text(probe_code, encoding="utf-8")
            try:
                res = subprocess.run(
                    [sys.executable, str(probe_file)],
                    cwd=tmpdir,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=self.sandbox_timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                return True, (
                    f"Probe timed out after {self.sandbox_timeout_seconds}s (infinite loop / complexity trap):\n"
                    f"```python\n{probe_code}\n```"
                )
            except OSError as exc:
                return False, str(exc)

            if res.returncode != 0:
                err_tail = (res.stdout + "\n" + res.stderr).strip()[-1200:]
                report = (
                    f"Probe Script:\n```python\n{probe_code}\n```\n"
                    f"Failure Output (exit {res.returncode}):\n{err_tail}"
                )
                return True, report
        return False, ""

    @staticmethod
    def _synthesize_ast_boundary_probes(
        workspace_root: pathlib.Path,
        py_files: list[str],
    ) -> list[tuple[str, str]]:
        probes: list[tuple[str, str]] = []
        for rel_path in py_files:
            fp = (workspace_root / rel_path).resolve()
            if not fp.is_file():
                continue
            try:
                tree = ast.parse(fp.read_text(encoding="utf-8"), filename=rel_path)
            except SyntaxError:
                continue

            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                pos_args = [a for a in node.args.args if a.arg not in ("self", "cls")]
                num_required = len(pos_args) - len(node.args.defaults)
                if num_required <= 0 or num_required > 3:
                    continue

                boundary_vectors = AdversarialFalsifierEvaluator._boundary_args_for_function(
                    pos_args[:num_required]
                )
                if not boundary_vectors:
                    continue

                for idx, arg_exprs in enumerate(boundary_vectors):
                    call_str = f"{node.name}({', '.join(arg_exprs)})"
                    script = (
                        "import importlib.util, pathlib\n"
                        f"spec = importlib.util.spec_from_file_location('target_mod', {str(fp)!r})\n"
                        "mod = importlib.util.module_from_spec(spec)\n"
                        "spec.loader.exec_module(mod)\n"
                        f"fn = getattr(mod, {node.name!r})\n"
                        "try:\n"
                        f"    fn({', '.join(arg_exprs)})\n"
                        "except (ValueError, TypeError, AssertionError):\n"
                        "    pass  # Explicit input validation is acceptable\n"
                        "except (ZeroDivisionError, IndexError, KeyError, AttributeError, UnboundLocalError) as exc:\n"
                        f"    raise RuntimeError(f'Unhandled boundary crash in {call_str}: {{type(exc).__name__}}: {{exc}}') from exc\n"
                    )
                    probes.append((f"ast_boundary_{node.name}_{idx}", script))
        return probes

    @staticmethod
    def _boundary_args_for_function(args: list[ast.arg]) -> list[list[str]]:
        per_arg_candidates: list[list[str]] = []
        for a in args:
            ann = ast.unparse(a.annotation).lower() if a.annotation else ""
            name = a.arg.lower()
            if "list" in ann or "seq" in ann or "iterable" in ann or any(
                k in name for k in ("items", "values", "numbers", "rows", "records", "list", "arr", "prices", "nums")
            ):
                per_arg_candidates.append(["[]", "[0.0]"])
            elif "dict" in ann or "map" in ann or "mapping" in ann or "data" in name or "config" in name:
                per_arg_candidates.append(["{}"])
            elif "int" in ann or "float" in ann or any(
                k in name for k in ("count", "total", "amount", "rate", "price", "num", "denom", "divisor", "n", "x", "y")
            ):
                per_arg_candidates.append(["0", "-1"])
            elif "str" in ann or any(k in name for k in ("text", "name", "query", "path", "s")):
                per_arg_candidates.append(["''"])
            else:
                per_arg_candidates.append(["[]", "0"])

        combos: list[list[str]] = []
        max_len = max(len(c) for c in per_arg_candidates)
        for i in range(min(2, max_len)):
            combos.append([c[min(i, len(c) - 1)] for c in per_arg_candidates])
        return combos


EVALUATOR_CLASS = AdversarialFalsifierEvaluator
