"""Core protocol, auto-discovery registry, telemetry state, and cascade runner for Crux-Evals."""

from crux_evals.core.protocol import (
    BaseEvaluator,
    CascadeReport,
    Decision,
    EnforcementMode,
    EvaluationFrame,
    EvaluationVerdict,
    EvaluatorManifest,
    EvaluatorParadigm,
    EvaluatorProtocol,
    HookEvent,
    LLMJudgeCallable,
    ToolRecord,
    sha256_short,
)
from crux_evals.core.registry import EvaluatorRegistry, register_evaluator
from crux_evals.core.runner import EvaluatorCascadeRunner, build_default_cascade
from crux_evals.core.state import (
    HarnessTelemetryStore,
    SessionTelemetry,
    capture_git_diff,
    compute_action_signature,
    compute_error_signature,
    discover_git_modified_files,
)

__all__ = [
    "BaseEvaluator",
    "CascadeReport",
    "Decision",
    "EnforcementMode",
    "EvaluationFrame",
    "EvaluationVerdict",
    "EvaluatorCascadeRunner",
    "EvaluatorManifest",
    "EvaluatorParadigm",
    "EvaluatorProtocol",
    "EvaluatorRegistry",
    "HarnessTelemetryStore",
    "HookEvent",
    "LLMJudgeCallable",
    "SessionTelemetry",
    "ToolRecord",
    "build_default_cascade",
    "capture_git_diff",
    "compute_action_signature",
    "compute_error_signature",
    "discover_git_modified_files",
    "register_evaluator",
    "sha256_short",
]
