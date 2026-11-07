"""Ground-Truth Session Telemetry Store & Disk Hydrator (`HarnessTelemetryStore`)."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pathlib
import re
import subprocess
from typing import Any

from crux_evals.core.protocol import EvaluationFrame, HookEvent, ToolRecord, sha256_short

VERIFICATION_CMD_RE = re.compile(
    r"\b(pytest|py\.test|unittest|python3?\s+-m\s+(pytest|unittest)|"
    r"python3?\s+[^\s]+_test\.py|python3?\s+test_[^\s]+\.py|"
    r"python3?\s+-c\b|cargo\s+(test|check)|go\s+test|npm\s+test|"
    r"pnpm\s+test|vitest|jest|ruff\s+check|mypy|pyright|tsc\b|bazel\s+test)\b",
    re.IGNORECASE,
)

HEX_ADDR_RE = re.compile(r"0x[0-9a-fA-F]+")
LINE_NUM_RE = re.compile(r"\bline\s+\d+", re.IGNORECASE)
TEMP_PATH_RE = re.compile(r"/tmp/[A-Za-z0-9_.-]+")


def compute_error_signature(tool_name: str, output_raw: str, is_error: bool) -> str:
    """Extracts a stable error fingerprint from stderr/stdout so repeated exceptions match across edits."""
    if not is_error and "Traceback" not in output_raw and "FAILED" not in output_raw and "Error:" not in output_raw:
        return ""
    lines = [ln.strip() for ln in output_raw.strip().splitlines() if ln.strip()]
    if not lines:
        return sha256_short(f"{tool_name}:empty_error")

    key_lines: list[str] = []
    for ln in reversed(lines):
        if any(
            tok in ln
            for tok in (
                "Error",
                "Exception",
                "FAILED",
                "AssertionError",
                "SyntaxError",
                "TypeError",
                "ValueError",
                "KeyError",
                "AttributeError",
                "ImportError",
                "ModuleNotFoundError",
                "exit_code=",
            )
        ):
            normalized = HEX_ADDR_RE.sub("0xADDR", ln)
            normalized = LINE_NUM_RE.sub("line N", normalized)
            normalized = TEMP_PATH_RE.sub("/tmp/PATH", normalized)
            key_lines.append(normalized)
            if len(key_lines) >= 2:
                break
    if not key_lines:
        normalized = HEX_ADDR_RE.sub("0xADDR", lines[-1][:200])
        normalized = LINE_NUM_RE.sub("line N", normalized)
        key_lines.append(normalized)
    return sha256_short(f"{tool_name}|{'|'.join(key_lines)}")


def compute_action_signature(tool_name: str, tool_args: dict[str, Any]) -> str:
    """Canonical hash of `(tool_name, sorted_args)`."""
    canonical_args = json.dumps(tool_args or {}, sort_keys=True, ensure_ascii=False)
    return sha256_short(f"{tool_name}|{canonical_args}")


def capture_git_diff(workspace_root: pathlib.Path, target_paths: list[str] | None = None) -> str:
    """Captures raw `git diff HEAD` from the workspace without trusting Actor reports."""
    if not (workspace_root / ".git").exists():
        return ""
    cmd = ["git", "diff", "HEAD", "--unified=3"]
    if target_paths:
        cmd.append("--")
        cmd.extend(target_paths)
    try:
        res = subprocess.run(
            cmd,
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0:
            return res.stdout
    except (subprocess.SubprocessError, OSError):
        pass
    return ""


def discover_git_modified_files(workspace_root: pathlib.Path) -> list[str]:
    """Returns workspace-relative modified/untracked files directly from `git status --porcelain`."""
    if not (workspace_root / ".git").exists():
        return []
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=3.0,
        )
        if res.returncode == 0:
            out: list[str] = []
            for line in res.stdout.splitlines():
                if len(line) > 3:
                    path_part = line[3:].strip().strip('"')
                    if " -> " in path_part:
                        path_part = path_part.split(" -> ", 1)[1].strip()
                    if path_part and path_part not in out:
                        out.append(path_part)
            return out
    except (subprocess.SubprocessError, OSError):
        pass
    return []


@dataclass
class SessionTelemetry:
    """Persisted per-session telemetry state managed by the Harness / MCP Server."""

    session_id: str
    workspace_root: str
    task_objective: str = ""
    declared_invariants: list[str] = field(default_factory=list)
    step_index: int = 0
    read_files: list[str] = field(default_factory=list)
    modified_files: list[str] = field(default_factory=list)
    last_edit_step: int = 0
    last_verification_step: int = 0
    last_verification_passed: bool | None = None
    last_verification_output: str = ""
    tool_history: list[ToolRecord] = field(default_factory=list)

    def record_after_tool(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        output_raw: str,
        is_error: bool,
        exit_code: int | None = None,
        current_diff: str = "",
    ) -> ToolRecord:
        self.step_index += 1
        ws = pathlib.Path(self.workspace_root).resolve()

        if tool_name == "read_file" and not is_error:
            raw_path = str(tool_args.get("path", ""))
            if raw_path:
                norm = self._normalize_path(ws, raw_path)
                if norm not in self.read_files:
                    self.read_files.append(norm)

        if tool_name in ("edit_file", "write_file") and not is_error:
            raw_path = str(tool_args.get("path", ""))
            if raw_path:
                norm = self._normalize_path(ws, raw_path)
                if norm not in self.modified_files:
                    self.modified_files.append(norm)
                if norm not in self.read_files:
                    self.read_files.append(norm)
            self.last_edit_step = self.step_index

        if tool_name in ("exec_pty", "bash", "run_command"):
            cmd = str(tool_args.get("command", ""))
            if VERIFICATION_CMD_RE.search(cmd):
                self.last_verification_step = self.step_index
                passed = (not is_error) and (exit_code in (None, 0)) and ("FAILED" not in output_raw)
                self.last_verification_passed = passed
                self.last_verification_output = output_raw[-4000:]

        act_sig = compute_action_signature(tool_name, tool_args)
        err_sig = compute_error_signature(tool_name, output_raw, is_error)
        diff_hash = sha256_short(current_diff) if current_diff.strip() else ""

        rec = ToolRecord(
            step_index=self.step_index,
            tool_name=tool_name,
            tool_args=tool_args,
            output_raw=output_raw[-4000:],
            is_error=is_error,
            exit_code=exit_code,
            action_signature=act_sig,
            error_signature=err_sig,
            diff_state_hash=diff_hash,
        )
        self.tool_history.append(rec)
        if len(self.tool_history) > 64:
            self.tool_history = self.tool_history[-64:]
        return rec

    def to_evaluation_frame(
        self,
        event: HookEvent,
        *,
        tool_name: str | None = None,
        tool_args: dict[str, Any] | None = None,
        tool_output_raw: str = "",
        tool_is_error: bool = False,
        tool_exit_code: int | None = None,
        unified_git_diff: str | None = None,
        target_paths: list[str] | None = None,
        actor_claim_text: str | None = None,
        actor_model_family: str | None = None,
    ) -> EvaluationFrame:
        ws = pathlib.Path(self.workspace_root).resolve()
        diff = unified_git_diff if unified_git_diff is not None else capture_git_diff(ws, target_paths)
        mod_files = list(self.modified_files)
        if target_paths:
            for tp in target_paths:
                norm = self._normalize_path(ws, tp)
                if norm not in mod_files:
                    mod_files.append(norm)
        elif not mod_files:
            mod_files = discover_git_modified_files(ws)

        return EvaluationFrame(
            session_id=self.session_id,
            workspace_root=str(ws),
            event=event,
            step_index=self.step_index,
            task_objective=self.task_objective,
            declared_invariants=tuple(self.declared_invariants),
            tool_name=tool_name,
            tool_args=dict(tool_args or {}),
            tool_output_raw=tool_output_raw,
            tool_is_error=tool_is_error,
            tool_exit_code=tool_exit_code,
            unified_git_diff=diff,
            modified_files=tuple(mod_files),
            read_files=tuple(self.read_files),
            tool_history=tuple(self.tool_history),
            last_edit_step=self.last_edit_step,
            last_verification_step=self.last_verification_step,
            last_verification_passed=self.last_verification_passed,
            last_verification_output=self.last_verification_output,
            actor_claim_text=actor_claim_text,
            actor_model_family=actor_model_family,
        )

    @staticmethod
    def _normalize_path(ws: pathlib.Path, raw_path: str) -> str:
        p = pathlib.Path(raw_path)
        if not p.is_absolute():
            p = (ws / p).resolve()
        else:
            p = p.resolve()
        try:
            return str(p.relative_to(ws))
        except ValueError:
            return str(p)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "workspace_root": self.workspace_root,
            "task_objective": self.task_objective,
            "declared_invariants": self.declared_invariants,
            "step_index": self.step_index,
            "read_files": self.read_files,
            "modified_files": self.modified_files,
            "last_edit_step": self.last_edit_step,
            "last_verification_step": self.last_verification_step,
            "last_verification_passed": self.last_verification_passed,
            "last_verification_output": self.last_verification_output,
            "tool_history": [r.to_dict() for r in self.tool_history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionTelemetry:
        return cls(
            session_id=str(data.get("session_id", "default")),
            workspace_root=str(data.get("workspace_root", ".")),
            task_objective=str(data.get("task_objective", "")),
            declared_invariants=list(data.get("declared_invariants") or []),
            step_index=int(data.get("step_index", 0)),
            read_files=list(data.get("read_files") or []),
            modified_files=list(data.get("modified_files") or []),
            last_edit_step=int(data.get("last_edit_step", 0)),
            last_verification_step=int(data.get("last_verification_step", 0)),
            last_verification_passed=data.get("last_verification_passed"),
            last_verification_output=str(data.get("last_verification_output", "")),
            tool_history=[ToolRecord.from_dict(x) for x in (data.get("tool_history") or [])],
        )


class HarnessTelemetryStore:
    """File-backed session telemetry store under `.ush/evaluator_state/`."""

    def __init__(self, state_dir: pathlib.Path) -> None:
        self.state_dir = state_dir.expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, session_id: str) -> pathlib.Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]", "_", session_id or "default")
        return self.state_dir / f"{safe_id}.json"

    def load(self, session_id: str, workspace_root: str) -> SessionTelemetry:
        fp = self._path_for(session_id)
        if fp.is_file():
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
                return SessionTelemetry.from_dict(data)
            except (OSError, json.JSONDecodeError):
                pass
        return SessionTelemetry(session_id=session_id, workspace_root=workspace_root)

    def save(self, telemetry: SessionTelemetry) -> None:
        fp = self._path_for(telemetry.session_id)
        fp.write_text(json.dumps(telemetry.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
