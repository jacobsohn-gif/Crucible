"""9-Event Hook Engine, 4-Scope + JIT Memory Inbox (`AGENTS.md`/`GEMINI.md`) & Extension Manager.

Implements CT-16 and CT-17:
1. `HookSystem`: `SessionStart`, `UserPromptSubmit`, `BeforeToolSelection`, `BeforeModel`, `AfterModel`,
   `BeforeTool`, `AfterTool`, `PreCompress`, and `Stop` (`exit 2` veto). Workspace hooks ONLY run when trusted!
2. `MemoryManager`: Global -> Extension -> Workspace Root -> JIT Subdirectory `USH.md`/`AGENTS.md`/`GEMINI.md`/`CLAUDE.md`
   loader with `(st_dev, st_ino)` symlink deduplication + untrusted-transcript secret-redacting `.inbox/*.patch` staging.
3. `ExtensionManager`: Loads & installs `ush-extension.json` and `gemini-extension.json` packages (clamping policies).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import difflib
import json
import os
import pathlib
import re
import shutil
import subprocess
import time
from typing import Any, Callable

from ush.core.types import CanonicalPart, ModelCapabilityProfile
from ush.kernel.security import PolicyEngine, redact_secrets, scrub_child_environment

HOOK_EVENTS: frozenset[str] = frozenset(
    {
        "SessionStart",
        "UserPromptSubmit",
        "BeforeToolSelection",
        "BeforeModel",
        "AfterModel",
        "BeforeTool",
        "AfterTool",
        "PreCompress",
        "Stop",
    }
)

MEMORY_FILENAMES: tuple[str, ...] = ("USH.md", "AGENTS.md", "GEMINI.md", "CLAUDE.md")


# ============================================================================
# 9-Event Lifecycle Hook System (CT-16)
# ============================================================================


@dataclass
class ShellHookSpec:
    event: str
    command: str
    matcher: str = "*"
    timeout_seconds: float = 10.0
    source: str = "user"


class HookSystem:
    """Executes shell (`hooks.json`) and in-process Python lifecycle hooks across all 9 agent loop events."""

    def __init__(self, workspace_root: pathlib.Path, *, is_workspace_trusted: bool = False) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.is_workspace_trusted = is_workspace_trusted
        self.shell_hooks: list[ShellHookSpec] = []
        self._tool_filter_callbacks: list[
            Callable[[set[str], ModelCapabilityProfile], set[str]]
        ] = []

    def add_tool_selection_filter(
        self, fn: Callable[[set[str], ModelCapabilityProfile], set[str]]
    ) -> None:
        self._tool_filter_callbacks.append(fn)

    def load_hooks_json(self, path: pathlib.Path, *, source: str) -> int:
        if source == "workspace" and not self.is_workspace_trusted:
            return 0
        if not path.is_file():
            return 0
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        hooks_block = data.get("hooks", data)
        if not isinstance(hooks_block, dict):
            return 0
        count = 0
        for event, entries in hooks_block.items():
            if event not in HOOK_EVENTS or not isinstance(entries, list):
                continue
            for entry in entries:
                matcher = str(entry.get("matcher", "*"))
                inner = entry.get("hooks", [entry])
                for h in inner:
                    cmd = h.get("command")
                    if isinstance(cmd, str) and cmd.strip():
                        self.shell_hooks.append(
                            ShellHookSpec(
                                event=event,
                                command=cmd.strip(),
                                matcher=matcher,
                                timeout_seconds=float(h.get("timeout", 10.0)),
                                source=source,
                            )
                        )
                        count += 1
        return count

    def _run_matching(self, event: str, target_name: str, payload: dict[str, Any]) -> tuple[int, str, dict[str, Any]]:
        notes: list[str] = []
        merged_json: dict[str, Any] = {}
        env = scrub_child_environment()
        stdin_str = json.dumps({"event": event, **payload})

        cwd_dir = self.workspace_root if self.workspace_root.is_dir() else pathlib.Path.cwd()
        for hk in self.shell_hooks:
            if hk.event != event:
                continue
            if hk.matcher != "*":
                try:
                    if not re.search(hk.matcher, target_name):
                        continue
                except re.error:
                    continue
            try:
                res = subprocess.run(
                    ["/bin/sh", "-c", hk.command],
                    input=stdin_str,
                    cwd=str(cwd_dir),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=hk.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                notes.append(f"[Hook timeout: {hk.event}]")
                continue
            except (subprocess.SubprocessError, OSError) as exc:
                notes.append(f"[Hook process error ({hk.event}): {exc}]")
                continue

            out = res.stdout.strip()
            err = res.stderr.strip()
            # Exit code 2 = explicit deterministic VETO across Claude/Gemini/Codex hook conventions!
            if res.returncode == 2:
                return 2, (err or out or f"Vetoed by {hk.event} hook"), merged_json
            if res.returncode != 0 and err and not out:
                notes.append(f"[Hook {hk.event} exited rc={res.returncode}: {err[:240]}]")
                continue
            if out:
                try:
                    parsed = json.loads(out)
                    if isinstance(parsed, dict):
                        merged_json.update(parsed)
                        if parsed.get("decision") in ("deny", "block"):
                            return 2, str(parsed.get("reason", "Blocked by hook")), merged_json
                        extra = parsed.get("additionalContext") or parsed.get("systemMessage")
                        if extra:
                            notes.append(str(extra))
                        continue
                except json.JSONDecodeError:
                    notes.append(out[:1000])
        return 0, "\n".join(notes), merged_json

    def fire(self, event: str, payload: dict[str, Any]) -> tuple[bool, str]:
        rc, msg, _ = self._run_matching(event, "*", payload)
        return rc != 2, msg

    def filter_tools(
        self,
        current_allowed: set[str] | None,
        all_tools: set[str],
        profile: ModelCapabilityProfile,
    ) -> set[str] | None:
        active = set(current_allowed) if current_allowed is not None else set(all_tools)
        for fn in self._tool_filter_callbacks:
            active = fn(active, profile)
        _, _, meta = self._run_matching(
            "BeforeToolSelection",
            profile.model_id,
            {"tools": sorted(active), "tier": profile.tier.value},
        )
        if isinstance(meta.get("allowedTools"), list):
            active &= {str(x) for x in meta["allowedTools"]}
        return None if (current_allowed is None and active == all_tools) else active

    def before_tool(self, tool_name: str, args: dict[str, Any]) -> tuple[bool, dict[str, Any], str]:
        rc, msg, meta = self._run_matching("BeforeTool", tool_name, {"tool_name": tool_name, "tool_input": args})
        if rc == 2:
            return False, args, msg
        updated = meta.get("updatedInput")
        if isinstance(updated, dict):
            args = {**args, **updated}
        return True, args, msg

    def after_tool(self, tool_name: str, result_part: CanonicalPart) -> str:
        _, msg, _ = self._run_matching(
            "AfterTool",
            tool_name,
            {"tool_name": tool_name, "is_error": result_part.is_error, "preview": result_part.text[:500]},
        )
        return msg


# ============================================================================
# 4-Scope + JIT Subdirectory Memory & `.inbox/*.patch` Auto-Extraction (CT-17)
# ============================================================================


@dataclass
class MemoryPatchProposal:
    patch_id: str
    path: pathlib.Path
    target_file: pathlib.Path
    diff_text: str
    proposed_content: str


class MemoryManager:
    """Loads global -> extension -> root -> JIT subdirectory memory with inode deduplication & `.inbox/` patches."""

    def __init__(
        self,
        workspace_root: pathlib.Path,
        *,
        ush_home: pathlib.Path | None = None,
        extension_context_files: list[pathlib.Path] | None = None,
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.ush_home = (ush_home or pathlib.Path("~/.ush")).expanduser().resolve()
        self.memory_dir = self.ush_home / "memory"
        self.memory_index_path = self.memory_dir / "MEMORY.md"
        self.inbox_dir = self.memory_dir / ".inbox"
        self.extension_context_files = list(extension_context_files or [])
        self._seen_inodes: set[tuple[int, int]] = set()
        self.loaded_paths: list[pathlib.Path] = []

    def _claim_inode(self, path: pathlib.Path) -> bool:
        try:
            st = path.stat()  # Follows symlinks -> deduplicates symlinked AGENTS.md <-> GEMINI.md! (CT-17)
        except OSError:
            return False
        ident = (st.st_dev, st.st_ino)
        if ident in self._seen_inodes:
            return False
        self._seen_inodes.add(ident)
        self.loaded_paths.append(path.resolve())
        return True

    def _read_with_imports(self, path: pathlib.Path, depth: int = 0) -> str:
        if depth > 3 or not self._claim_inode(path):
            return ""
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        out: list[str] = [f"<!-- source: {path} -->"]
        for line in raw.splitlines():
            stripped = line.strip()
            # `@relative/file.md` safe workspace-bounded include
            if stripped.startswith("@") and stripped.endswith(".md") and " " not in stripped:
                inc = (path.parent / stripped[1:]).resolve()
                if inc.is_file() and inc.is_relative_to(path.parent.resolve()):
                    sub = self._read_with_imports(inc, depth + 1)
                    if sub:
                        out.append(sub)
                        continue
            out.append(line)
        return "\n".join(out)[:12_000]

    def initial_memory_block(self) -> str:
        self._seen_inodes.clear()
        self.loaded_paths.clear()
        chunks: list[str] = []

        # Scope 1: User Global (`~/.ush/USH.md`, `~/.ush/memory/MEMORY.md`)
        for gp in (self.ush_home / "USH.md", self.memory_index_path):
            if gp.is_file():
                c = self._read_with_imports(gp)
                if c:
                    chunks.append(c)

        # Scope 2: Active Extensions
        for ep in self.extension_context_files:
            if ep.is_file():
                c = self._read_with_imports(ep)
                if c:
                    chunks.append(c)

        # Scope 3: Workspace Root (`USH.md`, `AGENTS.md`, `GEMINI.md`, `CLAUDE.md`)
        for fname in MEMORY_FILENAMES:
            fp = self.workspace_root / fname
            if fp.is_file():
                c = self._read_with_imports(fp)
                if c:
                    chunks.append(c)

        return "\n\n".join(chunks)

    def on_file_accessed(self, accessed_path: pathlib.Path) -> str:
        """Scope 4: JIT loads package-local `AGENTS.md`/`GEMINI.md` strictly once when package files are read (CT-17)."""
        resolved = accessed_path.resolve()
        if not resolved.is_relative_to(self.workspace_root):
            return ""
        dirs_to_check: list[pathlib.Path] = []
        cur = resolved.parent
        while cur != self.workspace_root and cur.is_relative_to(self.workspace_root):
            dirs_to_check.append(cur)
            cur = cur.parent
        dirs_to_check.reverse()

        discovered: list[str] = []
        for d in dirs_to_check:
            for fname in MEMORY_FILENAMES:
                cand = d / fname
                if cand.is_file():
                    txt = self._read_with_imports(cand)
                    if txt:
                        discovered.append(txt)
        if not discovered:
            return ""
        return "<jit_subdirectory_memory>\n" + "\n\n".join(discovered) + "\n</jit_subdirectory_memory>"

    # ---- Untrusted-Transcript-Safe `.inbox/*.patch` Extraction (`/learn`, `/memory inbox`) ----

    def stage_learning_patch(self, title: str, learned_bullets: list[str]) -> MemoryPatchProposal:
        """Redacts secrets and stages a unified `.patch` into `.inbox/` WITHOUT mutating `MEMORY.md` directly (CT-17)."""
        self.inbox_dir.mkdir(parents=True, exist_ok=True)
        safe_title = redact_secrets(title).strip()[:120]
        safe_bullets = [f"- {redact_secrets(b).strip()}" for b in learned_bullets if b.strip()]

        old_text = (
            self.memory_index_path.read_text(encoding="utf-8")
            if self.memory_index_path.is_file()
            else "# USH Durable Memory Index\n"
        )
        section = f"\n## {safe_title}\n" + "\n".join(safe_bullets) + "\n"
        new_text = old_text.rstrip("\n") + "\n" + section
        diff = "\n".join(
            difflib.unified_diff(
                old_text.splitlines(),
                new_text.splitlines(),
                fromfile="a/MEMORY.md",
                tofile="b/MEMORY.md",
                lineterm="",
            )
        )
        pid = f"learn_{int(time.time() * 1000)}"
        patch_file = self.inbox_dir / f"{pid}.patch.json"
        patch_file.write_text(
            json.dumps(
                {
                    "patch_id": pid,
                    "target_file": str(self.memory_index_path),
                    "diff_text": diff,
                    "proposed_content": new_text,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return MemoryPatchProposal(
            patch_id=pid,
            path=patch_file,
            target_file=self.memory_index_path,
            diff_text=diff,
            proposed_content=new_text,
        )

    def list_inbox(self) -> list[MemoryPatchProposal]:
        if not self.inbox_dir.is_dir():
            return []
        items: list[MemoryPatchProposal] = []
        for fp in sorted(self.inbox_dir.glob("*.patch.json")):
            try:
                d = json.loads(fp.read_text(encoding="utf-8"))
                items.append(
                    MemoryPatchProposal(
                        patch_id=d["patch_id"],
                        path=fp,
                        target_file=pathlib.Path(d["target_file"]),
                        diff_text=d["diff_text"],
                        proposed_content=d["proposed_content"],
                    )
                )
            except (OSError, KeyError, json.JSONDecodeError):
                continue
        return items

    def apply_inbox_patch(self, patch_id: str) -> bool:
        for prop in self.list_inbox():
            if prop.patch_id == patch_id:
                prop.target_file.parent.mkdir(parents=True, exist_ok=True)
                prop.target_file.write_text(prop.proposed_content, encoding="utf-8")
                prop.path.unlink(missing_ok=True)
                return True
        return False


# ============================================================================
# Extension Bundle Loader (`ush-extension.json` + `gemini-extension.json`)
# ============================================================================


@dataclass
class LoadedExtension:
    name: str
    version: str
    root: pathlib.Path
    skill_dir: pathlib.Path | None
    command_dir: pathlib.Path | None
    context_file: pathlib.Path | None


class ExtensionManager:
    """Loads & installs extension bundles combining skills, commands, hooks, and privilege-clamped policies."""

    def __init__(self, ush_home: pathlib.Path) -> None:
        self.ext_root = ush_home.expanduser().resolve() / "extensions"
        self.extensions: list[LoadedExtension] = []

    def discover_and_wire(self, policy_engine: PolicyEngine, hook_system: HookSystem) -> list[LoadedExtension]:
        self.extensions.clear()
        if not self.ext_root.is_dir():
            return []
        for child in sorted(self.ext_root.iterdir()):
            if not child.is_dir():
                continue
            manifest = next(
                (child / m for m in ("ush-extension.json", "gemini-extension.json") if (child / m).is_file()),
                None,
            )
            if not manifest:
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            name = str(data.get("name") or child.name)
            ver = str(data.get("version") or "0.1.0")
            sdir = child / "skills" if (child / "skills").is_dir() else None
            cdir = child / "commands" if (child / "commands").is_dir() else None
            ctx_name = str(data.get("contextFileName") or "USH.md")
            ctx_file = (
                child / ctx_name
                if (child / ctx_name).is_file()
                else (child / "GEMINI.md" if (child / "GEMINI.md").is_file() else None)
            )

            # Wire privilege-clamped extension policies (`allow`/`yolo` ignored!)
            pol_dir = child / "policies"
            if pol_dir.is_dir():
                for toml_f in sorted(pol_dir.glob("*.toml")):
                    policy_engine.load_toml_file(toml_f, tier="extension")

            hooks_f = child / "hooks" / "hooks.json"
            if hooks_f.is_file():
                hook_system.load_hooks_json(hooks_f, source=f"ext:{name}")

            self.extensions.append(LoadedExtension(name, ver, child, sdir, cdir, ctx_file))
        return self.extensions

    def install_local(self, source_dir: pathlib.Path) -> LoadedExtension:
        src = source_dir.expanduser().resolve()
        manifest = next(
            (src / m for m in ("ush-extension.json", "gemini-extension.json") if (src / m).is_file()),
            None,
        )
        if not manifest:
            raise FileNotFoundError(f"No `ush-extension.json` or `gemini-extension.json` found in `{src}`.")
        data = json.loads(manifest.read_text(encoding="utf-8"))
        slug = re.sub(r"[^a-zA-Z0-9_-]", "-", str(data.get("name") or src.name))
        dest = self.ext_root / slug
        self.ext_root.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        return LoadedExtension(slug, str(data.get("version", "0.1.0")), dest, None, None, None)
