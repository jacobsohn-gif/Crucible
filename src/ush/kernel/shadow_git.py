"""Out-of-Tree Shadow Git Checkpointer & Isolated Swarm Git Worktree Pool.

Implements CT-09, CT-18 (`clean_patch`), and CT-19 (`git merge-tree` + isolated worktrees):
- Stores snapshots inside `~/.ush/snapshots/<repo_hash>/` using separate `GIT_DIR` + `GIT_INDEX_FILE`
  pointed at `--work-tree=<workspace>` with `core.hooksPath=/dev/null` — leaving the user's real `.git/index`,
  mtime, staged files, and `git status` 100% untouched!
- Computes test-stripped `clean_patch` & flags test-assertion tampering for Stage-A Judges.
- Manages temporary Git worktrees for Gear 3/4 Swarm islands and `/arena` races.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

from ush.core.types import sha256_hex

TEST_PATH_RE = re.compile(
    r"(?:^|/)(?:tests?|__tests__|spec)/|(?:^|/)test_[^/]+\.[a-z0-9]+$|_test\.[a-z0-9]+$|\.spec\.[a-z0-9]+$",
    re.IGNORECASE,
)
ASSERTION_DELETE_RE = re.compile(
    r"^-\s*(?:assert\b|self\.assert|expect\(|pytest\.raises|should\.|assert_eq!|assert!)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class ShadowSnapshot:
    commit_sha: str
    tree_sha: str
    label: str


@dataclass(frozen=True)
class CleanPatchAudit:
    full_diff: str
    impl_only_diff: str
    touched_test_files: list[str]
    deleted_assertions: list[str]

    @property
    def tampered_with_existing_tests(self) -> bool:
        return bool(self.deleted_assertions)


class ShadowGitStore:
    """Zero-pollution out-of-tree snapshot engine backing `/restore`, `/diff`, and Judge `clean_patch`."""

    def __init__(self, workspace_root: pathlib.Path, ush_home: pathlib.Path | None = None) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        home = (ush_home or pathlib.Path(os.environ.get("USH_HOME", "~/.ush"))).expanduser().resolve()
        ws_hash = sha256_hex(str(self.workspace_root))[:20]
        self.git_dir = home / "snapshots" / ws_hash / "repo.git"
        self.index_file = home / "snapshots" / ws_hash / "shadow.index"
        self._initialized = False
        self.history: list[ShadowSnapshot] = []

    def _git_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env.update(
            {
                "GIT_DIR": str(self.git_dir),
                "GIT_WORK_TREE": str(self.workspace_root),
                "GIT_INDEX_FILE": str(self.index_file),
                "GIT_AUTHOR_NAME": "USH Shadow Snapshot",
                "GIT_AUTHOR_EMAIL": "shadow@ush.local",
                "GIT_COMMITTER_NAME": "USH Shadow Snapshot",
                "GIT_COMMITTER_EMAIL": "shadow@ush.local",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
        return env

    def _run(self, args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
        res = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=str(self.workspace_root),
            env=self._git_env(),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and res.returncode != 0:
            raise RuntimeError(f"Shadow git `{' '.join(args)}` failed: {res.stderr.strip()}")
        return res

    def ensure_initialized(self) -> None:
        if self._initialized and (self.git_dir / "HEAD").is_file():
            return
        self.git_dir.mkdir(parents=True, exist_ok=True)
        if not (self.git_dir / "HEAD").is_file():
            subprocess.run(
                ["git", "init", "--bare", "--quiet", str(self.git_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
            self._run(["config", "core.hooksPath", "/dev/null"])
            self._run(["config", "gc.auto", "0"])
        # Exclude nested `.git` directories, `.venv`, `node_modules`, and `.ush/sessions`
        excl = self.git_dir / "info" / "exclude"
        excl.parent.mkdir(parents=True, exist_ok=True)
        excl.write_text(
            ".git\n.git/\n.venv/\nnode_modules/\n__pycache__/\n*.pyc\n.ush/sessions/\n.ush/worktrees/\n",
            encoding="utf-8",
        )
        self._initialized = True

    def capture(self, label: str = "step") -> ShadowSnapshot:
        self.ensure_initialized()
        self._run(["add", "-A", "--", "."])
        tree_sha = self._run(["write-tree"]).stdout.strip()
        parent_args: list[str] = []
        head_check = self._run(["rev-parse", "--verify", "HEAD"], check=False)
        if head_check.returncode == 0 and head_check.stdout.strip():
            parent_args = ["-p", head_check.stdout.strip()]
        commit_sha = self._run(["commit-tree", tree_sha, *parent_args, "-m", label]).stdout.strip()
        self._run(["update-ref", "HEAD", commit_sha])
        snap = ShadowSnapshot(commit_sha=commit_sha, tree_sha=tree_sha, label=label)
        self.history.append(snap)
        return snap

    def diff_from(self, snapshot: ShadowSnapshot | str) -> str:
        self.ensure_initialized()
        cur = self.capture("ephemeral-diff")
        base_sha = snapshot.commit_sha if isinstance(snapshot, ShadowSnapshot) else snapshot
        return self._run(["diff", base_sha, cur.commit_sha]).stdout

    def restore(self, snapshot: ShadowSnapshot | str) -> list[str]:
        """Restores tracked workspace files & deletes untracked creations added after `snapshot` (CT-09)."""
        self.ensure_initialized()
        target_sha = snapshot.commit_sha if isinstance(snapshot, ShadowSnapshot) else snapshot
        cur = self.capture("pre-restore")

        # Identify files newly added after target_sha so we remove them cleanly too
        name_status = self._run(["diff", "--name-status", target_sha, cur.commit_sha]).stdout.splitlines()
        created_since: list[pathlib.Path] = []
        touched: list[str] = []
        for line in name_status:
            parts = line.split("\t")
            if len(parts) >= 2:
                status, rel = parts[0], parts[-1]
                touched.append(rel)
                if status.startswith("A"):
                    p = (self.workspace_root / rel).resolve()
                    if p.is_relative_to(self.workspace_root):
                        created_since.append(p)

        # Checkout exact files from target snapshot into work-tree via our out-of-tree shadow index
        self._run(["read-tree", target_sha])
        self._run(["checkout-index", "-f", "-a"])
        for added_file in created_since:
            if added_file.is_file() or added_file.is_symlink():
                added_file.unlink(missing_ok=True)
        self._run(["update-ref", "HEAD", target_sha])
        return touched

    @staticmethod
    def audit_clean_patch(unified_diff: str) -> CleanPatchAudit:
        """Splits a candidate patch into implementation-only diff vs test tampering audit (CT-18)."""
        impl_chunks: list[str] = []
        touched_tests: list[str] = []
        deleted_asserts: list[str] = []

        # Split by `diff --git a/... b/...` file sections
        sections = re.split(r"(?=^diff --git )", unified_diff, flags=re.MULTILINE)
        for sec in sections:
            if not sec.strip():
                continue
            m = re.match(r"diff --git a/(.+?) b/(.+)", sec.splitlines()[0])
            rel_path = m.group(2) if m else ""
            if TEST_PATH_RE.search(rel_path):
                touched_tests.append(rel_path)
                for dm in ASSERTION_DELETE_RE.finditer(sec):
                    deleted_asserts.append(f"{rel_path}: {dm.group(0).strip()}")
            else:
                impl_chunks.append(sec)

        return CleanPatchAudit(
            full_diff=unified_diff,
            impl_only_diff="".join(impl_chunks),
            touched_test_files=touched_tests,
            deleted_assertions=deleted_asserts,
        )


# ============================================================================
# Isolated Git Worktree Pool for Gear 3/4 Swarms & `/arena` (CT-19)
# ============================================================================


@dataclass
class ManagedWorktree:
    worker_id: str
    branch_name: str
    path: pathlib.Path
    base_commit: str


class GitWorktreePool:
    """Spawns isolated Git worktrees (`core.hooksPath=/dev/null`) and performs 3-way semantic/clean merges."""

    def __init__(self, repo_root: pathlib.Path, pool_dir: pathlib.Path | None = None) -> None:
        self.repo_root = repo_root.expanduser().resolve()
        self.pool_dir = (
            pool_dir
            or pathlib.Path(tempfile.gettempdir()) / f"ush_worktrees_{sha256_hex(str(self.repo_root))[:12]}"
        ).resolve()

    def _git(self, args: list[str], cwd: pathlib.Path | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
        res = subprocess.run(
            ["git", "-c", "core.hooksPath=/dev/null", *args],
            cwd=str(cwd or self.repo_root),
            capture_output=True,
            text=True,
            timeout=30,
        )
        if check and res.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {res.stderr.strip()}")
        return res

    def create(self, worker_id: str) -> ManagedWorktree:
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        safe_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", worker_id)
        branch = f"ush/swarm/{safe_slug}"
        wt_path = self.pool_dir / safe_slug
        if wt_path.exists():
            self._git(["worktree", "remove", "--force", str(wt_path)], check=False)
            shutil.rmtree(wt_path, ignore_errors=True)
        self._git(["branch", "-D", branch], check=False)
        base = self._git(["rev-parse", "HEAD"]).stdout.strip()
        self._git(["worktree", "add", "-b", branch, str(wt_path), base])
        self._git(["config", "core.hooksPath", "/dev/null"], cwd=wt_path, check=False)
        return ManagedWorktree(worker_id=worker_id, branch_name=branch, path=wt_path, base_commit=base)

    def commit_worktree(self, wt: ManagedWorktree, message: str = "swarm shard") -> str:
        self._git(["add", "-A"], cwd=wt.path)
        status = self._git(["status", "--porcelain"], cwd=wt.path).stdout.strip()
        if status:
            self._git(
                [
                    "-c",
                    "user.name=USH Swarm Worker",
                    "-c",
                    "user.email=swarm@ush.local",
                    "commit",
                    "-m",
                    message,
                ],
                cwd=wt.path,
            )
        return self._git(["rev-parse", "HEAD"], cwd=wt.path).stdout.strip()

    def worktree_diff(self, wt: ManagedWorktree) -> str:
        head = self.commit_worktree(wt, message=f"checkpoint {wt.worker_id}")
        return self._git(["diff", wt.base_commit, head]).stdout

    def can_merge_cleanly(self, branch_a: str, branch_b: str) -> tuple[bool, str]:
        """Non-destructive in-memory 3-way conflict check via `git merge-tree --write-tree` (Git >= 2.38)."""
        res = self._git(["merge-tree", "--write-tree", branch_a, branch_b], check=False)
        return res.returncode == 0, (res.stdout + "\n" + res.stderr).strip()

    def merge_into_current(self, wt: ManagedWorktree) -> tuple[bool, str]:
        self.commit_worktree(wt, message=f"finalize {wt.worker_id}")
        res = self._git(
            [
                "-c",
                "user.name=USH Integrator",
                "-c",
                "user.email=integrator@ush.local",
                "merge",
                "--no-edit",
                wt.branch_name,
            ],
            check=False,
        )
        if res.returncode != 0:
            self._git(["merge", "--abort"], check=False)
            return False, (res.stdout + "\n" + res.stderr).strip()
        return True, res.stdout.strip()

    def cleanup(self, wt: ManagedWorktree) -> None:
        self._git(["worktree", "remove", "--force", str(wt.path)], check=False)
        shutil.rmtree(wt.path, ignore_errors=True)
        self._git(["branch", "-D", wt.branch_name], check=False)
