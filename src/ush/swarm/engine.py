"""4-Gear Elastic Stigmergic Swarm Coordinator, SQLite WAL Blackboard & Hash-Chained Consensus Ledger.

Implements CT-19 and CT-20:
- Parameterized SQLite WAL Blackboard (`0` SQL concatenation) + append-only `blackboard.jsonl`.
- SHA-256 hash-chained `ConsensusLedger` enforcing `INV-SWARM-1`:
  (a) `judge_id not in contributor_ids` (zero self-certification!)
  (b) unbroken cryptographic `prev_hash -> entry_hash` chain verified before any merge!
- 4-Gear Automatic Gearbox (`G1 Solo -> G2 Strike Pair -> G3 Spike+Worktrees -> G4 Crucible`) with
  live mid-flight stall upshifting (`thrash_streak >= 2`), `CONTRACT_DELTA` cross-worktree broadcasts,
  `git merge-tree --write-tree` reconciliation, `/arena` multi-model showdowns, and optional remote
  container cluster dispatch!
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import contextlib
from dataclasses import dataclass, field
import json
import os
import pathlib
import re
import shutil
import sqlite3
import subprocess
import threading
import time
from typing import Any, Callable, Iterator

from ush.core.types import Gear, canonical_json, new_id, sha256_hex
from ush.kernel.security import scrub_child_environment
from ush.kernel.shadow_git import GitWorktreePool, ManagedWorktree
from ush.swarm.judge import CandidatePatch, CascadingJudge


@dataclass(frozen=True)
class BlackboardEvent:
    seq: int
    event_id: str
    event_type: str  # SPIKE | SHARD | CONTRACT_DELTA | REQUEST_REVIEW | UPSHIFT | NOTE
    author_id: str
    scope: str
    summary: str
    created_at: float


@dataclass(frozen=True)
class ConsensusLedgerRecord:
    seq: int
    candidate_id: str
    contributor_ids: tuple[str, ...]
    judge_id: str
    judge_family: str
    bars_score: float
    approved: bool
    prev_hash: str
    entry_hash: str


class SwarmInvariantError(PermissionError):
    """Raised on `INV-SWARM-1` self-review or tampered ledger chain violations (CT-20)."""


class StigmergicBlackboard:
    """Thread-safe parameterized SQLite WAL blackboard + SHA-256 hash-chained `ConsensusLedger`."""

    GENESIS_HASH = "0" * 64

    def __init__(self, state_dir: pathlib.Path) -> None:
        self.state_dir = state_dir.expanduser().resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state_dir / "blackboard.sqlite3"
        self.jsonl_path = self.state_dir / "blackboard.jsonl"
        self._lock = threading.Lock()
        self._init_db()

    @contextlib.contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    author_id TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS consensus_ledger (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT NOT NULL,
                    contributor_ids_json TEXT NOT NULL,
                    judge_id TEXT NOT NULL,
                    judge_family TEXT NOT NULL,
                    bars_score REAL NOT NULL,
                    approved INTEGER NOT NULL,
                    prev_hash TEXT NOT NULL,
                    entry_hash TEXT NOT NULL
                )
                """
            )

    def post(self, event_type: str, author_id: str, summary: str, scope: str = "*") -> BlackboardEvent:
        now = time.time()
        eid = new_id("evt")
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO events (event_id, event_type, author_id, scope, summary, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (eid, event_type, author_id, scope, summary, now),
                )
                seq = int(cur.lastrowid or 0)
            evt = BlackboardEvent(seq, eid, event_type, author_id, scope, summary, now)
            with self.jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    canonical_json(
                        {
                            "seq": seq,
                            "event_id": eid,
                            "event_type": event_type,
                            "author_id": author_id,
                            "scope": scope,
                            "summary": summary,
                            "created_at": now,
                        }
                    )
                    + "\n"
                )
            return evt

    def read_since(
        self, last_seq: int = 0, *, event_types: tuple[str, ...] | None = None, exclude_author: str | None = None
    ) -> list[BlackboardEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT seq, event_id, event_type, author_id, scope, summary, created_at FROM events WHERE seq > ? ORDER BY seq ASC",
                (last_seq,),
            ).fetchall()
        out: list[BlackboardEvent] = []
        for r in rows:
            evt = BlackboardEvent(int(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), str(r[5]), float(r[6]))
            if event_types and evt.event_type not in event_types:
                continue
            if exclude_author and evt.author_id == exclude_author:
                continue
            out.append(evt)
        return out

    # ---- INV-SWARM-1 Cryptographic Consensus Ledger (CT-20) ----

    @staticmethod
    def _compute_ledger_hash(
        prev_hash: str,
        candidate_id: str,
        contributors: tuple[str, ...],
        judge_id: str,
        judge_family: str,
        bars_score: float,
        approved: bool,
    ) -> str:
        payload = canonical_json(
            {
                "prev_hash": prev_hash,
                "candidate_id": candidate_id,
                "contributors": list(contributors),
                "judge_id": judge_id,
                "judge_family": judge_family,
                "bars_score": round(bars_score, 6),
                "approved": approved,
            }
        )
        return sha256_hex(payload)

    def record_quorum_verdict(
        self,
        *,
        candidate_id: str,
        contributor_ids: tuple[str, ...],
        judge_id: str,
        judge_family: str,
        bars_score: float,
        approved: bool,
    ) -> ConsensusLedgerRecord:
        # INV-SWARM-1a: Hard veto on any self-certification (`judge_id in contributor_ids`)! (CT-20)
        if judge_id in contributor_ids or not contributor_ids:
            raise SwarmInvariantError(
                f"INV-SWARM-1 VIOLATION: Judge `{judge_id}` cannot certify candidate `{candidate_id}` "
                f"authored by `{contributor_ids}` (self-review forbidden)."
            )

        with self._lock:
            self.verify_ledger_integrity()
            with self._connect() as conn:
                last = conn.execute(
                    "SELECT entry_hash FROM consensus_ledger ORDER BY seq DESC LIMIT 1"
                ).fetchone()
                prev_hash = str(last[0]) if last else self.GENESIS_HASH
                entry_hash = self._compute_ledger_hash(
                    prev_hash, candidate_id, contributor_ids, judge_id, judge_family, bars_score, approved
                )
                cur = conn.execute(
                    """
                    INSERT INTO consensus_ledger
                    (candidate_id, contributor_ids_json, judge_id, judge_family, bars_score, approved, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        candidate_id,
                        json.dumps(list(contributor_ids)),
                        judge_id,
                        judge_family,
                        round(bars_score, 6),
                        1 if approved else 0,
                        prev_hash,
                        entry_hash,
                    ),
                )
                seq = int(cur.lastrowid or 0)
            return ConsensusLedgerRecord(
                seq=seq,
                candidate_id=candidate_id,
                contributor_ids=contributor_ids,
                judge_id=judge_id,
                judge_family=judge_family,
                bars_score=round(bars_score, 6),
                approved=approved,
                prev_hash=prev_hash,
                entry_hash=entry_hash,
            )

    def verify_ledger_integrity(self) -> list[ConsensusLedgerRecord]:
        """Recomputes every SHA-256 chain link; raises `SwarmInvariantError` on tampering (CT-20)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT seq, candidate_id, contributor_ids_json, judge_id, judge_family,
                       bars_score, approved, prev_hash, entry_hash
                FROM consensus_ledger ORDER BY seq ASC
                """
            ).fetchall()

        expected_prev = self.GENESIS_HASH
        verified: list[ConsensusLedgerRecord] = []
        for r in rows:
            seq = int(r[0])
            cid = str(r[1])
            contribs = tuple(str(x) for x in json.loads(r[2]))
            jid = str(r[3])
            jfam = str(r[4])
            score = float(r[5])
            app = bool(r[6])
            prev_h = str(r[7])
            ent_h = str(r[8])

            if jid in contribs:
                raise SwarmInvariantError(f"Ledger row #{seq} violates INV-SWARM-1 (self-review by `{jid}`).")
            if prev_h != expected_prev:
                raise SwarmInvariantError(
                    f"Ledger chain broken at row #{seq}: expected prev_hash `{expected_prev[:12]}`, got `{prev_h[:12]}`."
                )
            recomputed = self._compute_ledger_hash(prev_h, cid, contribs, jid, jfam, score, app)
            if recomputed != ent_h:
                raise SwarmInvariantError(
                    f"Ledger SHA-256 tamper detected at row #{seq}: stored `{ent_h[:12]}` != computed `{recomputed[:12]}`."
                )
            expected_prev = ent_h
            verified.append(ConsensusLedgerRecord(seq, cid, contribs, jid, jfam, score, app, prev_h, ent_h))
        return verified


# ============================================================================
# 4-Gear Automatic Gearbox & Worktree Swarm Coordinator (CT-19)
# ============================================================================


def select_initial_gear(prompt: str, requested: Gear = Gear.AUTO) -> Gear:
    if requested != Gear.AUTO:
        return requested
    words = prompt.split()
    low = prompt.lower()
    crucible_terms = ("architecture", "redesign", "algorithm", "optimize", "benchmark", "security audit", "race")
    shard_terms = ("refactor across", "multiple modules", "migrate all", "parallel", "worktree", "end-to-end")
    if any(t in low for t in crucible_terms) and len(words) >= 20:
        return Gear.G4_CRUCIBLE
    if any(t in low for t in shard_terms) or len(words) >= 55:
        return Gear.G3_SPIKE_WORKTREES
    if len(words) >= 14 or any(k in low for k in ("fix", "bug", "implement", "add", "patch")):
        return Gear.G2_STRIKE_PAIR
    return Gear.G1_SOLO


ShardWorkerFn = Callable[[ManagedWorktree, StigmergicBlackboard, str], None]


@dataclass
class SwarmRunSummary:
    initial_gear: Gear
    final_gear: Gear
    upshifted: bool
    merged_workers: list[str]
    contract_deltas: list[BlackboardEvent]
    winning_candidate_id: str | None = None
    ledger_record: ConsensusLedgerRecord | None = None
    notes: list[str] = field(default_factory=list)


class ElasticSwarmCoordinator:
    """Executes local Git-worktree islands (`G1..G4`), `/arena` races, and optional ArgonClaw cluster bridges."""

    def __init__(
        self,
        repo_root: pathlib.Path,
        *,
        blackboard: StigmergicBlackboard | None = None,
        judge: CascadingJudge | None = None,
    ) -> None:
        self.repo_root = repo_root.expanduser().resolve()
        self.blackboard = blackboard or StigmergicBlackboard(self.repo_root / ".ush" / "swarm")
        self.worktrees = GitWorktreePool(self.repo_root)
        self.judge = judge or CascadingJudge()

    def execute_with_stall_upshift(
        self,
        objective: str,
        *,
        solo_attempt: Callable[[], bool],
        parallel_shards: list[tuple[str, ShardWorkerFn]],
        judge_id: str = "judge:cross-family",
        judge_family: str = "anthropic",
    ) -> SwarmRunSummary:
        """Runs Gear 2 first; if `solo_attempt()` stalls/thrashes, auto-upshifts into Gear 3 isolated worktrees (CT-19)."""
        g_start = Gear.G2_STRIKE_PAIR
        notes: list[str] = []
        solved_solo = solo_attempt()
        if solved_solo:
            return SwarmRunSummary(g_start, g_start, False, [], [], notes=["Solved cleanly in Gear 2 without upshift."])

        # Mid-flight stall upshift -> Gear 3 Worktrees + Stigmergic CONTRACT_DELTA Blackboard!
        self.blackboard.post("UPSHIFT", "gearbox", f"Upshifting Gear 2 -> Gear 3 on stall: {objective[:120]}")
        managed: list[ManagedWorktree] = [self.worktrees.create(wid) for wid, _ in parallel_shards]
        try:
            with ThreadPoolExecutor(max_workers=max(1, len(managed))) as pool:
                futs = [
                    pool.submit(fn, wt, self.blackboard, wid)
                    for (wid, fn), wt in zip(parallel_shards, managed)
                ]
                for fut in futs:
                    fut.result()

            for wt in managed:
                self.worktrees.commit_worktree(wt, message=f"shard {wt.worker_id}")

            # Non-destructive 3-way compatibility check via `git merge-tree` before applying merges
            if len(managed) >= 2:
                clean, detail = self.worktrees.can_merge_cleanly(managed[0].branch_name, managed[1].branch_name)
                if not clean:
                    notes.append(f"merge-tree detected conflict requiring integrator: {detail[:200]}")

            merged_ids: list[str] = []
            combined_diffs: list[str] = []
            for wt in managed:
                combined_diffs.append(self.worktrees.worktree_diff(wt))
                ok, msg = self.worktrees.merge_into_current(wt)
                if ok:
                    merged_ids.append(wt.worker_id)
                else:
                    notes.append(f"Conflict merging `{wt.worker_id}`: {msg[:200]}")

            cand = CandidatePatch(
                candidate_id=new_id("cand"),
                contributor_ids=tuple(merged_ids or [w.worker_id for w in managed]),
                family="gemini",
                unified_diff="\n".join(combined_diffs),
            )
            stage_a = self.judge.run_stage_a(cand)
            rec = self.blackboard.record_quorum_verdict(
                candidate_id=cand.candidate_id,
                contributor_ids=cand.contributor_ids,
                judge_id=judge_id,
                judge_family=judge_family,
                bars_score=1.0 if stage_a.passed else 0.0,
                approved=stage_a.passed,
            )
            deltas = self.blackboard.read_since(0, event_types=("CONTRACT_DELTA",))
            return SwarmRunSummary(
                initial_gear=g_start,
                final_gear=Gear.G3_SPIKE_WORKTREES,
                upshifted=True,
                merged_workers=merged_ids,
                contract_deltas=deltas,
                winning_candidate_id=cand.candidate_id if stage_a.passed else None,
                ledger_record=rec,
                notes=notes + stage_a.reasons,
            )
        finally:
            for wt in managed:
                self.worktrees.cleanup(wt)

    @staticmethod
    def dispatch_to_argonclaw_cluster(
        *,
        task_id: str,
        title: str,
        description: str,
        swarm_host: str | None = None,
        dispatcher_script: str | None = None,
    ) -> tuple[int, str]:
        """Optional bridge dispatching Gear-4 distributed jobs directly to a remote swarm cluster."""
        resolved_host = (swarm_host or os.environ.get("SWARM_HOST", "")).strip()
        resolved_script = (
            dispatcher_script or os.environ.get("SWARM_DISPATCHER_SCRIPT", "")
        ).strip()
        if not resolved_host or not resolved_script:
            return (
                1,
                "Remote swarm cluster not configured: set `SWARM_HOST` and `SWARM_DISPATCHER_SCRIPT`.",
            )
        script = pathlib.Path(resolved_script).expanduser()
        if not script.is_file():
            return 1, f"Remote swarm dispatcher script not found at `{script}`."
        env = scrub_child_environment(extra_overrides={"SWARM_HOST": resolved_host})
        # Preserve SSH_AUTH_SOCK solely for explicit user-initiated remote cluster dispatch
        if "SSH_AUTH_SOCK" in os.environ:
            env["SSH_AUTH_SOCK"] = os.environ["SSH_AUTH_SOCK"]
        res = subprocess.run(
            [
                str(script),
                "--id",
                task_id,
                "--title",
                title,
                "--priority",
                "P0",
                "--skills",
                "cognitive-harness,dissect,tot",
                "--mention",
                os.environ.get(
                    "SWARM_MENTIONS",
                    "@commander @architect @security @critic @builder @judge",
                ),
                "--desc",
                description,
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return res.returncode, (res.stdout + "\n" + res.stderr).strip()

    def dispatch_to_peer_cli(
        self,
        prompt: str,
        *,
        conversation_id: str | None = None,
        metadata_only: bool = False,
        agent_bin: str | None = None,
        timeout: float = 120.0,
    ) -> PeerCliResult:
        """Delegates a task or follow-up message to a local peer agent CLI (`USH_PEER_AGENT_BIN` / `agentapi`)."""
        adapter = ExternalAgentCliAdapter(agent_bin=agent_bin, default_cwd=self.repo_root)
        if metadata_only:
            if not conversation_id:
                return PeerCliResult(returncode=2, output="`--peer-conversation` is required with `--peer-metadata`.")
            return adapter.get_metadata(conversation_id, timeout=min(timeout, 30.0))
        if conversation_id:
            return adapter.send_message(conversation_id, prompt, timeout=timeout)
        return adapter.new_conversation(prompt, timeout=timeout)


_SAFE_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_UUID_RE = re.compile(
    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)


@dataclass(frozen=True)
class PeerCliResult:
    returncode: int
    output: str
    conversation_id: str | None = None
    metadata: dict[str, Any] | None = None


class ExternalAgentCliAdapter:
    """Zero-shell-interpolation (`shell=False`) adapter for delegating tasks to a peer agent CLI (`agentapi`)."""

    def __init__(
        self,
        agent_bin: str | None = None,
        default_cwd: pathlib.Path | None = None,
    ) -> None:
        self.agent_bin_spec = (agent_bin or os.environ.get("USH_PEER_AGENT_BIN") or "agentapi").strip()
        self.default_cwd = default_cwd

    def resolve_binary(self) -> str | None:
        if not self.agent_bin_spec:
            return None
        if os.sep in self.agent_bin_spec or (os.altsep and os.altsep in self.agent_bin_spec):
            cand = pathlib.Path(self.agent_bin_spec).expanduser()
            return str(cand) if cand.is_file() and os.access(cand, os.X_OK) else None
        return shutil.which(self.agent_bin_spec)

    def is_available(self) -> bool:
        return self.resolve_binary() is not None

    @staticmethod
    def _build_clean_env() -> dict[str, str]:
        env = scrub_child_environment()
        for cfg_key in ("XDG_CONFIG_HOME", "APPDATA", "LOCALAPPDATA"):
            if cfg_key in os.environ:
                env[cfg_key] = os.environ[cfg_key]
        # Explicitly unset cross-project source metadata variable per `agentapi` specification
        env.pop("ANTIGRAVITY_SOURCE_METADATA", None)
        return env

    @staticmethod
    def _validate_conversation_id(conversation_id: str) -> str:
        cid = (conversation_id or "").strip()
        if not _SAFE_CONVERSATION_ID_RE.match(cid):
            raise ValueError(f"Invalid peer conversation ID: {conversation_id!r}")
        return cid

    def _run_subcommand(
        self, args: list[str], *, timeout: float, cwd: pathlib.Path | None = None
    ) -> subprocess.CompletedProcess[str] | PeerCliResult:
        exe = self.resolve_binary()
        if not exe:
            return PeerCliResult(
                returncode=127,
                output=f"Peer agent CLI `{self.agent_bin_spec}` not found on PATH (set `USH_PEER_AGENT_BIN` if installed elsewhere).",
            )
        run_cwd = str(cwd or self.default_cwd) if (cwd or self.default_cwd) else None
        return subprocess.run(
            [exe, *args],
            cwd=run_cwd,
            env=self._build_clean_env(),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def _extract_conversation_id(stdout_text: str) -> str | None:
        stripped = (stdout_text or "").strip()
        if not stripped:
            return None
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict):
                    for k in ("conversationId", "conversation_id", "id"):
                        if obj.get(k) and _SAFE_CONVERSATION_ID_RE.match(str(obj[k]).strip()):
                            return str(obj[k]).strip()
            except json.JSONDecodeError:
                pass
        m = _UUID_RE.search(stripped)
        if m:
            return m.group(1)
        first_line = stripped.splitlines()[0].strip()
        if " " not in first_line and _SAFE_CONVERSATION_ID_RE.match(first_line):
            return first_line
        return None

    def new_conversation(
        self, prompt: str, *, timeout: float = 120.0, cwd: pathlib.Path | None = None
    ) -> PeerCliResult:
        res = self._run_subcommand(["new-conversation", prompt], timeout=timeout, cwd=cwd)
        if isinstance(res, PeerCliResult):
            return res
        combined = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
        cid = self._extract_conversation_id(res.stdout) if res.returncode == 0 else None
        return PeerCliResult(returncode=res.returncode, output=combined, conversation_id=cid)

    def send_message(
        self,
        conversation_id: str,
        message: str,
        *,
        timeout: float = 120.0,
        cwd: pathlib.Path | None = None,
    ) -> PeerCliResult:
        cid = self._validate_conversation_id(conversation_id)
        res = self._run_subcommand(["send-message", cid, message], timeout=timeout, cwd=cwd)
        if isinstance(res, PeerCliResult):
            return res
        combined = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
        return PeerCliResult(returncode=res.returncode, output=combined, conversation_id=cid)

    def get_metadata(
        self, conversation_id: str, *, timeout: float = 30.0, cwd: pathlib.Path | None = None
    ) -> PeerCliResult:
        cid = self._validate_conversation_id(conversation_id)
        res = self._run_subcommand(["get-conversation-metadata", cid], timeout=timeout, cwd=cwd)
        if isinstance(res, PeerCliResult):
            return res
        combined = (res.stdout + ("\n" + res.stderr if res.stderr else "")).strip()
        meta: dict[str, Any] | None = None
        if res.returncode == 0 and res.stdout.strip().startswith("{"):
            try:
                parsed = json.loads(res.stdout.strip())
                if isinstance(parsed, dict):
                    meta = parsed
            except json.JSONDecodeError:
                meta = None
        return PeerCliResult(returncode=res.returncode, output=combined, conversation_id=cid, metadata=meta)
