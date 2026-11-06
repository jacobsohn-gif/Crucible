"""Append-only DAG Session Tree (`LCA` Branch Rewind), Popperian Loop Guard & `0.85` Epistemic Compactor."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pathlib
import threading
from typing import Callable

from ush.core.types import CanonicalMessage, CanonicalPart, sha256_hex


# ============================================================================
# Popperian Falsification Loop Guard (CT-10)
# ============================================================================


@dataclass
class PopperianLoopGuard:
    """Detects identical failing tool/edit cycles and injects escalating falsification nudges."""

    recent_signatures: list[str] = field(default_factory=list)
    thrash_streak: int = 0

    def record(self, tool_name: str, args_summary: str, is_error: bool, result_head: str = "") -> str | None:
        if tool_name in ("read_file", "glob_files", "grep_files", "lsp_query") and not is_error:
            return None
        sig = sha256_hex(f"{tool_name}|{args_summary}|{is_error}|{result_head[:200]}")[:16]
        if self.recent_signatures and self.recent_signatures[-1] == sig:
            self.thrash_streak += 1
        elif sig in self.recent_signatures[-4:] and is_error:
            self.thrash_streak += 1
        else:
            self.thrash_streak = 1 if is_error else 0
        self.recent_signatures.append(sig)
        if len(self.recent_signatures) > 32:
            self.recent_signatures = self.recent_signatures[-32:]

        count = self.thrash_streak
        if count == 3:
            return (
                "[POPPERIAN_LOOP_GUARD streak=3]: You repeated the same failing edit/command 3 times. "
                "STOP patching blindly. State your active hypothesis explicitly and run ONE diagnostic read/command "
                "designed to FALSIFY it before editing again."
            )
        if count == 5:
            return (
                "[POPPERIAN_LOOP_GUARD streak=5]: Hypothesis falsified across 5 attempts. "
                "Revert assumptions, widen search with `grep_files`/`lsp_query`, or consider `/restore` / Gear-3 spike."
            )
        if count >= 8:
            return (
                f"[POPPERIAN_LOOP_GUARD streak={count} CRITICAL]: Hard cycle stall detected. "
                "Summarize exact blocker and falsified approaches instead of repeating the failing call."
            )
        return None


# ============================================================================
# DAG Conversation Tree + LCA Branch Summarizer + 0.85 Compactor (CT-10, CT-11)
# ============================================================================


class SessionDAG:
    """Tree-structured JSONL conversation log supporting `/tree`, `/fork`, LCA rewind, & atomic compaction."""

    def __init__(self, journal_path: pathlib.Path | None = None) -> None:
        self.journal_path = journal_path
        self._nodes: dict[str, CanonicalMessage] = {}
        self._order: list[str] = []
        self.leaf_id: str | None = None
        self.history_version: int = 0
        self.compaction_count: int = 0
        self._compacted_override: tuple[str | None, list[CanonicalMessage]] | None = None
        self._lock = threading.Lock()
        if journal_path and journal_path.is_file():
            self._load_journal(journal_path)

    def _load_journal(self, path: pathlib.Path) -> None:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta") == "head":
                self.leaf_id = rec.get("leaf_id")
                continue
            msg = CanonicalMessage.from_dict(rec)
            self._nodes[msg.entry_id] = msg
            self._order.append(msg.entry_id)
            self.leaf_id = msg.entry_id

    def _persist_record(self, payload: dict) -> None:
        if not self.journal_path:
            return
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def append(self, msg: CanonicalMessage) -> CanonicalMessage:
        with self._lock:
            msg.parent_entry_id = self.leaf_id
            self._nodes[msg.entry_id] = msg
            self._order.append(msg.entry_id)
            self.leaf_id = msg.entry_id
            self.history_version += 1
            self._persist_record(msg.to_dict())
            return msg

    def lineage(self, target_leaf_id: str | None = None) -> list[CanonicalMessage]:
        cur_id = target_leaf_id if target_leaf_id is not None else self.leaf_id
        chain: list[CanonicalMessage] = []
        seen: set[str] = set()
        while cur_id and cur_id in self._nodes and cur_id not in seen:
            seen.add(cur_id)
            node = self._nodes[cur_id]
            chain.append(node)
            cur_id = node.parent_entry_id
        chain.reverse()
        return chain

    def active_messages(self, apply_hygiene: bool = True) -> list[CanonicalMessage]:
        """Returns the active ancestor path (respecting atomic compaction anchor + stale-read tombstoning)."""
        raw = self.lineage()
        if self._compacted_override is not None:
            anchor_id, summary_msgs = self._compacted_override
            for idx, m in enumerate(raw):
                if m.entry_id == anchor_id:
                    raw = list(summary_msgs) + raw[idx + 1 :]
                    break
        return tombstone_superseded_reads(raw) if apply_hygiene else raw

    def approx_context_tokens(self) -> int:
        return sum(m.approx_tokens() for m in self.active_messages(apply_hygiene=True))

    def list_nodes(self) -> list[CanonicalMessage]:
        return [self._nodes[nid] for nid in self._order if nid in self._nodes]

    def find_lca(self, node_a: str | None, node_b: str | None) -> str | None:
        ancestors_a = {m.entry_id for m in self.lineage(node_a)}
        cur = node_b
        while cur and cur in self._nodes:
            if cur in ancestors_a:
                return cur
            cur = self._nodes[cur].parent_entry_id
        return None

    def rewind_to(
        self,
        target_entry_id: str,
        *,
        inject_branch_lessons: bool = True,
        failure_note: str = "",
    ) -> CanonicalMessage | None:
        """Rewinds active conversation leaf to `target_entry_id`, attaching an LCA summary of dead-end edits (CT-11)."""
        if target_entry_id not in self._nodes:
            raise KeyError(f"DAG entry `{target_entry_id}` not found.")
        old_leaf = self.leaf_id
        lca_id = self.find_lca(old_leaf, target_entry_id)

        abandoned_chain: list[CanonicalMessage] = []
        if old_leaf and old_leaf != target_entry_id:
            for m in self.lineage(old_leaf):
                if m.entry_id == lca_id:
                    abandoned_chain = []
                    continue
                abandoned_chain.append(m)

        with self._lock:
            self.leaf_id = target_entry_id
            self._compacted_override = None
            self.history_version += 1
            self._persist_record({"_meta": "head", "leaf_id": target_entry_id})

        if not inject_branch_lessons or not abandoned_chain:
            return None

        touched: list[str] = []
        errors: list[str] = []
        for m in abandoned_chain:
            for fp in m.modified_files:
                if fp not in touched:
                    touched.append(fp)
            for p in m.parts:
                if p.kind == "tool_call" and p.tool_name in ("edit_file", "write_file"):
                    fp = str((p.tool_args or {}).get("path", ""))
                    if fp and fp not in touched:
                        touched.append(fp)
                if p.kind == "tool_result" and p.is_error and p.text:
                    errors.append(p.text.splitlines()[0][:160])

        lesson_lines = [
            "<branch_lessons>",
            f"Rewound {len(abandoned_chain)} dead-end step(s) back to `{target_entry_id}` (LCA `{lca_id}`).",
            f"Modified files on abandoned branch: {', '.join(touched) if touched else 'none'}.",
        ]
        if failure_note:
            lesson_lines.append(f"Falsified approach / reason: {failure_note}")
        if errors:
            lesson_lines.append(f"Observed errors (DO NOT REPEAT): {' | '.join(errors[-3:])}")
        lesson_lines.append("</branch_lessons>")
        lesson_msg = CanonicalMessage(
            role="user",
            parts=[CanonicalPart(kind="text", text="\n".join(lesson_lines))],
        )
        return self.append(lesson_msg)

    def rollback_uncommitted(self, target_leaf_id: str | None) -> None:
        """Atomically rewinds `leaf_id` to `target_leaf_id` when Step 1 fails/cancels before any assistant output commits."""
        if target_leaf_id is not None and target_leaf_id not in self._nodes:
            return
        with self._lock:
            if self.leaf_id == target_leaf_id:
                return
            self.leaf_id = target_leaf_id
            if self._compacted_override is not None and self._compacted_override[0] != target_leaf_id:
                self._compacted_override = None
            self.history_version += 1
            self._persist_record({"_meta": "head", "leaf_id": target_leaf_id})

    # ---- Speculative / Atomic Version-Checked 0.85 Compactor (CT-10) ----

    def snapshot_for_compaction(self) -> tuple[int, str | None, list[CanonicalMessage]]:
        with self._lock:
            msgs = self.active_messages(apply_hygiene=True)
            return self.history_version, self.leaf_id, msgs

    def commit_compaction(
        self,
        expected_version: int,
        anchor_entry_id: str,
        summary_text: str,
        keep_tail: list[CanonicalMessage] | None = None,
    ) -> bool:
        """Commits compaction atomically ONLY if `history_version` has not drifted cleanly or tail-merges safely."""
        with self._lock:
            if self.history_version != expected_version:
                return False
            pointer = (
                f"\n[Full raw pre-compaction transcript preserved at: {self.journal_path}]"
                if self.journal_path
                else ""
            )
            summary_msg = CanonicalMessage(
                role="user",
                parts=[
                    CanonicalPart(
                        kind="text",
                        text=f"<epistemic_compaction_handoff>\n{summary_text.strip()}{pointer}\n</epistemic_compaction_handoff>",
                    )
                ],
            )
            tail = list(keep_tail or [])
            # Never start `keep_tail` on an orphaned `role == "tool"` turn
            while tail and tail[0].role == "tool":
                tail.pop(0)
            # Preserve strict `user -> assistant -> user` alternation when `tail` opens on a `user` turn
            if tail and tail[0].role != "assistant":
                ack_msg = CanonicalMessage(
                    role="assistant",
                    parts=[
                        CanonicalPart(
                            kind="text",
                            text="[Context checkpoint loaded. Continuing from preserved epistemic state.]",
                        )
                    ],
                )
                merged = [summary_msg, ack_msg] + tail
            else:
                merged = [summary_msg] + tail
            self._compacted_override = (anchor_entry_id, merged)
            self.compaction_count += 1
            self.history_version += 1
            return True

    def maybe_compact(
        self,
        context_limit_tokens: int,
        ratio: float = 0.85,
        summarizer: Callable[[list[CanonicalMessage]], str] | None = None,
    ) -> bool:
        threshold = max(256, int(context_limit_tokens * ratio))
        ver, leaf, msgs = self.snapshot_for_compaction()
        if not leaf or len(msgs) < 6:
            return False
        total_tok = sum(m.approx_tokens() for m in msgs)
        if total_tok < threshold:
            return False

        # Prefer snapping `split_idx` to a `role == "user"` turn boundary near `len(msgs) - 4`;
        # on single-turn 10+ step loops without a second `user` message, snap to a `role == "assistant"`
        # boundary (`isSplitTurn`) and NEVER cut on `role == "tool"`.
        target_idx = max(2, len(msgs) - 4)
        user_candidates = [k for k in range(1, len(msgs)) if msgs[k].role == "user"]
        if user_candidates:
            split_idx = min(user_candidates, key=lambda k: abs(k - target_idx))
        else:
            split_idx = target_idx
            while split_idx < len(msgs) and msgs[split_idx].role == "tool":
                split_idx += 1
            if split_idx >= len(msgs):
                split_idx = len(msgs) - 1
                while split_idx > 1 and msgs[split_idx].role == "tool":
                    split_idx -= 1
            if split_idx <= 0 or split_idx >= len(msgs) or msgs[split_idx].role == "tool":
                return False

        head = msgs[:split_idx]
        tail = msgs[split_idx:]
        anchor_id = msgs[-1].entry_id
        summary = summarizer(head) if summarizer else build_deterministic_epistemic_summary(head)
        return self.commit_compaction(ver, anchor_id, summary, keep_tail=tail)


def build_deterministic_epistemic_summary(messages: list[CanonicalMessage]) -> str:
    """Structured epistemic fallback/fast summary preserving paths, errors, and objectives verbatim."""
    objectives: list[str] = []
    files_seen: list[str] = []
    errors_seen: list[str] = []
    key_notes: list[str] = []

    for m in messages:
        if m.role == "user":
            txt = m.plain_text().strip()
            if txt and not txt.startswith("<epistemic_compaction"):
                objectives.append(txt.splitlines()[0][:220])
        for p in m.parts:
            if p.kind == "tool_call" and p.tool_args:
                fp = p.tool_args.get("path")
                if isinstance(fp, str) and fp not in files_seen:
                    files_seen.append(fp)
            elif p.kind == "tool_result" and p.is_error and p.text:
                errors_seen.append(p.text.splitlines()[0][:180])
        if m.role == "assistant":
            txt = m.plain_text().strip()
            if txt:
                key_notes.append(txt.splitlines()[0][:200])

    return "\n".join(
        [
            "1. OBJECTIVE & USER REQUESTS: " + (" -> ".join(objectives[:4]) or "Continue active task"),
            "2. VERIFIED TOUCHED/READ FILES: " + (", ".join(files_seen[-20:]) or "none yet"),
            "3. FALSIFIED PATHS / OBSERVED ERRORS (DO NOT REPEAT): " + (" | ".join(errors_seen[-5:]) or "none"),
            "4. RECENT PROGRESS NOTES: " + (" | ".join(key_notes[-4:]) or "in progress"),
        ]
    )


def tombstone_superseded_reads(messages: list[CanonicalMessage]) -> list[CanonicalMessage]:
    """Zero-LLM per-turn hygiene: replaces older duplicate full `read_file(path)` outputs with 1-line tombstones."""
    call_id_to_read_key: dict[str, str] = {}
    for m in messages:
        for p in m.parts:
            if p.kind == "tool_call" and p.tool_name == "read_file" and p.tool_call_id and p.tool_args:
                key = f"{p.tool_args.get('path')}:{p.tool_args.get('start_line')}:{p.tool_args.get('end_line')}"
                call_id_to_read_key[p.tool_call_id] = key

    latest_call_for_key: dict[str, tuple[str, int]] = {}
    for step_idx, m in enumerate(messages):
        for p in m.parts:
            if p.kind == "tool_result" and not p.is_error and p.tool_call_id in call_id_to_read_key:
                latest_call_for_key[call_id_to_read_key[p.tool_call_id]] = (p.tool_call_id, step_idx)

    cleaned: list[CanonicalMessage] = []
    for m in messages:
        changed = False
        new_parts: list[CanonicalPart] = []
        for p in m.parts:
            if p.kind == "tool_result" and not p.is_error and p.tool_call_id in call_id_to_read_key:
                rkey = call_id_to_read_key[p.tool_call_id]
                latest_id, latest_step = latest_call_for_key[rkey]
                if latest_id != p.tool_call_id and len(p.text) > 256:
                    changed = True
                    path_only = rkey.split(":", 1)[0]
                    new_parts.append(
                        CanonicalPart(
                            kind="tool_result",
                            tool_call_id=p.tool_call_id,
                            tool_name=p.tool_name,
                            text=f"[Superseded `read_file({path_only})` output replaced — latest contents at step #{latest_step}]",
                        )
                    )
                    continue
            new_parts.append(p)
        if changed:
            cp = CanonicalMessage(
                role=m.role,
                parts=new_parts,
                entry_id=m.entry_id,
                parent_entry_id=m.parent_entry_id,
                producer_provider=m.producer_provider,
                producer_model_id=m.producer_model_id,
                stop_reason=m.stop_reason,
                created_at=m.created_at,
                usage=m.usage,
                modified_files=list(m.modified_files),
            )
            cleaned.append(cp)
        else:
            cleaned.append(m)
    return cleaned
