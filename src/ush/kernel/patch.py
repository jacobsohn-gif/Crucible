"""4-Pass Fuzzy `edit_file` Engine, Stale-Read SHA-256 Guard & Deadlock-Free Sorted File Locks.

Implements CT-05 and CT-06:
- Pass 1: Exact character-sequence match.
- Pass 2: Line-end `rstrip` whitespace-insensitive match.
- Pass 3: Indentation-shifted trimmed block match (preserving relative indentation when applying replacement!).
- Pass 4: Unicode typographic quote / dash / non-breaking-space fold match.
- Strictly preserves UTF-8 BOM and dominant CRLF vs LF line endings across untouched and replaced lines.
- SHA-256 stale-read detector returns a clean unified diff instead of silently clobbering external edits.
- Lexicographically ordered canonical path lock table eliminates ABBA multi-file deadlocks.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import difflib
import os
import pathlib
import tempfile
import threading
from typing import Iterable, Iterator
import unicodedata

from ush.core.types import sha256_hex

UNICODE_FOLD_TABLE = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201A": "'",
        "\u201B": "'",
        "\u201C": '"',
        "\u201D": '"',
        "\u201E": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "\u00A0": " ",
        "\u2009": " ",
        "\u202F": " ",
        "\u3000": " ",
    }
)


def _fold_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s.translate(UNICODE_FOLD_TABLE)).rstrip()


class SortedFileLockTable:
    """Deadlock-free per-file mutex table acquiring canonical paths in strict lexicographic order (CT-06)."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._locks: dict[str, threading.Lock] = {}

    @contextmanager
    def acquire_sorted(self, paths: Iterable[pathlib.Path]) -> Iterator[None]:
        keys = sorted({str(p.resolve()) for p in paths})
        with self._guard:
            locks = [self._locks.setdefault(k, threading.Lock()) for k in keys]
        acquired: list[threading.Lock] = []
        try:
            for lk in locks:
                lk.acquire()
                acquired.append(lk)
            yield
        finally:
            for lk in reversed(acquired):
                lk.release()


GLOBAL_FILE_LOCKS = SortedFileLockTable()


class StaleReadTracker:
    """Tracks per-session file SHA-256 snapshots at last `read_file` / `edit_file` to catch concurrent edits."""

    def __init__(self) -> None:
        self._guard = threading.Lock()
        self._hashes: dict[str, tuple[str, str]] = {}  # abs_path -> (sha256, cached_text)

    def record_read(self, path: pathlib.Path, raw_bytes: bytes) -> None:
        key = str(path.resolve())
        text = raw_bytes.decode("utf-8", errors="replace")
        with self._guard:
            self._hashes[key] = (sha256_hex(raw_bytes), text)

    def check_fresh(self, path: pathlib.Path, current_bytes: bytes) -> str | None:
        """Returns None if unmodified since last read, or a unified diff error string if stale (CT-06)."""
        key = str(path.resolve())
        with self._guard:
            prev = self._hashes.get(key)
        if prev is None:
            return None
        prev_sha, prev_text = prev
        if sha256_hex(current_bytes) == prev_sha:
            return None
        cur_text = current_bytes.decode("utf-8", errors="replace")
        diff = "\n".join(
            difflib.unified_diff(
                prev_text.splitlines(),
                cur_text.splitlines(),
                fromfile=f"a/{path.name} (last read)",
                tofile=f"b/{path.name} (current on disk)",
                lineterm="",
                n=2,
            )
        )
        return (
            f"STALE_READ_CONFLICT: `{path.name}` was modified externally since you last read it. "
            f"Re-read or incorporate the external delta below before retrying `edit_file`:\n{diff[:3000]}"
        )


@dataclass
class PatchResult:
    success: bool
    pass_used: str  # exact | rstrip | trim_reindent | unicode_fold | error
    replacements: int
    diff_preview: str
    error: str = ""


def _leading_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip(" \t"))]


def _reindent_replacement(
    matched_doc_lines: list[str], target_lines: list[str], repl_lines: list[str]
) -> list[str]:
    """Shifts replacement block indentation by the exact column delta between file and model target."""
    doc_indents = [len(_leading_indent(ln)) for ln in matched_doc_lines if ln.strip()]
    tgt_indents = [len(_leading_indent(ln)) for ln in target_lines if ln.strip()]
    if not doc_indents or not tgt_indents:
        return repl_lines
    delta = min(doc_indents) - min(tgt_indents)
    if delta == 0:
        return repl_lines
    # Detect whether matched code prefers tabs or spaces
    sample_indent = next((_leading_indent(ln) for ln in matched_doc_lines if _leading_indent(ln)), " ")
    pad_char = "\t" if sample_indent.startswith("\t") else " "
    adjusted: list[str] = []
    for ln in repl_lines:
        if not ln.strip():
            adjusted.append("")
        elif delta > 0:
            adjusted.append((pad_char * delta) + ln)
        else:
            strip_n = min(-delta, len(_leading_indent(ln)))
            adjusted.append(ln[strip_n:])
    return adjusted


def apply_fuzzy_patch_to_bytes(raw_bytes: bytes, target: str, replacement: str, filename: str = "file") -> tuple[bytes | None, PatchResult]:
    """Applies a 4-pass fuzzy target->replacement patch while preserving UTF-8 BOM and CRLF/LF endings."""
    if not target:
        return None, PatchResult(False, "error", 0, "", "Target string must not be empty.")

    has_bom = raw_bytes.startswith(b"\xef\xbb\xbf")
    body_bytes = raw_bytes[3:] if has_bom else raw_bytes
    text = body_bytes.decode("utf-8")

    # Detect dominant line ending (`\r\n` vs `\n`) while normalizing working copy to `\n`
    crlf_count = text.count("\r\n")
    lf_only_count = text.count("\n") - crlf_count
    use_crlf = crlf_count > 0 and crlf_count >= lf_only_count

    norm_doc = text.replace("\r\n", "\n")
    norm_tgt = target.replace("\r\n", "\n")
    norm_rep = replacement.replace("\r\n", "\n")

    new_doc: str | None = None
    pass_name = "error"
    count = 0

    # --- Pass 1: Exact character match ---
    if norm_tgt in norm_doc:
        count = norm_doc.count(norm_tgt)
        new_doc = norm_doc.replace(norm_tgt, norm_rep)
        pass_name = "exact"
    else:
        # Line-oriented passes (Passes 2, 3, 4) preserve untouched lines verbatim!
        trailing_nl = norm_doc.endswith("\n")
        doc_lines = norm_doc.split("\n")
        if trailing_nl:
            doc_lines = doc_lines[:-1]
        tgt_lines = norm_tgt.split("\n")
        if norm_tgt.endswith("\n") and len(tgt_lines) > 1 and tgt_lines[-1] == "":
            tgt_lines = tgt_lines[:-1]
        rep_lines = norm_rep.split("\n") if norm_rep != "" else []
        if norm_rep.endswith("\n") and len(rep_lines) > 1 and rep_lines[-1] == "":
            rep_lines = rep_lines[:-1]

        m_len = len(tgt_lines)
        passes = (
            ("rstrip", lambda s: s.rstrip(), False),
            ("trim_reindent", lambda s: s.strip(), True),
            ("unicode_fold", lambda s: _fold_unicode(s).strip(), True),
        )
        for p_label, key_fn, do_reindent in passes:
            tgt_keys = [key_fn(x) for x in tgt_lines]
            doc_keys = [key_fn(x) for x in doc_lines]
            hits: list[int] = []
            idx = 0
            while idx <= len(doc_lines) - m_len:
                if doc_keys[idx : idx + m_len] == tgt_keys:
                    hits.append(idx)
                    idx += max(1, m_len)
                else:
                    idx += 1
            if len(hits) == 1:
                h = hits[0]
                matched_slice = doc_lines[h : h + m_len]
                final_rep = _reindent_replacement(matched_slice, tgt_lines, rep_lines) if do_reindent else rep_lines
                updated_lines = doc_lines[:h] + final_rep + doc_lines[h + m_len :]
                new_doc = "\n".join(updated_lines) + ("\n" if trailing_nl and updated_lines else "")
                pass_name = p_label
                count = 1
                break
            if len(hits) > 1:
                return None, PatchResult(
                    False,
                    "error",
                    0,
                    "",
                    f"Ambiguous fuzzy match ({p_label}): target matched {len(hits)} locations in `{filename}`. Include 2-3 surrounding context lines.",
                )

    if new_doc is None:
        return None, PatchResult(
            False,
            "error",
            0,
            "",
            f"Target snippet not found in `{filename}` across all 4 fuzzy passes (exact, rstrip, trim, unicode). Re-read surrounding lines with `read_file`.",
        )

    diff_preview = "\n".join(
        difflib.unified_diff(
            norm_doc.splitlines(),
            new_doc.splitlines(),
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}",
            lineterm="",
            n=3,
        )
    )
    encoded_text = new_doc.replace("\n", "\r\n") if use_crlf else new_doc
    out_bytes = (b"\xef\xbb\xbf" if has_bom else b"") + encoded_text.encode("utf-8")
    return out_bytes, PatchResult(True, pass_name, count, diff_preview)


def atomic_write_bytes(path: pathlib.Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = path.stat().st_mode if path.exists() else None
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".ush_edit_{path.name}_")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        if mode is not None:
            os.chmod(tmp_path, mode & 0o7777)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def edit_file_safe(
    path: pathlib.Path,
    target: str,
    replacement: str,
    stale_tracker: StaleReadTracker | None = None,
) -> PatchResult:
    resolved = path.resolve()
    with GLOBAL_FILE_LOCKS.acquire_sorted([resolved]):
        if not resolved.is_file():
            return PatchResult(False, "error", 0, "", f"File does not exist: `{resolved}`")
        raw = resolved.read_bytes()
        if stale_tracker is not None:
            conflict = stale_tracker.check_fresh(resolved, raw)
            if conflict:
                return PatchResult(False, "error", 0, "", conflict)
        new_bytes, res = apply_fuzzy_patch_to_bytes(raw, target, replacement, filename=resolved.name)
        if not res.success or new_bytes is None:
            return res
        atomic_write_bytes(resolved, new_bytes)
        if stale_tracker is not None:
            stale_tracker.record_read(resolved, new_bytes)
        return res
