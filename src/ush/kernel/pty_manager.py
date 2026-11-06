"""Persistent Stateful PTY Multiplexer (`OSC 633` + Prompt-Strength Heuristics + `1 MiB` HeadTailBuffer).

Implements CT-07 and CT-08:
1. Persistent primary shell PTY preserving `cd`, `export`, and activated virtualenvs across calls.
2. Stateful `OSC 633` completion/exit-code marker parser (`\x1b]633;D;<token>;<rc>\x07`) safe across chunk splits.
3. Markerless prompt-strength heuristics (`strong` prompts like `[y/N]`, `Password:`, `>>>` yield in `0.45s`;
   `weak` prompts yield after quiet window) so interactive commands NEVER hang the harness!
4. Hard `1 MiB` middle-out `HeadTailBuffer` (`512 KiB` head + `512 KiB` tail) protecting RAM against `1 GB` stdout floods,
   plus automatic raw spillover file save when output exceeds `preview_bytes`.
5. `is_daemon=True` background tasks + `write_stdin` interaction + clean `os.killpg(..., SIGKILL)` cancellation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import errno
import fcntl
import os
import pathlib
import pty
import re
import select
import signal
import subprocess
import termios
import threading
import time
import uuid

from ush.kernel.security import redact_secrets, scrub_child_environment

ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")
OSC_GENERIC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

STRONG_PROMPT_RE = re.compile(
    r"(?:\[[yY]/[nN]\]|\[[nN]/[yY]\]|\(y/n\)|[Pp]assword:\s*|[Pp]assphrase.*:\s*|continue connecting \(yes/no.*\)\?\s*|>>>\s*$|\(Pdb\)\s*$)",
    re.IGNORECASE,
)
WEAK_PROMPT_RE = re.compile(r"(?:[$#❯%]\s*$|>\s*$)")


class HeadTailBuffer:
    """Hard-capped `1 MiB` middle-out ring preserving initial headers and final assertion tracebacks (CT-07)."""

    def __init__(self, max_bytes: int = 1_048_576) -> None:
        self.head_cap = max_bytes // 2
        self.tail_cap = max_bytes - self.head_cap
        self._head = bytearray()
        self._tail = bytearray()
        self.total_bytes = 0

    def append(self, chunk: bytes) -> None:
        if not chunk:
            return
        self.total_bytes += len(chunk)
        if len(self._head) < self.head_cap:
            space = self.head_cap - len(self._head)
            take = chunk[:space]
            self._head.extend(take)
            rem = chunk[space:]
            if rem:
                self._append_tail(rem)
        else:
            self._append_tail(chunk)

    def _append_tail(self, data: bytes) -> None:
        if len(data) >= self.tail_cap:
            self._tail = bytearray(data[-self.tail_cap :])
            return
        overflow = len(self._tail) + len(data) - self.tail_cap
        if overflow > 0:
            del self._tail[:overflow]
        self._tail.extend(data)

    def render(self) -> str:
        head_txt = self._head.decode("utf-8", errors="replace")
        if not self._tail:
            return head_txt
        tail_txt = self._tail.decode("utf-8", errors="replace")
        dropped = max(0, self.total_bytes - len(self._head) - len(self._tail))
        if dropped > 0:
            return f"{head_txt}\n... [MIDDLE_TRUNCATED: {dropped:,} bytes omitted; capped at 1 MiB] ...\n{tail_txt}"
        return head_txt + tail_txt


def _clean_terminal_text(raw: str) -> str:
    txt = OSC_GENERIC_RE.sub("", raw)
    txt = ANSI_CSI_RE.sub("", txt)
    # Resolve carriage-return progress-bar overwrites line by line
    resolved_lines: list[str] = []
    for line in txt.replace("\r\n", "\n").split("\n"):
        if "\r" in line:
            segs = [s for s in line.split("\r") if s != ""]
            resolved_lines.append(segs[-1] if segs else "")
        else:
            resolved_lines.append(line)
    return "\n".join(resolved_lines)


@dataclass
class PtyExecResult:
    pty_id: str
    status: str  # completed | interactive_prompt | running_background | timed_out | killed
    exit_code: int | None
    output: str
    total_bytes: int
    cwd: str
    spillover_path: str | None = None


@dataclass
class _PtyHandle:
    pty_id: str
    proc: subprocess.Popen[bytes]
    pid: int
    master_fd: int
    is_persistent_primary: bool
    is_daemon: bool = False
    cwd: str = ""
    buf: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    lock: threading.Lock = field(default_factory=threading.Lock)
    alive: bool = True
    exit_code: int | None = None


class PersistentPtyManager:
    """Manages one stateful persistent shell PTY plus tracked interactive/daemon child PTYs."""

    PRIMARY_ID = "pty_main"

    def __init__(
        self,
        workspace_root: pathlib.Path,
        spillover_dir: pathlib.Path | None = None,
        allowed_env_keys: tuple[str, ...] = (),
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.spillover_dir = (spillover_dir or (self.workspace_root / ".ush" / "sessions" / "tool_outputs")).resolve()
        self.allowed_env_keys = allowed_env_keys
        self._handles: dict[str, _PtyHandle] = {}
        self._guard = threading.Lock()
        self._normalized_path = self._probe_login_path()

    @staticmethod
    def _probe_login_path() -> str:
        cur = os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")
        extra_dirs = [
            pathlib.Path("~/.local/bin").expanduser(),
            pathlib.Path("~/.cargo/bin").expanduser(),
            pathlib.Path("/opt/homebrew/bin"),
            pathlib.Path("/usr/local/bin"),
        ]
        parts = cur.split(":")
        for d in extra_dirs:
            s = str(d)
            if d.is_dir() and s not in parts:
                parts.insert(0, s)
        return ":".join(parts)

    @staticmethod
    def _default_shell_argv(command: str | None = None) -> list[str]:
        shell = "/bin/bash" if os.path.exists("/bin/bash") else "/bin/sh"
        if shell.endswith("bash"):
            base = [shell, "--norc", "--noprofile"]
        else:
            base = [shell]
        if command is not None:
            base.extend(["-c", command])
        return base

    def _spawn_pty_process(self, argv: list[str], *, pty_id: str, persistent: bool, is_daemon: bool) -> _PtyHandle:
        master_fd, slave_fd = pty.openpty()
        try:
            # Disable TTY input echo so injected OSC 633 wrapper commands don't pollute stdout!
            attrs = termios.tcgetattr(slave_fd)
            attrs[3] &= ~termios.ECHO
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)

            env = scrub_child_environment(
                explicit_allowlist=self.allowed_env_keys,
                extra_overrides={
                    "PATH": self._normalized_path,
                    "PS1": "",
                    "PROMPT": "",
                    "TERM": "xterm-256color",
                },
            )
            cwd_dir = self.workspace_root if self.workspace_root.is_dir() else pathlib.Path.cwd()
            proc = subprocess.Popen(
                argv,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=str(cwd_dir),
                env=env,
                start_new_session=True,  # Dedicated process group for leak-free os.killpg
                close_fds=True,
            )
        except BaseException:
            self._close_fd(slave_fd)
            self._close_fd(master_fd)
            raise
        self._close_fd(slave_fd)
        fl = fcntl.fcntl(master_fd, fcntl.F_GETFL)
        fcntl.fcntl(master_fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
        handle = _PtyHandle(
            pty_id=pty_id,
            proc=proc,
            pid=proc.pid,
            master_fd=master_fd,
            is_persistent_primary=persistent,
            is_daemon=is_daemon,
            cwd=str(cwd_dir),
        )
        with self._guard:
            self._handles[pty_id] = handle
        return handle

    def _ensure_primary(self) -> _PtyHandle:
        with self._guard:
            existing = self._handles.get(self.PRIMARY_ID)
        if existing and existing.alive:
            if existing.proc.poll() is None:
                return existing
            existing.alive = False
            self._close_fd(existing.master_fd)
        return self._spawn_pty_process(
            self._default_shell_argv(),
            pty_id=self.PRIMARY_ID,
            persistent=True,
            is_daemon=False,
        )

    def blocking_background_count(self) -> int:
        """Used by the 5-condition quiescence barrier (`is_daemon=True` tasks NEVER block turn completion)."""
        count = 0
        with self._guard:
            handles = list(self._handles.values())
        for h in handles:
            if h.is_persistent_primary or h.is_daemon or not h.alive:
                continue
            rc = h.proc.poll()
            if rc is None:
                count += 1
            else:
                h.alive = False
                h.exit_code = rc
                self._close_fd(h.master_fd)
        return count

    def execute(
        self,
        command: str,
        *,
        wait_seconds: float = 10.0,
        is_daemon: bool = False,
        dedicated_pty: bool = False,
        preview_bytes: int = 16_384,
        cancel_event: threading.Event | None = None,
    ) -> PtyExecResult:
        if is_daemon or dedicated_pty:
            pid_tag = f"pty_{uuid.uuid4().hex[:8]}"
            try:
                handle = self._spawn_pty_process(
                    self._default_shell_argv(command),
                    pty_id=pid_tag,
                    persistent=False,
                    is_daemon=is_daemon,
                )
            except OSError as exc:
                err_buf = HeadTailBuffer()
                err_buf.append(f"[PTY spawn failed: {exc}]".encode("utf-8"))
                return self._finalize_output(
                    pid_tag, "crashed", -1, err_buf, str(self.workspace_root), preview_bytes
                )
            effective_wait = min(wait_seconds, 0.4) if is_daemon else wait_seconds
            return self._collect_dedicated(handle, effective_wait, preview_bytes, cancel_event)

        for pty_attempt in range(2):
            try:
                handle = self._ensure_primary()
            except OSError as exc:
                err_buf = HeadTailBuffer()
                err_buf.append(f"[Primary PTY spawn failed: {exc}]".encode("utf-8"))
                return self._finalize_output(
                    self.PRIMARY_ID, "crashed", -1, err_buf, str(self.workspace_root), preview_bytes
                )

            retry_on_fresh_pty = False
            with handle.lock:
                token = uuid.uuid4().hex[:12]
                # OSC 633 completion + CWD frame emitted regardless of command exit status
                wrapped = (
                    f"{command}\n"
                    f'__ush_rc=$?; printf "\\033]633;P;Cwd=%s\\007\\033]633;D;{token};%d\\007" "$PWD" "$__ush_rc"\n'
                )
                try:
                    # Drain any stale bytes before issuing the wrapped command
                    self._drain_nonblocking(handle.master_fd)
                    os.write(handle.master_fd, wrapped.encode("utf-8"))
                except (OSError, ValueError) as exc:
                    self._reset_primary(handle)
                    if pty_attempt == 0:
                        continue
                    err_buf = HeadTailBuffer()
                    err_buf.append(f"[Primary PTY write failed after respawn: {exc}]".encode("utf-8"))
                    return self._finalize_output(
                        self.PRIMARY_ID, "crashed", -1, err_buf, handle.cwd, preview_bytes
                    )

                cmd_buf = HeadTailBuffer()
                tail_window = ""
                osc_re = re.compile(
                    rf"\x1b\]633;P;Cwd=([^\x07]*)\x07\x1b\]633;D;{token};(-?\d+)\x07"
                )
                deadline = time.monotonic() + wait_seconds
                last_io = time.monotonic()

                while True:
                    if cancel_event and cancel_event.is_set():
                        self._reset_primary(handle)
                        return self._finalize_output(
                            self.PRIMARY_ID, "killed", -9, cmd_buf, handle.cwd, preview_bytes
                        )

                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        # Reset stuck primary shell cleanly or transfer to interactive if prompt detected
                        cleaned_tail = _clean_terminal_text(tail_window[-512:])
                        last_line = cleaned_tail.splitlines()[-1] if cleaned_tail.splitlines() else ""
                        if STRONG_PROMPT_RE.search(last_line):
                            return self._finalize_output(
                                self.PRIMARY_ID, "interactive_prompt", None, cmd_buf, handle.cwd, preview_bytes
                            )
                        self._reset_primary(handle)
                        return self._finalize_output(
                            self.PRIMARY_ID, "timed_out", None, cmd_buf, handle.cwd, preview_bytes
                        )

                    poll_timeout = min(0.10, rem)
                    dead_fd = False
                    try:
                        rlist, _, _ = select.select([handle.master_fd], [], [], poll_timeout)
                    except (OSError, ValueError):
                        rlist = []
                        dead_fd = True

                    if rlist:
                        chunk = self._read_chunk(handle.master_fd)
                        if chunk:
                            last_io = time.monotonic()
                            decoded = chunk.decode("utf-8", errors="replace")
                            tail_window = (tail_window + decoded)[-4096:]
                            m = osc_re.search(tail_window)
                            if m:
                                handle.cwd = m.group(1) or handle.cwd
                                rc = int(m.group(2))
                                # Strip OSC 633 completion suffix from captured chunk
                                clean_chunk = osc_re.sub("", decoded).encode("utf-8")
                                cmd_buf.append(clean_chunk)
                                return self._finalize_output(
                                    self.PRIMARY_ID, "completed", rc, cmd_buf, handle.cwd, preview_bytes
                                )
                            cmd_buf.append(chunk)
                            continue
                        dead_fd = True

                    if dead_fd or handle.proc.poll() is not None:
                        # Primary shell exited or crashed (e.g., `exit`, `exec false`, or external SIGKILL)
                        for extra in self._drain_chunks(handle.master_fd):
                            dec_extra = extra.decode("utf-8", errors="replace")
                            tail_window = (tail_window + dec_extra)[-4096:]
                            m_extra = osc_re.search(tail_window)
                            if m_extra:
                                handle.cwd = m_extra.group(1) or handle.cwd
                                rc_ok = int(m_extra.group(2))
                                cmd_buf.append(osc_re.sub("", dec_extra).encode("utf-8"))
                                self._reset_primary(handle)
                                try:
                                    self._ensure_primary()
                                except OSError:
                                    pass
                                return self._finalize_output(
                                    self.PRIMARY_ID, "completed", rc_ok, cmd_buf, handle.cwd, preview_bytes
                                )
                            cmd_buf.append(extra)
                        try:
                            handle.proc.wait(timeout=0.5)
                        except (subprocess.TimeoutExpired, OSError):
                            pass
                        rc_exit = handle.proc.returncode if handle.proc.returncode is not None else -1
                        saved_cwd = handle.cwd
                        self._reset_primary(handle)
                        # If the shell had died externally from a signal before producing any output, replay once!
                        if pty_attempt == 0 and cmd_buf.total_bytes == 0 and rc_exit < 0:
                            retry_on_fresh_pty = True
                            break
                        try:
                            self._ensure_primary()
                        except OSError:
                            pass
                        if rc_exit != 0:
                            cmd_buf.append(
                                f"\n[Primary shell process exited (rc={rc_exit}) — fresh primary PTY auto-respawned]\n".encode(
                                    "utf-8"
                                )
                            )
                        return self._finalize_output(
                            self.PRIMARY_ID, "completed", rc_exit, cmd_buf, saved_cwd, preview_bytes
                        )

                    # Markerless strong-prompt stall detection (`[y/N]`, `Password:`, `>>>` -> yield in 0.45s!)
                    idle = time.monotonic() - last_io
                    if idle >= 0.45 and tail_window:
                        cleaned_tail = _clean_terminal_text(tail_window[-512:])
                        lines = [ln for ln in cleaned_tail.split("\n") if ln.strip()]
                        last_line = lines[-1] if lines else ""
                        if STRONG_PROMPT_RE.search(last_line):
                            return self._finalize_output(
                                self.PRIMARY_ID, "interactive_prompt", None, cmd_buf, handle.cwd, preview_bytes
                            )
            if retry_on_fresh_pty:
                continue

        return self._finalize_output(
            self.PRIMARY_ID, "crashed", -1, HeadTailBuffer(), str(self.workspace_root), preview_bytes
        )

    def _collect_dedicated(
        self,
        handle: _PtyHandle,
        wait_seconds: float,
        preview_bytes: int,
        cancel_event: threading.Event | None,
    ) -> PtyExecResult:
        deadline = time.monotonic() + wait_seconds
        last_io = time.monotonic()
        tail_window = ""
        while True:
            if cancel_event and cancel_event.is_set():
                self.kill(handle.pty_id)
                return self._finalize_output(handle.pty_id, "killed", -9, handle.buf, handle.cwd, preview_bytes)

            rc = handle.proc.poll()
            if rc is not None:
                for extra in self._drain_chunks(handle.master_fd):
                    handle.buf.append(extra)
                handle.alive = False
                handle.exit_code = rc
                self._close_fd(handle.master_fd)
                return self._finalize_output(
                    handle.pty_id, "completed", handle.exit_code, handle.buf, handle.cwd, preview_bytes
                )

            rem = deadline - time.monotonic()
            if rem <= 0:
                state = "running_background"
                return self._finalize_output(handle.pty_id, state, None, handle.buf, handle.cwd, preview_bytes)

            try:
                rlist, _, _ = select.select([handle.master_fd], [], [], min(0.08, rem))
            except (OSError, ValueError):
                time.sleep(0.02)
                continue
            if rlist:
                chunk = self._read_chunk(handle.master_fd)
                if chunk:
                    last_io = time.monotonic()
                    handle.buf.append(chunk)
                    tail_window = (tail_window + chunk.decode("utf-8", errors="replace"))[-2048:]
            elif not handle.is_daemon and tail_window:
                idle = time.monotonic() - last_io
                cleaned = _clean_terminal_text(tail_window[-512:])
                lines = [ln for ln in cleaned.split("\n") if ln.strip()]
                last_ln = lines[-1] if lines else ""
                if idle >= 0.45 and STRONG_PROMPT_RE.search(last_ln):
                    return self._finalize_output(
                        handle.pty_id, "interactive_prompt", None, handle.buf, handle.cwd, preview_bytes
                    )
                if idle >= 5.0 and WEAK_PROMPT_RE.search(last_ln):
                    return self._finalize_output(
                        handle.pty_id, "interactive_prompt", None, handle.buf, handle.cwd, preview_bytes
                    )

    def write_stdin(self, pty_id: str, chars: str = "", wait_seconds: float = 2.0, preview_bytes: int = 16_384) -> PtyExecResult:
        with self._guard:
            handle = self._handles.get(pty_id)
        if not handle:
            nf_buf = HeadTailBuffer()
            nf_buf.append(f"Unknown or closed PTY session `{pty_id}`.".encode("utf-8"))
            return self._finalize_output(
                pty_id, "not_found", -1, nf_buf, str(self.workspace_root), preview_bytes
            )
        if chars and handle.alive and handle.proc.poll() is None:
            try:
                os.write(handle.master_fd, chars.encode("utf-8"))
            except (OSError, ValueError):
                handle.alive = False
        if handle.is_persistent_primary:
            # Read response after interactive prompt input (e.g. answering `y\n` to `[y/N]`)
            time.sleep(min(0.25, wait_seconds))
            tmp_buf = HeadTailBuffer()
            deadline = time.monotonic() + wait_seconds
            while time.monotonic() < deadline:
                try:
                    r, _, _ = select.select([handle.master_fd], [], [], 0.08)
                except (OSError, ValueError):
                    break
                if not r:
                    break
                ch = self._read_chunk(handle.master_fd)
                if not ch:
                    break
                tmp_buf.append(ch)
            return self._finalize_output(pty_id, "completed", 0, tmp_buf, handle.cwd, preview_bytes)
        return self._collect_dedicated(handle, wait_seconds, preview_bytes, None)

    def kill(self, pty_id: str) -> bool:
        with self._guard:
            handle = self._handles.get(pty_id)
        if not handle:
            return False
        if handle.alive:
            try:
                os.killpg(os.getpgid(handle.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                try:
                    os.kill(handle.pid, signal.SIGKILL)
                except OSError:
                    pass
            try:
                handle.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                handle.proc.kill()
                try:
                    handle.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            handle.alive = False
            handle.exit_code = -9
        else:
            handle.proc.poll()
        self._close_fd(handle.master_fd)
        return True

    def close_all(self) -> None:
        with self._guard:
            ids = list(self._handles.keys())
        for pid in ids:
            self.kill(pid)

    def _reset_primary(self, handle: _PtyHandle) -> None:
        self.kill(handle.pty_id)
        with self._guard:
            self._handles.pop(handle.pty_id, None)

    def _finalize_output(
        self,
        pty_id: str,
        status: str,
        exit_code: int | None,
        buf: HeadTailBuffer,
        cwd: str,
        preview_bytes: int,
    ) -> PtyExecResult:
        rendered = redact_secrets(_clean_terminal_text(buf.render())).strip("\n")
        spill_path: str | None = None
        raw_utf8 = rendered.encode("utf-8")
        if len(raw_utf8) > preview_bytes:
            self.spillover_dir.mkdir(parents=True, exist_ok=True)
            sp = self.spillover_dir / f"{pty_id}_{int(time.time() * 1000)}.log"
            sp.write_text(rendered, encoding="utf-8")
            spill_path = str(sp)
            half = max(256, preview_bytes // 2)
            head = raw_utf8[:half].decode("utf-8", errors="ignore")
            tail = raw_utf8[-half:].decode("utf-8", errors="ignore")
            rendered = (
                f"{head}\n"
                f"... [OUTPUT SPILLOVER: {buf.total_bytes:,} total bytes; full log saved at {spill_path}] ...\n"
                f"{tail}"
            )
        return PtyExecResult(
            pty_id=pty_id,
            status=status,
            exit_code=exit_code,
            output=rendered,
            total_bytes=buf.total_bytes,
            cwd=cwd,
            spillover_path=spill_path,
        )

    @staticmethod
    def _read_chunk(fd: int) -> bytes:
        try:
            return os.read(fd, 65_536)
        except OSError as exc:
            if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                return b""
            return b""

    @classmethod
    def _drain_chunks(cls, fd: int) -> list[bytes]:
        out: list[bytes] = []
        while True:
            ch = cls._read_chunk(fd)
            if not ch:
                break
            out.append(ch)
        return out

    @classmethod
    def _drain_nonblocking(cls, fd: int) -> None:
        cls._drain_chunks(fd)

    @staticmethod
    def _close_fd(fd: int) -> None:
        try:
            os.close(fd)
        except OSError:
            pass
