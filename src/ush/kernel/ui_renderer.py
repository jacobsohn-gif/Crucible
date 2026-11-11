"""First-Class Headless Web/WebGL UI Renderer & Layout Diagnostics (`render_ui`).

Eliminates brittle shell-level headless browser flags, infinite `requestAnimationFrame` hangs,
`file://` CORS module blocks, and macOS SkiaGraphite/SwiftShader black WebGL canvas compositing.

Security invariants:
- Local files are strictly validated via `SecurityFirewall.resolve_and_validate(..., write=False)`.
- Loopback URLs are restricted strictly to `http://127.0.0.1:<port>` / `http://localhost:<port>`
  (blocking external hosts, RFC1918 LANs, and cloud metadata IPs `169.254.169.254`).
- Ephemeral HTTP server, sockets, subprocess pipes, and temp profile dirs are closed deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import http.server
import mimetypes
import os
import pathlib
import re
import shutil
import socketserver
import subprocess
import tempfile
import threading
import urllib.parse
from typing import Any, Callable

from ush.kernel.security import SecurityFirewall, redact_secrets

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"127.0.0.1", "localhost", "::1"})
_CHROME_CANDIDATES: tuple[str, ...] = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
)

_CONSOLE_ERR_RE = re.compile(r'INFO:CONSOLE[:(]\d+[)\]]+\s*"(.*)",\s*source:', re.IGNORECASE)


def discover_headless_browser() -> str | None:
    """Locates an installed Chromium-family browser binary for headless rendering."""
    override = (os.environ.get("CRUX_BROWSER_BIN") or os.environ.get("CHROME_BIN") or "").strip()
    if override:
        p = pathlib.Path(override).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        found = shutil.which(override)
        if found:
            return found
    for cand in _CHROME_CANDIDATES:
        if cand.startswith("/"):
            p = pathlib.Path(cand)
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def validate_render_target(
    target: str,
    firewall: SecurityFirewall,
) -> tuple[str, pathlib.Path | None, str]:
    """Validates `target` as `(kind='file', resolved_path, query_suffix)` or `(kind='url', None, safe_url)`.

    Raises `PermissionError` on any non-loopback URL or sandbox boundary violation.
    """
    raw = target.strip()
    if not raw:
        raise ValueError("Empty `target` passed to `render_ui`.")

    if raw.startswith(("http://", "https://")):
        parsed = urllib.parse.urlparse(raw)
        if parsed.scheme != "http":
            raise PermissionError(
                "Only local files or loopback `http://127.0.0.1:<port>` URLs are permitted in `render_ui`."
            )
        host = (parsed.hostname or "").lower()
        if host not in _LOOPBACK_HOSTS:
            raise PermissionError(
                f"SSRF blocked: `render_ui` only permits loopback hosts (`127.0.0.1` / `localhost`), got `{host}`."
            )
        if parsed.port is None or not (1 <= parsed.port <= 65535):
            raise PermissionError("Loopback URL in `render_ui` must specify an explicit port (1..65535).")
        return "url", None, raw

    if raw.startswith("file://"):
        parsed = urllib.parse.urlparse(raw)
        file_part = urllib.parse.unquote(parsed.path)
        query_sfx = f"?{parsed.query}" if parsed.query else ""
    elif "?" in raw:
        file_part, q = raw.split("?", 1)
        query_sfx = f"?{q}"
    else:
        file_part, query_sfx = raw, ""

    resolved = firewall.resolve_and_validate(file_part, write=False)
    if not resolved.is_file():
        raise FileNotFoundError(f"Render target file not found: `{resolved}`")
    return "file", resolved, query_sfx


def _build_probe_head_script(max_raf_frames: int, settle_ms: int) -> str:
    """Non-mutating in-memory JS shim preserving WebGL buffers, capping infinite rAF loops, & mirroring WebGL to 2D."""
    return (
        "<script>(function(){"
        "window.__cruxDiag={errors:[],webglCanvases:0};"
        "window.addEventListener('error',function(e){window.__cruxDiag.errors.push(String(e.message||e));});"
        "window.addEventListener('unhandledrejection',function(e){window.__cruxDiag.errors.push('UnhandledRejection: '+String(e.reason));});"
        "const _origErr=console.error.bind(console);"
        "console.error=function(){window.__cruxDiag.errors.push(Array.from(arguments).map(String).join(' '));_origErr.apply(console,arguments);};"
        "const _origGetCtx=HTMLCanvasElement.prototype.getContext;"
        "const _glCanvases=new Set();"
        "HTMLCanvasElement.prototype.getContext=function(t,a){"
        "if(t==='webgl'||t==='webgl2'||t==='experimental-webgl'){"
        "a=Object.assign({},a||{},{preserveDrawingBuffer:true});_glCanvases.add(this);"
        "}"
        "return _origGetCtx.call(this,t,a);};"
        "const _origRAF=window.requestAnimationFrame?window.requestAnimationFrame.bind(window):null;"
        f"let _rafCount=0;const _MAX_RAF={max(2, int(max_raf_frames))};"
        "if(_origRAF){window.requestAnimationFrame=function(cb){if(_rafCount>=_MAX_RAF)return 0;_rafCount++;return _origRAF(cb);};}"
        "function _snapshotWebGL(){try{"
        "_glCanvases.forEach(function(cv,idx){if(cv.width>0&&cv.height>0&&cv.parentElement){" 
        "const du=cv.toDataURL('image/png');if(du&&du.length>64){" 
        "const img=document.createElement('img');img.src=du;img.setAttribute('data-crux-gl-mirror',String(idx));" 
        "const cs=window.getComputedStyle(cv);" 
        "img.style.cssText='position:'+(cs.position==='fixed'?'fixed':'absolute')+';inset:0;width:'+cv.clientWidth+'px;height:'+cv.clientHeight+'px;z-index:'+(parseInt(cs.zIndex,10)||1)+';pointer-events:none;';" 
        "cv.parentElement.appendChild(img);}}"
        "});}catch(_e){}}"
        f"window.addEventListener('load',function(){{setTimeout(_snapshotWebGL,{max(60, int(settle_ms))});}});"
        "})();</script>"
    )


def inject_render_probe_html(html_text: str, *, max_raf_frames: int = 12, settle_ms: int = 260) -> str:
    """Injects the WebGL/rAF/console diagnostic shim into HTML text in memory without touching disk."""
    shim = _build_probe_head_script(max_raf_frames=max_raf_frames, settle_ms=settle_ms)
    m = re.search(r"<head[^>]*>", html_text, flags=re.IGNORECASE)
    if m:
        idx = m.end()
        return html_text[:idx] + shim + html_text[idx:]
    return shim + html_text


class _SilentThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _start_ephemeral_workspace_server(
    firewall: SecurityFirewall,
    *,
    max_raf_frames: int = 12,
    settle_ms: int = 260,
) -> tuple[_SilentThreadingHTTPServer, threading.Thread, int]:
    class _Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urllib.parse.urlparse(self.path)
            rel = urllib.parse.unquote(parsed.path).lstrip("/") or "index.html"
            try:
                fp = firewall.resolve_and_validate(rel, write=False)
            except Exception:
                self.send_error(403, "Forbidden")
                return
            if not fp.is_file():
                self.send_error(404, "Not Found")
                return
            try:
                raw = fp.read_bytes()
            except OSError:
                self.send_error(500, "Read Error")
                return
            ext = fp.suffix.lower()
            ctype = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
            if ext in (".html", ".htm"):
                txt = raw.decode("utf-8", errors="replace")
                injected = inject_render_probe_html(txt, max_raf_frames=max_raf_frames, settle_ms=settle_ms)
                raw = injected.encode("utf-8")
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(raw)
            except OSError:
                pass

    srv = _SilentThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    port = int(srv.server_address[1])
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    t.start()
    return srv, t, port


@dataclass
class RenderUiResult:
    success: bool
    screenshot_path: pathlib.Path | None = None
    png_bytes: bytes = b""
    width: int = 1440
    height: int = 900
    console_messages: list[str] = field(default_factory=list)
    summary: str = ""


def render_ui_capture(
    target: str,
    firewall: SecurityFirewall,
    *,
    output_path: str | None = None,
    width: int = 1440,
    height: int = 900,
    wait_ms: int = 350,
    max_raf_frames: int = 12,
    timeout_seconds: float = 22.0,
    allow_workspace_write: bool = True,
    browser_runner: Callable[[list[str], dict[str, str], float], tuple[int, str, str]] | None = None,
) -> RenderUiResult:
    """Renders a local HTML/SVG file or loopback URL to a PNG with WebGL preservation & console capture."""
    w = max(320, min(int(width), 3840))
    h = max(240, min(int(height), 2160))
    settle = max(60, min(int(wait_ms), 8000))
    kind, resolved_file, tail = validate_render_target(target, firewall)

    if output_path and allow_workspace_write:
        dest_png = firewall.resolve_and_validate(output_path, write=True)
        dest_png.parent.mkdir(parents=True, exist_ok=True)
    else:
        renders_dir = firewall.workspace_root / ".ush" / "renders"
        renders_dir.mkdir(parents=True, exist_ok=True)
        stem = resolved_file.stem if resolved_file is not None else "loopback"
        dest_png = renders_dir / f"{stem}_{w}x{h}.png"

    browser_bin = None if browser_runner is not None else discover_headless_browser()
    if browser_runner is None and not browser_bin:
        return RenderUiResult(
            success=False,
            width=w,
            height=h,
            summary="No headless Chromium/Chrome binary found on PATH or standard application paths (set `CRUX_BROWSER_BIN`).",
        )

    srv: _SilentThreadingHTTPServer | None = None
    srv_thread: threading.Thread | None = None
    tmp_profile = tempfile.mkdtemp(prefix="crux_ui_prof_")
    try:
        if kind == "file" and resolved_file is not None:
            srv, srv_thread, port = _start_ephemeral_workspace_server(
                firewall,
                max_raf_frames=max_raf_frames,
                settle_ms=settle,
            )
            rel_posix = resolved_file.relative_to(firewall.workspace_root).as_posix()
            quoted_rel = "/".join(urllib.parse.quote(seg) for seg in rel_posix.split("/"))
            render_url = f"http://127.0.0.1:{port}/{quoted_rel}{tail}"
        else:
            render_url = tail

        vtime_budget = settle + 900
        cmd = [
            browser_bin or "chromium",
            "--headless=new",
            f"--window-size={w},{h}",
            f"--user-data-dir={tmp_profile}",
            "--no-first-run",
            "--no-default-browser-check",
            "--disable-background-networking",
            "--disable-sync",
            "--enable-logging=stderr",
            "--v=0",
            "--enable-webgl",
            "--ignore-gpu-blocklist",
            "--enable-unsafe-swiftshader",
            "--use-gl=angle",
            "--use-angle=swiftshader",
            f"--virtual-time-budget={vtime_budget}",
            f"--screenshot={dest_png}",
            render_url,
        ]
        env = firewall.scrubbed_env()
        if browser_runner is not None:
            rc, out_s, err_s = browser_runner(cmd, env, timeout_seconds)
        else:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True,
            )
            try:
                out_s, err_s = proc.communicate(timeout=max(3.0, min(timeout_seconds, 60.0)))
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, 9)
                except OSError:
                    proc.kill()
                proc.communicate()
                return RenderUiResult(
                    success=False,
                    width=w,
                    height=h,
                    summary=f"`render_ui` timed out after {timeout_seconds:.1f}s while rendering `{target}`."
                )

        console_msgs: list[str] = []
        for m in _CONSOLE_ERR_RE.finditer(err_s or ""):
            msg = redact_secrets(m.group(1).strip())
            if "GPU stall due to ReadPixels" in msg or "Automatic fallback to software WebGL" in msg:
                continue
            if msg and msg not in console_msgs:
                console_msgs.append(msg[:240])

        if not dest_png.is_file() or dest_png.stat().st_size == 0:
            return RenderUiResult(
                success=False,
                width=w,
                height=h,
                console_messages=console_msgs,
                summary=f"Browser exited (rc={rc}) without producing screenshot `{dest_png.name}`."
            )

        png_bytes = dest_png.read_bytes()
        try:
            disp_path = dest_png.relative_to(firewall.workspace_root).as_posix()
        except ValueError:
            disp_path = str(dest_png)

        diag_str = (
            f"Console messages ({len(console_msgs)}): " + " | ".join(console_msgs[:5])
            if console_msgs
            else "Console errors: 0"
        )
        summary = (
            f"Rendered `{target}` -> `{disp_path}` ({w}x{h}, {len(png_bytes):,} bytes; WebGL buffer preserved).\n"
            f"{diag_str}"
        )
        return RenderUiResult(
            success=True,
            screenshot_path=dest_png,
            png_bytes=png_bytes,
            width=w,
            height=h,
            console_messages=console_msgs,
            summary=summary,
        )
    finally:
        if srv is not None:
            srv.shutdown()
            srv.server_close()
        if srv_thread is not None:
            srv_thread.join(timeout=1.5)
        shutil.rmtree(tmp_profile, ignore_errors=True)
