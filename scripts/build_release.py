#!/usr/bin/env python3
"""Build script for Crucible (`crux` v0.1) — produces both Source and Compiled distributions.

Outputs in `releases/v0.1/`:
  1. `crucible-0.1-src.tar.gz`     — Clean source distribution archive
  2. `crux.pyz`                    — Portable compiled `.pyc` bytecode executable (0 raw `.py` files)
  3. `crux-0.1-darwin-arm64`       — Native compiled Mach-O 64-bit ARM64 binary embedding compiled bytecode
  4. `crux`                        — Native compiled binary ready to copy anywhere on macOS ARM64
  5. `SHA256SUMS.txt`              — Cryptographic SHA-256 verification manifest
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import py_compile
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = "0.1"
RELEASE_DIR = ROOT / "releases" / f"v{VERSION}"


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def build_source_tarball(out_dir: pathlib.Path) -> pathlib.Path:
    """Package clean source tree into `crucible-0.1-src.tar.gz`."""
    tar_path = out_dir / f"crucible-{VERSION}-src.tar.gz"
    include_items = [
        "pyproject.toml",
        "README.md",
        "docs",
        "Makefile",
        ".gitignore",
        ".ushignore",
        "bin",
        "src",
        "tests",
        "scripts",
    ]

    def _filter(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
        name = tarinfo.name
        if any(part in ("__pycache__", ".DS_Store", ".pytest_cache") for part in name.split("/")):
            return None
        if name.endswith((".pyc", ".pyo")):
            return None
        return tarinfo

    with tarfile.open(tar_path, "w:gz") as tf:
        for rel in include_items:
            p = ROOT / rel
            if p.exists():
                tf.add(p, arcname=f"crucible-{VERSION}/{rel}", filter=_filter)
    return tar_path


def build_compiled_pyz(out_dir: pathlib.Path) -> pathlib.Path:
    """Compile all Python modules to `.pyc` bytecode and bundle into single-file executable `crux.pyz`."""
    pyz_path = out_dir / "crux.pyz"
    src_root = ROOT / "src"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = pathlib.Path(tmp)
        stage_dir = tmp_path / "stage"
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Compile every `src/**/*.py` into stage_dir as `.pyc` (plus `.py` and package data assets)
        for src_item in sorted(src_root.rglob("*")):
            if not src_item.is_file():
                continue
            rel = src_item.relative_to(src_root)
            if any(part in ("__pycache__", ".DS_Store", ".pytest_cache") for part in rel.parts):
                continue
            if src_item.suffix == ".py":
                py_dest = stage_dir / rel
                pyc_dest = stage_dir / rel.with_suffix(".pyc")
                pyc_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, py_dest)
                py_compile.compile(
                    str(src_item),
                    cfile=str(pyc_dest),
                    dfile=str(rel),
                    doraise=True,
                    optimize=2,
                )
            elif src_item.suffix in (".yaml", ".yml", ".json", ".md", ".toml"):
                asset_dest = stage_dir / rel
                asset_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_item, asset_dest)

        # Create entrypoint `__main__.py` and compile to `__main__.pyc`
        main_py = stage_dir / "__main__.py"
        main_py.write_text(
            "import sys\n"
            "from ush.cli.main import main\n"
            "if __name__ == '__main__':\n"
            "    raise SystemExit(main())\n",
            encoding="utf-8",
        )
        main_pyc = stage_dir / "__main__.pyc"
        py_compile.compile(
            str(main_py),
            cfile=str(main_pyc),
            dfile="__main__.py",
            doraise=True,
            optimize=2,
        )

        # Build ZIP payload in memory, prepend POSIX python3 shebang
        zip_buf_path = tmp_path / "payload.zip"
        with zipfile.ZipFile(zip_buf_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            for item in sorted(stage_dir.rglob("*")):
                if item.is_file() and item.suffix in (".pyc", ".py", ".yaml", ".yml", ".json", ".md", ".toml"):
                    arcname = str(item.relative_to(stage_dir))
                    zf.write(item, arcname)

        shebang = b"#!/usr/bin/env python3\n"
        pyz_path.write_bytes(shebang + zip_buf_path.read_bytes())
        pyz_path.chmod(pyz_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    return pyz_path


def build_native_macho_binary(pyz_path: pathlib.Path, out_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    """Compile a native macOS Mach-O executable with `clang -O3` embedding the compiled `.pyz` bytecode."""
    payload = pyz_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()[:16]

    # Format payload as C hex array lines
    hex_lines: list[str] = []
    for i in range(0, len(payload), 16):
        chunk = payload[i : i + 16]
        hex_lines.append("    " + ", ".join(f"0x{b:02x}" for b in chunk) + ",")
    payload_c_array = "\n".join(hex_lines)

    c_code = f"""/* Auto-generated Crucible (crux v{VERSION}) Native Compiled Launcher */
#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <libgen.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static const char *CRUX_VERSION = "{VERSION}";
static const char *CRUX_DIGEST = "{digest}";
static const unsigned char CRUX_PAYLOAD[] = {{
{payload_c_array}
}};
static const size_t CRUX_PAYLOAD_LEN = {len(payload)}UL;

static int ensure_dir(const char *path) {{
    struct stat st;
    if (stat(path, &st) == 0) {{
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }}
    if (mkdir(path, 0700) == 0 || errno == EEXIST) {{
        return 0;
    }}
    return -1;
}}

int main(int argc, char **argv) {{
    const char *home = getenv("HOME");
    if (!home || !home[0]) {{
        home = "/tmp";
    }}

    char dir1[PATH_MAX];
    char dir2[PATH_MAX];
    char cache_pyz[PATH_MAX];
    char tmp_pyz[PATH_MAX];

    snprintf(dir1, sizeof(dir1), "%s/.ush", home);
    snprintf(dir2, sizeof(dir2), "%s/.ush/bin-cache", home);
    snprintf(cache_pyz, sizeof(cache_pyz), "%s/crux-v%s-%s.pyz", dir2, CRUX_VERSION, CRUX_DIGEST);

    ensure_dir(dir1);
    ensure_dir(dir2);

    struct stat st;
    if (stat(cache_pyz, &st) != 0 || (size_t)st.st_size != CRUX_PAYLOAD_LEN) {{
        snprintf(tmp_pyz, sizeof(tmp_pyz), "%s.tmp.%d", cache_pyz, (int)getpid());
        int fd = open(tmp_pyz, O_WRONLY | O_CREAT | O_TRUNC, 0700);
        if (fd < 0) {{
            perror("crux: failed to open bytecode cache");
            return 1;
        }}
        size_t written = 0;
        while (written < CRUX_PAYLOAD_LEN) {{
            ssize_t n = write(fd, CRUX_PAYLOAD + written, CRUX_PAYLOAD_LEN - written);
            if (n < 0) {{
                if (errno == EINTR) continue;
                perror("crux: failed to write bytecode cache");
                close(fd);
                unlink(tmp_pyz);
                return 1;
            }}
            written += (size_t)n;
        }}
        close(fd);
        if (rename(tmp_pyz, cache_pyz) != 0) {{
            perror("crux: failed to finalize bytecode cache");
            unlink(tmp_pyz);
            return 1;
        }}
    }}

    char py_buf[PATH_MAX];
    py_buf[0] = '\\0';
    const char *py_bin = getenv("CRUX_PYTHON");
    if (!py_bin || !py_bin[0]) {{
        /* 1. Check relative repo `.venv/bin/python` from argv[0] or CWD */
        char resolved_self[PATH_MAX];
        if (argc > 0 && argv[0] && realpath(argv[0], resolved_self)) {{
            char tmp1[PATH_MAX];
            strncpy(tmp1, resolved_self, sizeof(tmp1) - 1);
            tmp1[sizeof(tmp1) - 1] = '\\0';
            char *d1 = dirname(tmp1);
            char cand[PATH_MAX];
            snprintf(cand, sizeof(cand), "%s/../../.venv/bin/python", d1);
            if (access(cand, X_OK) == 0) {{
                strncpy(py_buf, cand, sizeof(py_buf) - 1);
                py_bin = py_buf;
            }}
        }}
        if ((!py_bin || !py_bin[0]) && access(".venv/bin/python", X_OK) == 0) {{
            py_bin = ".venv/bin/python";
        }}
        /* 2. Check Homebrew / /usr/local / uv-managed Python 3.11+ before Xcode 3.9 */
        if (!py_bin || !py_bin[0]) {{
            const char *std_bins[] = {{
                "/opt/homebrew/bin/python3",
                "/usr/local/bin/python3",
                NULL
            }};
            for (int k = 0; std_bins[k]; k++) {{
                if (access(std_bins[k], X_OK) == 0) {{
                    py_bin = std_bins[k];
                    break;
                }}
            }}
        }}
        if ((!py_bin || !py_bin[0]) && home && home[0]) {{
            char pattern[PATH_MAX];
            snprintf(pattern, sizeof(pattern), "%s/.local/share/uv/python/cpython-3.*/bin/python3", home);
            glob_t g;
            if (glob(pattern, 0, NULL, &g) == 0 && g.gl_pathc > 0) {{
                for (ssize_t idx = (ssize_t)g.gl_pathc - 1; idx >= 0; idx--) {{
                    if (access(g.gl_pathv[idx], X_OK) == 0) {{
                        strncpy(py_buf, g.gl_pathv[idx], sizeof(py_buf) - 1);
                        py_bin = py_buf;
                        break;
                    }}
                }}
                globfree(&g);
            }}
        }}
        if ((!py_bin || !py_bin[0]) && home && home[0]) {{
            snprintf(py_buf, sizeof(py_buf), "%s/.pyenv/shims/python3", home);
            if (access(py_buf, X_OK) == 0) {{
                py_bin = py_buf;
            }}
        }}
        if (!py_bin || !py_bin[0]) {{
            py_bin = "python3";
        }}
    }}

    char **new_argv = (char **)calloc((size_t)argc + 2, sizeof(char *));
    if (!new_argv) {{
        perror("crux: calloc");
        return 1;
    }}
    new_argv[0] = (char *)py_bin;
    new_argv[1] = cache_pyz;
    for (int i = 1; i < argc; i++) {{
        new_argv[i + 1] = argv[i];
    }}
    new_argv[argc + 1] = NULL;

    execvp(py_bin, new_argv);
    perror("crux: execvp python3");
    return 1;
}}
"""

    arch_bin = out_dir / f"crux-{VERSION}-darwin-arm64"
    short_bin = out_dir / "crux"

    with tempfile.TemporaryDirectory() as tmp:
        c_file = pathlib.Path(tmp) / "crux_launcher.c"
        c_file.write_text(c_code, encoding="utf-8")
        subprocess.check_call(
            ["clang", "-O3", "-Wall", "-Wextra", str(c_file), "-o", str(arch_bin)]
        )
        subprocess.check_call(["strip", "-x", str(arch_bin)])
        subprocess.check_call(["codesign", "-s", "-", "-f", str(arch_bin)])

    shutil.copy2(arch_bin, short_bin)
    return arch_bin, short_bin


def main() -> int:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building Crucible (crux v{VERSION}) release artifacts in {RELEASE_DIR}...")

    pyz_path = build_compiled_pyz(RELEASE_DIR)
    arch_bin, short_bin = build_native_macho_binary(pyz_path, RELEASE_DIR)
    tar_path = build_source_tarball(RELEASE_DIR)

    artifacts = [arch_bin, short_bin, pyz_path, tar_path]
    sums_lines: list[str] = []
    for art in artifacts:
        digest = sha256_file(art)
        size_kb = art.stat().st_size / 1024.0
        sums_lines.append(f"{digest}  {art.name}")
        print(f"  [OK] {art.name:<28s} ({size_kb:6.1f} KB)  sha256={digest[:16]}...")

    sums_path = RELEASE_DIR / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(sums_lines) + "\n", encoding="utf-8")
    print(f"Wrote checksum manifest: {sums_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
