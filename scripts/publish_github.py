#!/usr/bin/env python3
"""Create/sync GitHub repository `jacobsohn-gif/Crucible`, push `main` + `v0.1`, and publish v0.1 Release assets."""

from __future__ import annotations

import json
import pathlib
import subprocess
import urllib.error
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
RELEASE_DIR = ROOT / "releases" / "v0.1"
OWNER = "jacobsohn-gif"
REPO = "Crucible"
TAG = "v0.1"


def get_github_token() -> str:
    out = subprocess.check_output(
        ["git", "credential-osxkeychain", "get"],
        input=b"protocol=https\nhost=github.com\n\n",
    ).decode("utf-8", errors="replace")
    creds = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    token = creds.get("password", "").strip()
    if not token:
        raise RuntimeError("No GitHub token found in git credential-osxkeychain.")
    return token


def api_request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str = "application/json",
) -> dict:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "Content-Type": content_type,
        "User-Agent": "crux-cli/0.1",
    }
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=60) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
        return json.loads(raw) if raw else {}


def ensure_repo(token: str) -> str:
    payload = json.dumps(
        {
            "name": REPO,
            "description": (
                "Crucible (crux v0.1): Standalone zero-dependency interactive "
                "thought-tracing coding & 4-Gear Elastic Swarm CLI (Source + Compiled distributions)."
            ),
            "private": False,
        }
    ).encode("utf-8")
    try:
        res = api_request("https://api.github.com/user/repos", token, method="POST", data=payload)
        print(f"Created GitHub repository: {res.get('html_url')}")
        return res.get("html_url", f"https://github.com/{OWNER}/{REPO}")
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            print(f"GitHub repository https://github.com/{OWNER}/{REPO} already exists.")
            return f"https://github.com/{OWNER}/{REPO}"
        raise


def push_git_repo() -> None:
    subprocess.check_call(
        ["git", "config", "--local", "url.https://github.com/.insteadOf", "git@github.com:"],
        cwd=ROOT,
    )
    subprocess.run(["git", "remote", "remove", "origin"], cwd=ROOT, check=False)
    subprocess.check_call(
        ["git", "remote", "add", "origin", f"git@github.com:{OWNER}/{REPO}.git"],
        cwd=ROOT,
    )
    subprocess.check_call(["git", "push", "-u", "origin", "main", "--force"], cwd=ROOT)
    subprocess.check_call(["git", "push", "origin", f"refs/tags/{TAG}", "--force"], cwd=ROOT)
    print(f"Pushed branch `main` and tag `{TAG}` to git@github.com:{OWNER}/{REPO}.git")


def publish_release(token: str) -> str:
    rel_body = (
        "## Crucible (`crux` `v0.1`) — Source & Compiled Release\n\n"
        "- **Command Name**: `crux` (with `ush` backwards-compatible alias)\n"
        "- **Compiled Native Binary (macOS ARM64 Mach-O)**: `crux-0.1-darwin-arm64` & `crux`\n"
        "- **Compiled Portable Bytecode Executable**: `crux.pyz`\n"
        "- **Source Distribution**: `crucible-0.1-src.tar.gz`\n"
        "- **Checksums**: `SHA256SUMS.txt`\n"
    )
    payload = json.dumps(
        {
            "tag_name": TAG,
            "target_commitish": "main",
            "name": "Crucible (crux) v0.1 — Source & Compiled Release",
            "body": rel_body,
            "draft": False,
            "prerelease": False,
        }
    ).encode("utf-8")

    try:
        rel = api_request(
            f"https://api.github.com/repos/{OWNER}/{REPO}/releases",
            token,
            method="POST",
            data=payload,
        )
    except urllib.error.HTTPError as exc:
        if exc.code == 422:
            rel = api_request(
                f"https://api.github.com/repos/{OWNER}/{REPO}/releases/tags/{TAG}",
                token,
                method="GET",
            )
        else:
            raise

    upload_url_tmpl = rel["upload_url"].split("{", 1)[0]
    existing_assets = {a["name"]: a["id"] for a in rel.get("assets", [])}

    for fname in ("crux-0.1-darwin-arm64", "crux.pyz", "crucible-0.1-src.tar.gz", "SHA256SUMS.txt"):
        fpath = RELEASE_DIR / fname
        if not fpath.exists():
            continue
        if fname in existing_assets:
            aid = existing_assets[fname]
            req = urllib.request.Request(
                f"https://api.github.com/repos/{OWNER}/{REPO}/releases/assets/{aid}",
                headers={"Authorization": f"Bearer {token}", "User-Agent": "crux-cli/0.1"},
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=30):
                pass
        q = urllib.parse.urlencode({"name": fname})
        up_url = f"{upload_url_tmpl}?{q}"
        res = api_request(
            up_url,
            token,
            method="POST",
            data=fpath.read_bytes(),
            content_type="application/octet-stream",
        )
        print(f"  Uploaded release asset: {fname} -> {res.get('browser_download_url')}")

    return rel.get("html_url", f"https://github.com/{OWNER}/{REPO}/releases/tag/{TAG}")


def main() -> int:
    token = get_github_token()
    repo_url = ensure_repo(token)
    push_git_repo()
    rel_url = publish_release(token)
    print(f"SUCCESS: Repo={repo_url} | Release={rel_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
