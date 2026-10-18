from __future__ import annotations

import fcntl
import hashlib
import shutil
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout, stderr=err)


def _clone(url: str, target: Path) -> None:
    if target.exists() and not (target / ".git").exists():
        shutil.rmtree(target)
    _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, str(target)])


def ensure_checkout(repo: str, commit: str, cache_root: str | Path) -> Path:
    """Create a read-only detached checkout for a public GitHub repo/base commit."""
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    safe_repo = repo.replace("/", "__")
    key = hashlib.sha1(f"{repo}@{commit}".encode()).hexdigest()[:10]
    target = cache_root / f"{safe_repo}__{commit[:10]}__{key}"
    if (target / ".git").exists():
        return target
    lock_path = cache_root / f".{safe_repo}__{commit[:10]}__{key}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target / ".git").exists():
            return target
        url = f"https://github.com/{repo}.git"
        try:
            _clone(url, target)
        except subprocess.CalledProcessError:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            _clone(url, target)
        try:
            _run(["git", "checkout", "--detach", commit], cwd=target)
        except subprocess.CalledProcessError:
            _run(["git", "fetch", "origin", commit, "--depth", "1"], cwd=target)
            _run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)
    return target
