"""
Run provenance: which code and which config produced a number (operating rule 8).

Every scorecard row carries the git SHA it was computed at and a hash of each
config it used, so any number in results/scorecard.csv can be traced back to the
exact code and settings behind it. The trainers will call the same helpers (P14-5)
so a checkpoint and the row that scores it share one vocabulary.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union

_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str, cwd: Optional[Path] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", *args], cwd=cwd or _REPO_ROOT, capture_output=True,
                             text=True, timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def git_info(cwd: Optional[Path] = None) -> Dict[str, Union[str, bool, None]]:
    """
    {"sha": full commit SHA or "unknown", "dirty": True if tracked files differ
    from HEAD, None when git is unavailable}.

    Untracked files are ignored: new checkpoints, logs and results appear on
    every run and do not change the code that produced the number.
    """
    sha = _git("rev-parse", "HEAD", cwd=cwd)
    if not sha:
        return {"sha": "unknown", "dirty": None}
    status = _git("status", "--porcelain", "--untracked-files=no", cwd=cwd)
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def config_hash(cfg: dict) -> str:
    """
    12-hex-character SHA-256 of the PARSED config.

    Hashing the parsed dict rather than the file bytes means reordering keys or
    editing comments does not change the hash, while any change to a value does.
    That matches rule 7: two ablation arms may differ in their comments, never in
    their settings.
    """
    canonical = json.dumps(cfg, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
