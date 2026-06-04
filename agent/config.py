"""
Zero-dependency .env loader for the Agentic Perception Platform.

Reads KEY=VALUE pairs from a .env file at the repo root and injects them into
os.environ. We use os.environ.setdefault so a real `export ANTHROPIC_API_KEY=...`
in your shell always wins over the file. Avoids a python-dotenv dependency.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def load_env(path: str | Path | None = None) -> None:
    """Populate os.environ from a .env file (no-op if the file is absent)."""
    env_path = Path(path) if path else _REPO_ROOT / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)
