"""Snapshot persistence helpers for world-model bootstrap.

Provides small utilities to load/save `WorldModelSnapshot` objects and compute
their SHA-256 digests. These helpers are intentionally lightweight so callers
can integrate snapshots into proofs or CLI tools.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .schemas import WorldModelSnapshot


def _compute_sha256_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()


def load_snapshot(path: str | Path) -> tuple[WorldModelSnapshot, str]:
    """Load a JSON snapshot from ``path`` and return (snapshot, sha256).

    Raises FileNotFoundError or pydantic.ValidationError on malformed input.
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    data = json.loads(text)
    snap = WorldModelSnapshot.model_validate(data)
    sha = _compute_sha256_text(text)
    return snap, sha


def save_snapshot(snapshot: WorldModelSnapshot, path: str | Path) -> str:
    """Write a `WorldModelSnapshot` to ``path`` (pretty JSON) and return its sha256."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(snapshot.model_dump(), sort_keys=True, ensure_ascii=True, indent=2)
    p.write_text(text, encoding="utf-8")
    return _compute_sha256_text(text)


__all__ = ["load_snapshot", "save_snapshot"]
