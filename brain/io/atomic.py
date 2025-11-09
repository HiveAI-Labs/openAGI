"""Atomic file writing helpers used across the brain codebase."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Union

__all__ = [
    "atomic_write_text",
    "atomic_write_bytes",
    "atomic_write_json",
    "atomic_write_json_lines",
]

PathLike = Union[str, os.PathLike]


def atomic_write_text(path: PathLike, data: str, encoding: str = "utf-8") -> None:
    """Atomically write text data to *path* using a temporary file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(dest.parent), encoding=encoding) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomically write binary data to *path* using a temporary file."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(dest.parent)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    tmp_path.replace(dest)


def atomic_write_json(path: PathLike, payload: Mapping[str, Any]) -> None:
    """Atomically write a single JSON document with deterministic key ordering."""
    serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True)
    atomic_write_text(path, serialized + "\n")


def atomic_write_json_lines(path: PathLike, records: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write an iterable of JSON records as newline-delimited JSON."""
    lines = [json.dumps(record, ensure_ascii=True, sort_keys=True) for record in records]
    data = "\n".join(lines)
    if data:
        data += "\n"
    atomic_write_text(path, data)

