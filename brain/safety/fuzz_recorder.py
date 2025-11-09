"""Utilities for recording fuzz and chaos test results into proof artifacts."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from brain.io.atomic import atomic_write_text

__all__ = ["FuzzRecord", "resolve_artifacts_dir", "record_fuzz_result", "read_fuzz_records"]


@dataclass(slots=True)
class FuzzRecord:
    """Structured representation of a single fuzz or chaos test iteration."""

    suite: str
    timestamp: str
    payload: dict[str, Any]
    sha256: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "timestamp": self.timestamp,
            "payload": self.payload,
            "sha256": self.sha256,
        }


def resolve_artifacts_dir() -> Path:
    """Return the configured artifacts directory (default: ``artifacts``)."""
    root = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
    return Path(root).resolve()


def _load_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return []
    lines = [line for line in raw.splitlines() if line.strip()]
    records: list[dict[str, Any]] = []
    for line in lines:
        try:
            parsed = json.loads(line)
        except Exception:
            continue
        if isinstance(parsed, dict):
            records.append(parsed)
    return records


def _serialize_records(records: Iterable[dict[str, Any]]) -> str:
    data = "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records if record)
    if data:
        data += "\n"
    return data


def record_fuzz_result(
    artifacts_dir: Path | str,
    suite: str,
    payload: dict[str, Any],
    *,
    sha256: str | None = None,
) -> Path:
    """Append a fuzz test result to ``artifacts/proof/fuzz/<suite>.jsonl``.

    The write is performed atomically (temp file + rename) to satisfy the
    repository's durability requirements.
    """
    if not suite or not isinstance(suite, str):
        raise ValueError("suite name must be a non-empty string")

    base = Path(artifacts_dir).resolve()
    out_dir = base / "proof" / "fuzz"
    out_path = out_dir / f"{suite}.jsonl"

    timestamp = datetime.now(UTC).isoformat()
    record = FuzzRecord(suite=suite, timestamp=timestamp, payload=dict(payload), sha256=sha256)

    existing = _load_existing_records(out_path)
    existing.append(record.to_json())
    serialized = _serialize_records(existing)
    atomic_write_text(out_path, serialized, encoding="utf-8")
    return out_path


def read_fuzz_records(path: Path) -> list[FuzzRecord]:
    """Load fuzz records from ``path`` for validation in tests."""
    records = _load_existing_records(path)
    parsed: list[FuzzRecord] = []
    for item in records:
        suite = str(item.get("suite"))
        timestamp = str(item.get("timestamp"))
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        sha256 = item.get("sha256")
        parsed.append(FuzzRecord(suite=suite, timestamp=timestamp, payload=dict(payload), sha256=sha256 if isinstance(sha256, str) else None))
    return parsed
