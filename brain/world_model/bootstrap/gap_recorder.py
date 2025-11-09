"""Persistence helpers for knowledge gap (prediction error) records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from brain.io.atomic import atomic_write_text

__all__ = ["PredictionGapRecorder"]


def _artifacts_root() -> Path:
    return Path(os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts").resolve()


@dataclass
class PredictionGapRecorder:
    """Write prediction gap records to artifacts/world_model/gaps."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._log_path = self.directory / "gaps.jsonl"

    @classmethod
    def default(cls) -> PredictionGapRecorder:
        base = _artifacts_root() / "world_model" / "gaps"
        return cls(base)

    def record(
        self,
        *,
        domain: str,
        success_rate: float,
        threshold: float,
        details: Mapping[str, Any],
    ) -> Path:
        timestamp = datetime.now(UTC).isoformat()
        payload: dict[str, Any] = {
            "timestamp": timestamp,
            "domain": domain,
            "success_rate": float(success_rate),
            "threshold": float(threshold),
            "details": dict(details),
        }
        line = json.dumps(payload, ensure_ascii=False)
        atomic_write_text(self._log_path, line + "\n", encoding="utf-8")
        return self._log_path

    def list_records(self, limit: int | None = None) -> Sequence[Mapping[str, Any]]:
        if not self._log_path.exists():
            return []
        try:
            raw = self._log_path.read_text(encoding="utf-8")
        except Exception:
            return []
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if limit is not None:
            lines = lines[-limit:]
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
        return records
