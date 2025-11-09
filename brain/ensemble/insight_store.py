"""Persistence helpers for recording ensemble query insights."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from brain.io.atomic import atomic_write_text

__all__ = ["EnsembleInsightStore", "record_ensemble_insight"]


def _artifacts_root() -> Path:
    base = os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
    return Path(base).resolve()


def _default_store_dir() -> Path:
    return _artifacts_root() / "ensemble" / "archive"


@dataclass
class EnsembleInsightStore:
    """Simple JSONL-backed store for ensemble answers."""

    directory: Path

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        question: str,
        context: Sequence[str],
        answer_payload: Mapping[str, Any],
        metadata: Mapping[str, Any],
    ) -> Path:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        filename = f"insight_{timestamp}.jsonl"
        path = self.directory / filename

        record: dict[str, Any] = {
            "timestamp": timestamp,
            "question": question,
            "context": list(context),
            "answer": answer_payload,
            "metadata": dict(metadata),
        }
        serialized = json.dumps(record, ensure_ascii=False)
        atomic_write_text(path, serialized + "\n", encoding="utf-8")
        return path


def record_ensemble_insight(
    question: str,
    context: Sequence[str],
    answer_payload: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    store: EnsembleInsightStore | None = None,
) -> Path:
    """Record an ensemble answer when logging is enabled."""
    active_store = store or EnsembleInsightStore(_default_store_dir())
    return active_store.record(
        question=question,
        context=context,
        answer_payload=answer_payload,
        metadata=metadata,
    )
