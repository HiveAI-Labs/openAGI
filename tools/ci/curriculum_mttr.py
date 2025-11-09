from __future__ import annotations

"""Shared helpers for curriculum acknowledgement MTTR gating."""

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from brain.io.atomic import atomic_write_json


def _safe_float(value: Any) -> Optional[float]:
    """Return ``float(value)`` when possible, otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    """Return ``int(value)`` when possible, otherwise ``None``."""

    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def resolve_mttr_config(
    *,
    tolerance_env: str = "PROOF_CURRICULUM_MTTR_TOLERANCE",
    baseline_min_env: str = "PROOF_CURRICULUM_MTTR_BASELINE_MIN",
    history_env: str = "PROOF_CURRICULUM_MTTR_HISTORY",
    default_tolerance: float = 0.10,
    default_baseline_min: int = 3,
    default_history: int = 30,
) -> Tuple[float, int, int]:
    """Resolve MTTR drift configuration from the environment."""

    tolerance_value = _safe_float(os.getenv(tolerance_env))
    if tolerance_value is None:
        tolerance_value = default_tolerance
    tolerance_value = max(0.0, float(tolerance_value))

    baseline_min = _safe_int(os.getenv(baseline_min_env))
    if baseline_min is None or baseline_min <= 0:
        baseline_min = default_baseline_min
    baseline_min = max(1, baseline_min)

    history_limit = _safe_int(os.getenv(history_env))
    if history_limit is None or history_limit <= 0:
        history_limit = default_history
    history_limit = max(baseline_min, history_limit)

    return tolerance_value, baseline_min, history_limit


def load_mttr_history(path: Path) -> List[Dict[str, float]]:
    """Load persisted MTTR history entries sorted by timestamp."""

    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    source: List[Dict[str, Any]]
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        source = [entry for entry in payload.get("history", []) if isinstance(entry, dict)]
    elif isinstance(payload, list):
        source = [entry for entry in payload if isinstance(entry, dict)]
    else:
        source = []

    history: List[Dict[str, float]] = []
    for entry in source:
        mean_value = _safe_float(entry.get("mean"))
        p95_value = _safe_float(entry.get("p95"))
        ts_value = _safe_float(entry.get("ts")) or 0.0
        if mean_value is None or p95_value is None:
            continue
        history.append({"ts": float(ts_value), "mean": float(mean_value), "p95": float(p95_value)})

    history.sort(key=lambda item: item.get("ts", 0.0))
    return history


def compute_mttr_baseline(history: List[Dict[str, float]], key: str) -> Optional[float]:
    """Return the arithmetic mean baseline for ``key`` across historic MTTR entries."""

    values = [float(entry.get(key)) for entry in history if _safe_float(entry.get(key)) is not None]
    if not values:
        return None
    try:
        return float(statistics.fmean(values))
    except statistics.StatisticsError:
        return None


def append_mttr_history(
    path: Path,
    history: List[Dict[str, float]],
    entry: Dict[str, Any],
    *,
    max_entries: int,
    timestamp: Optional[float] = None,
) -> List[Dict[str, float]]:
    """Append a MTTR sample to history and persist it atomically."""

    mean_value = _safe_float(entry.get("mean"))
    p95_value = _safe_float(entry.get("p95"))
    if mean_value is None or p95_value is None:
        return history

    ts_value = _safe_float(entry.get("ts"))
    if ts_value is None:
        ts_value = timestamp if timestamp is not None else time.time()

    items = [dict(item) for item in history]
    items.append({"ts": float(ts_value), "mean": float(mean_value), "p95": float(p95_value)})
    if max_entries > 0 and len(items) > max_entries:
        items = items[-max_entries:]

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(path, {"history": items})
    except Exception:
        pass

    items.sort(key=lambda item: item.get("ts", 0.0))
    return items


def build_mttr_gate_payload(
    mttr_stats: Optional[Dict[str, Any]],
    *,
    history_path: Path,
    tolerance: float,
    baseline_min: int,
    history_limit: int,
    update_history: bool = True,
    timestamp: Optional[float] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    """Derive the MTTR gate payload and optionally persist the newest sample."""

    history = load_mttr_history(history_path)
    baseline_mean = compute_mttr_baseline(history, "mean")
    baseline_p95 = compute_mttr_baseline(history, "p95")
    threshold_ratio = 1.0 + max(0.0, tolerance)

    gate_payload: Dict[str, Any] = {
        "baseline_samples": len(history),
        "tolerance_ratio": max(0.0, tolerance),
        "threshold_ratio": threshold_ratio,
        "baseline_mean_seconds": baseline_mean,
        "baseline_p95_seconds": baseline_p95,
        "ok": True,
    }

    breaches: List[str] = []
    mttr_count = _safe_float((mttr_stats or {}).get("count"))
    if mttr_stats and mttr_count and mttr_count > 0.0:
        mean_seconds = _safe_float(mttr_stats.get("mean")) or 0.0
        p95_seconds = _safe_float(mttr_stats.get("p95")) or 0.0
        gate_payload.update(
            {
                "mean_seconds": float(mean_seconds),
                "p95_seconds": float(p95_seconds),
                "sample_count": float(mttr_count),
            }
        )

        if baseline_mean and baseline_mean > 0.0 and len(history) >= baseline_min:
            mean_ratio = float(mean_seconds) / max(baseline_mean, 1e-9)
            gate_payload["mean_ratio"] = mean_ratio
            if mean_ratio > threshold_ratio:
                gate_payload["ok"] = False
                breaches.append("mean")
        else:
            gate_payload["mean_ratio"] = None

        if baseline_p95 and baseline_p95 > 0.0 and len(history) >= baseline_min:
            p95_ratio = float(p95_seconds) / max(baseline_p95, 1e-9)
            gate_payload["p95_ratio"] = p95_ratio
            if p95_ratio > threshold_ratio:
                gate_payload["ok"] = False
                breaches.append("p95")
        else:
            gate_payload["p95_ratio"] = None

        if update_history:
            history = append_mttr_history(
                history_path,
                history,
                {"ts": timestamp if timestamp is not None else time.time(), "mean": mean_seconds, "p95": p95_seconds},
                max_entries=history_limit,
                timestamp=timestamp,
            )
        gate_payload["baseline_samples_after"] = len(history)
    else:
        gate_payload["reason"] = "insufficient_mttr_samples"
        gate_payload["baseline_samples_after"] = len(history)

    if breaches:
        gate_payload["breach_dimensions"] = breaches

    return gate_payload, history
