"""Shared helpers for writing and summarizing selector consult events."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
import json
import math
import os
from pathlib import Path
import statistics
import threading
import time
from typing import Any

__all__ = [
    "append_consult_event",
    "consult_log_path",
    "load_consult_events",
    "consult_sessions",
    "summarize_consults",
]

_LOCK = threading.Lock()
_DISABLE_VALUES = {"1", "true", "yes", "on"}
_CONSULT_EVENTS = {"consult_request", "consult_reply", "consult_complete"}


def _persist_enabled() -> bool:
    flag = os.getenv("BRAIN_SELECTOR_DISABLE_PERSIST", "")
    return flag.strip().lower() not in _DISABLE_VALUES


def _artifacts_root() -> Path:
    base = os.getenv("BRAIN_SELECTOR_CONSULT_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or os.getenv("ARTIFACTS_DIR")
    return Path(base or "artifacts").resolve()


def consult_log_path() -> Path:
    path_override = os.getenv("BRAIN_SELECTOR_CONSULT_LOG")
    if path_override:
        return Path(path_override)
    return _artifacts_root() / "selector" / "consults.jsonl"


def append_consult_event(event: Mapping[str, Any]) -> None:
    """Append a consult event to the shared JSONL file."""
    if not _persist_enabled():
        return
    payload = dict(event)
    payload.setdefault("ts", time.time())
    path = consult_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _LOCK:
        try:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(serialized)
        except Exception:
            return


def load_consult_events(*, path: Path | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    """Load consult events from disk (most recent first when limit provided)."""
    log_path = path or consult_log_path()
    if not log_path.exists():
        return []
    try:
        rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except Exception:
        return []
    events = [row for row in rows if isinstance(row, Mapping) and row.get("event") in _CONSULT_EVENTS]
    if limit is not None:
        events = events[-int(max(1, limit)) :]
    return [dict(evt) for evt in events]


def consult_sessions(events: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Group consult events into sessions keyed by decision_id."""
    sessions: MutableMapping[str, dict[str, Any]] = {}
    order: list[str] = []
    for event in events:
        decision_id = str(event.get("decision_id") or "")
        if not decision_id:
            continue
        session = sessions.get(decision_id)
        if session is None:
            session = {
                "decision_id": decision_id,
                "request": None,
                "replies": [],
                "complete": None,
                "first_ts": None,
                "last_ts": None,
            }
            sessions[decision_id] = session
            order.append(decision_id)
        ts_val = event.get("ts")
        if isinstance(ts_val, (int, float)):
            ts = float(ts_val)
            session["first_ts"] = ts if session["first_ts"] is None else min(session["first_ts"], ts)
            session["last_ts"] = ts if session["last_ts"] is None else max(session["last_ts"], ts)
        kind = event.get("event")
        if kind == "consult_request":
            session["request"] = dict(event)
        elif kind == "consult_reply":
            session["replies"].append(dict(event))
        elif kind == "consult_complete":
            session["complete"] = dict(event)
    return [sessions[key] for key in order]


def _percentile(values: Sequence[float], pct: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    k = (len(data) - 1) * pct
    low = math.floor(k)
    high = math.ceil(k)
    if low == high:
        return float(data[int(k)])
    return float(data[low] + (data[high] - data[low]) * (k - low))


def summarize_consults(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute aggregate statistics across consult events."""
    generated = time.time()
    sessions = consult_sessions(events)
    request_count = 0
    reply_count = 0
    completion_count = 0
    reason_counts: dict[str, int] = {}
    model_counts: dict[str, int] = {}
    outcome_counts: dict[str, int] = {}
    timeouts = 0
    errors = 0
    reply_latencies: list[float] = []
    consensus_values: list[float] = []
    confidence_values: list[float] = []
    open_sessions = 0
    recent_sessions: list[dict[str, Any]] = []

    for session in sessions:
        request = session.get("request") or {}
        complete = session.get("complete")
        replies = session.get("replies") or []

        if request:
            request_count += 1
            labels = request.get("reason_labels") or []
            for label in labels:
                if not label:
                    continue
                key = str(label)
                reason_counts[key] = reason_counts.get(key, 0) + 1
            for name in request.get("models") or []:
                if not name:
                    continue
                key = str(name)
                model_counts[key] = model_counts.get(key, 0) + 1

        if complete:
            completion_count += 1
            outcome = str(complete.get("outcome") or "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
            consensus = complete.get("consensus")
            if isinstance(consensus, Mapping):
                avg = consensus.get("average_consensus")
                if isinstance(avg, (int, float)):
                    consensus_values.append(float(avg))
            conf = complete.get("confidence")
            if isinstance(conf, (int, float)):
                confidence_values.append(float(conf))
        else:
            open_sessions += 1

        for reply in replies:
            reply_count += 1
            latency = reply.get("latency_ms")
            if isinstance(latency, (int, float)):
                reply_latencies.append(float(latency))
            if reply.get("timed_out"):
                timeouts += 1
            if reply.get("error"):
                errors += 1

        recent_sessions.append(
            {
                "decision_id": session.get("decision_id"),
                "requested_at": (request or {}).get("ts"),
                "completed": bool(complete),
                "outcome": (complete or {}).get("outcome"),
                "reason_labels": list((request or {}).get("reason_labels") or []),
                "task_type": (request or {}).get("task_type"),
                "selector_confidence": (request or {}).get("selector_confidence"),
                "models": list((request or {}).get("models") or []),
                "model_count": len((request or {}).get("models") or []),
                "reply_count": len(replies),
                "duration_ms": (
                    (session.get("last_ts") - session.get("first_ts")) * 1000.0
                    if session.get("last_ts") is not None and session.get("first_ts") is not None
                    else None
                ),
            },
        )

    latency_summary = {
        "samples": len(reply_latencies),
        "p50": _percentile(reply_latencies, 0.5),
        "p95": _percentile(reply_latencies, 0.95),
        "max": max(reply_latencies) if reply_latencies else None,
    }

    completion_ratio = completion_count / request_count if request_count else None
    result = {
        "generated_at": generated,
        "events": len(events),
        "sessions": {
            "total": len(sessions),
            "completed": completion_count,
            "pending": open_sessions,
            "completion_ratio": completion_ratio,
        },
        "counts": {
            "requests": request_count,
            "replies": reply_count,
            "completions": completion_count,
            "timeouts": timeouts,
            "errors": errors,
        },
        "reasons": reason_counts,
        "models": model_counts,
        "outcomes": outcome_counts,
        "latency_ms": latency_summary,
        "consensus": {
            "avg": statistics.mean(consensus_values) if consensus_values else None,
            "samples": len(consensus_values),
        },
        "confidence": {
            "avg": statistics.mean(confidence_values) if confidence_values else None,
            "samples": len(confidence_values),
        },
        "recent_sessions": recent_sessions[-20:],
    }
    return result
