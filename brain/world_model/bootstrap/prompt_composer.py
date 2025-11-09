"""Prompt composition helpers for LLM-guided world-model bootstrap.

This module is intentionally small and optional. It can enrich the base
bootstrap prompt with recent curiosity/gap signals without changing any
defaults. Callers should guard usage behind feature flags.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
import os
from pathlib import Path
from typing import Any

from .gap_recorder import PredictionGapRecorder
from .registry import HypothesisRegistry
from .scheduler import RefinementScheduler

__all__ = [
    "append_gap_summary",
]


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _safe_truncate(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    # Leave a small indicator at the end when truncating
    tail = "\n... [truncated]"
    return s[: max(0, max_chars - len(tail))] + tail


def _summarize_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Build a compact, schema-free summary of a prediction gap record.

    Only uses primitive JSON types and a stable set of keys so it remains
    robust to recorder/detail format changes.
    """
    summary: dict[str, Any] = {}
    try:
        summary["domain"] = str(record.get("domain") or "unknown")
    except Exception:
        summary["domain"] = "unknown"
    try:
        summary["success_rate"] = float(record.get("success_rate") or 0.0)
    except Exception:
        summary["success_rate"] = 0.0
    try:
        summary["threshold"] = float(record.get("threshold") or 0.0)
    except Exception:
        summary["threshold"] = 0.0
    # Best-effort human-readable description
    desc = ""
    details = record.get("details")
    if isinstance(details, Mapping):
        hyp = details.get("hypothesis")
        if isinstance(hyp, Mapping):
            action = hyp.get("action")
            if isinstance(action, str) and action.strip():
                desc = f"hypothesis:{action.strip()}"
    if not desc:
        try:
            desc = str(details)[:120]
        except Exception:
            desc = ""
    if desc:
        summary["description"] = desc
    return summary


def append_gap_summary(
    base_prompt: str,
    *,
    gap_limit: int | None = None,
    max_chars: int | None = None,
    artifacts_dir: str | None = None,
) -> str:
    """Append a compact curiosity/gap summary block to an existing prompt.

    - Reads the most recent N prediction gap records from the recorder.
    - Appends a JSON array under a short heading to guide the LLM toward
      underperforming areas without changing the core schema section.
    - Truncates to ``max_chars`` if provided.
    """
    limit = gap_limit if gap_limit is not None else _int_env("WM_PROMPT_GAP_N", 3)
    try:
        recorder = (
            PredictionGapRecorder.default()
            if not artifacts_dir
            else PredictionGapRecorder(Path(artifacts_dir).expanduser().resolve() / "world_model" / "gaps")
        )
        records = recorder.list_records(limit=max(1, int(limit)))
    except Exception:
        records = []

    if not records:
        return base_prompt if max_chars is None else _safe_truncate(base_prompt, int(max_chars))

    # Sort by gap size descending (information gain proxy)
    scored: list[Tuple[float, dict[str, Any]]] = []
    for rec in records:
        try:
            s = _summarize_record(rec)
            gap = float(max(0.0, float(s.get("threshold", 0.0)) - float(s.get("success_rate", 0.0))))
            scored.append((gap, s))
        except Exception:
            continue
    scored.sort(key=lambda x: x[0], reverse=True)
    summaries = [item[1] for item in scored[: max(1, int(limit))]] if scored else []

    if not summaries:
        return base_prompt if max_chars is None else _safe_truncate(base_prompt, int(max_chars))

    block = (
        "\n\nCuriosity signals (recent prediction gaps):\n" + json.dumps(summaries, ensure_ascii=True)
    )
    composed = base_prompt + block
    if max_chars is not None:
        return _safe_truncate(composed, int(max_chars))
    return composed


def append_curiosity_tasks(
    base_prompt: str,
    *,
    task_limit: int | None = None,
    max_chars: int | None = None,
    artifacts_dir: str | None = None,
) -> str:
    """Append a compact summary of curiosity tasks from the refinement queue.

    Reads entries recorded via HypothesisRegistry.record_curiosity_task and
    includes the most recent ones tagged with task_type="curiosity_gap".
    """
    limit = task_limit if task_limit is not None else _int_env("WM_PROMPT_TASK_N", 3)
    try:
        reg = HypothesisRegistry(artifacts_dir=artifacts_dir) if artifacts_dir else HypothesisRegistry()
        sched = RefinementScheduler(reg)
        queue = list(sched.get_queue())
    except Exception:
        queue = []

    tasks = [
        item
        for item in queue
        if isinstance(item, Mapping) and str(item.get("task_type") or "") == "curiosity_gap"
    ]
    if not tasks:
        return base_prompt if max_chars is None else _safe_truncate(base_prompt, int(max_chars))

    # Most recent last in file; take tail
    # Sort tasks by priority or urgency*learnability
    scored_tasks: list[Tuple[float, Mapping[str, Any]]] = []
    for t in tasks:
        try:
            meta = t.get("metadata") or {}
            prio = float(meta.get("priority")) if isinstance(meta, Mapping) and "priority" in meta else None
        except Exception:
            prio = None
        if prio is None:
            try:
                prio = float(t.get("urgency", 0.0)) * float(t.get("learnability", 0.0))
            except Exception:
                prio = 0.0
        scored_tasks.append((float(prio), t))
    scored_tasks.sort(key=lambda x: x[0], reverse=True)
    picked = [item[1] for item in scored_tasks[: max(1, int(limit))]]
    summaries: list[dict[str, Any]] = []
    for t in picked:
        try:
            summaries.append(
                {
                    "domain": str(t.get("domain") or "unknown"),
                    "urgency": float(t.get("urgency") or 0.0),
                    "learnability": float(t.get("learnability") or 0.0),
                    "description": str(t.get("description") or ""),
                    "questions": list((t.get("metadata") or {}).get("questions") or []),
                },
            )
        except Exception:
            continue

    if not summaries:
        return base_prompt if max_chars is None else _safe_truncate(base_prompt, int(max_chars))

    block = (
        "\n\nCuriosity tasks (from refinement queue):\n" + json.dumps(summaries, ensure_ascii=True)
    )
    composed = base_prompt + block
    if max_chars is not None:
        return _safe_truncate(composed, int(max_chars))
    return composed
