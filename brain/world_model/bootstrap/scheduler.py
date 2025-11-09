"""Simple refinement scheduler for processing recorded snapshots.

This module provides a tiny scheduler that reads the registry's refinement queue
and allows a curator or operator to peek/pop entries for downstream processing.
It deliberately stays small — the registry owns the queue persistence and
metrics; the scheduler exposes a clean API for worker loops.
"""

from __future__ import annotations

from brain.io.atomic import atomic_write_json_lines
from brain.obs.metrics import (
    brain_wm_refinement_events_total,
    brain_wm_refinement_queue_size,
    brain_wm_retrain_events_total,
    brain_wm_retrain_queue_size,
)
from brain.world_model.bootstrap.registry import HypothesisRegistry


class RefinementScheduler:
    """Cursor over `HypothesisRegistry`'s refinement queue.

    Note: uses the registry's on-disk queue; operations acquire the registry lock
    to avoid races with concurrent writers.
    """

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry

    def peek_next(self) -> dict[str, object] | None:
        q = self.registry._load_refinement_queue()
        return q[0] if q else None

    def get_queue(self) -> list[dict[str, object]]:
        return list(self.registry._load_refinement_queue())

    def pop_next(self) -> dict[str, object] | None:
        with self.registry._lock:
            q = self.registry._load_refinement_queue()
            if not q:
                return None
            item = q.pop(0)
            atomic_write_json_lines(self.registry._refinement_queue_path, q)
            try:
                brain_wm_refinement_events_total.inc(event="popped")
                brain_wm_refinement_queue_size.set(len(q))
            except Exception:
                pass
            return item

    def pop_by_snapshot(self, snapshot_path: str) -> dict[str, object] | None:
        normalized = snapshot_path.strip()
        if not normalized:
            return None
        with self.registry._lock:
            q = self.registry._load_refinement_queue()
            match: dict[str, object] | None = None
            remaining: list[dict[str, object]] = []
            for item in q:
                item_path = str(item.get("snapshot_path", "")).strip()
                if match is None and item_path == normalized:
                    match = item
                    continue
                remaining.append(item)
            if match is None:
                return None
            atomic_write_json_lines(self.registry._refinement_queue_path, remaining)
            try:
                brain_wm_refinement_events_total.inc(event="popped")
                brain_wm_refinement_queue_size.set(len(remaining))
            except Exception:
                pass
            return match


class RetrainScheduler:
    """Scheduler for ensemble retrain queue records."""

    def __init__(self, registry: HypothesisRegistry) -> None:
        self.registry = registry

    def get_queue(self) -> list[dict[str, object]]:
        return list(self.registry._load_retrain_queue())

    def pop_next(self) -> dict[str, object] | None:
        with self.registry._lock:
            q = self.registry._load_retrain_queue()
            if not q:
                return None
            item = q.pop(0)
            atomic_write_json_lines(self.registry._retrain_queue_path, q)
            try:
                brain_wm_retrain_events_total.inc(event="popped")
                brain_wm_retrain_queue_size.set(len(q))
            except Exception:
                pass
            return item


__all__ = ["RefinementScheduler", "RetrainScheduler"]
