"""Background worker that consumes refinement queue entries.

This worker is intentionally small and synchronous — it pops the next
refinement queue entry, does a placeholder processing step (which a real
system would replace with a refinement job/validation), and emits a metric.

It's safe to run from a cron/loop or supervisor; callers can import `process_once`
for unit testing.
"""

from __future__ import annotations

from .registry import HypothesisRegistry
from .scheduler import RefinementScheduler, RetrainScheduler


def process_entry(record: dict) -> dict:
    """Process a single refinement queue record (placeholder).

    In real use this function would: load snapshot.json, schedule model retrain,
    kick off validation jobs, or hand off to a remote worker. Here we return a
    small result summary to keep the function testable.
    """
    # Minimal sanity: ensure snapshot_path present
    out = {"snapshot_path": record.get("snapshot_path"), "status": "processed"}
    # pretend we computed something
    out["processed_at"] = "now"
    return out


def process_once(registry: HypothesisRegistry | None = None) -> dict | None:
    """Pop next queued record and process it once. Returns result or None if queue empty."""
    reg = registry or HypothesisRegistry()
    sched = RefinementScheduler(reg)
    item = sched.pop_next()
    if not item:
        return None
    # In tests we'll pass small synthetic records
    result = process_entry(item)
    return result


def process_retrain_once(registry: HypothesisRegistry | None = None) -> dict | None:
    """Pop next retrain trigger and process it once."""
    reg = registry or HypothesisRegistry()
    sched = RetrainScheduler(reg)
    item = sched.pop_next()
    if not item:
        return None
    result = process_entry(item)
    result["queue"] = "retrain"
    return result


__all__ = ["process_once", "process_entry", "process_retrain_once"]
