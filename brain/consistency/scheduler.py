from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from brain.obs.metrics import (
    brain_consistency_goals_scheduled_total,
    brain_consistency_scheduler_budget,
    brain_consistency_scheduler_events_total,
    brain_consistency_scheduler_latency_ms,
)

QUOTAS_PATH = "consistency/quotas/quotas.json"

def _load_quotas(root: Path) -> dict[str, Any]:
    path = root / QUOTAS_PATH
    if not path.exists():
        return {"suites": {}, "updated_at": None}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"suites": {}, "updated_at": None}

def _persist_quotas(root: Path, quotas: dict[str, Any]) -> None:
    path = root / QUOTAS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(quotas, indent=2), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        # best-effort fallback
        try:
            path.write_text(json.dumps(quotas), encoding="utf-8")
        except Exception:
            pass


def _artifacts_root() -> Path:
    env_dir = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    return Path(env_dir) if env_dir else Path("artifacts")


def _load_latest_goals(root: Path) -> dict[str, Any]:
    latest = root / "consistency" / "goals" / "latest.json"
    if not latest.exists():
        return {"goals": [], "goal_count": 0}
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {"goals": [], "goal_count": 0}


def schedule_consistency_goals(limit: int | None = None) -> dict[str, Any]:
    """Schedule consistency remediation goals into the refinement queue.

    This writes entries to artifacts/world_model/bootstrap/refinement_queue.jsonl as a concrete,
    already-tracked queue. Enforces a hard safety switch via BRAIN_ENABLE_CONSISTENCY_SCHEDULER=1.

    Returns a summary including count and first/last ids.
    """
    if os.getenv("BRAIN_ENABLE_CONSISTENCY_SCHEDULER", "0").lower() not in {"1", "true", "yes", "on"}:
        brain_consistency_goals_scheduled_total.inc(outcome="disabled")
        brain_consistency_scheduler_events_total.inc(outcome="disabled")
        return {"ok": False, "reason": "disabled"}

    start_time = datetime.now(UTC)
    root = _artifacts_root()
    goals_payload = _load_latest_goals(root)
    goals = goals_payload.get("goals") or []
    if not isinstance(goals, list) or not goals:
        # Even when there are no goals, record an event and persist a minimal last_run
        brain_consistency_goals_scheduled_total.inc(outcome="empty")
        brain_consistency_scheduler_events_total.inc(outcome="empty")
        elapsed_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000.0
        # Record latency metric as well for observability
        try:
            brain_consistency_scheduler_latency_ms.observe(elapsed_ms)
        except Exception:
            pass
        summary: dict[str, Any] = {
            "ok": True,
            "scheduled": 0,
            "reason": "no_goals",
            "queue_path": str((root / "world_model" / "bootstrap" / "refinement_queue.jsonl").resolve()),
            "first_id": None,
            "last_id": None,
            "budget_before": None,
            "budget_remaining": None,
            "quotas_path": str((root / QUOTAS_PATH).resolve()),
            "suites": {},
            "timestamp": datetime.now(UTC).isoformat(),
            "latency_ms": elapsed_ms,
        }
        try:
            sched_dir = root / "consistency" / "scheduler"
            sched_dir.mkdir(parents=True, exist_ok=True)
            last_run = sched_dir / "last_run.json"
            tmp = last_run.with_suffix(".tmp")
            tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            tmp.replace(last_run)
        except Exception:
            pass
        return summary

    # Cap schedule size
    if limit is not None:
        try:
            n = max(0, int(limit))
        except Exception:
            n = 0
        if n >= 0:
            goals = goals[:n]

    # Global budget enforcement: env BRAIN_CONSISTENCY_SCHEDULER_BUDGET (default large)
    raw_budget = os.getenv("BRAIN_CONSISTENCY_SCHEDULER_BUDGET", "100")
    try:
        budget = max(0, int(raw_budget))
    except Exception:
        budget = 0
    if budget <= 0:
        brain_consistency_scheduler_events_total.inc(outcome="budget_exhausted")
        brain_consistency_scheduler_budget.set(0)
        return {"ok": True, "scheduled": 0, "reason": "budget_exhausted"}
    if len(goals) > budget:
        goals = goals[:budget]

    # Refinement queue: re-use world model bootstrap queue to focus contradiction fixes
    queue_path = root / "world_model" / "bootstrap" / "refinement_queue.jsonl"
    queue_path.parent.mkdir(parents=True, exist_ok=True)

    scheduled = 0
    first_id: str | None = None
    last_id: str | None = None

    # Risk scoring: recompute priority using simple frequency heuristic over existing queue
    existing: dict[str, int] = {}
    if queue_path.exists():
        try:
            for ln in queue_path.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                if rec.get("type") == "consistency_goal":
                    sid = rec.get("scenario")
                    if isinstance(sid, str):
                        existing[sid] = existing.get(sid, 0) + 1
        except Exception:
            pass

    # Load per-suite quotas (daily): structure {suites: {suite: {used:int, limit:int}}, updated_at: iso}
    quotas = _load_quotas(root)
    suites_quota = quotas.get("suites") if isinstance(quotas.get("suites"), dict) else {}
    # Reset quotas if date boundary crossed
    today = datetime.now(UTC).date().isoformat()
    updated_at = quotas.get("updated_at")
    if not isinstance(updated_at, str) or updated_at.split("T")[0] != today:
        suites_quota = {}
        quotas = {"suites": suites_quota, "updated_at": datetime.now(UTC).isoformat()}

    # Resolve configured limits from env: BRAIN_CONSISTENCY_SUITE_QUOTAS_JSON='{"suite_a":5,"suite_b":2}'
    limits_raw = os.getenv("BRAIN_CONSISTENCY_SUITE_QUOTAS_JSON", "{}")
    try:
        suite_limits = json.loads(limits_raw)
        if not isinstance(suite_limits, dict):
            suite_limits = {}
    except Exception:
        suite_limits = {}

    # Filter out goals exceeding per-suite remaining quotas within this run (considering already-used)
    filtered: list[dict[str, Any]] = []
    planned: dict[str, int] = {}
    for g in goals:
        if not isinstance(g, dict):
            continue
        suite = g.get("suite")
        if not isinstance(suite, str):
            continue
        limit = int(suite_limits.get(suite, 9999) or 0)
        state = suites_quota.setdefault(suite, {"used": 0, "limit": limit})
        state["limit"] = limit  # update limit if env changed
        used = int(state.get("used", 0))
        remaining = max(0, limit - used)
        count_planned = planned.get(suite, 0)
        if count_planned >= remaining:
            # No remaining budget for this suite in this run
            continue
        planned[suite] = count_planned + 1
        filtered.append(g)
    goals = filtered

    # Re-weight goals risk-aware:
    # - frequency boost: +0.1 per prior occurrence in queue (capped at 3)
    # - severity boost: +0.2 * normalized severity (from goal.severity or risk_score)
    # - recency boost: +0.1 * (1 - age_hours/48) clipped to [0,1]
    for g in goals:
        if isinstance(g, dict):
            scn = g.get("scenario")
            if isinstance(scn, str):
                freq = existing.get(scn, 0)
                try:
                    base = float(g.get("priority") or 1.0)
                except Exception:
                    base = 1.0
                # severity from goal payload
                sev_raw = g.get("severity")
                if sev_raw is None:
                    sev_raw = g.get("risk_score")
                try:
                    sev = max(0.0, float(sev_raw or 0.0))
                except Exception:
                    sev = 0.0
                # normalize severity with soft cap: sev_norm in [0,1] using sev/(sev+3)
                sev_norm = sev / (sev + 3.0) if sev > 0 else 0.0
                # recency from created_at
                recency_boost = 0.0
                ts = g.get("created_at")
                if isinstance(ts, str):
                    try:
                        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
                        age_h = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 3600.0)
                        recency_boost = max(0.0, min(1.0, 1.0 - (age_h / 48.0)))
                    except Exception:
                        recency_boost = 0.0
                g["priority"] = base + (min(3, freq) * 0.1) + (0.2 * sev_norm) + (0.1 * recency_boost)

    # Sort by descending priority (stable tie by id)
    goals = sorted(
        [g for g in goals if isinstance(g, dict)],
        key=lambda x: (float(x.get("priority") or 0.0), str(x.get("id") or "")),
        reverse=True,
    )

    # Persist scheduled goals updating per-suite quota usage
    with queue_path.open("a", encoding="utf-8") as handle:
        for goal in goals:
            if not isinstance(goal, dict):
                continue
            gid = goal.get("id")
            rec = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": "consistency_goal",
                "id": gid,
                "suite": goal.get("suite"),
                "scenario": goal.get("scenario"),
                "priority": goal.get("priority"),
                "reason": goal.get("reason"),
            }
            handle.write(json.dumps(rec) + "\n")
            scheduled += 1
            # quota accounting
            suite = rec.get("suite")
            if isinstance(suite, str):
                suites_quota.setdefault(suite, {"used": 0, "limit": int(suite_limits.get(suite, 9999) or 0)})
                suites_quota[suite]["used"] = suites_quota[suite]["used"] + 1
            if first_id is None:
                first_id = str(gid)
            last_id = str(gid)

    brain_consistency_goals_scheduled_total.inc(amount=scheduled, outcome="scheduled")
    brain_consistency_scheduler_events_total.inc(amount=scheduled, outcome="scheduled")
    remaining = max(0, budget - scheduled)
    brain_consistency_scheduler_budget.set(remaining)
    # Write quotas snapshot
    quotas["suites"] = suites_quota
    quotas["updated_at"] = quotas.get("updated_at") or datetime.now(UTC).isoformat()
    _persist_quotas(root, quotas)

    elapsed_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000.0
    try:
        brain_consistency_scheduler_latency_ms.observe(elapsed_ms)
    except Exception:
        pass

    summary: dict[str, Any] = {
        "ok": True,
        "scheduled": scheduled,
        "queue_path": str(queue_path.resolve()),
        "first_id": first_id,
        "last_id": last_id,
        "budget_before": budget,
        "budget_remaining": remaining,
        "quotas_path": str((root / QUOTAS_PATH).resolve()),
        "suites": suites_quota,
        "timestamp": datetime.now(UTC).isoformat(),
        "latency_ms": elapsed_ms,
    }

    # Persist last run summary for proof bundle embedding
    try:
        sched_dir = root / "consistency" / "scheduler"
        sched_dir.mkdir(parents=True, exist_ok=True)
        last_run = sched_dir / "last_run.json"
        tmp = last_run.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp.replace(last_run)
    except Exception:
        # best-effort only; do not fail scheduling on persistence errors
        pass

    return summary
