from __future__ import annotations

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from typing import Any

from brain.obs.metrics import brain_consistency_ledger_updates_total


def _artifacts_root() -> Path:
    env_dir = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    return Path(env_dir) if env_dir else Path("artifacts")


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if n > 0 and len(lines) > n:
            lines = lines[-n:]
        return lines
    except Exception:
        return []


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def update_consistency_ledger(
    contradictions: dict[str, Any],
    goals_summary: dict[str, Any] | None = None,
    *,
    window: int = 100,
) -> dict[str, Any]:
    """Append a snapshot to the consistency ledger and write latest summary.

    Inputs:
      contradictions: proof["contradictions"] payload from detector
      goals_summary: proof["consistency_goals"] payload (optional)
      window: number of recent entries to use for simple stats

    Artifacts written under artifacts/consistency/ledger/:
      - ledger.jsonl (append-only)
      - latest.json (summary of last entry + rolling stats)
    """
    root = _artifacts_root()
    ledger_dir = root / "consistency" / "ledger"
    ledger_dir.mkdir(parents=True, exist_ok=True)

    last_total = int(contradictions.get("total") or 0) if isinstance(contradictions, dict) else 0
    suites = contradictions.get("suites") if isinstance(contradictions, dict) else []
    token = contradictions.get("determinism_token") if isinstance(contradictions, dict) else None
    goal_count = 0
    if isinstance(goals_summary, dict):
        try:
            goal_count = int(goals_summary.get("goal_count") or 0)
        except Exception:
            goal_count = 0

    entry = {
        "timestamp": datetime.now(UTC).isoformat(),
        "contradictions_total": last_total,
        "goal_count": goal_count,
        "determinism_token": token,
        # keep suites light: only counts per suite for ledger row
        "per_suite": [
            {"suite": s.get("suite"), "contradictions": int(s.get("contradiction_count") or 0)}
            for s in (suites or [])
            if isinstance(s, dict)
        ],
    }

    ledger_path = ledger_dir / "ledger.jsonl"
    try:
        with ledger_path.open("a", encoding="utf-8") as h:
            h.write(json.dumps(entry) + "\n")
        brain_consistency_ledger_updates_total.inc(outcome="appended")
    except Exception:
        brain_consistency_ledger_updates_total.inc(outcome="error")

    # Compute rolling stats (including previous record for delta)
    lines = _tail_lines(ledger_path, max(window, 2))
    totals = []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        totals.append(int(rec.get("contradictions_total") or 0))

    prev = totals[-2] if len(totals) >= 2 else None
    last = totals[-1] if totals else last_total
    delta_abs = (last - prev) if prev is not None else 0
    delta_pct = (float(delta_abs) / float(prev)) * 100.0 if (prev is not None and prev > 0) else (0.0 if prev is not None else 0.0)

    stats = {
        "window": len(totals),
        "avg_contradictions": (sum(totals) / len(totals)) if totals else 0.0,
        "max_contradictions": max(totals) if totals else 0,
        "min_contradictions": min(totals) if totals else 0,
        "last_contradictions": last,
        "prev_contradictions": prev,
        "delta_abs": delta_abs,
        "delta_pct": delta_pct,
    }

    latest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "last_entry": entry,
        "stats": stats,
    }

    latest_path = ledger_dir / "latest.json"
    try:
        latest_path.write_text(json.dumps(latest), encoding="utf-8")
    except Exception:
        pass

    return {"latest_path": str(latest_path.resolve()), "ledger_path": str(ledger_path.resolve()), "stats": stats}
