
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

REASONS = {
    "low_coverage": "Predictions fail thresholds on many points.",
    "high_uncertainty": "Model uncertainty is high relative to signal.",
    "distribution_shift": "Inputs out of training domain.",
    "insufficient_support": "Too few support points for reliable fit.",
}


def make_critique(context: dict[str, Any]) -> dict[str, Any]:
    cov = float(context.get("coverage", 0.0))
    unc = float(context.get("uncertainty", 0.0))
    n = int(context.get("n_support", 0))
    reason = []
    if cov < 0.8:
        reason.append("low_coverage")
    if unc > 0.2:
        reason.append("high_uncertainty")
    if n < 6:
        reason.append("insufficient_support")
    if not reason:
        reason = ["high_uncertainty"] if unc > 0 else ["low_coverage"]
    return {
        "ts": time.time(),
        "reason": reason,
        "message": "; ".join(REASONS.get(r, r) for r in reason),
        "next_action": "Collect more points via /api/brain/active/design and refit; consider composition search.",
    }


def record_critique(eval_dir: Path, run_id: str, crit: dict[str, Any]) -> None:
    p = eval_dir / "safety" / "critiques.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    from core.schema import stamp

    p.open("a", encoding="utf-8").write(json.dumps(stamp({"ts": time.time(), "run_id": run_id, **crit})) + "\n")


def summary(eval_dir: Path) -> dict[str, Any]:
    p = eval_dir / "safety" / "critiques.jsonl"
    out = {"ok": True, "counts": {}, "last": None}
    if p.exists():
        import json

        last = None
        for ln in p.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            j = json.loads(ln)
            for r in j.get("reason", []):
                out["counts"][r] = 1 + out["counts"].get(r, 0)
            last = j
        out["last"] = last
    return out
