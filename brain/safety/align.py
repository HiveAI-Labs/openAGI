
from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

# Very conservative allow/deny check for "goals" described as free text + numeric constraints
DENY_KEYWORDS = {"harm", "weapon", "explosive", "attack", "bypass", "jailbreak", "suicide", "self-harm", "kill", "hate"}
ALLOW_KEYWORDS = {"optimize", "fit", "predict", "summarize", "classify", "control", "stabilize", "plan"}


def check_goal_text(goal: str) -> dict[str, Any]:
    s = (goal or "").lower()
    deny = [k for k in DENY_KEYWORDS if k in s]
    allow = [k for k in ALLOW_KEYWORDS if k in s]
    if deny:
        return {"ok": False, "allowed": False, "reason": "deny_keywords", "matches": deny}
    return {"ok": True, "allowed": True, "reason": "allow_keywords" if allow else "neutral"}


def consolidated_report(eval_dir: Path) -> dict[str, Any]:
    """Aggregate feasibility (/safety/check_goal), robust tests, red-team (if present), and alignment checks into a single view."""
    out: dict[str, Any] = {"ts": time.time(), "sections": {}}
    # robust
    robp = eval_dir / "robust" / "tests.jsonl"
    if robp.exists():
        try:
            lines = [json.loads(ln) for ln in robp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                out["sections"]["robust"] = {
                    "runs": len(lines),
                    "avg_failure_rate": sum(l.get("failure_rate", 0.0) for l in lines) / len(lines),
                    "avg_improvement": sum(l.get("mean_improvement", 0.0) for l in lines) / len(lines),
                }
        except Exception:
            pass
    # active loops
    loops = eval_dir / "active" / "loops.jsonl"
    if loops.exists():
        L = [ln for ln in loops.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out["sections"]["active_loops"] = {"runs": len(L)}
    # align decisions history
    alignp = eval_dir / "safety" / "align.jsonl"
    if alignp.exists():
        try:
            lines = [json.loads(ln) for ln in alignp.read_text(encoding="utf-8").splitlines() if ln.strip()]
            denies = sum(1 for l in lines if not l.get("allowed", True))
            out["sections"]["alignment"] = {"checks": len(lines), "denies": denies}
        except Exception:
            pass
    return out
