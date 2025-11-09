from __future__ import annotations

import json
import math
from pathlib import Path

PATH = Path("artifacts/brain/arm_bandit.json")
ARMS = ["diff", "py_edits", "tests_first"]

def _load():
    if PATH.exists():
        try: return json.loads(PATH.read_text(encoding="utf-8"))
        except Exception: return {}
    return {}

def _save(d):
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")

def select(skill: str) -> str:
    d = _load(); s = d.setdefault(skill, {"pulls":{}, "rewards":{}, "t":1})
    t = s.get("t",1)
    best, best_ucb = None, -1e9
    for a in ARMS:
        n = s["pulls"].get(a, 0)
        r = s["rewards"].get(a, 0.0)
        mean = (r / n) if n > 0 else 0.0
        ucb = mean + (2.0 * ( (math.log(max(2,t)) / (n+1)) ** 0.5  ))
        if ucb > best_ucb:
            best_ucb, best = ucb, a
    s["t"] = t + 1; d[skill] = s; _save(d)
    return best or "diff"

def reward(skill: str, arm: str, ok: bool):
    d = _load(); s = d.setdefault(skill, {"pulls":{}, "rewards":{}, "t":1})
    s["pulls"][arm] = s["pulls"].get(arm,0) + 1
    s["rewards"][arm] = s["rewards"].get(arm,0.0) + (1.0 if ok else -1.0)
    d[skill] = s; _save(d)

def reward_ex(skill: str, arm: str, bonus: float):
    """Add a fractional reward (can be negative/positive)."""
    d = _load(); s = d.setdefault(skill, {"pulls":{}, "rewards":{}, "t":1})
    s["pulls"][arm] = s["pulls"].get(arm,0) + 0  # pulls unchanged
    s["rewards"][arm] = s["rewards"].get(arm,0.0) + float(bonus)
    d[skill] = s; _save(d)
