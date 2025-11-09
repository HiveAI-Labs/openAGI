"""In-memory rolling window metrics for strategy selections.

Captures recent strategy decisions to surface navigation adoption of non-LLM
routes. This is intentionally lightweight and ephemeral (no persistence) so
it cannot skew production artifacts or introduce write contention.
"""
from __future__ import annotations

from collections import Counter, deque
from typing import Dict, Any

_WINDOW = 200
_recent = deque(maxlen=_WINDOW)

def record(decision_id: str, strategy: str, task_type: str) -> None:  # pragma: no cover - trivial
    try:
        _recent.append((strategy, task_type))
    except Exception:
        pass

def summary() -> Dict[str, Any]:
    c = Counter(str(s) for (s, _) in _recent)
    n = len(_recent) or 1
    nav_non_llm = sum(1 for (s, t) in _recent if t == "navigation" and s in {"policy", "world_model"})
    return {
        "window": len(_recent),
        "llm_share": c.get("llm", 0) / n,
        "policy_share": c.get("policy", 0) / n,
        "world_model_share": c.get("world_model", 0) / n,
        "navigation_non_llm_total": nav_non_llm,
    }

__all__ = ["record", "summary"]
