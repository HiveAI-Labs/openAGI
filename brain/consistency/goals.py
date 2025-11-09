from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class ConsistencyGoal:
    """A remediation goal generated from detected contradictions.

    id: stable identifier composed of suite+scenario.
    suite: name of the sandbox suite where contradiction was observed.
    scenario: scenario identifier (may include seed suffix from detector).
    priority: scheduling priority (higher = sooner).
    reason: short justification string.
    created_at: ISO timestamp in UTC when generated.
    risk_score: optional, for downstream safety-aware scheduling.
    """

    id: str
    suite: str
    scenario: str
    priority: float
    reason: str
    created_at: str
    risk_score: float | None = None
    severity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_goals_from_contradictions(
    contradictions_summary: dict[str, Any],
    *,
    base_priority: float = 1.0,
) -> list[ConsistencyGoal]:
    """Produce a list of consistency remediation goals from a contradictions summary.

    The expected input shape is what tools/ci/proof_bundle._collect_contradictions() emits
    under proof["contradictions"]. We generate one goal per (suite, scenario) contradiction.

    Priority is computed from a simple heuristic now (constant base). This can be
    extended to incorporate frequency, recentness, and failure ratios.
    """
    if not isinstance(contradictions_summary, dict):
        return []

    suites = contradictions_summary.get("suites")
    if not isinstance(suites, list):
        return []

    now = datetime.now(UTC).isoformat()
    goals: list[ConsistencyGoal] = []
    for suite_info in suites:
        if not isinstance(suite_info, dict):
            continue
        suite = suite_info.get("suite")
        if not isinstance(suite, str) or not suite:
            continue
        # Use per-suite contradiction_count as a coarse severity proxy
        try:
            suite_severity = float(suite_info.get("contradiction_count") or 0.0)
        except Exception:
            suite_severity = 0.0
        scenarios = suite_info.get("contradictions") or []
        for scenario in scenarios:
            if not isinstance(scenario, str) or not scenario:
                continue
            gid = f"{suite}:{scenario}"
            goal = ConsistencyGoal(
                id=gid,
                suite=suite,
                scenario=scenario,
                priority=float(base_priority),
                reason="scenario observed both success and failure in recent audit window",
                created_at=now,
                risk_score=suite_severity,
                severity=suite_severity,
            )
            goals.append(goal)
    return goals
