"""Cohort metrics helpers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CohortMetrics:
    """Aggregate metrics per cohort (e.g., seed family or universe)."""

    successes: int = 0
    total: int = 0
    latency_sum: float = 0.0
    latency_p95: float = 0.0  # aggregator stores last computed value
    _latencies: list[float] = field(default_factory=list, repr=False)

    def record(self, success: bool, latency: float) -> None:
        self.total += 1
        if success:
            self.successes += 1
        self.latency_sum += latency
        self._latencies.append(latency)
        self._latencies.sort()
        idx = max(0, int(round(0.95 * (len(self._latencies) - 1))))
        self.latency_p95 = self._latencies[idx] if self._latencies else 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def latency_mean(self) -> float:
        return self.latency_sum / self.total if self.total else 0.0


def update_cohort(metrics: dict[str, CohortMetrics], cohort: str, success: bool, latency: float) -> None:
    if cohort not in metrics:
        metrics[cohort] = CohortMetrics()
    metrics[cohort].record(success, latency)
