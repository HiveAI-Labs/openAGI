
from __future__ import annotations

import time
from typing import Any

metrics: dict[str, float] = {}


class AlertsEngine:
    def __init__(self, error_rate_threshold: float = 0.01, latency_p95_threshold_ms: float = 500.0, cpu_util_threshold: float = 0.9) -> None:
        self.error_rate_threshold = error_rate_threshold
        self.latency_p95_threshold_ms = latency_p95_threshold_ms
        self.cpu_util_threshold = cpu_util_threshold
        self._last_alert: dict | None = None
    def check_health(self) -> dict[str, Any]:
        """Check the global `metrics` dict and return an alert dict."""
        return self.evaluate(metrics)

    def evaluate(self, sample: dict[str, float]) -> dict[str, Any]:
        """Evaluate a provided metrics sample (dict) and return an alert dict.

        This mirrors the old `check_health` behavior but operates on the provided
        `sample` so callers can pass arbitrary measurement dicts.
        """
        err = float(sample.get("error_rate", 0.0))
        p95 = float(sample.get("latency_p95", 0.0))
        cpu = float(sample.get("cpu_util", 0.0))
        ok = (
            (err < self.error_rate_threshold)
            and (p95 < self.latency_p95_threshold_ms)
            and (cpu < self.cpu_util_threshold)
        )
        alert: dict[str, Any] = {"ok": ok, "error_rate": err, "latency_p95": p95, "cpu_util": cpu}
        if not ok:
            alert["ts"] = time.time()
            self._last_alert = alert
        return alert
    def last_alert(self) -> dict | None:
        return self._last_alert
