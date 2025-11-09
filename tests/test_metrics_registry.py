from __future__ import annotations

"""Behavioral tests for the lightweight metrics registry."""

import pytest

from brain.obs import metrics


def test_counter_labels_increment_with_kwargs():
    """Counters must record increments keyed by label tuples."""
    counter = metrics.Counter("requests_total", labels=("route",))
    counter.inc(route="brain/run")
    counter.inc(amount=2, route="brain/run")

    samples = counter.collect()
    assert samples[("brain/run",)] == pytest.approx(3.0)

    with pytest.raises(ValueError):
        counter.labels()

    cell = counter.labels("brain/run")
    cell.inc()
    assert counter.collect()[("brain/run",)] == pytest.approx(4.0)


def test_metrics_snapshot_reports_all_metric_families():
    """metrics_snapshot should emit counters, gauges, and histograms for a registry."""
    registry = metrics.MetricsRegistry()

    counter = registry.counter("selector_attempts_total", labels=("strategy",))
    counter.inc(strategy="policy")

    gauge = registry.gauge("selector_latency_ms", labels=("strategy",))
    gauge.set(42.5, strategy="policy")

    histogram = registry.histogram("selector_latency_distribution", labels=("strategy",), buckets=(1, 5, 10))
    histogram.observe(4.0, strategy="policy")
    histogram.observe(7.5, strategy="policy")

    snapshot = metrics.metrics_snapshot(registry)

    assert snapshot["counters"]["selector_attempts_total"][0]["value"] == pytest.approx(1.0)
    assert snapshot["gauges"]["selector_latency_ms"][0]["value"] == pytest.approx(42.5)

    hist_entry = snapshot["histograms"]["selector_latency_distribution"][0]
    assert hist_entry["count"] == pytest.approx(2.0)
    assert hist_entry["sum"] == pytest.approx(11.5)
    assert hist_entry["bucket_bounds"] == [1.0, 5.0, 10.0]
    assert sum(hist_entry["bucket_values"]) == pytest.approx(2.0)
    assert hist_entry["bucket_values"][1] >= 1.0
    assert hist_entry["bucket_values"][2] >= 1.0
