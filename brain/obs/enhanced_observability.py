# type: ignore
# Enhanced Observability System for HiveNet AI Brain
# Provides comprehensive metrics, alerting, and monitoring capabilities

import base64
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import hmac
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

from brain.io.atomic import atomic_write_json_lines
from brain.meta.consult_log import (
    consult_log_path as _consult_log_path,
    load_consult_events as _load_consult_events,
    summarize_consults as _summarize_consults,
)
from brain.obs.metrics import (
    brain_curriculum_ack_chaos_latency_ms,
    brain_curriculum_ack_mttr_seconds,
    brain_curriculum_ack_signature_checked,
    brain_curriculum_ack_signature_valid_ratio,
    brain_curriculum_ack_totals,
    metrics_snapshot,
    record_learning_budget_state,
)
from brain.universe.curriculum_sandbox import load_curriculum_summary
from tools.ci.curriculum_mttr import build_mttr_gate_payload, resolve_mttr_config

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create minimal in-memory metrics for when prometheus is not available
    import logging as _logging
    _LOG = _logging.getLogger(__name__)
    _warned_prom = False

    class _BaseMetric:
        def __init__(self, *args, **kwargs):
            self._values = {}
        def labels(self, **kwargs):
            key = tuple(sorted((kwargs or {}).items()))
            self._current = key
            if key not in self._values:
                self._values[key] = 0.0
            return self

    class Counter(_BaseMetric):
        def inc(self, amount=1):
            if not getattr(self, "_warned", False):
                _LOG.warning("Prometheus unavailable; using in-memory Counter (no export)")
                self._warned = True
            key = getattr(self, "_current", ())
            self._values[key] = float(self._values.get(key, 0.0)) + float(amount)

    class Gauge(_BaseMetric):
        def set(self, value):
            if not getattr(self, "_warned", False):
                _LOG.warning("Prometheus unavailable; using in-memory Gauge (no export)")
                self._warned = True
            key = getattr(self, "_current", ())
            self._values[key] = float(value)

    class Histogram(_BaseMetric):
        def observe(self, value):
            if not getattr(self, "_warned", False):
                _LOG.warning("Prometheus unavailable; using in-memory Histogram (no export)")
                self._warned = True
            key = getattr(self, "_current", ())
            self._values[key] = float(self._values.get(key, 0.0)) + 1.0

    class Summary(_BaseMetric):
        pass

    CollectorRegistry = type("CollectorRegistry", (), {})()
    generate_latest = lambda registry=None: b"# Prometheus client not available\n"
    CONTENT_TYPE_LATEST = "text/plain; charset=utf-8"

# Health check status
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class HealthCheck:
    """Represents a health check result."""

    name: str
    status: HealthStatus
    message: str
    timestamp: float
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "status": self.status.value,
            "timestamp": self.timestamp,
        }

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """Represents an alert condition."""

    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    labels: dict[str, str] | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "severity": self.severity.value,
            "timestamp": self.timestamp,
            "labels": self.labels or {},
        }

class EnhancedMetricsRegistry:
    """Enhanced metrics registry with Prometheus integration."""

    def __init__(self, registry: CollectorRegistry | None = None):
        if PROMETHEUS_AVAILABLE:
            self.registry = registry or CollectorRegistry()
        else:
            self.registry = None
        self._metrics = {}
        self._alerts = []
        self._health_checks = {}
        self._lock = threading.Lock()

        if PROMETHEUS_AVAILABLE:
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Set up Prometheus metrics."""
        # HTTP request metrics
        self.http_requests_total = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )

        self.http_request_duration_seconds = Histogram(
            "http_request_duration_seconds",
            "HTTP request duration in seconds",
            ["method", "endpoint"],
            registry=self.registry,
        )

        # Brain operation metrics
        self.brain_operations_total = Counter(
            "brain_operations_total",
            "Total brain operations",
            ["operation", "status"],
            registry=self.registry,
        )

        self.brain_operation_duration_seconds = Histogram(
            "brain_operation_duration_seconds",
            "Brain operation duration in seconds",
            ["operation"],
            registry=self.registry,
        )

        # Memory metrics
        self.memory_usage_bytes = Gauge(
            "memory_usage_bytes",
            "Current memory usage in bytes",
            ["type"],
            registry=self.registry,
        )

        # Tool execution metrics
        self.tool_executions_total = Counter(
            "tool_executions_total",
            "Total tool executions",
            ["tool", "status"],
            registry=self.registry,
        )

        # Error metrics
        self.errors_total = Counter(
            "errors_total",
            "Total errors",
            ["type", "component"],
            registry=self.registry,
        )

        # Performance metrics
        self.cpu_usage_percent = Gauge(
            "cpu_usage_percent",
            "Current CPU usage percentage",
            registry=self.registry,
        )

        self.active_connections = Gauge(
            "active_connections",
            "Number of active connections",
            registry=self.registry,
        )

    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics."""
        if PROMETHEUS_AVAILABLE:
            self.http_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
            self.http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(duration)

    def record_brain_operation(self, operation: str, status: str, duration: float):
        """Record brain operation metrics."""
        if PROMETHEUS_AVAILABLE:
            self.brain_operations_total.labels(operation=operation, status=status).inc()
            self.brain_operation_duration_seconds.labels(operation=operation).observe(duration)

    def record_memory_usage(self, heap_used: int, heap_total: int):
        """Record memory usage metrics."""
        if PROMETHEUS_AVAILABLE:
            self.memory_usage_bytes.labels(type="heap_used").set(heap_used)
            self.memory_usage_bytes.labels(type="heap_total").set(heap_total)

    def record_tool_execution(self, tool: str, status: str):
        """Record tool execution metrics."""
        if PROMETHEUS_AVAILABLE:
            self.tool_executions_total.labels(tool=tool, status=status).inc()

    def record_error(self, error_type: str, component: str):
        """Record error metrics."""
        if PROMETHEUS_AVAILABLE:
            self.errors_total.labels(type=error_type, component=component).inc()

    def record_cpu_usage(self, usage_percent: float):
        """Record CPU usage metrics."""
        if PROMETHEUS_AVAILABLE:
            self.cpu_usage_percent.set(usage_percent)

    def record_active_connections(self, count: int):
        """Record active connections."""
        if PROMETHEUS_AVAILABLE:
            self.active_connections.set(count)

    def add_health_check(self, check: HealthCheck):
        """Add a health check result."""
        with self._lock:
            self._health_checks[check.name] = check

    def get_health_status(self) -> dict[str, Any]:
        """Get overall health status."""
        with self._lock:
            checks = list(self._health_checks.values())

        healthy = sum(1 for c in checks if c.status == HealthStatus.HEALTHY)
        degraded = sum(1 for c in checks if c.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for c in checks if c.status == HealthStatus.UNHEALTHY)

        overall_status = HealthStatus.HEALTHY
        if unhealthy > 0:
            overall_status = HealthStatus.UNHEALTHY
        elif degraded > 0:
            overall_status = HealthStatus.DEGRADED

        return {
            "status": overall_status.value,
            "timestamp": time.time(),
            "checks": {c.name: c.to_dict() for c in checks},
            "summary": {
                "total": len(checks),
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
            },
        }

    def add_alert(self, alert: Alert):
        """Add an alert."""
        with self._lock:
            self._alerts.append(alert)
            # Keep only last 1000 alerts
            if len(self._alerts) > 1000:
                self._alerts = self._alerts[-1000:]
        try:
            _persist_alert_runbook_entry(alert.to_dict())
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "alert_runbook_persist_failed",
                extra={
                    "structured_data": {
                        "alert_name": alert.name,
                        "severity": alert.severity.value,
                        "error": str(exc),
                    },
                },
            )

    def get_alerts(self, severity: AlertSeverity | None = None,
                   limit: int = 100) -> list[dict[str, Any]]:
        """Get alerts, optionally filtered by severity."""
        with self._lock:
            alerts = self._alerts

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return [a.to_dict() for a in alerts[-limit:]]

    def generate_prometheus_output(self) -> bytes:
        """Generate Prometheus metrics output."""
        if PROMETHEUS_AVAILABLE:
            return generate_latest(self.registry)
        return b"# Prometheus client not available\n"

class AlertManager:
    """Manages alerts and alerting rules."""

    def __init__(self, metrics_registry: EnhancedMetricsRegistry):
        self.metrics = metrics_registry
        self._rules = []
        self._lock = threading.Lock()

    def add_rule(self, name: str, condition: Callable[[], bool],
                 severity: AlertSeverity, message: str,
                 labels: dict[str, str] | None = None):
        """Add an alerting rule."""
        with self._lock:
            self._rules.append({
                "name": name,
                "condition": condition,
                "severity": severity,
                "message": message,
                "labels": labels or {},
                "last_triggered": 0,
            })

    def evaluate_rules(self):
        """Evaluate all alerting rules."""
        with self._lock:
            for rule in self._rules:
                try:
                    if rule["condition"]():
                        # Only alert if not triggered in last 5 minutes
                        if time.time() - rule["last_triggered"] > 300:
                            alert = Alert(
                                name=rule["name"],
                                severity=rule["severity"],
                                message=rule["message"],
                                timestamp=time.time(),
                                labels=rule["labels"],
                            )
                            self.metrics.add_alert(alert)
                            rule["last_triggered"] = time.time()
                except Exception as e:
                    logging.exception(f"Error evaluating alert rule {rule['name']}: {e}")

class HealthChecker:
    """Performs health checks on system components."""

    def __init__(self, metrics_registry: EnhancedMetricsRegistry):
        self.metrics = metrics_registry
        self._checks = []
        self._lock = threading.Lock()

    def add_check(self, name: str, check_func: Callable[[], HealthCheck]):
        """Add a health check function."""
        with self._lock:
            self._checks.append({
                "name": name,
                "func": check_func,
            })

    def run_checks(self):
        """Run all health checks.

        Note: copy the checks list while holding the lock, but do not hold the
        lock while executing each check function. Some checks perform I/O or
        network requests and can block; holding the lock across execution can
        cause contention and deadlocks when run concurrently (e.g., background
        monitoring worker and an incoming HTTP request both calling
        run_checks()).
        """
        # Copy checks under lock to avoid holding the lock during slow I/O
        with self._lock:
            checks = list(self._checks)

        for check in checks:
            try:
                result = check["func"]()
                self.metrics.add_health_check(result)
            except Exception as e:
                error_check = HealthCheck(
                    name=check["name"],
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    timestamp=time.time(),
                )
                self.metrics.add_health_check(error_check)

class StructuredLogger:
    """Enhanced structured logging with context and correlation."""

    def __init__(self, base_logger: logging.Logger | None = None):
        self.logger = base_logger or logging.getLogger(__name__)
        self._context = threading.local()

    def set_context(self, **kwargs):
        """Set logging context for current thread."""
        if not hasattr(self._context, "data"):
            self._context.data = {}
        self._context.data.update(kwargs)

    def clear_context(self):
        """Clear logging context for current thread."""
        if hasattr(self._context, "data"):
            self._context.data = {}

    def get_context(self) -> dict[str, Any]:
        """Get current logging context."""
        if hasattr(self._context, "data"):
            return self._context.data.copy()
        return {}

    def log(self, level: int, message: str, **kwargs):
        """Log a structured message."""
        context = self.get_context()
        context.update(kwargs)

        # Add timestamp if not provided
        if "timestamp" not in context:
            context["timestamp"] = time.time()

        # Create structured log record
        extra = {
            "structured_data": context,
            "message": message,
        }

        self.logger.log(level, message, extra=extra)

    def info(self, message: str, **kwargs):
        """Log info level message."""
        self.log(logging.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning level message."""
        self.log(logging.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error level message."""
        self.log(logging.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical level message."""
        self.log(logging.CRITICAL, message, **kwargs)


# Curriculum observability thresholds tuned for early warning rather than final failure.
_PROGRESSION_WARNING = -0.05
_PROGRESSION_CRITICAL = -0.15
_MOMENTUM_WARNING = -0.05
_MOMENTUM_CRITICAL = -0.12
_FAILURE_RATIO_WARNING = 0.4
_FAILURE_RATIO_CRITICAL = 0.6
_FAILURE_STREAK_WARNING = 3
_FAILURE_STREAK_CRITICAL = 5
_STRATEGY_WEIGHT_WARNING = 0.6
_STRATEGY_WEIGHT_CRITICAL = 0.4
_STRATEGY_BLOCKERS_WARNING = 1
_STRATEGY_BLOCKERS_CRITICAL = 2

_CURRICULUM_METRICS = {
    "brain_curriculum_suite_success_latest",
    "brain_curriculum_suite_success_average",
    "brain_curriculum_suite_progression_delta",
    "brain_curriculum_suite_momentum",
    "brain_curriculum_suite_failure_ratio",
    "brain_curriculum_suite_streak",
}

_STRATEGY_METRICS = {
    "brain_strategy_curriculum_gate_weight",
    "brain_strategy_curriculum_gate_ok",
    "brain_strategy_curriculum_blockers_total",
    "brain_strategy_curriculum_progression",
    "brain_strategy_curriculum_momentum",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Coerce values recorded in metrics into bounded floats."""
    try:
        if value is None:
            return default
        casted = float(value)
        if math.isnan(casted) or math.isinf(casted):
            return default
        return casted
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    """Interpret gauge samples that should represent integral streak lengths."""
    try:
        if value is None:
            return default
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _suite_status(entry: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Compute qualitative status for a curriculum suite entry."""
    alerts: list[dict[str, Any]] = []
    progression = entry["progression"]["delta"]
    momentum = entry["progression"]["momentum"]
    failure_ratio = entry["failures"]["ratio"]
    failure_streak = entry["failures"]["failure_streak"]

    level = "ok"

    if progression <= _PROGRESSION_WARNING:
        severity = "critical" if progression <= _PROGRESSION_CRITICAL else "warning"
        alerts.append({
            "reason": "progression_backslide",
            "level": severity,
            "value": progression,
            "threshold": _PROGRESSION_WARNING if severity == "warning" else _PROGRESSION_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if momentum <= _MOMENTUM_WARNING:
        severity = "critical" if momentum <= _MOMENTUM_CRITICAL else "warning"
        alerts.append({
            "reason": "negative_momentum",
            "level": severity,
            "value": momentum,
            "threshold": _MOMENTUM_WARNING if severity == "warning" else _MOMENTUM_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if failure_ratio >= _FAILURE_RATIO_WARNING:
        severity = "critical" if failure_ratio >= _FAILURE_RATIO_CRITICAL else "warning"
        alerts.append({
            "reason": "failure_ratio",
            "level": severity,
            "value": failure_ratio,
            "threshold": _FAILURE_RATIO_WARNING if severity == "warning" else _FAILURE_RATIO_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if failure_streak >= _FAILURE_STREAK_WARNING:
        severity = "critical" if failure_streak >= _FAILURE_STREAK_CRITICAL else "warning"
        alerts.append({
            "reason": "failure_streak",
            "level": severity,
            "value": failure_streak,
            "threshold": _FAILURE_STREAK_WARNING if severity == "warning" else _FAILURE_STREAK_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    return level, alerts


def _strategy_status(entry: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Compute status flags for aggregated strategy gates."""
    alerts: list[dict[str, Any]] = []
    weight = entry["gate"]["weight"]
    ok_value = entry["gate"]["ok"]
    blockers = entry["gate"]["blockers"]
    progression = entry["curriculum"]["progression"]
    momentum = entry["curriculum"]["momentum"]

    level = "ok"

    if ok_value < 0.5:
        alerts.append({
            "reason": "gate_blocked",
            "level": "critical",
            "value": ok_value,
            "threshold": 0.5,
        })
        level = "critical"

    if weight <= _STRATEGY_WEIGHT_WARNING:
        severity = "critical" if weight <= _STRATEGY_WEIGHT_CRITICAL else "warning"
        alerts.append({
            "reason": "weight_degraded",
            "level": severity,
            "value": weight,
            "threshold": _STRATEGY_WEIGHT_WARNING if severity == "warning" else _STRATEGY_WEIGHT_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if blockers >= _STRATEGY_BLOCKERS_WARNING:
        severity = "critical" if blockers >= _STRATEGY_BLOCKERS_CRITICAL else "warning"
        alerts.append({
            "reason": "active_blockers",
            "level": severity,
            "value": blockers,
            "threshold": _STRATEGY_BLOCKERS_WARNING if severity == "warning" else _STRATEGY_BLOCKERS_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if progression <= _PROGRESSION_WARNING:
        severity = "critical" if progression <= _PROGRESSION_CRITICAL else "warning"
        alerts.append({
            "reason": "strategy_progression_backslide",
            "level": severity,
            "value": progression,
            "threshold": _PROGRESSION_WARNING if severity == "warning" else _PROGRESSION_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    if momentum <= _MOMENTUM_WARNING:
        severity = "critical" if momentum <= _MOMENTUM_CRITICAL else "warning"
        alerts.append({
            "reason": "strategy_negative_momentum",
            "level": severity,
            "value": momentum,
            "threshold": _MOMENTUM_WARNING if severity == "warning" else _MOMENTUM_CRITICAL,
        })
        if severity == "critical":
            level = "critical"
        elif level == "ok":
            level = "warning"

    return level, alerts


def _extract_suite_rows(gauges: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def ensure_row(suite: str, strategy: str) -> dict[str, Any]:
        key = (suite, strategy)
        if key not in rows:
            rows[key] = {
                "suite": suite,
                "strategy": strategy,
                "success": {"latest": 0.0, "average": 0.0},
                "progression": {"delta": 0.0, "momentum": 0.0},
                "failures": {"ratio": 0.0, "success_streak": 0, "failure_streak": 0},
                "alerts": [],
                "status": "ok",
            }
        return rows[key]

    for sample in gauges.get("brain_curriculum_suite_success_latest", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        if not suite or not strategy:
            continue
        ensure_row(suite, strategy)["success"]["latest"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_curriculum_suite_success_average", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        if not suite or not strategy:
            continue
        ensure_row(suite, strategy)["success"]["average"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_curriculum_suite_progression_delta", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        if not suite or not strategy:
            continue
        ensure_row(suite, strategy)["progression"]["delta"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_curriculum_suite_momentum", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        if not suite or not strategy:
            continue
        ensure_row(suite, strategy)["progression"]["momentum"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_curriculum_suite_failure_ratio", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        if not suite or not strategy:
            continue
        ensure_row(suite, strategy)["failures"]["ratio"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_curriculum_suite_streak", []):
        labels = sample.get("labels", {})
        suite = labels.get("suite")
        strategy = labels.get("strategy")
        streak_type = labels.get("type")
        if not suite or not strategy or not streak_type:
            continue
        row = ensure_row(suite, strategy)
        if streak_type == "success":
            row["failures"]["success_streak"] = _safe_int(sample.get("value"))
        elif streak_type == "failure":
            row["failures"]["failure_streak"] = _safe_int(sample.get("value"))

    sorted_rows = []
    for row in rows.values():
        status, alerts = _suite_status(row)
        row["status"] = status
        row["alerts"] = alerts
        sorted_rows.append(row)
    return sorted(sorted_rows, key=lambda r: (r["strategy"], r["suite"]))


def _extract_strategy_rows(gauges: Mapping[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}

    def ensure_row(strategy: str) -> dict[str, Any]:
        if strategy not in rows:
            rows[strategy] = {
                "strategy": strategy,
                "gate": {"weight": 1.0, "ok": 1.0, "blockers": 0.0},
                "curriculum": {"progression": 0.0, "momentum": 0.0},
                "alerts": [],
                "status": "ok",
            }
        return rows[strategy]

    for sample in gauges.get("brain_strategy_curriculum_gate_weight", []):
        labels = sample.get("labels", {})
        strategy = labels.get("strategy")
        if not strategy:
            continue
        ensure_row(strategy)["gate"]["weight"] = _safe_float(sample.get("value"), 1.0)

    for sample in gauges.get("brain_strategy_curriculum_gate_ok", []):
        labels = sample.get("labels", {})
        strategy = labels.get("strategy")
        if not strategy:
            continue
        ensure_row(strategy)["gate"]["ok"] = _safe_float(sample.get("value"), 1.0)

    for sample in gauges.get("brain_strategy_curriculum_blockers_total", []):
        labels = sample.get("labels", {})
        strategy = labels.get("strategy")
        if not strategy:
            continue
        ensure_row(strategy)["gate"]["blockers"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_strategy_curriculum_progression", []):
        labels = sample.get("labels", {})
        strategy = labels.get("strategy")
        if not strategy:
            continue
        ensure_row(strategy)["curriculum"]["progression"] = _safe_float(sample.get("value"))

    for sample in gauges.get("brain_strategy_curriculum_momentum", []):
        labels = sample.get("labels", {})
        strategy = labels.get("strategy")
        if not strategy:
            continue
        ensure_row(strategy)["curriculum"]["momentum"] = _safe_float(sample.get("value"))

    sorted_rows = []
    for row in rows.values():
        status, alerts = _strategy_status(row)
        row["status"] = status
        row["alerts"] = alerts
        sorted_rows.append(row)
    return sorted(sorted_rows, key=lambda r: r["strategy"])


def curriculum_dashboard_snapshot(metrics_data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build a structured view of curriculum telemetry for dashboards and alerts."""
    snapshot = metrics_data or metrics_snapshot()
    gauges: Mapping[str, list[dict[str, Any]]] = snapshot.get("gauges", {})  # type: ignore[assignment]

    sign_key = resolve_alert_ack_signing_key()
    records = load_curriculum_alert_records() or []
    chaos_metrics = _load_curriculum_ack_chaos_metrics()
    ack_summary = _curriculum_ack_summary(records, sign_key, chaos_metrics)
    incident_rollup_meta = _load_latest_incident_rollup()

    # Quick exit when curriculum telemetry has not been emitted yet.
    if not any(name in gauges for name in _CURRICULUM_METRICS.union(_STRATEGY_METRICS)):
        payload = {
            "generated_at": time.time(),
            "suites": [],
            "strategies": [],
            "alerts": [],
            "totals": {"suites": 0, "strategies": 0, "alerts": 0},
        }
        payload["acknowledgements"] = ack_summary
        payload["incident_rollup"] = incident_rollup_meta
        return payload

    suites = _extract_suite_rows(gauges)
    strategies = _extract_strategy_rows(gauges)

    alerts: list[dict[str, Any]] = []
    for row in suites:
        if row["status"] != "ok":
            alerts.append({
                "kind": "curriculum_suite",
                "suite": row["suite"],
                "strategy": row["strategy"],
                "level": row["status"],
                "reasons": row["alerts"],
            })

    for row in strategies:
        if row["status"] != "ok":
            alerts.append({
                "kind": "curriculum_strategy",
                "strategy": row["strategy"],
                "level": row["status"],
                "reasons": row["alerts"],
            })

    summary = _load_curriculum_summary_for_dashboard()
    suite_stats: dict[str, Any] = {}
    if summary:
        suite_stats = {stats.suite: stats for stats in summary.suites}

    for suite_row in suites:
        stats = suite_stats.get(suite_row["suite"])
        if not stats:
            continue
        suite_row["age_hours"] = stats.age_hours
        suite_row["stale"] = stats.stale
        suite_row["determinism_token"] = stats.determinism_token

    stale_suites: list[str] = []
    max_age_hours: float | None = None
    for suite_row in suites:
        age = suite_row.get("age_hours")
        if isinstance(age, (int, float)):
            max_age_hours = age if max_age_hours is None else max(max_age_hours, float(age))
        if suite_row.get("stale"):
            stale_suites.append(str(suite_row.get("suite")))

    payload: dict[str, Any] = {
        "generated_at": time.time(),
        "suites": suites,
        "strategies": strategies,
        "alerts": alerts,
        "totals": {
            "suites": len(suites),
            "strategies": len(strategies),
            "alerts": len(alerts),
        },
    }
    payload["acknowledgements"] = ack_summary
    payload["incident_rollup"] = incident_rollup_meta
    if summary:
        payload["sandbox_determinism_token"] = summary.determinism_token
        payload["sandbox_stale_window_hours"] = summary.stale_window_hours
    payload["stale_summary"] = {
        "count": len(stale_suites),
        "suites": stale_suites[:10],
        "max_age_hours": max_age_hours,
    }
    incident_analysis = _incident_analysis_summary()
    if incident_analysis:
        payload["incident_analysis"] = incident_analysis
    return payload


def selector_consult_snapshot(
    limit: int = 1000,
    *,
    log_path: Path | None = None,
) -> dict[str, Any]:
    """Summarize recent selector consult sessions for dashboards and proofs."""
    path = log_path or _consult_log_path()
    events = _load_consult_events(path=path, limit=limit)
    summary = _summarize_consults(events)
    summary.update(
        {
            "generated_at": time.time(),
            "log_path": str(path),
            "limit": limit,
            "available": path.exists(),
        },
    )
    return summary


def learning_budget_snapshot() -> dict[str, Any]:
    """Summarize adaptive learning budget files for dashboards and metrics JSON."""
    base = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
    budget_path = Path(base) / "training" / "learning_budget.json"
    state_path = Path(base) / "training" / "learning_budget_state.json"
    errors: dict[str, str] = {}

    def _load_json(path: Path, key: str) -> dict[str, Any] | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - defensive logging only
            errors[key] = f"load_failed:{exc}"
            return None

    budget = _load_json(budget_path, "budget")
    state = _load_json(state_path, "state")

    if isinstance(budget, Mapping):
        record_learning_budget_state(budget)
    else:
        record_learning_budget_state(None)

    summary: dict[str, Any] = {
        "score": None,
        "throttle_seconds": None,
        "max_runs_per_hour": None,
        "latency_multiplier": None,
        "cost_multiplier": None,
        "halt_training": None,
        "failing_gates": [],
        "notes": [],
        "updated_at": None,
        "recent_runs": [],
    }

    if isinstance(budget, Mapping):
        summary["score"] = budget.get("score")
        summary["throttle_seconds"] = budget.get("throttle_seconds")
        summary["max_runs_per_hour"] = budget.get("max_runs_per_hour")
        summary["latency_multiplier"] = budget.get("latency_multiplier")
        summary["cost_multiplier"] = budget.get("cost_multiplier")
        summary["halt_training"] = bool(budget.get("halt_training"))
        summary["failing_gates"] = list(budget.get("failing_gates") or [])
        summary["notes"] = list(budget.get("notes") or [])
        summary["updated_at"] = budget.get("updated_at")

    if isinstance(state, Mapping):
        summary["recent_runs"] = list(state.get("recent_runs") or [])

    payload: dict[str, Any] = {
        "generated_at": time.time(),
        "paths": {"budget": str(budget_path), "state": str(state_path)},
        "summary": summary,
        "budget": budget,
        "state": state,
    }
    if errors:
        payload["errors"] = errors
    return payload


def _load_curriculum_summary_for_dashboard():
    base = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
    try:
        return load_curriculum_summary(base)
    except Exception:
        return None


def load_latest_incident_analysis_snapshot(base_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Return the latest curriculum incident analysis snapshot payload and path."""
    if base_dir is None:
        base_dir = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
    base_path = Path(base_dir)
    directory = base_path / "ops" / "curriculum_incident_analysis"
    if not directory.exists():
        return None
    candidates = [
        path
        for path in directory.glob("analysis_*.json")
        if path.is_file()
    ]
    if not candidates:
        return None
    latest = max(candidates, key=lambda entry: entry.stat().st_mtime)
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return None
    return {
        "path": str(latest),
        "payload": payload,
    }


def _incident_analysis_summary(base_dir: str | Path | None = None) -> dict[str, Any] | None:
    snapshot = load_latest_incident_analysis_snapshot(base_dir)
    if snapshot is None:
        return None
    payload = snapshot.get("payload") or {}
    suite_reports = []
    for report in (payload.get("suite_reports") or [])[:5]:
        metrics = report.get("metrics") or {}
        suite_reports.append({
            "suite": report.get("suite"),
            "issues": (report.get("issues") or [])[:3],
            "success_rate_latest": metrics.get("success_rate_latest"),
            "success_progression_delta": metrics.get("success_progression_delta"),
            "age_hours": metrics.get("age_hours"),
        })
    return {
        "path": snapshot.get("path"),
        "generated_at": payload.get("generated_at"),
        "window_hours": payload.get("analysis_window_hours"),
        "alerts": payload.get("alerts"),
        "suite_reports": suite_reports,
    }


def curriculum_ack_summary(
    *,
    records: list[dict[str, Any]] | None = None,
    sign_key: str | None = None,
    include_chaos_metrics: bool = True,
) -> dict[str, Any]:
    """Public helper that returns the curriculum acknowledgement summary.

    Parameters
    ----------
    records:
        Optional preloaded alert records. When ``None`` the records are read
        from the runbook ledger under ``artifacts/ops/alerts.jsonl``.
    sign_key:
        Optional signing key override. When omitted the helper uses
        :func:`resolve_alert_ack_signing_key`.
    include_chaos_metrics:
        When ``True`` (default) include the persisted chaos drill metrics if
        available.

    """
    resolved_records = records if records is not None else (load_curriculum_alert_records() or [])
    resolved_key = sign_key if sign_key is not None else resolve_alert_ack_signing_key()
    chaos = _load_curriculum_ack_chaos_metrics() if include_chaos_metrics else None
    return _curriculum_ack_summary(resolved_records, resolved_key, chaos)

# Global instances
metrics_registry = EnhancedMetricsRegistry()
alert_manager = AlertManager(metrics_registry)
health_checker = HealthChecker(metrics_registry)
structured_logger = StructuredLogger()

# Convenience functions
def record_http_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    metrics_registry.record_http_request(method, endpoint, status, duration)

def record_brain_operation(operation: str, status: str, duration: float):
    """Record brain operation metrics."""
    metrics_registry.record_brain_operation(operation, status, duration)

def get_health_status() -> dict[str, Any]:
    """Get overall health status."""
    return metrics_registry.get_health_status()

def get_alerts(severity: AlertSeverity | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Get alerts."""
    return metrics_registry.get_alerts(severity, limit)

def generate_metrics_output() -> bytes:
    """Generate Prometheus metrics output."""
    return metrics_registry.generate_prometheus_output()


def _runbook_alerts_path() -> Path:
    base = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    artifacts_root = Path(base) if base else Path("artifacts")
    return artifacts_root / "ops" / "alerts.jsonl"


def _runbook_ack_chaos_path() -> Path:
    base = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    artifacts_root = Path(base) if base else Path("artifacts")
    return artifacts_root / "ops" / "curriculum_ack_chaos.json"


def _mttr_history_path() -> Path:
    base = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    artifacts_root = Path(base) if base else Path("artifacts")
    return artifacts_root / "ci" / "curriculum_ack_mttr.json"


def _incident_rollups_dir(artifacts_dir: Path | None = None) -> Path:
    base_path = artifacts_dir
    if base_path is None:
        base_env = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
        base_path = Path(base_env) if base_env else Path("artifacts")
    return base_path / "proof" / "incident_rollups"


def load_latest_incident_rollup_record(artifacts_dir: Path | None = None) -> dict[str, Any] | None:
    """Return the latest incident rollup payload and its path.

    Parameters
    ----------
    artifacts_dir:
        Optional override for the artifacts directory. When omitted the
        function resolves the directory from ``ARTIFACTS_DIR`` or
        ``BRAIN_ARTIFACTS_DIR``.

    Returns
    -------
    dict | None
        ``None`` when no rollup exists. Otherwise a dictionary containing the
        keys ``path`` (string path to the rollup JSON) and ``payload`` with the
        parsed JSON object. If parsing fails the dictionary contains only the
        ``path`` key.

    """
    directory = _incident_rollups_dir(artifacts_dir)
    if not directory.exists():
        return None
    candidates = sorted(directory.glob("incident_rollup_*.json"))
    if not candidates:
        return None
    latest = max(candidates, key=lambda path: path.stat().st_mtime)
    record: dict[str, Any] = {"path": str(latest)}
    try:
        record["payload"] = json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return record
    return record


def _load_latest_incident_rollup() -> dict[str, Any] | None:
    record = load_latest_incident_rollup_record()
    if record is None:
        return None

    metadata: dict[str, Any] = {
        "path": record.get("path"),
    }
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return metadata

    metadata["generated_at"] = payload.get("generated_at")
    proof_info = payload.get("proof_bundle")
    if isinstance(proof_info, dict):
        metadata["proof_bundle"] = proof_info
    ticket_payload = payload.get("ticket_export")
    if isinstance(ticket_payload, dict):
        metadata["ticket_export"] = ticket_payload
    ack = payload.get("acknowledgements")
    if isinstance(ack, dict):
        metadata["acknowledgements"] = {
            "pending": ack.get("pending"),
            "acknowledged": ack.get("acknowledged"),
            "blocking": ack.get("blocking"),
            "mttr_gate": ack.get("mttr_gate"),
            "signature_valid_ratio": ack.get("signature_valid_ratio"),
        }
    notes = payload.get("notes")
    if isinstance(notes, str) and notes:
        metadata["notes_preview"] = notes[:512]
    template_notes = payload.get("template_notes")
    if isinstance(template_notes, str) and template_notes:
        metadata["template_notes_preview"] = template_notes[:512]
    remediation_template = payload.get("remediation_template")
    if isinstance(remediation_template, dict):
        template_title = remediation_template.get("title") or remediation_template.get("name")
        if template_title:
            metadata["remediation_template"] = template_title
    drift = payload.get("drift_tolerances")
    if isinstance(drift, dict):
        metadata["drift_tolerances"] = drift
    return metadata


def _load_curriculum_ack_chaos_metrics(path: Path | None = None) -> dict[str, Any] | None:
    target = path or _runbook_ack_chaos_path()
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _resolve_alert_sign_key(preferred_env: str | None = None) -> str | None:
    candidates: list[str] = []
    if preferred_env:
        candidates.append(preferred_env)
    candidates.extend(
        [
            "OPS_ALERT_SIGN_KEY",
            "PROOF_CURRICULUM_ALERT_SIGN_KEY",
            "PROOF_SIGN_KEY",
            "BRAIN_PROOF_SIGN_KEY",
        ],
    )
    for env_name in candidates:
        if not env_name:
            continue
        value = os.getenv(env_name)
        if value:
            return value
    return None


def resolve_alert_ack_signing_key(preferred_env: str | None = None) -> str | None:
    """Resolve the signing key used for curriculum acknowledgements."""
    return _resolve_alert_sign_key(preferred_env)


def _alert_requires_runbook(alert: dict[str, Any]) -> bool:
    severity = str(alert.get("severity") or "info").lower()
    labels = alert.get("labels") or {}
    component = str(labels.get("component") or "").lower()
    return component == "curriculum" and severity in {"warning", "critical"}


def _alert_ack_token(alert: dict[str, Any]) -> str:
    material = "|".join(
        str(alert.get(key) or "")
        for key in ("name", "timestamp", "message")
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_existing_alert_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []
    records = []
    for line in lines:
        try:
            records.append(json.loads(line))
        except Exception:
            continue
    return records


def _alert_ack_signature_material(record: dict[str, Any]) -> bytes:
    payload = {
        "ack_token": record.get("ack_token"),
        "acknowledged_at": record.get("acknowledged_at"),
        "acknowledged_by": record.get("acknowledged_by"),
        "alert_id": record.get("alert_id"),
        "name": record.get("name"),
        "severity": record.get("severity"),
        "ts": record.get("ts"),
        "notes": record.get("notes") or "",
        "ack_version": int(record.get("ack_version") or 1),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return serialized.encode("utf-8")


def _alert_ack_signature(record: dict[str, Any], secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), _alert_ack_signature_material(record), hashlib.sha256)
    return base64.urlsafe_b64encode(digest.digest()).decode("ascii")


def generate_alert_ack_signature(record: dict[str, Any], secret: str) -> str:
    """Generate a deterministic acknowledgement signature for tooling/tests."""
    return _alert_ack_signature(record, secret)


def verify_alert_ack_signature(record: dict[str, Any], secret: str) -> bool:
    expected = record.get("ack_signature")
    if not expected:
        return False
    try:
        computed = _alert_ack_signature(record, secret)
    except Exception:
        return False
    return hmac.compare_digest(expected, computed)


def load_curriculum_alert_records(path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _runbook_alerts_path()
    return _load_existing_alert_records(target)


def _write_alert_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    atomic_write_json_lines(path, list(records))


def acknowledge_curriculum_alert(
    *,
    ack_token: str | None = None,
    alert_id: str | None = None,
    operator: str,
    notes: str = "",
    secret: str,
    artifacts_dir: Path | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    if not secret:
        raise ValueError("secret required to sign acknowledgement")
    path = (artifacts_dir / "ops" / "alerts.jsonl") if artifacts_dir else _runbook_alerts_path()
    records = load_curriculum_alert_records(path)
    if not records:
        raise FileNotFoundError("no curriculum alert records present")

    selected: dict[str, Any] | None = None
    for record in records:
        token_matches = ack_token and record.get("ack_token") == ack_token
        id_matches = alert_id and record.get("alert_id") == alert_id
        if token_matches or id_matches:
            selected = record
            break

    if not selected:
        raise ValueError("no matching alert record found for acknowledgement")

    current_time = float(timestamp) if timestamp is not None else time.time()
    selected["acknowledged"] = True
    selected["acknowledged_at"] = current_time
    selected["acknowledged_by"] = operator
    if notes:
        selected["notes"] = notes
    selected["ack_version"] = int(selected.get("ack_version") or 1)
    selected.pop("ack_signature", None)
    selected["ack_signature"] = _alert_ack_signature(selected, secret)

    _write_alert_records(path, records)
    return selected


def _summarize_alert_samples(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for record in records:
        sample = {key: record.get(key) for key in keys}
        sample = {k: v for k, v in sample.items() if v is not None}
        summaries.append(sample)
    return summaries


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    if fraction <= 0:
        return values[0]
    if fraction >= 1:
        return values[-1]
    position = fraction * (len(values) - 1)
    lower_index = int(math.floor(position))
    upper_index = int(math.ceil(position))
    if lower_index == upper_index:
        return values[lower_index]
    lower_value = values[lower_index]
    upper_value = values[upper_index]
    weight = position - float(lower_index)
    return (upper_value * weight) + (lower_value * (1.0 - weight))


def _compute_mttr_stats(durations: list[float]) -> dict[str, float] | None:
    if not durations:
        return None
    ordered = sorted(durations)
    count = len(ordered)
    total = sum(ordered)
    mean = total / float(count)
    median = _percentile(ordered, 0.5)
    p95 = _percentile(ordered, 0.95)
    return {
        "count": float(count),
        "mean": mean,
        "median": median,
        "p95": p95,
    }


def _curriculum_ack_summary(
    records: list[dict[str, Any]],
    sign_key: str | None,
    chaos_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:

    pending = [rec for rec in records if rec.get("requires_ack") and not rec.get("acknowledged")]
    acknowledged = [rec for rec in records if rec.get("requires_ack") and rec.get("acknowledged")]

    invalid: list[dict[str, Any]] = []
    signature_checked = bool(sign_key)
    if acknowledged:
        if sign_key:
            invalid = [rec for rec in acknowledged if not verify_alert_ack_signature(rec, sign_key)]
        else:
            # Without a signing key we treat acknowledgements as unverified for visibility.
            invalid = acknowledged

    blocking = len(pending) + len(invalid)
    valid_count = max(len(acknowledged) - len(invalid), 0)
    if len(acknowledged) > 0:
        valid_ratio = valid_count / float(len(acknowledged))
    else:
        valid_ratio = 1.0
    if not signature_checked and len(acknowledged) > 0:
        valid_ratio = 0.0

    chaos_duration: float | None = None
    if chaos_metrics is not None:
        try:
            chaos_duration = float(chaos_metrics.get("duration_ms"))
        except (TypeError, ValueError):
            chaos_duration = None

    # Update Prometheus-style gauges for observability.
    durations_all: list[float] = []
    durations_valid: list[float] = []
    invalid_tokens = {
        str(rec.get("ack_token") or rec.get("alert_id") or "")
        for rec in invalid
    }
    for rec in acknowledged:
        try:
            start = float(rec.get("ts"))
            end = float(rec.get("acknowledged_at"))
        except (TypeError, ValueError):
            continue
        delta = max(end - start, 0.0)
        durations_all.append(delta)
        token = str(rec.get("ack_token") or rec.get("alert_id") or "")
        if not invalid_tokens or token not in invalid_tokens:
            durations_valid.append(delta)

    mttr_stats = _compute_mttr_stats(durations_valid if signature_checked else durations_all)

    mttr_gate_payload: dict[str, Any] | None = None
    try:
        tolerance_value, baseline_min, history_limit = resolve_mttr_config()
        history_path = _mttr_history_path()
        payload, _ = build_mttr_gate_payload(
            mttr_stats,
            history_path=history_path,
            tolerance=tolerance_value,
            baseline_min=baseline_min,
            history_limit=history_limit,
            update_history=False,
        )
        mttr_gate_payload = {k: v for k, v in payload.items() if v not in (None, [], {})}
    except Exception:
        mttr_gate_payload = None

    try:
        brain_curriculum_ack_totals.set(float(len(pending)), state="pending")
        brain_curriculum_ack_totals.set(float(len(acknowledged)), state="acknowledged")
        brain_curriculum_ack_totals.set(float(len(invalid)), state="invalid")
        brain_curriculum_ack_totals.set(float(blocking), state="blocking")
        brain_curriculum_ack_signature_checked.set(1.0 if signature_checked else 0.0)
        brain_curriculum_ack_signature_valid_ratio.set(valid_ratio)
        brain_curriculum_ack_chaos_latency_ms.set(chaos_duration if chaos_duration is not None else 0.0)
        if mttr_stats:
            brain_curriculum_ack_mttr_seconds.set(mttr_stats["mean"], stat="mean")
            brain_curriculum_ack_mttr_seconds.set(mttr_stats["median"], stat="median")
            brain_curriculum_ack_mttr_seconds.set(mttr_stats["p95"], stat="p95")
        else:
            brain_curriculum_ack_mttr_seconds.set(0.0, stat="mean")
            brain_curriculum_ack_mttr_seconds.set(0.0, stat="median")
            brain_curriculum_ack_mttr_seconds.set(0.0, stat="p95")
    except Exception:
        pass

    latest_pending = sorted(pending, key=lambda rec: float(rec.get("ts") or 0.0))[-5:]
    latest_acknowledged = sorted(
        acknowledged,
        key=lambda rec: float(rec.get("acknowledged_at") or 0.0),
    )[-5:]

    summary: dict[str, Any] = {
        "total": len(records),
        "pending": len(pending),
        "acknowledged": len(acknowledged),
        "invalid_acknowledgements": len(invalid),
        "valid_acknowledgements": valid_count,
        "blocking": blocking,
        "signature_checked": signature_checked,
        "signature_valid_ratio": valid_ratio,
    }

    if mttr_stats:
        summary["mttr_seconds"] = mttr_stats

    if mttr_gate_payload:
        summary["mttr_gate"] = mttr_gate_payload

    if chaos_metrics:
        chaos_summary: dict[str, Any] = {
            "duration_ms": chaos_duration,
            "records_tested": int(chaos_metrics.get("records_tested") or 0),
            "iterations": int(chaos_metrics.get("iterations") or 0),
            "signing_key_present": bool(chaos_metrics.get("signing_key_present")),
        }
        generated_at = chaos_metrics.get("generated_at")
        if generated_at is not None:
            chaos_summary["generated_at"] = generated_at
        summary["chaos_drill"] = chaos_summary

    if latest_pending:
        summary["latest_pending"] = _summarize_alert_samples(
            latest_pending,
            ("alert_id", "severity", "ts", "ack_token"),
        )
    if latest_acknowledged:
        summary["latest_acknowledged"] = _summarize_alert_samples(
            latest_acknowledged,
            ("alert_id", "acknowledged_at", "acknowledged_by", "ack_signature"),
        )
    if invalid:
        summary["invalid_samples"] = _summarize_alert_samples(
            invalid,
            ("alert_id", "acknowledged_by", "ack_signature"),
        )
    return summary


def _persist_alert_runbook_entry(alert: dict[str, Any]) -> None:
    if not _alert_requires_runbook(alert):
        return

    path = _runbook_alerts_path()
    records = _load_existing_alert_records(path)

    ack_token = _alert_ack_token(alert)
    for record in records:
        if record.get("ack_token") == ack_token:
            return

    entry = {
        "alert_id": ack_token[:16],
        "ack_token": ack_token,
        "requires_ack": True,
        "acknowledged": False,
        "ack_version": 1,
        "ts": float(alert.get("timestamp") or time.time()),
        "name": alert.get("name"),
        "severity": alert.get("severity"),
        "message": alert.get("message"),
        "labels": alert.get("labels") or {},
        "source": "curriculum_dashboard",
        "value": alert.get("value"),
    }

    records.append(entry)
    _write_alert_records(path, records)
