# Alerting Rules for HiveNet AI Brain
# Defines alerting conditions and thresholds for system monitoring

import os
import time
from typing import Any

import psutil

from typing import TYPE_CHECKING

# The enhanced observability module is optional. Use TYPE_CHECKING so the
# static checker sees the actual symbols, and at runtime attempt to import the
# module and provide safe no-op fallbacks when it's absent.
if TYPE_CHECKING:  # pragma: no cover - types only
    # Declare symbol names as permissive Any types for static checking. Importing
    # the real module here can trigger attr-defined errors when the module is
    # partially implemented in certain test/dev environments.
    from typing import Any as _Any

    AlertSeverity: _Any
    alert_manager: _Any
    curriculum_dashboard_snapshot: _Any
    metrics_registry: _Any
    structured_logger: _Any
else:  # pragma: no branch - runtime fallback
    try:
        from brain.obs.enhanced_observability import (
            AlertSeverity,
            alert_manager,
            curriculum_dashboard_snapshot,
            metrics_registry,
            structured_logger,
        )
    except Exception:
        # Minimal fallbacks to keep alerting functional in environments where
        # enhanced observability is not installed. These are conservative
        # no-ops and do not attempt to replicate full behavior.
        class AlertSeverity:
            WARNING = "warning"
            ERROR = "error"
            CRITICAL = "critical"

        class _NoopAlertManager:
            def add_rule(self, *_, **__):
                return None

        alert_manager = _NoopAlertManager()

        def curriculum_dashboard_snapshot() -> dict:
            return {}

        class _NoopMetricsRegistry:
            def get_health_status(self) -> dict:
                return {"status": "healthy"}

        metrics_registry = _NoopMetricsRegistry()

        class _NoopStructuredLogger:
            def info(self, *_args, **_kwargs):
                return None

        structured_logger = _NoopStructuredLogger()
from brain.obs.metrics import clear_curriculum_metrics
from brain.universe.curriculum_sandbox import load_curriculum_summary


def _parse_int_env(name: str, default: int = 0) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return int(float(raw))
    except Exception:
        return default

def setup_alerting_rules():
    """Set up alerting rules for system monitoring."""
    clear_curriculum_metrics()

    # High CPU usage alert
    def check_high_cpu():
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent > 85

    alert_manager.add_rule(
        name="high_cpu_usage",
        condition=check_high_cpu,
        severity=AlertSeverity.WARNING,
        message="CPU usage is above 85%",
        labels={"component": "system", "metric": "cpu"},
    )

    # Critical CPU usage alert
    def check_critical_cpu():
        cpu_percent = psutil.cpu_percent(interval=1)
        return cpu_percent > 95

    alert_manager.add_rule(
        name="critical_cpu_usage",
        condition=check_critical_cpu,
        severity=AlertSeverity.CRITICAL,
        message="CPU usage is above 95%",
        labels={"component": "system", "metric": "cpu"},
    )

    # High memory usage alert
    def check_high_memory():
        memory = psutil.virtual_memory()
        return memory.percent > 85

    alert_manager.add_rule(
        name="high_memory_usage",
        condition=check_high_memory,
        severity=AlertSeverity.WARNING,
        message="Memory usage is above 85%",
        labels={"component": "system", "metric": "memory"},
    )

    # Critical memory usage alert
    def check_critical_memory():
        memory = psutil.virtual_memory()
        return memory.percent > 95

    alert_manager.add_rule(
        name="critical_memory_usage",
        condition=check_critical_memory,
        severity=AlertSeverity.CRITICAL,
        message="Memory usage is above 95%",
        labels={"component": "system", "metric": "memory"},
    )

    # Low disk space alert
    def check_low_disk_space():
        disk = psutil.disk_usage("/")
        return disk.percent > 90

    alert_manager.add_rule(
        name="low_disk_space",
        condition=check_low_disk_space,
        severity=AlertSeverity.ERROR,
        message="Disk space is below 10%",
        labels={"component": "system", "metric": "disk"},
    )

    # Critical disk space alert
    def check_critical_disk_space():
        disk = psutil.disk_usage("/")
        return disk.percent > 95

    alert_manager.add_rule(
        name="critical_disk_space",
        condition=check_critical_disk_space,
        severity=AlertSeverity.CRITICAL,
        message="Disk space is below 5%",
        labels={"component": "system", "metric": "disk"},
    )

    # High error rate alert
    last_error_count = {"value": 0, "timestamp": time.time()}

    def check_high_error_rate():
        nonlocal last_error_count
        current_time = time.time()

        # Get current error metrics (this is a simplified example)
        # In a real implementation, you'd query the metrics registry
        current_errors = getattr(metrics_registry, "errors_total", None)
        if current_errors is None:
            return False

        # Calculate error rate over last 5 minutes
        time_diff = current_time - last_error_count["timestamp"]
        if time_diff < 300:  # 5 minutes
            return False

        # This is a placeholder - you'd need to implement proper error rate calculation
        error_rate = 0.0  # Calculate from metrics

        last_error_count["timestamp"] = current_time
        return error_rate > 0.1  # 10% error rate

    alert_manager.add_rule(
        name="high_error_rate",
        condition=check_high_error_rate,
        severity=AlertSeverity.ERROR,
        message="Error rate is above 10% in the last 5 minutes",
        labels={"component": "application", "metric": "error_rate"},
    )

    # Service unavailable alert
    def check_service_unavailable():
        health_status = metrics_registry.get_health_status()
        return health_status.get("status") == "unhealthy"

    alert_manager.add_rule(
        name="service_unavailable",
        condition=check_service_unavailable,
        severity=AlertSeverity.CRITICAL,
        message="Service is in unhealthy state",
        labels={"component": "application", "metric": "health"},
    )

    # Brain operation failures alert
    brain_failures = {"count": 0, "timestamp": time.time()}

    def check_brain_operation_failures():
        nonlocal brain_failures
        current_time = time.time()

        # Reset counter every 10 minutes
        if current_time - brain_failures["timestamp"] > 600:
            brain_failures["count"] = 0
            brain_failures["timestamp"] = current_time

        # Check for recent brain operation failures
        # This would need to be integrated with actual brain operation tracking
        return brain_failures["count"] > 10  # More than 10 failures in 10 minutes

    alert_manager.add_rule(
        name="brain_operation_failures",
        condition=check_brain_operation_failures,
        severity=AlertSeverity.ERROR,
        message="High rate of brain operation failures detected",
        labels={"component": "brain", "metric": "operation_failures"},
    )

    def _curriculum_dashboard_state() -> dict[str, Any]:
        dashboard = curriculum_dashboard_snapshot()
        alerts = dashboard.get("alerts") or []
        warning_alerts = [a for a in alerts if str(a.get("level")) == "warning"]
        critical_alerts = [a for a in alerts if str(a.get("level")) == "critical"]
        totals = dashboard.get("totals") or {}
        return {
            "dashboard": dashboard,
            "warning_alerts": warning_alerts,
            "critical_alerts": critical_alerts,
            "total_suites": int(totals.get("suites") or 0),
        }

    curriculum_warning_labels = {
        "component": "curriculum",
        "metric": "dashboard",
        "alert_type": "warnings",
    }

    def check_curriculum_warning() -> bool:
        state = _curriculum_dashboard_state()
        total_suites = state["total_suites"]
        if total_suites <= 0:
            curriculum_warning_labels["count"] = "0"
            curriculum_warning_labels.pop("limit", None)
            return False
        warnings_limit = max(0, _parse_int_env("BRAIN_CURRICULUM_MAX_WARNINGS", 0))
        warning_count = len(state["warning_alerts"])
        curriculum_warning_labels["count"] = str(warning_count)
        curriculum_warning_labels["limit"] = str(warnings_limit)
        curriculum_warning_labels["suites"] = str(total_suites)
        return warning_count > warnings_limit

    alert_manager.add_rule(
        name="curriculum_dashboard_warning",
        condition=check_curriculum_warning,
        severity=AlertSeverity.WARNING,
        message="Curriculum dashboard warnings exceed allowed threshold",
        labels=curriculum_warning_labels,
    )

    curriculum_critical_labels = {
        "component": "curriculum",
        "metric": "dashboard",
        "alert_type": "critical",
    }

    def check_curriculum_critical() -> bool:
        state = _curriculum_dashboard_state()
        total_suites = state["total_suites"]
        if total_suites <= 0:
            curriculum_critical_labels["count"] = "0"
            curriculum_critical_labels.pop("limit", None)
            return False
        critical_limit = max(0, _parse_int_env("BRAIN_CURRICULUM_MAX_CRITICAL", 0))
        critical_count = len(state["critical_alerts"])
        curriculum_critical_labels["count"] = str(critical_count)
        curriculum_critical_labels["limit"] = str(critical_limit)
        curriculum_critical_labels["suites"] = str(total_suites)
        return critical_count > critical_limit

    alert_manager.add_rule(
        name="curriculum_dashboard_critical",
        condition=check_curriculum_critical,
        severity=AlertSeverity.CRITICAL,
        message="Curriculum dashboard critical alerts exceed allowed threshold",
        labels=curriculum_critical_labels,
    )

    sandbox_stale_labels = {
        "component": "curriculum",
        "metric": "sandbox",
        "alert_type": "stale",
    }

    def _curriculum_sandbox_state() -> dict[str, Any]:
        artifacts_root = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts"
        stale_suites: list[str] = []
        max_age: float | None = None
        summary_token: str | None = None
        stale_tokens: list[str] = []
        try:
            summary = load_curriculum_summary(artifacts_root)
        except Exception:
            summary = None
        if summary:
            summary_token = summary.determinism_token
            for stats in summary.suites:
                if stats.age_hours is not None:
                    max_age = stats.age_hours if max_age is None else max(max_age, stats.age_hours)
                if stats.stale:
                    stale_suites.append(stats.suite)
                    if stats.determinism_token:
                        stale_tokens.append(stats.determinism_token)
        return {
            "stale_suites": stale_suites,
            "max_age_hours": max_age,
            "determinism_token": stale_tokens[0] if stale_tokens else summary_token,
            "stale_tokens": stale_tokens,
            "summary_token": summary_token,
        }

    def check_curriculum_sandbox_stale() -> bool:
        state = _curriculum_sandbox_state()
        stale_suites = state["stale_suites"]
        max_age = state["max_age_hours"]
        sandbox_stale_labels["stale_suites"] = ",".join(stale_suites[:5]) if stale_suites else ""
        sandbox_stale_labels["stale_count"] = str(len(stale_suites))
        if max_age is not None:
            sandbox_stale_labels["max_age_hours"] = f"{max_age:.2f}"
        else:
            sandbox_stale_labels.pop("max_age_hours", None)
        if state["determinism_token"]:
            sandbox_stale_labels["determinism_token"] = str(state["determinism_token"])
        elif state["summary_token"]:
            sandbox_stale_labels["determinism_token"] = str(state["summary_token"])
        else:
            sandbox_stale_labels.pop("determinism_token", None)
        if state["summary_token"]:
            sandbox_stale_labels["summary_token"] = str(state["summary_token"])
        else:
            sandbox_stale_labels.pop("summary_token", None)
        return bool(stale_suites)

    alert_manager.add_rule(
        name="curriculum_sandbox_stale",
        condition=check_curriculum_sandbox_stale,
        severity=AlertSeverity.WARNING,
        message="Curriculum sandbox suites are stale",
        labels=sandbox_stale_labels,
    )

def evaluate_alerts():
    """Evaluate all alerting rules."""
    alert_manager.evaluate_rules()

def get_active_alerts() -> dict[str, Any]:
    """Get active alerts summary."""
    alerts = metrics_registry.get_alerts()

    # Group by severity
    summary = {
        "total": len(alerts),
        "by_severity": {
            "info": 0,
            "warning": 0,
            "error": 0,
            "critical": 0,
        },
        "recent_alerts": alerts[-10:],  # Last 10 alerts
    }

    for alert in alerts:
        severity = alert.get("severity", "info")
        if severity in summary["by_severity"]:
            summary["by_severity"][severity] += 1

    return summary

def log_alerts():
    """Log alerts using structured logging."""
    alerts = get_active_alerts()

    if alerts["total"] > 0:
        structured_logger.warning(
            f"Active alerts: {alerts['total']} total",
            alert_summary=alerts,
            alert_count=alerts["total"],
        )

        # Log critical alerts individually
        for alert in alerts["recent_alerts"]:
            if alert.get("severity") == "critical":
                structured_logger.error(
                    f"Critical alert: {alert.get('message', 'Unknown')}",
                    alert_name=alert.get("name"),
                    alert_severity=alert.get("severity"),
                    alert_timestamp=alert.get("timestamp"),
                )

# Initialize alerting rules
setup_alerting_rules()
