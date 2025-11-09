from __future__ import annotations

"""Regression tests for health check helpers in the observability package."""

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure we import the health check module from the openAGI tree instead of any
# pre-installed package from the parent repository.
for mod in ["brain.obs.health_checks", "brain.obs", "brain"]:
    sys.modules.pop(mod, None)

from brain.obs import health_checks as hc


def test_check_system_resources_degrades_without_psutil(monkeypatch):
    """check_system_resources should degrade gracefully when psutil is absent."""
    monkeypatch.setattr(hc, "PSUTIL_AVAILABLE", False)
    result = hc.check_system_resources()
    assert result.name == "system_resources"
    assert result.status == hc.HealthStatus.DEGRADED
    assert "psutil" in result.message


def test_check_brain_components_reports_missing_modules(monkeypatch):
    """Missing optional modules must mark the health check as degraded."""
    original_import = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "brain.meta.strategy_selector":
            raise ImportError("module missing")
        return original_import(name, package)

    monkeypatch.setattr(hc.importlib, "import_module", fake_import)
    result = hc.check_brain_components()

    assert result.name == "brain_components"
    assert result.status == hc.HealthStatus.DEGRADED
    assert "strategy_selector" in result.details["missing_components"]


def test_run_health_checks_noop_without_registry(monkeypatch):
    """run_health_checks should no-op when no registry is configured."""
    monkeypatch.setattr(hc, "health_checker", None)
    hc.run_health_checks()
