from __future__ import annotations

"""Smoke tests for the proof bundle generator with stubbed dependencies."""

import json
from pathlib import Path

import pytest

from tools.ci import proof_bundle


_STUB_FN_NAMES = (
    "_collect_plugin_inventory",
    "_collect_teach_phase_a_artifacts",
    "_collect_teach_phase_b_artifacts",
    "_collect_skill_artifacts",
    "_collect_bootstrap_artifacts",
    "_collect_retrain_validation_artifacts",
    "_collect_world_model_validation_artifacts",
    "_collect_resource_world_model_validation_artifacts",
    "_collect_coding_gate_artifacts",
    "_collect_policy_validation_artifacts",
    "_collect_curriculum_sandbox_artifacts",
    "_collect_consult_replay_artifacts",
    "_collect_selector_decisions",
    "_collect_selector_consults",
    "_curriculum_dashboard_gate",
    "_curriculum_alerts_gate",
    "_collect_contradictions",
    "_collect_contradiction_goals",
    "_collect_goal_lifecycle",
    "_collect_self_goals",
    "_world_model_retrain_gate",
    "_world_model_validation_gate",
    "_resource_world_model_validation_gate",
    "_coding_gate",
    "_policy_retrain_gate",
    "_collect_fuzz_summaries",
    "_plugin_host_metrics_summary",
    "_collect_packs_artifacts",
)


def _stub_optional_components(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace optional collectors with inert stubs."""

    for name in _STUB_FN_NAMES:
        monkeypatch.setattr(proof_bundle, name, lambda *args, **kwargs: None)

    monkeypatch.setattr(proof_bundle, "generate_coverage", lambda: {"ok": True})


def _stub_phase1_runner(monkeypatch: pytest.MonkeyPatch, artifacts_dir: Path, *, ok: bool) -> None:
    """Stub the phase 1 runner to avoid invoking pytest recursively."""

    log_path = artifacts_dir / "ci" / "phase1_tests.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stub phase1", encoding="utf-8")

    payload = {
        "ok": ok,
        "returncode": 0 if ok else 1,
        "command": ["pytest", "-q", "tests/research/test_stub.py"],
        "log_path": str(log_path),
    }
    monkeypatch.setattr(proof_bundle, "_run_phase1_tests", lambda _dir: payload)


def _stub_retrain_runner(monkeypatch: pytest.MonkeyPatch, artifacts_dir: Path, *, ok: bool) -> None:
    """Stub the world-model retrain validation runner."""

    log_path = artifacts_dir / "ci" / "retrain_validation.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("stub retrain", encoding="utf-8")

    payload = {
        "ok": ok,
        "returncode": 0 if ok else 1,
        "workspace": "default",
        "command": ["python", "scripts/retrain_stub.py"],
        "log_path": str(log_path),
    }
    monkeypatch.setattr(proof_bundle, "_run_world_model_retrain_validation", lambda _dir: payload)

    monkeypatch.setattr(proof_bundle, "_ensure_phase1_validation", lambda _dir: None)
    monkeypatch.setattr(proof_bundle, "_maybe_run_emergence_tests", lambda _dir: None)


def _prepare_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Setup a clean artifacts directory for the bundle run."""

    monkeypatch.chdir(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    monkeypatch.setenv("ARTIFACTS_DIR", str(artifacts))
    return artifacts


def _load_proof(path: Path) -> dict:
    proof_path = path / "proof.json"
    assert proof_path.exists(), "proof.json not produced"
    return json.loads(proof_path.read_text(encoding="utf-8"))


def _assert_proof_files(payload: dict) -> None:
    files = payload.get("files")
    assert isinstance(files, dict) and files, "proof files metadata missing"


def test_proof_bundle_minimal_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A minimal proof bundle run should succeed with stubbed collectors."""

    artifacts = _prepare_env(tmp_path, monkeypatch)
    _stub_optional_components(monkeypatch)
    _stub_phase1_runner(monkeypatch, artifacts, ok=True)
    _stub_retrain_runner(monkeypatch, artifacts, ok=True)

    proof_bundle.main()

    payload = _load_proof(tmp_path)
    _assert_proof_files(payload)

    gates = payload.get("gates")
    assert isinstance(gates, dict)
    assert gates["phase1_tests"]["ok"] is True
    assert gates["world_model_retrain_execution"]["ok"] is True

    summary = payload.get("gates_summary")
    assert summary and summary["all_pass"] is True


def test_proof_bundle_gate_failure_does_not_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under pytest, failing gates should not terminate the process even when fail flag is set."""

    artifacts = _prepare_env(tmp_path, monkeypatch)
    _stub_optional_components(monkeypatch)
    _stub_phase1_runner(monkeypatch, artifacts, ok=False)
    _stub_retrain_runner(monkeypatch, artifacts, ok=True)

    monkeypatch.setenv("PROOF_FAIL_ON_REQUIRED_GATES", "1")

    proof_bundle.main()

    payload = _load_proof(tmp_path)
    _assert_proof_files(payload)

    summary = payload.get("gates_summary")
    assert summary
    assert summary["all_pass"] is False
    assert summary["status"]["phase1_tests"] is False
    assert summary["status"]["world_model_retrain_execution"] is True

    monkeypatch.delenv("PROOF_FAIL_ON_REQUIRED_GATES", raising=False)
