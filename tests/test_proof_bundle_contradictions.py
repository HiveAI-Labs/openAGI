from __future__ import annotations

"""Tests for contradiction detectors and goal derivation in the proof bundle."""

import hashlib
import json
from pathlib import Path

import pytest

from tools.ci import proof_bundle


def _write_audit_entry(path: Path, *, scenario: str, success: bool, seed: int | None = None) -> None:
    payload = {
        "scenario": scenario,
        "seed": seed,
        "result": {"success": success, "name": scenario},
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def test_collect_contradictions_detects_mixed_outcomes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    audit_path = artifacts / "sandbox" / "suite_nav" / "audit.jsonl"
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    _write_audit_entry(audit_path, scenario="navigate_vault", success=True, seed=7)
    _write_audit_entry(audit_path, scenario="navigate_vault", success=False, seed=7)
    _write_audit_entry(audit_path, scenario="navigate_storage", success=True, seed=None)

    proof: dict[str, object] = {"gates": {}}
    proof_bundle._collect_contradictions(proof, artifacts, window=50)

    payload = proof.get("contradictions")
    assert isinstance(payload, dict)
    assert payload["total"] == 1
    suites = payload["suites"]
    assert isinstance(suites, list) and len(suites) == 1
    suite_entry = suites[0]
    assert suite_entry["suite"] == "suite_nav"
    assert suite_entry["contradictions"] == ["navigate_vault#seed=7"]

    expected_token = hashlib.sha256(
        json.dumps([{"suite": "suite_nav", "count": 1}], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert payload["determinism_token"] == expected_token

    gate = proof["gates"]["contradictions"]
    assert gate["total"] == 1
    assert gate["ok"] is False  # default max is 0


def test_collect_contradiction_goals_generates_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    proof: dict[str, object] = {
        "contradictions": {
            "total": 1,
            "suites": [
                {
                    "suite": "suite_nav",
                    "contradictions": ["navigate_vault#seed=7"],
                    "contradiction_count": 1,
                }
            ],
        },
        "gates": {},
    }

    proof_bundle._collect_contradiction_goals(proof, artifacts)

    payload = proof.get("consistency_goals")
    assert isinstance(payload, dict)
    assert payload["goal_count"] == 1
    latest_path = Path(payload["latest_path"])
    assert latest_path.exists()
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    assert latest["goal_count"] == 1
    assert latest["input_contradictions"] == 1

    goals_jsonl = latest_path.parent / "goals.jsonl"
    assert goals_jsonl.exists()
    lines = goals_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1

    gate = proof["gates"]["consistency_goals"]
    assert gate["ok"] is True
    assert gate["goal_count"] == 1


def test_collect_contradiction_goals_respects_minimum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = tmp_path / "artifacts"
    proof: dict[str, object] = {
        "contradictions": {
            "total": 1,
            "suites": [
                {
                    "suite": "suite_nav",
                    "contradictions": ["navigate_vault#seed=7"],
                    "contradiction_count": 1,
                }
            ],
        },
        "gates": {},
    }

    monkeypatch.setenv("PROOF_CONSISTENCY_GOALS_MIN", "3")
    proof_bundle._collect_contradiction_goals(proof, artifacts)

    gate = proof["gates"]["consistency_goals"]
    assert gate["ok"] is False
    assert gate["goal_count"] == 1
    assert gate["min_required_when_present"] == 3

    monkeypatch.delenv("PROOF_CONSISTENCY_GOALS_MIN", raising=False)
