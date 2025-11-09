from __future__ import annotations

"""Goal lifecycle and self-goal collectors for the proof bundle."""

import hashlib
import json
from pathlib import Path

from tools.ci import proof_bundle


def _sha256(text: str) -> str:
    digest = hashlib.sha256()
    digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def test_goal_lifecycle_collector_happy_path(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    goals_dir = artifacts / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)

    ledger_lines = [
        json.dumps({"goal_id": "g-1", "status": "proposed"}),
        json.dumps({"goal_id": "g-1", "status": "executed"}),
    ]
    ledger_text = "\n".join(ledger_lines)
    (goals_dir / "ledger.jsonl").write_text(ledger_text, encoding="utf-8")

    summary = {
        "total": 1,
        "open_status": {"evaluation": 0, "planning": 0, "execution": 0, "proposed": 0},
        "closed": {"succeeded": 1},
    }
    (goals_dir / "ledger.index.json").write_text(json.dumps(summary), encoding="utf-8")

    proof: dict[str, object] = {"gates": {}, "files": {}}
    proof_bundle._collect_goal_lifecycle(proof, artifacts)

    payload = proof.get("goal_lifecycle")
    assert isinstance(payload, dict)
    assert payload.get("present") is True
    assert payload.get("summary") == summary
    assert payload.get("ledger_sha256") == _sha256(ledger_text)

    gate = proof.get("gates", {}).get("goal_lifecycle")
    assert isinstance(gate, dict)
    assert gate.get("ok") is True
    assert gate.get("passed") is True
    assert gate.get("total") == summary["total"]
    metrics = gate.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics.get("total_goals") == summary["total"]

    files = proof.get("files", {})
    assert any(str(key).endswith("goals/ledger.jsonl") for key in files)
    assert any(str(key).endswith("goals/ledger.index.json") for key in files)


def test_goal_lifecycle_collector_absent(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    proof: dict[str, object] = {"gates": {}, "files": {}}
    proof_bundle._collect_goal_lifecycle(proof, artifacts)

    payload = proof.get("goal_lifecycle")
    assert isinstance(payload, dict)
    assert payload.get("present") is False

    gate = proof.get("gates", {}).get("goal_lifecycle")
    assert isinstance(gate, dict)
    assert gate.get("ok") is False
    assert gate.get("passed") is False
    assert gate.get("reason") == "missing_ledger"


def test_self_goals_collector_counts_autonomous_entries(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    goals_dir = artifacts / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)

    ledger_entries = [
        {
            "goal": {
                "id": "g-1",
                "status": "completed",
                "metadata": {"source": "self_goals"},
            }
        },
        {
            "goal": {
                "id": "g-2",
                "status": "planning",
                "metadata": {"source": "self_goals"},
            }
        },
        {
            "goal": {
                "id": "g-3",
                "status": "executed",
                "metadata": {"source": "manual"},
            }
        },
    ]
    lines = "\n".join(json.dumps(entry) for entry in ledger_entries)
    (goals_dir / "ledger.jsonl").write_text(lines, encoding="utf-8")

    proof: dict[str, object] = {"gates": {}, "files": {}}
    proof_bundle._collect_self_goals(proof, artifacts)

    payload = proof.get("self_goals")
    assert isinstance(payload, dict)
    assert payload.get("present") is True
    metrics = payload.get("metrics")
    assert isinstance(metrics, dict)
    assert metrics["self_goals_total"] == 2
    assert metrics["self_goals_completed_or_failed"] == 1
    assert metrics["self_goals_open"] == 1

    gate = proof.get("gates", {}).get("self_goals")
    assert isinstance(gate, dict)
    assert gate.get("ok") is True
    assert gate.get("count") == 2


def test_self_goals_collector_absent(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    proof: dict[str, object] = {"gates": {}, "files": {}}
    proof_bundle._collect_self_goals(proof, artifacts)

    payload = proof.get("self_goals")
    assert isinstance(payload, dict)
    assert payload.get("present") is False

    gate = proof.get("gates", {}).get("self_goals")
    assert isinstance(gate, dict)
    assert gate.get("ok") is True
    assert gate.get("present") is False
    assert gate.get("count") == 0
