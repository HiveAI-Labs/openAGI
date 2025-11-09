"""Check planner invariants gate status in the latest proof bundle.

Exit codes:
- 0: Pass or gate disabled or hard-fail disabled
- 2: Fail (planner or required gate) and BRAIN_INVARIANTS_GATES_HARD_FAIL=1

Usage:
  PYTHONPATH=. python tools/ci/check_proof_gate.py
Environment:
  BRAIN_INVARIANTS_GATES_HARD_FAIL: '1' to fail on gate failure (default '0')
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import sys


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}
    except Exception:
        return default


def _latest_proof_json(root: Path) -> Path | None:
    if not root.exists():
        return None
    cand = sorted([p for p in root.iterdir() if p.is_dir() and (p / "proof.json").exists()])
    if not cand:
        return None
    return cand[-1] / "proof.json"


def main() -> int:
    hard_fail = _bool_env("BRAIN_INVARIANTS_GATES_HARD_FAIL", False)
    proof_path = _latest_proof_json(Path("artifacts/proof"))
    if not proof_path or not proof_path.exists():
        print("No proof bundle found.")
        return 0
    data = json.loads(proof_path.read_text(encoding="utf-8"))
    planner_gates = (data.get("planner_invariants") or {}).get("gates") or {}
    enabled = bool(planner_gates.get("enabled"))
    status = str(planner_gates.get("status") or "unknown")

    summary: Dict[str, Any] = data.get("gates_summary") or {}
    required: List[str] = list(summary.get("required") or [])
    status_map: Dict[str, Any] = summary.get("status") or {}
    non_required_failures: List[str] = list(summary.get("non_required_failures") or [])

    failing_required = [gate for gate in required if not bool(status_map.get(gate))]

    gates_section = data.get("gates") or {}
    curriculum_gate = gates_section.get("curriculum_sandbox") if isinstance(gates_section, dict) else None

    payload: Dict[str, Any] = {
        "proof": str(proof_path),
        "planner_enabled": enabled,
        "planner_status": status,
    }
    if required:
        payload["required_gates"] = required
        payload["gate_status"] = {gate: bool(status_map.get(gate)) for gate in required}
    if non_required_failures:
        payload["non_required_failures"] = non_required_failures
    if failing_required:
        payload["failing_required_gates"] = failing_required
    if isinstance(curriculum_gate, dict):
        payload["curriculum_sandbox_gate"] = {
            k: v for k, v in curriculum_gate.items() if k not in {"path"} and v not in (None, [])
        }

    print(json.dumps(payload, indent=2))

    violations: List[str] = []
    if enabled and status == "fail":
        violations.append("planner_invariants")
    if failing_required:
        violations.extend(f"gate:{gate}" for gate in failing_required)
    if violations and hard_fail:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
