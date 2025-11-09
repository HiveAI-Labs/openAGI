from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

from brain.io.validators import NormalizedRequest, ValidationResult

__all__ = ["record_probe_alert"]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def record_probe_alert(
    artifacts_root: str | Path,
    run_id: str,
    req: NormalizedRequest,
    validation: ValidationResult,
) -> None:
    """Persist probe findings for audit and gating workflows.

    Writes an append-only ``probe_alerts.jsonl`` log and a ``probe_gate.json``
    summary that surfaces the most recent blocking/quarantine counts. Both live
    under ``<artifacts_root>/io`` alongside run artifacts so ToolForge gates and
    humans can review outstanding issues quickly.
    """
    findings = validation.probe_findings or {}
    issues = findings.get("issues") if isinstance(findings, dict) else None
    if not isinstance(issues, list):
        issues = []

    root = Path(artifacts_root)
    alerts_path = root / "io" / "probe_alerts.jsonl"
    gate_path = root / "io" / "probe_gate.json"
    record = {
        "ts": time.time(),
        "run_id": run_id,
        "decision": validation.decision,
        "query_hash": req.query_hash,
        "autonomy_level": req.autonomy_level,
        "issues": issues,
        "summary": findings.get("summary", {}),
    }

    if issues:
        _ensure_parent(alerts_path)
        with alerts_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    blocking = sum(1 for issue in issues if issue.get("severity") == "error")
    quarantines = sum(1 for issue in issues if issue.get("severity") == "warning")
    gate_summary: dict[str, Any] = {
        "ts": record["ts"],
        "run_id": run_id,
        "decision": validation.decision,
        "blocking_errors": blocking,
        "quarantine_warnings": quarantines,
        "last_issue_kinds": [issue.get("kind") for issue in issues],
    }
    _ensure_parent(gate_path)
    gate_path.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
