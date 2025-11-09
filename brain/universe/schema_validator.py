"""Small schema validator utilities for curriculum sandbox summaries.

This module intentionally avoids adding a heavy jsonschema dependency for a
lightweight repository check. It performs a pragmatic structural validation
that is sufficient for gating and tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_curriculum_summary_structure(payload: dict[str, Any]) -> list[str]:
    """Validate a loaded curriculum summary payload.

    Returns a list of string error messages. Empty list means validation passed.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["payload must be an object"]

    if "generated_at" not in payload:
        errors.append("missing generated_at")
    if "artifacts_dir" not in payload:
        errors.append("missing artifacts_dir")
    suites = payload.get("suites")
    if suites is None:
        errors.append("missing suites")
    elif not isinstance(suites, list):
        errors.append("suites must be a list")
    else:
        for i, s in enumerate(suites):
            if not isinstance(s, dict):
                errors.append(f"suite[{i}] must be object")
                continue
            if "suite" not in s:
                errors.append(f"suite[{i}] missing 'suite' name")
            if "runs_total" not in s:
                errors.append(f"suite[{i}] missing 'runs_total'")
    return errors


def load_schema_text() -> str:
    p = Path(__file__).with_name("schemas").joinpath("curriculum_summary_schema.json")
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return "{}"


def load_and_validate_from_artifacts(artifacts_dir: Path) -> list[str]:
    """Load the curriculum summary (if present) and validate structure.

    Returns errors (empty if OK). This is helpful for tests and lightweight CI.
    """
    from brain.universe.curriculum_sandbox import load_curriculum_summary

    summary = load_curriculum_summary(artifacts_dir)
    # Convert dataclass to dict for generic checks
    payload = json.loads(json.dumps(summary, default=lambda o: getattr(o, "__dict__", str(o))))
    return validate_curriculum_summary_structure(payload)
