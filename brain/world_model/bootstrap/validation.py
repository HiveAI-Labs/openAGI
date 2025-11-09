"""Validation helpers for the world-model bootstrap pipeline.

This module consolidates lightweight in-process validation (fast, deterministic
via `NavigationValidator`) and an empirical simulator-backed `HypothesisValidator`
used for end-to-end proof artifacts.

Exports:
 - validate_navigation_batch(batch, *, world_model=None, validator=None)
 - HypothesisValidator: run tests against the simulator and write proof JSONs
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

from brain.universe.sim_manager import run_scripted_episode

from .schemas import ActionHypothesis, HypothesisBatch, ValidatedRule, ValidationResult
from .validator import NavigationValidator


@dataclass(frozen=True)
class ValidationSummary:
    """Compact summary returned by :func:`validate_navigation_batch`."""

    validated_rules: Sequence[ValidatedRule]
    results: Mapping[str, ValidationResult]


def validate_navigation_batch(
    batch: HypothesisBatch,
    *,
    world_model: object | None = None,
    validator: NavigationValidator | None = None,
) -> ValidationSummary:
    """Validate ``batch`` of hypotheses using a `NavigationValidator`.

    If `validator` is provided it will be used directly; otherwise a new
    `NavigationValidator` is constructed and used. Returns a lightweight
    summary with promoted rules and full result map.
    """
    active_validator = validator or NavigationValidator(world_model=world_model)
    validated = active_validator.validate(batch)
    results: dict[str, ValidationResult] = dict(active_validator.last_results)
    return ValidationSummary(validated_rules=tuple(validated), results=results)


def _sha256_text(s: str) -> str:
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


class HypothesisValidator:
    """Empirical validator that executes test cases in the real simulator.

    This is used to produce a human-auditable proof artifact (JSON) under
    `artifacts/proof_bootstrap/` describing per-case outcomes. It is slower
    but provides stronger empirical evidence than the in-process validator.
    """

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.artifacts = Path(artifacts_dir or Path("artifacts"))
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self.out_dir = self.artifacts / "proof_bootstrap"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, hypothesis: ActionHypothesis) -> ValidationResult:
        runs: list[dict[str, object]] = []
        details: list[dict[str, object]] = []
        total = 0
        success = 0
        for case in hypothesis.test_cases:
            r1 = run_scripted_episode(
                workspace="bootstrap_validator",
                mode="generalization_plus",
                seed=case.seed,
                artifacts_dir=self.artifacts,
            )
            ok1 = bool(r1.get("success"))
            r2 = run_scripted_episode(
                workspace="bootstrap_validator",
                mode="generalization_plus",
                seed=case.seed,
                artifacts_dir=self.artifacts,
            )
            ok2 = bool(r2.get("success"))
            rep = ok1 == ok2
            case_payload = case.model_dump()
            case_hash = _sha256_text(json.dumps(case_payload, sort_keys=True, ensure_ascii=True))
            details.append(
                {
                    "test_case_hash": case_hash,
                    "success_first": ok1,
                    "success_second": ok2,
                    "reproducible": rep,
                },
            )
            runs.append(
                {
                    "test_case_hash": case_hash,
                    "success": ok1,
                    "latency_ms": 0.0,
                    "log_path": None,
                },
            )
            runs.append(
                {
                    "test_case_hash": case_hash,
                    "success": ok2,
                    "latency_ms": 0.0,
                    "log_path": None,
                },
            )
            total += 1
            if ok1:
                success += 1

        rate = (success / total) if total else 0.0
        reproducible = all(item["reproducible"] for item in details) if details else True

        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "hypothesis": json.loads(hypothesis.model_dump_json()),
            "results": details,
            "success_rate": rate,
            "reproducible": reproducible,
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        fname = f"validation_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.json"
        path = self.out_dir / fname
        path.write_text(text, encoding="utf-8")
        # Build a ValidationResult compatible with the bootstrap schemas (runs -> sequence of ValidationRun)
        vr_payload = {
            "hypothesis_hash": hypothesis.fingerprint() if hasattr(hypothesis, "fingerprint") else _sha256_text(json.dumps(hypothesis.model_dump())),
            "success_rate": rate,
            "reproducible": reproducible,
            "runs": runs,
            "proof_artifact": str(path),
        }
        # Use model validation to ensure shape compatibility
        return ValidationResult.model_validate(vr_payload)


__all__ = ["ValidationSummary", "validate_navigation_batch", "HypothesisValidator"]
