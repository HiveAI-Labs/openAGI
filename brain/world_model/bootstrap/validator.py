"""Validation harness for LLM-derived navigation hypotheses."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping, Sequence
import random
import time

from brain.obs.metrics import (
    brain_wm_domain_accuracy,
    brain_wm_hypotheses_total,
    brain_wm_validation_failures_total,
)
from brain.world_model.simple_model import SimpleState, SimpleWorldModel

from .gap_recorder import PredictionGapRecorder
from .registry import BatchHandle, HypothesisRegistry
from .schemas import (
    ActionHypothesis,
    HypothesisBatch,
    TestCase,
    ValidatedRule,
    ValidationResult,
    ValidationRun,
)


def _normalize_room(label: str) -> str:
    normalized = label.strip().lower().replace(" ", "_")
    if normalized == "vault":
        return "vault_room"
    if normalized == "storage":
        return "storage_room"
    if normalized == "start":
        return "start_room"
    return normalized


def _tokenize_state(state_str: str) -> Sequence[str]:
    return [token.strip() for token in state_str.split("|") if token.strip()]


class NavigationValidator:
    """Validate navigation hypotheses using a deterministic SimpleWorldModel."""

    def __init__(
        self,
        *,
        world_model: SimpleWorldModel | None = None,
        registry: HypothesisRegistry | None = None,
        promotion_threshold: float = 0.95,
        ensemble_threshold: float | None = None,
        gap_threshold: float = 0.8,
        gap_recorder: PredictionGapRecorder | None = None,
    ) -> None:
        self._world_model = world_model or SimpleWorldModel()
        self._registry = registry or HypothesisRegistry()
        self._promotion_threshold = max(0.0, min(1.0, promotion_threshold))
        if ensemble_threshold is None:
            ensemble_threshold = max(0.98, self._promotion_threshold)
        self._ensemble_threshold = max(self._promotion_threshold, float(ensemble_threshold))
        self._gap_threshold = max(0.0, min(1.0, float(gap_threshold)))
        self._gap_recorder = gap_recorder
        self._last_results: dict[str, ValidationResult] = {}
        self._last_handle: BatchHandle | None = None

    @property
    def last_results(self) -> Mapping[str, ValidationResult]:
        return dict(self._last_results)

    @property
    def last_batch_handle(self) -> BatchHandle | None:
        return self._last_handle

    def validate(self, batch: HypothesisBatch) -> list[ValidatedRule]:
        """Validate *batch* of hypotheses, producing validated rules when thresholds pass."""
        handle = self._registry.record_batch(batch)
        self._last_handle = handle
        validated: list[ValidatedRule] = []
        ensemble_hashes: set[str] = set()
        meta_hashes = batch.metadata.get("ensemble_hashes")
        if isinstance(meta_hashes, (list, tuple, set)):
            ensemble_hashes = {str(value) for value in meta_hashes}
        for hypothesis in handle.hypotheses:
            fingerprint = hypothesis.fingerprint()
            is_ensemble = fingerprint in ensemble_hashes
            threshold = self._ensemble_threshold if is_ensemble else self._promotion_threshold
            result = self._validate_hypothesis(handle, hypothesis, threshold, is_ensemble=is_ensemble)
            self._registry.record_validation_summary(handle, result)
            self._last_results[result.hypothesis_hash] = result
            if result.success_rate >= threshold and result.reproducible:
                brain_wm_hypotheses_total.inc(status="validated")
                validated.append(ValidatedRule(hypothesis=hypothesis, validation=result))
            else:
                brain_wm_hypotheses_total.inc(status="rejected")
            domain = batch.metadata.get("domain_fingerprint") or "unknown"
            try:
                brain_wm_domain_accuracy.labels(str(domain)).set(float(result.success_rate))
            except Exception:
                pass
            if self._gap_recorder and result.success_rate < self._gap_threshold:
                try:
                    self._gap_recorder.record(
                        domain=str(domain),
                        success_rate=float(result.success_rate),
                        threshold=float(self._gap_threshold),
                        details={
                            "hypothesis": hypothesis.model_dump(mode="json"),
                            "is_ensemble": bool(is_ensemble),
                            "promotion_threshold": threshold,
                        },
                    )
                except Exception:
                    pass
        return validated

    def _validate_hypothesis(
        self,
        handle: BatchHandle,
        hypothesis: ActionHypothesis,
        threshold: float,
        *,
        is_ensemble: bool,
    ) -> ValidationResult:
        runs: list[ValidationRun] = []
        success_count = 0
        for case in hypothesis.test_cases:
            run = self._execute_case(handle, hypothesis, case, is_ensemble=is_ensemble)
            runs.append(run)
            if run.success:
                success_count += 1
            else:
                brain_wm_validation_failures_total.inc(reason="simulation_mismatch")
        total = len(runs) or 1
        success_rate = success_count / total
        reproducible = success_rate >= threshold and success_count == len(runs)
        return ValidationResult(
            hypothesis_hash=hypothesis.fingerprint(),
            success_rate=success_rate,
            reproducible=reproducible,
            runs=runs,
        )

    def _execute_case(
        self,
        handle: BatchHandle,
        hypothesis: ActionHypothesis,
        case: TestCase,
        *,
        is_ensemble: bool,
    ) -> ValidationRun:
        random.seed(case.seed)
        start = time.perf_counter()
        details: MutableMapping[str, object] = {
            "hypothesis": hypothesis.model_dump(mode="json"),
            "ensemble_source": bool(is_ensemble),
        }
        try:
            initial_state = self._serialize_initial_state(case.initial_state)
            normalized_action = self._normalize_action(hypothesis.action)
            next_state, reward = self._world_model.predict_transition(initial_state, normalized_action)
            details.update(
                {
                    "initial_state": initial_state,
                    "next_state": next_state,
                    "action": normalized_action,
                    "reward": reward,
                },
            )
            success = self._matches_expected(next_state, reward, case.expected_outcome)
        except Exception as exc:  # pragma: no cover - defensive branch
            success = False
            details["error"] = str(exc)
        latency_ms = (time.perf_counter() - start) * 1000.0
        case_hash, log_path = self._registry.record_validation_run(
            handle,
            hypothesis,
            case,
            success=success,
            latency_ms=latency_ms,
            details=details,
        )
        return ValidationRun(
            test_case_hash=case_hash,
            success=success,
            latency_ms=latency_ms,
            log_path=log_path,
        )

    def _serialize_initial_state(self, payload: Mapping[str, object]) -> str:
        room = payload.get("room") or payload.get("location")
        if not isinstance(room, str) or not room.strip():
            raise ValueError("initial_state requires a room")
        holding = payload.get("holding")
        unlocked = payload.get("unlocked")
        unlocked_set = set()
        if isinstance(unlocked, Iterable) and not isinstance(unlocked, (str, bytes)):
            unlocked_set = {_normalize_room(str(value)) for value in unlocked}
        holding_value = None if holding is None else str(holding)
        state = SimpleState(location=_normalize_room(room), holding=holding_value, unlocked=set(unlocked_set))
        return str(state)

    def _normalize_action(self, action: str) -> str:
        normalized = action.strip()
        lower = normalized.lower()
        if lower.startswith("move_to_"):
            destination = _normalize_room(lower[len("move_to_"):])
            return f"navigate {destination}"
        if lower.startswith("go_to_"):
            destination = _normalize_room(lower[len("go_to_"):])
            return f"navigate {destination}"
        return normalized

    def _matches_expected(self, state_str: str, reward: float, expected: Mapping[str, object]) -> bool:
        tokens = set(_tokenize_state(state_str))
        if "room" in expected:
            expected_room = _normalize_room(str(expected["room"]))
            if f"at_{expected_room}" not in tokens:
                return False
        if "holding" in expected:
            holding = expected["holding"]
            if holding is None:
                if any(token.startswith("holding_") for token in tokens):
                    return False
            elif f"holding_{holding}" not in tokens:
                return False
        if "unlocked" in expected:
            expected_unlocked = {
                f"unlocked_{_normalize_room(str(item))}" for item in expected["unlocked"]  # type: ignore[arg-type]
            }
            if not expected_unlocked.issubset(tokens):
                return False
        if "reward" in expected:
            try:
                target_reward = float(expected["reward"])
            except (TypeError, ValueError):
                target_reward = reward
            if abs(reward - target_reward) > 1e-6:
                return False
        return True


__all__ = ["NavigationValidator"]
