"""Pydantic data models for the LLM-guided world model bootstrap pipeline."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _stable_dumps(payload: Any) -> str:
    """Serialize payload with stable ordering to support reproducible hashes."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def compute_sha256(text: str) -> str:
    """Return a hexadecimal SHA-256 digest for ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestCase(BaseModel):
    """A deterministic simulation probe used to validate action hypotheses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Prevent pytest from misidentifying this helper as a test container
    __test__ = False

    description: str = Field(..., min_length=1, max_length=256)
    initial_state: Mapping[str, Any] = Field(default_factory=dict)
    expected_outcome: Mapping[str, Any] = Field(default_factory=dict)
    max_steps: int = Field(default=10, ge=1, le=100)
    seed: int = Field(default=0, ge=0)
    timeout_s: float = Field(default=3.0, gt=0.0, le=10.0)

    @field_validator("initial_state", "expected_outcome", mode="before")
    @classmethod
    def _ensure_mapping(cls, value: Any) -> Mapping[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, dict):
            return dict(value)
        raise TypeError("initial_state and expected_outcome must be mappings")


class TransitionObservation(BaseModel):
    """Observed transition in the environment, used to seed the prompt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    state: str = Field(..., min_length=1, max_length=128)
    action: str = Field(..., min_length=1, max_length=64)
    next_state: str = Field(..., min_length=1, max_length=128)
    success: bool = Field(default=True)
    notes: str | None = Field(default=None, max_length=256)


class DomainSnapshot(BaseModel):
    """Minimal snapshot of the navigation domain shared with the LLM teacher."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rooms: Sequence[str] = Field(..., min_length=1)
    objects: Sequence[str] = Field(default_factory=list)
    transitions: Sequence[TransitionObservation] = Field(default_factory=list)
    constraints: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("rooms", "objects", mode="before")
    @classmethod
    def _ensure_sequence(cls, value: Any) -> Sequence[str]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [str(item) for item in value]
        raise TypeError("rooms and objects must be sequences of strings")

    def fingerprint(self) -> str:
        """Stable digest representing the snapshot content."""
        payload = self.model_dump()
        return compute_sha256(_stable_dumps(payload))


class ActionHypothesis(BaseModel):
    """A candidate action rule proposed by the LLM."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str = Field(..., min_length=1, max_length=64)
    parameters: Mapping[str, Any] = Field(default_factory=dict)
    preconditions: Sequence[str] = Field(default_factory=list)
    effects: Sequence[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    validation_steps: Sequence[str] = Field(..., min_length=1)
    test_cases: Sequence[TestCase] = Field(..., min_length=1)

    @field_validator("preconditions", "effects", "validation_steps", mode="before")
    @classmethod
    def _ensure_list(cls, value: Any) -> Sequence[str]:
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if not value:
                raise ValueError("sequences must not be empty")
            return [str(item) for item in value]
        raise TypeError("expected a sequence of strings")

    def fingerprint(self) -> str:
        """Return a stable hash for the hypothesis content."""
        payload = self.model_dump()
        return compute_sha256(_stable_dumps(payload))

    def procedural_text(self) -> str:
        """Return validation steps as a single string for safety filtering."""
        return "\n".join(step.strip() for step in self.validation_steps if step.strip())


class ValidationRun(BaseModel):
    """Outcome of a single deterministic validation execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    test_case_hash: str = Field(..., min_length=64, max_length=64)
    success: bool
    latency_ms: float = Field(default=0.0, ge=0.0)
    log_path: str | None = Field(default=None, max_length=256)


class ValidationResult(BaseModel):
    """Aggregated validation metrics for a hypothesis."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis_hash: str = Field(..., min_length=64, max_length=64)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    reproducible: bool = Field(default=False)
    runs: Sequence[ValidationRun] = Field(default_factory=list)
    proof_artifact: str | None = Field(default=None, max_length=256)

    @field_validator("runs", mode="before")
    @classmethod
    def _ensure_runs(cls, value: Any) -> Sequence[ValidationRun]:
        if isinstance(value, Sequence):
            return list(value)
        raise TypeError("runs must be a sequence")


class ValidatedRule(BaseModel):
    """A hypothesis that passed validation and is ready for promotion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: ActionHypothesis
    validation: ValidationResult
    promoted_at: str | None = Field(default=None, max_length=32)

    def fingerprint(self) -> str:
        """Hash the hypothesis and validation data for audit purposes."""
        payload = {
            "hypothesis": self.hypothesis.model_dump(),
            "validation": self.validation.model_dump(),
            "promoted_at": self.promoted_at,
        }
        return compute_sha256(_stable_dumps(payload))


class AppliedRuleRecord(BaseModel):
    """Serializable representation of an applied navigation rule."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: str = Field(..., min_length=1, max_length=128)
    from_room: str | None = Field(default=None, max_length=64)
    to_room: str | None = Field(default=None, max_length=64)
    required_key: str | None = Field(default=None, max_length=64)
    unlock_targets: Sequence[str] = Field(default_factory=tuple)
    reward: float = Field(default=0.0)
    hypothesis_hash: str = Field(..., min_length=64, max_length=64)
    validation_success_rate: float = Field(..., ge=0.0, le=1.0)
    proof_artifact: str | None = Field(default=None, max_length=256)

    @field_validator("unlock_targets", mode="before")
    @classmethod
    def _ensure_unlock_sequence(cls, value: Any) -> Sequence[str]:
        if value is None:
            return tuple()
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(str(item) for item in value)
        raise TypeError("unlock_targets must be a sequence of strings")


class WorldModelSnapshot(BaseModel):
    """Snapshot of the navigation world model following a bootstrap run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: str = Field(default="2.0.0", min_length=1, max_length=16)
    base_model: str = Field(default="SimpleWorldModel", min_length=1, max_length=64)
    recorded_at: str = Field(..., min_length=1, max_length=32)
    batch_timestamp: str = Field(..., min_length=1, max_length=32)
    batch_directory: str | None = Field(default=None, max_length=256)
    prompt_hash: str = Field(..., min_length=64, max_length=64)
    response_hash: str = Field(..., min_length=64, max_length=64)
    domain_fingerprint: str | None = Field(default=None, max_length=128)
    hypothesis_count: int = Field(..., ge=1)
    domain: DomainSnapshot
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    validated_rules: Sequence[ValidatedRule] = Field(default_factory=tuple)
    applied_rules: Sequence[AppliedRuleRecord] = Field(default_factory=tuple)

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata_mapping(cls, value: Any) -> Mapping[str, Any]:
        if value is None:
            return {}
        if isinstance(value, Mapping):
            return dict(value)
        raise TypeError("metadata must be a mapping")


class HypothesisBatch(BaseModel):
    """Container for prompts, responses, and parsed hypotheses."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prompt: str = Field(..., min_length=1)
    prompt_hash: str = Field(..., min_length=64, max_length=64)
    response: str = Field(..., min_length=1)
    response_hash: str = Field(..., min_length=64, max_length=64)
    hypotheses: tuple[ActionHypothesis, ...] = Field(..., min_length=1)
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @classmethod
    def build(cls, *, prompt: str, response: str, hypotheses: Iterable[ActionHypothesis], metadata: Mapping[str, Any] | None = None) -> HypothesisBatch:
        hyp_tuple = tuple(hypotheses)
        if not hyp_tuple:
            raise ValueError("hypotheses must not be empty")
        prompt_hash = compute_sha256(prompt)
        response_hash = compute_sha256(response)
        return cls(
            prompt=prompt,
            prompt_hash=prompt_hash,
            response=response,
            response_hash=response_hash,
            hypotheses=hyp_tuple,
            metadata=dict(metadata or {}),
        )


__all__ = [
    "ActionHypothesis",
    "AppliedRuleRecord",
    "DomainSnapshot",
    "HypothesisBatch",
    "TestCase",
    "TransitionObservation",
    "ValidationResult",
    "ValidationRun",
    "ValidatedRule",
    "WorldModelSnapshot",
    "compute_sha256",
]
