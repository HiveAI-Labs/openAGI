"""World-model bootstrapping (schemas, LLM teacher, registry, validation)."""

from __future__ import annotations

from .builder import (
    BootstrapError,
    BootstrapParseError,
    BootstrapSafetyError,
    BootstrapTimeoutError,
    LLMWorldModelBuilder,
)
from .insight_provider import EnsembleHypothesisProvider
from .orchestrator import (
    AppliedRuleSummary,
    NavigationBootstrapOrchestrator,
    apply_validated_rules,
    build_navigation_snapshot,
)
from .registry import BatchHandle, HypothesisRegistry
from .schemas import (
    ActionHypothesis,
    AppliedRuleRecord,
    DomainSnapshot,
    HypothesisBatch,
    TestCase,
    TransitionObservation,
    ValidatedRule,
    ValidationResult,
    ValidationRun,
    WorldModelSnapshot,
    compute_sha256,
)
from .validation import ValidationSummary, validate_navigation_batch
from .validator import NavigationValidator

__all__ = [
    "ActionHypothesis",
    "AppliedRuleRecord",
    "AppliedRuleSummary",
    "BatchHandle",
    "BootstrapError",
    "BootstrapParseError",
    "BootstrapSafetyError",
    "BootstrapTimeoutError",
    "DomainSnapshot",
    "HypothesisBatch",
    "HypothesisRegistry",
    "LLMWorldModelBuilder",
    "NavigationBootstrapOrchestrator",
    "NavigationValidator",
    "TestCase",
    "TransitionObservation",
    "ValidatedRule",
    "ValidationResult",
    "ValidationRun",
    "ValidationSummary",
    "WorldModelSnapshot",
    "EnsembleHypothesisProvider",
    "apply_validated_rules",
    "build_navigation_snapshot",
    "compute_sha256",
    "validate_navigation_batch",
]
