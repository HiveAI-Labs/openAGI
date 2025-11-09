"""Coordinator for running LLM-assisted navigation bootstrap."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import time
from typing import Any

from brain.obs.metrics import (
    brain_wm_bootstrap_latency_ms,
    brain_wm_bootstrap_runs_total,
    brain_wm_refinement_events_total,
)
from brain.obs.tracing import global_tracer

from ..simple_model import NavigationRule, SimpleState, SimpleWorldModel
from .builder import LLMWorldModelBuilder
from .persistence import load_snapshot, save_snapshot
from .registry import HypothesisRegistry
from .scheduler import RefinementScheduler
from .schemas import (
    ActionHypothesis,
    AppliedRuleRecord,
    DomainSnapshot,
    TransitionObservation,
    ValidatedRule,
)
from .validator import NavigationValidator

try:
    from brain.arm_bandit import reward_ex as _bandit_reward_ex  # reuse existing bandit
except Exception:  # pragma: no cover
    _bandit_reward_ex = None  # type: ignore


@dataclass(frozen=True)
class AppliedRuleSummary:
    """Summary of a rule applied to the runtime world model."""

    action: str
    from_room: str | None
    to_room: str | None
    required_key: str | None
    unlock_targets: tuple[str, ...]
    reward: float
    hypothesis_hash: str
    validation_success_rate: float
    proof_artifact: str | None


def _parse_entries(entries: Sequence[str]) -> dict[str, list[str]]:
    parsed: dict[str, list[str]] = {}
    for entry in entries:
        if not isinstance(entry, str):
            continue
        token = entry.strip()
        if not token:
            continue
        if ":" in token:
            key, value = token.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
        else:
            key = token.strip().lower()
            value = ""
        if not key:
            continue
        parsed.setdefault(key, []).append(value)
    return parsed


def _infer_navigation_details(hypothesis: ActionHypothesis) -> tuple[str | None, str | None, str | None, tuple[str, ...]]:
    """Derive navigation metadata from hypothesis content and tests."""
    prec = _parse_entries(hypothesis.preconditions)
    eff = _parse_entries(hypothesis.effects)

    from_room: str | None = None
    to_room: str | None = None

    for field in ("room", "location"):
        values = prec.get(field)
        if values:
            from_room = values[0]
            break

    for field in ("room", "location"):
        values = eff.get(field)
        if values:
            to_room = values[0]
            break

    required_key: str | None = None
    for field in ("has_key", "holding", "inventory"):
        values = prec.get(field)
        if values:
            required_key = values[0]
            break

    unlock_targets: list[str] = []
    for field in ("unlocked", "unlock", "open"):
        unlock_targets.extend(eff.get(field, []))

    if to_room is None:
        for case in hypothesis.test_cases:
            value = case.expected_outcome.get("room") or case.expected_outcome.get("location")
            if value:
                to_room = str(value)
                break

    if from_room is None:
        for case in hypothesis.test_cases:
            value = case.initial_state.get("room") or case.initial_state.get("location")
            if value:
                from_room = str(value)
                break

    return from_room, to_room, required_key, tuple(unlock_targets)


def apply_validated_rules(
    world_model: SimpleWorldModel,
    rules: Sequence[ValidatedRule],
) -> list[AppliedRuleSummary]:
    """Promote validated rules into the supplied world model."""
    applied: list[AppliedRuleSummary] = []
    for rule in rules:
        from_room, to_room, required_key, unlock_targets = _infer_navigation_details(rule.hypothesis)
        navigation_rule: NavigationRule = world_model.register_navigation_rule(
            action=rule.hypothesis.action,
            from_room=from_room,
            to_room=to_room,
            required_key=required_key,
            unlock_targets=unlock_targets,
        )
        applied.append(
            AppliedRuleSummary(
                action=navigation_rule.action,
                from_room=navigation_rule.from_room,
                to_room=navigation_rule.to_room,
                required_key=navigation_rule.required_key,
                unlock_targets=navigation_rule.unlock_targets,
                reward=navigation_rule.reward,
                hypothesis_hash=rule.validation.hypothesis_hash,
                validation_success_rate=rule.validation.success_rate,
                proof_artifact=rule.validation.proof_artifact,
            ),
        )
    return applied


def build_navigation_snapshot(world_model: SimpleWorldModel) -> DomainSnapshot:
    """Create a deterministic snapshot of the navigation domain."""
    room_set = set(world_model.connections.keys())
    for neighbors in world_model.connections.values():
        room_set.update(neighbors)
    rooms = sorted(room_set)

    objects = sorted(world_model.key_locations.keys())

    transitions: list[TransitionObservation] = []
    seen_edges: set[tuple[str, str]] = set()
    for source, neighbors in world_model.connections.items():
        for target in neighbors:
            edge = (source, target)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            transitions.append(
                TransitionObservation(
                    state=str(SimpleState(location=source)),
                    action=f"navigate {target}",
                    next_state=str(SimpleState(location=target)),
                    success=True,
                    notes="bootstrap_snapshot",
                ),
            )

    constraints: dict[str, Any] = {
        "locked_doors": [
            {"from": src, "to": dst, "required_key": key}
            for (src, dst), key in world_model.locked_doors.items()
        ],
        "bootstrap_aliases": world_model.bootstrap_aliases(),
    }

    return DomainSnapshot(
        rooms=rooms,
        objects=objects,
        transitions=transitions,
        constraints=constraints,
    )


class NavigationBootstrapOrchestrator:
    """High-level coordinator for bootstrap: build -> validate -> integrate."""

    def __init__(
        self,
        *,
        world_model: SimpleWorldModel,
        builder: LLMWorldModelBuilder,
        validator: NavigationValidator,
        registry: HypothesisRegistry,
        require_manual_approval: bool = False,
        snapshot_version: str = "2.0.0",
    ) -> None:
        self._world_model = world_model
        self._builder = builder
        self._validator = validator
        self._registry = registry
        self._require_manual_approval = bool(require_manual_approval)
        self._snapshot_version = snapshot_version
        self._base_model_name = type(world_model).__name__

    def build_snapshot(self) -> DomainSnapshot:
        """Return the default domain snapshot for the associated world model."""
        return build_navigation_snapshot(self._world_model)

    def run(self, *, snapshot_override: DomainSnapshot | None = None) -> dict[str, Any]:
        """Execute the bootstrap pipeline and apply resulting rules."""
        snapshot = snapshot_override or self.build_snapshot()
        tracer = global_tracer()
        start = time.perf_counter()

        with tracer.span("world_model.bootstrap") as span:
            span.tags["snapshot.rooms"] = len(snapshot.rooms)
            span.tags["snapshot.objects"] = len(snapshot.objects)

            batch = self._builder.propose_navigation_rules(snapshot)
            span.tags["hypotheses.proposed"] = len(batch.hypotheses)

            validated = self._validator.validate(batch)
            span.tags["hypotheses.validated"] = len(validated)

            if self._require_manual_approval:
                applied = []
                span.tags["rules.applied"] = 0
            else:
                applied = apply_validated_rules(self._world_model, validated)
                span.tags["rules.applied"] = len(applied)
            applied_payload = [asdict(entry) for entry in applied]

        duration_ms = (time.perf_counter() - start) * 1000.0
        outcome = "applied" if applied else ("validated" if validated else "none")
        brain_wm_bootstrap_latency_ms.observe(duration_ms)
        brain_wm_bootstrap_runs_total.inc(outcome=outcome)

        handle = self._validator.last_batch_handle
        snapshot_path = None
        if handle is not None:
            snapshot_path = self._registry.record_world_model_snapshot(
                handle,
                snapshot=snapshot,
                batch=batch,
                validated=validated,
                applied_rules=applied_payload,
                base_model=self._base_model_name,
                version=self._snapshot_version,
                approval_required=self._require_manual_approval,
            )

        # Optional: update existing bandit with fractional reward
        try:
            use_bandit = str(os.getenv("WM_PROMPT_USE_BANDIT", "0")).strip().lower() in {"1", "true", "yes", "on"}
        except Exception:
            use_bandit = False
        if use_bandit and _bandit_reward_ex is not None:
            try:
                tmpl = str(batch.metadata.get("prompt_template_id") or "base")
                total = max(1, int(batch.metadata.get("hypothesis_count") or len(batch.hypotheses)))
                reward = float(len(validated)) / float(total)
                _bandit_reward_ex("wm_prompt_enrichment", tmpl, reward)
            except Exception:
                pass

        registry_status = self._registry.status()
        return {
            "prompt_hash": batch.prompt_hash,
            "response_hash": batch.response_hash,
            "metadata": dict(batch.metadata),
            "validated_rules": [
                {
                    "action": item.hypothesis.action,
                    "hypothesis_hash": item.validation.hypothesis_hash,
                    "success_rate": item.validation.success_rate,
                    "reproducible": item.validation.reproducible,
                    "proof_artifact": item.validation.proof_artifact,
                }
                for item in validated
            ],
            "applied_rules": applied_payload,
            "batch_directory": str(handle.directory) if handle else None,
            "registry_base_dir": str(self._registry.base_dir),
            "snapshot_path": snapshot_path,
            "registry_status": registry_status,
            "duration_ms": duration_ms,
        }

    def status(self) -> dict[str, Any]:
        """Return current bootstrap registry status and last validation results."""
        handle = self._validator.last_batch_handle
        last_batch_timestamp = handle.timestamp if handle else None
        return {
            "registry": self._registry.status(),
            "last_batch_timestamp": last_batch_timestamp,
            "last_validation_results": {
                "count": len(self._validator.last_results),
                "hypotheses": list(self._validator.last_results.keys()),
            },
        }

    def approve_pending_rules(self, *, snapshot_path: str | None = None) -> dict[str, Any]:
        """Apply validated rules from a snapshot once manual approval is granted."""
        scheduler = RefinementScheduler(self._registry)
        queue_entry: dict[str, Any] | None = None
        resolved_absolute: Path | None = None
        queue_key: str | None = None

        if snapshot_path:
            resolved_absolute, queue_key = self._resolve_snapshot_path(snapshot_path)
            queue_entry = (
                scheduler.pop_by_snapshot(queue_key)
                or scheduler.pop_by_snapshot(str(resolved_absolute))
            )
        else:
            queue = scheduler.get_queue()
            for candidate in queue:
                if candidate.get("approval_required"):
                    queue_key = str(candidate.get("snapshot_path"))
                    queue_entry = scheduler.pop_by_snapshot(queue_key)
                    break
            if queue_key is None:
                return {
                    "ok": False,
                    "error": "no_pending_approvals",
                    "queue_length": len(queue),
                }
            if queue_entry is None:
                return {
                    "ok": False,
                    "error": "approval_entry_conflict",
                    "snapshot_path": queue_key,
                }
            resolved_absolute, queue_key = self._resolve_snapshot_path(queue_key or "")

        if resolved_absolute is None or queue_key is None:
            return {"ok": False, "error": "invalid_snapshot_path"}

        if not resolved_absolute.exists():
            return {
                "ok": False,
                "error": "snapshot_missing",
                "snapshot_path": str(resolved_absolute),
            }

        snapshot, _ = load_snapshot(resolved_absolute)

        existing_hashes = {record.hypothesis_hash for record in snapshot.applied_rules}
        pending_rules = [
            rule
            for rule in snapshot.validated_rules
            if rule.validation.hypothesis_hash not in existing_hashes
        ]
        if not pending_rules:
            return {
                "ok": False,
                "error": "no_pending_rules",
                "snapshot_path": str(resolved_absolute),
            }

        applied_summaries = apply_validated_rules(self._world_model, pending_rules)
        applied_records = tuple(
            AppliedRuleRecord.model_validate(asdict(summary))
            for summary in applied_summaries
        )

        metadata = dict(snapshot.metadata)
        approvals = list(metadata.get("approvals", []))
        approvals.append(
            {
                "approved_at": datetime.now(UTC).isoformat(),
                "applied_count": len(applied_records),
                "snapshot_path": str(queue_key),
            },
        )
        metadata["approvals"] = approvals

        updated_snapshot = snapshot.model_copy(
            update={
                "applied_rules": tuple(snapshot.applied_rules) + applied_records,
                "metadata": metadata,
            },
        )
        sha = save_snapshot(updated_snapshot, resolved_absolute)

        try:
            brain_wm_refinement_events_total.inc(event="approved")
        except Exception:
            pass

        registry_status = self._registry.status()
        return {
            "ok": True,
            "snapshot_path": str(resolved_absolute),
            "applied_rules": [asdict(summary) for summary in applied_summaries],
            "applied_count": len(applied_summaries),
            "snapshot_sha256": sha,
            "queue_entry_removed": queue_entry is not None,
            "queue_remaining": registry_status.get("refinement_queue_length"),
            "approvals": approvals,
        }

    def _resolve_snapshot_path(self, snapshot_path: str) -> tuple[Path, str]:
        raw = snapshot_path.strip()
        if not raw:
            raise ValueError("snapshot_path must be non-empty")
        candidate = Path(raw)
        if candidate.is_absolute():
            absolute = candidate
            try:
                relative = str(absolute.resolve().relative_to(self._registry.artifact_root))
            except ValueError:
                relative = str(absolute.resolve())
        else:
            relative = str(candidate)
            absolute = (self._registry.artifact_root / candidate).resolve()
        return absolute, relative
