"""Artifact registry for LLM-derived world model hypotheses."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableSequence, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import threading

from brain.io.atomic import atomic_write_json, atomic_write_json_lines, atomic_write_text
from brain.obs.metrics import (
    brain_wm_hypotheses_total,
    brain_wm_refinement_events_total,
    brain_wm_refinement_queue_size,
    brain_wm_retrain_events_total,
    brain_wm_retrain_queue_size,
    brain_wm_snapshots_total,
)

from .schemas import (
    ActionHypothesis,
    AppliedRuleRecord,
    DomainSnapshot,
    HypothesisBatch,
    TestCase,
    ValidatedRule,
    ValidationResult,
    WorldModelSnapshot,
    compute_sha256,
)

_TIME_FORMAT = "%Y%m%dT%H%M%S%fZ"


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _stable_dumps(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class BatchHandle:
    """Handle describing where a hypothesis batch is stored."""

    batch: HypothesisBatch
    directory: Path
    timestamp: str

    @property
    def hypotheses(self) -> tuple[ActionHypothesis, ...]:
        return self.batch.hypotheses


class HypothesisRegistry:
    """Persist LLM hypotheses and validation runs under artifacts/world_model."""

    def __init__(
        self,
        *,
        artifacts_dir: str | os.PathLike[str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        base_dir = Path(artifacts_dir or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts").resolve()
        self._artifact_root = base_dir
        self._base_dir = base_dir / "world_model" / "bootstrap"
        self._clock = clock or _default_clock
        self._index_path = self._base_dir / "index.jsonl"
        self._lock = threading.Lock()
        self._refinement_queue_path = self._base_dir / "refinement_queue.jsonl"
        self._retrain_queue_path = self._base_dir / "retrain_queue.jsonl"
        self._refinement_queue_limit = 200
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def artifact_root(self) -> Path:
        return self._artifact_root

    def record_batch(self, batch: HypothesisBatch) -> BatchHandle:
        """Write prompt/response artifacts for *batch* and return a handle."""
        timestamp = self._clock().strftime(_TIME_FORMAT)
        slug = f"{timestamp}_{batch.response_hash[:12]}"
        batch_dir = self._base_dir / timestamp[:8] / slug
        batch_dir.mkdir(parents=True, exist_ok=True)

        atomic_write_text(batch_dir / "prompt.txt", batch.prompt)
        atomic_write_text(batch_dir / "response.json", batch.response)

        hypotheses_payload = [hyp.model_dump(mode="json") for hyp in batch.hypotheses]
        atomic_write_json_lines(batch_dir / "hypotheses.jsonl", hypotheses_payload)

        ensemble_hashes = set()
        meta_hashes = batch.metadata.get("ensemble_hashes")
        if isinstance(meta_hashes, (list, tuple, set)):
            ensemble_hashes = {str(value) for value in meta_hashes}

        metadata_payload = {
            "prompt_hash": batch.prompt_hash,
            "response_hash": batch.response_hash,
            "stored_at": timestamp,
            "metadata": dict(batch.metadata),
            "hypotheses": [
                {
                    "hash": hyp.fingerprint(),
                    "action": hyp.action,
                    "confidence": hyp.confidence,
                    "preconditions": list(hyp.preconditions),
                    "effects": list(hyp.effects),
                    "source": "ensemble" if hyp.fingerprint() in ensemble_hashes else "llm",
                }
                for hyp in batch.hypotheses
            ],
        }
        atomic_write_json(batch_dir / "metadata.json", metadata_payload)

        index_record = {
            "timestamp": timestamp,
            "prompt_hash": batch.prompt_hash,
            "response_hash": batch.response_hash,
            "count": len(batch.hypotheses),
            "domain_fingerprint": batch.metadata.get("domain_fingerprint"),
            "path": str(self._relative_path(batch_dir)),
        }
        if ensemble_hashes:
            index_record["ensemble_hypotheses"] = len(ensemble_hashes)
        self._append_index(index_record)

        brain_wm_hypotheses_total.inc(amount=len(batch.hypotheses), status="proposed")

        return BatchHandle(batch=batch, directory=batch_dir, timestamp=timestamp)

    def record_validation_run(
        self,
        handle: BatchHandle,
        hypothesis: ActionHypothesis,
        case: TestCase,
        *,
        success: bool,
        latency_ms: float,
        details: Mapping[str, object],
    ) -> tuple[str, str]:
        """Persist a single validation run and return (test_case_hash, log_path)."""
        hypothesis_hash = hypothesis.fingerprint()
        case_payload = case.model_dump(mode="json")
        case_hash = compute_sha256(_stable_dumps(case_payload))

        run_dir = handle.directory / "validation" / hypothesis_hash
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "recorded_at": self._clock().strftime(_TIME_FORMAT),
            "hypothesis_hash": hypothesis_hash,
            "test_case_hash": case_hash,
            "success": bool(success),
            "latency_ms": float(latency_ms),
            "seed": case.seed,
            "case": case_payload,
            "details": dict(details),
        }
        log_path = run_dir / f"{case_hash}.json"
        atomic_write_json(log_path, payload)

        rel_path = str(self._relative_path(log_path))
        return case_hash, rel_path

    def record_validation_summary(self, handle: BatchHandle, result: ValidationResult) -> str:
        """Persist aggregated validation metrics and return the summary path."""
        summary_dir = handle.directory / "validation"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"{result.hypothesis_hash}_summary.json"
        payload = {
            "recorded_at": self._clock().strftime(_TIME_FORMAT),
            "result": result.model_dump(mode="json"),
        }
        atomic_write_json(summary_path, payload)
        return str(self._relative_path(summary_path))

    def record_world_model_snapshot(
        self,
        handle: BatchHandle,
        *,
        snapshot: DomainSnapshot,
        batch: HypothesisBatch,
        validated: Sequence[ValidatedRule],
        applied_rules: Sequence[Mapping[str, object]],
        base_model: str,
        version: str = "2.0.0",
        approval_required: bool = False,
    ) -> str:
        """Persist a snapshot of the world model following bootstrap."""
        recorded_at = self._clock().strftime(_TIME_FORMAT)
        ensemble_hashes: set[str] = set()
        meta_hashes = batch.metadata.get("ensemble_hashes")
        if isinstance(meta_hashes, (list, tuple, set)):
            ensemble_hashes = {str(value) for value in meta_hashes}
        applied_payload = tuple(
            AppliedRuleRecord.model_validate(rule)
            for rule in applied_rules
        )
        fingerprint = batch.metadata.get("domain_fingerprint")
        fingerprint_str = None if fingerprint is None else str(fingerprint)
        snapshot_payload = WorldModelSnapshot(
            version=version,
            base_model=base_model,
            recorded_at=recorded_at,
            batch_timestamp=handle.timestamp,
            batch_directory=str(self._relative_path(handle.directory)),
            prompt_hash=batch.prompt_hash,
            response_hash=batch.response_hash,
            domain_fingerprint=fingerprint_str,
            hypothesis_count=len(batch.hypotheses),
            domain=snapshot,
            metadata=dict(batch.metadata),
            validated_rules=tuple(validated),
            applied_rules=applied_payload,
        )

        target_path = handle.directory / "snapshot.json"
        atomic_write_json(target_path, snapshot_payload.model_dump(mode="json"))

        queue_record = {
            "recorded_at": recorded_at,
            "batch_timestamp": handle.timestamp,
            "snapshot_path": str(self._relative_path(target_path)),
            "prompt_hash": batch.prompt_hash,
            "validated_rules": [item.validation.hypothesis_hash for item in validated],
            "applied_count": len(applied_payload),
            "approval_required": bool(approval_required),
        }
        ensemble_count = int(batch.metadata.get("ensemble_hypotheses") or 0)
        if ensemble_count:
            queue_record["ensemble_hypotheses"] = ensemble_count
        queue_length = self._append_refinement_queue(queue_record)

        brain_wm_snapshots_total.inc(status="recorded")
        brain_wm_refinement_events_total.inc(event="snapshot_recorded")
        try:
            brain_wm_refinement_queue_size.set(queue_length)
        except Exception:
            pass

        if ensemble_hashes:
            promoted = []
            for rule in validated:
                if rule.hypothesis.fingerprint() in ensemble_hashes:
                    promoted.append(
                        {
                            "hypothesis_hash": rule.validation.hypothesis_hash,
                            "action": rule.hypothesis.action,
                            "success_rate": rule.validation.success_rate,
                        },
                    )
            if promoted:
                retrain_record = {
                    "recorded_at": recorded_at,
                    "snapshot_path": str(self._relative_path(target_path)),
                    "prompt_hash": batch.prompt_hash,
                    "batch_timestamp": handle.timestamp,
                    "ensemble_promotions": promoted,
                }
                self._append_retrain_queue(retrain_record)

        return str(self._relative_path(target_path))

    def record_curiosity_task(
        self,
        *,
        domain: str,
        description: str,
        urgency: float,
        learnability: float,
        source: str = "curiosity_engine",
        metadata: Mapping[str, object] | None = None,
    ) -> int:
        """Append a curiosity-driven learning task to the refinement queue."""
        recorded_at = self._clock().strftime(_TIME_FORMAT)
        queue_record: dict[str, object] = {
            "recorded_at": recorded_at,
            "task_type": "curiosity_gap",
            "domain": domain,
            "description": description,
            "urgency": max(0.0, min(1.0, float(urgency))),
            "learnability": max(0.0, min(1.0, float(learnability))),
            "source": source,
        }
        if metadata:
            queue_record["metadata"] = dict(metadata)
        queue_length = self._append_refinement_queue(queue_record)
        try:
            brain_wm_refinement_events_total.inc(event="curiosity_task_recorded")
            brain_wm_refinement_queue_size.set(queue_length)
        except Exception:
            pass
        return queue_length

    def status(self) -> dict[str, object]:
        """Return a summary of registry and refinement queue state."""
        retrain_queue: list[dict[str, object]] = []
        retrain_latest: Optional[dict[str, object]] = None
        with self._lock:
            queue = self._load_refinement_queue()
            latest = dict(queue[-1]) if queue else None
            index_entries = len(queue)
            if self._index_path.exists():
                try:
                    raw = self._index_path.read_text(encoding="utf-8")
                    lines = [ln for ln in raw.splitlines() if ln.strip()]
                    index_entries = len(lines)
                except Exception:
                    pass

            retrain_queue = self._load_retrain_queue()
            if retrain_queue:
                retrain_latest = dict(retrain_queue[-1])

        summary: dict[str, object] = {
            "base_dir": str(self._relative_path(self._base_dir)),
            "index_path": str(self._relative_path(self._index_path)),
            "index_entries": index_entries,
            "refinement_queue_length": len(queue),
        }
        if latest:
            summary["latest_snapshot"] = latest
        summary["retrain_queue_length"] = len(retrain_queue)
        if retrain_latest:
            summary["latest_retrain"] = retrain_latest
        return summary

    def _append_index(self, record: Mapping[str, object]) -> None:
        with self._lock:
            existing: MutableSequence[Mapping[str, object]] = []
            if self._index_path.exists():
                raw = self._index_path.read_text(encoding="utf-8")
                if raw.strip():
                    existing = [json.loads(line) for line in raw.splitlines() if line.strip()]
            existing.append(dict(record))
            atomic_write_json_lines(self._index_path, existing)  # type: ignore[arg-type]

    def _relative_path(self, path: Path) -> Path:
        try:
            return Path(path.resolve().relative_to(self._artifact_root))
        except ValueError:
            return path.resolve()

    def _load_refinement_queue(self) -> list[dict[str, object]]:
        if not self._refinement_queue_path.exists():
            return []
        try:
            raw = self._refinement_queue_path.read_text(encoding="utf-8")
        except Exception:
            return []
        if not raw.strip():
            return []
        entries: list[dict[str, object]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _append_refinement_queue(self, record: Mapping[str, object]) -> int:
        with self._lock:
            queue = self._load_refinement_queue()
            queue.append(dict(record))
            if len(queue) > self._refinement_queue_limit:
                queue = queue[-self._refinement_queue_limit :]
            atomic_write_json_lines(self._refinement_queue_path, queue)  # type: ignore[arg-type]
            return len(queue)

    def _load_retrain_queue(self) -> list[dict[str, object]]:
        if not self._retrain_queue_path.exists():
            return []
        try:
            raw = self._retrain_queue_path.read_text(encoding="utf-8")
        except Exception:
            return []
        if not raw.strip():
            return []
        entries: list[dict[str, object]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                entries.append(payload)
        return entries

    def _append_retrain_queue(self, record: Mapping[str, object]) -> int:
        with self._lock:
            queue = self._load_retrain_queue()
            queue.append(dict(record))
            atomic_write_json_lines(self._retrain_queue_path, queue)  # type: ignore[arg-type]
            try:
                brain_wm_retrain_events_total.inc(event="queued")
                brain_wm_retrain_queue_size.set(len(queue))
            except Exception:
                pass
            return len(queue)


__all__ = ["BatchHandle", "HypothesisRegistry"]
