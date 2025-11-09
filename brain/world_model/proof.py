"""Proof helpers for world-model retraining validation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from brain.io.atomic import atomic_write_json


def _stable_dump(payload: Mapping[str, Any]) -> str:
    """Return a deterministic JSON serialization of *payload*."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _compact_stats(stats: Mapping[str, Any]) -> Mapping[str, Any]:
    """Extract a lightweight summary from the full retraining stats."""
    summary: dict[str, Any] = {
        "samples": stats.get("samples"),
        "states": stats.get("states"),
        "top1_accuracy": stats.get("top1_accuracy"),
        "episodes": stats.get("episodes"),
        "avg_reward": stats.get("avg_reward"),
    }
    per_mode = stats.get("per_mode")
    if isinstance(per_mode, Mapping):
        summary["per_mode"] = {
            str(mode): {
                "samples": value.get("samples"),
                "top1_accuracy": value.get("top1_accuracy"),
                "success_rate": value.get("success_rate"),
                "avg_steps": value.get("avg_steps"),
                "avg_reward": value.get("avg_reward"),
            }
            for mode, value in per_mode.items()
            if isinstance(value, Mapping)
        }
    v2_stats = stats.get("v2")
    if isinstance(v2_stats, Mapping):
        summary["v2"] = {
            "policy_entries": v2_stats.get("policy_entries"),
            "transition_entries": v2_stats.get("transition_entries"),
            "alpha": v2_stats.get("alpha"),
        }
    neural_stats = stats.get("neural")
    if isinstance(neural_stats, Mapping):
        summary["neural"] = {
            "samples": neural_stats.get("samples"),
            "top1_accuracy": neural_stats.get("top1_accuracy"),
            "train_top1_accuracy": neural_stats.get("train_top1_accuracy"),
            "test_top1_accuracy": neural_stats.get("test_top1_accuracy"),
        }
    return summary


def _compact_validation_stats(stats: Mapping[str, Any]) -> Mapping[str, Any]:
    """Project world-model validation stats into a deterministic summary."""
    summary: dict[str, Any] = {
        "samples": stats.get("samples"),
        "matches": stats.get("matches"),
        "match_rate": stats.get("match_rate"),
        "avg_reward_delta": stats.get("avg_reward_delta"),
        "avg_abs_reward_delta": stats.get("avg_abs_reward_delta"),
        "reward_abs_p95": stats.get("reward_abs_p95"),
        "latency_avg_ms": stats.get("latency_avg_ms"),
        "latency_p95_ms": stats.get("latency_p95_ms"),
        "latency_p99_ms": stats.get("latency_p99_ms"),
        "avg_confidence": stats.get("avg_confidence"),
        "brier_score": stats.get("brier_score"),
        "negative_log_likelihood": stats.get("negative_log_likelihood"),
        "reward_rmse": stats.get("reward_rmse"),
    }
    per_action = stats.get("per_action") if isinstance(stats.get("per_action"), Mapping) else None
    if isinstance(per_action, Mapping):
        compact: dict[str, Any] = {}
        for action, payload in per_action.items():
            if not isinstance(payload, Mapping):
                continue
            compact[str(action)] = {
                "samples": payload.get("samples"),
                "match_rate": payload.get("match_rate"),
                "avg_reward_delta": payload.get("avg_reward_delta"),
                "avg_abs_reward_delta": payload.get("avg_abs_reward_delta"),
            }
        summary["per_action"] = compact
    return summary


def write_retrain_validation_proof(
    *,
    artifacts_dir: Path,
    workspace: str,
    stats: Mapping[str, Any],
    validation: Mapping[str, Any],
) -> Path:
    """Persist a proof record tying retraining stats to validation outcomes."""
    artifacts = Path(artifacts_dir)
    proof_dir = artifacts / "proof" / "retrain_validation"
    proof_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    ts_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    # Build deterministic digests for traceability
    stats_text = _stable_dump(stats)
    validation_text = _stable_dump(validation)
    stats_digest = "sha256:" + hashlib.sha256(stats_text.encode("utf-8")).hexdigest()
    validation_digest = "sha256:" + hashlib.sha256(validation_text.encode("utf-8")).hexdigest()

    payload: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "workspace": workspace,
        "stats_summary": _compact_stats(stats),
        "stats_digest": stats_digest,
        "validation": dict(validation),
        "validation_digest": validation_digest,
    }

    out_path = proof_dir / f"retrain_validation_{ts_token}.json"
    atomic_write_json(out_path, payload)
    return out_path


def write_world_model_validation_proof(
    *,
    artifacts_dir: Path,
    workspace: str,
    stats: Mapping[str, Any],
    validation: Mapping[str, Any],
    summary: Mapping[str, Any],
    summary_digest: str,
    stress_summary: Mapping[str, Any] | None = None,
    proof_subdir: str = "world_model_validation",
    domains: Mapping[str, Mapping[str, Any]] | None = None,
) -> Path:
    """Persist a proof record for world-model validation runs."""
    artifacts = Path(artifacts_dir)
    proof_dir = artifacts / "proof" / proof_subdir
    proof_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC)
    ts_token = timestamp.strftime("%Y%m%dT%H%M%S%fZ")

    stats_text = _stable_dump(stats)
    validation_text = _stable_dump(validation)
    summary_text = _stable_dump(summary)

    stats_digest = "sha256:" + hashlib.sha256(stats_text.encode("utf-8")).hexdigest()
    validation_digest = "sha256:" + hashlib.sha256(validation_text.encode("utf-8")).hexdigest()

    payload: dict[str, Any] = {
        "timestamp": timestamp.isoformat(),
        "workspace": workspace,
        "stats_summary": _compact_validation_stats(stats),
        "stats_digest": stats_digest,
        "validation": dict(validation),
        "validation_digest": validation_digest,
        "summary": dict(summary),
        "summary_digest": summary_digest,
        "summary_compact_digest": "sha256:" + hashlib.sha256(summary_text.encode("utf-8")).hexdigest(),
    }
    if stress_summary:
        payload["stress"] = dict(stress_summary)
    if domains:
        payload["domains"] = {name: dict(value) for name, value in domains.items()}

    out_path = proof_dir / f"world_model_validation_{ts_token}.json"
    atomic_write_json(out_path, payload)
    return out_path


__all__ = ["write_retrain_validation_proof", "write_world_model_validation_proof"]
