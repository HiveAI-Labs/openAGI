"""Helpers for loading LLM ensemble inventory metadata."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCapability:
    """Structured capability data for a local LLM backend."""

    latency_ms_p95: float | None = None
    latency_ms_p50: float | None = None
    cost_per_1k_tokens_usd: float | None = None
    quality_score: float | None = None
    safety_tier: str | None = None
    max_context_tokens: int | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the capability record."""
        return {
            "latency_ms_p95": self.latency_ms_p95,
            "latency_ms_p50": self.latency_ms_p50,
            "cost_per_1k_tokens_usd": self.cost_per_1k_tokens_usd,
            "quality_score": self.quality_score,
            "safety_tier": self.safety_tier,
            "max_context_tokens": self.max_context_tokens,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ModelRecord:
    """Inventory entry for a specialist LLM."""

    name: str
    backend: str | None
    capability: ModelCapability
    roles: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    provenance: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "backend": self.backend,
            "roles": list(self.roles),
            "tags": list(self.tags),
            "capability": self.capability.to_dict(),
            "provenance": self.provenance,
        }


class ModelInventory:
    """In-memory catalogue of available LLM specialists."""

    def __init__(self, records: Mapping[str, ModelRecord]) -> None:
        self._records: dict[str, ModelRecord] = dict(records)

    def get(self, name: str) -> ModelRecord | None:
        return self._records.get(name)

    def available_models(self) -> Iterable[ModelRecord]:
        return tuple(self._records.values())

    def to_matrix(self) -> Iterable[dict[str, Any]]:
        for record in self._records.values():
            yield record.to_dict()

    @classmethod
    def from_payload(cls, payload: Any) -> ModelInventory:
        if payload is None:
            return cls({})
        try:
            entries = list(payload.values()) if isinstance(payload, Mapping) else list(payload)
        except Exception as exc:
            raise ValueError(f"Inventory payload must be list/dict; got {type(payload)!r}: {exc}")

        records: MutableMapping[str, ModelRecord] = {}
        for item in entries:
            if not isinstance(item, Mapping):
                _LOG.warning("Skipping non-mapping inventory entry: %r", item)
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                _LOG.warning("Skipping inventory entry missing name: %r", item)
                continue
            backend = item.get("backend")
            if backend is not None:
                backend = str(backend)
            capability_payload = item.get("capability") or {}
            if not isinstance(capability_payload, Mapping):
                _LOG.warning("Inventory entry %s has invalid capability payload", name)
                capability_payload = {}
            capability = ModelCapability(
                latency_ms_p95=_coerce_float(capability_payload.get("latency_ms_p95")),
                latency_ms_p50=_coerce_float(capability_payload.get("latency_ms_p50")),
                cost_per_1k_tokens_usd=_coerce_float(capability_payload.get("cost_per_1k_tokens_usd")),
                quality_score=_coerce_float(capability_payload.get("quality_score")),
                safety_tier=_coerce_str(capability_payload.get("safety_tier")),
                max_context_tokens=_coerce_int(capability_payload.get("max_context_tokens")),
                notes=_coerce_str(capability_payload.get("notes")),
            )
            roles = tuple(_coerce_str(role) for role in _ensure_iterable(item.get("roles")))
            tags = tuple(_coerce_str(tag) for tag in _ensure_iterable(item.get("tags")))
            provenance = _coerce_str(item.get("provenance"))
            records[name] = ModelRecord(
                name=name,
                backend=backend,
                capability=capability,
                roles=roles,
                tags=tags,
                provenance=provenance,
            )
        return cls(records)


def load_inventory_from_env() -> ModelInventory | None:
    """Load inventory metadata from env-provided JSON or file path."""
    raw = os.getenv("BRAIN_LLM_INVENTORY_JSON")
    if raw:
        try:
            payload = json.loads(raw)
            return ModelInventory.from_payload(payload)
        except Exception as exc:
            _LOG.warning("Failed to parse BRAIN_LLM_INVENTORY_JSON: %s", exc)
            return None

    path = os.getenv("BRAIN_LLM_INVENTORY_PATH")
    if not path:
        return None
    candidate = Path(path)
    try:
        text = candidate.read_text(encoding="utf-8")
    except FileNotFoundError:
        _LOG.warning("Inventory file %s not found", candidate)
        return None
    except OSError as exc:
        _LOG.warning("Failed to read inventory file %s: %s", candidate, exc)
        return None

    try:
        payload = json.loads(text)
    except Exception as exc:
        _LOG.warning("Failed to parse inventory file %s: %s", candidate, exc)
        return None
    return ModelInventory.from_payload(payload)


def _ensure_iterable(value: Any) -> Iterable[Any]:
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set)):
        return value
    return (value,)


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "ModelCapability",
    "ModelInventory",
    "ModelRecord",
    "load_inventory_from_env",
]
