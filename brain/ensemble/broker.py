"""Concurrent broker for LLM ensemble queries."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import concurrent.futures
from dataclasses import dataclass
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from brain.ensemble.orchestrator import Specialist

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrokerResult:
    """Result emitted by the broker for a single specialist call."""

    specialist: Specialist
    content: str
    usage: Mapping[str, object]
    latency_ms: float
    timed_out: bool = False
    error: str | None = None

    def to_dict(self) -> Mapping[str, object]:
        return {
            "specialist": self.specialist.name,
            "role": self.specialist.role,
            "latency_ms": self.latency_ms,
            "timed_out": self.timed_out,
            "error": self.error,
        }


class EnsembleBroker:
    """Abstract base-class for firing specialist queries."""

    def gather(
        self,
        specialists: Sequence[Specialist],
        messages: Sequence[Mapping[str, object]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[BrokerResult]:  # pragma: no cover - interface only
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:  # pragma: no cover - interface only
        return {}


class ParallelEnsembleBroker(EnsembleBroker):
    """Thread-pool based broker that fan-outs requests to specialists."""

    def __init__(
        self,
        *,
        max_workers: int | None = None,
        request_timeout_s: float = 8.0,
        allowed_roles: Iterable[str] | None = None,
    ) -> None:
        self._timeout_s = max(0.001, float(request_timeout_s))
        self._max_workers = int(max_workers) if max_workers else None
        if allowed_roles:
            cleaned = {role.strip() for role in allowed_roles if role and str(role).strip()}
            self._allowed_roles = cleaned or None
        else:
            self._allowed_roles = None

    def describe(self) -> dict[str, Any]:
        allowed = sorted(self._allowed_roles) if self._allowed_roles else None
        return {
            "timeout_ms": float(self._timeout_s * 1000.0),
            "max_workers": self._max_workers,
            "allowed_roles": allowed,
        }

    def gather(
        self,
        specialists: Sequence[Specialist],
        messages: Sequence[Mapping[str, object]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> list[BrokerResult]:
        results: list[BrokerResult] = []
        if not specialists:
            return results

        order = {spec: idx for idx, spec in enumerate(specialists)}

        allowed_roles = self._allowed_roles
        pending: list[tuple[int, Specialist, concurrent.futures.Future[BrokerResult]]] = []

        def _invoke(spec: Specialist) -> BrokerResult:
            start = time.time()
            try:
                text, usage = spec.client.generate(  # type: ignore[attr-defined]
                    list(messages),
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                elapsed = max(0.0, time.time() - start) * 1000.0
                return BrokerResult(spec, str(text).strip(), dict(usage), latency_ms=elapsed)
            except Exception as exc:  # pragma: no cover - defensive
                elapsed = max(0.0, time.time() - start) * 1000.0
                _LOG.warning("Broker specialist %s failed: %s", spec.name, exc)
                return BrokerResult(spec, "", {}, latency_ms=elapsed, error=str(exc))

        with concurrent.futures.ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            for idx, spec in enumerate(specialists):
                if allowed_roles and spec.role not in allowed_roles:
                    results.append(
                        BrokerResult(
                            spec,
                            "",
                            {},
                            latency_ms=0.0,
                            error=f"role_not_allowed:{spec.role}",
                        ),
                    )
                    continue
                future = executor.submit(_invoke, spec)
                pending.append((idx, spec, future))

            for idx, spec, future in pending:
                try:
                    result = future.result(timeout=self._timeout_s)
                except concurrent.futures.TimeoutError:
                    result = BrokerResult(
                        spec,
                        "",
                        {},
                        latency_ms=self._timeout_s * 1000.0,
                        timed_out=True,
                        error="timeout",
                    )
                except Exception as exc:  # pragma: no cover - executor raised
                    result = BrokerResult(
                        spec,
                        "",
                        {},
                        latency_ms=self._timeout_s * 1000.0,
                        error=str(exc),
                    )
                results.append(result)

        # Preserve original ordering for downstream determinism
        results.sort(key=lambda item: order.get(item.specialist, 0))
        return results


__all__ = [
    "BrokerResult",
    "EnsembleBroker",
    "ParallelEnsembleBroker",
]
