"""Performance tracker for ensemble specialists."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from .orchestrator import EnsembleAnswer, Specialist, VerifiedAnswer

__all__ = ["EnsemblePerformanceTracker", "SpecialistStats"]


@dataclass
class SpecialistStats:
    """Rolling statistics for a specialist."""

    queries: int = 0
    successes: int = 0
    confidence_sum: float = 0.0
    last_confidence: float = 0.0

    @property
    def success_rate(self) -> float:
        return self.successes / self.queries if self.queries else 0.0

    @property
    def average_confidence(self) -> float:
        return self.confidence_sum / self.queries if self.queries else 0.0


class EnsemblePerformanceTracker:
    """Tracks per-specialist performance and provides adaptive rankings."""

    def __init__(self, *, history_size: int = 256) -> None:
        self._stats: dict[str, SpecialistStats] = defaultdict(SpecialistStats)
        self._history: deque[Mapping[str, object]] = deque(maxlen=int(history_size))
        self._last_token: Mapping[str, object] | None = None

    def record_query(
        self,
        *,
        question: str,
        context: Sequence[str],
        answer: EnsembleAnswer,
        provenance: Sequence[VerifiedAnswer],
        outcome: str,
    ) -> Mapping[str, object]:
        timestamp = datetime.now(UTC).isoformat()
        specialists: list[str] = []
        for verified in provenance:
            name = verified.specialist.name
            stats = self._stats[name]
            stats.queries += 1
            stats.confidence_sum += float(verified.confidence)
            stats.last_confidence = float(verified.confidence)
            specialists.append(name)
        entry: dict[str, object] = {
            "timestamp": timestamp,
            "question": question,
            "context": list(context),
            "answer_confidence": float(answer.confidence),
            "outcome": outcome,
            "specialists": list(specialists),
        }
        self._history.append(entry)
        token = {
            "specialists": list(specialists),
            "timestamp": timestamp,
        }
        self._last_token = token
        return token

    def record_feedback(self, success: bool) -> None:
        token = self._last_token
        if not token:
            return
        for name in token.get("specialists", []):
            stats = self._stats[name]
            stats.successes += 1 if success else 0
        self._last_token = None

    def rank_specialists(self, specialists: Sequence[Specialist]) -> list[Specialist]:
        if not specialists:
            return []
        scores: dict[str, tuple[float, float]] = {}
        for spec in specialists:
            stats = self._stats.get(spec.name)
            if not stats or stats.queries == 0:
                scores[spec.name] = (0.0, 0.0)
            else:
                scores[spec.name] = (stats.success_rate, stats.average_confidence)
        # Stable sort: tie-break by original position.
        indexed = list(enumerate(specialists))
        indexed.sort(
            key=lambda item: (
                -scores[item[1].name][0],
                -scores[item[1].name][1],
                item[0],
            ),
        )
        return [spec for _, spec in indexed]

    def snapshot(self) -> Mapping[str, Mapping[str, float]]:
        return {
            name: {
                "queries": float(stats.queries),
                "successes": float(stats.successes),
                "success_rate": stats.success_rate,
                "average_confidence": stats.average_confidence,
            }
            for name, stats in self._stats.items()
        }

    def history(self) -> Sequence[Mapping[str, object]]:
        return list(self._history)
