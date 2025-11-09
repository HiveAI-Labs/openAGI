"""Utilities for orchestrating multi-model ensembles with simple cross checks."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
import difflib
import logging
import statistics
from typing import Any

from brain.core.model_client import ModelClient
from brain.ensemble.broker import BrokerResult, EnsembleBroker, ParallelEnsembleBroker
from brain.ensemble.inventory import ModelCapability

# Optional consensus broker for metrics/persistence of consult results.
try:  # pragma: no cover - metrics side-effect only
    from brain.tools.llm_consensus import Candidate as _Cand, LLMConsensusBroker
    _CONSENSUS_BROKER: LLMConsensusBroker | None = LLMConsensusBroker()
except Exception:
    _CONSENSUS_BROKER = None

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Specialist:
    """Description of a specialist model participating in the ensemble."""

    name: str
    role: str
    client: ModelClient
    capability: ModelCapability | None = None
    tags: tuple[str, ...] = ()
    provenance: str | None = None


@dataclass(frozen=True)
class RawAnswer:
    """Response emitted by a specialist before verification."""

    specialist: Specialist
    content: str
    usage: Mapping[str, object]


@dataclass(frozen=True)
class VerifiedAnswer:
    """Answer that passed ensemble pipeline checks."""

    raw: RawAnswer
    confidence: float
    verification_metadata: Mapping[str, float]

    @property
    def content(self) -> str:
        return self.raw.content

    @property
    def specialist(self) -> Specialist:
        return self.raw.specialist

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the verified answer."""
        return {
            "specialist": {
                "name": self.specialist.name,
                "role": self.specialist.role,
            },
            "content": self.content,
            "confidence": float(self.confidence),
            "verification": dict(self.verification_metadata),
            "usage": dict(self.raw.usage),
        }


@dataclass(frozen=True)
class EnsembleAnswer:
    """Final fused answer returned by the ensemble."""

    answer: str
    confidence: float
    provenance: Sequence[VerifiedAnswer]
    consensus_metrics: Mapping[str, float]
    broker_events: Sequence[Mapping[str, object]] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the ensemble answer."""
        return {
            "answer": self.answer,
            "confidence": float(self.confidence),
            "provenance": [item.to_dict() for item in self.provenance],
            "consensus_metrics": dict(self.consensus_metrics),
            "broker_events": [dict(evt) for evt in self.broker_events],
        }


def _tokenize(content: str) -> list[str]:
    tokens = [token.strip(" ,.;:!?") for token in content.lower().split()]
    return [token for token in tokens if token]


def _sentence_chunks(content: str) -> list[str]:
    chunks: list[str] = []
    current = []
    for token in content.split():
        current.append(token)
        if token.endswith((".", "!", "?")):
            chunks.append(" ".join(current).strip())
            current = []
    if current:
        chunks.append(" ".join(current).strip())
    return chunks


class CrossVerificationPipeline:
    """Performs lightweight filtering and consensus scoring across answers."""

    def __init__(
        self,
        *,
        consensus_threshold: float = 0.35,
        hallucination_terms: Iterable[str] | None = None,
        syntax_policy: str = "skip",
    ) -> None:
        self._consensus_threshold = max(0.0, min(1.0, float(consensus_threshold)))
        self._hallucination_terms = tuple(term.lower() for term in (hallucination_terms or ("???", "fabricated")))
        self._syntax_policy = syntax_policy if syntax_policy in ("skip", "penalize") else "skip"

    def verify(self, raw_answers: Sequence[RawAnswer]) -> list[VerifiedAnswer]:
        verified: list[VerifiedAnswer] = []
        for raw in raw_answers:
            if not raw.content.strip():
                continue
            # Basic structural consistency checks (braces, etc.)
            if not self._is_consistent(raw.content):
                continue
            # Syntax check for likely Python code snippets. If a snippet looks
            # like Python and fails to parse, treat it as invalid and skip it.
            syntax_ok, syntax_err = self._syntax_ok(raw.content)
            if not syntax_ok:
                if self._syntax_policy == "skip":
                    # Skip obviously invalid code answers
                    continue
                # 'penalize' policy: continue but we will reduce factual_score below
            factual_score = self._factual_score(raw.content)
            # If syntax is invalid and policy is 'penalize', reduce factual score
            if not syntax_ok and self._syntax_policy == "penalize":
                factual_score = max(0.0, factual_score - 0.4)
            if factual_score < 0.5:
                continue
            hallucination_prob = self._hallucination_probability(raw.content)
            if hallucination_prob > 0.6:
                continue
            consensus = self._consensus_with_others(raw, raw_answers)
            if consensus < self._consensus_threshold:
                continue
            confidence = factual_score * (1.0 - hallucination_prob) * consensus
            verified.append(
                VerifiedAnswer(
                    raw=raw,
                    confidence=confidence,
                    verification_metadata={
                        "factual_score": factual_score,
                        "hallucination_prob": hallucination_prob,
                        "consensus": consensus,
                        "syntax_ok": bool(syntax_ok),
                    },
                ),
            )
        verified.sort(key=lambda item: item.confidence, reverse=True)
        return verified

    def _syntax_ok(self, content: str) -> tuple[bool, str | None]:
        """Return (True, None) if content appears syntactically valid for
        Python code when code-like; otherwise (True, None) if no code detected.
        If code is detected and parsing fails, return (False, error).
        """
        import ast
        import re

        text = content or ""
        code: str | None = None

        # Prefer fenced code blocks with an optional language marker. Use a
        # regex that captures the code body robustly even when the fence has a
        # language token on the same line.
        m = re.search(r"```(?:\s*(?P<lang>py(?:thon)?))?\n(?P<code>[\s\S]*?)```", text, re.IGNORECASE)
        if m:
            code = m.group("code")

        if code is None:
            # Heuristic: treat as code only if it contains strong Python signals.
            # Avoid flagging natural language plans like "Plan:" as code.
            lower = text.lower()
            strong_signals = ("def ", "import ", "return ", "class ", "try:", "except ", "with ", " lambda ")
            if any(tok in lower for tok in strong_signals):
                code = text

        if not code:
            return True, None

        # If code contains a leading language name line (e.g. "python\n..."),
        # drop that first line.
        lines = code.splitlines()
        if lines and lines[0].strip().lower() in ("python", "py"):
            code = "\n".join(lines[1:])

        # Repair common mid-identifier line-breaks produced by errant wrapping
        # in some model outputs: join letter-newline-letter occurrences.
        code = re.sub(r"(?<=[A-Za-z0-9_])\n(?=[A-Za-z0-9_])", "", code)

        try:
            ast.parse(code)
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _is_consistent(self, content: str) -> bool:
        braces = 0
        for char in content:
            if char == "{":
                braces += 1
            elif char == "}":
                braces -= 1
                if braces < 0:
                    return False
        return braces == 0

    def _factual_score(self, content: str) -> float:
        penalty = 0.0
        if "TODO" in content or "FIXME" in content:
            penalty += 0.4
        if any(term in content.lower() for term in ("i guess", "maybe", "might be")):
            penalty += 0.3
        length = len(_tokenize(content))
        if length < 6:
            penalty += 0.2
        score = 1.0 - min(penalty, 0.9)
        return max(0.0, min(1.0, score))

    def _hallucination_probability(self, content: str) -> float:
        text = content.lower()
        if any(term in text for term in self._hallucination_terms):
            return 0.7
        capitalised = sum(1 for token in content.split() if token.isupper() and len(token) > 3)
        return min(0.6, 0.1 + 0.05 * capitalised)

    def _consensus_with_others(self, raw: RawAnswer, population: Sequence[RawAnswer]) -> float:
        # Prefer to compute consensus on code blocks when present to avoid
        # penalising legitimate stylistic differences in accompanying prose.
        import re
        m = re.search(r"```(?:\s*\w+)?\n([\s\S]*?)```", raw.content, re.IGNORECASE)
        source = m.group(1) if m else raw.content
        # If the source appears code-like, extract identifier tokens rather than
        # naive whitespace tokens. This yields more stable token overlap across
        # stylistic variations (e.g. different prose around the same code).
        import re
        if "\n" in source and ("def " in source or "return" in source or "[" in source):
            idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", source.lower())
            tokens = set(idents)
        else:
            tokens = set(_tokenize(source))
        if not tokens:
            return 0.0
        overlaps: list[float] = []
        # If source appears code-like, use sequence similarity on the code body
        # (more robust to identifier naming differences) otherwise fall back to
        # token Jaccard.
        is_code_like = "\n" in source and ("def " in source or "return" in source or "[" in source)
        if is_code_like:
            import difflib
            for other in population:
                if other is raw:
                    continue
                m2 = re.search(r"```(?:\s*\w+)?\n([\s\S]*?)```", other.content, re.IGNORECASE)
                other_source = m2.group(1) if m2 else other.content
                # normalize whitespace for fair comparison
                a = "\n".join(line.strip() for line in source.splitlines() if line.strip())
                b = "\n".join(line.strip() for line in other_source.splitlines() if line.strip())
                if not a or not b:
                    continue
                ratio = difflib.SequenceMatcher(None, a, b).ratio()
                overlaps.append(ratio)
        else:
            for other in population:
                if other is raw:
                    continue
                m2 = re.search(r"```(?:\s*\w+)?\n([\s\S]*?)```", other.content, re.IGNORECASE)
                other_source = m2.group(1) if m2 else other.content
                if "\n" in other_source and ("def " in other_source or "return" in other_source or "[" in other_source):
                    other_idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", other_source.lower())
                    other_tokens = set(other_idents)
                else:
                    other_tokens = set(_tokenize(other_source))
                if not other_tokens:
                    continue
                # Use overlap relative to the smaller token set to avoid
                # penalising legitimate extensions (superset answers).
                intersection = len(tokens & other_tokens)
                denom = min(len(tokens), len(other_tokens)) or 1
                overlaps.append(intersection / denom)
        if not overlaps:
            return 0.5
        return sum(overlaps) / len(overlaps)


class AnswerFusionEngine:
    """Synthesises a final response by combining high-confidence fragments."""

    def fuse(self, answers: Sequence[VerifiedAnswer]) -> str:
        if not answers:
            return ""
        primary = answers[0]
        supplemental = OrderedDict[str, None]()
        primary_sentences = {_normalize_sentence(chunk) for chunk in _sentence_chunks(primary.content)}
        seen_sentences = set(primary_sentences)
        # Always include novel numbered steps or sentences from other answers.
        for candidate in answers[1:]:
            for chunk in _sentence_chunks(candidate.content):
                if not chunk:
                    continue
                normalized = _normalize_sentence(chunk)
                if _is_redundant_sentence(normalized, seen_sentences):
                    continue
                supplemental[chunk] = None
                seen_sentences.add(normalized)
        if not supplemental:
            return primary.content
        combined = [primary.content, "\n\nSupplemental insights:"]
        for chunk in supplemental.keys():
            combined.append(f"- {chunk}")
        return "\n".join(combined)


def _normalize_sentence(sentence: str) -> str:
    return " ".join(sentence.strip().split()).lower()


def _is_redundant_sentence(candidate: str, seen: set[str]) -> bool:
    for existing in seen:
        if existing == candidate:
            return True
        # Treat sentences with high token overlap as redundant to avoid
        # reiterating conflicting scalar values (e.g., timestamps).
        ratio = difflib.SequenceMatcher(None, existing, candidate).ratio()
        if ratio >= 0.8:
            return True
    return False


class ConfidenceCalibrator:
    """Normalises per-answer confidence into an ensemble score."""

    def calibrate(self, answers: Sequence[VerifiedAnswer]) -> float:
        if not answers:
            return 0.0
        scores = [answer.confidence for answer in answers]
        peak = max(scores)
        spread = statistics.pstdev(scores) if len(scores) > 1 else 0.0
        calibrated = peak - 0.4 * spread
        # Penalise if any verified answers flagged syntax issues (rare since
        # we skip syntax-invalid answers) — keep this defensive to widen gaps.
        syntax_issue = any(not item.verification_metadata.get("syntax_ok", True) for item in answers)
        if syntax_issue:
            calibrated = max(0.0, calibrated - 0.2)
        return max(0.0, min(0.99, calibrated))


class EnsembleOrchestrator:
    """Coordinates specialist queries and assembles a fused answer."""

    def __init__(
        self,
        specialists: Sequence[Specialist],
        *,
        cross_verifier: CrossVerificationPipeline | None = None,
        fusion_engine: AnswerFusionEngine | None = None,
        confidence_calibrator: ConfidenceCalibrator | None = None,
        broker: EnsembleBroker | None = None,
    ) -> None:
        if not specialists:
            raise ValueError("At least one specialist is required")
        self._specialists = list(specialists)
        self._cross_verifier = cross_verifier or CrossVerificationPipeline()
        self._fusion_engine = fusion_engine or AnswerFusionEngine()
        self._confidence_calibrator = confidence_calibrator or ConfidenceCalibrator()
        self._broker = broker or ParallelEnsembleBroker(max_workers=len(self._specialists))

    def specialists(self) -> tuple[Specialist, ...]:
        return tuple(self._specialists)

    def reorder_specialists(self, ordered: Sequence[Specialist]) -> None:
        desired = list(ordered)
        if len(desired) != len(self._specialists):
            raise ValueError("specialist list mismatch")
        current_ids = {id(spec) for spec in self._specialists}
        desired_ids = {id(spec) for spec in desired}
        if current_ids != desired_ids:
            raise ValueError("ordered specialists must be drawn from existing roster")
        self._specialists = desired

    def broker_metadata(self) -> dict[str, Any]:
        try:
            return dict(self._broker.describe())
        except Exception:
            return {}

    def query(
        self,
        question: str,
        *,
        context: Sequence[str] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> EnsembleAnswer:
        context = list(context or [])
        messages = [{"role": "system", "content": ctx} for ctx in context]
        messages.append({"role": "user", "content": question})

        raw_answers, broker_events = self._collect_answers(messages, temperature=temperature, max_tokens=max_tokens)

        verified = self._cross_verifier.verify(raw_answers)
        if not verified or len(verified) < 2:
            # Fallback behavior: return first raw answer verbatim for deterministic
            # semantics when consensus too low or only one verified answer survived.
            # This matches legacy expectations in tests that a low-confidence path
            # yields the original (possibly placeholder) content for triage.
            first = raw_answers[0]
            return EnsembleAnswer(
                answer=first.content,
                confidence=0.2,
                provenance=(),
                consensus_metrics={"count": float(len(verified)), "average_consensus": 0.0},
                broker_events=broker_events,
            )

        fused_answer = self._fusion_engine.fuse(verified)
        confidence = self._confidence_calibrator.calibrate(verified)
        consensus_stats = self._build_consensus_metrics(verified)
        # Side-effect: emit consensus metrics and persist a consult_result row
        # using a deterministic, pure scoring broker. This does not alter the
        # orchestrator's fused answer.
        try:  # pragma: no cover - side-effect only
            if _CONSENSUS_BROKER is not None:
                _cands = [_Cand(model=v.specialist.name, content=v.content) for v in verified]
                _CONSENSUS_BROKER.score_session(None, _cands)
        except Exception:
            pass
        return EnsembleAnswer(
            answer=fused_answer,
            confidence=confidence,
            provenance=tuple(verified),
            consensus_metrics=consensus_stats,
            broker_events=broker_events,
        )

    def _build_consensus_metrics(self, verified: Sequence[VerifiedAnswer]) -> MutableMapping[str, float]:
        if not verified:
            return {"count": 0.0, "average_consensus": 0.0}
        consensuses = [item.verification_metadata.get("consensus", 0.0) for item in verified]
        avg = sum(consensuses) / len(consensuses)
        harmonic = 0.0
        if all(value > 0 for value in consensuses):
            harmonic = len(consensuses) / sum(1.0 / value for value in consensuses)
        return {
            "count": float(len(verified)),
            "average_consensus": avg,
            "harmonic_consensus": harmonic,
        }

    def _collect_answers(
        self,
        messages: Sequence[Mapping[str, object]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> tuple[list[RawAnswer], tuple[Mapping[str, object], ...]]:
        broker_results: list[BrokerResult] = []
        if self._broker is not None:
            try:
                broker_results = self._broker.gather(
                    self._specialists,
                    messages,
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
            except Exception as exc:  # pragma: no cover - defensive
                _LOG.warning("Ensemble broker failed, falling back to sequential mode: %s", exc)
                broker_results = []

        raw_answers: list[RawAnswer] = []
        broker_events: list[Mapping[str, object]] = []

        if broker_results:
            for result in broker_results:
                usage = dict(result.usage)
                usage.setdefault("broker_latency_ms", result.latency_ms)
                if result.error:
                    usage.setdefault("broker_error", result.error)
                if result.timed_out:
                    usage.setdefault("broker_timeout", True)
                raw_answers.append(
                    RawAnswer(
                        specialist=result.specialist,
                        content=result.content,
                        usage=usage,
                    ),
                )
                broker_events.append(result.to_dict())
        else:
            for specialist in self._specialists:
                text, usage = specialist.client.generate(  # type: ignore[attr-defined]
                    list(messages),
                    temperature=float(temperature),
                    max_tokens=int(max_tokens),
                )
                raw_answers.append(
                    RawAnswer(
                        specialist=specialist,
                        content=str(text).strip(),
                        usage=dict(usage),
                    ),
                )

        if not raw_answers:
            raise RuntimeError("Ensemble orchestrator was unable to gather answers")

        return raw_answers, tuple(broker_events)


__all__ = [
    "AnswerFusionEngine",
    "ConfidenceCalibrator",
    "CrossVerificationPipeline",
    "EnsembleAnswer",
    "EnsembleOrchestrator",
    "RawAnswer",
    "Specialist",
    "VerifiedAnswer",
]
