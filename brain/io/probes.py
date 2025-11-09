from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import re

from brain.io.validators import ContextEntry, LLMResponse, NormalizedRequest

__all__ = ["ProbeIssue", "ProbeReport", "run_probes"]

# Basic NLP helpers kept lightweight so they run in constrained test envs.
_TOKEN_RE = re.compile(r"[A-Za-z0-9']+")
_STOPWORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
}
_NEGATION_TOKENS: set[str] = {
    "no",
    "not",
    "never",
    "none",
    "nobody",
    "without",
    "cannot",
    "can't",
    "isn't",
    "aren't",
    "wasn't",
    "weren't",
    "doesn't",
    "don't",
    "didn't",
    "won't",
    "wouldn't",
    "shouldn't",
    "fails",
    "failure",
    "disabled",
    "denies",
    "denied",
    "forbidden",
    "forbids",
}
_POSITIVE_TOKENS: set[str] = {
    "enable",
    "enabled",
    "enables",
    "allows",
    "allow",
    "supports",
    "supported",
    "successful",
    "success",
    "permitted",
    "permits",
    "available",
    "provided",
    "provides",
    "on",
}


@dataclass
class ProbeIssue:
    kind: str
    severity: str  # "warning" or "error"
    message: str
    context_ids: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "context_ids": list(self.context_ids),
        }


@dataclass
class ProbeReport:
    issues: list[ProbeIssue]
    causal: list[dict[str, object]]
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "issues": [issue.to_dict() for issue in self.issues],
            "causal": self.causal,
            "summary": self.summary,
        }


def run_probes(req: NormalizedRequest, response: LLMResponse) -> ProbeReport:
    context_tokens = {entry.cid: _keywords(entry.text) for entry in req.context}
    context_negation = {entry.cid: _has_negation(entry.text) for entry in req.context}
    context_positive = {entry.cid: _has_positive(entry.text) for entry in req.context}
    issues: list[ProbeIssue] = []

    # Detect contradictory context pairs using simple lexical heuristics.
    issues.extend(
        _detect_context_contradictions(req.context, context_tokens, context_negation, context_positive),
    )

    cited_ids = {
        citation.get("id")
        for citation in response.citations
        if isinstance(citation, dict) and isinstance(citation.get("id"), str)
    }
    cited_ids.discard(None)

    answer_tokens = _keywords(response.answer)
    answer_has_neg = _has_negation(response.answer)

    if cited_ids:
        cited_neg_flags = [context_negation.get(cid) for cid in cited_ids if cid in context_negation]
        if cited_neg_flags:
            # Flag if majority of cited context disagrees with the answer on negation polarity.
            neg_majority = _majority_flag(cited_neg_flags)
            if neg_majority is not None and neg_majority != answer_has_neg:
                issues.append(
                    ProbeIssue(
                        kind="answer_context_polarity_mismatch",
                        severity="warning",
                        message="Answer polarity differs from cited context",
                        context_ids=sorted(cited_ids),
                    ),
                )

    causal_records: list[dict[str, object]] = []
    for entry in req.context:
        keywords = context_tokens[entry.cid]
        overlap_terms = sorted(answer_tokens & keywords)
        support_ratio = 0.0
        if keywords:
            support_ratio = len(overlap_terms) / float(len(keywords))
        record = {
            "id": entry.cid,
            "referenced": entry.cid in cited_ids,
            "overlap_terms": overlap_terms[:8],
            "support_ratio": round(support_ratio, 3),
            "negation": context_negation[entry.cid],
            "token_count": len(keywords),
        }
        causal_records.append(record)
        if (not record["referenced"]) and len(overlap_terms) >= 4 and support_ratio >= 0.3:
            issues.append(
                ProbeIssue(
                    kind="unused_high_overlap_context",
                    severity="warning",
                    message=f"Context {entry.cid} overlaps with answer but was not cited",
                    context_ids=[entry.cid],
                ),
            )

    summary = {
        "answer_has_negation": answer_has_neg,
        "cited_ids": sorted(cited_ids),
        "context_negation": {cid: context_negation[cid] for cid in context_negation},
    }

    return ProbeReport(issues=issues, causal=causal_records, summary=summary)


def _keywords(text: str) -> set[str]:
    tokens = [tok.lower() for tok in _TOKEN_RE.findall(text)]
    return {tok for tok in tokens if tok not in _STOPWORDS}


def _has_negation(text: str) -> bool:
    tokens = _keywords(text)
    return any(tok in _NEGATION_TOKENS for tok in tokens)


def _has_positive(text: str) -> bool:
    tokens = _keywords(text)
    return any(tok in _POSITIVE_TOKENS for tok in tokens)


def _detect_context_contradictions(
    context: Sequence[ContextEntry],
    tokens: dict[str, set[str] | None],
    negation: dict[str, bool],
    positive: dict[str, bool],
) -> list[ProbeIssue]:
    issues: list[ProbeIssue] = []
    for idx, entry_a in enumerate(context):
        tokens_a = tokens.get(entry_a.cid) or set()
        if not tokens_a:
            continue
        for entry_b in context[idx + 1 :]:
            tokens_b = tokens.get(entry_b.cid) or set()
            if not tokens_b:
                continue
            overlap = tokens_a & tokens_b
            if len(overlap) < 5:
                continue
            neg_a = negation.get(entry_a.cid, False)
            neg_b = negation.get(entry_b.cid, False)
            pos_a = positive.get(entry_a.cid, False)
            pos_b = positive.get(entry_b.cid, False)
            polarity_disagree = (neg_a != neg_b) or (pos_a != pos_b)
            conflicting_keywords = _detect_conflicting_keywords(tokens_a, tokens_b)
            if polarity_disagree or conflicting_keywords:
                shared_terms = ", ".join(sorted(list(overlap))[:6])
                message = (
                    f"Potential contradiction between {entry_a.cid} and {entry_b.cid} (shared terms: {shared_terms})"
                )
                if conflicting_keywords:
                    message += f"; conflicting verbs: {', '.join(sorted(conflicting_keywords))}"
                issues.append(
                    ProbeIssue(
                        kind="context_contradiction",
                        severity="warning",
                        message=message,
                        context_ids=[entry_a.cid, entry_b.cid],
                    ),
                )
    return issues


def _detect_conflicting_keywords(tokens_a: set[str], tokens_b: set[str]) -> set[str]:
    """Identify obvious antonym or polarity disagreements."""
    conflicts: set[str] = set()
    antonym_pairs = [
        ("enable", "disable"),
        ("enabled", "disabled"),
        ("allow", "forbid"),
        ("allows", "forbids"),
        ("increase", "decrease"),
        ("allowed", "prohibited"),
        ("supports", "prevents"),
    ]
    for pos_word, neg_word in antonym_pairs:
        if (pos_word in tokens_a and neg_word in tokens_b) or (pos_word in tokens_b and neg_word in tokens_a):
            conflicts.update({pos_word, neg_word})
    return conflicts


def _majority_flag(flags: Iterable[bool | None]) -> bool | None:
    votes = [flag for flag in flags if flag is not None]
    if not votes:
        return None
    positives = sum(1 for flag in votes if flag)
    negatives = len(votes) - positives
    if positives == negatives:
        return None
    return positives > negatives
