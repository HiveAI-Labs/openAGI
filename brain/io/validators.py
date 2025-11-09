from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

__all__ = [
    "ValidationError",
    "ContextEntry",
    "NormalizedRequest",
    "LLMResponse",
    "ValidationResult",
    "normalize_request",
    "build_messages",
    "parse_llm_response",
    "validate_response",
    "compute_citations_hash",
]


class ValidationError(Exception):
    """Raised when the IO query payload is malformed."""


@dataclass
class ContextEntry:
    cid: str
    text: str
    sha256: str
    source_index: int


@dataclass
class NormalizedRequest:
    query: str
    mode: str
    autonomy_level: str
    budget_seconds: float
    budget_tokens: int
    metadata: dict[str, Any]
    context: list[ContextEntry]
    query_hash: str


@dataclass
class LLMResponse:
    answer: str
    citations: list[dict[str, Any]]
    confidence: float | None
    raw: str


@dataclass
class ValidationResult:
    ok: bool
    decision: str
    errors: list[str]
    warnings: list[str]
    parsed: LLMResponse | None
    latency_seconds: float
    usage: dict[str, Any]
    citations_hash: str | None
    probe_findings: dict[str, Any] | None = None

    def update_decision(self) -> None:
        """Recompute decision/ok fields after mutating errors or warnings."""
        self.ok = not self.errors
        if self.errors:
            self.decision = "reject"
        elif self.warnings:
            self.decision = "quarantine"
        else:
            self.decision = "accept"


_ALLOWED_MODES = {"assist", "semi"}
_CODE_FENCE_PREFIX = "```"
_PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN-like
    re.compile(r"\b\d{16}\b"),  # 16 digit account/order numbers
    re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE),
]


def _ensure_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_request(payload: dict[str, Any], *, autonomy_level: str) -> NormalizedRequest:
    """Validate incoming request payload and normalize defaults."""
    if not isinstance(payload, dict):
        raise ValidationError("request payload must be an object")

    query = str(payload.get("query") or "").strip()
    if not query:
        raise ValidationError("query is required")

    context_raw = payload.get("context")
    if not isinstance(context_raw, list) or not context_raw:
        raise ValidationError("context must be a non-empty list")

    context: list[ContextEntry] = []
    for idx, item in enumerate(context_raw):
        if not isinstance(item, dict):
            raise ValidationError(f"context[{idx}] must be an object")
        cid = str(item.get("id") or "").strip()
        text = str(item.get("text") or "").strip()
        sha = str(item.get("sha256") or "").strip()
        if not cid or not text:
            raise ValidationError(f"context[{idx}] missing id or text")
        computed = _ensure_sha256(text)
        if sha and sha != computed:
            raise ValidationError(f"context[{idx}] sha256 mismatch")
        if not sha:
            sha = computed
        context.append(ContextEntry(cid=cid, text=text, sha256=sha, source_index=idx))

    mode = str(payload.get("mode") or "assist").strip().lower()
    if mode not in _ALLOWED_MODES:
        raise ValidationError(f"mode must be one of {_ALLOWED_MODES}")

    budget = payload.get("budget") or {}
    if not isinstance(budget, dict):
        raise ValidationError("budget must be an object when provided")

    seconds = float(budget.get("seconds", 5.0))
    if seconds <= 0:
        raise ValidationError("budget.seconds must be > 0")
    tokens = int(budget.get("tokens", 1024))
    if tokens <= 0:
        raise ValidationError("budget.tokens must be > 0")

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValidationError("metadata must be an object when provided")

    query_hash = _ensure_sha256(query)

    return NormalizedRequest(
        query=query,
        mode=mode,
        autonomy_level=str(autonomy_level or "full"),
        budget_seconds=seconds,
        budget_tokens=tokens,
        metadata=dict(metadata),
        context=context,
        query_hash=query_hash,
    )


def build_messages(req: NormalizedRequest) -> list[dict[str, str]]:
    """Construct chat messages for the model client."""
    context_lines = []
    for entry in req.context:
        context_lines.append(
            f"ID: {entry.cid}\nSHA256: {entry.sha256}\nTEXT: {entry.text}",
        )
    context_blob = "\n\n".join(context_lines)
    system_prompt = (
        "You must answer in pure JSON with keys exactly "
        '{"answer": string, "citations": [string], "confidence": float}. '
        "No markdown, code fences, or commentary. Use only context IDs in citations."
    )
    user_prompt = (
        "Question:\n"
        f"{req.query}\n\n"
        "Approved context (each entry includes id and hash):\n"
        f"{context_blob}\n\n"
        "Rules: cite every claim with at least one provided id. "
        "List citations as strings matching the context ids. Output valid JSON only."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_llm_response(raw: str) -> LLMResponse:
    if raw is None:
        raise ValidationError("llm returned empty response")

    # Some model clients (e.g., HF adapters) return structured payloads with a
    # top-level `text` field alongside tokens/logits metadata. Accept those by
    # projecting to the raw text so downstream validation can continue without
    # forcing every caller to special-case the payload shape.
    if isinstance(raw, dict):
        if isinstance(raw.get("text"), str):
            raw = raw["text"]
        elif isinstance(raw.get("output"), str):
            raw = raw["output"]
        else:
            raise ValidationError("llm response is not valid JSON: missing text field")

    def _strip_code_fence(text: str) -> str:
        stripped = text.strip()
        if not stripped.startswith(_CODE_FENCE_PREFIX):
            return stripped
        # Drop leading fence including optional language tag (e.g., ```json)
        body = stripped[len(_CODE_FENCE_PREFIX) :]
        body = body.lstrip()
        newline = body.find("\n")
        if newline != -1:
            body = body[newline + 1 :]
        else:
            body = ""
        if body.endswith(_CODE_FENCE_PREFIX):
            body = body[: -len(_CODE_FENCE_PREFIX)]
        elif _CODE_FENCE_PREFIX in body:
            body = body.rsplit(_CODE_FENCE_PREFIX, 1)[0]
        return body.strip()

    def _json_candidates(text: str) -> list[str]:
        candidates = []
        stripped = text.strip()
        if stripped:
            candidates.append(stripped)
        fenced = _strip_code_fence(text)
        if fenced and fenced not in candidates:
            candidates.append(fenced)
        first = stripped.find("{")
        last = stripped.rfind("}")
        if first != -1 and last != -1 and last > first:
            inner = stripped[first : last + 1]
            if inner and inner not in candidates:
                candidates.append(inner)
        # Scan for embedded JSON objects using raw_decode so we can recover
        # valid payloads even when the model surrounds them with narrative
        # text or multiple fenced examples.
        try:
            decoder = json.JSONDecoder()
            idx = 0
            text_len = len(text)
            while idx < text_len:
                ch = text[idx]
                if ch == "{" or ch == "[":
                    try:
                        obj, consumed = decoder.raw_decode(text[idx:])
                    except ValueError:
                        idx += 1
                        continue
                    snippet = text[idx : idx + consumed]
                    if snippet and snippet not in candidates:
                        candidates.append(snippet)
                    idx += max(consumed, 1)
                    if len(candidates) >= 8:
                        break
                    continue
                idx += 1
        except Exception:
            pass
        return candidates

    payload = None
    last_err: json.JSONDecodeError | None = None
    for candidate in _json_candidates(raw):
        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            last_err = exc
            continue

    if payload is None:
        if last_err is None:
            raise ValidationError("llm response is not valid JSON")
        raise ValidationError(f"llm response is not valid JSON: {last_err}") from last_err
    if not isinstance(payload, dict):
        raise ValidationError("llm response must be a JSON object")

    answer = str(payload.get("answer") or "").strip()
    citations_raw = payload.get("citations") or []
    if not isinstance(citations_raw, list):
        raise ValidationError("citations must be a list")
    citations: list[dict[str, Any]] = []
    for idx, item in enumerate(citations_raw):
        if isinstance(item, str):
            cid = item.strip()
            if not cid:
                raise ValidationError(f"citations[{idx}] must be a non-empty string or object")
            citations.append({"id": cid, "sha256": "", "evidence": ""})
            continue
        if not isinstance(item, dict):
            raise ValidationError(f"citations[{idx}] must be an object")
        cid = str(item.get("id") or "").strip()
        sha = str(item.get("sha256") or "").strip()
        evidence = str(item.get("evidence") or "").strip()
        citations.append({"id": cid, "sha256": sha, "evidence": evidence})

    confidence_val = payload.get("confidence")
    confidence: float | None
    if confidence_val is None:
        confidence = None
    else:
        try:
            confidence = float(confidence_val)
        except (TypeError, ValueError):
            raise ValidationError("confidence must be numeric")
        if confidence < 0 or confidence > 1:
            raise ValidationError("confidence must be between 0 and 1")

    return LLMResponse(answer=answer, citations=citations, confidence=confidence, raw=raw)


def compute_citations_hash(citations: Sequence[dict[str, Any]]) -> str:
    if not citations:
        return _ensure_sha256("none")
    parts = [f"{c.get('id','')}:{c.get('sha256','')}" for c in citations]
    joined = "|".join(sorted(parts))
    return _ensure_sha256(joined)


def _scan_for_pii(text: str) -> bool:
    if not text:
        return False
    for pattern in _PII_PATTERNS:
        if pattern.search(text):
            return True
    return False


def _detect_contradiction(answer: str, context: Iterable[ContextEntry]) -> bool:
    """Placeholder contradiction detector.

    Current heuristic: flag when the answer explicitly states that context is
    unavailable even though context was provided. This is a weak guard but
    keeps the skeleton honest until richer NLI checks ship.
    """
    if not answer:
        return False
    lowered = answer.lower()
    if "no information" in lowered or "cannot find" in lowered:
        return True
    return False


def validate_response(
    req: NormalizedRequest,
    response: LLMResponse,
    *,
    latency_seconds: float,
    usage: dict[str, Any] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    context_by_id = {entry.cid: entry for entry in req.context}

    if not response.citations:
        errors.append("missing_citations")

    seen_ids = set()
    for idx, citation in enumerate(response.citations):
        cid = citation.get("id") or ""
        sha = citation.get("sha256") or ""
        if not cid:
            errors.append(f"citations[{idx}].id_missing")
            continue
        entry = context_by_id.get(cid)
        if not entry:
            errors.append(f"citations[{idx}].unknown_id")
            continue
        if sha and sha != entry.sha256:
            errors.append(f"citations[{idx}].hash_mismatch")
            continue
        if not sha:
            if entry.sha256:
                citation["sha256"] = entry.sha256
                sha = entry.sha256
            else:
                warnings.append(f"citations[{idx}].hash_missing")
        seen_ids.add(cid)

    if _scan_for_pii(response.answer):
        warnings.append("pii_detected")

    if _detect_contradiction(response.answer, req.context):
        warnings.append("contradiction_suspected")

    if latency_seconds > req.budget_seconds:
        errors.append("latency_over_budget")

    if usage:
        try:
            completion = int(usage.get("completion_tokens") or 0)
            if completion > req.budget_tokens:
                errors.append("token_budget_exceeded")
        except Exception:
            pass

    ok = not errors
    decision = "accept"
    if errors:
        decision = "reject"
    elif warnings:
        decision = "quarantine"

    citations_hash = compute_citations_hash(response.citations)

    return ValidationResult(
        ok=ok,
        decision=decision,
        errors=errors,
        warnings=warnings,
        parsed=response,
        latency_seconds=float(latency_seconds),
        usage=dict(usage or {}),
        citations_hash=citations_hash,
    )
