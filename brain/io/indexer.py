from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from brain.io.validators import LLMResponse, NormalizedRequest

__all__ = ["index_if_accepted"]


def _hash_blob(blob: str) -> str:
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _doc_id(req: NormalizedRequest, response: LLMResponse, citations_hash: str) -> str:
    return f"io:{req.query_hash}:{citations_hash}"


def index_if_accepted(brain, req: NormalizedRequest, response: LLMResponse, citations_hash: str) -> None:
    """Persist accepted answers into semantic/vector stores with deduping."""
    if not hasattr(brain, "memory_semantic") and not hasattr(brain, "memory_vector"):
        return

    meta: dict[str, Any] = {
        "source": "io_query",
        "query_hash": req.query_hash,
        "citations_hash": citations_hash,
        "timestamp": time.time(),
    }
    if response.confidence is not None:
        meta["confidence"] = response.confidence
    if req.metadata:
        meta["metadata"] = req.metadata

    citations_blob = json.dumps(response.citations, ensure_ascii=False)
    text_payload = (
        f"QUESTION: {req.query}\n"
        f"ANSWER: {response.answer}\n"
        f"CITATIONS: {citations_blob}"
    )

    doc_id = _doc_id(req, response, citations_hash)

    semantic = getattr(brain, "memory_semantic", None)
    if semantic and hasattr(semantic, "has_doc") and hasattr(semantic, "add"):
        try:
            if semantic.has_doc(doc_id):
                # Refresh existing entry by re-adding (store handles replace)
                semantic.add(doc_id, text_payload, meta=meta)
            else:
                semantic.add(doc_id, text_payload, meta=meta)
        except Exception:
            pass

    vector = getattr(brain, "memory_vector", None)
    if vector and hasattr(vector, "has_doc") and hasattr(vector, "add"):
        try:
            if vector.has_doc(doc_id):
                vector.add(doc_id, text_payload, meta=meta)
            else:
                vector.add(doc_id, text_payload, meta=meta)
        except Exception:
            pass
