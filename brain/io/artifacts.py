from __future__ import annotations

from collections.abc import Iterable, Sequence
import json
from pathlib import Path
from typing import Any

from brain.io.audit import record_probe_alert
from brain.io.validators import NormalizedRequest, ValidationResult

__all__ = ["write_artifacts"]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_artifacts(
    base_dir: str | Path,
    run_id: str,
    req: NormalizedRequest,
    llm_messages: Sequence[dict[str, str]],
    llm_output: str,
    validation: ValidationResult,
) -> Path:
    """Persist artifacts for the IO query run.

    Returns the directory path used to store the artifacts.
    """
    base = Path(base_dir) / "io" / "runs" / run_id
    _ensure_dir(base)

    _write_json(base / "query.json", {
        "query": req.query,
        "mode": req.mode,
        "autonomy_level": req.autonomy_level,
        "budget": {
            "seconds": req.budget_seconds,
            "tokens": req.budget_tokens,
        },
        "metadata": req.metadata,
    })

    _write_jsonl(
        base / "context.jsonl",
        (
            {
                "id": entry.cid,
                "sha256": entry.sha256,
                "text": entry.text,
                "source_index": entry.source_index,
            }
            for entry in req.context
        ),
    )

    _write_json(base / "llm_request.json", list(llm_messages))
    (base / "llm_raw.txt").write_text(llm_output, encoding="utf-8")

    response_payload: dict[str, Any] = {
        "decision": validation.decision,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "latency_seconds": validation.latency_seconds,
        "usage": validation.usage,
    }
    if validation.probe_findings:
        response_payload["probe_findings"] = validation.probe_findings
    if validation.parsed:
        response_payload["answer"] = validation.parsed.answer
        response_payload["citations"] = validation.parsed.citations
        response_payload["confidence"] = validation.parsed.confidence
    else:
        response_payload["answer"] = ""
        response_payload["citations"] = []
        response_payload["confidence"] = None

    _write_json(base / "response.json", response_payload)
    validation_payload = {
        "ok": validation.ok,
        "decision": validation.decision,
        "errors": validation.errors,
        "warnings": validation.warnings,
        "citations_hash": validation.citations_hash,
        "latency_seconds": validation.latency_seconds,
        "usage": validation.usage,
        "probe_findings": validation.probe_findings,
    }
    _write_json(base / "validation.json", validation_payload)

    record_probe_alert(base_dir, run_id, req, validation)

    _write_json(
        base / "bundle.meta.json",
        {
            "run_id": run_id,
            "query_hash": req.query_hash,
            "context_ids": [entry.cid for entry in req.context],
            "citations_hash": validation.citations_hash,
        },
    )

    return base
