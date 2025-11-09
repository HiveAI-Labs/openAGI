"""Materialise evaluated skill candidates into signed registry entries.

This script converts the JSON summary emitted by
``experiments.skills.run`` into concrete skill documents suitable for
loading into the `skills` registry. Each generated skill is signed with
a SHA-256 HMAC keyed by ``PROOF_SIGN_KEY`` (if provided) and stored
alongside a `_capabilities.json` manifest for downstream composition
checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = 1


def sig(blob: bytes, key: str | None) -> str:
    if not key:
        return ""
    h = hashlib.sha256()
    h.update((key + "|").encode("utf-8"))
    h.update(blob)
    return h.hexdigest()


def _action_policy(actions: Iterable[str]) -> list[dict[str, str]]:
    return [{"if": "*", "do": action} for action in actions]


def as_skill(
    candidate: dict[str, Any],
    dataset_path: str,
    run_id: str,
    sign_key: str | None,
) -> dict[str, Any]:
    """Convert a candidate JSON object into a signed skill document."""

    name_slug = candidate["name"].replace("::", "__").replace("+", "__")
    actions = list(candidate.get("actions", []))
    doc: dict[str, Any] = {
        "name": f"opt__{name_slug}",
        "domain": candidate.get("domain"),
        "version": SCHEMA_VERSION,
        "preconditions": ["*"],
        "postconditions": ["progress:+1"],
        "policy": _action_policy(actions),
        "provenance": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "source": "evaluation_candidates",
            "evaluation_file": dataset_path,
            "run_id": run_id,
            "support": candidate.get("support", 0),
            "success_rate": candidate.get("success_rate"),
            "tasks": candidate.get("tasks", []),
        },
    }
    blob = json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    doc["signature"] = sig(blob, sign_key)
    return doc


def _write_json(target: Path, payload: Any) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(target)


def register_skills(
    evaluation_json_path: str,
    outdir: str,
    run_id: str,
    limit: int | None,
) -> dict[str, Any]:
    """Create skill documents from the supplied evaluation summary."""

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    payload = json.loads(Path(evaluation_json_path).read_text(encoding="utf-8"))
    candidates: list[dict[str, Any]] = list(payload.get("candidates", []))
    if limit is not None:
        candidates = candidates[:limit]
    sign_key = os.environ.get("PROOF_SIGN_KEY")

    written: list[str] = []
    for candidate in candidates:
        doc = as_skill(candidate, payload.get("evaluation_file", ""), run_id, sign_key)
        dest = out_path / f"{doc['name']}.json"
        _write_json(dest, doc)
        written.append(str(dest))

    capabilities = payload.get("tasks", [])
    _write_json(out_path / "_capabilities.json", capabilities)

    summary = {"ok": True, "count": len(written), "dir": str(out_path.resolve()), "skills": written}
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-json",
        required=True,
        help="Path to the JSON summary emitted by experiments.skills.run",
    )
    parser.add_argument("--outdir", default="skills/registry", help="Directory to store generated skills")
    parser.add_argument("--run-id", default="live_eval_20251108", help="Run identifier recorded in provenance")
    parser.add_argument("--limit", type=int, help="If provided, limit the number of skills emitted")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = register_skills(args.eval_json, args.outdir, args.run_id, args.limit)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI tests
    raise SystemExit(main())
