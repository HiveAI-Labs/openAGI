from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _artifacts_root() -> Path:
    env_dir = os.getenv("ARTIFACTS_DIR") or os.getenv("BRAIN_ARTIFACTS_DIR")
    return Path(env_dir) if env_dir else Path("artifacts")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _tail_lines(path: Path, n: int) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if n > 0 and len(lines) > n:
            lines = lines[-n:]
        return lines
    except Exception:
        return []


def get_latest() -> dict[str, Any]:
    root = _artifacts_root()
    latest_path = root / "consistency" / "ledger" / "latest.json"
    doc = _read_json(latest_path)
    if doc is None:
        return {"ok": True, "found": False}
    return {"ok": True, "found": True, "latest": doc}


def get_stats(*, window: int = 50) -> dict[str, Any]:
    root = _artifacts_root()
    ledger_path = root / "consistency" / "ledger" / "ledger.jsonl"
    lines = _tail_lines(ledger_path, window)
    vals: list[int] = []
    for ln in lines:
        try:
            rec = json.loads(ln)
        except Exception:
            continue
        try:
            vals.append(int(rec.get("contradictions_total") or 0))
        except Exception:
            vals.append(0)
    if not vals:
        return {"ok": True, "found": False, "window": 0, "avg": 0.0, "min": 0, "max": 0, "trend": []}
    return {
        "ok": True,
        "found": True,
        "window": len(vals),
        "avg": sum(vals) / len(vals),
        "min": min(vals),
        "max": max(vals),
        "trend": vals,
    }
