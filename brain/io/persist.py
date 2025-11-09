
from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); n = 0
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); n += 1
    return n
def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path); out = []
    if not p.exists(): return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip(): out.append(json.loads(line))
    return out
