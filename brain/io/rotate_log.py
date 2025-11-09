
from __future__ import annotations

import os
from pathlib import Path


class RotatingLog:
    def __init__(self, path: str | os.PathLike, max_bytes: int = 1_000_000, backups: int = 3) -> None:
        self.path = Path(path); self.max_bytes = int(max_bytes); self.backups = int(backups)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists(): self.path.write_text("")
    def _rotate(self) -> None:
        oldest = self.path.with_suffix(self.path.suffix + f".{self.backups}")
        if oldest.exists(): oldest.unlink()
        for i in range(self.backups-1, 0, -1):
            src = self.path.with_suffix(self.path.suffix + f".{i}")
            dst = self.path.with_suffix(self.path.suffix + f".{i+1}")
            if src.exists(): src.replace(dst)
        if self.path.exists(): self.path.replace(self.path.with_suffix(self.path.suffix + ".1"))
        self.path.write_text("")
    def append(self, line: str) -> None:
        data = line if line.endswith("\n") else line + "\n"
        if self.path.exists() and self.path.stat().st_size + len(data.encode("utf-8")) > self.max_bytes:
            self._rotate()
        with self.path.open("a", encoding="utf-8") as f:
            f.write(data)
