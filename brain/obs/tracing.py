
from __future__ import annotations

from contextlib import contextmanager
import json
import threading
import time

__all__ = ["Tracer","Span","global_tracer"]

class Span:
    __slots__ = ("name","start","end","tags")
    def __init__(self, name: str) -> None:
        self.name = name; self.start = time.time(); self.end = None; self.tags: dict[str, float | str] = {}
    def finish(self):
        if self.end is None: self.end = time.time()
    def duration(self) -> float:
        return (self.end or time.time()) - self.start

class Tracer:
    def __init__(self, max_spans: int = 10000, sink_path: str | None = None) -> None:
        self._lock = threading.Lock()
        self._spans: list[Span] = []
        self.max_spans = max_spans
        self.sink_path = sink_path
    @contextmanager
    def span(self, name: str):
        sp = Span(name)
        try:
            yield sp
        finally:
            sp.finish()
            with self._lock:
                self._spans.append(sp)
                if len(self._spans) > self.max_spans:
                    self._spans.pop(0)
                if self.sink_path:
                    try:
                        with open(self.sink_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"name": sp.name, "start": sp.start, "end": sp.end, "duration": sp.duration(), "tags": sp.tags}) + "\n")
                    except Exception:
                        pass
    def recent(self, n: int = 100) -> list[Span]:
        with self._lock:
            return list(self._spans[-n:])

_global = Tracer()
def global_tracer() -> Tracer: return _global
