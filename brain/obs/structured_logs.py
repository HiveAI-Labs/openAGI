
from __future__ import annotations

import json
import queue
import threading
import time
from typing import Any

import requests


def _parse_labels(lbl: str) -> dict[str, str]:
    out, cur, key, in_str = {}, "", None, False
    for ch in lbl.strip():
        if ch == "{": continue
        if ch == "}":
            if key is not None: out[key] = cur
            break
        if ch == '"':
            in_str = not in_str; continue
        if ch == "=" and not in_str and key is None:
            key, cur = cur, ""; continue
        if ch == "," and not in_str:
            if key is not None: out[key] = cur; key=None; cur=""; continue
        if ch == "\\": continue
        cur += ch
    return out

class _BaseShip:
    def __init__(self) -> None:
        self.q: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1000)
        # Provide a well-typed _run attribute so static checkers know the
        # method exists on instances (subclasses override with the real
        # implementation). The thread target resolves to the bound method on
        # the actual instance (subclass), so runtime behavior is unchanged.
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
    def _run(self) -> None:
        """Placeholder run loop; subclasses should override.

        A no-op implementation is provided only for typing/static analysis.
        """
        return None
    def emit(self, record: dict[str, Any]) -> None:
        try: self.q.put_nowait(record)
        except Exception: pass

class LokiShip(_BaseShip):
    def __init__(self, loki_url: str, labels: str) -> None:
        self.url = loki_url
        self.labels = labels
        super().__init__()
    def _run(self) -> None:
        batch, last = [], time.time()
        while True:
            try: item = self.q.get(timeout=1.0); batch.append(item)
            except Exception: pass
            if batch and (len(batch) >= 50 or (time.time()-last) > 2.0):
                try:
                    streams = [{
                        "stream": _parse_labels(self.labels),
                        "values": [[str(int(r.get("ts", time.time())*1e9)), json.dumps(r)] for r in batch],
                    }]
                    requests.post(self.url, json={"streams": streams}, timeout=2.5)
                except Exception: pass
                batch=[]; last=time.time()

class ElkShip(_BaseShip):
    def __init__(self, url: str) -> None:
        self.url = url
        super().__init__()
    def _run(self) -> None:
        batch, last = [], time.time()
        while True:
            try: item = self.q.get(timeout=1.0); batch.append(item)
            except Exception: pass
            if batch and (len(batch) >= 100 or (time.time()-last) > 2.0):
                try:
                    body = "\n".join(json.dumps(r) for r in batch)
                    requests.post(self.url, data=body, headers={"Content-Type":"application/json"}, timeout=2.5)
                except Exception: pass
                batch=[]; last=time.time()
