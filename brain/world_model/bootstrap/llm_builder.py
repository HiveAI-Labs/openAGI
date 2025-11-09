from __future__ import annotations

from collections.abc import Sequence
import json
import os
from typing import Any
import urllib.parse

from pydantic import ValidationError

from brain.core.model_client import ModelClient
from brain.core.ollama_client import OllamaChatClient

from .schemas import ActionHypothesis, DomainSnapshot


class LLMWorldModelBuilder:
    """Safety-minded LLM teacher that proposes navigation rules.

    Guards:
    - 10s default timeout for LLM calls
    - localhost-only by default (host allowlist)
    - Strict JSON schema validation for returned hypotheses
    """

    def __init__(
        self,
        client: ModelClient | None = None,
        *,
        timeout_s: float = 10.0,
        allow_hosts: tuple[str, ...] = ("localhost", "127.0.0.1"),
        model: str | None = None,
        base_url: str | None = None,
        extra_providers: Sequence[Any] | None = None,
    ) -> None:
        if client is None:
            base_url = (base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
            self._enforce_allowlist(base_url, allow_hosts)
            client = OllamaChatClient(base_url=base_url, model=(model or os.getenv("OLLAMA_MODEL") or "qwen2.5-coder"), timeout_s=float(timeout_s), retries=0)
        else:
            # Best-effort: attempt to read base_url and enforce allowlist if present
            try:
                bu = getattr(client, "base_url", None)
                if isinstance(bu, str) and bu.strip():
                    self._enforce_allowlist(bu, allow_hosts)
            except Exception:
                pass
        self.client = client
        self.timeout_s = float(timeout_s)
        self._extra_providers = list(extra_providers or [])

    @staticmethod
    def _enforce_allowlist(url: str, allow_hosts: tuple[str, ...]) -> None:
        try:
            host = urllib.parse.urlparse(url).hostname or ""
        except Exception:
            host = ""
        if host not in allow_hosts:
            raise ValueError(f"LLM endpoint host '{host}' not in allowlist {allow_hosts}")

    def propose_navigation_rules(self, domain_snapshot: DomainSnapshot, *, temperature: float = 0.2, max_tokens: int = 768) -> list[ActionHypothesis]:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a careful assistant that proposes symbolic navigation rules. "
                    "Return ONLY JSON. No prose. Output a JSON array of objects with keys: "
                    "action, preconditions, effects, confidence, rationale, test_cases. "
                    "test_cases items have: start_room, goal_room, max_steps, seed."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "domain": {
                            "facts": list(domain_snapshot.facts or []),
                            "notes": domain_snapshot.notes or "",
                        },
                        "constraints": {
                            "return": "json_array_only",
                            "confidence_range": [0.0, 1.0],
                            "max_test_cases": 8,
                            "no_network": True,
                        },
                        "example_schema": {
                            "action": "unlock",
                            "preconditions": ["has(key)", "at(hall)", "door(vault)"],
                            "effects": ["door_unlocked(vault)"],
                            "confidence": 0.7,
                            "rationale": "Key unlocks the vault door from hall",
                            "test_cases": [
                                {"start_room": "hall", "goal_room": "vault", "max_steps": 24, "seed": 1},
                            ],
                        },
                    },
                ),
            },
        ]

        text, usage = self.client.generate(messages, temperature=temperature, max_tokens=max_tokens)
        raw = self._extract_json(text)
        items = raw if isinstance(raw, list) else []
        out: list[ActionHypothesis] = []
        for item in items:
            try:
                hyp = ActionHypothesis.model_validate(item)
            except ValidationError:
                continue
            # Enforce max test cases limit defensively
            if len(hyp.test_cases) > 8:
                hyp.test_cases = hyp.test_cases[:8]
            out.append(hyp)
        extras = self._collect_extra_hypotheses(domain_snapshot)
        if extras:
            combined = {}
            for hyp in out:
                combined[hyp.fingerprint()] = hyp
            for hyp in extras:
                combined.setdefault(hyp.fingerprint(), hyp)
            out = list(combined.values())
        return out

    @staticmethod
    def _extract_json(text: str):
        """Attempt to parse text into JSON, tolerating wrappers by scanning for [] or {} blocks."""
        text = text.strip()
        # Fast path
        try:
            return json.loads(text)
        except Exception:
            pass
        # Scan for an array
        arr = LLMWorldModelBuilder._scan_brackets(text, "[", "]")
        if arr is not None:
            try:
                return json.loads(arr)
            except Exception:
                pass
        # Fallback: scan for a single object
        obj = LLMWorldModelBuilder._scan_brackets(text, "{", "}")
        if obj is not None:
            try:
                return json.loads(obj)
            except Exception:
                pass
        return []

    @staticmethod
    def _scan_brackets(text: str, open_b: str, close_b: str) -> str | None:
        start = text.find(open_b)
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == open_b:
                depth += 1
            elif ch == close_b:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _collect_extra_hypotheses(self, snapshot: DomainSnapshot) -> list[ActionHypothesis]:
        collected: list[ActionHypothesis] = []
        for provider in self._extra_providers:
            try:
                if hasattr(provider, "collect"):
                    extra = provider.collect(snapshot)
                else:
                    extra = provider(snapshot)  # type: ignore[operator]
                if extra:
                    collected.extend(list(extra))
            except Exception:
                continue
        return collected
