"""Safety filters applied before caching or learning from LLM answers."""

from __future__ import annotations

from collections.abc import Iterable
import re
from typing import Any

PROCEDURAL_KEYWORDS: tuple[str, ...] = (
    "how to",
    "how do",
    "steps to",
    "procedure",
    "process for",
    "method for",
    "workflow",
    "recipe",
    "guide to",
    "instructions",
)

UNCERTAINTY_CUES: tuple[str, ...] = (
    "i think",
    "i believe",
    "maybe",
    "perhaps",
    "probably",
    "might",
    "could",
    "unsure",
    "not sure",
    "i guess",
)

DISALLOWED_PREFIXES: tuple[str, ...] = (
    "as an ai",
    "as a language model",
    "i am an ai",
)


class LLMAnswerSafetyFilter:
    """Best-effort guardrail ensuring only procedural, grounded answers are cached."""

    def __init__(self, *, procedural_only: bool = True) -> None:
        self._procedural_only = procedural_only

    def approve(self, *, task: str | None, args: Any, answer: str) -> bool:
        """Return True when the answer looks safe to reuse."""
        prompt = self._extract_prompt(args)
        combined = " ".join(token for token in (task, prompt) if token).strip().lower()
        text = (answer or "").strip()
        if not text:
            return False
        lowered = text.lower()

        if self._procedural_only and not self._looks_procedural(combined):
            return False
        if any(prefix in lowered for prefix in DISALLOWED_PREFIXES):
            return False
        if self._has_uncertainty(lowered):
            return False
        if not self._has_concrete_structure(text):
            return False
        if len(text.split()) < 6:
            return False
        return True

    def _extract_prompt(self, args: Any) -> str:
        if isinstance(args, dict):
            if "messages" in args and isinstance(args["messages"], Iterable):
                for message in reversed(list(args["messages"])):
                    if isinstance(message, dict) and str(message.get("role", "user")).lower() == "user":
                        return str(message.get("content", ""))
            if "prompt" in args:
                return str(args.get("prompt", ""))
        return str(args)

    def _looks_procedural(self, text: str) -> bool:
        return any(keyword in text for keyword in PROCEDURAL_KEYWORDS)

    def _has_uncertainty(self, lowered_answer: str) -> bool:
        return any(cue in lowered_answer for cue in UNCERTAINTY_CUES)

    def _has_concrete_structure(self, answer: str) -> bool:
        normalized = answer.lower()
        if "step" in normalized:
            return True
        if re.search(r"\b\d+\. ", answer):
            return True
        if normalized.count("\n") >= 1:
            return True
        return False
