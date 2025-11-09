"""Derive navigation hypotheses from stored ensemble insights."""

from __future__ import annotations

from collections.abc import Sequence
import json
import os
from pathlib import Path
import re

from .schemas import ActionHypothesis, DomainSnapshot, TestCase

__all__ = ["EnsembleHypothesisProvider"]


class EnsembleHypothesisProvider:
    """Lightweight provider that interprets ensemble insights as hypotheses."""

    def __init__(
        self,
        *,
        artifacts_dir: str | os.PathLike[str] | None = None,
        max_records: int = 5,
    ) -> None:
        base_dir = Path(artifacts_dir or os.getenv("BRAIN_ARTIFACTS_DIR") or "artifacts")
        self._archive_dir = base_dir.resolve() / "ensemble" / "archive"
        self._max_records = max(1, int(max_records))
        self._action_pattern = re.compile(r"Action:\s*(.+)", re.IGNORECASE)

    def collect(self, snapshot: DomainSnapshot) -> list[ActionHypothesis]:
        if not self._archive_dir.exists():
            return []
        candidates: list[ActionHypothesis] = []
        records = sorted(self._archive_dir.glob("insight_*.jsonl"), reverse=True)
        for path in records[: self._max_records]:
            try:
                text = path.read_text(encoding="utf-8").strip()
                if not text:
                    continue
                record = json.loads(text)
            except Exception:
                continue
            answer_payload = record.get("answer")
            if isinstance(answer_payload, dict):
                answer_text = str(answer_payload.get("answer") or "")
                provenance = answer_payload.get("provenance") or []
            else:
                answer_text = str(answer_payload or "")
                provenance = []

            candidates.extend(self._from_text(answer_text, snapshot))
            for entry in provenance:
                content = entry.get("content")
                if isinstance(content, str):
                    candidates.extend(self._from_text(content, snapshot))
        # deduplicate by fingerprint
        unique: dict[str, ActionHypothesis] = {}
        for hyp in candidates:
            unique.setdefault(hyp.fingerprint(), hyp)
        return list(unique.values())

    def _from_text(self, content: str, snapshot: DomainSnapshot) -> list[ActionHypothesis]:
        matches = self._action_pattern.findall(content or "")
        if not matches:
            return []
        hypotheses: list[ActionHypothesis] = []
        for raw_action in matches:
            action_name = self._normalize_action(raw_action)
            if not action_name:
                continue
            hypotheses.append(self._build_hypothesis(action_name, snapshot))
        return hypotheses

    @staticmethod
    def _normalize_action(raw: str) -> str:
        name = raw.strip().split("\n")[0]
        name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return name or ""

    def _build_hypothesis(self, action: str, snapshot: DomainSnapshot) -> ActionHypothesis:
        start_room = self._first_room(snapshot)
        target_room = self._second_room(snapshot) or start_room
        test_case = TestCase(
            description=f"Validate ensemble action '{action}'",
            initial_state={"room": start_room},
            expected_outcome={"room": target_room},
            max_steps=8,
            seed=0,
            timeout_s=2.0,
        )
        return ActionHypothesis(
            action=action or "ensemble_action",
            parameters={},
            preconditions=[f"ensemble_insight({action})"],
            effects=[f"potential_transition({target_room})"],
            confidence=0.55,
            validation_steps=[
                "Step 1: replay ensemble insight in simulation.",
                "Step 2: confirm resulting room matches expected outcome.",
            ],
            test_cases=[test_case],
        )

    @staticmethod
    def _first_room(snapshot: DomainSnapshot) -> str:
        rooms = getattr(snapshot, "rooms", None)
        if isinstance(rooms, Sequence) and rooms:
            return str(rooms[0])
        return "start_room"

    @staticmethod
    def _second_room(snapshot: DomainSnapshot) -> str | None:
        rooms = getattr(snapshot, "rooms", None)
        if isinstance(rooms, Sequence) and len(rooms) > 1:
            return str(rooms[1])
        return None
