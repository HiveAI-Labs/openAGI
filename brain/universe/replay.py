"""Replay utilities for deterministic universes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

from .abi import Action, Observation, SeedSpec, StepResult
from .runtime import DeterministicGridUniverse, UniverseConfig


@dataclass
class ReplayEvent:
    """Recorded step information for deterministic replay."""

    step: int
    action: dict
    result: dict

    def to_json(self) -> str:
        return json.dumps({"step": self.step, "action": self.action, "result": self.result})


def run_episode(
    actions: Iterable[Action],
    *,
    seed: SeedSpec,
    config: UniverseConfig | None = None,
) -> list[ReplayEvent]:
    """Execute a full episode and return its replay events."""
    env = DeterministicGridUniverse(config=config)
    init_obs: Observation = env.reset(seed)
    events: list[ReplayEvent] = [
        ReplayEvent(step=0, action={}, result={"observation": init_obs.as_dict(), "reward": 0.0, "done": False, "info": {}}),
    ]

    for idx, action in enumerate(actions, start=1):
        result: StepResult = env.step(action)
        events.append(
            ReplayEvent(
                step=idx,
                action=action.as_dict(),
                result=result.as_dict(),
            ),
        )
        if result.done:
            break
    return events


def save_replay(events: Iterable[ReplayEvent], path: str | Path) -> None:
    """Persist replay events to JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(ev.to_json() + "\n")


def load_replay(path: str | Path) -> list[ReplayEvent]:
    """Load replay events from JSONL."""
    out: list[ReplayEvent] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            payload = json.loads(line)
            out.append(
                ReplayEvent(
                    step=payload["step"],
                    action=payload.get("action", {}),
                    result=payload["result"],
                ),
            )
    return out


def verify_replay(
    events: Iterable[ReplayEvent],
    *,
    seed: SeedSpec,
    config: UniverseConfig | None = None,
) -> bool:
    """Verify recorded events re-execute identically."""
    env = DeterministicGridUniverse(config=config)
    env.reset(seed)
    iterator = iter(events)

    try:
        first = next(iterator)
    except StopIteration:
        return False
    expected_obs = first.result.get("observation")
    if expected_obs is None:
        return False
    if _normalize(env._build_observation().as_dict()) != _normalize(expected_obs):  # type: ignore[attr-defined]
        return False

    for ev in iterator:
        action = Action.from_payload(ev.action or {"move": "STAY"})
        result = env.step(action)
        if _normalize(result.as_dict()) != _normalize(ev.result):
            return False
        if result.done:
            break
    return True


def _normalize(obj: dict) -> dict:
    return json.loads(json.dumps(obj, sort_keys=True))
