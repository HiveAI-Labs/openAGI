"""Strict data models describing the Universe runtime ABI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Limits:
    """Resource budgets allowed per environment tick."""

    cpu_ms: int
    gpu_ms: int
    memory_mb: int
    io_bytes: int

    def as_dict(self) -> dict[str, int]:
        return {
            "cpu_ms": self.cpu_ms,
            "gpu_ms": self.gpu_ms,
            "memory_mb": self.memory_mb,
            "io_bytes": self.io_bytes,
        }


@dataclass(frozen=True)
class Observation:
    """Observation payload returned to the agent."""

    grid: tuple[tuple[int, ...], ...]
    agent_pos: tuple[int, int]
    goal_pos: tuple[int, int]
    step: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "grid": self.grid,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
            "step": self.step,
        }


@dataclass(frozen=True)
class Action:
    """Agent action interface."""

    move: str  # "UP", "DOWN", "LEFT", "RIGHT", "STAY"

    def as_dict(self) -> dict[str, str]:
        return {"move": self.move}

    @staticmethod
    def valid_moves() -> Iterable[str]:
        return ("UP", "DOWN", "LEFT", "RIGHT", "STAY")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Action:
        move = str(payload.get("move", "")).upper()
        if move not in cls.valid_moves():
            raise ValueError(f"invalid move: {move!r}")
        return cls(move=move)


@dataclass(frozen=True)
class StepResult:
    """Result returned by Universe.step()."""

    observation: Observation
    reward: float
    done: bool
    info: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.as_dict(),
            "reward": self.reward,
            "done": self.done,
            "info": dict(self.info),
        }


@dataclass(frozen=True)
class SeedSpec:
    """Deterministic seed partitioning used for reproducibility."""

    global_seed: int
    universe_seed: int
    episode_seed: int
    agent_seed: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.global_seed,
            self.universe_seed,
            self.episode_seed,
            self.agent_seed,
        )

