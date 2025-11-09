"""Deterministic grid-world universe implementation."""

from __future__ import annotations

from dataclasses import dataclass
import random

from .abi import Action, Limits, Observation, SeedSpec, StepResult


@dataclass
class UniverseConfig:
    """Configuration for the grid universe."""

    size: tuple[int, int] = (5, 5)
    max_steps: int = 40
    step_penalty: float = -0.01
    goal_reward: float = 1.0
    limits: Limits = Limits(cpu_ms=10, gpu_ms=0, memory_mb=32, io_bytes=4096)


class DeterministicGridUniverse:
    """Minimal deterministic universe for fast experiments."""

    def __init__(self, config: UniverseConfig | None = None) -> None:
        self.config = config or UniverseConfig()
        self._rng_global = random.Random()
        self._rng_universe = random.Random()
        self._rng_episode = random.Random()
        self._rng_agent = random.Random()
        self._step = 0
        self._agent_pos = (0, 0)
        self._goal_pos = (self.config.size[0] - 1, self.config.size[1] - 1)

    def reseed(self, seed: SeedSpec) -> None:
        """Reseed the deterministic RNG streams."""
        self._rng_global.seed(seed.global_seed)
        self._rng_universe.seed(seed.universe_seed)
        self._rng_episode.seed(seed.episode_seed)
        self._rng_agent.seed(seed.agent_seed)

    def reset(self, seed: SeedSpec | None = None) -> Observation:
        """Reset the universe state."""
        if seed:
            self.reseed(seed)
        self._step = 0
        self._agent_pos = (
            self._rng_episode.randrange(self.config.size[0]),
            self._rng_episode.randrange(self.config.size[1]),
        )
        gx = self._rng_universe.randrange(self.config.size[0])
        gy = self._rng_universe.randrange(self.config.size[1])
        self._goal_pos = (gx, gy)
        if self._goal_pos == self._agent_pos:
            self._goal_pos = ((gx + 1) % self.config.size[0], (gy + 1) % self.config.size[1])
        return self._build_observation()

    def step(self, action: Action) -> StepResult:
        """Advance simulation by a single step."""
        self._step += 1
        ax, ay = self._agent_pos
        move = action.move.upper()
        if move == "UP":
            ay = max(0, ay - 1)
        elif move == "DOWN":
            ay = min(self.config.size[1] - 1, ay + 1)
        elif move == "LEFT":
            ax = max(0, ax - 1)
        elif move == "RIGHT":
            ax = min(self.config.size[0] - 1, ax + 1)
        self._agent_pos = (ax, ay)

        reward = self.config.step_penalty
        done = False
        info: dict[str, object] = {"limits": self.config.limits.as_dict()}
        if self._agent_pos == self._goal_pos:
            reward += self.config.goal_reward
            done = True
            info["event"] = "goal"
        elif self._step >= self.config.max_steps:
            done = True
            info["event"] = "max_steps"

        obs = self._build_observation()
        return StepResult(observation=obs, reward=reward, done=done, info=info)

    def _build_observation(self) -> Observation:
        width, height = self.config.size
        grid: list[list[int]] = [[0 for _ in range(height)] for _ in range(width)]
        ax, ay = self._agent_pos
        gx, gy = self._goal_pos
        grid[gx][gy] = 2
        grid[ax][ay] = 1
        return Observation(
            grid=tuple(tuple(row) for row in grid),
            agent_pos=self._agent_pos,
            goal_pos=self._goal_pos,
            step=self._step,
        )

