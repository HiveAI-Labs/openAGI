"""Registry for multiple deterministic universes and adapter loading."""

from __future__ import annotations

from collections.abc import Callable

from .abi import Action, Observation, SeedSpec, StepResult
from .runtime import DeterministicGridUniverse, UniverseConfig


class UniverseAdapter:
    """Standard interface for universe implementations."""

    def __init__(self, config: dict | None = None) -> None:
        raise NotImplementedError

    def reset(self, seed: SeedSpec | None = None) -> Observation:
        raise NotImplementedError

    def step(self, action: Action) -> StepResult:
        raise NotImplementedError


class GridAdapter(UniverseAdapter):
    """Adapter that wraps DeterministicGridUniverse."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = UniverseConfig(**(config or {}))
        self._env = DeterministicGridUniverse(cfg)

    def reset(self, seed: SeedSpec | None = None) -> Observation:
        return self._env.reset(seed)

    def step(self, action: Action) -> StepResult:
        return self._env.step(action)


_REGISTRY: dict[str, Callable[[dict | None], UniverseAdapter]] = {
    "grid": GridAdapter,
}


def register_universe(name: str, factory: Callable[[dict | None], UniverseAdapter]) -> None:
    if not name or not callable(factory):
        raise ValueError("name and callable factory required")
    _REGISTRY[name] = factory


def available_universes() -> dict[str, Callable[[dict | None], UniverseAdapter]]:
    return dict(_REGISTRY)


def create_universe(name: str, config: dict | None = None) -> UniverseAdapter:
    factory = _REGISTRY.get(name)
    if not factory:
        raise ValueError(f"unknown universe: {name}")
    return factory(config)

