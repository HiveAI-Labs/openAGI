"""World model package for planning and simulation.

Provides tree search algorithms and planning capabilities using learned
world models to reduce LLM dependency.

Modules:
- tree_search: Beam search and MCTS algorithms
- planner: World model-based planner
- workspace_sim: Workspace simulation (imported from parent)

Author: OneBrain Team
Created: November 3, 2025
"""

from __future__ import annotations

# Make workspace_sim available from this package
try:
    from brain.universe.workspace_sim import Action, WorkspaceSim
except ImportError:
    # Fallback if workspace_sim not available
    WorkspaceSim = None
    Action = None

try:
    from brain.world_model.concept_memory import ConceptMemory, ConceptMemoryRecord
except ImportError:
    ConceptMemory = None
    ConceptMemoryRecord = None

try:
    from brain.world_model.causal_graph import build_default_workspace_graph
except ImportError:
    build_default_workspace_graph = None

try:
    from brain.world_model.multidomain import MultiDomainWorldModel
    from brain.world_model.simulator_hub import (
        EnvironmentSpec,
        SimulatorHub,
        SimulatorRequest,
        SimulatorResult,
    )
except ImportError:  # pragma: no cover - optional import guard
    MultiDomainWorldModel = None
    EnvironmentSpec = None
    SimulatorHub = None
    SimulatorRequest = None
    SimulatorResult = None

__all__ = [
    "WorkspaceSim",
    "Action",
    "ConceptMemory",
    "ConceptMemoryRecord",
    "build_default_workspace_graph",
    "MultiDomainWorldModel",
    "SimulatorHub",
    "EnvironmentSpec",
    "SimulatorRequest",
    "SimulatorResult",
]
