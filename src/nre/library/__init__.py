"""CoreThink primitive library — concrete primitive implementations.

Tool-calling kernel stages live in :mod:`nre.library.hybrid` and
:mod:`nre.library.deterministic` alongside demo / algorithmic building blocks.
"""

from .deterministic import (
    RuleEngine,
    Unifier,
    ConstraintSolver,
    CycleDetect,
    TopologicalSort,
    KnowledgeBase,
    InitScratchpad,
    InitScratchpadStateManagement,
    topological_levels,
)
from .hybrid import (
    Embedder,
    Classifier,
    Similarity,
    ScoredRule,
    GuidedAttention,
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)

__all__ = [
    # Deterministic
    "RuleEngine",
    "Unifier",
    "ConstraintSolver",
    "CycleDetect",
    "TopologicalSort",
    "topological_levels",
    "KnowledgeBase",
    "InitScratchpad",
    "InitScratchpadStateManagement",
    # Hybrid (demo + kernel)
    "Embedder",
    "Classifier",
    "Similarity",
    "ScoredRule",
    "GuidedAttention",
    "DecomposeInput",
    "OrderUnitTasks",
    "CheckPrerequisites",
    "MatchFuncsAndParams",
]
