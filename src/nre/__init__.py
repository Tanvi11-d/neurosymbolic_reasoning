"""CoreThink — composable neural + symbolic primitives with a Python DSL."""

from .dsl import Engine, DomainContext, Var, PipeResult
from .primitives import (
    CheckPrerequisites,
    DecomposeInput,
    InitScratchpad,
    MatchFuncsAndParams,
    MonotonicScratchpad,
    OrderUnitTasks,
    Primitive,
    PrimitiveRegistry,
    DeterministicPrimitive,
    HybridPrimitive,
    Fact,
    Rule,
    Constraint,
    UnitTask,
)
from .library import (
    RuleEngine,
    Unifier,
    ConstraintSolver,
    CycleDetect,
    TopologicalSort,
    KnowledgeBase,
    Embedder,
    Classifier,
    Similarity,
    ScoredRule,
    GuidedAttention,
)

__all__ = [
    # Engine & DSL
    "Engine",
    "DomainContext",
    "Var",
    "PipeResult",
    # Primitive declarations
    "Primitive",
    "PrimitiveRegistry",
    "DeterministicPrimitive",
    "HybridPrimitive",
    "Fact",
    "Rule",
    "Constraint",
    "UnitTask",
    "InitScratchpad",
    "MonotonicScratchpad",
    "OrderUnitTasks",
    "CheckPrerequisites",
    "DecomposeInput",
    "MatchFuncsAndParams",
    # Library — deterministic
    "RuleEngine",
    "Unifier",
    "ConstraintSolver",
    "CycleDetect",
    "TopologicalSort",
    "KnowledgeBase",
    # Library — hybrid
    "Embedder",
    "Classifier",
    "Similarity",
    "ScoredRule",
    "GuidedAttention",
]
