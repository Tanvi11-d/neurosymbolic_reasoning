"""Primitive bases, facts/rules, and tool-calling kernel entry types."""

from .base import (
    HybridPrimitive,
    Primitive,
    PrimitiveLog,
    PrimitiveRegistry,
    ensure_primitive_logging_visible,
)
from .log_buffer import (
    format_primitive_log_for_api,
    get_primitive_log_buffer,
    primitive_log_session,
)
from .deterministic import (
    Constraint,
    DeterministicPrimitive,
    Fact,
    InitScratchpad,
    MonotonicScratchpad,
    Rule,
    UnitTask,
    apply_init_scratchpad,
    merge_init_scratchpad,
)
from .llm_bridge import llm_task_item_to_unit_task
from .hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
    normalize_decompose_result,
)

__all__ = [
    "HybridPrimitive",
    "Primitive",
    "PrimitiveLog",
    "PrimitiveRegistry",
    "ensure_primitive_logging_visible",
    "primitive_log_session",
    "get_primitive_log_buffer",
    "format_primitive_log_for_api",
    "DeterministicPrimitive",
    "Fact",
    "Rule",
    "Constraint",
    "UnitTask",
    "llm_task_item_to_unit_task",
    "InitScratchpad",
    "MonotonicScratchpad",
    "merge_init_scratchpad",
    "apply_init_scratchpad",
    "OrderUnitTasks",
    "CheckPrerequisites",
    "DecomposeInput",
    "MatchFuncsAndParams",
    "normalize_decompose_result",
]
