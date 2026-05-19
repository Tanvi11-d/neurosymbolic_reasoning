"""NRE internal DSL — :class:`Engine` and the :class:`DomainContext` language surface.

Domain logic (the turn-level pipeline) lives in
:mod:`nre.examples.agentic_reasoner` and is registered on an
:class:`Engine` via :func:`~nre.examples.agentic_reasoner.register_agent_turn`.
"""

from .engine import Engine
from .context import (
    DomainContext,
    Var,
    PipeResult,
    WhenBlock,
    EachBlock,
    TimesBlock,
    UntilBlock,
)

__all__ = [
    "Engine",
    "DomainContext",
    "Var",
    "PipeResult",
    "WhenBlock",
    "EachBlock",
    "TimesBlock",
    "UntilBlock",
]
