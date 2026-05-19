"""Deterministic primitives: facts, rules, constraints, INIT-SCRATCHPAD, and bases."""

from __future__ import annotations

import abc
from collections import UserDict
from dataclasses import dataclass, field
from typing import Any, Callable

from .base import Primitive

# ── Data structures ──────────────────────────────────────────────────

@dataclass
class Fact:
    predicate: str
    args: tuple[Any, ...]

    def __hash__(self):     return hash((self.predicate, self.args))
    def __eq__(self, other): return NotImplemented if not isinstance(other, Fact) else (self.predicate == other.predicate and self.args == other.args)
    def __repr__(self):      return f"{self.predicate}({', '.join(repr(a) for a in self.args)})"

@dataclass
class Rule:
    """An if-then rule: when *condition* holds, derive *action*."""
    name: str
    condition: Callable[[dict[str, Any]], bool]
    action: Callable[[dict[str, Any]], Any]
    priority: int = 0

    def __repr__(self): return f"Rule({self.name!r}, priority={self.priority})"

@dataclass
class Constraint:
    """A named constraint over a set of variables."""
    name: str
    variables: list[str]
    check: Callable[[dict[str, Any]], bool]

@dataclass
class UnitTask:
    id: str
    text: str = ""
    depends_on: tuple[str, ...] = ()
    tool_name: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)

# ── Abstract base ────────────────────────────────────────────────────

class MonotonicScratchpad(UserDict[str, Any]):
    """Working memory ``S`` — keys may be added or updated, never deleted (INV-S)."""
    _err = "scratchpad keys may not be deleted (INV-S)"

    def __delitem__(self, key):         raise TypeError(self._err)
    def pop(self, key, *args):          raise TypeError(self._err)
    def clear(self):                    raise TypeError("scratchpad may not be cleared (INV-S)")
    def popitem(self):                  raise TypeError(self._err)

def merge_init_scratchpad(
    *,
    prior: dict[str, Any] | None = None,
    gamma_seeds: dict[str, Any] | None = None,
    decompose_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Shallow-merge INIT-SCRATCHPAD layers: prior → gamma_seeds → decompose_config."""
    return {**(prior or {}), **(gamma_seeds or {}), **(decompose_config or {})}

def apply_init_scratchpad(
    seed_from_decompose: dict[str, Any],
    *,
    carry: dict[str, Any] | None = None,
    gamma_seeds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply INIT-SCRATCHPAD merge for one turn (carry updated in-place if given)."""
    if carry is not None:
        if gamma_seeds:
            carry.update(gamma_seeds)
        carry.update(seed_from_decompose)
        return carry
    return merge_init_scratchpad(gamma_seeds=gamma_seeds, decompose_config=seed_from_decompose)

class DeterministicPrimitive(Primitive, abc.ABC):
    """Abstract base for deterministic (pure, no learned params) primitives."""
    primitive_type = "deterministic"

def __getattr__(name: str) -> Any:
    """Lazy-import InitScratchpad from nre.library.deterministic."""
    if name == "InitScratchpad":
        from nre.library.deterministic import InitScratchpad as _InitScratchpad
        return _InitScratchpad
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")