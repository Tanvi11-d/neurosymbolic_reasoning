"""The CoreThink engine — top-level entry point for defining and running domains."""

from __future__ import annotations

from typing import Any, Callable

from ..primitives.base import Primitive, PrimitiveRegistry
from .context import DomainContext


class Engine:
    """CoreThink reasoning engine.

    Usage::

        engine = Engine()

        # Register primitives
        engine.register(Embedder, name="embed", dim=128)

        # Define domain logic
        @engine.domain("my_domain")
        def my_domain(ctx):
            x = ctx.declare("input_data")
            ...
            return ctx.returns(result=x)

        # Execute
        result = engine.run("my_domain", input_data="hello")
    """

    def __init__(self) -> None:
        self.registry = PrimitiveRegistry()
        self._domains: dict[str, Callable[[DomainContext], Any]] = {}

    # ── Primitive registration ────────────────────────────────────

    def register(
        self,
        primitive_or_cls: Primitive | type,
        name: str | None = None,
        **kwargs: Any,
    ) -> Primitive:
        """Register a primitive instance or class (instantiated with *kwargs*)."""
        if isinstance(primitive_or_cls, type):
            prim = primitive_or_cls(name=name, **kwargs)
        else:
            prim = primitive_or_cls
            if name:
                prim.name = name
        self.registry.register(prim)
        return prim

    # ── Domain definition ─────────────────────────────────────────

    def domain(self, name: str) -> Callable:
        """Decorator that registers a domain-logic function.

        The decorated function receives a :class:`DomainContext` and
        should return ``ctx.returns(...)``.
        """
        def decorator(fn: Callable[[DomainContext], Any]) -> Callable:
            self._domains[name] = fn
            return fn
        return decorator

    def add_domain(self, name: str, fn: Callable[[DomainContext], Any]) -> None:
        """Programmatic (non-decorator) domain registration."""
        self._domains[name] = fn

    # ── Execution ─────────────────────────────────────────────────

    def run(self, domain_name: str, **inputs: Any) -> dict[str, Any]:
        """Execute a named domain with the given inputs."""
        if domain_name not in self._domains:
            raise KeyError(
                f"Domain {domain_name!r} not found. "
                f"Registered: {list(self._domains.keys())}"
            )
        ctx = DomainContext(self, inputs)
        return self._domains[domain_name](ctx)

    # ── Introspection ─────────────────────────────────────────────

    def list_domains(self) -> list[str]:
        return list(self._domains.keys())

    def list_primitives(self) -> list[str]:
        return self.registry.all_names()

    def __repr__(self) -> str:
        return (
            f"<Engine domains={self.list_domains()} "
            f"primitives={self.list_primitives()}>"
        )
