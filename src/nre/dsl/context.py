"""DSL context objects — the language surface for CoreThink programs."""

from __future__ import annotations

from typing import Any, Callable, Iterator

from ..primitives.base import Primitive


# ── Pipe result (makes >> chainable) ─────────────────────────────────

class PipeResult:
    """Wraps a value so it can continue a ``>>`` chain."""

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __rshift__(self, other: Any) -> PipeResult:
        if isinstance(other, Primitive):
            return PipeResult(other(self.value))
        if callable(other):
            return PipeResult(other(self.value))
        raise TypeError(f"Cannot pipe into {type(other).__name__}")

    def __repr__(self) -> str:
        return f"PipeResult({self.value!r})"


# ── DSL Variable ─────────────────────────────────────────────────────

class Var:
    """A named, typed variable in the DSL.

    Supports:
    - ``<<`` for assignment:  ``x << 42``
    - ``>>`` for piping:      ``x >> embed >> classify``
    - Comparisons return real bools for use in ``when()``
    """

    def __init__(self, name: str, type_hint: type = object, default: Any = None) -> None:
        self.name = name
        self.type_hint = type_hint
        self._value: Any = default

    # ── value access ──────────────────────────────────────────────

    @property
    def val(self) -> Any:
        return self._value

    @property
    def value(self) -> Any:
        return self._value

    # ── << assignment ─────────────────────────────────────────────

    def __lshift__(self, value: Any) -> Var:
        """Assign: ``x << 42``."""
        actual = value.value if isinstance(value, (Var, PipeResult)) else value
        if self.type_hint is not object and not isinstance(actual, self.type_hint):
            raise TypeError(
                f"{self.name}: expected {self.type_hint.__name__}, "
                f"got {type(actual).__name__}"
            )
        self._value = actual
        return self

    def set(self, value: Any) -> None:
        """Imperative alias for <<."""
        self << value

    # ── >> pipe ───────────────────────────────────────────────────

    def __rshift__(self, other: Any) -> PipeResult:
        """Pipe: ``x >> embed`` calls embed(x.value)."""
        if isinstance(other, Primitive):
            return PipeResult(other(self._value))
        if callable(other):
            return PipeResult(other(self._value))
        raise TypeError(f"Cannot pipe into {type(other).__name__}")

    # ── comparisons (return real bools for when()) ────────────────

    def _resolve(self, other: Any) -> Any:
        return other._value if isinstance(other, Var) else other

    def __gt__(self, other: Any) -> bool:
        return self._value > self._resolve(other)

    def __lt__(self, other: Any) -> bool:
        return self._value < self._resolve(other)

    def __ge__(self, other: Any) -> bool:
        return self._value >= self._resolve(other)

    def __le__(self, other: Any) -> bool:
        return self._value <= self._resolve(other)

    def __eq__(self, other: Any) -> bool:
        return self._value == self._resolve(other)

    def __ne__(self, other: Any) -> bool:
        return self._value != self._resolve(other)

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        return f"Var({self.name!r}, {self._value!r})"


# ── Let descriptor (makes ctx.let[type] work) ────────────────────────

class _LetDescriptor:
    """Accessed as ``ctx.let[float]`` to declare a typed variable.

    Returns a fresh :class:`Var` whose name is set by the caller via
    ``__lshift__`` or left as auto-generated.
    """

    _counter: int = 0

    def __init__(self, ctx: DomainContext) -> None:
        self._ctx = ctx

    def __getitem__(self, type_hint: type) -> Var:
        _LetDescriptor._counter += 1
        name = f"_v{_LetDescriptor._counter}"
        var = Var(name, type_hint)
        self._ctx._vars[name] = var
        return var


# ── Loop blocks ──────────────────────────────────────────────────────

class EachBlock:
    """``with ctx.each(items) as item:``"""

    def __init__(self, iterable: Any) -> None:
        self._iterable = iterable
        self.iteration = 0

    def __enter__(self) -> EachBlock:
        self.iteration = 0
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def __iter__(self) -> Iterator[Any]:
        src = self._iterable
        if isinstance(src, Var):
            src = src.value
        for item in src:
            self.iteration += 1
            yield item


class TimesBlock:
    """``with ctx.times(n) as i:``"""

    def __init__(self, n: int) -> None:
        self._n = n
        self.iteration = 0

    def __enter__(self) -> TimesBlock:
        self.iteration = 0
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def __iter__(self) -> Iterator[int]:
        for i in range(self._n):
            self.iteration = i
            yield i


class UntilBlock:
    """``with ctx.until(lambda: done) as step:``"""

    def __init__(self, predicate: Callable[[], bool], max_iter: int = 1000) -> None:
        self._predicate = predicate
        self._max_iter = max_iter
        self.iteration = 0

    def __enter__(self) -> UntilBlock:
        self.iteration = 0
        return self

    def __exit__(self, *exc: Any) -> None:
        pass

    def __iter__(self) -> Iterator[int]:
        for i in range(self._max_iter):
            if self._predicate():
                break
            self.iteration = i
            yield i


# ── Conditional blocks ───────────────────────────────────────────────

class WhenBlock:
    """``with ctx.when(condition):`` — body runs only if truthy."""

    def __init__(self, condition: bool, ctx: DomainContext) -> None:
        self.condition = bool(condition)
        self._ctx = ctx

    def __enter__(self) -> bool:
        self._ctx._last_when = self.condition
        return self.condition

    def __exit__(self, *exc: Any) -> None:
        pass


class _OtherwiseDescriptor:
    """``with ctx.otherwise:`` — runs when the last ``when`` was False."""

    def __init__(self, ctx: DomainContext) -> None:
        self._ctx = ctx

    def __enter__(self) -> bool:
        cond = not self._ctx._last_when
        return cond

    def __exit__(self, *exc: Any) -> None:
        pass


# ── Domain Context ───────────────────────────────────────────────────

class DomainContext:
    """The execution context — this IS the language surface.

    DSL cheat-sheet::

        # declare
        x = ctx.let[float] << 0.0

        # pipe
        result = x >> ctx.embed >> ctx.classify

        # loop
        with ctx.each(items) as item:  ...
        with ctx.times(5) as i:        ...
        with ctx.until(pred) as step:  ...

        # conditional
        with ctx.when(x > 0.5) as ok:
            if ok: ...
        with ctx.otherwise as fallback:
            if fallback: ...

        # return
        return ctx.returns(x, y, extra=z)

        # init a new primitive inline
        emb = ctx.init(Embedder, "emb2", dim=64)

        # trace / debug
        ctx.trace("step_name", key=value)

        # connect primitives declaratively
        ctx.connect("embed", "classify")
    """

    def __init__(self, engine: Any, inputs: dict[str, Any]) -> None:
        self._engine = engine
        self._inputs = dict(inputs)
        self._vars: dict[str, Var] = {}
        self._pipeline: list[tuple[str, str]] = []
        self._trace_log: list[dict[str, Any]] = []
        self._last_when: bool = True

        # Expose inputs as pre-declared vars
        for k, v in inputs.items():
            var = Var(k, type(v), v)
            self._vars[k] = var

    # ── let ───────────────────────────────────────────────────────

    @property
    def let(self) -> _LetDescriptor:
        """Declare a typed variable: ``x = ctx.let[float] << 0.0``"""
        return _LetDescriptor(self)

    # ── loops ─────────────────────────────────────────────────────

    @staticmethod
    def each(iterable: Any) -> EachBlock:
        """Iterate over a collection: ``with ctx.each(items) as loop:``"""
        return EachBlock(iterable)

    @staticmethod
    def times(n: int) -> TimesBlock:
        """Counted loop: ``with ctx.times(5) as loop:``"""
        return TimesBlock(n)

    @staticmethod
    def until(predicate: Callable[[], bool], max_iter: int = 1000) -> UntilBlock:
        """Loop until predicate is true: ``with ctx.until(pred) as loop:``"""
        return UntilBlock(predicate, max_iter)

    # ── conditionals ──────────────────────────────────────────────

    def when(self, condition: bool) -> WhenBlock:
        """Conditional: ``with ctx.when(x > 0.5) as ok:``"""
        return WhenBlock(condition, self)

    @property
    def otherwise(self) -> _OtherwiseDescriptor:
        """Else branch: ``with ctx.otherwise as fallback:``"""
        return _OtherwiseDescriptor(self)

    # ── init primitive inline ─────────────────────────────────────

    def init(self, primitive_cls: type, name: str | None = None, **kwargs: Any) -> Any:
        prim = primitive_cls(name=name, **kwargs)
        self._engine.registry.register(prim)
        return prim

    # ── returns ───────────────────────────────────────────────────

    def returns(self, *positional: Any, **named: Any) -> dict[str, Any]:
        """Package return values.

        Positional :class:`Var` args use their name as key.
        All Var / PipeResult values are auto-resolved.
        """
        result: dict[str, Any] = {}
        for arg in positional:
            if isinstance(arg, Var):
                result[arg.name] = arg.value
            elif isinstance(arg, PipeResult):
                result[f"result_{len(result)}"] = arg.value
            else:
                result[f"result_{len(result)}"] = arg
        for key, val in named.items():
            if isinstance(val, Var):
                result[key] = val.value
            elif isinstance(val, PipeResult):
                result[key] = val.value
            else:
                result[key] = val
        result["_trace"] = list(self._trace_log)
        return result

    # ── connections ───────────────────────────────────────────────

    def connect(self, source: str, target: str) -> None:
        """Declare a dataflow edge between two primitives."""
        self._pipeline.append((source, target))

    def pipe(self, *primitives: Any) -> PipeResult:
        """Explicit pipeline: ``ctx.pipe(a, b, c)``."""
        result: Any = None
        for i, prim in enumerate(primitives):
            result = prim if i == 0 else prim(result)
        return PipeResult(result)

    # ── tracing ───────────────────────────────────────────────────

    def trace(self, label: str, **data: Any) -> None:
        entry: dict[str, Any] = {"step": label}
        for k, v in data.items():
            entry[k] = v.value if isinstance(v, (Var, PipeResult)) else v
        self._trace_log.append(entry)

    # ── primitive dispatch ────────────────────────────────────────

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        try:
            return self._engine.registry.get(name)
        except KeyError:
            raise AttributeError(
                f"DomainContext has no attribute or primitive {name!r}"
            )
