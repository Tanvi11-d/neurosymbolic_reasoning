"""Shared base classes that every CoreThink primitive builds on."""

from __future__ import annotations

import abc
import logging
import time
from typing import Any

from .log_buffer import (
    append_primitive_log_event,
    get_primitive_log_buffer,
    primitive_log_event,
)

def show_debug_logs():
    """Turn on DEBUG logs for all primitives. Call once at startup."""
    logging.getLogger("nre.primitive").setLevel(logging.DEBUG)

ensure_primitive_logging_visible = show_debug_logs

class PrimitiveLog:
    """Logger attached to every primitive as self.log."""

    __slots__ = ("_primitive", "_logger")

    def __init__(self, primitive: Primitive) -> None:
        self._primitive = primitive
        self._logger = logging.getLogger(f"nre.primitive.{primitive.name}")

    def emit(self, level: int, msg: str, args: tuple[Any, ...]) -> None:
        # Do nothing if logging is turned off for this primitive.
        if not self._primitive.primitive_log_enabled():
            return
        self._logger.log(level, msg, *args)

        if get_primitive_log_buffer() is not None:
            if not args:
                plain_text = str(msg)
            else:
                try:
                    plain_text = msg % args
                except (TypeError, ValueError):
                    plain_text = f"{msg} {args!r}"

            append_primitive_log_event(primitive_log_event(
                ts=time.time(),
                level=logging.getLevelName(level),
                levelno=level,
                primitive=self._primitive.name,
                primitive_class=self._primitive.__class__.__name__,
                message=plain_text,
            ))

    def debug(self, msg, *args):
        self.emit(logging.DEBUG,   msg, args)
    def info(self, msg, *args):
        self.emit(logging.INFO,    msg, args)
    def warning(self, msg, *args):
        self.emit(logging.WARNING, msg, args)
    def error(self, msg, *args):
        self.emit(logging.ERROR,   msg, args)


class Primitive(abc.ABC):
    """Base class for every primitive. Subclasses must implement forward().

    Logging options (pass when creating the primitive):
        log_primitive=True  — master on/off switch.
        log_llm=True        — show LLM call logs.
        log_bodies=True     — show full prompts and raw JSON.
    """

    primitive_type = "base"

    def __init__(self, name: str | None = None, **config: Any) -> None:
        self.name = name or self.__class__.__name__
        self.config = config
        self._initialized = False
        self.log = PrimitiveLog(self)

    def primitive_log_enabled(self) -> bool:
        """Is logging on for this primitive? Default: yes."""
        return bool(self.config.get("log_primitive", True))

    def log_llm_enabled(self) -> bool:
        """Should LLM call logs show up? Default: yes."""
        return self.primitive_log_enabled() and bool(self.config.get("log_llm", True))

    def log_bodies_enabled(self) -> bool:
        """Should full prompt/JSON logs show up? Default: yes."""
        return self.primitive_log_enabled() and bool(self.config.get("log_bodies", True))

    def initialize(self) -> None:
        """One-time setup. Override in subclass if needed, call super() at the end."""
        self._initialized = True

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """The actual work. Must be implemented by every subclass."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # Run setup the first time only, then do the actual work.
        if not self._initialized:
            self.initialize()
        return self.forward(*args, **kwargs)

    def printf(self, *args, separator=" ", line_end="\n", flush=False):
        """Same as print() but goes to self.log. flush is ignored."""
        del flush
        self.log.info("%s%s", separator.join(str(a) for a in args), line_end)

    def __repr__(self):
        return f"<{self.__class__.__name__} name={self.name!r}>"


class HybridPrimitive(Primitive, abc.ABC):
    """Base for primitives that mix LLM calls with rule-based logic."""
    primitive_type = "hybrid"

class PrimitiveRegistry:
    """Stores primitives by name so the DSL engine can look them up.

    registry.register(my_step)   # add
    registry.get("my_step")      # fetch
    "my_step" in registry        # check
    """

    def __init__(self):
        self._primitives = {}

    def register(self, primitive: Primitive) -> None:
        if not primitive.name:
            raise ValueError(f"Cannot register {primitive!r} — name cannot be empty.")
        self._primitives[primitive.name] = primitive

    def get(self, name: str) -> Primitive:
        if name not in self._primitives:
            raise KeyError(
                f"No primitive named {name!r}. "
                f"Registered: {self.all_names()}"
            )
        return self._primitives[name]

    def __contains__(self, name: str) -> bool:
        return name in self._primitives
    def __iter__(self):
        return iter(self._primitives.values())
    def all_names(self) -> list[str]:
        return list(self._primitives.keys())
