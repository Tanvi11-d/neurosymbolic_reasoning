"""Collects log lines from primitives during a request, so the API can return them."""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator

# One log list per request. Each request gets its own so they don't mix.
primitive_log_buffer: contextvars.ContextVar[list[dict[str, Any]] | None] = (
    contextvars.ContextVar("nre_primitive_log_buffer", default=None)
)

def get_primitive_log_buffer() -> list[dict[str, Any]] | None:
    # Returns the current log list, or None if we're not inside a request.
    return primitive_log_buffer.get()


def append_primitive_log_event(entry: dict[str, Any]) -> None:
    # Write one log line into the current request's list.
    buffer = primitive_log_buffer.get()
    if buffer is not None:
        buffer.append(entry)


def primitive_log_event(*, ts, level, levelno, primitive, primitive_class, message):
    # Build one log row. Convert the raw timestamp to a readable ISO string too.
    ts_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "ts":              ts,
        "ts_iso":          ts_iso,
        "level":           level,
        "levelno":         levelno,
        "primitive":       primitive,
        "primitive_class": primitive_class,
        "data":            {"message": message},
    }


def format_primitive_log_for_api(entries: list[dict[str, Any]]) -> dict[str, Any]:
    # Turn the flat list of log lines into two views for the API:
    #   events       → all lines in time order
    #   by_primitive → same lines grouped by who wrote them

    all_events = list(entries)
    groups = {}         # primitive name → its log lines
    seen_order = []     # order we first saw each primitive
    class_by_name = {}  # primitive name → its class name

    for event in all_events:
        prim_name  = str(event.get("primitive", ""))
        prim_class = str(event.get("primitive_class", ""))

        # First time seeing this primitive — open a group for it.
        if prim_name not in groups:
            groups[prim_name] = []
            seen_order.append(prim_name)

        # Remember the class name (only need to save it once).
        if prim_name not in class_by_name and prim_class:
            class_by_name[prim_name] = prim_class

        # Add the line to this primitive's group.
        # Skip primitive/class fields here — they're already on the outer group.
        groups[prim_name].append({
            "ts":      event.get("ts"),
            "ts_iso":  event.get("ts_iso"),
            "level":   event.get("level"),
            "levelno": event.get("levelno"),
            "data":    dict(event.get("data") or {}),
        })

    by_primitive = [
        {
            "primitive":       name,
            "primitive_class": class_by_name.get(name, ""),
            "entries":         groups[name],
        }
        for name in seen_order
    ]

    return {
        "schema_version": 1,
        "events":         all_events,
        "by_primitive":   by_primitive,
    }


@contextmanager
def primitive_log_session() -> Generator[list[dict[str, Any]], None, None]:
    log_list = []
    token = primitive_log_buffer.set(log_list)
    try:
        yield log_list
    finally:
        primitive_log_buffer.reset(token)
