"""JSON-safe views of :func:`nre.dsl.run_turn.run_turn` results (no callables, no UnitTask)."""

from __future__ import annotations

from typing import Any

from nre.primitives.deterministic import UnitTask


def _task_as_dict(t: UnitTask) -> dict[str, Any]:
    return {
        "id": t.id,
        "text": t.text,
        "depends_on": list(t.depends_on),
        "tool_name": t.tool_name,
        "parameters": dict(t.parameters),
    }


def _strip_tool_registry(tools: Any) -> dict[str, Any]:
    if not isinstance(tools, dict):
        return {}
    out: dict[str, Any] = {}
    for name, spec in tools.items():
        if not isinstance(spec, dict):
            continue
        entry = {k: v for k, v in spec.items() if k != "execute"}
        out[str(name)] = entry
    return out


def json_safe_run_turn_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *result* safe for ``json.dumps``."""
    dec = result.get("decompose")
    dec_out: dict[str, Any] | None = None
    if isinstance(dec, dict):
        turns = dec.get("turns") or []
        dec_out = {
            "turns": [
                _task_as_dict(t) if isinstance(t, UnitTask) else t
                for t in turns
            ],
            "config": dict(dec.get("config") or {}),
            "tools": _strip_tool_registry(dec.get("tools")),
            "reasoning": dec.get("reasoning", ""),
        }

    return {
        "scratchpad": dict(result.get("scratchpad") or {}),
        "ordered_task_ids": list(result.get("ordered_task_ids") or []),
        "steps": list(result.get("steps") or []),
        "cycle": bool(result.get("cycle")),
        "cycles": list(result.get("cycles") or []),
        "decompose": dec_out,
    }
