#!/usr/bin/env python3
"""Regenerate tests/fixtures/match_benchmark/cases.json (deterministic + LLM rows)."""

from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "tests" / "fixtures" / "match_benchmark" / "cases.json"


def _task(**kw: object) -> dict:
    return {
        "id": kw.get("id", "t0"),
        "text": kw.get("text", "") or "",
        "depends_on": list(kw.get("depends_on", ()) or ()),
        "tool_name": kw.get("tool_name", "") or "",
        "parameters": dict(kw.get("parameters") or {}),
    }


CASES: list[dict] = [
    {
        "id": "m01",
        "description": "Scratchpad binds double(x)",
        "task": _task(id="t1", text="double", tool_name="double"),
        "scratchpad": {"x": 3},
        "turn_context": {},
        "tools": ["double"],
        "match_func_returns": None,
        "expected": {"status": "success", "function": "double", "result": 6},
    },
    {
        "id": "m02",
        "description": "τ wins over S for same key",
        "task": _task(id="t1", tool_name="add"),
        "scratchpad": {"a": 9, "b": 2},
        "turn_context": {"a": 1},
        "tools": ["add"],
        "match_func_returns": None,
        "expected": {
            "status": "success",
            "function": "add",
            "result": 3,
            "bindings": {"a": 1, "b": 2},
        },
    },
    {
        "id": "m03",
        "description": "task.parameters before τ/S",
        "task": _task(id="t1", tool_name="mul", parameters={"a": 3, "b": 4}),
        "scratchpad": {"a": 100},
        "turn_context": {},
        "tools": ["mul"],
        "match_func_returns": None,
        "expected": {
            "status": "success",
            "function": "mul",
            "result": 12,
            "bindings": {"a": 3, "b": 4},
        },
    },
    {
        "id": "m04",
        "description": "Prefer task.tool_name over match_func_returns",
        "task": _task(id="t1", tool_name="double", parameters={"x": 4}),
        "scratchpad": {},
        "turn_context": {},
        "tools": ["double", "other"],
        "match_func_returns": "other",
        "expected": {"status": "success", "function": "double", "result": 8},
    },
    {
        "id": "m05",
        "description": "No execute → bound",
        "task": _task(id="t1", tool_name="spec_only"),
        "scratchpad": {"a": 1},
        "turn_context": {},
        "tools": ["spec_only"],
        "match_func_returns": None,
        "expected": {
            "status": "bound",
            "function": "spec_only",
            "bindings": {"a": 1},
        },
    },
    {
        "id": "m06",
        "description": "match_func_returns chooses add",
        "task": _task(id="t1", text="sum", tool_name=""),
        "scratchpad": {"a": 2, "b": 3},
        "turn_context": {},
        "tools": ["add", "mul"],
        "match_func_returns": "add",
        "expected": {"status": "success", "function": "add", "result": 5},
    },
    {
        "id": "m07",
        "description": "Zero-arity tool",
        "task": _task(id="t1", tool_name="noop"),
        "scratchpad": {},
        "turn_context": {},
        "tools": ["noop"],
        "match_func_returns": None,
        "expected": {"status": "success", "function": "noop", "result": 0},
    },
    {
        "id": "m08",
        "description": "identity from scratchpad",
        "task": _task(id="t1", tool_name="identity"),
        "scratchpad": {"x": "keep"},
        "turn_context": {},
        "tools": ["identity"],
        "match_func_returns": None,
        "expected": {"status": "success", "function": "identity", "result": "keep"},
    },
    {
        "id": "m09",
        "description": "OpenRouter MATCH-FUNC picks tool (no symbolic match)",
        "llm_case": True,
        "task": _task(
            id="t1",
            text="Invoke the pickme helper",
            tool_name="",
        ),
        "scratchpad": {},
        "turn_context": {},
        "tools": ["pickme", "other"],
        "match_func_returns": None,
        "expected": {"status": "success", "function": "pickme", "result": 7},
    },
    {
        "id": "m10",
        "description": "OpenRouter MATCH-FUNC + param fill for add",
        "llm_case": True,
        "task": _task(
            id="t1",
            text="Add 2 and 3",
            tool_name="",
        ),
        "scratchpad": {},
        "turn_context": {},
        "tools": ["add"],
        "match_func_returns": None,
        "match_config": {"llm_fill_missing_params": True},
        "expected": {"status": "success", "function": "add", "result": 5},
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(CASES, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
