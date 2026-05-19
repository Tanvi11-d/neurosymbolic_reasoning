#!/usr/bin/env python3
"""Regenerate tests/fixtures/check_benchmark/cases.json with 20 CHECK-PREREQUISITES scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
OUT = _ROOT / "tests" / "fixtures" / "check_benchmark" / "cases.json"


def _task(**kw: object) -> dict:
    d = {
        "id": kw.get("id", "t0"),
        "text": kw.get("text", "") or "",
        "depends_on": list(kw.get("depends_on", ()) or ()),
        "tool_name": kw.get("tool_name", "") or "",
        "parameters": dict(kw.get("parameters") or {}),
    }
    return d


CASES: list[dict] = [
    {
        "id": "c01",
        "description": "No prerequisites → ready",
        "task": _task(id="run"),
        "prereq_keys": [],
        "scratchpad": {},
        "mode": "non_null",
        "check_trace": False,
        "expected": {"status": "ready"},
    },
    {
        "id": "c02",
        "description": "Single satisfied key",
        "task": _task(id="run"),
        "prereq_keys": ["a"],
        "scratchpad": {"a": 1},
        "mode": "non_null",
        "expected": {"status": "ready"},
    },
    {
        "id": "c03",
        "description": "Key absent from scratchpad",
        "task": _task(id="run"),
        "prereq_keys": ["missing"],
        "scratchpad": {},
        "mode": "non_null",
        "expected": {"status": "absent", "key": "missing"},
    },
    {
        "id": "c04",
        "description": "Key present but null → blocked",
        "task": _task(id="run"),
        "prereq_keys": ["n"],
        "scratchpad": {"n": None},
        "mode": "non_null",
        "expected": {"status": "blocked", "key": "n"},
    },
    {
        "id": "c05",
        "description": "equals: value mismatch → blocked",
        "task": _task(id="run"),
        "prereq_keys": ["k"],
        "scratchpad": {"k": "bad"},
        "mode": "equals",
        "equals": {"k": "ok"},
        "expected": {"status": "blocked", "key": "k"},
    },
    {
        "id": "c06",
        "description": "equals: value match → ready",
        "task": _task(id="run"),
        "prereq_keys": ["k"],
        "scratchpad": {"k": "ok"},
        "mode": "equals",
        "equals": {"k": "ok"},
        "expected": {"status": "ready"},
    },
    {
        "id": "c07",
        "description": "equals: extra key uses non_null branch",
        "task": _task(id="run"),
        "prereq_keys": ["only_in_scratch"],
        "scratchpad": {"only_in_scratch": 0},
        "mode": "equals",
        "equals": {},
        "expected": {"status": "ready"},
    },
    {
        "id": "c08",
        "description": "classify: met then absent",
        "task": _task(id="run"),
        "prereq_keys": ["x", "y"],
        "scratchpad": {},
        "mode": "classify",
        "classify_outcomes": {"x": "met", "y": "absent"},
        "expected": {"status": "absent", "key": "y"},
    },
    {
        "id": "c09",
        "description": "classify: not_met",
        "task": _task(id="run"),
        "prereq_keys": ["a"],
        "scratchpad": {},
        "mode": "classify",
        "classify_outcomes": {"a": "not_met"},
        "expected": {"status": "blocked", "key": "a"},
    },
    {
        "id": "c10",
        "description": "classify: all met → ready",
        "task": _task(id="run"),
        "prereq_keys": ["u", "v"],
        "scratchpad": {},
        "mode": "classify",
        "classify_outcomes": {"u": "met", "v": "met"},
        "expected": {"status": "ready"},
    },
    {
        "id": "c11",
        "description": "OpenRouter proposes remediation when prerequisite is absent (no symbolic hook)",
        "remediation_kind": "llm",
        "task": _task(id="run", text="Continue after need is available", tool_name="go"),
        "prereq_keys": ["need"],
        "scratchpad": {},
        "mode": "non_null",
        "tools_registry_for_llm": {
            "fetch": {
                "description": "Load or retrieve a missing value into the scratchpad",
                "params": ["target"],
            },
            "auth_login": {
                "description": "Obtain credentials or session material",
                "params": ["user"],
            },
        },
        "expected": {
            "status": "absent",
            "key": "need",
            "remediation_non_null": True,
            "remediation_tool_in": ["fetch", "auth_login"],
        },
    },
    {
        "id": "c12",
        "description": "OpenRouter proposes remediation when prerequisite is not_met",
        "remediation_kind": "llm",
        "task": _task(id="run", text="Use validated field bad", tool_name="consume"),
        "prereq_keys": ["bad"],
        "scratchpad": {"bad": None},
        "mode": "non_null",
        "tools_registry_for_llm": {
            "set_value": {
                "description": "Assign a valid value to a scratchpad key",
                "params": ["key", "value"],
            },
            "clear_flag": {
                "description": "Reset a scratchpad key",
                "params": ["key"],
            },
        },
        "expected": {
            "status": "blocked",
            "key": "bad",
            "remediation_non_null": True,
            "remediation_tool_in": ["set_value", "clear_flag"],
        },
    },
    {
        "id": "c13",
        "description": "First failing key wins (second would be met)",
        "task": _task(id="run"),
        "prereq_keys": ["first", "second"],
        "scratchpad": {"second": 1},
        "mode": "non_null",
        "expected": {"status": "absent", "key": "first"},
    },
    {
        "id": "c14",
        "description": "check_trace on ready (two checks)",
        "task": _task(id="run"),
        "prereq_keys": ["a", "b"],
        "scratchpad": {"a": 1, "b": 2},
        "mode": "non_null",
        "check_trace": True,
        "expected": {"status": "ready"},
    },
    {
        "id": "c15",
        "description": "check_trace on absent includes prior met keys",
        "task": _task(id="run"),
        "prereq_keys": ["p", "q"],
        "scratchpad": {"p": 1},
        "mode": "non_null",
        "check_trace": True,
        "expected": {"status": "absent", "key": "q"},
    },
    {
        "id": "c16",
        "description": "Two-key chain blocked on second",
        "task": _task(id="run"),
        "prereq_keys": ["gate", "data"],
        "scratchpad": {"gate": True, "data": None},
        "mode": "non_null",
        "expected": {"status": "blocked", "key": "data"},
    },
    {
        "id": "c17",
        "description": "remediation hook does not fire for wrong key",
        "task": _task(id="run"),
        "prereq_keys": ["p", "q"],
        "scratchpad": {},
        "mode": "remediation",
        "remediation": {
            "when_key": "q",
            "when_outcome": "absent",
            "task": _task(id="wrong_rem", tool_name="noop"),
        },
        "expected": {"status": "absent", "key": "p"},
    },
    {
        "id": "c18",
        "description": "Empty string value counts as present for non_null",
        "task": _task(id="run"),
        "prereq_keys": ["s"],
        "scratchpad": {"s": ""},
        "mode": "non_null",
        "expected": {"status": "ready"},
    },
    {
        "id": "c19",
        "description": "Zero is satisfied (non_null)",
        "task": _task(id="run"),
        "prereq_keys": ["z"],
        "scratchpad": {"z": 0},
        "mode": "non_null",
        "expected": {"status": "ready"},
    },
    {
        "id": "c20",
        "description": "Task metadata preserved in output task_id",
        "task": _task(id="special_id", text="do thing", tool="t"),
        "prereq_keys": ["r"],
        "scratchpad": {},
        "mode": "non_null",
        "expected": {"status": "absent", "key": "r", "task_id": "special_id"},
    },
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(CASES, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
