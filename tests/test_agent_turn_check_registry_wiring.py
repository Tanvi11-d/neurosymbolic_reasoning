"""P1 wiring — ``agent_turn`` must expose the merged tools registry to
``CheckPrerequisites`` so its registry-driven defaults (spec Sub-process 4
GET-PREREQS / Satisfied / TASK-TO-CHECK) are actually consulted during a run.

Without this wiring the server-side ``CheckPrerequisites`` would see no
``tools_registry`` (it's built once per engine, and the live Ω is only known
per-request), so production runs would still return ``status=ready`` for
every task.
"""

from __future__ import annotations

from nre.examples.agentic_reasoner import build_turn_engine
from nre.primitives.deterministic import InitScratchpad, UnitTask
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)


def _engine_with_check(tasks):
    """Build an engine whose CheckPrerequisites starts with no registry wired;
    the agent_turn domain is responsible for threading it in at call time."""

    def decompose(_i, _g, _tools):
        return {"turns": list(tasks), "config": {}, "tools": {}}

    return build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),  # no tools_registry at init
        match_funcs=MatchFuncsAndParams(
            match_func=lambda _t, tools: _t.tool_name if _t.tool_name in tools else None
        ),
    )


def test_registry_requires_detected_via_check_at_turn_time():
    """Tools registry declares ``requires``; agent_turn must expose it to
    CheckPrerequisites so it sees the absent key and reports ``absent``."""
    tasks = [UnitTask("t1", tool_name="estimate_distance")]
    engine = _engine_with_check(tasks)

    tools = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["zipcode_A"],
        },
    }
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools=tools,
        scratchpad_carry=None,
    )
    # The check primitive should have reported 'absent' for zipcode_A
    prereq_steps = [s for s in out["steps"] if s.get("phase") == "prereq"]
    assert prereq_steps, f"expected a prereq step, got {out['steps']}"
    assert prereq_steps[0]["status"] == "absent"
    assert prereq_steps[0]["key"] == "zipcode_A"


def test_registry_produces_drives_remediation_task():
    tasks = [UnitTask("t1", tool_name="estimate_distance")]
    engine = _engine_with_check(tasks)
    tools = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["zipcode_A"],
        },
        "get_zipcode_A": {
            "params": [],
            "produces": ["zipcode_A"],
        },
    }
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools=tools,
    )
    prereq_steps = [s for s in out["steps"] if s.get("phase") == "prereq"]
    assert prereq_steps
    assert prereq_steps[0].get("remediation_task") is not None
    assert prereq_steps[0]["remediation_task"].tool_name == "get_zipcode_A"


def test_registry_satisfied_with_gamma_flattened_value():
    """End-to-end: γ provides a valid zipcode, Satisfied accepts it, task becomes ready."""
    tasks = [UnitTask("t1", tool_name="estimate_distance")]
    engine = _engine_with_check(tasks)
    tools = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["VehicleControlAPI.zipA"],
            "param_types": {"VehicleControlAPI.zipA": "zipcode"},
        },
    }
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={"VehicleControlAPI": {"zipA": "94016"}},
        tools=tools,
    )
    match_steps = [s for s in out["steps"] if s.get("phase") == "match"]
    prereq_blocks = [s for s in out["steps"] if s.get("phase") == "prereq"]
    # γ-flatten put a valid zipcode into S; Satisfied passes; no prereq block.
    assert not prereq_blocks, f"unexpected prereq block: {prereq_blocks}"
    assert match_steps, f"expected a match step, got {out['steps']}"


def test_registry_satisfied_rejects_gamma_non_zipcode_value():
    tasks = [UnitTask("t1", tool_name="estimate_distance")]
    engine = _engine_with_check(tasks)
    tools = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["VehicleControlAPI.zipA"],
            "param_types": {"VehicleControlAPI.zipA": "zipcode"},
        },
    }
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={"VehicleControlAPI": {"zipA": "Rivermist"}},
        tools=tools,
    )
    prereq_steps = [s for s in out["steps"] if s.get("phase") == "prereq"]
    assert prereq_steps
    # 'Rivermist' is not a zipcode → Satisfied fails → blocked
    assert prereq_steps[0]["status"] == "blocked"
