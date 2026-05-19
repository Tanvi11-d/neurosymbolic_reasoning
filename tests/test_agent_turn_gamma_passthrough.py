"""P3 wiring — ``agent_turn`` must pass ``γ`` to ``InitScratchpad``.

The primitive now supports ``forward(config, gamma=...)``. The DSL domain must
actually use it. Without this wiring the production server would still see
``S = ∅`` for every request.
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


def _bare_engine():
    t1 = UnitTask("t1", tool_name="noop")

    def decompose(_i, _g, _tools):
        return {"turns": [t1], "config": {}, "tools": {}}

    return build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: "noop"),
    )


def test_agent_turn_flattens_gamma_into_scratchpad():
    """γ with a nested BFCL-style config ends up as dot-path keys in S."""
    engine = _bare_engine()
    gamma = {
        "VehicleControlAPI": {
            "fuelLevel": 10.5,
            "engineState": "stopped",
        }
    }
    out = engine.run(
        "agent_turn",
        iota="",
        gamma=gamma,
        tools={"noop": {"params": []}},
    )
    assert out["scratchpad"]["VehicleControlAPI.fuelLevel"] == 10.5
    assert out["scratchpad"]["VehicleControlAPI.engineState"] == "stopped"


def test_agent_turn_config_overrides_gamma_when_both_present():
    """DECOMPOSE ``config_keys`` still win — spec authoritative."""
    t1 = UnitTask("t1", tool_name="noop")

    def decompose(_i, _g, _tools):
        return {
            "turns": [t1],
            "config": {"VehicleControlAPI.fuelLevel": 99.9},
            "tools": {},
        }

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: "noop"),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={"VehicleControlAPI": {"fuelLevel": 10.5}},
        tools={"noop": {"params": []}},
    )
    assert out["scratchpad"]["VehicleControlAPI.fuelLevel"] == 99.9


def test_agent_turn_no_gamma_still_works():
    """Backward compat: empty γ should behave exactly like before."""
    engine = _bare_engine()
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools={"noop": {"params": []}},
    )
    # scratchpad may contain whatever carry / gamma_seeds / config put there,
    # but with all of those empty we expect an empty dict.
    assert out["scratchpad"] == {}


def test_agent_turn_gamma_seeds_still_layer_correctly():
    """``gamma_seeds`` (caller-provided flat overlay) still layers between
    carry and DECOMPOSE config. γ-flatten is *in addition*, not a replacement."""
    engine = _bare_engine()
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={"VehicleControlAPI": {"fuelLevel": 10.5}},
        tools={"noop": {"params": []}},
        gamma_seeds={"manual_override": 42},
    )
    assert out["scratchpad"]["VehicleControlAPI.fuelLevel"] == 10.5
    assert out["scratchpad"]["manual_override"] == 42


def test_agent_turn_gamma_flatten_reaches_match_via_lookup():
    """End-to-end: flattened γ key is reachable by MATCH via S lookup."""
    t1 = UnitTask("t1", tool_name="read_fuel")

    def decompose(_i, _g, _tools):
        return {"turns": [t1], "config": {}, "tools": {}}

    # Tool declares ``fuelLevel`` as a required param; S supplies it via γ-flatten.
    tools = {
        "read_fuel": {
            "params": ["VehicleControlAPI.fuelLevel"],
            "execute": lambda **kw: kw,
        }
    }
    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: "read_fuel"),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={"VehicleControlAPI": {"fuelLevel": 10.5}},
        tools=tools,
    )
    step = out["steps"][0]
    assert step["phase"] == "match"
    assert step["status"] == "success"
    assert step["bindings"] == {"VehicleControlAPI.fuelLevel": 10.5}
