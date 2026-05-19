"""INIT-SCRATCHPAD merge layers and MonotonicScratchpad (INV-S)."""

import pytest

from nre.examples.agentic_reasoner import build_turn_engine
from nre.primitives.deterministic import (
    InitScratchpad,
    MonotonicScratchpad,
    UnitTask,
    apply_init_scratchpad,
    merge_init_scratchpad,
)
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)


def test_merge_init_scratchpad_order():
    assert merge_init_scratchpad(
        prior={"a": 1, "b": 1},
        gamma_seeds={"b": 2, "c": 3},
        decompose_config={"c": 4, "d": 5},
    ) == {"a": 1, "b": 2, "c": 4, "d": 5}


def test_apply_init_scratchpad_inplace_preserves_identity():
    carry: dict = {"p": 0}
    seed = {"x": 1}
    gamma = {"y": 2}
    out = apply_init_scratchpad(seed, carry=carry, gamma_seeds=gamma)
    assert out is carry
    assert carry == {"p": 0, "y": 2, "x": 1}


def test_apply_init_scratchpad_fresh_dict():
    assert apply_init_scratchpad(
        {"a": 1},
        carry=None,
        gamma_seeds={"b": 2},
    ) == {"b": 2, "a": 1}


def test_monotonic_scratchpad_no_delete():
    s = MonotonicScratchpad({"k": 1})
    s["k"] = 2
    assert s["k"] == 2
    with pytest.raises(TypeError, match="INV-S"):
        del s["k"]
    with pytest.raises(TypeError, match="INV-S"):
        s.pop("k")
    with pytest.raises(TypeError, match="INV-S"):
        s.clear()


def test_run_turn_gamma_seeds_before_decompose_seed():
    t1 = UnitTask("t1")

    def decompose(_i, _g, _tools):
        return {"turns": [t1], "config": {"from_dec": 2}, "tools": {}}

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: None),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools={},
        gamma_seeds={"from_gamma": 1, "from_dec": 0},
        scratchpad_carry=None,
    )
    assert out["scratchpad"]["from_gamma"] == 1
    assert out["scratchpad"]["from_dec"] == 2


def test_golden_turn1_style_seed_layers():
    """Aligns with ``docs/golden-contracts/BFCL-multi_turn_base_0-turns-1-2.md`` INIT-SCRATCHPAD."""
    gamma_seeds = {
        "cwd": "/",
        "workspace_contents": ["report.txt", "notes.txt"],
    }
    decompose_seed = {
        "cwd_is_workspace": False,
        "archive_exists": False,
    }
    s = merge_init_scratchpad(
        prior=None,
        gamma_seeds=gamma_seeds,
        decompose_config=decompose_seed,
    )
    assert s["cwd"] == "/"
    assert s["workspace_contents"] == ["report.txt", "notes.txt"]
    assert s["cwd_is_workspace"] is False
    assert s["archive_exists"] is False


def test_run_turn_with_monotonic_scratchpad_carry():
    """Carry may be a MonotonicScratchpad; INIT merge uses .update (INV-S preserved)."""
    t1 = UnitTask("t1")
    carry = MonotonicScratchpad({"pre": 1})

    def decompose(_i, _g, _tools):
        return {"turns": [t1], "config": {"post": 2}, "tools": {}}

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: None),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools={},
        gamma_seeds={"mid": 0},
        scratchpad_carry=carry,
    )
    assert out["scratchpad"] is carry
    assert dict(carry) == {"pre": 1, "mid": 0, "post": 2}
    with pytest.raises(TypeError, match="INV-S"):
        del carry["pre"]
