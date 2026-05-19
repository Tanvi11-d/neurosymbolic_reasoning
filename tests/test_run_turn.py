from nre.examples.agentic_reasoner import build_turn_engine
from nre.primitives.deterministic import InitScratchpad, UnitTask
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)


def test_run_turn_reports_cycle():
    t1 = UnitTask("t1", depends_on=("t2",))
    t2 = UnitTask("t2", depends_on=("t1",))

    def decompose(_i, _g, tools):
        return {"turns": [t1, t2], "config": {}, "tools": {}}

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: None),
    )
    out = engine.run("agent_turn", iota="", gamma={}, tools={})
    assert out["cycle"] is True
    assert out["cycles"]
    assert out["steps"] == []


def test_run_turn_prereq_skips_match():
    t1 = UnitTask("t1")

    def decompose(_i, _g, tools):
        return {"turns": [t1], "config": {}, "tools": {}}

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(
            get_prereqs=lambda _t: ["missing"],
        ),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: "x"),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools={},
        scratchpad_carry={},
        turn_context_by_task_id={},
    )
    assert len(out["steps"]) == 1
    assert out["steps"][0]["phase"] == "prereq"
    assert out["steps"][0]["status"] == "absent"


def test_run_turn_success_match():
    t1 = UnitTask("t1")
    recorded: list[tuple[str, dict]] = []
    state: dict = {}

    def decompose(_i, _g, tools):
        return {"turns": [t1], "config": {"x": 0}, "tools": {}}

    def f():
        recorded.append(("f", {}))
        return 42

    tools = {"f": {"params": [], "execute": f}}

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, _tools: "f"),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools=tools,
        scratchpad_carry=state,
    )
    assert out["cycle"] is False
    assert out["steps"][0]["status"] == "success"
    assert out["steps"][0]["result"] == 42
    assert recorded == [("f", {})]


def test_run_turn_merges_decompose_tools():
    t1 = UnitTask("t1")

    def decompose(_i, _g, reg):
        return {
            "turns": [t1],
            "config": {},
            "tools": {"extra": {"params": [], "execute": lambda: 1}},
        }

    engine = build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(match_func=lambda _t, tools: "extra"),
    )
    out = engine.run(
        "agent_turn",
        iota="",
        gamma={},
        tools={"base": 1},
        scratchpad_carry={},
    )
    assert out["steps"][0]["status"] == "success"
