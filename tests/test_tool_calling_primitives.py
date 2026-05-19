from unittest.mock import MagicMock

from nre.library.deterministic import InitScratchpadStateManagement
from nre.schemas import LLMMatchFillParams, LLMMatchFuncOutput
from nre.primitives.deterministic import InitScratchpad, UnitTask
from nre.primitives.hybrid import (
    CheckPrerequisites,
    MatchFuncsAndParams,
    OrderUnitTasks,
)


def test_init_scratchpad_empty_and_copy():
    assert InitScratchpad().forward() == {}
    assert InitScratchpad().forward(None) == {}
    cfg = {"a": 1, "b": [2]}
    s = InitScratchpad().forward(cfg)
    assert s == cfg
    assert s is not cfg
    cfg["a"] = 99
    assert s["a"] == 1


def test_init_scratchpad_state_management_alias():
    assert InitScratchpadStateManagement().forward({"x": 1}) == {"x": 1}


def test_order_unit_tasks_chain_and_single():
    t1 = UnitTask("t1", "a")
    t2 = UnitTask("t2", "b", depends_on=("t1",))
    t3 = UnitTask("t3", "c", depends_on=("t2",))
    out = OrderUnitTasks().forward([t3, t1, t2])
    assert out["has_cycle"] is False
    assert out["ordered"] == ["t1", "t2", "t3"]
    assert out["is_dag"] is True
    solo = OrderUnitTasks().forward([UnitTask("only")])
    assert solo["ordered"] == ["only"]


def test_order_unit_tasks_parallel_stable():
    a = UnitTask("a")
    b = UnitTask("b")
    out = OrderUnitTasks().forward([b, a])
    assert out["has_cycle"] is False
    assert set(out["ordered"]) == {"a", "b"}
    assert out["ordered"] == sorted(out["ordered"])


def test_order_unit_tasks_cycle():
    t1 = UnitTask("t1", depends_on=("t2",))
    t2 = UnitTask("t2", depends_on=("t1",))
    out = OrderUnitTasks().forward([t1, t2])
    assert out["has_cycle"] is True
    assert out["ordered"] == []
    assert out["is_dag"] is False
    assert out["cycles"]


def test_order_unit_tasks_dep_outside_list():
    t = UnitTask("t", depends_on=("ghost",))
    out = OrderUnitTasks().forward([t])
    assert out["has_cycle"] is False
    assert out["ordered"] == ["ghost", "t"]


def test_check_prerequisites_ready_default():
    t = UnitTask("t1")
    r = CheckPrerequisites().forward(t, {})
    assert r == {"status": "ready", "task_id": "t1"}


def test_check_prerequisites_absent():
    t = UnitTask("t1")
    cp = CheckPrerequisites(
        get_prereqs=lambda _x: ["need"],
    )
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    assert r["key"] == "need"


def test_check_prerequisites_blocked():
    t = UnitTask("t1")
    cp = CheckPrerequisites(
        get_prereqs=lambda _x: ["k"],
        satisfied=lambda key, val, _s: val == "ok",
    )
    r = cp.forward(t, {"k": "bad"})
    assert r["status"] == "blocked"
    assert r["key"] == "k"


def test_check_prerequisites_order_stops_first():
    t = UnitTask("t1")
    seen: list[str] = []

    def get_prereqs(_x):
        return ["a", "b"]

    def satisfied(key, val, _s):
        seen.append(key)
        return val

    cp = CheckPrerequisites(get_prereqs=get_prereqs, satisfied=satisfied)
    r = cp.forward(t, {"a": False, "b": True})
    assert r["status"] == "blocked"
    assert seen == ["a"]


def test_match_missing_function():
    t = UnitTask("t")
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: None)
    r = m(t, {}, {}, {})
    assert r["status"] == "failure"
    assert r["reason"] == "missing_function"


def test_match_missing_param():
    t = UnitTask("t")
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: "f")
    r = m(t, {}, {}, tools)
    assert r["status"] == "failure"
    assert r["reason"] == "missing_param"
    assert r["param"] == "x"


def test_match_turn_context_over_scratchpad():
    t = UnitTask("t")
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: "f")
    r = m(t, {"x": "s"}, {"x": "tau"}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "tau"


def test_match_param_enum_canonicalization():
    t = UnitTask(
        "t1",
        tool_name="trade",
        parameters={"order_type": "buy"},
    )
    tools = {
        "trade": {
            "params": ["order_type"],
            "param_enums": {"order_type": ["Buy", "Sell"]},
            "execute": lambda order_type: order_type,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["order_type"] == "Buy"
    assert r["result"] == "Buy"


def test_match_from_scratchpad():
    t = UnitTask("t")
    tools = {"f": {"params": ["x"], "execute": lambda x: x * 2}}
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: "f")
    r = m(t, {"x": 4}, {}, tools)
    assert r["status"] == "success"
    assert r["result"] == 8


def test_match_retrieve_fills_scratchpad():
    t = UnitTask("t")
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    scratch: dict = {}

    def retrieve(name, _task, s, _ctx):
        if name == "x":
            s["x"] = 7
            return 7
        return None

    m = MatchFuncsAndParams(
        match_func=lambda _task, _tools: "f",
        retrieve=retrieve,
    )
    r = m(t, scratch, {}, tools)
    assert r["status"] == "success"
    assert scratch.get("x") == 7


def test_match_execute_multiparam():
    t = UnitTask("t")
    tools = {
        "g": {
            "params": ["a", "b"],
            "execute": lambda a, b: a + b,
        }
    }
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: "g")
    r = m(t, {"b": 2}, {"a": 1}, tools)
    assert r["status"] == "success"
    assert r["result"] == 3
    assert r["function"] == "g"


def test_match_prefers_task_tool_name():
    t = UnitTask("t", tool_name="wanted")
    tools = {
        "wanted": {"params": [], "execute": lambda: 1},
        "other": {"params": [], "execute": lambda: 2},
    }
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: "other")
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["function"] == "wanted"
    assert r["result"] == 1


def test_match_llm_resolves_tool_when_match_returns_none():
    t = UnitTask("t", text="list files", tool_name="")
    tools = {
        "ls": {"description": "List directory", "params": [], "execute": lambda: 99},
    }
    client = MagicMock()
    client.chat.return_value = LLMMatchFuncOutput(tool_name="ls", reasoning="")
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: None, llm_client=client)
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["function"] == "ls"
    assert r["result"] == 99
    client.chat.assert_called_once()


def test_match_llm_fills_missing_param():
    t = UnitTask("t", text="run with x", parameters={})
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    client = MagicMock()

    def chat(messages, *, response_model=None, **kwargs):
        if response_model is LLMMatchFuncOutput:
            return LLMMatchFuncOutput(tool_name="f", reasoning="")
        if response_model is LLMMatchFillParams:
            return LLMMatchFillParams(values={"x": 42})
        raise AssertionError(response_model)

    client.chat.side_effect = chat
    m = MatchFuncsAndParams(
        match_func=lambda _task, _tools: None,
        llm_client=client,
        llm_fill_missing_params=True,
    )
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == 42
    assert r["result"] == 42
    assert client.chat.call_count == 2


def test_match_registry_without_execute_returns_bound():
    t = UnitTask("t", tool_name="spec_only")
    tools = {"spec_only": {"description": "Metadata only", "params": ["a"]}}
    m = MatchFuncsAndParams(match_func=lambda _task, _tools: None)
    r = m(t, {"a": 1}, {}, tools)
    assert r["status"] == "bound"
    assert r["bindings"] == {"a": 1}
    assert "result" not in r
