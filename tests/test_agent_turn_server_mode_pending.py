"""Server-mode (``status=bound``) pending-executor sentinel.

In server mode the kernel does NOT execute tools — the host/base LLM does.
So after a MATCH returns ``status=bound``, the tool's ``produces`` keys are
not yet populated with real values at kernel-run time. We write a sentinel
``{"__nre_pending__": <task_id>, "tool": <tool_name>}`` under each produced
key so downstream CHECK can accept that the prereq is "committed" and the
consumer task can proceed in the plan.

Without this sentinel, the retry loop would oscillate forever on the same
prereq because no real values ever enter S in server mode.
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


def _engine(tasks):
    def decompose(_i, _g, _tools):
        return {"turns": list(tasks), "config": {}, "tools": {}}

    return build_turn_engine(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(
            match_func=lambda t, tools: t.tool_name if t.tool_name in tools else None,
        ),
    )


def test_bound_producer_populates_s_with_pending_sentinel():
    """Producer tool has no ``execute``. After MATCH returns ``bound``,
    S must contain the pending sentinel under the produces key."""
    tasks = [UnitTask("t1", tool_name="lookup")]
    engine = _engine(tasks)
    tools = {
        "lookup": {
            "params": [],
            "produces": ["the_key"],
            # No execute — server mode.
        },
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    v = out["scratchpad"].get("the_key")
    assert isinstance(v, dict)
    assert v.get("__nre_pending__") == "t1"
    assert v.get("tool") == "lookup"


def test_bound_producer_unblocks_downstream_consumer():
    """Plan: t1 produces zipcode_A (bound), t2 requires zipcode_A. t2 must
    pass CHECK because S(zipcode_A) = pending sentinel."""
    tasks = [
        UnitTask("t1", tool_name="get_zipcode_A", depends_on=()),
        UnitTask("t2", tool_name="consumer", depends_on=("t1",)),
    ]
    engine = _engine(tasks)
    tools = {
        "get_zipcode_A": {"params": [], "produces": ["zipcode_A"]},
        "consumer": {"params": [], "requires": ["zipcode_A"]},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    match_steps = [s for s in out["steps"] if s.get("phase") == "match"]
    ids = [s["task_id"] for s in match_steps]
    assert "t1" in ids
    assert "t2" in ids
    # Neither should have been blocked on retry.
    prereq_steps = [s for s in out["steps"] if s.get("phase") == "prereq"]
    assert not prereq_steps


def test_bound_sentinel_respects_typed_requires_for_consumer():
    """Even though the sentinel is not a real zipcode, ``Satisfied`` accepts
    it under ``param_types: zipcode`` so the retry loop doesn't wedge when
    the kernel is only planning."""
    tasks = [
        UnitTask("t1", tool_name="get_zip", depends_on=()),
        UnitTask("t2", tool_name="estimate", depends_on=("t1",)),
    ]
    engine = _engine(tasks)
    tools = {
        "get_zip": {"params": [], "produces": ["zip_A"]},
        "estimate": {
            "params": ["a"],
            "requires": ["zip_A"],
            "param_types": {"zip_A": "zipcode"},
        },
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    prereq_steps = [s for s in out["steps"] if s.get("phase") == "prereq"]
    assert not prereq_steps, f"unexpected blocks: {prereq_steps}"


def test_success_producer_still_writes_real_result_not_sentinel():
    """Regression: when the producer has a real ``execute`` (kernel-owned tools),
    the result value is still the real return — not a sentinel."""
    tasks = [UnitTask("t1", tool_name="real_producer")]
    engine = _engine(tasks)
    tools = {
        "real_producer": {
            "params": [],
            "produces": ["answer"],
            "execute": lambda: 42,
        },
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    assert out["scratchpad"].get("answer") == 42
