"""P — ``agent_turn`` consults ``PolicyConstraints`` (when registered) after
MATCH produces bindings and before any scratchpad write.

Policy gating is **opt-in**: the domain only fires the gate if a primitive
named ``policy_constraints`` is present in the engine registry. Engines
without policy rules behave exactly as before (full backward compat).

When the gate denies:
- The step is appended as ``{phase: "match", status: "denied", policy,
  reason, function, task_id}``.
- The ``produces`` keys for that tool are NOT written into S (the task is
  abandoned — spec INV-4 non-fatal).
- The retry loop proceeds to the next task.

When the gate allows:
- Behavior is identical to today; step is appended with its normal
  ``status=bound`` / ``status=success``, produces are written.
"""

from __future__ import annotations

from nre.examples.agentic_reasoner import build_turn_engine
from nre.primitives.deterministic import InitScratchpad, UnitTask
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
    PolicyConstraints,
)


def _engine(tasks, policy_rules=None):
    def decompose(_i, _g, _tools):
        return {"turns": list(tasks), "config": {}, "tools": {}}

    kwargs = dict(
        decompose_input=DecomposeInput(decompose=decompose),
        init_scratchpad=InitScratchpad(),
        order_unit_tasks=OrderUnitTasks(),
        check_prerequisites=CheckPrerequisites(),
        match_funcs=MatchFuncsAndParams(
            match_func=lambda t, tools: t.tool_name if t.tool_name in tools else None
        ),
    )
    engine = build_turn_engine(**kwargs)
    if policy_rules is not None:
        engine.register(
            PolicyConstraints(name="policy_constraints", policies=policy_rules)
        )
    return engine


# ── Allow path — no policy registered, behavior unchanged ────────────


def test_no_policy_primitive_registered_behaves_as_before():
    tasks = [UnitTask("t1", tool_name="book")]
    engine = _engine(tasks)  # no policies
    tools = {"book": {"params": [], "execute": lambda: "ok"}}
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    match_steps = [s for s in out["steps"] if s.get("phase") == "match"]
    assert match_steps and match_steps[0]["status"] == "success"
    assert not any(s.get("status") == "denied" for s in out["steps"])


# ── Deny path — policy refuses a mutation ────────────────────────────


def test_policy_deny_abandons_task_and_continues():
    """t1 should deny (basic_economy immutable); t2 should proceed normally."""
    tasks = [
        UnitTask(
            "t1", tool_name="update_flights", parameters={"cabin": "basic_economy"}
        ),
        UnitTask("t2", tool_name="cheap_read"),
    ]
    policies = {
        "basic_economy_immutable": {
            "applies_to": ["update_flights"],
            "predicate": lambda b, _s, _tau: b.get("cabin") != "basic_economy",
            "reason": "Basic economy flights cannot be modified.",
        }
    }
    engine = _engine(tasks, policies)
    tools = {
        "update_flights": {"params": ["cabin"], "execute": lambda cabin: cabin},
        "cheap_read": {"params": [], "execute": lambda: "ok"},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    steps_by_id = {s["task_id"]: s for s in out["steps"] if s.get("phase") == "match"}
    # t1 denied.
    assert steps_by_id["t1"]["status"] == "denied"
    assert steps_by_id["t1"]["policy"] == "basic_economy_immutable"
    assert "basic economy" in steps_by_id["t1"]["reason"].lower()
    # t2 still ran.
    assert steps_by_id["t2"]["status"] == "success"


def test_policy_deny_does_not_write_produces():
    """When a producer is denied, its ``produces`` keys must NOT be written to S
    — otherwise a downstream CHECK would incorrectly think the prereq is met."""
    tasks = [UnitTask("t1", tool_name="producer")]
    policies = {
        "always_deny_producer": {
            "applies_to": ["producer"],
            "predicate": lambda *_: False,
            "reason": "nope",
        }
    }
    engine = _engine(tasks, policies)
    tools = {"producer": {"params": [], "produces": ["k"], "execute": lambda: 42}}
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    assert "k" not in out["scratchpad"]


def test_policy_allow_passes_through_unchanged():
    """Allow path must return exactly the same shape as the no-policy case
    (aside from any trace data we choose to add)."""
    tasks = [UnitTask("t1", tool_name="book")]
    policies = {
        "always_allow": {
            "applies_to": ["book"],
            "predicate": lambda *_: True,
            "reason": "",
        }
    }
    engine = _engine(tasks, policies)
    tools = {"book": {"params": [], "execute": lambda: "booked"}}
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    match = [s for s in out["steps"] if s.get("phase") == "match"][0]
    assert match["status"] == "success"
    assert match.get("result") == "booked"


def test_policy_gate_sees_bound_status_in_server_mode():
    """In server mode (no ``execute``), MATCH returns ``status=bound``. The
    policy gate must still run and can deny. Denied bound calls are recorded
    as ``status=denied`` and the pending sentinel is NOT written to S."""
    tasks = [UnitTask("t1", tool_name="mutate", parameters={"cabin": "basic_economy"})]
    policies = {
        "basic_economy_immutable": {
            "applies_to": ["mutate"],
            "predicate": lambda b, _s, _tau: b.get("cabin") != "basic_economy",
            "reason": "no.",
        }
    }
    engine = _engine(tasks, policies)
    tools = {"mutate": {"params": ["cabin"], "produces": ["side_effect"]}}  # no execute
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    match = [s for s in out["steps"] if s.get("phase") == "match"][0]
    assert match["status"] == "denied"
    assert "side_effect" not in out["scratchpad"]
