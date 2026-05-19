"""P2 — ``agent_turn`` fuel-bounded retry loop (spec Sub-process 4).

The spec says: when CHECK returns ``Blocked(t')`` or ``Absent`` with a
remediation task, the caller must run ``t'`` **and then retry t**. The current
implementation walks the ordered task list once and simply logs the
remediation task as metadata — it never executes it.

This test suite pins the retry semantics:

- Absent/blocked task with a ``remediation_task`` → remediation runs first,
  then the original task is re-checked.
- MATCH failure → abandons only that task (spec INV-4), no retry.
- A fuel counter caps the loop so cycles in remediation cannot wedge the
  engine (spec INV-T finiteness).
- If the second attempt at ``t`` still blocks and no new remediation surfaces,
  ``t`` is abandoned after logging both attempts.
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


# ── Helpers ──────────────────────────────────────────────────────────


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


def _steps_by_task(out):
    return [(s.get("task_id"), s.get("phase"), s.get("status")) for s in out["steps"]]


# ── Absent + remediation → remediation runs, task retried ────────────


def test_absent_with_remediation_runs_producer_then_retries_task():
    """``estimate_distance`` needs ``zipcode_A``; ``get_zipcode_A`` produces it.
    Expected step sequence:
      1. prereq   t1  absent     (first check: key missing, remediation emitted)
      2. match    t1__prereq_zipcode_A  success  (remediation executes, writes S)
      3. match    t1  success    (retry now finds key in S → MATCH succeeds)
    """
    tasks = [UnitTask("t1", tool_name="estimate_distance")]
    engine = _engine(tasks)

    def fake_get_zip():
        return "94016"

    tools = {
        "estimate_distance": {
            "params": [],
            "requires": ["zipcode_A"],
            "execute": lambda: "distance_ok",
        },
        "get_zipcode_A": {
            "params": [],
            "produces": ["zipcode_A"],
            "execute": lambda: "94016",
        },
    }

    # Side-channel to confirm the producer's ``execute`` was actually called.
    execute_order = []
    tools["get_zipcode_A"]["execute"] = lambda: (
        execute_order.append("get_zipcode_A"),
        "94016",
    )[1]
    tools["estimate_distance"]["execute"] = lambda: (
        execute_order.append("estimate_distance"),
        "distance_ok",
    )[1]

    # Writing S on producer-success is the job of the retry loop, not the
    # primitive. The loop copies the remediation's result under the produced
    # key(s) into the scratchpad so the retry of t sees it.
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)

    # Absent → prereq step for t1.
    assert ("t1", "prereq", "absent") in _steps_by_task(out)
    # Remediation step ran with success.
    rem_steps = [
        s for s in out["steps"] if s.get("task_id", "").startswith("t1__prereq")
    ]
    assert rem_steps
    assert rem_steps[0]["phase"] == "match"
    assert rem_steps[0]["status"] == "success"
    # Execution order: producer before consumer.
    assert execute_order == ["get_zipcode_A", "estimate_distance"]
    # Final step for t1 is a successful match.
    match_t1 = [
        s
        for s in out["steps"]
        if s.get("task_id") == "t1" and s.get("phase") == "match"
    ]
    assert match_t1 and match_t1[-1]["status"] == "success"


def test_producer_populates_scratchpad_with_produces_keys():
    """After the remediation tool executes, its ``produces`` keys are written
    into S so the retry of t sees them."""
    tasks = [UnitTask("t1", tool_name="consumer")]
    engine = _engine(tasks)
    tools = {
        "consumer": {
            "params": [],
            "requires": ["the_key"],
            "execute": lambda: "consumed",
        },
        "producer": {
            "params": [],
            "produces": ["the_key"],
            "execute": lambda: 42,
        },
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    assert out["scratchpad"].get("the_key") == 42


# ── MATCH failure is non-fatal, no retry (spec INV-4) ────────────────


def test_match_missing_param_failure_abandons_only_that_task():
    """Spec INV-4: MATCH failure abandons only t; subsequent tasks still run."""
    t1 = UnitTask(
        "t1", tool_name="needs_param"
    )  # no task.parameters, no S, no τ → fail
    t2 = UnitTask("t2", tool_name="fine")
    engine = _engine([t1, t2])
    tools = {
        "needs_param": {"params": ["x"], "execute": lambda x: x},
        "fine": {"params": [], "execute": lambda: "ok"},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    by_id = {s["task_id"]: s for s in out["steps"] if s.get("phase") == "match"}
    assert by_id["t1"]["status"] == "failure"
    assert by_id["t1"]["reason"] == "missing_param"
    assert by_id["t2"]["status"] == "success"


# ── Fuel cap: can't wedge on pathological remediation cycles ─────────


def test_fuel_cap_abandons_task_on_repeated_block():
    """If remediation never satisfies the prereq (producer tool fails to
    populate the required key), the loop must not spin forever — the task is
    abandoned after a bounded number of retries and subsequent tasks proceed."""
    tasks = [
        UnitTask("stuck", tool_name="consumer"),
        UnitTask("later", tool_name="fine"),
    ]
    engine = _engine(tasks)
    tools = {
        "consumer": {
            "params": [],
            "requires": ["never_produced"],
            "execute": lambda: "should_not_run",
        },
        "bad_producer": {
            "params": [],
            "produces": ["never_produced"],
            # Does NOT actually populate the key when it runs.
            "execute": lambda: None,
        },
        "fine": {"params": [], "execute": lambda: "ok"},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    # 'later' must still have been reached and succeeded.
    match_later = [
        s
        for s in out["steps"]
        if s.get("task_id") == "later" and s.get("phase") == "match"
    ]
    assert match_later and match_later[-1]["status"] == "success"


# ── Back-compat: no remediation, no requires → single pass ───────────


def test_no_requires_no_retry_behaves_as_before():
    """When tools declare no requires, every task runs once in order."""
    t1 = UnitTask("t1", tool_name="a")
    t2 = UnitTask("t2", tool_name="b", depends_on=("t1",))
    engine = _engine([t1, t2])
    tools = {
        "a": {"params": [], "execute": lambda: "A"},
        "b": {"params": [], "execute": lambda: "B"},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    match_steps = [s for s in out["steps"] if s.get("phase") == "match"]
    assert [s["task_id"] for s in match_steps] == ["t1", "t2"]
    assert all(s["status"] == "success" for s in match_steps)


def test_blocked_without_remediation_abandons_task_no_retry():
    """Blocked (¬Met) with no remediation_task: task abandoned, next task runs."""
    t1 = UnitTask("t1", tool_name="consumer")
    t2 = UnitTask("t2", tool_name="fine")
    engine = _engine([t1, t2])
    tools = {
        "consumer": {
            "params": [],
            "requires": ["missing_key"],
            # No producer registered for missing_key.
        },
        "fine": {"params": [], "execute": lambda: "ok"},
    }
    out = engine.run("agent_turn", iota="", gamma={}, tools=tools)
    by_id_phase = [
        (s.get("task_id"), s.get("phase"), s.get("status")) for s in out["steps"]
    ]
    # t1 prereq step present; no match step for t1; t2 match success.
    assert ("t1", "prereq", "absent") in by_id_phase or (
        "t1",
        "prereq",
        "blocked",
    ) in by_id_phase
    assert not any(p == "match" and tid == "t1" for tid, p, _ in by_id_phase)
    assert ("t2", "match", "success") in by_id_phase
