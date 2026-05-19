"""CHECK-PREREQUISITES extensions: classify, remediation hook, optional LLM."""

from unittest.mock import MagicMock

import pytest

from nre.schemas import LLMCheckRemediationOutput, LLMTaskItem
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import CheckPrerequisites


def test_classify_prereq_override():
    t = UnitTask("t1")

    def classify(key, _task, s):
        if key == "x":
            return "absent" if "x" not in s else "met"
        return "met"

    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["x"],
        classify_prereq=classify,
    )
    assert cp.forward(t, {})["status"] == "absent"
    assert cp.forward(t, {"x": 1})["status"] == "ready"


def test_classify_invalid_raises():
    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["a"],
        classify_prereq=lambda *_a: "maybe",
    )
    with pytest.raises(ValueError, match="classify_prereq"):
        cp.forward(UnitTask("t"), {})


def test_remediation_for_hook():
    t = UnitTask("main")
    fix = UnitTask("fix1", text="retrieve", tool_name="fetch", depends_on=())

    def remediation(key, _task, _s, outcome):
        if key == "need" and outcome == "absent":
            return fix
        return None

    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["need"],
        remediation_for=remediation,
    )
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    assert r["remediation_task"] is fix


def test_check_trace_lists_all_keys_on_success():
    t = UnitTask("t1")
    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["a", "b"],
        satisfied=lambda _k, v, _s: bool(v),
        check_trace=True,
    )
    r = cp.forward(t, {"a": True, "b": True})
    assert r["status"] == "ready"
    assert r["checks"] == [
        {"key": "a", "outcome": "met"},
        {"key": "b", "outcome": "met"},
    ]


def test_llm_remediation_when_hook_returns_none():
    t = UnitTask("t_main", text="Continue", tool_name="go")
    rem = LLMTaskItem(
        id="sub1",
        description="Fetch missing key",
        tool_name="get",
        depends_on=[],
        parameters={"k": "need"},
    )
    client = MagicMock()
    client.chat.return_value = LLMCheckRemediationOutput(task=rem, reasoning="fetch")

    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["need"],
        llm_client=client,
        tools_registry_for_llm={
            "get": {"description": "Get", "params": ["k"]},
        },
    )
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    assert r["remediation_task"].id == "sub1"
    assert r["remediation_task"].tool_name == "get"
    client.chat.assert_called_once()


def test_llm_skipped_without_tools_registry():
    t = UnitTask("t1")
    client = MagicMock()
    cp = CheckPrerequisites(
        get_prereqs=lambda _u: ["x"],
        llm_client=client,
    )
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    assert "remediation_task" not in r
    client.chat.assert_not_called()
