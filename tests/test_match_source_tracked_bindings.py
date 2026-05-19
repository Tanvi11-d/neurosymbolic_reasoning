"""Source-tracked bindings in MATCH-FUNCS-AND-PARAMS.

Motivation from τ² airline task_0: the agent bound ``payment_methods`` to
user-invented certificate IDs ("CERT67890") instead of the real IDs from the
user profile (``certificate_7504069``). Policy explicitly says "All payment
methods must already be in user profile for safety reasons." The fix is to
give every binding a *provenance* and let a tool declare which provenances
are acceptable for each parameter.

Provenance vocabulary (MATCH emits one of these per binding):

- ``"turn_context"``   — value came from τ (user-asserted unless τ is tagged).
- ``"scratchpad"``     — value came from S (populated by γ-flatten, producer
                          tool output, or retrieve hook).
- ``"task.parameters"`` — value came from the DECOMPOSE-emitted plan hint.
- ``"default"``        — value came from registry ``defaults`` map.
- ``"retrieve"``       — value came from a caller ``retrieve`` hook.
- ``"llm_fill"``       — value was synthesized by ``run_llm_fill_missing_params``.
- ``"tool_result"``    — value came from a prior tool call's return (explicitly
                          tagged by the caller when writing τ or S).

Opt-in ``param_sources`` field on a registry entry constrains which
provenances are acceptable for each param. When set and an incoming binding's
source is not in the allow-list, MATCH returns
``status="failure", reason="forbidden_source"``. When unset, any provenance is
fine — full backward compatibility.

τ and S can carry either a plain value ``v`` or a structured tag
``{"value": v, "source": "<tag>"}``. The tagged shape is used by callers that
want to declare provenance; untagged values default to the natural source
(``turn_context`` from τ, ``scratchpad`` from S).

MATCH emits ``binding_sources: {param: source}`` alongside ``bindings`` when
the feature is in use (either any tag was seen, or a ``param_sources`` gate
was configured).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import MatchFuncsAndParams
from nre.schemas import LLMMatchFillParams, LLMMatchFuncOutput


# ── 1. Baseline emission — untagged inputs still produce sources ─────


def test_untagged_tau_value_reported_as_turn_context_source():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["turn_context"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {"x": "v"}, tools)
    assert r["status"] == "success"
    assert r["binding_sources"] == {"x": "turn_context"}


def test_untagged_scratchpad_value_reported_as_scratchpad_source():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["scratchpad"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {"x": "v"}, {}, tools)
    assert r["status"] == "success"
    assert r["binding_sources"] == {"x": "scratchpad"}


def test_task_parameter_fallback_reported_as_task_parameters_source():
    t = UnitTask("t", tool_name="f", parameters={"x": "hint"})
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["task.parameters"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["binding_sources"] == {"x": "task.parameters"}


def test_registry_default_reported_as_default_source():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "defaults": {"x": 7},
            "param_sources": {"x": ["default"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["binding_sources"] == {"x": "default"}


def test_llm_fill_reported_as_llm_fill_source():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["llm_fill"]},
            "execute": lambda x: x,
        }
    }
    client = MagicMock()

    def chat(messages, *, response_model=None, **kwargs):
        if response_model is LLMMatchFillParams:
            return LLMMatchFillParams(values={"x": "synth"})
        if response_model is LLMMatchFuncOutput:
            return LLMMatchFuncOutput(tool_name="f", reasoning="")
        raise AssertionError(response_model)

    client.chat.side_effect = chat
    m = MatchFuncsAndParams(llm_client=client, llm_fill_missing_params=True)
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["binding_sources"] == {"x": "llm_fill"}


# ── 2. Tagged inputs override the default provenance ────────────────


def test_tau_entry_with_source_tool_result_reported_correctly():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["tool_result"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    # τ entry is a structured tag; its ``value`` is used for the binding and
    # its ``source`` becomes the recorded provenance.
    r = m(t, {}, {"x": {"value": "fromtool", "source": "tool_result"}}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "fromtool"
    assert r["binding_sources"] == {"x": "tool_result"}


def test_scratchpad_entry_with_source_tool_result_reported_correctly():
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["tool_result"]},
            "execute": lambda x: x,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {"x": {"value": "fromtool", "source": "tool_result"}}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "fromtool"
    assert r["binding_sources"] == {"x": "tool_result"}


# ── 3. Gate: forbidden source → failure ─────────────────────────────


def test_param_sources_forbids_turn_context_when_only_tool_result_allowed():
    """τ² airline task_0: user-asserted payment IDs must be rejected when the
    tool declares ``param_sources: {payment_methods: ["scratchpad","tool_result"]}``."""
    t = UnitTask(
        "t", tool_name="book_reservation", parameters={"payment_methods": ["CERT67890"]}
    )
    tools = {
        "book_reservation": {
            "params": ["payment_methods"],
            "param_sources": {"payment_methods": ["scratchpad", "tool_result"]},
            "execute": lambda payment_methods: payment_methods,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "failure"
    assert r["reason"] == "forbidden_source"
    assert r["param"] == "payment_methods"
    assert r["source"] == "task.parameters"


def test_param_sources_allows_tagged_tool_result_value():
    """Same tool; when the value comes in tagged as tool_result via τ, accept."""
    t = UnitTask("t", tool_name="book_reservation")
    tools = {
        "book_reservation": {
            "params": ["payment_methods"],
            "param_sources": {"payment_methods": ["scratchpad", "tool_result"]},
            "execute": lambda payment_methods: payment_methods,
        }
    }
    m = MatchFuncsAndParams()
    r = m(
        t,
        {},
        {
            "payment_methods": {
                "value": ["certificate_7504069", "credit_card_4421486"],
                "source": "tool_result",
            }
        },
        tools,
    )
    assert r["status"] == "success"
    assert r["bindings"]["payment_methods"] == [
        "certificate_7504069",
        "credit_card_4421486",
    ]
    assert r["binding_sources"]["payment_methods"] == "tool_result"


def test_param_sources_unset_means_no_gate_full_back_compat():
    """Tools without ``param_sources`` behave exactly as before. No source
    emitted in the result payload unless the feature is actually in use."""
    t = UnitTask("t", tool_name="f", parameters={"x": "anything"})
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "anything"
    assert "binding_sources" not in r, "no source payload when feature unused"


# ── 4. Mixed — some params gated, others open ────────────────────────


def test_partial_gating_only_gated_param_checked():
    t = UnitTask(
        "t",
        tool_name="book_reservation",
        parameters={"user_id": "alice", "payment_methods": ["CERT67890"]},
    )
    tools = {
        "book_reservation": {
            "params": ["user_id", "payment_methods"],
            "param_sources": {
                # user_id gate only allows task.parameters; payment_methods only allows tool_result.
                "user_id": ["task.parameters"],
                "payment_methods": ["tool_result"],
            },
            "execute": lambda user_id, payment_methods: (user_id, payment_methods),
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "failure"
    assert r["param"] == "payment_methods"
    assert r["source"] == "task.parameters"


def test_source_propagates_through_tagged_scratchpad_read():
    """A value written into S with a tag must retain its source when MATCH
    reads it back, even though the raw scratchpad dict may hold the tag dict
    instead of the bare value."""
    t = UnitTask("t", tool_name="f")
    tools = {
        "f": {
            "params": ["x"],
            "param_sources": {"x": ["tool_result"]},
            "execute": lambda x: x,
        }
    }
    scratch = {"x": {"value": 42, "source": "tool_result"}}
    m = MatchFuncsAndParams()
    r = m(t, scratch, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == 42
    assert r["binding_sources"]["x"] == "tool_result"
