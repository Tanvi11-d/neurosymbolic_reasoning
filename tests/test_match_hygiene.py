"""P4 — MATCH-FUNCS-AND-PARAMS hygiene (spec Sub-process 5).

Three independent primitive-level behaviors, tested in isolation:

1. **Unknown-parameter drop.** Spec Stage 2 iterates ``m ∈ PARAMS(f)``. Keys that
   live on ``task.parameters`` but not on ``PARAMS(f)`` must not leak into
   ``bindings`` — otherwise paramless tools like ``ls`` get phantom args.

2. **LOOKUP priority.** Spec INV-3 says ``τ`` is always consulted before ``S``;
   the implementation additionally accepts ``task.parameters`` as a DECOMPOSE
   hint, **but τ must still beat it** (a live turn context must never be
   silently overridden by a stale decompose hint).

3. **Registry-declared defaults.** Spec Stage 2 treats every ``m ∈ PARAMS(f)``
   as required. In practice tools have optional params with natural defaults
   (``mentions=[]`` for ``post_tweet``). A ``defaults`` map on the registry
   entry is consulted *before* invoking the LLM fill — LLM fill only runs for
   params that are still unresolved.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import MatchFuncsAndParams
from nre.schemas import LLMMatchFillParams, LLMMatchFuncOutput


# ── 1. Unknown-parameter drop ────────────────────────────────────────


def test_unknown_task_param_is_dropped_for_paramless_tool():
    """``ls`` has no declared params; ``task.parameters={'a': ''}`` must not leak."""
    t = UnitTask("t1", tool_name="ls", parameters={"a": ""})
    tools = {"ls": {"params": [], "execute": lambda: "files"}}
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"] == {}, f"phantom param leaked: {r['bindings']}"


def test_unknown_task_param_is_dropped_with_declared_params():
    """Declared param kept, undeclared extras dropped."""
    t = UnitTask(
        "t1",
        tool_name="cp",
        parameters={"source": "a.txt", "destination": "b/", "extra": "junk"},
    )
    tools = {
        "cp": {
            "params": ["source", "destination"],
            "execute": lambda source, destination: (source, destination),
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"] == {"source": "a.txt", "destination": "b/"}


# ── 2. LOOKUP priority (τ > S > task.parameters) ─────────────────────


def test_turn_context_beats_task_parameters():
    """Spec INV-3: τ must beat any other source for the same key, including
    a stale DECOMPOSE hint in ``task.parameters``."""
    t = UnitTask("t1", tool_name="f", parameters={"x": "stale"})
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams()
    r = m(t, {}, {"x": "fresh"}, tools)  # scratchpad={}, turn_context={"x": "fresh"}
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "fresh", "turn_context τ must override task.parameters"


def test_scratchpad_beats_task_parameters():
    """With τ empty, S is the spec's second source. S must still beat
    ``task.parameters`` (the DECOMPOSE hint is the fallback, not primary)."""
    t = UnitTask("t1", tool_name="f", parameters={"x": "from_task"})
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams()
    r = m(t, {"x": "from_s"}, {}, tools)  # scratchpad={"x": "from_s"}, τ={}
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "from_s"


def test_task_parameters_used_when_tau_and_s_both_miss():
    """With τ and S both empty, ``task.parameters`` is the fallback source."""
    t = UnitTask("t1", tool_name="f", parameters={"x": "from_task"})
    tools = {"f": {"params": ["x"], "execute": lambda x: x}}
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["x"] == "from_task"


# ── 3. Registry-declared defaults ────────────────────────────────────


def test_registry_defaults_applied_before_llm_fill():
    """A ``defaults`` map on the registry entry is applied before LLM MATCH_FILL.
    If defaults resolve all missing params, LLM fill is never called."""
    t = UnitTask("t1", tool_name="post_tweet", parameters={"content": "hi", "tags": []})
    tools = {
        "post_tweet": {
            "params": ["content", "tags", "mentions"],
            "defaults": {"mentions": []},
            "execute": lambda content, tags, mentions: {"ok": True, "m": mentions},
        }
    }
    client = MagicMock()
    m = MatchFuncsAndParams(llm_client=client, llm_fill_missing_params=True)
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["mentions"] == []
    assert client.chat.call_count == 0, (
        "LLM fill must not fire when defaults close the gap"
    )


def test_llm_fill_still_runs_for_params_not_covered_by_defaults():
    """Defaults fill some params; LLM fill covers the remainder."""
    t = UnitTask("t1", tool_name="post_tweet", parameters={"tags": []})
    tools = {
        "post_tweet": {
            "params": ["content", "tags", "mentions"],
            "defaults": {"mentions": []},
            "execute": lambda content, tags, mentions: content,
        }
    }
    client = MagicMock()

    def chat(messages, *, response_model=None, **kwargs):
        if response_model is LLMMatchFillParams:
            return LLMMatchFillParams(values={"content": "llm_supplied"})
        if response_model is LLMMatchFuncOutput:
            return LLMMatchFuncOutput(tool_name="post_tweet", reasoning="")
        raise AssertionError(response_model)

    client.chat.side_effect = chat
    m = MatchFuncsAndParams(llm_client=client, llm_fill_missing_params=True)
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["content"] == "llm_supplied"
    assert r["bindings"]["mentions"] == []
    # LLM fill was called (once) — but only for 'content', not 'mentions'
    # (Tool match may or may not require a second LLM call depending on task.tool_name hint.)
    assert client.chat.call_count >= 1


def test_defaults_do_not_override_existing_bindings():
    """If τ / S / task.parameters already supplied a value, defaults must not clobber it."""
    t = UnitTask(
        "t1",
        tool_name="post_tweet",
        parameters={"content": "hi", "tags": [], "mentions": ["@alice"]},
    )
    tools = {
        "post_tweet": {
            "params": ["content", "tags", "mentions"],
            "defaults": {"mentions": []},
            "execute": lambda content, tags, mentions: mentions,
        }
    }
    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "success"
    assert r["bindings"]["mentions"] == ["@alice"]
