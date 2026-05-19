"""P1 — CHECK-PREREQUISITES registry-driven defaults (spec Sub-process 4).

The spec describes three callables that make CHECK operational:

- ``GET-PREREQS(t)`` — set of scratchpad keys task ``t`` requires.
- ``Satisfied(p, v)`` — predicate validating S(p) is a valid value for key p.
- ``TASK-TO-CHECK(p)`` — generate a remediation task to populate absent p.

In the shipping implementation, CheckPrerequisites accepts these as callable
hooks via config; the production ``build_engine()`` wires none of them, so
CHECK is an inert no-op for every task.

This test suite pins the **registry-driven default** behavior: when the
caller passes ``tools_registry`` (the same Ω that MATCH consults) with
``requires``, ``produces``, and ``param_types`` fields on entries, the
primitive synthesizes the three callables automatically.

Contract for the generic schema, applied consistently across any tool family:

- ``entry["requires"]: list[str]`` — scratchpad keys that MUST be present and
  satisfied before this tool can run.
- ``entry["produces"]: list[str]`` — scratchpad keys this tool writes into S.
- ``entry["param_types"]: dict[str, str]`` — per-param value-type tag used by
  ``Satisfied``. Reserved tags this suite checks:
     ``"zipcode"`` — 5-digit numeric string.
     ``"non_empty_str"`` — str with len > 0.
     any other tag ⇒ value is non-None.

Caller-supplied hooks always take precedence over these defaults.
"""

from __future__ import annotations

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import CheckPrerequisites


# ── GET-PREREQS from registry.requires ───────────────────────────────


def test_default_get_prereqs_reads_requires_from_registry():
    registry = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["zipcode.cityA", "zipcode.cityB"],
        },
        "get_zipcode_based_on_city": {
            "params": ["city"],
            "produces": ["zipcode.{city}"],
        },
    }
    t = UnitTask("t1", tool_name="estimate_distance")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    # Both required keys are absent.
    assert r["status"] == "absent"
    assert r["key"] in {"zipcode.cityA", "zipcode.cityB"}


def test_default_get_prereqs_empty_requires_returns_ready():
    registry = {"pwd": {"params": [], "requires": []}}
    t = UnitTask("t1", tool_name="pwd")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    assert r["status"] == "ready"


def test_default_get_prereqs_unknown_tool_returns_ready():
    """Tool not in registry: no prereqs known, trivially ready (spec POST-3)."""
    t = UnitTask("t1", tool_name="not_in_registry")
    cp = CheckPrerequisites(tools_registry={})
    r = cp.forward(t, {})
    assert r["status"] == "ready"


# ── Satisfied from registry.param_types ──────────────────────────────


def test_default_satisfied_zipcode_validator_rejects_city_name():
    """City names are not zipcodes; S(cityA)='San Francisco' must fail Satisfied."""
    registry = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["cityA", "cityB"],
            "param_types": {"cityA": "zipcode", "cityB": "zipcode"},
        },
    }
    t = UnitTask("t1", tool_name="estimate_distance")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {"cityA": "San Francisco", "cityB": "Rivermist"})
    assert r["status"] == "blocked"  # spec '¬Met'
    assert r["key"] in {"cityA", "cityB"}


def test_default_satisfied_zipcode_validator_accepts_five_digits():
    registry = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["cityA", "cityB"],
            "param_types": {"cityA": "zipcode", "cityB": "zipcode"},
        },
    }
    t = UnitTask("t1", tool_name="estimate_distance")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {"cityA": "94016", "cityB": "83214"})
    assert r["status"] == "ready"


def test_default_satisfied_non_empty_str_rejects_empty():
    registry = {
        "cat": {
            "params": ["file_name"],
            "requires": ["file_name"],
            "param_types": {"file_name": "non_empty_str"},
        },
    }
    t = UnitTask("t1", tool_name="cat")
    cp = CheckPrerequisites(tools_registry=registry)
    assert cp.forward(t, {"file_name": ""})["status"] == "blocked"
    assert cp.forward(t, {"file_name": "report.txt"})["status"] == "ready"


def test_default_satisfied_unknown_tag_falls_back_to_non_null():
    registry = {
        "f": {
            "params": ["x"],
            "requires": ["x"],
            "param_types": {"x": "some_unrecognized_tag"},
        },
    }
    t = UnitTask("t1", tool_name="f")
    cp = CheckPrerequisites(tools_registry=registry)
    assert cp.forward(t, {"x": None})["status"] == "blocked"
    assert cp.forward(t, {"x": "anything"})["status"] == "ready"


# ── TASK-TO-CHECK synthesized from registry.produces ─────────────────


def test_default_remediation_finds_producer_for_absent_key():
    registry = {
        "estimate_distance": {
            "params": ["cityA", "cityB"],
            "requires": ["zipcode_A", "zipcode_B"],
        },
        "get_zipcode_A": {
            "params": [],
            "produces": ["zipcode_A"],
        },
    }
    t = UnitTask("t1", tool_name="estimate_distance")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    rem = r.get("remediation_task")
    assert rem is not None
    assert rem.tool_name == "get_zipcode_A"


def test_default_remediation_none_when_no_producer():
    registry = {
        "consumer": {"params": [], "requires": ["mystery_key"]},
        # no tool declares produces: [mystery_key]
    }
    t = UnitTask("t1", tool_name="consumer")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    assert r["status"] == "absent"
    assert "remediation_task" not in r


def test_default_remediation_picks_first_producer_deterministically():
    """Two tools produce the same key — first-in-sorted-order wins (deterministic)."""
    registry = {
        "consumer": {"params": [], "requires": ["k"]},
        "b_producer": {"params": [], "produces": ["k"]},
        "a_producer": {"params": [], "produces": ["k"]},
    }
    t = UnitTask("t1", tool_name="consumer")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    rem = r.get("remediation_task")
    assert rem is not None
    assert rem.tool_name == "a_producer"  # sorted order


# ── Caller hooks still override defaults ─────────────────────────────


def test_caller_get_prereqs_overrides_registry_requires():
    registry = {"t": {"requires": ["from_registry"]}}
    t = UnitTask("t1", tool_name="t")
    cp = CheckPrerequisites(
        tools_registry=registry,
        get_prereqs=lambda _t: ["from_caller"],
    )
    r = cp.forward(t, {})
    # Prereq must be 'from_caller' (not 'from_registry')
    assert r.get("key") == "from_caller"


def test_caller_classify_overrides_default_satisfied():
    """Caller's classify_prereq always wins, even when registry has param_types."""
    registry = {
        "t": {"requires": ["x"], "param_types": {"x": "zipcode"}},
    }
    t = UnitTask("t1", tool_name="t")
    cp = CheckPrerequisites(
        tools_registry=registry,
        classify_prereq=lambda _k, _t, _s: "met",  # force met
    )
    # Even though S(x)='not a zipcode', classify forces 'met'.
    r = cp.forward(t, {"x": "not a zipcode"})
    assert r["status"] == "ready"


def test_caller_remediation_for_overrides_registry_producer():
    registry = {
        "consumer": {"requires": ["k"]},
        "registry_producer": {"produces": ["k"]},
    }
    custom_rem = UnitTask("custom", tool_name="caller_producer")
    t = UnitTask("t1", tool_name="consumer")
    cp = CheckPrerequisites(
        tools_registry=registry,
        remediation_for=lambda _k, _t, _s, _o: custom_rem,
    )
    r = cp.forward(t, {})
    assert r["remediation_task"] is custom_rem


# ── Backward compatibility ───────────────────────────────────────────


def test_no_registry_and_no_hooks_still_ready():
    """Current production default: no hooks, no registry → task is trivially ready."""
    t = UnitTask("t1", tool_name="anything")
    cp = CheckPrerequisites()
    r = cp.forward(t, {})
    assert r["status"] == "ready"


def test_registry_with_no_requires_field_does_nothing():
    """Registry entries without 'requires' are not prereq sources."""
    registry = {"t": {"params": ["x"]}}  # no 'requires'
    t = UnitTask("t1", tool_name="t")
    cp = CheckPrerequisites(tools_registry=registry)
    r = cp.forward(t, {})
    assert r["status"] == "ready"
