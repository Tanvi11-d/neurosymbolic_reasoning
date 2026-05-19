"""PolicyConstraints primitive — declarative pre-mutation policy gate.

Motivation from τ² airline:

- task 15 — user asked to REMOVE a passenger. Policy: "The user can modify
  passengers but cannot modify the *number* of passengers." Ground truth was
  empty (refuse). Agent wrongly proceeded with ``cancel_reservation``.
- Other policy rules: "Basic economy flights cannot be modified." "All
  payment methods must already be in user profile." Many of these are
  symbolic predicates expressible as Python functions over ``bindings +
  scratchpad``.

The primitive evaluates a set of named policy rules against a candidate
tool call and returns ``{status: allow}`` or ``{status: deny, policy, reason}``.

Design:

    PolicyRule = {
        "applies_to": list[str] OR callable(tool_name, task, bindings) -> bool,
        "predicate": callable(bindings, scratchpad, turn_context) -> bool,
        "reason":    str,
    }

    PolicyConstraints.forward(
        tool_name, task, bindings, scratchpad, turn_context
    ) -> {"status": "allow"}
       | {"status": "deny", "policy": name, "reason": str}

Evaluated rules are recorded under ``evaluated: list[{policy, applies, result}]``
when ``trace=True``. First denial short-circuits (spec-style single block).
"""

from __future__ import annotations

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import PolicyConstraints


# ── Allow path ───────────────────────────────────────────────────────


def test_no_policies_configured_always_allows():
    p = PolicyConstraints()
    r = p("f", UnitTask("t"), {"x": 1}, {}, {})
    assert r["status"] == "allow"


def test_policy_that_does_not_apply_is_skipped():
    policies = {
        "only_for_book": {
            "applies_to": ["book_reservation"],
            "predicate": lambda *_: False,  # would always deny if applied
            "reason": "would deny",
        }
    }
    p = PolicyConstraints(policies=policies)
    r = p("cancel_reservation", UnitTask("t"), {}, {}, {})
    assert r["status"] == "allow"


def test_applicable_policy_passes_allow():
    policies = {
        "has_user_id": {
            "applies_to": ["book_reservation"],
            "predicate": lambda b, _s, _tau: bool(b.get("user_id")),
            "reason": "user_id must be set",
        }
    }
    p = PolicyConstraints(policies=policies)
    r = p("book_reservation", UnitTask("t"), {"user_id": "u1"}, {}, {})
    assert r["status"] == "allow"


# ── Deny path ────────────────────────────────────────────────────────


def test_single_policy_denies():
    policies = {
        "needs_confirm": {
            "applies_to": ["book_reservation"],
            "predicate": lambda _b, s, _tau: s.get("confirmed.book") is True,
            "reason": "user confirmation required before booking",
        }
    }
    p = PolicyConstraints(policies=policies)
    r = p("book_reservation", UnitTask("t"), {"user_id": "u1"}, {}, {})
    assert r["status"] == "deny"
    assert r["policy"] == "needs_confirm"
    assert "confirmation" in r["reason"]


def test_first_denial_short_circuits():
    """Evaluation is deterministic by sorted policy name; first denial wins."""
    order = []

    def make_pred(name, result):
        def pred(_b, _s, _tau):
            order.append(name)
            return result

        return pred

    policies = {
        "b_always_deny": {
            "applies_to": ["f"],
            "predicate": make_pred("b", False),
            "reason": "B said no",
        },
        "a_always_pass": {
            "applies_to": ["f"],
            "predicate": make_pred("a", True),
            "reason": "A passed",
        },
        "c_also_deny": {
            "applies_to": ["f"],
            "predicate": make_pred("c", False),
            "reason": "C said no",
        },
    }
    p = PolicyConstraints(policies=policies)
    r = p("f", UnitTask("t"), {}, {}, {})
    assert r["status"] == "deny"
    # Sorted name order: a (pass), b (deny → stop). c must NOT have evaluated.
    assert order == ["a", "b"]
    assert r["policy"] == "b_always_deny"


def test_applies_to_callable():
    """``applies_to`` may be a callable for dynamic dispatch."""

    def dynamic(tool_name, _task, bindings):
        return (
            tool_name.startswith("update_") and bindings.get("cabin") == "basic_economy"
        )

    policies = {
        "basic_economy_immutable": {
            "applies_to": dynamic,
            "predicate": lambda *_: False,
            "reason": "Basic economy flights cannot be modified.",
        }
    }
    p = PolicyConstraints(policies=policies)
    # Applies: update_ + basic_economy → deny
    r = p(
        "update_reservation_flights",
        UnitTask("t"),
        {"cabin": "basic_economy", "flights": []},
        {},
        {},
    )
    assert r["status"] == "deny"
    assert r["policy"] == "basic_economy_immutable"

    # Doesn't apply: update_ + economy → allow
    r = p(
        "update_reservation_flights",
        UnitTask("t"),
        {"cabin": "economy", "flights": []},
        {},
        {},
    )
    assert r["status"] == "allow"

    # Doesn't apply: book_ + basic_economy → allow
    r = p(
        "book_reservation",
        UnitTask("t"),
        {"cabin": "basic_economy"},
        {},
        {},
    )
    assert r["status"] == "allow"


# ── Tracing ──────────────────────────────────────────────────────────


def test_trace_lists_evaluations_on_allow():
    policies = {
        "p1": {"applies_to": ["f"], "predicate": lambda *_: True, "reason": ""},
        "p2": {"applies_to": ["g"], "predicate": lambda *_: True, "reason": ""},
    }
    p = PolicyConstraints(policies=policies, trace=True)
    r = p("f", UnitTask("t"), {}, {}, {})
    assert r["status"] == "allow"
    assert r["evaluated"] == [
        {"policy": "p1", "applies": True, "result": "allow"},
        {"policy": "p2", "applies": False, "result": "skip"},
    ]


def test_trace_lists_evaluations_on_deny():
    policies = {
        "p1": {"applies_to": ["f"], "predicate": lambda *_: True, "reason": ""},
        "p2": {"applies_to": ["f"], "predicate": lambda *_: False, "reason": "bad"},
    }
    p = PolicyConstraints(policies=policies, trace=True)
    r = p("f", UnitTask("t"), {}, {}, {})
    assert r["status"] == "deny"
    # Only p1 + p2 evaluated; p2 denies → short-circuits.
    assert r["evaluated"] == [
        {"policy": "p1", "applies": True, "result": "allow"},
        {"policy": "p2", "applies": True, "result": "deny"},
    ]
