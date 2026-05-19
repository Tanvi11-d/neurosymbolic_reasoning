"""τ² airline — PolicyConstraints catches the real policy-violation failures.

Fixtures used:
  - ``task_15_policy_refuse.json`` — user asked to remove a passenger; policy
    forbids changing the number of passengers. Ground truth: refuse.
  - ``task_basic_economy_cannot_modify.json`` — policy: basic economy flights
    cannot be modified. Ground truth: refuse.

Each test constructs a realistic ``airline_policies`` dict (the same shape a
production τ² harness adapter would wire into ``agent_turn``) and asserts
that PolicyConstraints denies the offending call with the correct reason.
"""

from __future__ import annotations

import json
from pathlib import Path

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import PolicyConstraints


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "tau2" / "airline"


def _load(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def airline_policies() -> dict:
    """Realistic subset of the airline wiki encoded as predicates. The full
    wiki has ~15 rules; these are the ones that show up in the failing
    trajectories and can be expressed without LLM reasoning.
    """

    def _passenger_count_preserved(bindings, _s, _tau):
        # Policy: "The user can modify passengers but cannot modify the
        # number of passengers."
        before = bindings.get("_current_passenger_count")
        after = bindings.get("passengers")
        if after is None or before is None:
            return True  # can't evaluate → don't block
        return len(after) == before

    def _not_basic_economy(bindings, _s, _tau):
        # Policy: "Basic economy flights cannot be modified."
        return bindings.get("cabin") != "basic_economy"

    def _payment_ids_from_profile(bindings, scratchpad, _tau):
        # Policy: "All payment methods must already be in user profile."
        profile_ids = scratchpad.get("user.payment_method_ids")
        if not profile_ids:
            return True  # no profile info → don't block in policy layer
        proposed = bindings.get("payment_methods") or []
        return all(p.get("payment_id") in profile_ids for p in proposed)

    return {
        "passenger_count_immutable": {
            "applies_to": ["update_reservation_passengers"],
            "predicate": _passenger_count_preserved,
            "reason": "The number of passengers on a reservation cannot be changed.",
        },
        "basic_economy_immutable": {
            "applies_to": [
                "update_reservation_flights",
                "update_reservation_baggages",
            ],
            "predicate": _not_basic_economy,
            "reason": "Basic economy flights cannot be modified.",
        },
        "payment_from_profile_only": {
            "applies_to": ["book_reservation"],
            "predicate": _payment_ids_from_profile,
            "reason": "All payment methods must already be in the user profile.",
        },
    }


# ── Task 15: passenger-count refusal ─────────────────────────────────


def test_task_15_refuse_remove_passenger():
    fx = _load("task_15_policy_refuse.json")
    p = PolicyConstraints(policies=airline_policies())

    # Simulate the agent's candidate call: drop Sophia, keep only James.
    bindings = {
        "reservation_id": fx["reservation_from_tool"]["reservation_id"],
        "passengers": fx["proposed_new_passengers"],
        "_current_passenger_count": len(fx["reservation_from_tool"]["passengers"]),
    }
    r = p("update_reservation_passengers", UnitTask("t"), bindings, {}, {})
    assert r["status"] == "deny"
    assert r["policy"] == "passenger_count_immutable"
    assert "number of passengers" in r["reason"].lower()


def test_task_15_allow_when_count_preserved():
    """Sanity: same tool, same fixture, but a swap (not a removal) must allow."""
    fx = _load("task_15_policy_refuse.json")
    p = PolicyConstraints(policies=airline_policies())
    # Swap Sophia for someone else — count preserved.
    bindings = {
        "reservation_id": fx["reservation_from_tool"]["reservation_id"],
        "passengers": [
            {"first_name": "James", "last_name": "Patel", "dob": "1970-06-15"},
            {"first_name": "Nadia", "last_name": "Patel", "dob": "1972-09-01"},
        ],
        "_current_passenger_count": len(fx["reservation_from_tool"]["passengers"]),
    }
    r = p("update_reservation_passengers", UnitTask("t"), bindings, {}, {})
    assert r["status"] == "allow"


# ── Basic economy refusal ────────────────────────────────────────────


def test_basic_economy_cannot_be_modified():
    fx = _load("task_basic_economy_cannot_modify.json")
    p = PolicyConstraints(policies=airline_policies())
    bindings = {
        "reservation_id": fx["reservation_from_tool"]["reservation_id"],
        "cabin": fx["reservation_from_tool"]["cabin"],
        "flights": fx["reservation_from_tool"]["flights"],
    }
    r = p("update_reservation_flights", UnitTask("t"), bindings, {}, {})
    assert r["status"] == "deny"
    assert r["policy"] == "basic_economy_immutable"
    assert "basic economy" in r["reason"].lower()


def test_economy_modification_allowed():
    p = PolicyConstraints(policies=airline_policies())
    bindings = {
        "reservation_id": "RES123",
        "cabin": "economy",
        "flights": [{"flight_number": "HAT001", "date": "2024-05-20"}],
    }
    r = p("update_reservation_flights", UnitTask("t"), bindings, {}, {})
    assert r["status"] == "allow"


# ── Payment source policy (complements source-tracked bindings) ──────


def test_payment_methods_must_be_from_profile():
    """The PolicyConstraints layer is a second line of defense beyond the
    source-gate: even if a fake ID somehow slipped past the source check,
    the policy still verifies the value is in the known profile list."""
    p = PolicyConstraints(policies=airline_policies())
    scratch = {
        "user.payment_method_ids": ["certificate_7504069", "credit_card_4421486"]
    }
    bindings = {
        "payment_methods": [
            {"payment_id": "CERT67890", "amount": 250},  # NOT in profile
        ],
    }
    r = p("book_reservation", UnitTask("t"), bindings, scratch, {})
    assert r["status"] == "deny"
    assert r["policy"] == "payment_from_profile_only"


def test_payment_methods_allowed_when_in_profile():
    p = PolicyConstraints(policies=airline_policies())
    scratch = {
        "user.payment_method_ids": ["certificate_7504069", "credit_card_4421486"]
    }
    bindings = {
        "payment_methods": [
            {"payment_id": "certificate_7504069", "amount": 250},
            {"payment_id": "credit_card_4421486", "amount": 5},
        ],
    }
    r = p("book_reservation", UnitTask("t"), bindings, scratch, {})
    assert r["status"] == "allow"
