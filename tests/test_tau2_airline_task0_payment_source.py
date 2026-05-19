"""τ² airline task_0 — source-tracked bindings prevent the original failure.

Fixture: ``tests/fixtures/tau2/airline/task_0_payment_source.json``.

The original trajectory (from tau-bench run):

- The simulated user volunteered fake certificate IDs ("CERT67890" /
  "CERT12345") rather than the real ones from their user profile.
- The agent dutifully passed those fake IDs into ``book_reservation``.
- Ground truth used the REAL IDs (``certificate_7504069`` /
  ``credit_card_4421486``) which the policy requires ("All payment methods
  must already be in user profile for safety reasons").
- Reward was 0.0 — wrong action kwargs.

This test replays the exact binding situation through MATCH with the
appropriate ``param_sources`` gate on ``book_reservation.payment_methods``.
Two scenarios:

1. The fake IDs arrive via ``task.parameters`` (as if DECOMPOSE took the
   user's claim at face value). MATCH must return
   ``status="failure", reason="forbidden_source"``.
2. The real IDs arrive via τ tagged as ``tool_result`` (as if a prior
   ``get_user_details`` call's output was surfaced into τ by the harness).
   MATCH must accept and bind successfully.
"""

from __future__ import annotations

import json
from pathlib import Path

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import MatchFuncsAndParams


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "tau2"
    / "airline"
    / "task_0_payment_source.json"
)


def _load():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _book_reservation_tool():
    """Registry entry for ``book_reservation`` with the τ² policy encoded as
    a ``param_sources`` gate: ``payment_methods`` must be scratchpad- or
    tool_result-derived, never user-asserted."""
    return {
        "book_reservation": {
            "params": [
                "user_id",
                "origin",
                "destination",
                "flight_type",
                "cabin",
                "flights",
                "passengers",
                "payment_methods",
                "total_baggages",
                "nonfree_baggages",
                "insurance",
            ],
            "param_sources": {
                "payment_methods": ["scratchpad", "tool_result"],
                "passengers": ["scratchpad", "tool_result", "task.parameters"],
            },
            # No execute — bound-only path (server mode).
        }
    }


def test_rejects_user_asserted_fake_payment_ids():
    """Replicates the real failure: DECOMPOSE put the user's claimed fake IDs
    into ``task.parameters``. The gate rejects."""
    fx = _load()
    tools = _book_reservation_tool()

    # Build a UnitTask shaped like what DECOMPOSE would emit after trusting the
    # user's fake IDs. The key detail is that ``payment_methods`` arrives via
    # ``task.parameters``, i.e. source="task.parameters".
    fake_payments = [
        {"payment_id": "CERT67890", "amount": 250},
        {"payment_id": "CERT12345", "amount": 5},
    ]
    t = UnitTask(
        id="book_1",
        tool_name="book_reservation",
        parameters={
            "user_id": fx["user_profile_from_tool"]["user_id"],
            "origin": "JFK",
            "destination": "SEA",
            "flight_type": "one_way",
            "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}],
            "passengers": [
                {"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}
            ],
            "payment_methods": fake_payments,
            "total_baggages": 3,
            "nonfree_baggages": 0,
            "insurance": "no",
        },
    )

    m = MatchFuncsAndParams()
    r = m(t, {}, {}, tools)
    assert r["status"] == "failure"
    assert r["reason"] == "forbidden_source"
    assert r["param"] == "payment_methods"
    assert r["source"] == "task.parameters"
    # And confirm the rejected value was the fake-ID shape (helpful for
    # downstream telemetry — source tracking didn't drop the info).
    assert r["allowed"] == ["scratchpad", "tool_result"]


def test_accepts_real_payment_ids_when_sourced_from_tool_result():
    """Correct trajectory: a prior ``get_user_details`` call's output lands
    in τ tagged as ``source="tool_result"``. MATCH accepts and binds."""
    fx = _load()
    tools = _book_reservation_tool()

    real_payments = fx["expected_payment_methods"]
    # Simulate the harness / orchestrator tagging τ entries with provenance.
    tau = {
        "payment_methods": {"value": real_payments, "source": "tool_result"},
        "passengers": {
            "value": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
            "source": "tool_result",
        },
    }
    t = UnitTask(
        id="book_1",
        tool_name="book_reservation",
        parameters={
            "user_id": fx["user_profile_from_tool"]["user_id"],
            "origin": "JFK",
            "destination": "SEA",
            "flight_type": "one_way",
            "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}],
            "total_baggages": 3,
            "nonfree_baggages": 0,
            "insurance": "no",
        },
    )

    m = MatchFuncsAndParams()
    r = m(t, {}, tau, tools)
    assert r["status"] == "bound"
    assert r["bindings"]["payment_methods"] == real_payments
    assert r["binding_sources"]["payment_methods"] == "tool_result"
    assert r["binding_sources"]["passengers"] == "tool_result"


def test_payment_ids_from_gamma_flattened_profile_also_accepted():
    """Realistic end-to-end flow: γ (initial_config) contains the user's
    payment methods; γ-flatten puts them in S under a stable key; the tool's
    ``param_sources`` allows ``scratchpad``. No manual τ tagging needed.

    This is the shape a τ² harness adapter would use: write the profile into
    γ once, let γ-flatten populate S, let the source gate do the rest."""
    fx = _load()
    tools = _book_reservation_tool()

    # The harness already flattened user_profile.payment_methods into S under
    # the key ``payment_methods``. The scratchpad's source is the default
    # ``scratchpad`` (no tag), which is in the allow-list.
    real_payments = fx["expected_payment_methods"]
    scratch = {
        "payment_methods": real_payments,
        "passengers": [{"first_name": "Mia", "last_name": "Li", "dob": "1990-04-05"}],
    }
    t = UnitTask(
        id="book_1",
        tool_name="book_reservation",
        parameters={
            "user_id": fx["user_profile_from_tool"]["user_id"],
            "origin": "JFK",
            "destination": "SEA",
            "flight_type": "one_way",
            "cabin": "economy",
            "flights": [{"flight_number": "HAT136", "date": "2024-05-20"}],
            "total_baggages": 3,
            "nonfree_baggages": 0,
            "insurance": "no",
        },
    )

    m = MatchFuncsAndParams()
    r = m(t, scratch, {}, tools)
    assert r["status"] == "bound"
    assert r["bindings"]["payment_methods"] == real_payments
    assert r["binding_sources"]["payment_methods"] == "scratchpad"
