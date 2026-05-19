"""Live end-to-end regression — BFCL v4 zipcode prerequisite class.

Covers all 7 cases from run ``run_4bfb657b8373`` whose failure shape is
"DECOMPOSE emitted ``estimate_distance(cityA=<city_name>, cityB=<city_name>)``
instead of planning ``get_zipcode_based_on_city`` lookups first". Each case
is tested at turn 0 (the one where the failure originates).

The fix is entirely in the registry pipeline: it now preserves per-parameter
descriptions and response-schema field names that the host already sends.
No keyword lists, no tag canonicalization, no prompt changes. Once the LLM
sees the semantic hints the host already wrote, it plans the correct
sequence.

This test is gated by ``OPENROUTER_API_KEY`` so it's skipped in default runs.
It is SLOW (~15s per case, so ~100s for the full suite against minimax-m2.7).
Run with ``-v`` to see per-case pass/fail.

Correctness criteria per case:

1. DECOMPOSE output must include at least one ``get_zipcode_based_on_city``
   task before the ``estimate_distance`` task.
2. Each city the ground truth resolves must be the ``city`` argument of at
   least one lookup task.

These are the minimal symbolic properties the failing bundle showed DECOMPOSE
was violating. We do NOT assert exact parity with ground truth (count of
tasks, task ordering of unrelated steps, etc.) because the test should pass
as long as the zipcode-lookup prerequisite is planned correctly.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

pytest.importorskip("openai")

from nre.library.hybrid import run_llm_decompose
from nre.llm.openrouter import OpenRouterClient, OpenRouterSettings
from nre.server.schemas import openai_tools_to_nre_registry


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "bfcl_v4_live"

ZIPCODE_CASES = [
    "multi_turn_base_56",
    "multi_turn_base_57",
    "multi_turn_base_67",
    "multi_turn_base_69",
    "multi_turn_base_71",
    "multi_turn_base_77",
    "multi_turn_base_97",
]

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY", "").strip(),
    reason="OPENROUTER_API_KEY not set — live test skipped",
)


def _load_fixture(case_id: str) -> dict:
    return json.loads(
        (FIXTURE_DIR / f"zipcode_prereq_{case_id}.json").read_text(encoding="utf-8")
    )


def _tool_names_from_plan(raw: dict) -> list[str]:
    out = []
    for t in raw.get("turns") or []:
        tn = (
            t.tool_name
            if hasattr(t, "tool_name")
            else (t.get("tool_name") if isinstance(t, dict) else None)
        )
        if tn:
            out.append(tn)
    return out


def _lookup_cities_from_plan(raw: dict) -> list[str]:
    cities: list[str] = []
    for t in raw.get("turns") or []:
        tn = t.tool_name if hasattr(t, "tool_name") else t.get("tool_name")
        params = t.parameters if hasattr(t, "parameters") else t.get("parameters", {})
        if tn == "get_zipcode_based_on_city":
            city = (params or {}).get("city")
            if city:
                cities.append(str(city))
    return cities


_GT_CITY_RE = re.compile(r"get_zipcode_based_on_city\((?:city=)?['\"]([^'\"]+)['\"]\)")


def _cities_from_ground_truth(gt_turn: list[str]) -> list[str]:
    """Extract the city names the ground truth passes to get_zipcode_based_on_city."""
    out = []
    for call in gt_turn or []:
        m = _GT_CITY_RE.search(call)
        if m:
            out.append(m.group(1))
    return out


@pytest.fixture(scope="module")
def client() -> OpenRouterClient:
    """Shared client — one OpenRouterClient for the whole module."""
    return OpenRouterClient(OpenRouterSettings())


@pytest.fixture(scope="module")
def plans(client: OpenRouterClient) -> dict[str, dict]:
    """Run DECOMPOSE once per case, cache for downstream assertions.

    Module-scoped so we don't pay the LLM cost twice when both assertions
    (ordering and city-binding) run over the same case.
    """
    out: dict[str, dict] = {}
    for cid in ZIPCODE_CASES:
        fx = _load_fixture(cid)
        registry = openai_tools_to_nre_registry(fx["tools"])
        raw = run_llm_decompose(
            client,
            iota=fx["user_message"],
            gamma={},
            tools_registry=registry,
        )
        out[cid] = raw
    return out


@pytest.mark.parametrize("case_id", ZIPCODE_CASES)
def test_zipcode_lookup_planned_before_estimate_distance(case_id: str, plans: dict):
    """DECOMPOSE must include at least one get_zipcode_based_on_city task
    that precedes the estimate_distance task in the plan order."""
    raw = plans[case_id]
    plan = _tool_names_from_plan(raw)

    assert plan, f"[{case_id}] DECOMPOSE produced empty plan"
    assert "estimate_distance" in plan, (
        f"[{case_id}] estimate_distance missing from plan: {plan}"
    )

    zip_idx = [i for i, t in enumerate(plan) if t == "get_zipcode_based_on_city"]
    est_idx = plan.index("estimate_distance")

    assert zip_idx, (
        f"[{case_id}] no get_zipcode_based_on_city in plan. "
        f"plan={plan}"
    )
    assert min(zip_idx) < est_idx, (
        f"[{case_id}] get_zipcode_based_on_city must precede estimate_distance. "
        f"plan={plan}"
    )


@pytest.mark.parametrize("case_id", ZIPCODE_CASES)
def test_zipcode_lookups_bind_expected_cities(case_id: str, plans: dict):
    """Each city mentioned in the ground-truth lookup sequence must be passed
    to a ``get_zipcode_based_on_city`` task in the produced plan. Match is
    case-insensitive substring (tolerates small casing/punctuation drift)."""
    fx = _load_fixture(case_id)
    expected_cities = _cities_from_ground_truth(fx["ground_truth_tool_sequence"])
    assert expected_cities, (
        f"[{case_id}] fixture ground truth has no zipcode lookups to check "
        f"against: {fx['ground_truth_tool_sequence']}"
    )

    produced_cities = [c.lower() for c in _lookup_cities_from_plan(plans[case_id])]
    for want in expected_cities:
        want_l = want.lower()
        assert any(want_l in c or c in want_l for c in produced_cities), (
            f"[{case_id}] DECOMPOSE did not plan a lookup for city {want!r}. "
            f"produced_lookups={produced_cities}"
        )
