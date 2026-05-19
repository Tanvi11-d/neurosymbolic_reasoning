"""Live smoke — GLM 4.7 via Cerebras provider via OpenRouter.

Gated on ``OPENROUTER_API_KEY`` so it's skipped by default. Exercises the
new config-file path end-to-end:

1. Writes a temporary ``nre.toml`` with ``model = "z-ai/glm-4.7"`` and
   ``provider_order = ["cerebras/fp16"]``.
2. Points ``NRE_CONFIG`` at it.
3. Runs one real ``run_llm_decompose`` call against OpenRouter.
4. Asserts the request succeeded and the plan is coherent.

Does NOT assert a specific plan — different models pick different
decompositions for the same input. The test only verifies the transport
plumbing works with a non-MiniMax provider.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("openai")

from nre.library.hybrid import run_llm_decompose
from nre.llm.openrouter import OpenRouterClient, OpenRouterSettings
from nre.server.schemas import openai_tools_to_nre_registry


FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "bfcl_v4_live"
    / "zipcode_prereq_multi_turn_base_56.json"
)

pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY", "").strip(),
    reason="OPENROUTER_API_KEY not set — live test skipped",
)


def test_glm_4_7_via_cerebras_decomposes_a_real_bfcl_case(
    tmp_path: Path, monkeypatch
):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        """
llm:
  model: z-ai/glm-4.7
  openrouter:
    provider_order:
      - cerebras/fp16
    # GLM 4.7 is a reasoning model; disable reasoning for structured-JSON
    # kernel calls (otherwise completion budget is spent on chain-of-thought
    # and the JSON answer never arrives — finish_reason='length').
    reasoning:
      enabled: false
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.setenv("NRE_CONFIG", str(cfg))

    settings = OpenRouterSettings()
    assert settings.model == "z-ai/glm-4.7"
    assert settings.extra_body == {
        "provider": {"order": ["cerebras/fp16"]},
        "reasoning": {"enabled": False},
    }

    fx = json.loads(FIXTURE.read_text(encoding="utf-8"))
    registry = openai_tools_to_nre_registry(fx["tools"])
    client = OpenRouterClient(settings)

    raw = run_llm_decompose(
        client,
        iota=fx["user_message"],
        gamma={},
        tools_registry=registry,
    )
    turns = raw.get("turns") or []
    assert turns, f"GLM 4.7 (Cerebras) produced empty plan; raw={raw}"
    # Light sanity: at least one planned task references estimate_distance
    # OR a zipcode lookup (any coherent plan, not a fabrication).
    tool_names = []
    for t in turns:
        tn = t.tool_name if hasattr(t, "tool_name") else t.get("tool_name")
        if tn:
            tool_names.append(tn)
    assert any(
        tn in {"estimate_distance", "get_zipcode_based_on_city"} for tn in tool_names
    ), f"GLM 4.7 plan doesn't reference the distance workflow: {tool_names}"
