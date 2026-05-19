"""DECOMPOSE benchmark: fixtures on disk; scoring helpers; stub client unit tests."""

import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nre.benchmark.decompose_harness import (
    load_benchmark_cases,
    ordered_tool_names,
    run_benchmark,
    run_decompose_for_case,
)
from nre.schemas import LLMDecomposeOutput, LLMTaskItem
from nre.primitives.deterministic import UnitTask


def _cases_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "decompose_benchmark"
        / "cases.json"
    )


def test_benchmark_cases_file_exists():
    assert _cases_path().is_file(), "run scripts/gen_decompose_benchmark_json.py"


def test_run_decompose_for_case_uses_llm_client():
    """Harness must call the provided client (unit test; not OpenRouter)."""
    case = load_benchmark_cases(_cases_path())[0]
    parsed = LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="t1",
                description="x",
                tool_name="cd",
                depends_on=[],
                parameters={"folder": "workspace"},
            ),
            LLMTaskItem(
                id="t2",
                description="y",
                tool_name="mkdir",
                depends_on=["t1"],
                parameters={"dir_name": "archive"},
            ),
            LLMTaskItem(
                id="t3",
                description="z",
                tool_name="mv",
                depends_on=["t2"],
                parameters={"source": "a", "destination": "b"},
            ),
        ],
        config_keys={},
        reasoning="unit",
    )
    client = MagicMock()
    client.chat.return_value = parsed
    out = run_decompose_for_case(case, llm_client=client)
    client.chat.assert_called()
    assert [t.tool_name for t in out["turns"]] == ["cd", "mkdir", "mv"]


def test_run_benchmark_with_stub_client():
    cases = load_benchmark_cases(_cases_path())[:2]
    calls = {"n": 0}
    lock = threading.Lock()

    def side(*_a, **_k):
        with lock:
            i = calls["n"]
            calls["n"] += 1
        first = cases[i]["expected"]["ordered_tool_names"][0]
        return LLMDecomposeOutput(
            tasks=[
                LLMTaskItem(
                    id="t1",
                    description="",
                    tool_name=first,
                    depends_on=[],
                    parameters={},
                )
            ],
            config_keys={},
            reasoning="",
        )

    stub = MagicMock()
    stub.chat.side_effect = side
    summary = run_benchmark(cases=cases, llm_client=stub, max_workers=4)
    assert summary.total == 2
    assert summary.workers_used == 2
    assert stub.chat.call_count == 2


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY", "").strip(),
    reason="OPENROUTER_API_KEY not set",
)
def test_one_benchmark_case_live_smoke():
    """Optional: one real OpenRouter call against first fixture."""
    case = load_benchmark_cases(_cases_path())[0]
    out = run_decompose_for_case(case)
    assert "turns" in out
    assert isinstance(out["turns"], list)


def test_ordered_tool_names_empty():
    assert ordered_tool_names([]) == []


def test_ordered_tool_names_cycle():
    a = UnitTask("a", tool_name="x", depends_on=("b",))
    b = UnitTask("b", tool_name="y", depends_on=("a",))
    assert ordered_tool_names([a, b]) is None
