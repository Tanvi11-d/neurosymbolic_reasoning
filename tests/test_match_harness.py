"""MATCH harness: smoke fixtures, benchmark JSON, OpenRouter mocks for LLM rows."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nre.benchmark.match_harness import (
    evaluate_match_case,
    load_match_benchmark_cases,
    run_match_benchmark,
)
from nre.schemas import LLMMatchFillParams, LLMMatchFuncOutput


def _smoke_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "match_smoke"


def _bench_path() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "match_benchmark" / "cases.json"


def test_match_benchmark_file_exists():
    assert _bench_path().is_file(), "run scripts/gen_match_benchmark_json.py"


def test_ten_benchmark_cases():
    cases = load_match_benchmark_cases(_bench_path())
    assert len(cases) == 10
    assert sum(1 for c in cases if c.get("llm_case")) == 2


@pytest.mark.parametrize(
    "stem",
    [
        "scratchpad",
        "tau_wins",
        "tool_name_prefers",
        "bound",
        "match_func",
        "task_params",
    ],
)
def test_match_smoke_fixtures_evaluate(stem: str):
    path = _smoke_dir() / f"{stem}.json"
    assert path.is_file()
    raw = json.loads(path.read_text(encoding="utf-8"))
    case = {k: v for k, v in raw.items() if k != "description"}
    out = evaluate_match_case(case)
    exp = case["expected"]
    assert out["status"] == exp["status"]
    if "function" in exp:
        assert out["function"] == exp["function"]
    if "result" in exp:
        assert out["result"] == exp["result"]
    if "bindings" in exp:
        for k, v in exp["bindings"].items():
            assert out["bindings"][k] == v


@pytest.mark.parametrize(
    "case",
    [c for c in load_match_benchmark_cases(_bench_path()) if not c.get("llm_case")],
    ids=lambda c: c["id"],
)
def test_match_benchmark_deterministic_rows(case):
    out = evaluate_match_case(case)
    exp = case["expected"]
    assert out["status"] == exp["status"]
    if "function" in exp:
        assert out["function"] == exp["function"]
    if "result" in exp:
        assert out["result"] == exp["result"]
    if "bindings" in exp:
        for k, v in exp["bindings"].items():
            assert out["bindings"][k] == v


def _llm_mock_match_benchmark() -> MagicMock:
    client = MagicMock()

    def chat(messages, *, response_model=None, **kwargs):
        if response_model is LLMMatchFuncOutput:
            text = messages[-1]["content"].lower()
            if "pickme" in text:
                return LLMMatchFuncOutput(tool_name="pickme", reasoning="")
            return LLMMatchFuncOutput(tool_name="add", reasoning="")
        if response_model is LLMMatchFillParams:
            return LLMMatchFillParams(values={"a": 2, "b": 3})
        raise AssertionError(response_model)

    client.chat.side_effect = chat
    return client


def test_match_benchmark_llm_rows_with_mock():
    cases = [c for c in load_match_benchmark_cases(_bench_path()) if c.get("llm_case")]
    assert len(cases) == 2
    mock = _llm_mock_match_benchmark()
    for c in cases:
        out = evaluate_match_case(c, llm_client=mock)
        exp = c["expected"]
        assert out["status"] == exp["status"]
        assert out["function"] == exp["function"]
        assert out["result"] == exp["result"]
    assert mock.chat.call_count == 3


def test_run_match_benchmark_parallel_no_llm_subset():
    cases = [c for c in load_match_benchmark_cases(_bench_path()) if not c.get("llm_case")][:4]
    summary = run_match_benchmark(cases=cases, llm_client=None, max_workers=4)
    assert summary.total == 4
    assert summary.matched == 4


def test_run_match_benchmark_full_suite_mock():
    mock = _llm_mock_match_benchmark()
    cases = load_match_benchmark_cases(_bench_path())
    summary = run_match_benchmark(cases=cases, llm_client=mock, max_workers=1)
    assert summary.total == 10
    assert summary.matched == 10
    assert mock.chat.call_count == 3
