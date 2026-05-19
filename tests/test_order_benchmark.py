"""GET-ORDER benchmark fixtures and harness (mock OpenRouter returns sorted sibling batch)."""

import re
from pathlib import Path

import pytest

from nre.benchmark.order_harness import (
    is_valid_topological_order,
    load_order_benchmark_cases,
    run_order_benchmark,
    tasks_from_order_case,
)
from unittest.mock import MagicMock

from nre.primitives.hybrid import OrderUnitTasks


def _cases_path() -> Path:
    return (
        Path(__file__).resolve().parent
        / "fixtures"
        / "order_benchmark"
        / "cases.json"
    )


def test_order_benchmark_file_exists():
    assert _cases_path().is_file(), "run scripts/gen_order_benchmark_json.py"


def test_twenty_cases_with_expected():
    cases = load_order_benchmark_cases(_cases_path())
    assert len(cases) == 20
    for c in cases:
        assert "id" in c and "tasks" in c and "expected_ordered" in c


@pytest.mark.parametrize("case", load_order_benchmark_cases(_cases_path()), ids=lambda c: c["id"])
def test_deterministic_order_matches_baseline(case):
    tasks, _ = tasks_from_order_case(case)
    out = OrderUnitTasks().forward(tasks)
    assert out["has_cycle"] is False
    assert out["ordered"] == case["expected_ordered"]


@pytest.mark.parametrize("case", load_order_benchmark_cases(_cases_path()), ids=lambda c: c["id"])
def test_baseline_is_valid_extension(case):
    tasks, _ = tasks_from_order_case(case)
    assert is_valid_topological_order(case["expected_ordered"], tasks)


def _fake_order_chat(messages: list[dict], *, response_model=None, **kwargs):
    text = messages[-1]["content"]
    ids = re.findall(r"- \*\*([^*]+)\*\*", text)
    if not ids or response_model is None:
        raise RuntimeError("unexpected mock chat")
    return response_model(ordered_task_ids=sorted(ids))


def test_run_order_benchmark_parallel_mock():
    cases = load_order_benchmark_cases(_cases_path())[:5]
    client = MagicMock()
    client.chat.side_effect = _fake_order_chat
    summary = run_order_benchmark(cases=cases, llm_client=client, max_workers=4)
    assert summary.total == 5
    assert summary.valid_count == 5
    assert summary.matched_expected_count == 5
    assert summary.workers_used == 4
    assert client.chat.call_count >= 1
