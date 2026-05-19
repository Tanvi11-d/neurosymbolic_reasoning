"""CHECK-PREREQUISITES benchmark fixtures and harness."""

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nre.benchmark.check_harness import (
    evaluate_check_case,
    load_check_benchmark_cases,
    run_check_benchmark,
)
from nre.schemas import LLMCheckRemediationOutput, LLMTaskItem


def _cases_path() -> Path:
    return (
        Path(__file__).resolve().parent / "fixtures" / "check_benchmark" / "cases.json"
    )


def _client_for_case(case: dict) -> MagicMock | None:
    if case.get("remediation_kind") != "llm":
        return None
    client = MagicMock()
    tools = (case.get("expected") or {}).get("remediation_tool_in") or ["fetch"]
    tool = tools[0]
    client.chat.return_value = LLMCheckRemediationOutput(
        task=LLMTaskItem(
            id="mock_repair",
            description="mock remediation",
            tool_name=tool,
            depends_on=[],
            parameters={},
        ),
        reasoning="unit test",
    )
    return client


def _bench_llm_mock() -> MagicMock:
    """Return remediation tasks keyed by prerequisite from the prompt (c11/c12)."""

    client = MagicMock()

    def chat(messages: list, *, response_model=None, **kwargs):
        text = messages[-1]["content"]
        m = re.search(r"- key: `([^`]+)`", text)
        key = m.group(1) if m else ""
        if key == "need":
            tn = "fetch"
        elif key == "bad":
            tn = "set_value"
        else:
            tn = "fetch"
        return LLMCheckRemediationOutput(
            task=LLMTaskItem(
                id="bench_llm",
                description="mock",
                tool_name=tn,
                depends_on=[],
                parameters={},
            ),
            reasoning="",
        )

    client.chat.side_effect = chat
    return client


def test_check_benchmark_file_exists():
    assert _cases_path().is_file(), "run scripts/gen_check_benchmark_json.py"


def test_twenty_cases_with_expected():
    cases = load_check_benchmark_cases(_cases_path())
    assert len(cases) == 20
    for c in cases:
        assert "id" in c and "expected" in c
        assert "status" in c["expected"]


@pytest.mark.parametrize(
    "case",
    load_check_benchmark_cases(_cases_path()),
    ids=lambda c: c["id"],
)
def test_evaluate_case_matches_expected(case):
    client = _client_for_case(case)
    out = evaluate_check_case(case, llm_client=client)
    exp = case["expected"]
    assert out["status"] == exp["status"]
    if "key" in exp:
        assert out.get("key") == exp["key"]
    if "task_id" in exp:
        assert out.get("task_id") == exp["task_id"]
    if exp.get("remediation_task_id"):
        rt = out.get("remediation_task")
        assert rt is not None
        assert rt.id == exp["remediation_task_id"]
    if exp.get("remediation_non_null"):
        assert out.get("remediation_task") is not None
    if exp.get("remediation_tool_in"):
        rt = out.get("remediation_task")
        assert rt is not None
        assert rt.tool_name in exp["remediation_tool_in"]
    if client is not None:
        client.chat.assert_called_once()


@pytest.mark.parametrize(
    "case",
    [c for c in load_check_benchmark_cases(_cases_path()) if c.get("check_trace")],
    ids=lambda c: c["id"],
)
def test_check_trace_has_checks(case):
    out = evaluate_check_case(case)
    assert "checks" in out
    keys = case["prereq_keys"]
    exp_len = len(keys)
    assert len(out["checks"]) == exp_len
    for i, chk in enumerate(out["checks"]):
        assert chk["key"] == keys[i]
        assert chk["outcome"] in ("met", "absent", "not_met")


def test_run_check_benchmark_parallel():
    cases = load_check_benchmark_cases(_cases_path())[:8]
    mock = _bench_llm_mock()
    summary = run_check_benchmark(cases=cases, llm_client=mock, max_workers=4)
    assert summary.total == 8
    assert summary.matched == 8
    assert summary.workers_used == 4
    mock.chat.assert_not_called()


def test_run_check_benchmark_all_match_full_suite():
    cases = load_check_benchmark_cases(_cases_path())
    mock = _bench_llm_mock()
    summary = run_check_benchmark(cases=cases, llm_client=mock, max_workers=1)
    assert summary.total == 20
    assert summary.matched == 20
    failed = [r for r in summary.results if not r.ok]
    assert not failed, failed
    assert mock.chat.call_count == 2
