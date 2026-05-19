"""Smoke fixtures used by nre-verify-check (same layout as order_smoke for verify-order)."""

import json
from pathlib import Path

import pytest

from nre.benchmark.check_harness import evaluate_check_case


def _smoke_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures" / "check_smoke"


@pytest.mark.parametrize(
    "name",
    ["ready", "absent", "blocked", "remediation"],
)
def test_check_smoke_fixtures_evaluate(name: str):
    path = _smoke_dir() / f"{name}.json"
    assert path.is_file(), f"missing {path}"
    raw = json.loads(path.read_text(encoding="utf-8"))
    case = {k: v for k, v in raw.items() if k != "description"}
    out = evaluate_check_case(case)
    exp = case["expected"]
    assert out["status"] == exp["status"]
    if "key" in exp:
        assert out.get("key") == exp["key"]
    if exp.get("remediation_task_id"):
        assert out.get("remediation_task") is not None
        assert out["remediation_task"].id == exp["remediation_task_id"]
