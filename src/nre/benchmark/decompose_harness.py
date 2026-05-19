"""DECOMPOSE benchmark: load cases, score ordered tool names vs expected (OpenRouter)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nre.llm import OpenRouterClient, OpenRouterSettings
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import DecomposeInput, OrderUnitTasks


def _repo_root_from_here() -> Path:
    # src/nre/benchmark/decompose_harness.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def default_benchmark_json_path() -> Path:
    return _repo_root_from_here() / "tests" / "fixtures" / "decompose_benchmark" / "cases.json"


def benchmark_cases_paths() -> list[Path]:
    root = _repo_root_from_here() / "tests" / "fixtures" / "decompose_benchmark"
    if not root.is_dir():
        return []
    paths = sorted(root.glob("case_*.json"))
    if paths:
        return paths
    p = root / "cases.json"
    return [p] if p.is_file() else []


def load_benchmark_cases(path: Path | None = None) -> list[dict[str, Any]]:
    """Load benchmark cases from ``cases.json`` or ``case_*.json`` shards."""
    if path is not None:
        if path.is_dir():
            files = sorted(path.glob("case_*.json"))
            if not files and (path / "cases.json").is_file():
                files = [path / "cases.json"]
        elif path.is_file():
            files = [path]
        else:
            files = []
        cases: list[dict[str, Any]] = []
        for f in files:
            data = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(data, list):
                cases.extend(data)
            elif isinstance(data, dict) and "cases" in data:
                cases.extend(data["cases"])
            elif isinstance(data, dict):
                cases.append(data)
        return cases

    paths = benchmark_cases_paths()
    if not paths:
        return []
    if len(paths) == 1 and paths[0].name == "cases.json":
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "cases" in data:
            return list(data["cases"])
        return [data]
    cases = []
    for p in paths:
        cases.extend(load_benchmark_cases(p))
    return cases


def ordered_tool_names(turns: list[UnitTask]) -> list[str] | None:
    """Topological order of tasks; return ``None`` if cycle or empty."""
    if not turns:
        return []
    order = OrderUnitTasks().forward(turns)
    if order["has_cycle"]:
        return None
    by_id = {t.id: t for t in turns}
    return [by_id[tid].tool_name for tid in order["ordered"]]


def score_case(
    turns: list[UnitTask],
    expected_tool_names: list[str],
) -> tuple[bool, list[str], list[str] | None]:
    """Return (ok, expected, actual) where actual is None on cycle."""
    actual = ordered_tool_names(turns)
    if actual is None:
        return False, expected_tool_names, None
    ok = actual == expected_tool_names
    return ok, expected_tool_names, actual


def run_decompose_for_case(
    case: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """Run DECOMPOSE for one case via OpenRouter (or the given client)."""
    iota = case["iota"]
    gamma = case.get("gamma") or {}
    tools = case["tools_registry"]
    client = llm_client if llm_client is not None else OpenRouterClient(OpenRouterSettings())
    prim = DecomposeInput(llm_client=client)
    return prim(iota, gamma, tools)


@dataclass
class BenchmarkRunResult:
    case_id: str
    ok: bool
    expected: list[str]
    actual: list[str] | None
    error: str | None = None


def evaluate_benchmark_case(
    case: dict[str, Any],
    llm_client: Any,
) -> BenchmarkRunResult:
    """Run DECOMPOSE for one case and score vs ``expected.ordered_tool_names``."""
    cid = str(case.get("id", "unknown"))
    exp = list(case.get("expected", {}).get("ordered_tool_names", []))
    try:
        out = run_decompose_for_case(case, llm_client=llm_client)
        turns = list(out["turns"])
        ok, _, actual = score_case(turns, exp)
        return BenchmarkRunResult(case_id=cid, ok=ok, expected=exp, actual=actual)
    except Exception as e:
        return BenchmarkRunResult(
            case_id=cid,
            ok=False,
            expected=exp,
            actual=None,
            error=str(e),
        )


@dataclass
class BenchmarkSummary:
    results: list[BenchmarkRunResult] = field(default_factory=list)
    workers_used: int = 1

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def matched(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def accuracy_pct(self) -> float:
        if not self.results:
            return 0.0
        return 100.0 * self.matched / self.total


def run_benchmark(
    *,
    cases: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    progress: Callable[[BenchmarkRunResult], None] | None = None,
    max_workers: int | None = None,
) -> BenchmarkSummary:
    data = cases if cases is not None else load_benchmark_cases()
    client = llm_client if llm_client is not None else OpenRouterClient(OpenRouterSettings())
    n = len(data)
    summary = BenchmarkSummary()
    if n == 0:
        return summary

    if max_workers is None or max_workers <= 0:
        workers = min(16, n)
    else:
        workers = min(max_workers, n)

    if workers <= 1:

        def _one(c: dict[str, Any]) -> BenchmarkRunResult:
            return evaluate_benchmark_case(c, client)

        results = [_one(c) for c in data]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(evaluate_benchmark_case, data, [client] * n))

    summary.workers_used = workers
    for row in results:
        summary.results.append(row)
        if progress is not None:
            progress(row)
    return summary
