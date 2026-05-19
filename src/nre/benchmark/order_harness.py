"""GET-ORDER benchmark: load task DAG fixtures, run OpenRouter sibling ordering in parallel."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from nre.llm import OpenRouterClient, OpenRouterSettings
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import OrderUnitTasks


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def order_benchmark_cases_paths() -> list[Path]:
    root = _repo_root_from_here() / "tests" / "fixtures" / "order_benchmark"
    if not root.is_dir():
        return []
    paths = sorted(root.glob("case_*.json"))
    if paths:
        return paths
    p = root / "cases.json"
    return [p] if p.is_file() else []


def load_order_benchmark_cases(path: Path | None = None) -> list[dict[str, Any]]:
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

    paths = order_benchmark_cases_paths()
    if not paths:
        return []
    if len(paths) == 1 and paths[0].name == "cases.json":
        data = json.loads(paths[0].read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "cases" in data:
            return list(data["cases"])
        return [data]
    cases: list[dict[str, Any]] = []
    for p in paths:
        cases.extend(load_order_benchmark_cases(p))
    return cases


def tasks_from_order_case(case: dict[str, Any]) -> tuple[list[UnitTask], str]:
    tasks_raw = case.get("tasks") or []
    summary = case.get("conversation_summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)
    turns: list[UnitTask] = []
    for i, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise ValueError(f"tasks[{i}] must be an object")
        tid = item.get("id")
        if not tid:
            raise ValueError(f"tasks[{i}].id required")
        deps = item.get("depends_on", [])
        if isinstance(deps, tuple):
            deps_t = deps
        elif isinstance(deps, list):
            deps_t = tuple(deps)
        else:
            raise ValueError(f"tasks[{i}].depends_on must be a list")
        turns.append(
            UnitTask(
                id=str(tid),
                text=str(item.get("text", "") or ""),
                depends_on=deps_t,
                tool_name=str(item.get("tool_name", "") or ""),
                parameters=dict(item.get("parameters") or {}),
            )
        )
    return turns, summary


def is_valid_topological_order(order: list[str], tasks: list[UnitTask]) -> bool:
    ids = {t.id for t in tasks}
    if set(order) != ids or len(order) != len(ids):
        return False
    pos = {tid: i for i, tid in enumerate(order)}
    for t in tasks:
        for dep in t.depends_on:
            if dep not in pos or pos[dep] >= pos[t.id]:
                return False
    return True


@dataclass
class OrderBenchmarkRunResult:
    case_id: str
    valid_extension: bool
    matched_expected: bool | None
    expected: list[str] | None
    ordered: list[str] | None
    used_llm_sibling_order: bool | None
    error: str | None = None


@dataclass
class OrderBenchmarkSummary:
    results: list[OrderBenchmarkRunResult] = field(default_factory=list)
    workers_used: int = 1

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def valid_count(self) -> int:
        return sum(1 for r in self.results if r.valid_extension)

    @property
    def matched_expected_count(self) -> int:
        return sum(1 for r in self.results if r.matched_expected is True)

    @property
    def expected_total(self) -> int:
        return sum(1 for r in self.results if r.expected is not None)

    @property
    def validity_pct(self) -> float:
        if not self.results:
            return 0.0
        return 100.0 * self.valid_count / self.total

    @property
    def match_pct(self) -> float:
        n = self.expected_total
        if n == 0:
            return 0.0
        return 100.0 * self.matched_expected_count / n


def evaluate_order_benchmark_case(
    case: dict[str, Any],
    llm_client: Any,
) -> OrderBenchmarkRunResult:
    cid = str(case.get("id", "unknown"))
    exp = case.get("expected_ordered")
    if exp is not None:
        exp = list(exp)
    try:
        tasks, summary = tasks_from_order_case(case)
        orderer = OrderUnitTasks(
            llm_client=llm_client,
            conversation_summary=summary,
        )
        out = orderer(tasks)
        if out["has_cycle"]:
            return OrderBenchmarkRunResult(
                case_id=cid,
                valid_extension=False,
                matched_expected=False if exp else None,
                expected=exp,
                ordered=out.get("ordered"),
                used_llm_sibling_order=out.get("used_llm_sibling_order"),
                error="cycle in graph",
            )
        ord_ids = list(out["ordered"])
        valid = is_valid_topological_order(ord_ids, tasks)
        matched: bool | None = None
        if exp is not None:
            matched = ord_ids == exp
        return OrderBenchmarkRunResult(
            case_id=cid,
            valid_extension=valid,
            matched_expected=matched,
            expected=exp,
            ordered=ord_ids,
            used_llm_sibling_order=out.get("used_llm_sibling_order"),
            error=None if valid else "invalid linear extension",
        )
    except Exception as e:
        return OrderBenchmarkRunResult(
            case_id=cid,
            valid_extension=False,
            matched_expected=False if exp else None,
            expected=exp,
            ordered=None,
            used_llm_sibling_order=None,
            error=str(e),
        )


def run_order_benchmark(
    *,
    cases: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    progress: Callable[[OrderBenchmarkRunResult], None] | None = None,
    max_workers: int | None = None,
) -> OrderBenchmarkSummary:
    data = cases if cases is not None else load_order_benchmark_cases()
    client = llm_client if llm_client is not None else OpenRouterClient(OpenRouterSettings())
    n = len(data)
    summary = OrderBenchmarkSummary()
    if n == 0:
        return summary

    if max_workers is None or max_workers <= 0:
        workers = min(16, n)
    else:
        workers = min(max_workers, n)

    def _one(c: dict[str, Any]) -> OrderBenchmarkRunResult:
        return evaluate_order_benchmark_case(c, client)

    if workers <= 1:
        results = [_one(c) for c in data]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, data))

    summary.workers_used = workers
    for row in results:
        summary.results.append(row)
        if progress is not None:
            progress(row)
    return summary
