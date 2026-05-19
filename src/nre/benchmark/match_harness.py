"""MATCH-FUNCS-AND-PARAMS harness: demo tool registry, JSON cases, optional OpenRouter."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nre.benchmark.check_harness import unit_task_from_dict
from nre.primitives.hybrid import MatchFuncsAndParams


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def demo_tool_registry(tool_ids: list[str]) -> dict[str, Any]:
    """Built-in tools for smoke/benchmark JSON (real ``execute`` callables)."""
    full: dict[str, Any] = {
        "add": {
            "description": "Sum a and b",
            "params": ["a", "b"],
            "execute": lambda a, b: int(a) + int(b),
        },
        "mul": {
            "description": "Multiply a and b",
            "params": ["a", "b"],
            "execute": lambda a, b: int(a) * int(b),
        },
        "double": {
            "description": "Double x",
            "params": ["x"],
            "execute": lambda x: int(x) * 2,
        },
        "identity": {
            "description": "Return x unchanged",
            "params": ["x"],
            "execute": lambda x: x,
        },
        "noop": {
            "description": "No parameters",
            "params": [],
            "execute": lambda: 0,
        },
        "spec_only": {
            "description": "Metadata only (no execute)",
            "params": ["a"],
        },
        "pickme": {
            "description": "LLM should pick this tool when ambiguous",
            "params": [],
            "execute": lambda: 7,
        },
        "other": {
            "description": "Wrong tool for preference tests",
            "params": [],
            "execute": lambda: -1,
        },
    }
    out: dict[str, Any] = {}
    for n in tool_ids:
        if n not in full:
            raise KeyError(f"unknown demo tool {n!r}; known: {sorted(full)}")
        entry = full[n]
        out[n] = {k: v for k, v in entry.items()}
    return out


def match_benchmark_cases_paths() -> list[Path]:
    root = _repo_root_from_here() / "tests" / "fixtures" / "match_benchmark"
    if not root.is_dir():
        return []
    paths = sorted(root.glob("case_*.json"))
    if paths:
        return paths
    p = root / "cases.json"
    return [p] if p.is_file() else []


def load_match_benchmark_cases(path: Path | None = None) -> list[dict[str, Any]]:
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

    paths = match_benchmark_cases_paths()
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
        cases.extend(load_match_benchmark_cases(p))
    return cases


def build_match_funcs_from_case(
    case: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> MatchFuncsAndParams:
    """Instantiate :class:`MatchFuncsAndParams` from a JSON case dict."""
    mc = dict(case.get("match_config") or {})
    ret = case.get("match_func_returns")

    def match_func(_task: Any, tools: dict[str, Any]) -> str | None:
        if ret is None:
            return None
        if isinstance(ret, str) and ret in tools:
            return ret
        return None

    kwargs: dict[str, Any] = {
        "match_func": match_func,
        **mc,
    }
    if llm_client is not None:
        kwargs["llm_client"] = llm_client
    return MatchFuncsAndParams(**kwargs)


def evaluate_match_case(
    case: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    task = unit_task_from_dict(case["task"])
    scratchpad = dict(case.get("scratchpad") or {})
    turn_context = dict(case.get("turn_context") or {})
    tools = demo_tool_registry(list(case.get("tools") or []))
    m = build_match_funcs_from_case(case, llm_client=llm_client)
    return m.forward(task, scratchpad, turn_context, tools)


def _expected_ok(out: dict[str, Any], exp: dict[str, Any]) -> bool:
    if out.get("status") != exp.get("status"):
        return False
    if "function" in exp and out.get("function") != exp["function"]:
        return False
    if "result" in exp and out.get("result") != exp["result"]:
        return False
    if "bindings" in exp:
        ob = out.get("bindings") or {}
        for k, v in exp["bindings"].items():
            if ob.get(k) != v:
                return False
    return True


@dataclass
class MatchBenchmarkRunResult:
    case_id: str
    ok: bool
    detail: str | None = None


@dataclass
class MatchBenchmarkSummary:
    results: list[MatchBenchmarkRunResult] = field(default_factory=list)
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


def run_match_benchmark(
    *,
    cases: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    max_workers: int | None = None,
) -> MatchBenchmarkSummary:
    data = cases if cases is not None else load_match_benchmark_cases()
    n = len(data)
    summary = MatchBenchmarkSummary()
    if n == 0:
        return summary

    if max_workers is None or max_workers <= 0:
        workers = min(16, n)
    else:
        workers = min(max_workers, n)

    def _one(c: dict[str, Any]) -> MatchBenchmarkRunResult:
        cid = str(c.get("id", "unknown"))
        exp = c.get("expected") or {}
        try:
            out = evaluate_match_case(c, llm_client=llm_client)
            ok = _expected_ok(out, exp)
            detail = None if ok else json.dumps(out, default=str)[:500]
            return MatchBenchmarkRunResult(case_id=cid, ok=ok, detail=detail)
        except Exception as e:
            return MatchBenchmarkRunResult(case_id=cid, ok=False, detail=str(e))

    if workers <= 1:
        results = [_one(c) for c in data]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, data))

    summary.workers_used = workers
    summary.results.extend(results)
    return summary
