"""CHECK benchmark: load JSON cases, build :class:`CheckPrerequisites`, evaluate (optional parallel)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import CheckPrerequisites


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def check_benchmark_cases_paths() -> list[Path]:
    root = _repo_root_from_here() / "tests" / "fixtures" / "check_benchmark"
    if not root.is_dir():
        return []
    paths = sorted(root.glob("case_*.json"))
    if paths:
        return paths
    p = root / "cases.json"
    return [p] if p.is_file() else []


def load_check_benchmark_cases(path: Path | None = None) -> list[dict[str, Any]]:
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

    paths = check_benchmark_cases_paths()
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
        cases.extend(load_check_benchmark_cases(p))
    return cases


def unit_task_from_dict(d: dict[str, Any]) -> UnitTask:
    deps = d.get("depends_on", [])
    if isinstance(deps, tuple):
        dt = deps
    else:
        dt = tuple(deps or [])
    return UnitTask(
        id=str(d["id"]),
        text=str(d.get("text", "") or ""),
        depends_on=dt,
        tool_name=str(d.get("tool_name", "") or ""),
        parameters=dict(d.get("parameters") or {}),
    )


def build_check_prerequisites_from_case(
    case: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> CheckPrerequisites:
    """Build ``CheckPrerequisites`` from a benchmark case dict.

    ``mode``:
    - ``non_null`` — default ``satisfied`` (value is not None).
    - ``equals`` — requires ``equals`` map: ``scratchpad[key] == equals[key]`` for Met.
    - ``classify`` — uses ``classify_outcomes`` map: key -> met|absent|not_met.
    - ``remediation`` — adds ``remediation_for`` from ``remediation`` block.
    """
    keys: list[str] = list(case.get("prereq_keys") or [])
    mode = case.get("mode") or "non_null"
    trace = bool(case.get("check_trace", False))
    tools_for_llm = dict(case.get("tools_registry_for_llm") or {})

    cfg: dict[str, Any] = {
        "get_prereqs": lambda _t, ks=keys: list(ks),
        "check_trace": trace,
    }

    if mode == "non_null":
        cfg["satisfied"] = lambda _k, v, _s: v is not None
    elif mode == "equals":
        equals_map: dict[str, Any] = dict(case.get("equals") or {})

        def satisfied(k: str, v: Any, _s: dict[str, Any]) -> bool:
            if k not in equals_map:
                return v is not None
            return v == equals_map[k]

        cfg["satisfied"] = satisfied
    elif mode == "classify":
        outcomes: dict[str, str] = dict(case.get("classify_outcomes") or {})

        def classify_prereq(k: str, _task: UnitTask, _s: dict[str, Any]) -> str:
            return str(outcomes[k])

        cfg["classify_prereq"] = classify_prereq
    elif mode == "remediation":
        cfg["satisfied"] = lambda _k, v, _s: v is not None
        rem_spec = case.get("remediation") or {}
        fix = unit_task_from_dict(rem_spec["task"])

        def remediation_for(
            key: str,
            _task: UnitTask,
            _s: dict[str, Any],
            outcome: str,
        ) -> UnitTask | None:
            if (
                key == rem_spec.get("when_key")
                and outcome == rem_spec.get("when_outcome", "absent")
            ):
                return fix
            return None

        # Symbolic hook only; LLM fallback still runs when hook returns None and
        # ``llm_client`` + ``tools_registry_for_llm`` are set (see c17 + future cases).
        cfg["remediation_for"] = remediation_for
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    if llm_client is not None:
        cfg["llm_client"] = llm_client
    if tools_for_llm:
        cfg["tools_registry_for_llm"] = tools_for_llm

    return CheckPrerequisites(**cfg)


def evaluate_check_case(
    case: dict[str, Any],
    *,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    task = unit_task_from_dict(case.get("task") or {"id": "t0"})
    scratchpad = dict(case.get("scratchpad") or {})
    cp = build_check_prerequisites_from_case(case, llm_client=llm_client)
    return cp.forward(task, scratchpad)


@dataclass
class CheckBenchmarkRunResult:
    case_id: str
    ok: bool
    expected_status: str
    actual_status: str
    detail: str | None = None


@dataclass
class CheckBenchmarkSummary:
    results: list[CheckBenchmarkRunResult] = field(default_factory=list)
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


def run_check_benchmark(
    *,
    cases: list[dict[str, Any]] | None = None,
    llm_client: Any | None = None,
    max_workers: int | None = None,
) -> CheckBenchmarkSummary:
    data = cases if cases is not None else load_check_benchmark_cases()
    n = len(data)
    summary = CheckBenchmarkSummary()
    if n == 0:
        return summary

    if max_workers is None or max_workers <= 0:
        workers = min(16, n)
    else:
        workers = min(max_workers, n)

    def _one(c: dict[str, Any]) -> CheckBenchmarkRunResult:
        cid = str(c.get("id", "unknown"))
        exp = c.get("expected") or {}
        exp_status = str(exp.get("status", ""))
        try:
            out = evaluate_check_case(c, llm_client=llm_client)
            act = str(out.get("status", ""))
            ok = act == exp_status
            if ok and "key" in exp:
                ok = out.get("key") == exp["key"]
            if ok and "task_id" in exp:
                ok = out.get("task_id") == exp["task_id"]
            if ok and exp.get("remediation_task_id"):
                rt = out.get("remediation_task")
                ok = rt is not None and getattr(rt, "id", None) == exp["remediation_task_id"]
            if ok and exp.get("remediation_non_null"):
                ok = out.get("remediation_task") is not None
            if ok and exp.get("remediation_tool_in"):
                rt = out.get("remediation_task")
                tools = exp["remediation_tool_in"]
                ok = (
                    rt is not None
                    and getattr(rt, "tool_name", None) in tools
                )
            detail = None
            if not ok:
                detail = json.dumps(out, default=str)[:500]
            return CheckBenchmarkRunResult(
                case_id=cid,
                ok=ok,
                expected_status=exp_status,
                actual_status=act,
                detail=detail,
            )
        except Exception as e:
            return CheckBenchmarkRunResult(
                case_id=cid,
                ok=False,
                expected_status=exp_status,
                actual_status="error",
                detail=str(e),
            )

    if workers <= 1:
        results = [_one(c) for c in data]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_one, data))

    summary.workers_used = workers
    summary.results.extend(results)
    return summary
