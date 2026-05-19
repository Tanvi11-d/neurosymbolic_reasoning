"""Manual GET-ORDER (OrderUnitTasks) verification — OpenRouter only (no deterministic shortcut).

Requires ``OPENROUTER_API_KEY`` and ``nre`` importable.

Examples::

    nre-verify-order parallel
    nre-verify-order chain
    nre-verify-order --fixture path/to.json
    nre-verify-order benchmark
    nre-verify-order benchmark --jobs 8

Without install::

    python scripts/verify_order.py parallel
    python scripts/verify_order.py benchmark --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nre.benchmark.order_harness import (
    load_order_benchmark_cases,
    run_order_benchmark,
)
from nre.llm import OpenRouterClient, OpenRouterSettings
from nre.primitives.deterministic import UnitTask
from nre.primitives.hybrid import OrderUnitTasks


def _repo_root_candidates(here: Path) -> list[Path]:
    return [here.parents[3] / "tests" / "fixtures" / "order_smoke", Path.cwd() / "tests" / "fixtures" / "order_smoke"]


def _fixture_path(name: str) -> Path:
    stem = name if name.endswith(".json") else f"{name}.json"
    here = Path(__file__).resolve()
    for base in _repo_root_candidates(here):
        p = base / stem
        if p.is_file():
            return p
    raise SystemExit(
        f"Fixture not found for {name!r}. Tried order_smoke/{stem} under repo tests/fixtures."
    )


def _load_tasks_from_data(data: dict[str, Any]) -> tuple[list[UnitTask], str]:
    tasks_raw = data.get("tasks") or []
    summary = data.get("conversation_summary") or ""
    if not isinstance(summary, str):
        summary = str(summary)
    turns: list[UnitTask] = []
    for i, item in enumerate(tasks_raw):
        if not isinstance(item, dict):
            raise SystemExit(f"tasks[{i}] must be an object")
        tid = item.get("id")
        if not tid:
            raise SystemExit(f"tasks[{i}].id required")
        deps = item.get("depends_on", [])
        if isinstance(deps, tuple):
            deps_t = deps
        elif isinstance(deps, list):
            deps_t = tuple(deps)
        else:
            raise SystemExit(f"tasks[{i}].depends_on must be a list")
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


def _main_benchmark(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run GET-ORDER benchmark (20 cases, OpenRouter, parallel requests)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel OpenRouter calls (default: min(16, number of cases))",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id",
    )
    args = parser.parse_args(argv)

    cases = load_order_benchmark_cases()
    if not cases:
        print(
            "No cases found. Run: python scripts/gen_order_benchmark_json.py",
            file=sys.stderr,
        )
        sys.exit(2)

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)

    summary = run_order_benchmark(
        cases=cases,
        llm_client=client,
        max_workers=args.jobs,
    )

    print(
        f"=== GET-ORDER benchmark ({len(cases)} cases, OpenRouter, "
        f"max_workers={summary.workers_used}) ===\n",
    )
    for r in summary.results:
        if r.error:
            line = f"FAIL {r.case_id}  error={r.error}"
        elif not r.valid_extension:
            line = f"FAIL {r.case_id}  invalid order={r.ordered!r}"
        elif r.matched_expected is True:
            line = f"OK   {r.case_id}  ordered={' -> '.join(r.ordered or [])}  (matched baseline)"
        elif r.matched_expected is False:
            exp = " -> ".join(r.expected or [])
            act = " -> ".join(r.ordered or [])
            line = f"MIS  {r.case_id}  baseline=[{exp}]  actual=[{act}]  (still valid={r.valid_extension})"
        else:
            line = f"OK   {r.case_id}  ordered={' -> '.join(r.ordered or [])}"
        print(line)

    print(
        f"\n--- VALID {summary.valid_count}/{summary.total} "
        f"({summary.validity_pct:.1f}%) | MATCH baseline {summary.matched_expected_count}/"
        f"{summary.expected_total} ({summary.match_pct:.1f}%) ---",
    )
    if summary.valid_count < summary.total:
        sys.exit(1)


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and argv[0] == "benchmark":
        return _main_benchmark(argv[1:])

    parser = argparse.ArgumentParser(
        description="Run GET-ORDER (OrderUnitTasks) on a JSON task list via OpenRouter.",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="parallel",
        help="Fixture stem under tests/fixtures/order_smoke/ (default: parallel)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to JSON with tasks[] and optional conversation_summary",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id (default from OpenRouterSettings)",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Override conversation_summary from fixture",
    )
    args = parser.parse_args(argv)

    path = args.fixture if args.fixture is not None else _fixture_path(args.preset)
    raw = json.loads(path.read_text(encoding="utf-8"))
    tasks, summary = _load_tasks_from_data(raw)
    if args.summary is not None:
        summary = args.summary

    print(f"=== GET-ORDER fixture: {path.name} ===")
    if raw.get("description"):
        print("---", raw["description"])
    print("\n--- tasks ---")
    for t in tasks:
        print(f"  {t.id}  tool={t.tool_name!r}  deps={list(t.depends_on)}  {t.text!r}")

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)
    orderer = OrderUnitTasks(
        llm_client=client,
        conversation_summary=summary,
    )

    print("\n--- running OrderUnitTasks (OpenRouter sibling batches) ---\n")
    out = orderer(tasks)
    print(json.dumps(
        {
            "ordered": out["ordered"],
            "has_cycle": out["has_cycle"],
            "used_llm_sibling_order": out.get("used_llm_sibling_order"),
            "cycles": out.get("cycles"),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
