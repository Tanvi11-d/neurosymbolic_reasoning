"""Manual CHECK-PREREQUISITES (CheckPrerequisites) verification — smoke + OpenRouter benchmark.

- **Smoke presets** (``ready``, ``absent``, …): deterministic; no API key.
- **benchmark** subcommand: runs the 20-case suite; cases **c11** and **c12** call OpenRouter
  for LLM remediation (requires ``OPENROUTER_API_KEY``). Other cases stay deterministic.

Examples::

    nre-verify-check ready
    nre-verify-check absent
    nre-verify-check --fixture path/to.json
    nre-verify-check benchmark
    nre-verify-check benchmark --jobs 8

Without install::

    python scripts/verify_check.py ready
    python scripts/verify_check.py benchmark --jobs 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nre.benchmark.check_harness import (
    evaluate_check_case,
    load_check_benchmark_cases,
    run_check_benchmark,
)
from nre.llm import OpenRouterClient, OpenRouterSettings


def _repo_root_candidates(here: Path) -> list[Path]:
    return [
        here.parents[3] / "tests" / "fixtures" / "check_smoke",
        Path.cwd() / "tests" / "fixtures" / "check_smoke",
    ]


def _fixture_path(name: str) -> Path:
    stem = name if name.endswith(".json") else f"{name}.json"
    here = Path(__file__).resolve()
    for base in _repo_root_candidates(here):
        p = base / stem
        if p.is_file():
            return p
    raise SystemExit(
        f"Fixture not found for {name!r}. Tried check_smoke/{stem} under repo tests/fixtures.",
    )


def _load_check_case(raw: dict[str, Any]) -> dict[str, Any]:
    """Return a case dict for the harness (strip smoke-only keys)."""
    skip = {"description"}
    return {k: v for k, v in raw.items() if k not in skip}


def _main_benchmark(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run CHECK benchmark (20 cases): OpenRouter remediation on LLM cases, "
            "parallel workers for case evaluation"
        ),
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel workers (default: min(16, number of cases))",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to cases.json or directory of case_*.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id (default from OpenRouterSettings)",
    )
    args = parser.parse_args(argv)

    cases = (
        load_check_benchmark_cases(args.fixture)
        if args.fixture is not None
        else load_check_benchmark_cases()
    )
    if not cases:
        print(
            "No cases found. Run: python scripts/gen_check_benchmark_json.py",
            file=sys.stderr,
        )
        sys.exit(2)

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)

    summary = run_check_benchmark(
        cases=cases,
        llm_client=client,
        max_workers=args.jobs,
    )

    print(
        f"=== CHECK benchmark ({len(cases)} cases, OpenRouter remediation where configured, "
        f"max_workers={summary.workers_used}) ===\n",
    )
    for r in summary.results:
        if r.ok:
            print(f"OK   {r.case_id}  expected={r.expected_status}")
        else:
            print(
                f"FAIL {r.case_id}  expected={r.expected_status}  "
                f"actual={r.actual_status}  detail={r.detail}",
            )

    print(
        f"\n--- MATCH {summary.matched}/{summary.total} "
        f"({summary.accuracy_pct:.1f}%) ---",
    )
    if summary.matched < summary.total:
        sys.exit(1)


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and argv[0] == "benchmark":
        return _main_benchmark(argv[1:])

    parser = argparse.ArgumentParser(
        description="Run CHECK-PREREQUISITES on a JSON case via CheckPrerequisites.",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="ready",
        help="Fixture stem under tests/fixtures/check_smoke/ (default: ready)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to one CHECK case JSON (same schema as benchmark cases)",
    )
    args = parser.parse_args(argv)

    path = args.fixture if args.fixture is not None else _fixture_path(args.preset)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("Fixture must be a single JSON object", file=sys.stderr)
        sys.exit(2)

    case = _load_check_case(raw)

    print(f"=== CHECK fixture: {path.name} ===")
    if raw.get("description"):
        print("---", raw["description"])

    task = case.get("task") or {}
    print("\n--- task ---")
    print(
        f"  {task.get('id', '?')}  tool={task.get('tool_name', '')!r}  "
        f"deps={task.get('depends_on', [])}  {task.get('text', '')!r}",
    )
    print("\n--- prereq_keys ---")
    for k in case.get("prereq_keys") or []:
        print(f"  {k!r}")
    print("\n--- scratchpad ---")
    sp = case.get("scratchpad") or {}
    for k, v in sorted(sp.items()):
        print(f"  {k!r}: {v!r}")
    print("\n--- mode / check_trace ---")
    print(f"  mode={case.get('mode', 'non_null')!r}  check_trace={case.get('check_trace', False)}")

    print("\n--- running CheckPrerequisites ---\n")
    out = evaluate_check_case(case)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
