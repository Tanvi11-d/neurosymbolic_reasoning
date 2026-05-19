"""Manual MATCH-FUNCS-AND-PARAMS verification — smoke fixtures + benchmark.

- **Smoke presets** under ``tests/fixtures/match_smoke/``: deterministic demo registry (no API key).
- **benchmark**: 10 cases; **m09** and **m10** use OpenRouter for tool choice / param fill (requires key).

Examples::

    nre-verify-match scratchpad
    nre-verify-match tau_wins
    nre-verify-match --fixture path/to/case.json
    nre-verify-match benchmark
    nre-verify-match benchmark --jobs 4

Without install::

    python scripts/verify_match.py scratchpad
    python scripts/verify_match.py benchmark
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nre.benchmark.match_harness import (
    evaluate_match_case,
    load_match_benchmark_cases,
    run_match_benchmark,
)
from nre.llm import OpenRouterClient, OpenRouterSettings


def _repo_root_candidates(here: Path) -> list[Path]:
    return [
        here.parents[3] / "tests" / "fixtures" / "match_smoke",
        Path.cwd() / "tests" / "fixtures" / "match_smoke",
    ]


def _fixture_path(name: str) -> Path:
    stem = name if name.endswith(".json") else f"{name}.json"
    here = Path(__file__).resolve()
    for base in _repo_root_candidates(here):
        p = base / stem
        if p.is_file():
            return p
    raise SystemExit(
        f"Fixture not found for {name!r}. Tried match_smoke/{stem} under repo tests/fixtures.",
    )


def _load_case(raw: dict[str, Any]) -> dict[str, Any]:
    skip = {"description"}
    return {k: v for k, v in raw.items() if k not in skip}


def _main_benchmark(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run MATCH benchmark (10 cases): OpenRouter for m09/m10, parallel workers optional"
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
        help="Path to cases.json",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id",
    )
    args = parser.parse_args(argv)

    cases = (
        load_match_benchmark_cases(args.fixture)
        if args.fixture is not None
        else load_match_benchmark_cases()
    )
    if not cases:
        print(
            "No cases found. Run: python scripts/gen_match_benchmark_json.py",
            file=sys.stderr,
        )
        sys.exit(2)

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)

    summary = run_match_benchmark(
        cases=cases,
        llm_client=client,
        max_workers=args.jobs,
    )

    print(
        f"=== MATCH benchmark ({len(cases)} cases, OpenRouter for LLM rows, "
        f"max_workers={summary.workers_used}) ===\n",
    )
    for r in summary.results:
        if r.ok:
            print(f"OK   {r.case_id}")
        else:
            print(f"FAIL {r.case_id}  detail={r.detail}")

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
        description="Run MatchFuncsAndParams on a JSON case (demo tool registry).",
    )
    parser.add_argument(
        "preset",
        nargs="?",
        default="scratchpad",
        help="Fixture stem under tests/fixtures/match_smoke/ (default: scratchpad)",
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to one MATCH case JSON",
    )
    args = parser.parse_args(argv)

    path = args.fixture if args.fixture is not None else _fixture_path(args.preset)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        print("Fixture must be a single JSON object", file=sys.stderr)
        sys.exit(2)

    case = _load_case(raw)

    print(f"=== MATCH fixture: {path.name} ===")
    if raw.get("description"):
        print("---", raw["description"])

    task = case.get("task") or {}
    print("\n--- task ---")
    print(
        f"  {task.get('id', '?')}  tool={task.get('tool_name', '')!r}  "
        f"deps={task.get('depends_on', [])}  params={task.get('parameters', {})!r}",
    )
    print("\n--- scratchpad ---")
    for k, v in sorted((case.get("scratchpad") or {}).items()):
        print(f"  {k!r}: {v!r}")
    print("\n--- turn_context ---")
    for k, v in sorted((case.get("turn_context") or {}).items()):
        print(f"  {k!r}: {v!r}")
    print("\n--- demo tools ---")
    for t in case.get("tools") or []:
        print(f"  {t!r}")

    print("\n--- running MatchFuncsAndParams ---\n")
    out = evaluate_match_case(case)
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
