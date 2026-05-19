"""Manual LLM DECOMPOSE verification against BFCL-style JSON fixtures.

``python -m nre.cli.verify_decompose`` only works if ``nre`` is importable (e.g. after
``pip install -e .`` / ``uv sync``, or with ``PYTHONPATH=src``).

From a fresh clone without installing the package (from repo root, with
``OPENROUTER_API_KEY`` set)::

    python scripts/verify_decompose.py turn1
    python scripts/verify_decompose.py turn2

With ``src`` on ``PYTHONPATH``::

    PYTHONPATH=src python -m nre.cli.verify_decompose turn1

After editable install, the ``nre-verify-decompose`` console script is also available.

Benchmark (OpenRouter only; needs ``OPENROUTER_API_KEY``; parallel by default; ``-j 1`` for sequential)::

    python scripts/verify_decompose.py benchmark
    nre-verify-decompose benchmark --jobs 8

Prints ι, γ summary, tool list, then the normalized decompose result (turns, config, tools).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from nre.benchmark.decompose_harness import (
    load_benchmark_cases,
    run_benchmark,
    run_decompose_for_case,
)
from nre.llm import OpenRouterClient, OpenRouterSettings
from nre.primitives.hybrid import DecomposeInput


def _fixture_path(name: str) -> Path:
    here = Path(__file__).resolve()
    rel = Path("tests") / "fixtures" / "bfcl_multi_turn_base_0" / f"{name}.json"
    candidates: list[Path] = []
    seen_resolved: set[Path] = set()
    for base in (here.parents[3], Path.cwd()):
        p = (base / rel).resolve()
        if p not in seen_resolved:
            seen_resolved.add(p)
            candidates.append(p)
        if p.is_file():
            return p
    raise SystemExit(
        "Fixture not found. Tried:\n  " + "\n  ".join(str(p) for p in candidates)
    )


def _load_fixture(name: str) -> dict[str, Any]:
    path = _fixture_path(name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _main_benchmark(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Run DECOMPOSE benchmark over tests/fixtures/decompose_benchmark/cases.json (OpenRouter)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id (default from OpenRouterSettings)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print full JSON for each case",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        metavar="N",
        help="Parallel OpenRouter requests (default: min(16, number of cases); 1 = sequential)",
    )
    args = parser.parse_args(argv)

    cases = load_benchmark_cases()
    if not cases:
        print(
            "No benchmark cases found. Expected tests/fixtures/decompose_benchmark/cases.json",
            file=sys.stderr,
        )
        sys.exit(2)

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)

    summary = run_benchmark(cases=cases, llm_client=client, max_workers=args.jobs)

    print(
        f"=== DECOMPOSE benchmark ({len(cases)} cases, OpenRouter, "
        f"max_workers={summary.workers_used}) ===\n",
    )
    for r in summary.results:
        if r.error:
            line = f"FAIL {r.case_id}  error={r.error}"
        elif r.ok:
            line = f"OK   {r.case_id}  tools={' -> '.join(r.actual or [])}"
        else:
            exp = " -> ".join(r.expected)
            act = " -> ".join(r.actual or []) if r.actual is not None else "None"
            line = f"FAIL {r.case_id}  expected=[{exp}] actual=[{act}]"
        print(line)
        if args.verbose and not r.error:
            c = next((x for x in cases if str(x.get("id")) == r.case_id), None)
            if c:
                out = run_decompose_for_case(c, llm_client=client)
                print(json.dumps(
                    {
                        "turns": [
                            {
                                "id": t.id,
                                "tool_name": t.tool_name,
                                "depends_on": list(t.depends_on),
                                "parameters": dict(t.parameters),
                            }
                            for t in out["turns"]
                        ]
                    },
                    indent=2,
                ))

    pct = summary.accuracy_pct
    print(
        f"\n--- TOTAL {summary.matched}/{summary.total} matched ({pct:.1f}%) [OpenRouter] ---",
    )
    if summary.matched < summary.total:
        sys.exit(1)


def main() -> None:
    argv = list(sys.argv[1:])
    if argv and argv[0] == "benchmark":
        return _main_benchmark(argv[1:])

    parser = argparse.ArgumentParser(description="Run LLM DECOMPOSE on a BFCL fixture.")
    parser.add_argument(
        "fixture",
        nargs="?",
        default="turn1",
        help="Fixture stem: turn1 or turn2 (tests/fixtures/bfcl_multi_turn_base_0/<stem>.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override OpenRouter model id (default from OpenRouterSettings)",
    )
    args = parser.parse_args(argv)

    data = _load_fixture(args.fixture)
    iota = data["iota"]
    gamma = data.get("gamma")
    if gamma is None:
        gamma = data.get("gamma_minimal") or {}
    tools = data["tools_registry"]

    settings = OpenRouterSettings()
    if args.model:
        settings.model = args.model
    client = OpenRouterClient(settings)
    prim = DecomposeInput(llm_client=client)

    print("=== Fixture:", data.get("scenario"), "turn", data.get("turn"), "===")
    print("\n--- iota ---\n", iota)
    print("\n--- gamma keys ---\n", list(gamma.keys()))
    print("\n--- tools ---\n", sorted(tools.keys()))
    if "expected_from_md" in data:
        print("\n--- expected_from_md (reference only) ---")
        print(json.dumps(data["expected_from_md"], indent=2))

    print("\n--- calling OpenRouter DECOMPOSE ---\n")
    out = prim(iota, gamma, tools)
    print(json.dumps(
        {
            "turns": [
                {
                    "id": t.id,
                    "text": t.text,
                    "depends_on": list(t.depends_on),
                    "tool_name": t.tool_name,
                    "parameters": dict(t.parameters),
                }
                for t in out["turns"]
            ],
            "config": out["config"],
            "tools": out["tools"],
            "reasoning": out.get("reasoning", ""),
        },
        indent=2,
        default=str,
    ))


if __name__ == "__main__":
    main()
