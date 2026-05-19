#!/usr/bin/env python3
"""Regenerate tests/fixtures/order_benchmark/cases.json with 20 GET-ORDER scenarios."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root / src on path for generator
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from nre.primitives.deterministic import UnitTask # noqa: E402
from nre.primitives.hybrid import OrderUnitTasks  # noqa: E402


def _u(
    tid: str,
    text: str,
    *,
    deps: tuple[str, ...] = (),
    tool: str = "",
) -> dict:
    return {
        "id": tid,
        "text": text,
        "depends_on": list(deps),
        "tool_name": tool,
    }


def _expected(tasks_dicts: list[dict]) -> list[str]:
    turns = [
        UnitTask(
            id=d["id"],
            text=d.get("text", ""),
            depends_on=tuple(d.get("depends_on", [])),
            tool_name=d.get("tool_name", ""),
        )
        for d in tasks_dicts
    ]
    return list(OrderUnitTasks().forward(turns)["ordered"])


def _case(
    cid: str,
    description: str,
    summary: str,
    tasks: list[dict],
) -> dict:
    return {
        "id": cid,
        "description": description,
        "conversation_summary": summary,
        "tasks": tasks,
        "expected_ordered": _expected(tasks),
    }


def build_cases() -> list[dict]:
    return [
        _case(
            "01_chain_3",
            "Linear chain of three tools",
            "",
            [
                _u("a1", "Step one", tool="x"),
                _u("a2", "Step two", deps=("a1",), tool="y"),
                _u("a3", "Step three", deps=("a2",), tool="z"),
            ],
        ),
        _case(
            "02_parallel_pair",
            "Two independent tasks; context prefers listing before touch",
            "List directory contents before creating a marker file.",
            [
                _u("t_ls", "List workspace", tool="ls"),
                _u("t_touch", "Create flag file", tool="touch"),
            ],
        ),
        _case(
            "03_parallel_pair_swap_hint",
            "Context prefers touch before ls",
            "Create the lock file first, then list files to verify.",
            [
                _u("p1", "List files", tool="ls"),
                _u("p2", "Create lock", tool="touch"),
            ],
        ),
        _case(
            "04_fan_out",
            "One root, two children",
            "",
            [
                _u("r1", "Root", tool="a"),
                _u("c2", "Child B", deps=("r1",), tool="b"),
                _u("c1", "Child A", deps=("r1",), tool="c"),
            ],
        ),
        _case(
            "05_parallel_three",
            "Three independent API calls",
            "Refresh cache before fetching user then posting analytics.",
            [
                _u("z3", "Post analytics", tool="post"),
                _u("z1", "Refresh cache", tool="refresh"),
                _u("z2", "Fetch user", tool="get"),
            ],
        ),
        _case(
            "06_chain_4",
            "Four-step pipeline",
            "",
            [
                _u("s1", "Connect", tool="conn"),
                _u("s2", "Auth", deps=("s1",), tool="auth"),
                _u("s3", "Query", deps=("s2",), tool="q"),
                _u("s4", "Close", deps=("s3",), tool="close"),
            ],
        ),
        _case(
            "07_diamond",
            "Diamond: join after fork",
            "",
            [
                _u("d1", "Start", tool="a"),
                _u("d2", "Left", deps=("d1",), tool="b"),
                _u("d3", "Right", deps=("d1",), tool="c"),
                _u("d4", "Join", deps=("d2", "d3"), tool="d"),
            ],
        ),
        _case(
            "08_two_waves_parallel_second",
            "Second wave has two parallel tasks",
            "",
            [
                _u("w1", "Setup", tool="setup"),
                _u("w3", "Task B", deps=("w1",), tool="b"),
                _u("w2", "Task A", deps=("w1",), tool="a"),
            ],
        ),
        _case(
            "09_wide_parallel_four",
            "Four roots",
            "Encrypt, then compress, then upload, then notify — by urgency.",
            [
                _u("f2", "Compress", tool="zip"),
                _u("f4", "Notify", tool="mail"),
                _u("f1", "Encrypt", tool="enc"),
                _u("f3", "Upload", tool="s3"),
            ],
        ),
        _case(
            "10_chain_2",
            "Short chain",
            "",
            [
                _u("x1", "Open", tool="open"),
                _u("x2", "Read", deps=("x1",), tool="read"),
            ],
        ),
        _case(
            "11_merge_three_to_one",
            "Three parents to one sink",
            "",
            [
                _u("m1", "A", tool="a"),
                _u("m2", "B", tool="b"),
                _u("m3", "C", tool="c"),
                _u("m4", "Merge", deps=("m1", "m2", "m3"), tool="merge"),
            ],
        ),
        _case(
            "12_staggered",
            "Partial overlap deps",
            "",
            [
                _u("g1", "G1", tool="a"),
                _u("g2", "G2", deps=("g1",), tool="b"),
                _u("g3", "G3", deps=("g1",), tool="c"),
                _u("g4", "G4", deps=("g2", "g3"), tool="d"),
            ],
        ),
        _case(
            "13_parallel_with_summary",
            "NL prefers validate before submit",
            "Always validate the form before submitting to the server.",
            [
                _u("sub", "Submit form", tool="submit"),
                _u("val", "Validate form", tool="validate"),
            ],
        ),
        _case(
            "14_three_chain_parallel_branch",
            "",
            "",
            [
                _u("h1", "H1", tool="a"),
                _u("h2", "H2", deps=("h1",), tool="b"),
                _u("h4", "H4", deps=("h3",), tool="d"),
                _u("h3", "H3", deps=("h2",), tool="c"),
            ],
        ),
        _case(
            "15_dual_roots_join",
            "Two roots merging — add dummy deps as two chains into join",
            "",
            [
                _u("i1", "I1", tool="a"),
                _u("i2", "I2", tool="b"),
                _u("i3", "I3", deps=("i1",), tool="c"),
                _u("i4", "I4", deps=("i2",), tool="d"),
                _u("i5", "I5", deps=("i3", "i4"), tool="e"),
            ],
        ),
        _case(
            "16_parallel_binary_names",
            "Binary-style ids",
            "Run checksum before download.",
            [
                _u("b10", "Download", tool="get"),
                _u("b01", "Checksum", tool="hash"),
            ],
        ),
        _case(
            "17_wave_three_children",
            "",
            "",
            [
                _u("v0", "V0", tool="a"),
                _u("v2", "V2", deps=("v0",), tool="c"),
                _u("v1", "V1", deps=("v0",), tool="b"),
                _u("v3", "V3", deps=("v0",), tool="d"),
            ],
        ),
        _case(
            "18_short_parallel",
            "",
            "Backup then delete old files.",
            [
                _u("del", "Delete old", tool="rm"),
                _u("bak", "Backup", tool="cp"),
            ],
        ),
        _case(
            "19_linear_5",
            "Longer chain",
            "",
            [
                _u("n1", "N1", tool="a"),
                _u("n2", "N2", deps=("n1",), tool="b"),
                _u("n3", "N3", deps=("n2",), tool="c"),
                _u("n4", "N4", deps=("n3",), tool="d"),
                _u("n5", "N5", deps=("n4",), tool="e"),
            ],
        ),
        _case(
            "20_complex_fan",
            "Root, two parallel, one depends on both",
            "",
            [
                _u("q1", "Q1", tool="a"),
                _u("q2", "Q2", deps=("q1",), tool="b"),
                _u("q3", "Q3", deps=("q1",), tool="c"),
                _u("q4", "Q4", deps=("q2",), tool="d"),
                _u("q5", "Q5", deps=("q3", "q4"), tool="e"),
            ],
        ),
    ]


def main() -> None:
    out = _ROOT / "tests" / "fixtures" / "order_benchmark" / "cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cases = build_cases()
    payload = {
        "version": 1,
        "description": "GET-ORDER benchmark: expected_ordered = deterministic topo baseline; LLM may differ.",
        "cases": cases,
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {out}")


if __name__ == "__main__":
    main()
