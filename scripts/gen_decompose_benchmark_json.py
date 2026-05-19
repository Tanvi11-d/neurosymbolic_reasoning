#!/usr/bin/env python3
"""Regenerate tests/fixtures/decompose_benchmark/cases.json (OpenRouter benchmark inputs only)."""

from __future__ import annotations

import json
from pathlib import Path


def _case(
    cid: str,
    title: str,
    iota: str,
    registry: dict[str, list[str]],
    order: list[str],
    gamma: dict | None = None,
) -> dict:
    tools_registry = {
        name: {"description": f"Tool {name}", "params": params}
        for name, params in registry.items()
    }
    return {
        "id": cid,
        "title": title,
        "iota": iota,
        "gamma": gamma or {},
        "tools_registry": tools_registry,
        "expected": {"ordered_tool_names": order, "task_count": len(order)},
    }


def build_cases() -> list[dict]:
    return [
        _case(
            "01_archive_mv",
            "Archive folder and move file",
            "Create directory 'archive' under workspace and move report.txt into it.",
            {
                "cd": ["folder"],
                "mkdir": ["dir_name"],
                "mv": ["source", "destination"],
            },
            ["cd", "mkdir", "mv"],
        ),
        _case(
            "02_grep_revenue",
            "Cd then grep",
            "Change to folder 'archive' and search report.txt for the word 'revenue'.",
            {"cd": ["folder"], "grep": ["file_name", "pattern"]},
            ["cd", "grep"],
        ),
        _case(
            "03_mkdir_only",
            "Single mkdir",
            "Create a top-level directory named 'logs'.",
            {"mkdir": ["dir_name"]},
            ["mkdir"],
        ),
        _case(
            "04_cd_cat",
            "Read file after cd",
            "Go to workspace and print config.yaml contents.",
            {"cd": ["folder"], "cat": ["file_path"]},
            ["cd", "cat"],
        ),
        _case(
            "05_cd_cp",
            "Copy file",
            "In workspace, copy notes.txt to notes.bak.",
            {"cd": ["folder"], "cp": ["src", "dest"]},
            ["cd", "cp"],
        ),
        _case(
            "06_cd_rm",
            "Remove file",
            "From workspace, delete tmp.dat.",
            {"cd": ["folder"], "rm": ["path"]},
            ["cd", "rm"],
        ),
        _case(
            "07_cd_touch",
            "Create empty file",
            "In workspace, create an empty file named done.flag.",
            {"cd": ["folder"], "touch": ["path"]},
            ["cd", "touch"],
        ),
        _case(
            "08_login_post",
            "Auth then post",
            "Log in to the app, then publish status 'deployed'.",
            {"login": ["user", "token"], "post_status": ["text"]},
            ["login", "post_status"],
        ),
        _case(
            "09_fetch_parse",
            "Fetch then parse",
            "Download URL https://example.com/data.json then parse JSON.",
            {"fetch_url": ["url"], "parse_json": ["blob"]},
            ["fetch_url", "parse_json"],
        ),
        _case(
            "10_save_sync",
            "Save then sync",
            "Save document id 7 then sync to cloud bucket 'backup'.",
            {"save_doc": ["doc_id"], "sync_cloud": ["bucket"]},
            ["save_doc", "sync_cloud"],
        ),
        _case(
            "11_validate_submit",
            "Validate then submit",
            "Validate order form f-12 then submit it.",
            {"validate_form": ["form_id"], "submit_form": ["form_id"]},
            ["validate_form", "submit_form"],
        ),
        _case(
            "12_build_search",
            "Index then search",
            "Build search index for corpus 'docs' then query 'deadline'.",
            {"build_index": ["corpus"], "search_index": ["query"]},
            ["build_index", "search_index"],
        ),
        _case(
            "13_compress_upload",
            "Compress then upload",
            "Compress folder release/ then upload archive to s3 key 'out.zip'.",
            {"compress": ["path"], "upload_s3": ["key"]},
            ["compress", "upload_s3"],
        ),
        _case(
            "14_cd_mkdir_cp",
            "Prepare dir and copy in",
            "Enter workspace, create 'staging', copy build.bin into staging.",
            {"cd": ["folder"], "mkdir": ["dir_name"], "cp": ["src", "dest"]},
            ["cd", "mkdir", "cp"],
        ),
        _case(
            "15_connect_query",
            "DB connect and query",
            "Connect to database 'analytics' and run SQL 'SELECT 1'.",
            {"connect_db": ["name"], "run_sql": ["sql"]},
            ["connect_db", "run_sql"],
        ),
        _case(
            "16_encrypt_send",
            "Encrypt then send",
            "Encrypt message 'hi' with key k9 then send to queue 'jobs'.",
            {"encrypt": ["message", "key"], "send_queue": ["queue_name"]},
            ["encrypt", "send_queue"],
        ),
        _case(
            "17_ticket_notify",
            "Ticket then notify",
            "Create ticket 'bug-1' then notify channel 'ops'.",
            {"create_ticket": ["title"], "notify_channel": ["name"]},
            ["create_ticket", "notify_channel"],
        ),
        _case(
            "18_reserve_confirm",
            "Reserve then confirm",
            "Reserve table T5 at 7pm then confirm reservation id r42.",
            {"reserve_table": ["table", "time"], "confirm_reservation": ["res_id"]},
            ["reserve_table", "confirm_reservation"],
        ),
    ]


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    out = root / "tests" / "fixtures" / "decompose_benchmark" / "cases.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "description": "DECOMPOSE benchmark: score OpenRouter decompositions vs expected tool order.",
        "cases": build_cases(),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(payload['cases'])} cases to {out}")


if __name__ == "__main__":
    main()
