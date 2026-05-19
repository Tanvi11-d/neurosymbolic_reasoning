"""Robustness of JSON extraction from mixed prose + JSON LLM output.

Provider models sometimes include preamble or trailer text alongside the
requested JSON object, even when ``response_format: json_object`` is set.
The kernel needs to extract the JSON object and fail only when there is
genuinely no JSON present.

These tests pin the behavior of the extraction helper used by
:meth:`nre.llm.openrouter.OpenRouterClient.chat` when a response_model is
requested.
"""

from __future__ import annotations

import pytest

from nre.llm.openrouter import _strip_json_fence


# ── Baseline cases (already worked) ──────────────────────────────────

def test_plain_json_passes_through():
    s = '{"a": 1}'
    assert _strip_json_fence(s) == '{"a": 1}'


def test_fenced_json_block_alone_unwraps():
    s = '```json\n{"a": 1}\n```'
    assert _strip_json_fence(s) == '{"a": 1}'


def test_plain_fence_without_language_tag():
    s = '```\n{"a": 1}\n```'
    assert _strip_json_fence(s) == '{"a": 1}'


# ── The real-world failure case ──────────────────────────────────────

def test_preamble_prose_followed_by_fenced_json_is_extracted():
    """Model said 'Looking at this request, I need to...' then gave JSON in
    a fenced block. Extractor must return just the JSON."""
    s = (
        "Looking at this request, I need to estimate distance.\n"
        "\n"
        "```json\n"
        '{"tasks": [{"id": "t1"}], "config_keys": {}}\n'
        "```\n"
        "\n"
        "Done."
    )
    out = _strip_json_fence(s)
    import json as _j
    parsed = _j.loads(out)
    assert parsed == {"tasks": [{"id": "t1"}], "config_keys": {}}


def test_preamble_prose_followed_by_bare_json_is_extracted():
    """Some providers emit prose then a bare JSON object with no fence."""
    s = (
        "Reasoning: I need to plan the tasks.\n"
        "\n"
        '{"tasks": [{"id": "t1"}]}\n'
    )
    out = _strip_json_fence(s)
    import json as _j
    parsed = _j.loads(out)
    assert parsed == {"tasks": [{"id": "t1"}]}


def test_trailing_prose_after_json_is_stripped():
    s = (
        '{"tasks": []}\n'
        "\n"
        "I hope that helps!"
    )
    out = _strip_json_fence(s)
    import json as _j
    parsed = _j.loads(out)
    assert parsed == {"tasks": []}


def test_multiple_fenced_blocks_picks_largest_valid_json():
    """If the model emits a short example block then the full answer, take
    the full answer (the largest parsing JSON object)."""
    s = (
        "Example output shape:\n"
        "```json\n"
        '{"tasks": []}\n'
        "```\n"
        "\n"
        "Actual answer:\n"
        "```json\n"
        '{"tasks": [{"id": "t1"}, {"id": "t2"}], "config_keys": {"k": "v"}}\n'
        "```"
    )
    out = _strip_json_fence(s)
    import json as _j
    parsed = _j.loads(out)
    assert parsed == {"tasks": [{"id": "t1"}, {"id": "t2"}], "config_keys": {"k": "v"}}


def test_nested_braces_inside_json_not_confused():
    s = (
        "Here is the plan:\n"
        '{"tasks": [{"id": "t1", "parameters": {"nested": {"a": 1}}}]}'
    )
    out = _strip_json_fence(s)
    import json as _j
    parsed = _j.loads(out)
    assert parsed["tasks"][0]["parameters"]["nested"]["a"] == 1


# ── Negative cases — must NOT fabricate JSON ─────────────────────────

def test_prose_with_no_json_returns_unchanged_for_downstream_error():
    """If the content has no JSON object at all, return it unchanged so
    downstream ``json.loads`` produces a clear parse error instead of a
    silent mis-parse."""
    s = "I cannot complete this task."
    out = _strip_json_fence(s)
    # Either unchanged, or empty — just must not be accidentally valid JSON.
    import json as _j
    with pytest.raises(_j.JSONDecodeError):
        _j.loads(out)


def test_empty_string_returns_empty():
    assert _strip_json_fence("") == ""
