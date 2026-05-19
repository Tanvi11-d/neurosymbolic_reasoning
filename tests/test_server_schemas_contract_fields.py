"""P1 wiring — ``openai_tools_to_nre_registry`` surfaces the generic contract
fields (``requires``, ``produces``, ``param_types``, ``defaults``) from the
OpenAI function JSON so CheckPrerequisites and MATCH can use them.

We accept two locations (both common in OpenAPI extensions):
1. Plain top-level fields on the ``function`` dict (concise).
2. ``x-corethink`` vendor extension nested under ``function``.

Either shape is tolerated. Unknown / absent fields are simply not emitted
(backward compat with existing OpenAI tool JSON).
"""

from __future__ import annotations

from nre.server.schemas import openai_tools_to_nre_registry


def _tool(fn_body):
    return [{"type": "function", "function": fn_body}]


def test_requires_surfaced_from_top_level():
    body = {
        "name": "estimate_distance",
        "description": "Distance between two cities",
        "parameters": {"type": "object", "properties": {"cityA": {}, "cityB": {}}},
        "requires": ["zipcode_A", "zipcode_B"],
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["estimate_distance"]["requires"] == ["zipcode_A", "zipcode_B"]


def test_produces_surfaced_from_top_level():
    body = {
        "name": "get_zipcode_based_on_city",
        "parameters": {"type": "object", "properties": {"city": {}}},
        "produces": ["zipcode"],
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["get_zipcode_based_on_city"]["produces"] == ["zipcode"]


def test_param_types_surfaced_from_top_level():
    body = {
        "name": "estimate_distance",
        "parameters": {"type": "object", "properties": {"cityA": {}, "cityB": {}}},
        "param_types": {"cityA": "zipcode", "cityB": "zipcode"},
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["estimate_distance"]["param_types"] == {
        "cityA": "zipcode",
        "cityB": "zipcode",
    }


def test_defaults_surfaced_from_top_level():
    body = {
        "name": "post_tweet",
        "parameters": {"type": "object", "properties": {"content": {}, "mentions": {}}},
        "defaults": {"mentions": []},
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["post_tweet"]["defaults"] == {"mentions": []}


def test_fields_absent_when_not_declared():
    """Backward compat: OpenAI tool JSON without these fields must not emit them."""
    body = {
        "name": "pwd",
        "parameters": {"type": "object", "properties": {}},
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    entry = reg["pwd"]
    for key in ("requires", "produces", "param_types", "defaults"):
        assert key not in entry, f"unexpected {key!r} in {entry}"


def test_vendor_extension_x_corethink_fields_surface():
    """OpenAI tool JSON using ``x-corethink`` extension block."""
    body = {
        "name": "post_tweet",
        "parameters": {"type": "object", "properties": {"content": {}, "mentions": {}}},
        "x-corethink": {
            "requires": ["authenticated"],
            "produces": ["tweet_id"],
            "param_types": {"content": "non_empty_str"},
            "defaults": {"mentions": []},
        },
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    entry = reg["post_tweet"]
    assert entry["requires"] == ["authenticated"]
    assert entry["produces"] == ["tweet_id"]
    assert entry["param_types"] == {"content": "non_empty_str"}
    assert entry["defaults"] == {"mentions": []}


def test_top_level_wins_over_vendor_extension():
    """If both locations set the same field, top-level wins (explicit over extension)."""
    body = {
        "name": "tool",
        "parameters": {"type": "object", "properties": {"x": {}}},
        "requires": ["top"],
        "x-corethink": {"requires": ["ext"]},
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["tool"]["requires"] == ["top"]


def test_enum_metadata_still_works():
    """Regression: existing enum surface must still be emitted."""
    body = {
        "name": "trade",
        "parameters": {
            "type": "object",
            "properties": {
                "order_type": {"enum": ["Buy", "Sell"]},
            },
        },
    }
    reg = openai_tools_to_nre_registry(_tool(body))
    assert reg["trade"]["param_enums"] == {"order_type": ["Buy", "Sell"]}
