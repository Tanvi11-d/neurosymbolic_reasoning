"""Registry pipeline preserves parameter descriptions and response schemas.

When the host sends OpenAI tool JSON, each parameter typically has a
``description`` and the function may have a ``response`` schema. Both are
semantic signals that DECOMPOSE needs to plan correctly (e.g. "this param
is a zipcode; another tool produces zipcode → plan a prerequisite").

The current pipeline collapses tool entries to just ``{description, params}``,
losing all per-parameter descriptions and the entire response schema. This
prevents DECOMPOSE from cross-referencing.

Tests below pin faithful preservation of this data — no keyword matching,
no tag canonicalization. Just pass through what the caller sent.
"""

from __future__ import annotations

from nre.server.schemas import openai_tools_to_nre_registry
from nre.utils.tool_registry import format_tool_schemas, tool_registry_to_schemas


def _fn(name, params=None, response=None, description=None):
    props = {}
    required = []
    for p, info in (params or {}).items():
        props[p] = info
        required.append(p)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"{name} description.",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
            **({"response": response} if response else {}),
        },
    }


# ── Registry: preserve param descriptions ───────────────────────────

def test_registry_preserves_param_descriptions():
    tools = [
        _fn(
            "estimate_distance",
            params={
                "cityA": {"type": "string", "description": "The zipcode of the first city."},
                "cityB": {"type": "string", "description": "The zipcode of the second city."},
            },
        ),
    ]
    reg = openai_tools_to_nre_registry(tools)
    pd = reg["estimate_distance"].get("param_descriptions") or {}
    assert pd.get("cityA") == "The zipcode of the first city."
    assert pd.get("cityB") == "The zipcode of the second city."


def test_registry_omits_param_descriptions_when_empty():
    """Tools with no parameter descriptions don't emit an empty dict."""
    tools = [_fn("noop", params={"x": {"type": "string"}})]
    reg = openai_tools_to_nre_registry(tools)
    assert "param_descriptions" not in reg["noop"]


def test_registry_partial_param_descriptions():
    """Only the params that have descriptions are recorded."""
    tools = [
        _fn(
            "mixed",
            params={
                "a": {"type": "string", "description": "first."},
                "b": {"type": "string"},
            },
        ),
    ]
    reg = openai_tools_to_nre_registry(tools)
    pd = reg["mixed"].get("param_descriptions") or {}
    assert pd == {"a": "first."}


# ── Registry: preserve response schema ──────────────────────────────

def test_registry_preserves_response_properties():
    tools = [
        _fn(
            "get_zipcode_based_on_city",
            params={"city": {"type": "string"}},
            response={"type": "dict", "properties": {"zipcode": {"type": "string"}}},
        ),
    ]
    reg = openai_tools_to_nre_registry(tools)
    rs = reg["get_zipcode_based_on_city"].get("response_properties") or {}
    assert "zipcode" in rs


def test_registry_response_properties_with_nested_description():
    """Full schema-per-field preservation — not just the field name set.

    Downstream consumers may want to read the field-level description, so we
    carry the sub-schema dict through verbatim.
    """
    tools = [
        _fn(
            "get_zipcode_based_on_city",
            params={"city": {"type": "string"}},
            response={
                "type": "dict",
                "properties": {
                    "zipcode": {
                        "type": "string",
                        "description": "The 5-digit postal code of the city.",
                    }
                },
            },
        ),
    ]
    reg = openai_tools_to_nre_registry(tools)
    rs = reg["get_zipcode_based_on_city"].get("response_properties") or {}
    assert rs["zipcode"]["description"] == "The 5-digit postal code of the city."


def test_registry_omits_response_properties_when_absent():
    tools = [_fn("noop", params={})]
    reg = openai_tools_to_nre_registry(tools)
    assert "response_properties" not in reg["noop"]


# ── format_tool_schemas surfaces both to the LLM ────────────────────

def test_format_schemas_renders_param_descriptions_inline():
    registry = {
        "estimate_distance": {
            "description": "Estimate distance.",
            "params": ["cityA", "cityB"],
            "param_descriptions": {
                "cityA": "The zipcode of the first city.",
                "cityB": "The zipcode of the second city.",
            },
        }
    }
    text = format_tool_schemas(tool_registry_to_schemas(registry))
    # Descriptions must appear in the rendered output — substring match is
    # fine; the exact formatting is free.
    assert "zipcode of the first city" in text
    assert "zipcode of the second city" in text


def test_format_schemas_renders_response_fields_inline():
    registry = {
        "get_zipcode_based_on_city": {
            "description": "Get zipcode.",
            "params": ["city"],
            "response_properties": {
                "zipcode": {"type": "string", "description": "Postal code."},
            },
        }
    }
    text = format_tool_schemas(tool_registry_to_schemas(registry))
    # The response field name ("zipcode") must appear so the LLM can
    # cross-reference against parameter descriptions in other tools.
    assert "zipcode" in text
    # Response section is distinct from parameters section.
    assert "→" in text or "returns" in text.lower() or "produces" in text.lower()


def test_format_schemas_back_compat_no_extras():
    """Registry entries without these new fields render as before — no stray
    ``→`` or empty ``produces:`` sections."""
    registry = {
        "noop": {"description": "noop.", "params": ["x"]},
    }
    text = format_tool_schemas(tool_registry_to_schemas(registry))
    # Simple signature + description, nothing else.
    assert text.count("\n") == 0, f"unexpected multi-line render: {text!r}"
    assert "→" not in text
    assert "returns" not in text.lower()


# ── End-to-end: what the LLM actually sees ──────────────────────────

def test_end_to_end_bfcl_style_zipcode_signal_reaches_user_prompt():
    tools = [
        _fn(
            "estimate_distance",
            params={
                "cityA": {"type": "string", "description": "The zipcode of the first city."},
                "cityB": {"type": "string", "description": "The zipcode of the second city."},
            },
            response={"type": "dict", "properties": {"distance": {"type": "number"}}},
        ),
        _fn(
            "get_zipcode_based_on_city",
            params={"city": {"type": "string", "description": "The name of the city."}},
            response={"type": "dict", "properties": {"zipcode": {"type": "string"}}},
        ),
    ]
    reg = openai_tools_to_nre_registry(tools)
    text = format_tool_schemas(tool_registry_to_schemas(reg))

    # Both the consumer's "zipcode"-typed parameter description and the
    # producer's "zipcode" response field must be in the rendered text.
    assert "zipcode of the first city" in text
    # "zipcode" as a response-field name must appear in the producer's block.
    assert "zipcode" in text
