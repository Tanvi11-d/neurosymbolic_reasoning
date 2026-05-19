"""Pydantic models for the NRE HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """OpenAI-style chat message. Extra keys used for tool loops must be listed here — otherwise
    Pydantic drops them and the base LLM receives an invalid conversation (provider 400)."""

    role: str
    content: str | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat + tools: run the kernel, then call the **base** LLM with the trace."""

    model: str = Field(
        "nre-kernel",
        description=(
            "Optional client hint (e.g. ``nre-kernel``); ignored for routing. "
            "Responses always use the CoreThink public model id."
        ),
    )
    base_model: str | None = Field(
        default=None,
        description="Override OpenRouter model id for the post-kernel completion (default: OpenRouterSettings.model).",
    )
    messages: list[ChatMessage] = Field(default_factory=list)
    tools: list[dict[str, Any]] | None = None
    turn_context_by_task_id: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-task τ map (task id → parameter hints) for MATCH.",
    )
    conversation_summary: str | None = Field(
        default=None,
        description="Optional summary fed into LLM-backed kernel primitives.",
    )
    gamma: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Environment configuration γ (spec Sub-process 1 ``Extract(γ)``). "
            "Deep-flattened into the scratchpad at INIT so MATCH's LOOKUP chain "
            "has a populated S. Caller-supplied structured state (e.g. BFCL "
            "``initial_config``)."
        ),
    )
    gamma_seeds: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Pre-flattened overlay of seed keys that must win over γ-derived "
            "values but lose to DECOMPOSE config. Rarely needed by callers."
        ),
    )
    temperature: float | None = None
    max_tokens: int | None = None
    include_kernel_trace_in_response: bool = Field(
        default=True,
        description=(
            "If true, add ``nre_kernel_trace`` and ``nre.kernel_trace`` (kernel JSON after "
            "``run_turn``)."
        ),
    )
    include_primitive_log_in_response: bool = Field(
        default=True,
        description=(
            "If true, add ``nre_primitive_log`` and ``nre.primitive_log`` (structured primitive log)."
        ),
    )


_CORETHINK_CONTRACT_FIELDS: tuple[str, ...] = (
    "requires",
    "produces",
    "param_types",
    "defaults",
)


def openai_tools_to_nre_registry(tools: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Map OpenAI ``tools`` JSON to flat NRE registry entries (no ``execute``).

    In addition to the standard OpenAI fields, the following CoreThink contract
    fields are surfaced onto the registry entry when present, so CHECK's
    registry-driven defaults (spec Sub-process 4) and MATCH's defaults
    (spec Sub-process 5 Stage 2) can consult them:

    - ``requires: list[str]``        — S-keys this tool needs before it runs.
    - ``produces: list[str]``        — S-keys this tool writes on success.
    - ``param_types: dict[str, str]`` — per-param value-type tag for Satisfied.
    - ``defaults: dict[str, Any]``    — optional-param defaults for MATCH.

    Fields may live at two locations on the OpenAI ``function`` dict:

    1. **Top-level** (concise): e.g. ``function.requires``. Preferred.
    2. **Vendor extension** ``function["x-corethink"]`` — nested dict.

    If the same field is present in both, top-level wins.
    """
    if not tools:
        return {}
    out: dict[str, Any] = {}
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        desc = fn.get("description") or f"Tool {name}"
        params_obj = fn.get("parameters") or {}
        props: dict[str, Any] = {}
        if isinstance(params_obj, dict):
            props = params_obj.get("properties") or {}
        if not isinstance(props, dict):
            props = {}
        param_names = list(props.keys())
        param_enums: dict[str, list[Any]] = {}
        # Per-parameter descriptions are carried through verbatim. DECOMPOSE
        # and other downstream consumers use them to infer what *kind* of
        # value a param wants (e.g. "The zipcode of the first city") and
        # cross-reference against other tools' response schemas.
        param_descriptions: dict[str, str] = {}
        for pname, prop in props.items():
            if not isinstance(prop, dict):
                continue
            ev = prop.get("enum")
            if isinstance(ev, list) and ev:
                param_enums[str(pname)] = list(ev)
            desc_prop = prop.get("description")
            if isinstance(desc_prop, str) and desc_prop.strip():
                param_descriptions[str(pname)] = desc_prop
        entry: dict[str, Any] = {"description": desc, "params": param_names}
        if param_enums:
            entry["param_enums"] = param_enums
        if param_descriptions:
            entry["param_descriptions"] = param_descriptions

        # Preserve the tool's response schema. ``response.properties`` is the
        # set of named output fields; downstream (and the LLM) can read this
        # to match a caller's "I need a zipcode" hint against a producer
        # that actually emits a ``zipcode`` field.
        resp = fn.get("response")
        if isinstance(resp, dict):
            resp_props = resp.get("properties")
            if isinstance(resp_props, dict) and resp_props:
                entry["response_properties"] = dict(resp_props)

        ext = fn.get("x-corethink") if isinstance(fn.get("x-corethink"), dict) else {}
        for field in _CORETHINK_CONTRACT_FIELDS:
            if field in fn:
                entry[field] = fn[field]
            elif field in ext:
                entry[field] = ext[field]

        out[str(name)] = entry
    return out


def openai_tool_summaries_for_classifier(
    tools: list[dict[str, Any]] | None,
) -> list[str]:
    """Compact tool lines for the HTTP route classifier (name + description)."""
    if not tools:
        return []
    lines: list[str] = []
    for item in tools:
        if not isinstance(item, dict) or item.get("type") != "function":
            continue
        fn = item.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        desc = (fn.get("description") or "").strip()
        lines.append(f"{name}: {desc}" if desc else str(name))
    return lines


def last_user_content(messages: list[ChatMessage]) -> str:
    for m in reversed(messages):
        if m.role == "user" and m.content:
            return m.content.strip()
    return ""


def chat_messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    return [m.model_dump(mode="python", exclude_none=True) for m in messages]


def messages_prior_to_last_user(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """All chat messages strictly before the last user message that supplies kernel ``iota``."""
    idx: int | None = None
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        if m.role == "user" and (m.content or "").strip():
            idx = i
            break
    if idx is None or idx == 0:
        return []
    return chat_messages_to_dicts(messages[:idx])
