from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError
from nre.llm import OpenRouterClient

_log = logging.getLogger("nre.route_classify")

# When something goes wrong, default to the more capable path so we never
# accidentally strip the kernel from a request that needed it.
_SAFE_DEFAULT_ROUTE: Literal["reasoner", "direct"] = "reasoner"


class ChatRouteDecision(BaseModel):
    """The label the classifier LLM is asked to produce."""

    route: Literal["reasoner", "direct"] = Field(
        ...,
        description="reasoner = run the NRE kernel; direct = single completion call",
    )


_CLASSIFIER_SYSTEM = """You are an API request router for a neurosymbolic tool-calling engine.

Decide whether this turn should go through the **reasoner** pipeline (DECOMPOSE → task DAG → prerequisites → bound tool plan) or take a **direct** completion path.

Pick **reasoner** when the user clearly needs an agentic, multi-step workflow: several ordered tool calls, dependencies between steps, prerequisite data gathering, filesystem or agent benchmarks, or explicit planning across tools.

Pick **direct** for everyday chat, single-shot Q&A, creative writing, opinion, or when tools are offered but the request isn't actually a structured multi-step mission (including simple one-off tool use that doesn't need a planner).

Reply with JSON only, matching the schema: {"route":"reasoner"} or {"route":"direct"}."""

def _build_classifier_prompt(
    iota: str,
    tool_summaries: list[str],
    conversation_summary: str | None,
) -> str:
    """Compose the user-side message for the classifier from the per-turn inputs."""
    tools_block = (
        "\n".join(f"- {line}" for line in tool_summaries)
        if tool_summaries
        else "(no tools)"
    )

    sections: list[str] = []
    summary = (conversation_summary or "").strip()
    if summary:
        sections.append(f"Conversation summary:\n{summary}")
    sections.append("Registered tools and descriptions:\n" + tools_block)
    sections.append(f"Latest user message:\n{iota.strip()}")
    return "\n\n".join(sections)

def _safe_default(reason: str, exc: Exception | None = None) -> ChatRouteDecision:
    """Log why we're bailing on the classifier and return the safe default route."""
    if exc is not None:
        _log.warning(
            "classifier fell back to route=%s — %s (%s: %s)",
            _SAFE_DEFAULT_ROUTE,
            reason,
            type(exc).__name__,
            exc,
        )
    else:
        _log.warning(
            "classifier fell back to route=%s — %s",
            _SAFE_DEFAULT_ROUTE,
            reason,
        )
    return ChatRouteDecision(route=_SAFE_DEFAULT_ROUTE)

def classify_tool_call_route(
    *,
    client: OpenRouterClient,
    iota: str,
    tool_summaries: list[str],
    conversation_summary: str | None,
    max_tokens: int = 128,
) -> ChatRouteDecision:
    """Ask the classifier LLM whether this turn needs the full kernel.

    Always returns a :class:`ChatRouteDecision`. If the upstream call raises,
    or the response doesn't parse into a valid decision, we log it and return
    the safe default (``reasoner``) so the request still goes through.
    """
    # An empty user message is a degenerate input — the classifier has nothing
    # to classify, so skip the round-trip and go straight to the default.
    if not iota or not iota.strip():
        return _safe_default("empty iota — nothing to classify")

    user_message = _build_classifier_prompt(iota, tool_summaries, conversation_summary)

    try:
        parsed = client.chat(
            [
                {"role": "system", "content": _CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            response_model=ChatRouteDecision,
            max_tokens=max_tokens,
        )
    except ValidationError as e:
        return _safe_default("response failed schema validation", e)
    except Exception as e:
        return _safe_default("upstream classifier call failed", e)

    if not isinstance(parsed, ChatRouteDecision):
        return _safe_default(
            f"unexpected payload type {type(parsed).__name__}"
        )

    _log.info("classifier decision: route=%s", parsed.route)
    return parsed