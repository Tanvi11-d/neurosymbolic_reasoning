"""FastAPI app: HTTP front-end for the NRE tool-calling kernel.

- ``POST /v1/chat/completions`` — run the kernel, inject a **reasoning trace** into the
  prompt, call OpenRouter again with the client ``tools`` (optional
  ``llm.tool_completion`` model/provider in ``config.yaml``), and return
  OpenAI-style ``tool_calls`` / ``content`` (plus optional ``nre_kernel_trace``).

Run with uvicorn::

    uvicorn nre.server.app:app --host 0.0.0.0 --port 8000

Or::

    python run_server.py
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from nre.examples.agentic_reasoner import build_engine
from nre.llm import (
    OpenRouterClient,
    OpenRouterError,
    OpenRouterSettings,
    resolve_classifier_settings,
    resolve_direct_route_settings,
    resolve_tool_completion_settings,
)
from nre.llm.config import load_discovered_config
from nre.server.classify import ChatRouteDecision, classify_tool_call_route
from nre.server.schemas import (
    ChatCompletionRequest,
    chat_messages_to_dicts,
    last_user_content,
    messages_prior_to_last_user,
    openai_tools_to_nre_registry,
    openai_tool_summaries_for_classifier,
)
from nre.server.pipeline_helpers import (
    _count_tool_messages,
    _error_recovery_message,
    _file_write_reminder,
    _live_data_reminder,
    _loop_breaker_message,
    _scrub_host_injected_tool_args,
    _should_run_kernel,
)
from nre.server.serialize import json_safe_run_turn_result
from nre.server.trace import build_kernel_augmented_messages
from nre.primitives import ensure_primitive_logging_visible
from nre.primitives.log_buffer import (
    format_primitive_log_for_api,
    primitive_log_session,
)

_log_api = logging.getLogger("nre.api")
_log_engine = logging.getLogger("nre.engine")
_log_serialize = logging.getLogger("nre.serialize")
_log_openrouter = logging.getLogger("nre.openrouter")

# OpenAI-style responses and ``GET /v1/models`` must not surface upstream /
# vendor model slugs — only this CoreThink-facing id.
CORETHINK_PUBLIC_MODEL_ID = "corethink/nre-kernel"


def _http_routing_enabled() -> bool:
    cfg = load_discovered_config()
    return bool(cfg and cfg.routing and cfg.routing.enabled)


def _ensure_server_log_level() -> None:
    """So ``uvicorn nre.server.app:app`` still shows pipeline INFO without ``nre-serve``."""
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
    logging.getLogger("nre").setLevel(logging.INFO)
    ensure_primitive_logging_visible()


def _error_chat_completion(detail: str, *, model: str) -> dict[str, Any]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"error": detail}),
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _chat_completions_pipeline(body: ChatCompletionRequest) -> dict[str, Any]:
    iota = last_user_content(body.messages)
    registry = openai_tools_to_nre_registry(body.tools)
    _log_api.info(
        "chat/completions: request iota_chars=%s tools_in_omega=%s tau_tasks=%s",
        len(iota),
        len(registry),
        len(body.turn_context_by_task_id or {}),
    )
    if not iota:
        _log_api.warning("chat/completions: abort — no user message content")
        return _error_chat_completion(
            "no user message with content",
            model=CORETHINK_PUBLIC_MODEL_ID,
        )

    settings = OpenRouterSettings()
    routing_on = _http_routing_enabled()

    # No OpenAI-style tools → empty Ω: skip kernel and call OpenRouter directly.
    if not registry:
        _log_api.info(
            "chat/completions: empty tool registry — passthrough (%s)",
            "routed direct model" if routing_on else "kernel settings model",
        )
        route_settings = (
            resolve_direct_route_settings(settings) if routing_on else settings
        )
        base_model = route_settings.model
        if body.base_model or (body.model and body.model != "nre-kernel"):
            _log_api.info(
                "chat/completions: ignoring client model overrides "
                "(body.model=%r base_model=%r) — routing to %s",
                body.model,
                body.base_model,
                base_model,
            )
        client = OpenRouterClient(route_settings)
        convo = chat_messages_to_dicts(body.messages)
        try:
            or_resp = client.chat_completions_create(
                messages=convo,
                tools=body.tools if body.tools else None,
                model=base_model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
            )
        except OpenRouterError as e:
            _log_openrouter.error("base LLM (passthrough) failed: %s", e)
            raise HTTPException(status_code=503, detail=str(e)) from e
        c0 = or_resp["choices"][0]
        msg = c0["message"]
        _scrub_host_injected_tool_args(msg.get("tool_calls"))
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        out: dict[str, Any] = {
            "id": cid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": CORETHINK_PUBLIC_MODEL_ID,
            "choices": [
                {
                    "index": 0,
                    "message": msg,
                    "finish_reason": c0.get("finish_reason") or "stop",
                }
            ],
            "usage": or_resp.get("usage")
            or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        if routing_on:
            out["nre"] = {"route": "direct", "path": "passthrough"}
        return out

    if routing_on:
        cl_settings = resolve_classifier_settings(settings)
        cl_client = OpenRouterClient(cl_settings)
        tool_lines = openai_tool_summaries_for_classifier(body.tools)
        try:
            decision = classify_tool_call_route(
                client=cl_client,
                iota=iota,
                tool_summaries=tool_lines,
                conversation_summary=body.conversation_summary,
            )
        except Exception as e:
            _log_api.warning(
                "chat/completions: route classifier failed; defaulting to reasoner: %s",
                e,
            )
            decision = ChatRouteDecision(route="reasoner")
        if decision.route == "direct":
            direct = resolve_direct_route_settings(settings)
            _log_api.info(
                "chat/completions: classifier=direct model=%s (skip kernel)",
                direct.model,
            )
            client = OpenRouterClient(direct)
            convo = chat_messages_to_dicts(body.messages)
            try:
                or_resp = client.chat_completions_create(
                    messages=convo,
                    tools=body.tools,
                    model=direct.model,
                    temperature=body.temperature,
                    max_tokens=body.max_tokens,
                )
            except OpenRouterError as e:
                _log_openrouter.error("direct model (classifier) failed: %s", e)
                raise HTTPException(status_code=503, detail=str(e)) from e
            c0 = or_resp["choices"][0]
            _scrub_host_injected_tool_args(c0["message"].get("tool_calls"))
            cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            return {
                "id": cid,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": CORETHINK_PUBLIC_MODEL_ID,
                "choices": [
                    {
                        "index": 0,
                        "message": c0["message"],
                        "finish_reason": c0.get("finish_reason") or "stop",
                    }
                ],
                "usage": or_resp.get("usage")
                or {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
                "nre": {"route": "direct", "path": "classifier"},
            }

    # The kernel earns its keep on turn 1 (initial plan) and when the previous
    # tool call errored (the plan needs to route around the failure). Every
    # other turn the base LLM is just executing the existing plan, so a fresh
    # kernel run would only burn the working-time budget.
    n_tool_turns = _count_tool_messages(body.messages)
    run_kernel, kernel_reason = _should_run_kernel(body.messages, n_tool_turns)

    trace: dict[str, Any] = {}
    primitive_log = None
    if not run_kernel:
        _log_api.info(
            "chat/completions: skipping kernel (tool_msgs=%s) — %s",
            n_tool_turns,
            kernel_reason,
        )
    else:
        _log_api.info(
            "chat/completions: running kernel (tool_msgs=%s) — %s",
            n_tool_turns,
            kernel_reason,
        )
        engine = build_engine()
        _log_engine.info(
            "Engine registry primitives=%s", sorted(engine.registry.all_names())
        )
        try:
            with primitive_log_session() as plog:
                raw = engine.run(
                    "agent_turn",
                    iota=iota,
                    gamma=body.gamma or {},
                    tools=registry,
                    turn_context_by_task_id=body.turn_context_by_task_id or {},
                    conversation_summary=body.conversation_summary,
                    prior_turns=messages_prior_to_last_user(body.messages) or None,
                    gamma_seeds=body.gamma_seeds,
                )
                primitive_log = plog
        except ValueError as e:
            _log_api.warning("chat/completions: kernel ValueError: %s", e)
            return _error_chat_completion(str(e), model=CORETHINK_PUBLIC_MODEL_ID)

        trace = json_safe_run_turn_result(raw)
        _log_serialize.info(
            "json_safe_run_turn_result: decompose_turns=%s",
            len(trace.get("decompose", {}).get("turns", [])),
        )

    convo = chat_messages_to_dicts(body.messages)
    effective_tools = body.tools

    # Dynamic, content-driven nudges. Each helper looks at the actual messages
    # and decides whether to inject a targeted guidance line; none of them
    # rely on hardcoded thresholds. Order matters — error recovery first
    # (most specific), then loop-breaker (mid-task drift), then the live-data
    # and file-write reminders (drive the model toward the right kind of tool
    # call before it submits).
    for nudge_helper, label in (
        (lambda: _error_recovery_message(body.messages), "error recovery"),
        (lambda: _loop_breaker_message(body.messages), "loop breaker"),
        (lambda: _live_data_reminder(iota, body.messages), "live data reminder"),
        (lambda: _file_write_reminder(iota, body.messages), "file write reminder"),
    ):
        nudge = nudge_helper()
        if nudge is not None:
            _log_api.info(
                "chat/completions: injected %s nudge (tool_msgs=%s)",
                label,
                n_tool_turns,
            )
            convo = [*convo, nudge]

    messages = build_kernel_augmented_messages(trace, convo)

    # Post-kernel OpenRouter call: ``llm.tool_completion`` in config.yaml when
    # set (e.g. MiniMax + SambaNova); otherwise same settings as kernel.
    # Client ``body.model`` / ``body.base_model`` are ignored for routing.
    base_settings = resolve_tool_completion_settings(settings)
    base_model = base_settings.model
    if body.base_model or (body.model and body.model != "nre-kernel"):
        _log_api.info(
            "chat/completions: ignoring client model overrides "
            "(body.model=%r base_model=%r) — routing to %s",
            body.model,
            body.base_model,
            base_model,
        )

    client = OpenRouterClient(base_settings)
    _log_openrouter.info(
        "base LLM request: model=%s openai_tools=%s temp=%s max_tokens=%s",
        base_model,
        len(effective_tools or []),
        body.temperature,
        body.max_tokens,
    )
    try:
        or_resp = client.chat_completions_create(
            messages=messages,
            tools=effective_tools,
            model=base_model,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except OpenRouterError as e:
        _log_openrouter.error("base LLM failed: %s", e)
        raise HTTPException(status_code=503, detail=str(e)) from e

    c0 = or_resp["choices"][0]
    msg = c0["message"]
    _scrub_host_injected_tool_args(msg.get("tool_calls"))
    tc = msg.get("tool_calls") or []
    _log_openrouter.info(
        "base LLM response: finish_reason=%s content_chars=%s tool_calls=%s usage=%s",
        c0.get("finish_reason"),
        len(msg.get("content") or "") if msg.get("content") else 0,
        len(tc),
        or_resp.get("usage"),
    )
    for j, call in enumerate(tc):
        fn = call.get("function") or {}
        _log_openrouter.info(
            "tool_call[%s] name=%s args_preview=%s",
            j,
            fn.get("name"),
            (fn.get("arguments") or "")[:120],
        )

    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    out: dict[str, Any] = {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": CORETHINK_PUBLIC_MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": c0["message"],
                "finish_reason": c0.get("finish_reason") or "stop",
            }
        ],
        "usage": or_resp.get("usage")
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    # primitive_log is None on continuation turns where we skipped the kernel —
    # there's nothing to format, so don't pretend there is.
    shaped_primitive_log = (
        format_primitive_log_for_api(primitive_log)
        if body.include_primitive_log_in_response and primitive_log is not None
        else None
    )
    if body.include_kernel_trace_in_response:
        out["nre_kernel_trace"] = trace
    if shaped_primitive_log is not None:
        out["nre_primitive_log"] = shaped_primitive_log

    nre_bundle: dict[str, Any] = {"route": "reasoner"}
    if body.include_kernel_trace_in_response:
        nre_bundle["kernel_trace"] = trace
    if shaped_primitive_log is not None:
        nre_bundle["primitive_log"] = shaped_primitive_log
    if nre_bundle:
        out["nre"] = nre_bundle

    _log_api.info(
        "chat/completions: response id=%s include_kernel_trace=%s include_primitive_log=%s nre_keys=%s",
        cid,
        body.include_kernel_trace_in_response,
        body.include_primitive_log_in_response,
        list(nre_bundle.keys()),
    )
    return out


def create_app() -> Any:
    _ensure_server_log_level()
    app = FastAPI(
        title="NRE kernel server",
        description="Neurosymbolic reasoning engine — kernel trace + base LLM tool calls",
        version="0.1.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Service index (useful on Cloud Run / load balancers)."""
        return {
            "service": "nre-kernel",
            "title": app.title,
            "version": app.version,
            "description": app.description,
            "endpoints": {
                "health": "/health",
                "models": "/v1/models",
                "chat_completions": "/v1/chat/completions",
            },
            "docs": "/docs",
        }

    @app.get("/health")
    async def health() -> dict[str, str]:
        _log_api.debug("GET /health")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        now = int(time.time())
        data = [
            {
                "id": CORETHINK_PUBLIC_MODEL_ID,
                "object": "model",
                "created": now,
                "owned_by": "corethink",
            },
        ]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat_completions(body: ChatCompletionRequest) -> dict[str, Any]:
        """Run the five primitives, attach trace, call base OpenRouter, return tool_calls."""
        return _chat_completions_pipeline(body)

    return app


# Uvicorn string import target
app = create_app()
