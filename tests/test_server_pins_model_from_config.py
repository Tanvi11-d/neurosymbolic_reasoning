"""Server pins every base-model call to the configured model.

Whatever the client puts in ``body.model`` or ``body.base_model`` is
**ignored** for routing. This guarantees a Cloud Run deployment running
``config.yaml`` always uses the operator-chosen OpenRouter model, even when a
benchmark harness sends ``"model": "openai/gpt-4o"``.

Covers both code paths:

- **Passthrough** (no ``tools`` on the request): server bypasses the
  kernel and calls OpenRouter directly. The model sent to OpenRouter
  must be ``OpenRouterSettings().model`` (main ``llm`` block).
- **Kernel path** (request has ``tools``): after DECOMPOSE/CHECK/MATCH, the
  server calls OpenRouter once more with the client tools. That call uses
  ``resolve_tool_completion_settings(OpenRouterSettings())`` — when
  ``llm.tool_completion`` is set in YAML, that is a different model/provider
  than the kernel primitives.

The response envelope ``model`` is always the CoreThink public id
(``corethink/nre-kernel``), never upstream or client-requested vendor slugs.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nre.schemas import LLMDecomposeOutput, LLMTaskItem
from nre.server.app import create_app


def _mock_decompose():
    return LLMDecomposeOutput(
        tasks=[LLMTaskItem(id="t1", description="d", tool_name="noop", depends_on=[], parameters={})],
        config_keys={},
    )


def _canned_base_response():
    return {
        "id": "or-test",
        "model": "whatever-openrouter-echoed",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_passthrough_ignores_body_model_and_uses_config_model(
    mock_kernel_cls: MagicMock, mock_app_cls: MagicMock
) -> None:
    instance = MagicMock()
    mock_kernel_cls.return_value = instance
    mock_app_cls.return_value = instance
    instance.chat_completions_create.return_value = _canned_base_response()

    # settings.model is whatever config.yaml resolves to at import time; we
    # don't assert its exact value here, only that ``body.model`` is ignored.
    # We compare the model sent to OpenRouter against the server's settings.
    app = create_app()
    from nre.llm.openrouter import OpenRouterSettings, resolve_direct_route_settings

    pinned_model = resolve_direct_route_settings(OpenRouterSettings()).model

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",  # client asks for something else
            "base_model": "anthropic/claude-sonnet-4.5",  # client also overrides
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        },
    )
    assert resp.status_code == 200
    assert instance.chat_completions_create.call_count == 1
    call_kw = instance.chat_completions_create.call_args_list[0][1]
    assert call_kw["model"] == pinned_model, (
        f"passthrough must use config model {pinned_model!r}, got {call_kw['model']!r}"
    )


@patch("nre.server.app.classify_tool_call_route")
@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_kernel_path_ignores_body_model_and_uses_config_model(
    mock_kernel_cls: MagicMock,
    mock_app_cls: MagicMock,
    mock_classify: MagicMock,
) -> None:
    """Same guarantee on the kernel + base-LLM path."""
    from nre.server.classify import ChatRouteDecision

    mock_classify.return_value = ChatRouteDecision(route="reasoner")
    instance = MagicMock()
    mock_kernel_cls.return_value = instance
    mock_app_cls.return_value = instance
    instance.chat.return_value = _mock_decompose()
    instance.chat_completions_create.return_value = _canned_base_response()

    app = create_app()
    from nre.llm.openrouter import OpenRouterSettings, resolve_tool_completion_settings

    pinned_model = resolve_tool_completion_settings(OpenRouterSettings()).model

    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "base_model": "anthropic/claude-sonnet-4.5",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "description": "no-op",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
    )
    assert resp.status_code == 200
    # The single chat_completions_create call (post-kernel tool completion)
    # must use the tool-completion routing from config (not the client's
    # requested model names).
    assert instance.chat_completions_create.call_count == 1
    call_kw = instance.chat_completions_create.call_args_list[0][1]
    assert call_kw["model"] == pinned_model, (
        f"post-kernel tool completion must use routed model {pinned_model!r}, "
        f"got {call_kw['model']!r}"
    )


@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_response_envelope_uses_corethink_public_model(
    mock_kernel_cls: MagicMock, mock_app_cls: MagicMock
) -> None:
    """Never leak proprietary or client-supplied model names in the API."""
    instance = MagicMock()
    mock_kernel_cls.return_value = instance
    mock_app_cls.return_value = instance
    instance.chat_completions_create.return_value = _canned_base_response()

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == "corethink/nre-kernel", data["model"]
