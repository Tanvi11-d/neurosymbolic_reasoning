"""HTTP routing: classifier sends non-agentic tool turns to the direct model."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nre.server.app import create_app
from nre.server.classify import ChatRouteDecision


@patch("nre.server.app.classify_tool_call_route")
@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_classifier_direct_skips_kernel(
    mock_kernel_or: MagicMock,
    mock_app_or: MagicMock,
    mock_classify: MagicMock,
) -> None:
    mock_classify.return_value = ChatRouteDecision(route="direct")
    instance = MagicMock()
    mock_app_or.return_value = instance
    mock_kernel_or.return_value = instance
    instance.chat_completions_create.return_value = {
        "id": "or-d",
        "model": "anthropic/claude-opus-4.7",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "plain"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    mock_engine = MagicMock()

    app = create_app()
    client = TestClient(app)
    with patch("nre.server.app.build_engine", mock_engine):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "calculator",
                            "description": "Compute arithmetic",
                            "parameters": {
                                "type": "object",
                                "properties": {"expr": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
        )

    assert resp.status_code == 200
    assert mock_engine.call_count == 0
    mock_classify.assert_called_once()
    assert instance.chat_completions_create.call_count == 1
    data = resp.json()
    assert data["nre"]["route"] == "direct"
    assert data["nre"]["path"] == "classifier"
    assert data["model"] == "corethink/nre-kernel"
