"""Server-level γ plumbing.

The ``ChatCompletionRequest`` schema must accept a ``gamma`` payload and the
app must thread it into ``engine.run("agent_turn", gamma=...)`` so the γ→S
flatten (P3) is available server-side. Without this, server-mode traffic
never benefits from γ extraction — only the agentic-reasoner CLI would.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nre.schemas import LLMDecomposeOutput, LLMTaskItem
from nre.server.app import create_app


def _decompose_one_task():
    return LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="t1",
                description="do thing",
                tool_name="read_fuel",
                depends_on=[],
                parameters={},
            ),
        ],
        config_keys={},
    )


@patch("nre.server.app.classify_tool_call_route")
@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_gamma_flattens_into_scratchpad_via_server(
    mock_kernel_or: MagicMock,
    mock_app_or: MagicMock,
    mock_classify: MagicMock,
) -> None:
    from nre.server.classify import ChatRouteDecision

    mock_classify.return_value = ChatRouteDecision(route="reasoner")
    instance = MagicMock()
    mock_kernel_or.return_value = instance
    mock_app_or.return_value = instance
    instance.chat.return_value = _decompose_one_task()
    instance.chat_completions_create.return_value = {
        "id": "x",
        "model": "mock/m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "nre-kernel",
            "messages": [{"role": "user", "content": "check the fuel"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_fuel",
                        "description": "Read the fuel level.",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                }
            ],
            "gamma": {
                "VehicleControlAPI": {
                    "fuelLevel": 10.5,
                    "engineState": "stopped",
                }
            },
            "include_kernel_trace_in_response": True,
        },
    )
    assert resp.status_code == 200
    trace = resp.json().get("nre_kernel_trace") or {}
    scratch = trace.get("scratchpad") or {}
    assert scratch.get("VehicleControlAPI.fuelLevel") == 10.5
    assert scratch.get("VehicleControlAPI.engineState") == "stopped"


@patch("nre.server.app.classify_tool_call_route")
@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_no_gamma_still_works_back_compat(
    mock_kernel_or: MagicMock,
    mock_app_or: MagicMock,
    mock_classify: MagicMock,
) -> None:
    from nre.server.classify import ChatRouteDecision

    mock_classify.return_value = ChatRouteDecision(route="reasoner")
    """Request without γ must behave as before (scratchpad may be empty)."""
    instance = MagicMock()
    mock_kernel_or.return_value = instance
    mock_app_or.return_value = instance
    instance.chat.return_value = _decompose_one_task()
    instance.chat_completions_create.return_value = {
        "id": "x",
        "model": "mock/m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "ok"},
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    app = create_app()
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "nre-kernel",
            "messages": [{"role": "user", "content": "check the fuel"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read_fuel",
                        "description": "Read the fuel level.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "include_kernel_trace_in_response": True,
        },
    )
    assert resp.status_code == 200
    trace = resp.json().get("nre_kernel_trace") or {}
    assert trace.get("scratchpad") == {} or not any(
        str(k).startswith("VehicleControlAPI.")
        for k in (trace.get("scratchpad") or {}).keys()
    )
