"""FastAPI server smoke tests (requires optional ``[server]`` deps)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from nre.examples.agentic_reasoner import IOTA_1, _make_turn1_tools
from nre.schemas import LLMDecomposeOutput, LLMTaskItem
from nre.server.app import create_app
from nre.server.schemas import chat_messages_to_dicts, openai_tools_to_nre_registry


def _mock_kernel_decompose() -> LLMDecomposeOutput:
    return LLMDecomposeOutput(
        tasks=[
            LLMTaskItem(
                id="t1",
                description="nav",
                tool_name="cd",
                depends_on=[],
                parameters={},
            ),
            LLMTaskItem(
                id="t2",
                description="mkdir archive",
                tool_name="mkdir",
                depends_on=["t1"],
                parameters={},
            ),
            LLMTaskItem(
                id="t3",
                description="move report",
                tool_name="mv",
                depends_on=["t2"],
                parameters={},
            ),
        ],
        config_keys={
            "cwd": "/",
            "cwd_is_workspace": False,
            "archive_exists": False,
        },
    )


def test_root_lists_endpoints() -> None:
    app = create_app()
    client = TestClient(app)
    data = client.get("/").json()
    assert data["service"] == "nre-kernel"
    assert "endpoints" in data
    assert data["endpoints"]["chat_completions"] == "/v1/chat/completions"


def test_chat_messages_to_dicts_preserves_tool_turn() -> None:
    """Regression: multi-turn tool history must not be stripped (OpenRouter provider 400)."""
    from nre.server.schemas import ChatMessage

    msgs = [
        ChatMessage(role="user", content="hello"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "pwd", "arguments": "{}"},
                }
            ],
        ),
        ChatMessage(role="tool", tool_call_id="call_1", content="ok"),
    ]
    d = chat_messages_to_dicts(msgs)
    assert d[1]["tool_calls"][0]["id"] == "call_1"
    assert d[2]["tool_call_id"] == "call_1"
    assert d[2]["role"] == "tool"


def test_messages_prior_to_last_user() -> None:
    from nre.server.schemas import ChatMessage, last_user_content, messages_prior_to_last_user

    msgs = [
        ChatMessage(role="user", content="first"),
        ChatMessage(role="assistant", content="ok"),
        ChatMessage(role="user", content="last"),
    ]
    prior = messages_prior_to_last_user(msgs)
    assert len(prior) == 2
    assert prior[0]["role"] == "user" and prior[0]["content"] == "first"
    assert last_user_content(msgs) == "last"


def test_build_decompose_user_prompt_block_order_prior_turns() -> None:
    from nre.library.hybrid import build_decompose_user_prompt

    text = build_decompose_user_prompt(
        "current",
        {},
        {"x": {"description": "d", "params": []}},
        conversation_summary="SUMMARY",
        prior_turns=[{"role": "user", "content": "earlier"}],
    )
    assert text.index("## Conversation history (prior turns)") < text.index(
        "## Prior conversation (tools already run)"
    )
    assert text.index("## Prior conversation (tools already run)") < text.index(
        "## Current user request"
    )
    assert "earlier" in text and "SUMMARY" in text and "current" in text


def test_openai_tools_to_nre_registry() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "mkdir",
                "description": "Make directory",
                "parameters": {
                    "type": "object",
                    "properties": {"dir_name": {"type": "string"}},
                    "required": ["dir_name"],
                },
            },
        }
    ]
    reg = openai_tools_to_nre_registry(tools)
    assert "mkdir" in reg
    assert reg["mkdir"]["params"] == ["dir_name"]


def test_openai_tools_to_nre_registry_extracts_json_schema_enum() -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "place_order",
                "description": "Trade",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "order_type": {"type": "string", "enum": ["Buy", "Sell"]},
                    },
                },
            },
        }
    ]
    reg = openai_tools_to_nre_registry(tools)
    assert reg["place_order"]["param_enums"]["order_type"] == ["Buy", "Sell"]


def test_format_tool_schemas_shows_allowed_enum() -> None:
    from nre.utils.tool_registry import format_tool_schemas, tool_registry_to_schemas

    reg = {
        "x": {
            "description": "d",
            "params": ["order_type"],
            "param_enums": {"order_type": ["Buy", "Sell"]},
        }
    }
    text = format_tool_schemas(tool_registry_to_schemas(reg))
    assert "allowed:" in text
    assert "Buy" in text and "Sell" in text


def test_kernel_turn_to_chat_messages() -> None:
    from nre.utils.kernel_transcript import kernel_turn_to_chat_messages

    raw = {
        "steps": [
            {
                "phase": "match",
                "function": "mean",
                "status": "success",
                "bindings": {"numbers": [1, 2]},
                "result": 1.5,
            }
        ]
    }
    msgs = kernel_turn_to_chat_messages(iota="average", raw_turn=raw)
    assert msgs[0] == {"role": "user", "content": "average"}
    assert msgs[1]["role"] == "assistant"
    assert "mean" in msgs[1]["content"]


def test_health_ok() -> None:
    app = create_app()
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}


def test_models_lists_corethink_only() -> None:
    app = create_app()
    client = TestClient(app)
    data = client.get("/v1/models").json()
    ids = {m["id"] for m in data["data"]}
    assert ids == {"corethink/nre-kernel"}


@patch("nre.server.app.classify_tool_call_route")
@patch("nre.server.app.OpenRouterClient")
@patch("nre.examples.agentic_reasoner.OpenRouterClient")
def test_chat_completions_kernel_then_base_llm(
    mock_kernel_or: MagicMock,
    mock_app_or: MagicMock,
    mock_classify: MagicMock,
) -> None:
    """Kernel DECOMPOSE is mocked; base model is mocked; response carries tool_calls + trace."""
    from nre.server.classify import ChatRouteDecision

    mock_classify.return_value = ChatRouteDecision(route="reasoner")
    instance = MagicMock()
    mock_kernel_or.return_value = instance
    mock_app_or.return_value = instance
    instance.chat.return_value = _mock_kernel_decompose()
    passthrough_resp = {
        "id": "or-0",
        "model": "mock/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hi!"},
            }
        ],
        "usage": {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        },
    }
    kernel_then_base_resp = {
        "id": "or-1",
        "model": "mock/model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "cd",
                                "arguments": '{"folder": "workspace"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    instance.chat_completions_create.side_effect = [
        passthrough_resp,
        kernel_then_base_resp,
    ]

    app = create_app()
    client = TestClient(app)
    r0 = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello"}]},
    )
    assert r0.status_code == 200
    j0 = r0.json()
    assert j0["choices"][0]["message"]["content"] == "Hi!"
    assert "nre_kernel_trace" not in j0
    assert "nre_primitive_log" not in j0
    assert j0.get("nre", {}).get("route") == "direct"
    assert j0.get("nre", {}).get("path") == "passthrough"

    state: dict = {}
    log: list = []
    reg = _make_turn1_tools(state, log)
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": str(v.get("description", "")),
                "parameters": {
                    "type": "object",
                    "properties": {p: {"type": "string"} for p in v.get("params", [])},
                },
            },
        }
        for n, v in reg.items()
        if isinstance(v, dict)
    ]
    r2 = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": IOTA_1}],
            "tools": openai_tools,
            "turn_context_by_task_id": {
                "t1": {"folder": "workspace"},
                "t2": {"dir_name": "archive"},
                "t3": {"source": "report.txt", "destination": "archive"},
            },
        },
    )
    assert r2.status_code == 200, r2.text
    payload = r2.json()
    msg = payload["choices"][0]["message"]
    assert msg["role"] == "assistant"
    assert msg["tool_calls"][0]["function"]["name"] == "cd"
    assert payload["usage"]["total_tokens"] == 15
    assert "nre_kernel_trace" in payload
    assert payload["nre_kernel_trace"]["ordered_task_ids"] == ["t1", "t2", "t3"]
    assert "nre_primitive_log" in payload
    assert len(payload["nre_primitive_log"]["events"]) >= 2
    assert "nre" in payload
    assert payload["nre"]["kernel_trace"] == payload["nre_kernel_trace"]
    assert payload["nre"]["primitive_log"] == payload["nre_primitive_log"]
    assert payload["nre"]["route"] == "reasoner"

    assert instance.chat_completions_create.call_count == 2
    call_kw = instance.chat_completions_create.call_args_list[1][1]
    assert "Kernel reasoning trace" in call_kw["messages"][0]["content"]
    assert call_kw["tools"] == openai_tools


def test_chat_completions_can_omit_trace_in_response() -> None:
    from nre.server.classify import ChatRouteDecision

    with (
        patch("nre.server.app.classify_tool_call_route") as mock_classify,
        patch("nre.examples.agentic_reasoner.OpenRouterClient") as mock_k,
        patch("nre.server.app.OpenRouterClient") as mock_or_cls,
    ):
        mock_classify.return_value = ChatRouteDecision(route="reasoner")
        instance = MagicMock()
        mock_k.return_value = instance
        mock_or_cls.return_value = instance
        instance.chat.return_value = _mock_kernel_decompose()
        instance.chat_completions_create.return_value = {
            "id": "or-2",
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "ok"},
                }
            ],
            "usage": None,
        }
        app = create_app()
        client = TestClient(app)
        reg = _make_turn1_tools({}, [])
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": n,
                    "description": str(v.get("description", "")),
                    "parameters": {
                        "type": "object",
                        "properties": {p: {"type": "string"} for p in v.get("params", [])},
                    },
                },
            }
            for n, v in reg.items()
            if isinstance(v, dict)
        ]
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": IOTA_1}],
                "tools": openai_tools,
                "include_kernel_trace_in_response": False,
                "turn_context_by_task_id": {
                    "t1": {"folder": "workspace"},
                    "t2": {"dir_name": "archive"},
                    "t3": {"source": "report.txt", "destination": "archive"},
                },
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert "nre_kernel_trace" not in body
        assert "nre_primitive_log" in body
        assert "nre" in body
        assert body["nre"]["route"] == "reasoner"
        assert "kernel_trace" not in body["nre"]
        assert "primitive_log" in body["nre"]
        r2 = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": IOTA_1}],
                "tools": openai_tools,
                "include_kernel_trace_in_response": False,
                "include_primitive_log_in_response": False,
                "turn_context_by_task_id": {
                    "t1": {"folder": "workspace"},
                    "t2": {"dir_name": "archive"},
                    "t3": {"source": "report.txt", "destination": "archive"},
                },
            },
        )
        assert r2.status_code == 200
        r2j = r2.json()
        assert "nre_primitive_log" not in r2j
        assert r2j.get("nre") == {"route": "reasoner"}
