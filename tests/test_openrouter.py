from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from nre.llm.openrouter import (
    OpenRouterClient,
    OpenRouterError,
    OpenRouterParseError,
    OpenRouterSettings,
    _assistant_message_text,
    _strip_json_fence,
)


def _isolate_from_repo_config_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """No discovered YAML — :class:`OpenRouterSettings` uses code defaults."""
    monkeypatch.delenv("NRE_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path / "_home"))


def _mock_raw_completion(parsed, *, body="{}", status=200):
    raw = MagicMock()
    raw.http_response.text = body
    raw.http_response.status_code = status
    raw.parse.return_value = parsed
    return raw


def test_strip_json_fence():
    assert _strip_json_fence('{"a": 1}') == '{"a": 1}'
    assert _strip_json_fence("```json\n{\"x\": 2}\n```") == '{"x": 2}'
    assert _strip_json_fence("```\n{}\n```") == "{}"


def test_assistant_message_text_list_parts():
    msg = MagicMock()
    msg.content = [{"type": "text", "text": '{"a": 1}'}]
    assert _assistant_message_text(msg) == '{"a": 1}'


def test_assistant_message_text_reasoning_json_fallback():
    msg = MagicMock()
    msg.content = None
    msg.reasoning = '{"n": 1}'
    assert _assistant_message_text(msg) == '{"n": 1}'


def test_settings_default_extra_body_minimax_fp8(monkeypatch, tmp_path):
    _isolate_from_repo_config_yaml(monkeypatch, tmp_path)
    s = OpenRouterSettings(api_key="k")
    assert s.extra_body == {"provider": {"order": ["minimax/fp8"]}}


def test_settings_extra_body_override():
    d = {"provider": {"order": ["fireworks"]}}
    assert OpenRouterSettings(api_key="k", extra_body=d).extra_body == d


def test_settings_extra_body_none_disables_routing():
    assert OpenRouterSettings(api_key="k", extra_body=None).extra_body is None


def test_settings_max_tokens():
    s = OpenRouterSettings(api_key="k")
    assert s.max_tokens == 32768
    assert OpenRouterSettings(api_key="k", max_tokens=8192).max_tokens == 8192


def test_client_missing_key():
    c = OpenRouterClient(OpenRouterSettings(api_key=""))
    with pytest.raises(OpenRouterError, match="API key"):
        c.chat([{"role": "user", "content": "hi"}])


@patch("nre.llm.openrouter.OpenAI")
def test_chat_plain_text(mock_openai_cls, monkeypatch, tmp_path):
    _isolate_from_repo_config_yaml(monkeypatch, tmp_path)
    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=[MagicMock(message=MagicMock(content="ok"))])
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x", max_tokens=16384))
    out = c.chat([{"role": "user", "content": "ping"}])

    assert out == "ok"
    inst.with_raw_response.chat.completions.create.assert_called_once()
    call_kw = inst.with_raw_response.chat.completions.create.call_args.kwargs
    assert call_kw["model"] == "minimax/minimax-m2.7"
    assert call_kw["max_tokens"] == 16384
    assert "response_format" not in call_kw
    assert call_kw["extra_body"] == {"provider": {"order": ["minimax/fp8"]}}


@patch("nre.llm.openrouter.OpenAI")
def test_chat_max_tokens_override(mock_openai_cls):
    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=[MagicMock(message=MagicMock(content="x"))])
    )
    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x", max_tokens=4096))
    c.chat([{"role": "user", "content": "p"}], max_tokens=8192)
    call_kw = inst.with_raw_response.chat.completions.create.call_args.kwargs
    assert call_kw["max_tokens"] == 8192


@patch("nre.llm.openrouter.OpenAI")
def test_chat_structured(mock_openai_cls, monkeypatch, tmp_path):
    _isolate_from_repo_config_yaml(monkeypatch, tmp_path)
    class Out(BaseModel):
        n: int
        label: str

    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"n": 3, "label": "z"}'))]
        )
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x"))
    parsed = c.chat(
        [{"role": "user", "content": "emit json"}],
        response_model=Out,
    )
    assert isinstance(parsed, Out)
    assert parsed.n == 3
    assert parsed.label == "z"
    call_kw = inst.with_raw_response.chat.completions.create.call_args.kwargs
    msgs = call_kw["messages"]
    assert msgs[0]["role"] == "system"
    assert "JSON Schema" in msgs[0]["content"]
    assert call_kw["response_format"] == {"type": "json_object"}
    assert call_kw["extra_body"] == {
        "provider": {"order": ["minimax/fp8"]},
        "reasoning": {"enabled": False},
    }


@patch("nre.llm.openrouter.OpenAI")
def test_chat_structured_empty_content_retries_then_ok(mock_openai_cls):
    class Out(BaseModel):
        n: int

    inst = MagicMock()
    mock_openai_cls.return_value = inst
    empty_msg = MagicMock(message=MagicMock(content=""))
    ok_msg = MagicMock(message=MagicMock(content='{"n": 7}'))
    inst.with_raw_response.chat.completions.create.side_effect = [
        _mock_raw_completion(MagicMock(choices=[empty_msg])),
        _mock_raw_completion(MagicMock(choices=[ok_msg])),
    ]

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x", max_retries=3))
    with patch("nre.llm.openrouter.time.sleep"):
        parsed = c.chat(
            [{"role": "user", "content": "x"}],
            response_model=Out,
        )
    assert parsed.n == 7
    assert inst.with_raw_response.chat.completions.create.call_count == 2


@patch("nre.llm.openrouter.OpenAI")
def test_chat_structured_from_tool_call_arguments(mock_openai_cls):
    class Out(BaseModel):
        n: int

    inst = MagicMock()
    mock_openai_cls.return_value = inst
    fn = MagicMock(arguments='{"n": 42}')
    tc = MagicMock(function=fn)
    msg = MagicMock(content="", tool_calls=[tc])
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=[MagicMock(message=msg)])
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x"))
    parsed = c.chat(
        [{"role": "user", "content": "x"}],
        response_model=Out,
    )
    assert parsed.n == 42


@patch("nre.llm.openrouter.OpenAI")
def test_chat_structured_invalid_json(mock_openai_cls):
    class Out(BaseModel):
        n: int

    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=[MagicMock(message=MagicMock(content="not json"))])
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x"))
    with pytest.raises(OpenRouterParseError):
        c.chat([{"role": "user", "content": "x"}], response_model=Out)


@patch("nre.llm.openrouter.OpenAI")
def test_chat_completions_null_choices(mock_openai_cls):
    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=None, id=None, model=None),
        body='{"choices":null}',
        status=200,
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x"))
    with pytest.raises(OpenRouterError, match="no choices"):
        c.chat_completions_create([{"role": "user", "content": "hi"}])


@patch("nre.llm.openrouter.OpenAI")
def test_chat_null_choices(mock_openai_cls):
    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=None, id=None, model=None),
        body='{"choices":null}',
        status=200,
    )

    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x"))
    with pytest.raises(OpenRouterError, match="no choices"):
        c.chat([{"role": "user", "content": "hi"}])


@patch("nre.llm.openrouter.OpenAI")
def test_chat_passes_extra_body(mock_openai_cls):
    inst = MagicMock()
    mock_openai_cls.return_value = inst
    inst.with_raw_response.chat.completions.create.return_value = _mock_raw_completion(
        MagicMock(choices=[MagicMock(message=MagicMock(content="hi"))])
    )

    extra = {"provider": {"order": ["SambaNova"]}}
    c = OpenRouterClient(OpenRouterSettings(api_key="sk-x", extra_body=extra))
    c.chat([{"role": "user", "content": "x"}])

    call_kw = inst.with_raw_response.chat.completions.create.call_args.kwargs
    assert call_kw["extra_body"] == extra
