from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

import httpx
from openai import OpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

_DEFAULT_BASE = "https://openrouter.ai/api/v1"
_DEFAULT_MODEL = "minimax/minimax-m2.7"
_DEFAULT_MAX_TOKENS = 32768
_MIN_MAX_TOKENS = 1024
_MAX_MAX_TOKENS = 262144

# OpenRouter endpoint slug for MiniMax M2.7 (see model ``/endpoints`` on openrouter.ai).
_DEFAULT_PROVIDER_ORDER_MINIMAX_FP8: tuple[str, ...] = ("minimax/fp8",)


def _default_openrouter_extra_body() -> dict[str, Any]:
    return {"provider": {"order": list(_DEFAULT_PROVIDER_ORDER_MINIMAX_FP8)}}

_DISCOVERED_CONFIG = None
try:
    from nre.llm.config import load_discovered_config
    _DISCOVERED_CONFIG = load_discovered_config()
except Exception:
    pass

def _get_open_router_setting_parameters():
    def _build_extra_body() -> dict[str, Any] | None:
        if _DISCOVERED_CONFIG is None:
            return _default_openrouter_extra_body()
        extra: dict[str, Any] = {}
        if _DISCOVERED_CONFIG.provider_order is not None:
            if len(_DISCOVERED_CONFIG.provider_order) > 0:
                extra["provider"] = {"order": list(_DISCOVERED_CONFIG.provider_order)}
        else:
            extra["provider"] = {"order": list(_DEFAULT_PROVIDER_ORDER_MINIMAX_FP8)}
        if _DISCOVERED_CONFIG.reasoning is not None:
            extra["reasoning"] = dict(_DISCOVERED_CONFIG.reasoning)
        return extra if extra else None
    
    return {
        "model": (
            _DISCOVERED_CONFIG.model
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.model)
            else _DEFAULT_MODEL),
        "base_url": (
            _DISCOVERED_CONFIG.base_url
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.base_url)         
            else _DEFAULT_BASE),
        "temperature": (
            _DISCOVERED_CONFIG.temperature
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.temperature      is not None) 
            else 0.0),
        "max_tokens": (
            _DISCOVERED_CONFIG.max_tokens
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.max_tokens       is not None) 
            else _DEFAULT_MAX_TOKENS),
        "max_retries": (
            _DISCOVERED_CONFIG.max_retries
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.max_retries      is not None) 
            else 3),
        "timeout_seconds": (
            _DISCOVERED_CONFIG.timeout_seconds  
            if (_DISCOVERED_CONFIG is not None and _DISCOVERED_CONFIG.timeout_seconds  is not None) 
            else 120.0),
        "extra_body": (
            _build_extra_body() 
            if _DISCOVERED_CONFIG is not None 
            else _default_openrouter_extra_body()),
    }

def resolve_tool_completion_settings(kernel: OpenRouterSettings) -> OpenRouterSettings:
    """Build settings for the tool completion call. Falls back to kernel settings if not configured."""

    # If no tool completion model is set in config, just use the main kernel settings.
    if (
        _DISCOVERED_CONFIG is None
        or _DISCOVERED_CONFIG.tool_completion is None
        or not _DISCOVERED_CONFIG.tool_completion.model
    ):
        return kernel

    tc = _DISCOVERED_CONFIG.tool_completion

    # Build the extra_body from tool completion config only.
    # We don't reuse the main llm provider pin — tool completion
    # models are usually hosted on different providers.
    extra_body = {}
    if tc.provider_order is not None and len(tc.provider_order) > 0:
        extra_body["provider"] = {"order": list(tc.provider_order)}
    if tc.reasoning is not None:
        extra_body["reasoning"] = dict(tc.reasoning)

    # If nothing was added, set extra_body to None so OpenRouter auto-routes.
    final_extra_body = extra_body if extra_body else None

    # Use tool completion value if set, otherwise fall back to kernel value.
    def use_tc_or_kernel(tc_value, kernel_value):
        return kernel_value if tc_value is None else tc_value

    return OpenRouterSettings(
        api_key         = kernel.api_key,
        base_url        = tc.base_url or kernel.base_url,
        model           = tc.model,
        temperature     = float(use_tc_or_kernel(tc.temperature,     kernel.temperature)),
        max_tokens      = int(use_tc_or_kernel(tc.max_tokens,        kernel.max_tokens)),
        max_retries     = int(use_tc_or_kernel(tc.max_retries,       kernel.max_retries)),
        timeout_seconds = float(use_tc_or_kernel(tc.timeout_seconds, kernel.timeout_seconds)),
        http_referer    = kernel.http_referer,
        x_title         = kernel.x_title,
        extra_body      = final_extra_body,
    )

DEFAULT_DIRECT_ROUTE_MODEL = "anthropic/claude-opus-4.7"


def resolve_direct_route_settings(kernel: OpenRouterSettings) -> OpenRouterSettings:
    """Build settings for direct HTTP completions (no kernel).
    Returns kernel unchanged if routing is off or not configured.
    """
    cfg = _DISCOVERED_CONFIG

    if cfg is None or cfg.routing is None or not cfg.routing.enabled:
        return kernel

    routing = cfg.routing
    model = routing.direct_model or DEFAULT_DIRECT_ROUTE_MODEL

    extra_body = {}
    if routing.provider_order is not None and len(routing.provider_order) > 0:
        extra_body["provider"] = {"order": list(routing.provider_order)}
    if routing.reasoning is not None:
        extra_body["reasoning"] = dict(routing.reasoning)

    final_extra_body = extra_body if extra_body else None
    def use_routing_or_kernel(routing_value, kernel_value):
        return kernel_value if routing_value is None else routing_value

    return OpenRouterSettings(
        api_key         = kernel.api_key,
        base_url        = routing.base_url or kernel.base_url,
        model           = model,
        temperature     = float(use_routing_or_kernel(routing.temperature,     kernel.temperature)),
        max_tokens      = int(use_routing_or_kernel(routing.max_tokens,        kernel.max_tokens)),
        max_retries     = int(use_routing_or_kernel(routing.max_retries,       kernel.max_retries)),
        timeout_seconds = float(use_routing_or_kernel(routing.timeout_seconds, kernel.timeout_seconds)),
        http_referer    = kernel.http_referer,
        x_title         = kernel.x_title,
        extra_body      = final_extra_body,
    )

def resolve_classifier_settings(kernel: OpenRouterSettings) -> OpenRouterSettings:
    """OpenRouter settings for the HTTP route **classifier** only (JSON schema).

    When ``llm.routing.classifier.model`` is set, returns dedicated
    :class:`OpenRouterSettings`; otherwise returns *kernel* (same as main ``llm``).
    """
    cfg = _DISCOVERED_CONFIG
    if (
        cfg is None
        or cfg.routing is None
        or not cfg.routing.enabled
        or cfg.routing.classifier is None
        or not cfg.routing.classifier.model
    ):
        return kernel
    cf = cfg.routing.classifier
    extra: dict[str, Any] = {}
    if cf.provider_order is not None and len(cf.provider_order) > 0:
        extra["provider"] = {"order": list(cf.provider_order)}
    if cf.reasoning is not None:
        extra["reasoning"] = dict(cf.reasoning)
    extra_body: dict[str, Any] | None = extra if extra else None

    def _coalesce_cf(over: float | int | None, base: float | int) -> float | int:
        return base if over is None else over

    return OpenRouterSettings(
        api_key=kernel.api_key,
        base_url=cf.base_url or kernel.base_url,
        model=cf.model,
        temperature=float(_coalesce_cf(cf.temperature, kernel.temperature)),
        max_tokens=int(_coalesce_cf(cf.max_tokens, kernel.max_tokens)),
        max_retries=int(_coalesce_cf(cf.max_retries, kernel.max_retries)),
        timeout_seconds=float(_coalesce_cf(cf.timeout_seconds, kernel.timeout_seconds)),
        http_referer=kernel.http_referer,
        x_title=kernel.x_title,
        extra_body=extra_body,
    )

_LOG = logging.getLogger(__name__)
_LOG_SEP = "=" * 72


def _message_entry_content_len(m: Any) -> int:
    if not isinstance(m, dict):
        return len(str(m))
    c = m.get("content")
    if c is None:
        return 0
    if isinstance(c, str):
        return len(c)
    if isinstance(c, list):
        n = 0
        for b in c:
            if isinstance(b, dict):
                t = b.get("text")
                n += len(t) if isinstance(t, str) else len(str(b))
            else:
                n += len(str(b))
        return n
    return len(str(c))


def _format_messages_overview(messages: Any, *, preview_chars: int = 320) -> str:
    if not isinstance(messages, list):
        return f"  (messages not a list: {type(messages).__name__})"
    lines: list[str] = []
    total = 0
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            lines.append(f"  [{i}] <non-dict {type(m).__name__}>")
            continue
        role = m.get("role", "?")
        n = _message_entry_content_len(m)
        total += n
        c = m.get("content")
        prev = ""
        if isinstance(c, str) and c:
            frag = c[:preview_chars].replace("\r", "").replace("\n", " ")
            if len(c) > preview_chars:
                frag += f" … [{len(c) - preview_chars} more chars]"
            prev = frag
        lines.append(f"  [{i}] role={role} content_chars={n} preview={prev!r}")
    lines.insert(0, f"  message_count={len(messages)} total_content_chars={total}")
    return "\n".join(lines)


def _log_chat_completion_request(
    call_site: str,
    attempt: int,
    max_retries: int,
    kwargs_for_api: dict[str, Any],
) -> None:
    """Log OpenAI-style parameters and a compact view of messages (not full bodies)."""
    meta = {k: v for k, v in kwargs_for_api.items() if k != "messages"}
    tools = meta.pop("tools", None)
    tool_choice = meta.pop("tool_choice", None)
    extra_body = meta.pop("extra_body", None)
    blocks = [
        _LOG_SEP,
        f"OpenRouter chat.completions.create  {call_site}  "
        f"attempt {attempt + 1}/{max_retries}",
        _LOG_SEP,
        "OpenAI-compatible parameters:",
        json.dumps(meta, indent=2, default=str),
    ]
    if tools is not None:
        blocks.append(f"tools: list length = {len(tools)}")
    if tool_choice is not None:
        blocks.append(f"tool_choice: {tool_choice!r}")
    if extra_body is not None:
        blocks.append("extra_body (OpenRouter):")
        try:
            blocks.append(json.dumps(extra_body, indent=2, default=str))
        except TypeError:
            blocks.append(str(extra_body))
    blocks.append("messages (overview; full text not logged):")
    blocks.append(_format_messages_overview(kwargs_for_api.get("messages")))
    blocks.append(_LOG_SEP)
    _LOG.info("\n".join(blocks))


class OpenRouterError(Exception):
    pass


class OpenRouterParseError(OpenRouterError):
    pass


def _strip_json_fence(raw: str) -> str:
    """Extract the JSON payload from an LLM response.

    Handles three realistic shapes:

    - Pure JSON: returned unchanged.
    - ``` ```json ... ``` ``` fenced block (optionally with preamble / trailer
      prose): the fenced block's contents are returned.
    - Bare JSON object preceded or followed by prose (e.g. the model wrote
      a "reasoning" paragraph before the object even though
      ``response_format: json_object`` was set): the largest balanced-brace
      substring that parses as JSON is returned.

    Returns empty string for empty input; returns the input unchanged when
    no JSON object is detectable (so downstream ``json.loads`` surfaces a
    clear parse error rather than silently mis-parsing).
    """
    text = raw.strip()
    if not text:
        return ""

    # 1) Pure JSON fast-path.
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass  # fall through to brace scanner

    # 2) Fenced-block shape(s). Strip ```json or ``` wrappers when they
    #    exactly bound the text (legacy behavior).
    if text.startswith("```json"):
        inner = text[7:]
        if inner.endswith("```"):
            inner = inner[:-3]
        return inner.strip()
    if text.startswith("```"):
        inner = text[3:]
        if inner.endswith("```"):
            inner = inner[:-3]
        return inner.strip()

    # 3) Prose + JSON (with or without fences). Scan for balanced-brace
    #    substrings and return the LONGEST one that parses cleanly. This
    #    handles preamble, trailing text, and embedded example blocks.
    best: str | None = None
    for opener, closer in (("{", "}"), ("[", "]")):
        depth = 0
        start = -1
        in_str = False
        escape = False
        for i, ch in enumerate(text):
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch == opener:
                if depth == 0:
                    start = i
                depth += 1
            elif ch == closer:
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidate = text[start : i + 1]
                        try:
                            json.loads(candidate)
                            if best is None or len(candidate) > len(best):
                                best = candidate
                        except json.JSONDecodeError:
                            pass
                        start = -1

    if best is not None:
        return best
    return text


def _assistant_message_text(msg: Any) -> str:
    if msg is None:
        return ""
    raw = getattr(msg, "content", None)
    if raw is None:
        s = ""
    elif isinstance(raw, str):
        s = raw
    elif isinstance(raw, list):
        parts: list[str] = []
        for block in raw:
            if isinstance(block, dict):
                if block.get("type") == "text" and block.get("text"):
                    parts.append(str(block["text"]))
                elif block.get("text"):
                    parts.append(str(block["text"]))
            else:
                txt = getattr(block, "text", None)
                if txt:
                    parts.append(str(txt))
        s = "".join(parts)
    else:
        s = str(raw)
    if s.strip():
        return s
    reasoning = getattr(msg, "reasoning", None)
    if isinstance(reasoning, str):
        r = reasoning.strip()
        if len(r) >= 2 and r[0] in "{[":
            return reasoning
    return s


def _usage_tokens(resp: Any) -> tuple[int | None, int | None]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _json_from_tool_call_arguments(msg: Any) -> str | None:
    if msg is None:
        return None
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return None
    try:
        tc0 = tcs[0]
        fn = getattr(tc0, "function", None)
        if fn is None:
            return None
        args = getattr(fn, "arguments", None)
        if args is None or not isinstance(args, str):
            return None
        s = args.strip()
        if len(s) < 2 or s[0] not in "{[":
            return None
        return s
    except (IndexError, TypeError):
        return None


def _response_choices(resp: Any) -> list[Any]:
    raw = getattr(resp, "choices", None)
    return list(raw) if raw is not None else []


def _chat_completion_create_with_http(
    client: OpenAI, **kwargs: Any
) -> tuple[Any, httpx.Response]:
    raw = client.with_raw_response.chat.completions.create(**kwargs)
    http = raw.http_response
    try:
        parsed: Any = raw.parse()
    except Exception as e:
        body = (http.text or "")[:4000]
        raise OpenRouterError(
            f"OpenRouter response parse failed ({type(e).__name__}: {e}); "
            f"HTTP {http.status_code}; body_preview={body!r}"
        ) from e
    return parsed, http


def _http_body_preview(http: httpx.Response, *, limit: int = 4000) -> str:
    return (http.text or "")[:limit]


def _choice_message_to_dict(msg: Any) -> dict[str, Any]:
    if msg is None:
        return {"role": "assistant", "content": None}
    md: dict[str, Any] = {
        "role": getattr(msg, "role", None) or "assistant",
        "content": getattr(msg, "content", None),
    }
    tcs = getattr(msg, "tool_calls", None)
    if not tcs:
        return md
    tool_calls_out: list[dict[str, Any]] = []
    for tc in tcs:
        fn = getattr(tc, "function", None)
        name = getattr(fn, "name", None) if fn is not None else None
        arguments = getattr(fn, "arguments", None) if fn is not None else None
        tool_calls_out.append(
            {
                "id": getattr(tc, "id", None),
                "type": getattr(tc, "type", None) or "function",
                "function": {
                    "name": name or "",
                    "arguments": arguments if arguments is not None else "{}",
                },
            }
        )
    md["tool_calls"] = tool_calls_out
    return md


@dataclass
class OpenRouterSettings:
    """Transport configuration for :class:`OpenRouterClient`.

    Field defaults are resolved lazily at instance creation, in this order
    of precedence (highest wins):

    1. Explicit constructor arguments.
    2. Environment variable (``OPENROUTER_API_KEY`` — api_key only).
    3. TOML config file (discovered via ``NRE_CONFIG`` / ``./nre.toml`` /
       ``~/.config/nre/config.toml``; see :mod:`nre.llm.config`).
    4. Hardcoded defaults shown on each field below.

    Back-compat: when no config file is found, behaviour is identical to
    the previous hardcoded-only defaults.
    """

    openrouter_config_defaults = _get_open_router_setting_parameters()

    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", "").strip()
    )
    base_url: str = field(default=str(openrouter_config_defaults["base_url"]))
    model: str = field(default=str(openrouter_config_defaults["model"]))
    temperature: float = field(default=float(openrouter_config_defaults["temperature"]))
    max_tokens: int = field(default=int(openrouter_config_defaults["max_tokens"]))
    max_retries: int = field(default=int(openrouter_config_defaults["max_retries"]))
    timeout_seconds: float = field(default=float(openrouter_config_defaults["timeout_seconds"]))
    http_referer: str = "https://github.com/CoreThink-AI/neurosymbolic-reasoning-engine"
    x_title: str = "CoreThink Neurosymbolic Reasoning Engine"
    # OpenRouter ``extra_body``; default pins ``provider.order`` to direct
    # MiniMax FP8 unless overridden by the TOML config file. Pass ``None``
    # for auto-routing.
    extra_body: dict[str, Any] | None = field(
        default_factory=lambda _v=openrouter_config_defaults["extra_body"]: dict(_v)
    )


class OpenRouterClient:
    def __init__(self, settings: OpenRouterSettings | None = None) -> None:
        self._settings = settings or OpenRouterSettings()
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            key = self._settings.api_key
            if not key:
                raise OpenRouterError(
                    "Missing API key: set OPENROUTER_API_KEY or OpenRouterSettings.api_key"
                )
            headers = {
                "HTTP-Referer": self._settings.http_referer,
                "X-Title": self._settings.x_title,
            }
            self._client = OpenAI(
                api_key=key,
                base_url=self._settings.base_url,
                timeout=httpx.Timeout(self._settings.timeout_seconds),
                max_retries=0,
                default_headers=headers,
            )
        return self._client

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        response_model: type[T] | None = None,
        max_tokens: int | None = None,
    ) -> T | str:
        client = self._get_client()
        send_messages = list(messages)
        budget = self._settings.max_tokens if max_tokens is None else max_tokens
        budget = max(_MIN_MAX_TOKENS, min(int(budget), _MAX_MAX_TOKENS))

        if response_model is not None:
            schema = response_model.model_json_schema()
            send_messages = [
                {
                    "role": "system",
                    "content": (
                        "You must respond with valid JSON that matches this schema exactly. "
                        "Do not include any text outside the JSON object.\n\n"
                        f"JSON Schema:\n{json.dumps(schema, indent=2)}"
                    ),
                },
                *send_messages,
            ]

        last_err: Exception | None = None
        for attempt in range(self._settings.max_retries):
            create_kw: dict[str, Any] = {
                "model": self._settings.model,
                "messages": send_messages,
                "temperature": self._settings.temperature,
                "max_tokens": budget,
            }
            if response_model is not None:
                create_kw["response_format"] = {"type": "json_object"}
            if self._settings.extra_body:
                create_kw["extra_body"] = dict(self._settings.extra_body)
            # For structured-JSON calls (response_model set), reasoning must be
            # disabled — reasoning models burn the entire completion budget on
            # chain-of-thought and never emit the JSON payload. The base-model
            # call (chat_completions_create) does NOT hit this code path, so
            # reasoning stays on for tool-calling completions as configured.
            if response_model is not None:
                eb = create_kw.get("extra_body") or {}
                eb["reasoning"] = {"enabled": False}
                create_kw["extra_body"] = eb
            _log_chat_completion_request(
                "OpenRouterClient.chat",
                attempt,
                self._settings.max_retries,
                create_kw,
            )
            try:
                resp, http = _chat_completion_create_with_http(client, **create_kw)
                ch = _response_choices(resp)
                if not ch:
                    raise OpenRouterError(
                        "OpenRouter returned no choices (empty or null); "
                        f"HTTP {http.status_code}; "
                        f"id={getattr(resp, 'id', None)!r} "
                        f"model={getattr(resp, 'model', None)!r}; "
                        f"body_preview={_http_body_preview(http)!r}"
                    )
                msg0 = getattr(ch[0], "message", None)
                content = _assistant_message_text(msg0)
                if not content.strip():
                    alt = _json_from_tool_call_arguments(msg0)
                    if alt:
                        content = alt
                if response_model is None:
                    return content
                finish = getattr(ch[0], "finish_reason", None)
                refusal = getattr(msg0, "refusal", None) if msg0 is not None else None
                if not content.strip():
                    ptok, ctok = _usage_tokens(resp)
                    err = OpenRouterParseError(
                        "Empty assistant content for structured response "
                        f"(finish_reason={finish!r}, refusal={refusal!r}); "
                        f"usage.prompt_tokens={ptok!r} usage.completion_tokens={ctok!r}; "
                        f"HTTP {http.status_code}; id={getattr(resp, 'id', None)!r}; "
                        f"body_preview={_http_body_preview(http)!r}"
                    )
                    if attempt < self._settings.max_retries - 1:
                        time.sleep(min(0.5 * (2**attempt), 4.0))
                        last_err = err
                        continue
                    raise err
                try:
                    data = json.loads(_strip_json_fence(content))
                    return response_model.model_validate(data)
                except (json.JSONDecodeError, ValueError) as e:
                    raise OpenRouterParseError(
                        f"Invalid JSON or schema: {e}\nContent: {content[:800]!r}\n"
                        f"finish_reason={finish!r} refusal={refusal!r}"
                    ) from e
            except OpenRouterParseError as e:
                # Schema mismatch or empty content is usually a transient
                # provider hiccup — the model returns '{}' once and a valid
                # object on the retry. Burn an attempt and try again; only
                # bubble up after the last attempt.
                if attempt < self._settings.max_retries - 1:
                    time.sleep(min(0.5 * (2**attempt), 4.0))
                    last_err = e
                    continue
                raise
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "429" in msg or "rate" in msg:
                    time.sleep(2**attempt)
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {e}") from e

        raise OpenRouterError("OpenRouter max retries exceeded") from last_err

    def chat_completions_create(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = "auto",
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        client = self._get_client()
        kwargs: dict[str, Any] = {
            "model": model or self._settings.model,
            "messages": messages,
            "temperature": (
                self._settings.temperature if temperature is None else temperature
            ),
            "max_tokens": (
                self._settings.max_tokens if max_tokens is None else max_tokens
            ),
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = tool_choice
        if self._settings.extra_body:
            kwargs["extra_body"] = self._settings.extra_body

        last_err: Exception | None = None
        for attempt in range(self._settings.max_retries):
            _log_chat_completion_request(
                "OpenRouterClient.chat_completions_create",
                attempt,
                self._settings.max_retries,
                kwargs,
            )
            try:
                resp, http = _chat_completion_create_with_http(client, **kwargs)
                usage: dict[str, Any] | None = None
                if resp.usage is not None:
                    usage = {
                        "prompt_tokens": resp.usage.prompt_tokens or 0,
                        "completion_tokens": resp.usage.completion_tokens or 0,
                        "total_tokens": resp.usage.total_tokens or 0,
                    }
                choices_raw = _response_choices(resp)
                if not choices_raw:
                    raise OpenRouterError(
                        "OpenRouter returned no choices (empty or null); "
                        f"HTTP {http.status_code}; "
                        f"id={getattr(resp, 'id', None)!r} "
                        f"model={getattr(resp, 'model', None)!r}; "
                        f"body_preview={_http_body_preview(http)!r}"
                    )
                choices_out: list[dict[str, Any]] = []
                for i, c in enumerate(choices_raw):
                    msg = getattr(c, "message", None)
                    md = _choice_message_to_dict(msg)
                    choices_out.append(
                        {
                            "index": i,
                            "finish_reason": getattr(c, "finish_reason", None),
                            "message": md,
                        }
                    )
                return {
                    "id": resp.id,
                    "model": resp.model,
                    "choices": choices_out,
                    "usage": usage,
                }
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                if "429" in msg or "rate" in msg:
                    time.sleep(2**attempt)
                    continue
                raise OpenRouterError(f"OpenRouter request failed: {e}") from e

        raise OpenRouterError("OpenRouter max retries exceeded") from last_err