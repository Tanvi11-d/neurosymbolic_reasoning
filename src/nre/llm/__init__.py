"""OpenRouter client (OpenAI-compatible SDK) for kernel JSON and server chat.

- **Transport:** :class:`OpenRouterClient`, :class:`OpenRouterSettings`.
- **Kernel LLM calls:** :mod:`nre.library.hybrid` (DECOMPOSE, ORDER, CHECK, MATCH).
- **Structured output:** ``client.chat(..., response_model=...)`` uses ``response_format=json_object``.
- **OpenRouter routing:** default :attr:`OpenRouterSettings.extra_body` sets ``provider.order`` to ``minimax/fp8``; set ``extra_body=None`` for OpenRouter auto-routing, or replace with another ``provider`` dict.
"""

from .openrouter import (
    OpenRouterClient,
    OpenRouterError,
    OpenRouterParseError,
    OpenRouterSettings,
    resolve_classifier_settings,
    resolve_direct_route_settings,
    resolve_tool_completion_settings,
)

__all__ = [
    "OpenRouterClient",
    "OpenRouterError",
    "OpenRouterParseError",
    "OpenRouterSettings",
    "resolve_classifier_settings",
    "resolve_direct_route_settings",
    "resolve_tool_completion_settings",
]
