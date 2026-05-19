"""Build the system message we send to the base LLM after the kernel runs.

The kernel produces a structured reasoning trace; we glue that onto a system
prompt with a little guidance on how to use it, then prepend the whole thing
to the conversation before the upstream call.
"""

from __future__ import annotations

import json
import logging
from typing import Any

_logger = logging.getLogger("nre.trace")

# Kept deliberately short. Every bullet here exists because we watched it fail
# in a benchmark run — adding more rules just pushes the actual task out of
# the model's attention.
_TRACE_INSTRUCTIONS = """A neurosymbolic kernel has already pre-processed the user's request. Below you'll find the kernel's reasoning trace (task DAG, scratchpad, per-task match results) and then the original conversation. Use the trace to decide what to do.

If a step in the trace is `"status": "bound"` or `"status": "success"`, just call that tool with the arguments the kernel already resolved — don't reach for a generic alternative when a specific one was planned.

If the kernel didn't bind a tool but you still need outside information, fall back to web search. `searxng` is much more reliable than `duckduckgo-search` (DDG rate-limits hard). If one search server returns nothing useful, try a different query or a different server before concluding the task can't be done — at least two attempts before you give up.

Don't camp on `route`. If a couple of `route` calls in a row don't reveal a clearly relevant server, stop discovering and just use `searxng` / `duckduckgo-search` for the topic. The working-time budget is finite — keep tool calls focused, and always finish with `submit`.

When the task says something like "save to /root/...", "write to file", or "generate a PDF/xlsx/docx at <path>", you have to actually call a filesystem write tool after gathering the content. Describing what you would save doesn't count. If the obvious server isn't in the registry, call `route` with a query like "write file", "save xlsx", "create pdf", "filesystem", "office" — there is almost always a way to produce the file (filesystem servers, office servers, conversion servers). Don't conclude "no tool can do this" without at least one targeted `route` for the write step.

When the task is a recommendation / listing request ("recommend a gaming PC", "find cheap goodies", "give me the latest prices"), even if no dedicated shopping/commerce server exists, you can still answer using `searxng` or `duckduckgo-search` to gather current info. Don't surrender with "no shopping tool available" — web search is the fallback.

Use the file path exactly as the user wrote it. Don't rename or invent variations.

Your final answer comes from tool results that actually came back in this conversation, not from your training data. Don't write "saved successfully" or "found N items" unless a tool returned that.

Two host quirks for the meta-tools (`route`, `execute-tool`):

The `ctx` parameter is filled in by the host — leave it out of your `arguments` entirely. Setting `"ctx": "{}"` or `"ctx": "null"` crashes the call with "Context is not available outside of a request".

For tools that take a `server_name`, only use values you've actually seen returned by a prior `route` call. Names like `search`, `web`, `maps`, `audio_server`, `web_scraper` aren't real servers and will be rejected — if you don't know the right server yet, call `route` first.

When tools are needed, respond with `tool_calls`. When a plain answer is enough, respond with `content`."""


def build_kernel_augmented_messages(
    trace: dict[str, Any] | None,
    conversation: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Wrap the kernel trace in a system message and prepend it to the conversation.

    On the first turn ``trace`` carries the kernel's fresh reasoning output. On
    continuation turns it can be empty/None — we skip the trace block but keep
    the same guidance so the model's behaviour stays consistent across turns.

    Any system messages the client sent are folded in at the end so the caller's
    own setup wins on conflicts with our defaults.
    """
    system_body = _TRACE_INSTRUCTIONS
    if trace:
        trace_block = json.dumps(trace, indent=2, default=str)
        system_body += "\n\n## Kernel reasoning trace\n" + trace_block

    client_system_lines: list[str] = []
    other_messages: list[dict[str, Any]] = []
    for message in conversation:
        if message.get("role") == "system" and message.get("content"):
            client_system_lines.append(str(message["content"]))
        else:
            other_messages.append(dict(message))

    if client_system_lines:
        system_body += (
            "\n\n## Additional instructions from the client\n"
            + "\n\n".join(client_system_lines)
        )

    trace_tasks = 0
    if isinstance(trace, dict):
        trace_tasks = len(trace.get("decompose", {}).get("turns", []))
    _logger.info(
        "built augmented messages (system_chars=%s convo_messages=%s trace_tasks=%s)",
        len(system_body),
        len(other_messages),
        trace_tasks,
    )
    return [{"role": "system", "content": system_body}, *other_messages]
