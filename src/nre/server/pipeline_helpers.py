"""Pure helper functions for the ``/v1/chat/completions`` pipeline.

These were originally inlined in :mod:`nre.server.app`; pulling them out keeps
the FastAPI route handler short and makes the individual helpers easier to
read and test on their own.

What lives here:

- **kernel-run decision** — ``_should_run_kernel`` and the predicates it uses
  (``_last_tool_call_errored``, ``_user_added_new_intent``).
- **loop detection** — ``_loop_breaker_message`` plus its signature collectors
  and the per-tool guidance text.
- **file-write reminder** — ``_file_write_reminder`` plus its task-shape and
  history checks (``_task_requires_file_output``, ``_model_has_written_file``).
- **live-data reminder** — ``_live_data_reminder`` plus its task-shape and
  fallback-detection helpers.
- **error recovery** — ``_error_recovery_message`` keyed off recent tool errors.
- **outgoing-call cleanup** — ``_scrub_host_injected_tool_args`` strips host-
  injected fields the model hallucinated into tool arguments.

Everything is pure: helpers look at the messages they're given and return a
value. No global state, no I/O other than a module-level logger.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

_log = logging.getLogger("nre.pipeline_helpers")


# ---------------------------------------------------------------------------
# Tiny shared accessors — every helper below reads message attributes the
# same way, so pulling these out keeps the call sites readable in English.
# ---------------------------------------------------------------------------


def _role(message: Any) -> str | None:
    """The role of a message (``"user"``, ``"assistant"``, ``"tool"``, ...)."""
    return getattr(message, "role", None)


def _content(message: Any) -> str:
    """The text content of a message, or an empty string when there is none."""
    return getattr(message, "content", None) or ""


def _tool_calls(message: Any) -> list[Any]:
    """The list of tool calls attached to an assistant message (may be empty)."""
    return getattr(message, "tool_calls", None) or []


def _user_message(text: str) -> dict[str, Any]:
    """Build a user-role chat message dict in the OpenAI shape."""
    return {"role": "user", "content": text}


def _find_last_message(messages: list[Any], role: str) -> Any | None:
    """Return the most recent message with the given role, or ``None``."""
    for message in reversed(messages):
        if _role(message) == role:
            return message
    return None


# ---------------------------------------------------------------------------
# Conversation-shape helpers — used to decide whether to re-run the kernel.
# ---------------------------------------------------------------------------


def _count_tool_messages(messages: list[Any]) -> int:
    """How many tool-response messages are already in this conversation."""
    return sum(1 for message in messages if _role(message) == "tool")


# Substrings that mark a ``role=tool`` message as an error. MCP hosts surface
# errors as plain text rather than a structured field, so we match by substring.
_TOOL_ERROR_MARKERS: tuple[str, ...] = (
    "error executing tool",
    "error:",
    "connection closed",
    "rate limit",
    "is not defined",
    "validation error",
    "timed out",
    "timeout",
)


def _last_tool_call_errored(messages: list[Any]) -> bool:
    """Did the most recent ``role=tool`` message look like an error response?

    We use this as a "the previous plan may now be wrong" signal — when a tool
    call comes back with an error string, replanning is worth the extra time
    because the plan probably needs to route around the failure.
    """
    last_tool = _find_last_message(messages, "tool")
    if last_tool is None:
        return False
    content = _content(last_tool).lower()
    return any(marker in content for marker in _TOOL_ERROR_MARKERS)


def _user_added_new_intent(messages: list[Any]) -> bool:
    """Did the user send a fresh message after the most recent tool exchange?

    Most benchmarks (LiveMCPBench included) send one user message at the start
    and then drive everything else via tool/assistant turns. But a general
    OpenAI client may interrupt mid-conversation with a brand-new ask — in
    that case the existing plan is stale and the kernel should replan.

    We treat "new intent" as a user message that appears strictly after the
    last tool-role message. The very first message (index 0) is the original
    opening user request, which doesn't count.
    """
    last_user_index = -1
    last_tool_index = -1
    for index, message in enumerate(messages):
        role = _role(message)
        if role == "user" and _content(message).strip():
            last_user_index = index
        elif role == "tool":
            last_tool_index = index
    if last_user_index <= 0:
        return False
    return last_user_index > last_tool_index


def _should_run_kernel(messages: list[Any], n_tool_turns: int) -> tuple[bool, str]:
    """Decide whether the kernel should run on this turn.

    Returns ``(run, reason)`` — ``reason`` is a short tag for the logs so we
    can see at a glance why we made each choice in production.

    We re-plan in three cases:
      - Turn 1 (no tool exchanges yet) — the kernel produces the initial plan.
      - The user sent a new message after the last tool result — new intent.
      - Last tool call errored — the plan needs to route around the failure.
    Otherwise we trust the existing plan and let the base LLM walk through it.
    """
    if n_tool_turns == 0:
        return True, "first turn"
    if _user_added_new_intent(messages):
        return True, "new user message — replanning for fresh intent"
    if _last_tool_call_errored(messages):
        return True, "last tool errored — refreshing plan"
    return False, "continuation (last tool succeeded)"


# ---------------------------------------------------------------------------
# Loop detection — looks at recent tool calls and nudges the model out of ruts.
# ---------------------------------------------------------------------------


# How many recent tool calls we keep around when checking for repetition loops.
_LOOP_LOOKBACK = 6


def _tool_call_signature(tool_call: Any) -> tuple[str, str] | None:
    """Reduce one tool-call dict to a ``(name, short_args_signature)`` pair.

    The signature normalises whitespace and key order so the same logical
    call rendered slightly differently still compares equal. Returns ``None``
    if the entry isn't a usable tool call (missing name, wrong shape, etc.).
    """
    if not isinstance(tool_call, dict):
        return None
    function = tool_call.get("function") or {}
    name = function.get("name")
    if not name:
        return None

    raw_args = function.get("arguments") or ""
    if isinstance(raw_args, str):
        try:
            parsed_args: Any = json.loads(raw_args)
        except json.JSONDecodeError:
            parsed_args = None
    else:
        parsed_args = raw_args

    if isinstance(parsed_args, dict):
        signature = json.dumps(parsed_args, sort_keys=True)[:200]
    else:
        signature = str(raw_args)[:200]
    return name, signature


def _recent_tool_call_signatures(messages: list[Any]) -> list[tuple[str, str]]:
    """Walk back through the conversation and collect recent tool-call shapes.

    Returns ``(function_name, short_args_signature)`` pairs in newest-first
    order, capped at ``_LOOP_LOOKBACK`` so we only see what the model is
    doing right now. We use this to detect repetition / loop patterns.
    """
    signatures: list[tuple[str, str]] = []
    for message in reversed(messages):
        if _role(message) != "assistant":
            continue
        for tool_call in _tool_calls(message):
            sig = _tool_call_signature(tool_call)
            if sig is not None:
                signatures.append(sig)
        if len(signatures) >= _LOOP_LOOKBACK:
            break
    return signatures


def _looping_on_tool_message(tool_name: str) -> dict[str, Any]:
    """Build the user-role nudge for a detected loop on ``tool_name``."""
    if tool_name == "route":
        return _user_message(
            "You keep calling `route` over and over. Stop discovering. "
            "Call `execute-tool` with `server_name=searxng` (or "
            "`duckduckgo-search`) and a general web-search tool to handle "
            "the remaining information needs. If you already have enough "
            "data, call `submit` right now."
        )
    return _user_message(
        f"You have been calling `{tool_name}` repeatedly without progress. "
        "Switch to a different tool or different server — or call "
        "`submit` with whatever you already have. Don't keep repeating "
        "the same call pattern."
    )


def _loop_breaker_message(messages: list[Any]) -> dict[str, Any] | None:
    """If the model is stuck repeating itself, inject a tactical nudge.

    Two conservative patterns we watch for:
      1. Same tool + same args twice in a row — verbatim repeat.
      2. Same tool name three+ times in a row — consecutive-name loop.

    We deliberately keep this conservative. An earlier version also fired on
    "dominance within the last 10 calls" but that triggered too easily on
    legitimate tasks where the model needs to call ``route`` a few times in
    sequence to find the right server — the false positives hurt more than
    the extra catches helped.
    """
    recent = _recent_tool_call_signatures(messages)
    if len(recent) < 2:
        return None

    # Pattern 1: same tool + same args (verbatim repeat).
    if recent[0] == recent[1]:
        repeated_name = recent[0][0]
        return _user_message(
            f"You just called `{repeated_name}` with the same arguments twice "
            "in a row. That call did not work the first time and it won't "
            "work the second. Pick a different server, change the arguments "
            "meaningfully, or call `submit` with what you already have."
        )

    # Pattern 2: same tool name three+ times in a row.
    if len(recent) >= 3:
        last_three_names = [name for name, _ in recent[:3]]
        if len(set(last_three_names)) == 1:
            return _looping_on_tool_message(last_three_names[0])

    return None


# ---------------------------------------------------------------------------
# File-write reminder — fires when the user asked for a file output but the
# model hasn't actually written it yet.
# ---------------------------------------------------------------------------


# Phrases in the user's original message that signal "this task expects a
# file on disk at the end". Combined with a known file extension, they
# trigger the file-write reminder.
_FILE_OUTPUT_PATH_MARKERS: tuple[str, ...] = (
    "/root/",
    "save to",
    "save the",
    "write to",
    "save it to",
)
_FILE_OUTPUT_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".xlsx",
    ".docx",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".xls",
)
# Substrings we expect to see in a tool's name or arguments when the model is
# actually writing a file. If none show up, the model is about to submit a
# text-only answer for what is really a file-output task.
_FILE_WRITE_TOOL_HINTS: tuple[str, ...] = (
    "write_file",
    "write-file",
    "save_file",
    "save-file",
    "create_file",
    "write_excel",
    "write_pdf",
    "write_docx",
    "write_markdown",
    "filesystem",
)

# Pre-compiled regex for pulling a ``/root/...`` file path out of the user's
# message. Built once at import time so each call to ``_file_path_from_iota``
# doesn't pay the regex-compile cost.
_FILE_PATH_PATTERN = re.compile(
    r"(/root/[^\s\"'`)]+?(?:"
    + "|".join(re.escape(ext) for ext in _FILE_OUTPUT_EXTENSIONS)
    + r"))"
)


def _task_requires_file_output(iota: str) -> bool:
    """Does the user's original request ask for a file to be written?"""
    text = (iota or "").lower()
    has_path_intent = any(marker in text for marker in _FILE_OUTPUT_PATH_MARKERS)
    has_file_extension = any(ext in text for ext in _FILE_OUTPUT_EXTENSIONS)
    return has_path_intent and has_file_extension


def _model_has_written_file(messages: list[Any]) -> bool:
    """Did any prior assistant tool call look like a filesystem write?"""
    for message in messages:
        if _role(message) != "assistant":
            continue
        for tool_call in _tool_calls(message):
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function") or {}
            args_lower = (function.get("arguments") or "").lower()
            if any(hint in args_lower for hint in _FILE_WRITE_TOOL_HINTS):
                return True
    return False


def _file_path_from_iota(iota: str) -> str | None:
    """Pull the first ``/root/...`` file path out of the user's message so we
    can name it back to the model in the reminder. Returns ``None`` when no
    path in the message looks like an explicit file output target.
    """
    if not iota:
        return None
    match = _FILE_PATH_PATTERN.search(iota)
    return match.group(1) if match else None


def _file_write_reminder(iota: str, messages: list[Any]) -> dict[str, Any] | None:
    """When the user asked for a file output but the model hasn't written it
    yet, inject a short reminder on every continuation turn.

    The previous version only fired at "about to submit" moments, but in
    benchmarks like LiveMCPBench the task ends the instant the model calls
    ``submit`` — there is no further turn for us to react on. So we keep the
    reminder short and persistent: it appears every turn until the model
    actually calls a filesystem-writing tool, which is the only signal that
    the requested file is on disk.
    """
    if not _task_requires_file_output(iota):
        return None
    if _model_has_written_file(messages):
        return None
    if _count_tool_messages(messages) == 0:
        # Model hasn't started yet; nothing to remind about.
        return None

    target_path = _file_path_from_iota(iota) or "the requested path"
    return _user_message(
        f"Reminder: the user asked you to save output to {target_path}. "
        "You have not called a filesystem-writing tool yet — the grader "
        "checks that the file exists on disk and a text-only answer will "
        "fail. Before you call `submit`, call `execute-tool` with "
        "`server_name=filesystem` and `tool_name=write_file` (or another "
        "server that writes the requested file format)."
    )


# ---------------------------------------------------------------------------
# Error recovery — turns specific tool-error patterns into next-call guidance.
# ---------------------------------------------------------------------------


# Each entry maps a substring we look for in a tool-error string to the
# guidance we want to inject next turn.
_ERROR_SWITCH_GUIDANCE: tuple[tuple[str, str], ...] = (
    (
        "ddg detected",
        "The DuckDuckGo search just refused as rate-limited. Stop using "
        "`duckduckgo-search`. Switch to `execute-tool` with "
        "`server_name=searxng` and use its web-search tool for the rest of "
        "this task.",
    ),
    (
        "you are likely making requests too quickly",
        "The DuckDuckGo search is rate-limiting. Switch to "
        "`server_name=searxng` for the remaining web searches.",
    ),
    (
        "is not defined in the configuration",
        "The server you just tried does not exist in the registry. Stop "
        "guessing server names. Call `route` once with a clear query to "
        "discover what is actually available, then use only the names that "
        "`route` returns.",
    ),
)


def _error_recovery_message(messages: list[Any]) -> dict[str, Any] | None:
    """If the most recent tool result is an error we know how to respond to,
    inject specific guidance for the next call."""
    last_tool = _find_last_message(messages, "tool")
    if last_tool is None:
        return None
    content = _content(last_tool).lower()
    if not content:
        return None
    for marker, guidance in _ERROR_SWITCH_GUIDANCE:
        if marker in content:
            return _user_message(guidance)
    return None


# ---------------------------------------------------------------------------
# Live-data reminder — fires when the user asked for current / today's info
# but the model is likely to answer from training data.
# ---------------------------------------------------------------------------


# Words in the user's request that signal "this task expects fresh,
# present-time information". When any appear, answering from training data
# alone will almost certainly be wrong by the grader's standard.
_LIVE_DATA_MARKERS: tuple[str, ...] = (
    "today",
    "today's",
    "todays",
    "latest",
    "currently",
    "current",
    "right now",
    "this week",
    "this month",
    "this year",
    "recent",
    "recently",
    "now",
    "tonight",
    "yesterday",
    "live ",
    "real-time",
    "realtime",
)

# Phrases the model uses when it has fallen back to training data instead of
# fetching live info. Seeing one in the model's last reply is our cue to
# push it toward search tools.
_TRAINING_FALLBACK_MARKERS: tuple[str, ...] = (
    "training data",
    "knowledge cutoff",
    "as of my knowledge",
    "based on my knowledge",
    "i don't have access to real-time",
    "i don't have access to live",
    "i cannot access live",
)


def _task_needs_live_data(iota: str) -> bool:
    """Does the user's request expect fresh, present-time information?

    False positives are cheap (one extra reminder); missing a real live-data
    task tends to let the model fall back to training data, which the grader
    scores as a fail.
    """
    text = (iota or "").lower()
    return any(marker in text for marker in _LIVE_DATA_MARKERS)


def _last_assistant_content(messages: list[Any]) -> str:
    """Plain text of the most recent assistant message (empty when missing)."""
    last_assistant = _find_last_message(messages, "assistant")
    if last_assistant is None:
        return ""
    return _content(last_assistant)


def _live_data_reminder(iota: str, messages: list[Any]) -> dict[str, Any] | None:
    """Push the model to use live search when the task wants current info.

    Fires when both of these hold:
      - The user's request mentions a temporal "current/today" marker, AND
      - The model's last assistant turn admitted to falling back to training
        data (says "as of my knowledge cutoff…", "based on my training data",
        and similar).

    We deliberately stay quiet on every other turn — nagging when the model
    isn't visibly off-track just bloats the prompt.
    """
    if not _task_needs_live_data(iota):
        return None
    if _count_tool_messages(messages) == 0:
        # Turn 1 — the system prompt already covers this; no need to repeat.
        return None

    last_text = _last_assistant_content(messages).lower()
    is_falling_back = any(
        marker in last_text for marker in _TRAINING_FALLBACK_MARKERS
    )
    if not is_falling_back:
        return None

    return _user_message(
        "Reminder: the user asked for current / today's information. "
        "Don't answer from your training data — that data will be wrong "
        "for the user's request. Use `execute-tool` with "
        "`server_name=searxng` (or `duckduckgo-search`) and a clear "
        "query for the topic. If one search server returns nothing, try "
        "a different query or the alternate server before giving up."
    )


# ---------------------------------------------------------------------------
# Outgoing-call cleanup — strips host-injected fields from the model's args.
# ---------------------------------------------------------------------------


# Argument keys the host fills in automatically (request context, etc.). The
# model occasionally hallucinates these into ``arguments`` as garbage strings
# like ``"ctx": "{}"`` or ``"ctx": "null"``, which crashes the host. We strip
# them on the way out so the host only ever sees fields it expects to see.
_HOST_INJECTED_ARG_KEYS = ("ctx", "context", "_ctx", "request_context")


def _scrub_one_tool_call(call: Any) -> None:
    """Strip host-injected keys from a single tool call, in place.

    A "call" is the OpenAI tool-call shape: ``{"function": {"arguments": ...}}``.
    We only touch arguments that parse cleanly as JSON dicts — malformed
    payloads are left alone so the host can surface the underlying parse
    error itself.
    """
    if not isinstance(call, dict):
        return
    function = call.get("function")
    if not isinstance(function, dict):
        return
    raw_args = function.get("arguments")
    if not isinstance(raw_args, str) or not raw_args.strip():
        return
    try:
        parsed_args = json.loads(raw_args)
    except json.JSONDecodeError:
        return
    if not isinstance(parsed_args, dict):
        return

    removed_keys = [key for key in _HOST_INJECTED_ARG_KEYS if key in parsed_args]
    if not removed_keys:
        return

    for key in removed_keys:
        parsed_args.pop(key, None)
    function["arguments"] = json.dumps(parsed_args)
    _log.info(
        "scrubbed host-injected args from tool_call name=%s removed=%s",
        function.get("name"),
        removed_keys,
    )


def _scrub_host_injected_tool_args(tool_calls: list[dict[str, Any]] | None) -> None:
    """Strip host-injected argument keys (``ctx``, ``context``, etc.) from the
    LLM's tool calls before we hand them back to the caller.

    The model sometimes guesses values for fields the host fills automatically,
    producing payloads like ``{"ctx": "{}", "query": "..."}`` that crash the
    MCP host with "Context is not available outside of a request".
    """
    if not tool_calls:
        return
    for call in tool_calls:
        _scrub_one_tool_call(call)
