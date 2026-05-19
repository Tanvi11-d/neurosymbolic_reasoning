"""Synthesize OpenAI-style ``prior_turns`` entries from ``agent_turn`` output."""

from __future__ import annotations

import json
from typing import Any


def kernel_turn_to_chat_messages(
    *,
    iota: str,
    raw_turn: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build messages for one completed turn to **append** to cumulative ``prior_turns``.

    Does not alter primitive APIs — integrators merge the result into the next request's
    ``prior_turns`` or chat ``messages``.
    """
    msgs: list[dict[str, Any]] = [{"role": "user", "content": iota}]
    lines: list[str] = []
    for st in raw_turn.get("steps") or []:
        if not isinstance(st, dict) or st.get("phase") != "match":
            continue
        fn = st.get("function")
        status = st.get("status")
        if not fn:
            continue
        try:
            b = json.dumps(st.get("bindings"), default=str)
        except (TypeError, ValueError):
            b = str(st.get("bindings"))
        line = f"- {fn} status={status} bindings={b}"
        if st.get("status") == "success" and "result" in st:
            try:
                r = json.dumps(st["result"], default=str)
            except (TypeError, ValueError):
                r = str(st["result"])
            if len(r) > 1800:
                r = r[:1797] + "..."
            line += f" result={r}"
        lines.append(line)
    if lines:
        msgs.append({
            "role": "assistant",
            "content": "Kernel match/execute summary:\n" + "\n".join(lines),
        })
    return msgs


def extend_prior_turns_with_kernel_result(
    prior_turns: list[dict[str, Any]] | None,
    *,
    iota: str,
    raw_turn: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return ``prior_turns`` (copy) plus messages derived from *(iota, raw_turn)*."""
    base = list(prior_turns) if prior_turns else []
    base.extend(kernel_turn_to_chat_messages(iota=iota, raw_turn=raw_turn))
    return base
