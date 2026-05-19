"""Hybrid primitive library — tool-calling kernel stages and demo implementations.

Kernel: DECOMPOSE, ORDER, CHECK, MATCH (OpenRouter helpers). Demo: Embedder, Classifier, …
Structured I/O: :mod:`nre.schemas`. Transport: :mod:`nre.llm.openrouter`.
"""

from __future__ import annotations

import json
import math
import random
from typing import Any, Callable, Sequence

from nre.llm.openrouter import OpenRouterClient
from nre.schemas import (
    LLMCheckRemediationOutput,
    LLMDecomposeOutput,
    LLMMatchFillParams,
    LLMMatchFuncOutput,
    LLMSiblingOrderOutput,
)
from nre.utils.tool_registry import (
    canonicalize_bindings_with_registry,
    format_tool_schemas,
    tool_registry_to_schemas,
    tools_registry_metadata,
)

from nre.primitives.base import HybridPrimitive, Primitive, PrimitiveLog
from nre.primitives.deterministic import Fact, UnitTask
from nre.primitives.llm_bridge import llm_task_item_to_unit_task


def _llm_logging(primitive: Primitive | None) -> tuple[PrimitiveLog | None, bool]:
    """``(log, log_full_bodies)`` for LLM runners — driven by :class:`Primitive` config."""
    if primitive is None or not primitive.log_llm_enabled():
        return None, False
    return primitive.log, primitive.log_bodies_enabled()


def _kernel_stage_max_tokens(primitive: Primitive | None) -> int | None:
    """Optional per-primitive ``config['llm_max_tokens']`` for this kernel LLM call.

    If unset, :class:`nre.llm.openrouter.OpenRouterSettings.max_tokens` applies (raise it on
    the client passed into :func:`build_engine`, or set per-primitive ``llm_max_tokens`` here).
    """
    if primitive is None:
        return None
    v = primitive.config.get("llm_max_tokens")
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def normalize_decompose_result(raw: Any) -> dict[str, Any]:
    """Validate hook output for DECOMPOSE: ``turns``, ``config``, ``tools``."""
    if not isinstance(raw, dict):
        raise TypeError("decompose must return a dict")
    turns_raw = raw.get("turns", [])
    if turns_raw is None:
        turns_raw = []
    if not isinstance(turns_raw, list):
        raise TypeError("decompose result 'turns' must be a list")
    turns: list[UnitTask] = []
    seen: set[str] = set()
    for i, item in enumerate(turns_raw):
        if isinstance(item, UnitTask):
            t = item
        elif isinstance(item, dict):
            tid = item.get("id")
            if not tid or not isinstance(tid, str):
                raise ValueError(f"turns[{i}]: 'id' must be a non-empty string")
            deps = item.get("depends_on", [])
            if isinstance(deps, tuple):
                deps_t = deps
            elif isinstance(deps, list):
                deps_t = tuple(deps)
            else:
                raise TypeError(f"turns[{i}]: 'depends_on' must be a list or tuple")
            text = item.get("text", "")
            if text is not None and not isinstance(text, str):
                raise TypeError(f"turns[{i}]: 'text' must be a string")
            tn = item.get("tool_name", "") or ""
            if tn is not None and not isinstance(tn, str):
                raise TypeError(f"turns[{i}]: 'tool_name' must be a string")
            params = item.get("parameters", {})
            if params is None:
                params = {}
            if not isinstance(params, dict):
                raise TypeError(f"turns[{i}]: 'parameters' must be a dict")
            t = UnitTask(
                id=tid,
                text=text or "",
                depends_on=deps_t,
                tool_name=tn,
                parameters=dict(params),
            )
        else:
            raise TypeError(
                f"turns[{i}]: expected UnitTask or dict, got {type(item).__name__}"
            )
        if t.id in seen:
            raise ValueError(f"duplicate task id: {t.id!r}")
        seen.add(t.id)
        turns.append(t)

    config = raw.get("config", {})
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise TypeError("decompose result 'config' must be a dict or omitted")

    tools = raw.get("tools", {})
    if tools is None:
        tools = {}
    if not isinstance(tools, dict):
        raise TypeError("decompose result 'tools' must be a dict or omitted")

    out: dict[str, Any] = {"turns": turns, "config": config, "tools": tools}
    if "reasoning" in raw and isinstance(raw.get("reasoning"), str):
        out["reasoning"] = raw["reasoning"]
    return out


# ── Kernel LLM prompts + OpenRouter runners (DECOMPOSE / CHECK / ORDER / MATCH) ──

DECOMPOSE_SYSTEM_PROMPT = """You are a task decomposition engine for the CoreThink tool-calling system.

Given a user request, environment configuration (gamma), and the available tools, you must:
1. Break the request into atomic unit tasks, each calling exactly one tool from the registry.
2. Identify dependencies between tasks (which task ids must complete before others).
3. Extract key-value pairs for scratchpad seeding (config_keys) from gamma and explicit user text.
4. For each task, set parameters from: (a) the **current user request**, (b) **gamma**, and (c) when the user message includes **## Prior conversation**, from that section (earlier user turns, tool results, numbers, paths). Do not invent numbers or paths that none of these sources support.

Multi-turn / anaphora (when **## Prior conversation** is present in the user message):
- Phrases like "those same values", "the numbers from before", "using the previous answer", or "as in the last step" refer to **literals already shown** in that prior block (e.g. numeric lists, means, file contents). **Reuse those literals** in tool parameters (e.g. pass the same list to ``standard_deviation`` or ``mean``).
- Do **not** return an empty task list solely because the current sentence omits digits if the prior block contains the values you need.

Writing summaries to files (e.g. ``echo``):
- If the user asks to write "results", "summary", or "analysis" to a file, ``content`` must be the **substantive computed text** from earlier steps (e.g. ``Mean: ...`` and ``Standard Deviation: ...`` on **separate lines** when reporting multiple statistics), not the literal English word "results" unless they explicitly asked for that exact string.

Incremental decomposition (“baby steps”):
- Prefer **small, ordered steps** with clear ``depends_on`` links instead of one leap that assumes the filesystem is already in the right shape.
- **Observation before mutation** when layout is not already known: if **gamma** (or **## Prior conversation**) does **not** give a reliable current-directory listing, working path, or explicit canonical paths for what the user described, start with **discovery** tasks the registry actually provides (commonly ``pwd`` and/or ``ls`` — use the **exact** tool and parameter names from the tool list). Place them **before** ``cd``, ``mkdir``, ``touch``, ``cp``, etc. that depend on “what exists here”.
- After discovery tasks in the same plan, **cd** / path parameters should reflect **what could be observed**: tie later tasks to earlier ones via ``depends_on`` so order is: see what’s there → move/create as needed → operate on files. Do **not** skip discovery just to shorten the task list when the user only gave colloquial folder wording.
- If **gamma** or prior tool results already state cwd contents or paths, **do not** add redundant listing steps; use that state in parameters.

Rules:
- Use EXACT tool_name and parameter names from the tool list.
- When a parameter has **allowed** values listed (e.g. ``allowed: 'Buy', 'Sell'``), use one of those **exact** strings — same spelling and capitalization as shown (simulators often compare literally).
- depends_on must form a DAG (no cycles).
- Task ids must be unique (t1, t2, ... or similar).
- If prior conversation context is provided, use it for state (files moved, etc.) and for numeric or path parameters as above.

Filesystem / working-directory discipline (Gorilla-style tools: paths are usually relative to the **current working directory**):
- After you are in the right directory, use **short relative paths** in later tasks (e.g. ``touch`` with a file name only, not long prefixed paths) when that matches the cwd you established.
- Do not invent path segments; only use folders/files the user, **gamma**, or **prior conversation** supports — or that follow naturally **after** explicit discovery tasks in your plan.
- Keep the plan **minimal but complete**: fewest tasks that still respect observation-first ordering when the environment is underspecified.
"""

_DECOMPOSE_PRIOR_TURNS_MAX_CHARS_DEFAULT = 24_000


def format_prior_turns_for_decompose(
    prior_turns: list[dict[str, Any]] | None,
    *,
    max_chars: int | None = None,
) -> str:
    """Render OpenAI-style message dicts for the DECOMPOSE user prompt (bounded size)."""
    if not prior_turns:
        return ""
    budget = (
        max_chars if max_chars is not None else _DECOMPOSE_PRIOR_TURNS_MAX_CHARS_DEFAULT
    )
    lines: list[str] = []
    for i, m in enumerate(prior_turns):
        if not isinstance(m, dict):
            continue
        role = m.get("role", "?")
        bits: list[str] = [f"[{i}] role={role}"]
        c = m.get("content")
        if c is not None and str(c).strip():
            try:
                bits.append("content=" + json.dumps(str(c), ensure_ascii=False))
            except (TypeError, ValueError):
                bits.append(f"content={str(c)!r}")
        tc = m.get("tool_calls")
        if tc:
            try:
                tc_s = json.dumps(tc, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                tc_s = str(tc)
            if len(tc_s) > 4000:
                tc_s = tc_s[:3997] + "..."
            bits.append("tool_calls=" + tc_s)
        if m.get("tool_call_id"):
            bits.append(f"tool_call_id={m['tool_call_id']!r}")
        if m.get("name"):
            bits.append(f"name={m['name']!r}")
        lines.append(" | ".join(bits))
    text = "\n".join(lines)
    if len(text) > budget:
        omit = len(text) - budget + 20
        text = f"… ({omit} chars omitted)\n" + text[omit:]
    return text


def build_decompose_user_prompt(
    iota: str,
    gamma: dict[str, Any],
    tools_registry: dict[str, Any],
    *,
    conversation_summary: str = "",
    prior_turns: list[dict[str, Any]] | None = None,
    prior_turns_max_chars: int | None = None,
) -> str:
    blocks: list[str] = []
    hist = format_prior_turns_for_decompose(
        prior_turns, max_chars=prior_turns_max_chars
    )
    if hist:
        blocks.append("## Conversation history (prior turns)\n" + hist)
    if conversation_summary:
        blocks.append(
            "## Prior conversation (tools already run)\n" + conversation_summary
        )
    blocks.append(f"## Current user request\n{iota}")
    schemas = tool_registry_to_schemas(tools_registry)
    if schemas:
        blocks.append(
            "## Available tools (EXACT names and parameter names; * = required)\n"
            + format_tool_schemas(schemas)
        )
    else:
        blocks.append("## Available tools\n(none registered — infer no tool calls)")
    blocks.append(
        "## Environment (gamma)\n```json\n"
        + json.dumps(gamma, indent=2, default=str)
        + "\n```"
    )
    blocks.append(
        "Decompose into tasks and config_keys. "
        "Return JSON matching the response schema exactly."
    )
    return "\n\n".join(blocks)


def run_llm_decompose(
    client: OpenRouterClient,
    iota: str,
    gamma: dict[str, Any],
    tools_registry: dict[str, Any],
    *,
    conversation_summary: str = "",
    prior_turns: list[dict[str, Any]] | None = None,
    prior_turns_max_chars: int | None = None,
    primitive: Primitive | None = None,
) -> dict[str, Any]:
    user_content = build_decompose_user_prompt(
        iota,
        gamma,
        tools_registry,
        conversation_summary=conversation_summary,
        prior_turns=prior_turns,
        prior_turns_max_chars=prior_turns_max_chars,
    )
    messages = [
        {"role": "system", "content": DECOMPOSE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    plog, bodies = _llm_logging(primitive)
    if plog is not None:
        plog.info(
            "LLM DECOMPOSE request: iota_chars=%s gamma_keys=%s tool_names=%s "
            "conv_summary_chars=%s prior_turns_msgs=%s",
            len(iota),
            sorted(gamma.keys()),
            sorted(tools_registry.keys()) if isinstance(tools_registry, dict) else 0,
            len(conversation_summary or ""),
            len(prior_turns or []),
        )
        if bodies:
            plog.debug("LLM DECOMPOSE system: %s", DECOMPOSE_SYSTEM_PROMPT)
            plog.debug("LLM DECOMPOSE user: %s", user_content)
    parsed = client.chat(
        messages,
        response_model=LLMDecomposeOutput,
        max_tokens=_kernel_stage_max_tokens(primitive),
    )
    assert isinstance(parsed, LLMDecomposeOutput)
    raw = parsed.to_raw_decompose_dict()
    if plog is not None:
        turns = raw.get("turns") if isinstance(raw, dict) else []
        n_tasks = len(turns) if isinstance(turns, list) else 0
        cfg = raw.get("config") if isinstance(raw, dict) else {}
        extra_tools = raw.get("tools") if isinstance(raw, dict) else {}
        plog.info(
            "LLM DECOMPOSE response: task_count=%s config_keys=%s decompose_tool_keys=%s reasoning_chars=%s",
            n_tasks,
            sorted(cfg.keys()) if isinstance(cfg, dict) else [],
            sorted(extra_tools.keys()) if isinstance(extra_tools, dict) else [],
            len(raw.get("reasoning", "") or "") if isinstance(raw, dict) else 0,
        )
        if bodies:
            try:
                payload = json.dumps(raw, indent=2, default=str)
            except (TypeError, ValueError):
                payload = repr(raw)
            plog.debug("LLM DECOMPOSE parsed JSON: %s", payload)
    return raw


CHECK_REMEDIATION_SYSTEM_PROMPT = """You assist the CoreThink CHECK-PREREQUISITES sub-process.

For a **main task** waiting on a symbolic prerequisite key in scratchpad ``S``:
- **absent**: the key is missing from ``S`` (needs retrieval or a setup action).
- **not_met**: the key is present but the predicate is false (needs a remediation action).

Propose **at most one** small **unit task** (single tool from the registry) that would plausibly advance CHECK toward **Met** for that key. Use only tools and parameter names from the registry. If no sensible tool exists, set task to null.

Return JSON matching the schema exactly."""


def build_check_remediation_user_prompt(
    *,
    prereq_key: str,
    outcome: str,
    task: UnitTask,
    scratchpad: dict[str, Any],
    tools_registry: dict[str, Any],
) -> str:
    schemas = tool_registry_to_schemas(tools_registry)
    tool_block = (
        format_tool_schemas(schemas)
        if schemas
        else "(no tools registered — return task: null)"
    )
    blocks = [
        f"## Prerequisite outcome\n- key: `{prereq_key}`\n- outcome: `{outcome}` "
        f"(absent = missing in S; not_met = present but predicate false)",
        f"## Main task\n- id: `{task.id}`\n- text: {task.text!r}\n"
        f"- tool_name: {task.tool_name!r}\n- depends_on: {list(task.depends_on)}",
        "## Current scratchpad S (subset)\n```json\n"
        + json.dumps(scratchpad, indent=2, default=str)
        + "\n```",
        "## Available tools\n" + tool_block,
        "Propose remediation `task` or null.",
    ]
    return "\n\n".join(blocks)


def run_llm_prereq_remediation(
    client: OpenRouterClient,
    *,
    prereq_key: str,
    outcome: str,
    task: UnitTask,
    scratchpad: dict[str, Any],
    tools_registry: dict[str, Any],
    primitive: Primitive | None = None,
) -> UnitTask | None:
    user = build_check_remediation_user_prompt(
        prereq_key=prereq_key,
        outcome=outcome,
        task=task,
        scratchpad=scratchpad,
        tools_registry=tools_registry,
    )
    messages = [
        {"role": "system", "content": CHECK_REMEDIATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    plog, bodies = _llm_logging(primitive)
    if plog is not None:
        plog.info(
            "LLM CHECK_REMEDIATION request: key=%s outcome=%s task_id=%s scratchpad_keys=%s registry_tools=%s",
            prereq_key,
            outcome,
            task.id,
            sorted(scratchpad.keys()),
            sorted(tools_registry.keys()) if isinstance(tools_registry, dict) else 0,
        )
        if bodies:
            plog.debug(
                "LLM CHECK_REMEDIATION system: %s", CHECK_REMEDIATION_SYSTEM_PROMPT
            )
            plog.debug("LLM CHECK_REMEDIATION user: %s", user)
    parsed = client.chat(
        messages,
        response_model=LLMCheckRemediationOutput,
        max_tokens=_kernel_stage_max_tokens(primitive),
    )
    assert isinstance(parsed, LLMCheckRemediationOutput)
    if parsed.task is None:
        if plog is not None:
            plog.info(
                "LLM CHECK_REMEDIATION response: task=null (no remediation proposed)"
            )
        return None
    rem = llm_task_item_to_unit_task(parsed.task)
    if plog is not None:
        plog.info(
            "LLM CHECK_REMEDIATION response: remediation id=%s tool_name=%s deps=%s",
            rem.id,
            rem.tool_name,
            list(rem.depends_on),
        )
        if bodies:
            try:
                td = parsed.model_dump()
            except Exception:
                td = str(parsed)
            plog.debug(
                "LLM CHECK_REMEDIATION raw task payload: %s",
                json.dumps(td, default=str),
            )
    return rem


ORDER_SIBLINGS_SYSTEM_PROMPT = """You are a task scheduling assistant for the CoreThink tool-calling kernel (GET-ORDER).

You receive a set of unit tasks that are **mutually ready at the same time**: no task in the set depends on another task in the same set. All dependencies on **earlier** tasks are already satisfied.

Your job is to output a **total ordering** of exactly those task ids (a permutation) that best fits:
- The user's underlying goal and natural workflow (read each task's description and tool hint).
- Any prior conversation or environment context provided.

Hard rules:
- Output **only** task ids from the given batch — same multiset, no extras, no omissions, no duplicates.
- Do **not** invent dependencies; the DAG is fixed in code. You only **rank** siblings.
- If indifferent, use lexicographic order by id.
"""


def build_sibling_order_user_prompt(
    batch: list[str],
    tasks_by_id: dict[str, Any],
    *,
    conversation_summary: str = "",
) -> str:
    lines: list[str] = []
    if conversation_summary:
        lines.append("## Prior context\n" + conversation_summary.strip())
    lines.append("## Tasks in this batch (pick one order to run them)\n")
    for tid in batch:
        t = tasks_by_id[tid]
        text = getattr(t, "text", "") or ""
        tool = getattr(t, "tool_name", "") or ""
        deps = list(getattr(t, "depends_on", ()) or ())
        lines.append(
            f"- **{tid}** tool={tool!r}  depends_on={deps}  description: {text}"
        )
    lines.append(
        "\nReturn JSON matching the schema: `ordered_task_ids` must list every id above exactly once."
    )
    return "\n".join(lines)


def run_llm_sibling_order_batch(
    client: OpenRouterClient,
    batch: list[str],
    tasks_by_id: dict[str, Any],
    *,
    conversation_summary: str = "",
    primitive: Primitive | None = None,
) -> LLMSiblingOrderOutput:
    user = build_sibling_order_user_prompt(
        batch, tasks_by_id, conversation_summary=conversation_summary
    )
    messages = [
        {"role": "system", "content": ORDER_SIBLINGS_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    plog, bodies = _llm_logging(primitive)
    if plog is not None:
        plog.info(
            "LLM SIBLING_ORDER request: batch=%s conv_summary_chars=%s",
            batch,
            len(conversation_summary or ""),
        )
        if bodies:
            plog.debug("LLM SIBLING_ORDER system: %s", ORDER_SIBLINGS_SYSTEM_PROMPT)
            plog.debug("LLM SIBLING_ORDER user: %s", user)
    out = client.chat(
        messages,
        response_model=LLMSiblingOrderOutput,
        max_tokens=_kernel_stage_max_tokens(primitive),
    )
    assert isinstance(out, LLMSiblingOrderOutput)
    if plog is not None:
        perm = list(out.ordered_task_ids)
        ok = validate_sibling_permutation(perm, batch)
        plog.info(
            "LLM SIBLING_ORDER response: ordered=%s valid_permutation=%s",
            perm,
            ok,
        )
        if bodies:
            try:
                od = out.model_dump()
            except Exception:
                od = str(out)
            plog.debug("LLM SIBLING_ORDER parsed: %s", json.dumps(od, default=str))
    return out


def validate_sibling_permutation(ordered: list[str], batch: list[str]) -> bool:
    if len(ordered) != len(batch):
        return False
    return sorted(ordered) == sorted(batch)


MATCH_FUNC_SYSTEM_PROMPT = """You are the CoreThink MATCH-FUNC step.

Given one **unit task** and a **tool registry** (names, signatures, descriptions), choose **at most one**
tool that best implements the task. The tool name must match the registry exactly.

When parameters later list **allowed** values, downstream steps must use those exact literals (case-sensitive).

If no tool is appropriate, return `"tool_name": null`."""

MATCH_FILL_SYSTEM_PROMPT = """You are the CoreThink parameter-completion step for MATCH.

Some required parameters could not be read from the task, turn context, or scratchpad.
Propose **only** values you can justify from the task text, explicit task.parameters,
or the scratchpad JSON. Use exact JSON-serializable values. Omit keys you cannot ground.

If the tool spec lists **allowed** values for a parameter, you **must** output one of those strings exactly (same capitalization)."""


def build_match_func_user_prompt(
    *, task: UnitTask, tools_registry: dict[str, Any]
) -> str:
    meta = tools_registry_metadata(tools_registry)
    schemas = tool_registry_to_schemas(meta)
    tool_block = format_tool_schemas(schemas) if schemas else "(empty registry)"
    blocks = [
        "## Unit task",
        f"- id: `{task.id}`",
        f"- text: {task.text!r}",
        f"- tool_name hint: {task.tool_name!r}",
        f"- depends_on: {list(task.depends_on)}",
        f"- parameters: {json.dumps(dict(task.parameters), default=str)}",
        "## Registry",
        tool_block,
        "Return JSON: chosen `tool_name` or null.",
    ]
    return "\n\n".join(blocks)


def run_llm_match_func(
    client: OpenRouterClient,
    *,
    task: UnitTask,
    tools_registry: dict[str, Any],
    primitive: Primitive | None = None,
) -> str | None:
    plog, bodies = _llm_logging(primitive)
    meta = tools_registry_metadata(tools_registry)
    if not meta:
        if plog is not None:
            plog.info("LLM MATCH_FUNC: skip — empty tool metadata registry")
        return None
    user = build_match_func_user_prompt(task=task, tools_registry=tools_registry)
    messages = [
        {"role": "system", "content": MATCH_FUNC_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    if plog is not None:
        plog.info(
            "LLM MATCH_FUNC request: task_id=%s hint_tool=%s registry_size=%s",
            task.id,
            task.tool_name,
            len(meta),
        )
        if bodies:
            plog.debug("LLM MATCH_FUNC system: %s", MATCH_FUNC_SYSTEM_PROMPT)
            plog.debug("LLM MATCH_FUNC user: %s", user)
    parsed = client.chat(
        messages,
        response_model=LLMMatchFuncOutput,
        max_tokens=_kernel_stage_max_tokens(primitive),
    )
    assert isinstance(parsed, LLMMatchFuncOutput)
    if parsed.tool_name is None:
        if plog is not None:
            plog.info("LLM MATCH_FUNC response: tool_name=null")
        return None
    name = str(parsed.tool_name).strip()
    if name not in meta:
        if plog is not None:
            plog.warning(
                "LLM MATCH_FUNC response: model returned unknown tool %r (not in registry)",
                name,
            )
        return None
    if plog is not None:
        plog.info("LLM MATCH_FUNC response: tool_name=%r", name)
        if bodies:
            try:
                md = parsed.model_dump()
            except Exception:
                md = str(parsed)
            plog.debug("LLM MATCH_FUNC parsed: %s", json.dumps(md, default=str))
    return name


def build_match_fill_user_prompt(
    *,
    task: UnitTask,
    missing_params: list[str],
    scratchpad: dict[str, Any],
    turn_context: dict[str, Any],
    tool_name: str,
    tools_registry: dict[str, Any],
) -> str:
    meta = tools_registry_metadata(tools_registry)
    spec = meta.get(tool_name, {})
    blocks = [
        "## Missing parameters",
        ", ".join(f"`{p}`" for p in missing_params),
        f"## Tool `{tool_name}`",
        json.dumps(spec, indent=2),
        "## Unit task",
        json.dumps(
            {
                "id": task.id,
                "text": task.text,
                "tool_name": task.tool_name,
                "parameters": dict(task.parameters),
            },
            indent=2,
        ),
        "## Turn context τ",
        json.dumps(dict(turn_context), indent=2, default=str),
        "## Scratchpad S",
        json.dumps(dict(scratchpad), indent=2, default=str),
        "Return JSON object `values` mapping param names to proposed values.",
    ]
    return "\n\n".join(blocks)


def run_llm_fill_missing_params(
    client: OpenRouterClient,
    *,
    task: UnitTask,
    tool_name: str,
    missing_params: list[str],
    scratchpad: dict[str, Any],
    turn_context: dict[str, Any],
    tools_registry: dict[str, Any],
    primitive: Primitive | None = None,
) -> dict[str, Any]:
    if not missing_params:
        return {}
    user = build_match_fill_user_prompt(
        task=task,
        missing_params=missing_params,
        scratchpad=scratchpad,
        turn_context=turn_context,
        tool_name=tool_name,
        tools_registry=tools_registry,
    )
    messages = [
        {"role": "system", "content": MATCH_FILL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    plog, bodies = _llm_logging(primitive)
    if plog is not None:
        plog.info(
            "LLM MATCH_FILL request: task_id=%s tool=%s missing=%s τ_keys=%s S_keys=%s",
            task.id,
            tool_name,
            missing_params,
            sorted(turn_context.keys()),
            sorted(scratchpad.keys()),
        )
        if bodies:
            plog.debug("LLM MATCH_FILL system: %s", MATCH_FILL_SYSTEM_PROMPT)
            plog.debug("LLM MATCH_FILL user: %s", user)
    parsed = client.chat(
        messages,
        response_model=LLMMatchFillParams,
        max_tokens=_kernel_stage_max_tokens(primitive),
    )
    assert isinstance(parsed, LLMMatchFillParams)
    filled = dict(parsed.values or {})
    if plog is not None:
        plog.info(
            "LLM MATCH_FILL response: filled_keys=%s",
            sorted(filled.keys()),
        )
        if bodies:
            plog.debug("LLM MATCH_FILL values: %s", json.dumps(filled, default=str))
    return filled


class OrderUnitTasks(HybridPrimitive):
    """GET-ORDER: DAG topo + deterministic sort, or LLM sibling ranking per Kahn wave.

    With ``config['llm_client']`` set, each batch of mutually ready tasks (same wave) is
    sent to the model; the returned permutation must match the batch exactly or the batch
    falls back to lexicographic order by id.
    """

    def forward(self, tasks: list[UnitTask]) -> dict[str, Any]:
        ids = [t.id for t in tasks]
        graph: dict[str, list[str]] = {i: [] for i in ids}
        for t in tasks:
            for dep in t.depends_on:
                if dep not in graph:
                    graph[dep] = []
                graph[dep].append(t.id)

        from nre.library.deterministic import CycleDetect, TopologicalSort

        self.log.info("OrderUnitTasks: unit_tasks=%s ids=%s", len(tasks), ids)
        cyc = CycleDetect().forward(graph)
        if cyc["has_cycle"]:
            self.log.warning(
                "OrderUnitTasks: cycle detected tasks=%s cycles=%s",
                ids,
                cyc.get("cycles"),
            )
            return {
                "ordered": [],
                "has_cycle": True,
                "cycles": cyc["cycles"],
                "is_dag": False,
            }

        n_edges = sum(len(v) for v in graph.values())
        self.log.debug("OrderUnitTasks: DAG edges=%s adj_size=%s", n_edges, len(graph))

        llm_client = self.config.get("llm_client")
        summary = self.config.get("conversation_summary") or ""
        if not isinstance(summary, str):
            summary = str(summary)

        if llm_client is None:
            topo = TopologicalSort().forward(graph)
            self.log.info(
                "OrderUnitTasks: deterministic topo (no llm_client) ordered_len=%s is_dag=%s head=%s",
                len(topo["sorted"]),
                topo["is_dag"],
                topo["sorted"][:12],
            )
            return {
                "ordered": topo["sorted"],
                "has_cycle": False,
                "cycles": [],
                "is_dag": topo["is_dag"],
                "used_llm_sibling_order": False,
            }

        levels = TopologicalSort.levels(graph)
        self.log.info(
            "OrderUnitTasks: LLM sibling waves=%s summary_chars=%s",
            len(levels),
            len(summary),
        )
        by_id = {t.id: t for t in tasks}
        ordered: list[str] = []
        used_llm = False
        for batch in levels:
            if len(batch) <= 1:
                ordered.extend(batch)
                continue
            parsed = run_llm_sibling_order_batch(
                llm_client,
                batch,
                by_id,
                conversation_summary=summary,
                primitive=self,
            )
            perm = list(parsed.ordered_task_ids)
            if validate_sibling_permutation(perm, batch):
                ordered.extend(perm)
                used_llm = True
            else:
                self.log.warning(
                    "OrderUnitTasks: invalid LLM permutation, fallback lex batch=%s perm=%s",
                    batch,
                    perm,
                )
                ordered.extend(sorted(batch))

        self.log.info(
            "OrderUnitTasks: final ordered_len=%s used_llm_sibling_order=%s head=%s",
            len(ordered),
            used_llm,
            ordered[:12],
        )
        return {
            "ordered": ordered,
            "has_cycle": False,
            "cycles": [],
            "is_dag": True,
            "used_llm_sibling_order": used_llm,
        }


# ── Registry-driven defaults for CheckPrerequisites ──────────────────
#
# These helpers implement spec Sub-process 4's ``GET-PREREQS``, ``Satisfied``
# and ``TASK-TO-CHECK`` from declarative fields on tool registry entries
# (``requires``, ``param_types``, ``produces``). They are used only when the
# caller wires a ``tools_registry`` into the primitive config *and* does not
# supply an explicit hook — caller hooks always win.


def _default_get_prereqs(task: UnitTask, tools_registry: dict[str, Any]) -> list[str]:
    """Read ``requires`` off the tool's registry entry. Spec GET-PREREQS."""
    entry = tools_registry.get(task.tool_name) if tools_registry else None
    if not isinstance(entry, dict):
        return []
    req = entry.get("requires")
    if isinstance(req, (list, tuple)):
        return [str(k) for k in req]
    return []


def _default_satisfied(
    key: str,
    value: Any,
    task: UnitTask,
    tools_registry: dict[str, Any],
) -> bool:
    """Validate ``value`` under the param-type tag declared for ``key``.

    Recognized tags:
      - ``"zipcode"``            — string of exactly 5 decimal digits.
      - ``"non_empty_str"``      — string with at least one character.
    Any other tag (or no tag) falls back to ``value is not None``.

    **Pending-executor sentinel.** The agent_turn retry loop writes
    ``{"__nre_pending__": <task_id>, "tool": <tool_name>}`` under a produced
    key when the producer ran with ``status=bound`` (server mode — the base
    LLM is the actual executor, the kernel is only planning). ``Satisfied``
    accepts this sentinel as met so the consumer task can proceed past
    CHECK; the real value will be supplied by the base LLM's tool_call result
    at runtime.
    """
    if value is None:
        return False
    if isinstance(value, dict) and "__nre_pending__" in value:
        return True
    entry = tools_registry.get(task.tool_name) if tools_registry else None
    if not isinstance(entry, dict):
        return True
    tags = entry.get("param_types")
    if not isinstance(tags, dict):
        return True
    tag = tags.get(key)
    if tag == "zipcode":
        return isinstance(value, str) and len(value) == 5 and value.isdigit()
    if tag == "non_empty_str":
        return isinstance(value, str) and len(value) > 0
    return True


def _default_remediation(
    key: str,
    task: UnitTask,
    tools_registry: dict[str, Any],
) -> UnitTask | None:
    """Spec ``TASK-TO-CHECK(p)``: find a tool whose ``produces`` list contains
    ``key`` and synthesize a UnitTask that calls it. Deterministic tie-break:
    sort candidate tool names alphabetically and pick the first.
    """
    if not tools_registry:
        return None
    producers: list[str] = []
    for name, entry in tools_registry.items():
        if not isinstance(entry, dict):
            continue
        prod = entry.get("produces")
        if isinstance(prod, (list, tuple)) and key in prod:
            producers.append(str(name))
    if not producers:
        return None
    producers.sort()
    chosen = producers[0]
    return UnitTask(
        id=f"{task.id}__prereq_{key}",
        text=f"Populate prerequisite {key!r} for {task.id}",
        depends_on=(),
        tool_name=chosen,
        parameters={},
    )


class CheckPrerequisites(HybridPrimitive):
    """CHECK-PREREQUISITES: Met / Absent / ¬Met (blocked) over ordered prerequisite keys.

    Optional:

    - ``classify_prereq(key, task, scratchpad) -> "met"|"absent"|"not_met"`` — override default key-presence + ``satisfied`` logic.
    - ``remediation_for(key, task, scratchpad, outcome) -> UnitTask | None`` — symbolic t′
      (``outcome`` is ``"absent"`` or ``"not_met"``).
    - ``llm_client`` + ``tools_registry_for_llm`` — OpenRouter proposes remediation when the
      hook returns ``None``.
    - ``check_trace=True`` — include ``checks`` list of ``{key, outcome}`` for every key
      examined (including on failure).

    **Registry-driven defaults (spec Sub-process 4 wiring).**
    When ``tools_registry`` is supplied (the same Ω that MATCH consults) and the
    hooks above are *not* provided, the primitive synthesizes them:

    - ``get_prereqs(task)`` reads ``tools_registry[task.tool_name]["requires"]``
      (list of scratchpad keys).
    - ``Satisfied(key, val, S)`` consults ``tools_registry[task.tool_name]
      ["param_types"][key]``. Recognized tags: ``"zipcode"`` (5-digit string),
      ``"non_empty_str"``. Any other tag falls back to ``v is not None``.
    - ``TASK-TO-CHECK(key)`` scans the whole registry for an entry whose
      ``produces`` list contains ``key``; the first such tool (sorted by name,
      for determinism) becomes the remediation ``UnitTask``.

    Caller-supplied hooks always override these defaults.
    """

    def forward(self, task: UnitTask, scratchpad: dict[str, Any]) -> dict[str, Any]:
        tools_registry: dict[str, Any] = self.config.get("tools_registry") or {}
        get_prereqs: Callable[[UnitTask], list[str]] = self.config.get(
            "get_prereqs",
            lambda t: _default_get_prereqs(t, tools_registry),
        )
        satisfied: Callable[[str, Any, dict[str, Any]], bool] = self.config.get(
            "satisfied",
            lambda k, v, _s: _default_satisfied(k, v, task, tools_registry),
        )
        classify: Callable[[str, UnitTask, dict[str, Any]], str] | None = (
            self.config.get("classify_prereq")
        )
        remediation_for: (
            Callable[[str, UnitTask, dict[str, Any], str], UnitTask | None] | None
        ) = (
            self.config.get(
                "remediation_for",
                lambda k, t, _s, _o: _default_remediation(k, t, tools_registry),
            )
            if tools_registry
            else self.config.get("remediation_for")
        )
        llm_client = self.config.get("llm_client")
        tools_for_llm: dict[str, Any] = self.config.get("tools_registry_for_llm") or {}
        trace = bool(self.config.get("check_trace", False))
        checks_trace: list[dict[str, Any]] = []
        prereq_keys = get_prereqs(task)
        self.log.info(
            "CheckPrerequisites: task_id=%s prereq_keys=%s scratchpad_keys=%s llm_enabled=%s trace=%s",
            task.id,
            prereq_keys,
            sorted(scratchpad.keys()),
            llm_client is not None and bool(tools_for_llm),
            trace,
        )

        def _outcome_for_key(key: str) -> str:
            if classify is not None:
                o = classify(key, task, scratchpad)
                if o not in ("met", "absent", "not_met"):
                    raise ValueError(
                        f"classify_prereq must return 'met', 'absent', or 'not_met', got {o!r}"
                    )
                return o
            if key not in scratchpad:
                return "absent"
            if not satisfied(key, scratchpad[key], scratchpad):
                return "not_met"
            return "met"

        for key in prereq_keys:
            outcome = _outcome_for_key(key)
            if trace:
                checks_trace.append({"key": key, "outcome": outcome})
            self.log.debug("CheckPrerequisites: key=%s outcome=%s", key, outcome)
            if outcome == "met":
                continue
            status = "absent" if outcome == "absent" else "blocked"
            rem: UnitTask | None = None
            if remediation_for is not None:
                rem = remediation_for(key, task, scratchpad, outcome)
                if rem is not None:
                    self.log.info(
                        "CheckPrerequisites: hook remediation key=%s -> task_id=%s tool=%s",
                        key,
                        rem.id,
                        rem.tool_name,
                    )
            if rem is None and llm_client is not None and tools_for_llm:
                self.log.info(
                    "CheckPrerequisites: invoking LLM remediation key=%s outcome=%s",
                    key,
                    outcome,
                )
                rem = run_llm_prereq_remediation(
                    llm_client,
                    prereq_key=key,
                    outcome=outcome,
                    task=task,
                    scratchpad=scratchpad,
                    tools_registry=tools_for_llm,
                    primitive=self,
                )
            elif rem is None and outcome != "met":
                self.log.debug(
                    "CheckPrerequisites: no remediation (hook=None llm=%s tools_empty=%s)",
                    llm_client is not None,
                    not bool(tools_for_llm),
                )
            out: dict[str, Any] = {"status": status, "key": key, "task_id": task.id}
            if rem is not None:
                out["remediation_task"] = rem
            if trace:
                out["checks"] = checks_trace
            if rem is not None:
                self.log.info(
                    "CheckPrerequisites: returning blocked/absent with remediation task_id=%s",
                    rem.id,
                )
            return out

        out_ready: dict[str, Any] = {"status": "ready", "task_id": task.id}
        if trace:
            out_ready["checks"] = checks_trace
        self.log.info("CheckPrerequisites: all prereqs met task_id=%s", task.id)
        return out_ready


class DecomposeInput(HybridPrimitive):
    def forward(
        self,
        iota: str,
        gamma: dict[str, Any],
        tools_registry: dict[str, Any],
    ) -> dict[str, Any]:
        self.log.debug(
            "DecomposeInput: iota_chars=%s gamma_keys=%s tool_count=%s llm_client=%s",
            len(iota),
            sorted(gamma.keys()),
            len(tools_registry) if isinstance(tools_registry, dict) else 0,
            self.config.get("llm_client") is not None,
        )
        llm_client = self.config.get("llm_client")
        if llm_client is not None:
            summary = self.config.get("conversation_summary") or ""
            if not isinstance(summary, str):
                summary = str(summary)
            pt_raw = self.config.get("prior_turns")
            prior_turns: list[dict[str, Any]] | None = None
            if isinstance(pt_raw, list) and pt_raw:
                prior_turns = [x for x in pt_raw if isinstance(x, dict)]
                if not prior_turns:
                    prior_turns = None
            pt_max = self.config.get("prior_turns_max_chars")
            pt_max_i: int | None = None
            if pt_max is not None:
                try:
                    pt_max_i = int(pt_max)
                except (TypeError, ValueError):
                    pt_max_i = None
            raw = run_llm_decompose(
                llm_client,
                iota,
                gamma,
                tools_registry,
                conversation_summary=summary,
                prior_turns=prior_turns,
                prior_turns_max_chars=pt_max_i,
                primitive=self,
            )
            norm = normalize_decompose_result(raw)
            turns = norm.get("turns") if isinstance(norm, dict) else []
            n_turns = len(turns) if isinstance(turns, list) else 0
            self.log.info(
                "DecomposeInput: normalized (llm) turns=%s config_keys=%s tool_keys=%s",
                n_turns,
                sorted((norm.get("config") or {}).keys())
                if isinstance(norm, dict)
                else [],
                sorted((norm.get("tools") or {}).keys())
                if isinstance(norm.get("tools"), dict)
                else [],
            )
            if self.log_bodies_enabled():
                try:
                    blob = json.dumps(norm, indent=2, default=str)
                except (TypeError, ValueError):
                    blob = repr(norm)
                self.log.debug("DecomposeInput: normalized payload: %s", blob)
            return norm

        decompose = self.config.get("decompose")
        if decompose is None:
            raise ValueError(
                "Provide config['llm_client'] (OpenRouterClient) or config['decompose'] callable"
            )
        self.log.info("DecomposeInput: using callable decompose() (no llm_client)")
        raw = decompose(iota, gamma, tools_registry)
        self.log.debug(
            "DecomposeInput: callable raw type=%s keys=%s",
            type(raw).__name__,
            sorted(raw.keys()) if isinstance(raw, dict) else None,
        )
        norm = normalize_decompose_result(raw)
        turns = norm.get("turns") if isinstance(norm, dict) else []
        n_turns = len(turns) if isinstance(turns, list) else 0
        self.log.info(
            "DecomposeInput: normalized (callable) turns=%s config_keys=%s",
            n_turns,
            sorted((norm.get("config") or {}).keys()) if isinstance(norm, dict) else [],
        )
        if self.log_bodies_enabled():
            try:
                blob = json.dumps(norm, indent=2, default=str)
            except (TypeError, ValueError):
                blob = repr(norm)
            self.log.debug("DecomposeInput: normalized (callable) payload: %s", blob)
        return norm


class MatchFuncsAndParams(HybridPrimitive):
    """MATCH-FUNCS-AND-PARAMS: resolve ``f ∈ Ω``, bind parameters (τ then ``S``), optionally ``EXECUTE``.

    Resolution order for **function** (first hit wins):

    1. ``task.tool_name`` if ``prefer_task_tool_name`` (default True) and it names a registry entry.
    2. ``match_func(task, tools)`` when configured.
    3. **OpenRouter** — ``run_llm_match_func`` when ``llm_client`` is set and the name is still unknown.

    Parameters: ``task.parameters`` → ``turn_context`` → ``scratchpad`` → optional ``retrieve`` hook;
    then, if ``llm_fill_missing_params`` and ``llm_client`` are set, **OpenRouter** may fill remaining
    keys via ``run_llm_fill_missing_params``.

    If the registry entry has no ``execute`` callable, returns ``status="bound"`` with bindings only
    (orchestrator / host may run the real tool).

    **Binding normalization**

    After all parameters are non-null, values are passed through
    :func:`nre.utils.tool_registry.canonicalize_bindings_with_registry` so strings match
    declared ``enum`` / ``param_enums`` when there is a unique case-insensitive hit (e.g. ``buy`` → ``Buy``).

    Optional ``config['normalize_bindings']``: ``Callable[[tool_name, bindings, registry_entry], dict]``
    for host-specific rules; receives and must return a **bindings** dict.
    """

    def forward(
        self,
        task: UnitTask,
        scratchpad: dict[str, Any],
        turn_context: dict[str, Any],
        tools: dict[str, Any],
    ) -> dict[str, Any]:
        prefer_tool = bool(self.config.get("prefer_task_tool_name", True))
        match_fn: Callable[[UnitTask, dict[str, Any]], str | None] | None = (
            self.config.get("match_func")
        )
        llm_client = self.config.get("llm_client")
        fill_missing = bool(self.config.get("llm_fill_missing_params", False))
        retrieve = self.config.get("retrieve")

        def _tool_names() -> set[str]:
            return {k for k, v in tools.items() if isinstance(v, dict)}

        def _resolve_name() -> str | None:
            names = _tool_names()
            if prefer_tool and task.tool_name and task.tool_name in names:
                self.log.info(
                    "MatchFuncsAndParams: resolved tool from task hint task_id=%s tool=%s",
                    task.id,
                    task.tool_name,
                )
                return task.tool_name
            if match_fn is not None:
                hit = match_fn(task, tools)
                if hit is not None:
                    self.log.info(
                        "MatchFuncsAndParams: match_func hook task_id=%s -> %s",
                        task.id,
                        hit,
                    )
                    return hit
            if llm_client is not None and names:
                self.log.info(
                    "MatchFuncsAndParams: calling LLM match_func task_id=%s registry=%s",
                    task.id,
                    len(names),
                )
                hit = run_llm_match_func(
                    llm_client,
                    task=task,
                    tools_registry=tools,
                    primitive=self,
                )
                if hit is not None:
                    return hit
            return None

        self.log.debug(
            "MatchFuncsAndParams: task_id=%s prefer_hint=%s fill_missing=%s registry_tools=%s",
            task.id,
            prefer_tool,
            fill_missing,
            sorted(_tool_names()),
        )
        name = _resolve_name()
        if name is None or name not in tools or not isinstance(tools[name], dict):
            self.log.warning(
                "MatchFuncsAndParams: missing_function task_id=%s resolved=%r",
                task.id,
                name,
            )
            return {
                "status": "failure",
                "reason": "missing_function",
                "task_id": task.id,
            }

        spec = tools[name]
        raw_params = spec.get("params") or spec.get("parameters") or []
        if isinstance(raw_params, list):
            param_names = [
                p if isinstance(p, str) else str(p.get("name", "")) for p in raw_params
            ]
            param_names = [p for p in param_names if p]
        else:
            param_names = []

        execute = spec.get("execute")
        defaults = (
            spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
        )
        param_sources_gate: dict[str, list[str]] = (
            spec.get("param_sources")
            if isinstance(spec.get("param_sources"), dict)
            else {}
        )
        bindings: dict[str, Any] = {}
        # Per-binding provenance: one of
        # turn_context | scratchpad | task.parameters | default | retrieve | llm_fill | tool_result
        # ``None`` means the param is unbound so far.
        sources: dict[str, str | None] = {m: None for m in param_names}
        # Flag whether the source-tracking feature is in active use — determines
        # whether ``binding_sources`` is emitted in the result payload (kept off
        # by default so untouched callers see no shape change).
        source_feature_active = bool(param_sources_gate)

        def _unpack(value: Any, default_src: str) -> tuple[Any, str]:
            """Extract (value, source) from either a bare value or a source-tagged
            dict ``{"value": v, "source": s}``. The tagged form lets callers
            declare where a τ or S entry originated (e.g. a tool_result)."""
            nonlocal source_feature_active
            if (
                isinstance(value, dict)
                and "value" in value
                and "source" in value
                and isinstance(value.get("source"), str)
            ):
                source_feature_active = True
                return value["value"], value["source"]
            return value, default_src

        # Spec Sub-process 5 LOOKUP (INV-3): τ is always consulted before S.
        # ``task.parameters`` is a DECOMPOSE-emitted hint and acts only as a
        # fallback after τ and S both miss — this prevents a stale plan hint
        # from silently overriding live turn context.
        for m in param_names:
            if m in turn_context:
                v, src = _unpack(turn_context[m], "turn_context")
            elif m in scratchpad:
                v, src = _unpack(scratchpad[m], "scratchpad")
            elif m in task.parameters:
                v, src = _unpack(task.parameters[m], "task.parameters")
            else:
                v, src = None, None
            if v is None and retrieve is not None:
                retrieved = retrieve(m, task, scratchpad, turn_context)
                if retrieved is not None:
                    scratchpad[m] = retrieved
                    v, src = retrieved, "retrieve"
            bindings[m] = v
            sources[m] = src

        # Apply registry-declared defaults for params not otherwise bound.
        # Spec treats every ``m ∈ PARAMS(f)`` as required; defaults are a host-
        # side contract that closes the gap for optional params without
        # inventing values via the LLM.
        if defaults:
            for m in param_names:
                if bindings[m] is None and m in defaults:
                    bindings[m] = defaults[m]
                    sources[m] = "default"

        missing = [m for m in param_names if bindings[m] is None]
        self.log.debug(
            "MatchFuncsAndParams: tool=%s param_names=%s missing_before_fill=%s",
            name,
            param_names,
            missing,
        )
        if missing and fill_missing and llm_client is not None:
            filled = run_llm_fill_missing_params(
                llm_client,
                task=task,
                tool_name=name,
                missing_params=missing,
                scratchpad=scratchpad,
                turn_context=turn_context,
                tools_registry=tools,
                primitive=self,
            )
            for k, v in filled.items():
                if k in missing and v is not None:
                    bindings[k] = v
                    scratchpad[k] = v
                    sources[k] = "llm_fill"
            still = [m for m in param_names if bindings[m] is None]
            self.log.info(
                "MatchFuncsAndParams: after LLM fill missing=%s filled_keys=%s",
                still,
                sorted(filled.keys()),
            )

        for m in param_names:
            if bindings[m] is None:
                self.log.warning(
                    "MatchFuncsAndParams: missing_param task_id=%s tool=%s param=%r",
                    task.id,
                    name,
                    m,
                )
                return {
                    "status": "failure",
                    "reason": "missing_param",
                    "param": m,
                    "task_id": task.id,
                }

        # Source gate: reject a bound value whose provenance is not in the
        # tool's allow-list. Enables the host to declare "payment_methods must
        # come from tool_result, not from user-asserted task.parameters" —
        # closing τ² airline task_0's class of failures.
        if param_sources_gate:
            for m, allowed in param_sources_gate.items():
                if m not in bindings or not isinstance(allowed, (list, tuple)):
                    continue
                src = sources.get(m)
                if src is not None and src not in allowed:
                    self.log.warning(
                        "MatchFuncsAndParams: forbidden_source task_id=%s tool=%s "
                        "param=%r source=%r allowed=%s",
                        task.id,
                        name,
                        m,
                        src,
                        list(allowed),
                    )
                    return {
                        "status": "failure",
                        "reason": "forbidden_source",
                        "param": m,
                        "source": src,
                        "allowed": list(allowed),
                        "task_id": task.id,
                    }

        _pre_norm = dict(bindings)
        bindings = canonicalize_bindings_with_registry(bindings, spec)
        norm_cb = self.config.get("normalize_bindings")
        if callable(norm_cb):
            bindings = dict(norm_cb(name, bindings, spec))
        for m in param_names:
            if m in bindings and bindings.get(m) != _pre_norm.get(m):
                scratchpad[m] = bindings[m]

        base_out: dict[str, Any] = {
            "function": name,
            "bindings": bindings,
            "task_id": task.id,
        }
        # Emit provenance only when the feature is actively in use (param_sources
        # gate set or any value arrived as a source tag). Keeps legacy callers'
        # response shape unchanged.
        if source_feature_active:
            base_out["binding_sources"] = {
                k: v for k, v in sources.items() if v is not None
            }
        if self.log_bodies_enabled():
            try:
                bind_blob = json.dumps(bindings, default=str)
            except (TypeError, ValueError):
                bind_blob = repr(bindings)
            self.log.debug("MatchFuncsAndParams: final bindings: %s", bind_blob)

        if execute is None or not callable(execute):
            self.log.info(
                "MatchFuncsAndParams: bound (no execute) task_id=%s tool=%s",
                task.id,
                name,
            )
            return {
                "status": "bound",
                **base_out,
            }

        result = execute(**bindings)
        self.log.info(
            "MatchFuncsAndParams: execute success task_id=%s tool=%s result_type=%s",
            task.id,
            name,
            type(result).__name__,
        )
        if self.log_bodies_enabled():
            self.log.debug("MatchFuncsAndParams: execute result: %r", result)
        return {
            "status": "success",
            **base_out,
            "result": result,
        }


class PolicyConstraints(HybridPrimitive):
    """Pre-mutation policy gate: evaluate declarative rules against a candidate
    tool call, return **allow** or **deny(policy, reason)**.

    Many tool-calling benchmarks (τ², live-mcp-bench) fail not because of
    planning errors but because the agent violated a prose policy that could
    be encoded as a symbolic predicate (e.g. "Basic economy flights cannot
    be modified", "All payment methods must already be in user profile").
    This primitive expresses such policies as Python callables and provides
    a uniform gate in the ``agent_turn`` loop.

    **Policy rule shape** (``config["policies"]``):

        {
            "<policy_name>": {
                "applies_to": list[str] | callable(tool_name, task, bindings) -> bool,
                "predicate": callable(bindings, scratchpad, turn_context) -> bool,
                "reason":    str,          # human-readable refusal reason
            },
            ...
        }

    ``applies_to`` as a list of tool names is the common case. A callable is
    used when applicability depends on the bound argument shape (e.g. only
    gate ``update_reservation_flights`` when ``cabin == "basic_economy"``).

    ``predicate`` returns ``True`` when the policy is **satisfied**. If it
    returns ``False``, the call is denied with that rule's ``reason``.

    Evaluation order is deterministic (sorted by policy name) and
    short-circuits on the first denial — matches spec Sub-process 4's
    single-block-per-call convention.

    With ``trace=True`` in config, the result includes ``evaluated``, a list
    of ``{policy, applies, result}`` rows for every rule considered.
    """

    def forward(
        self,
        tool_name: str,
        task: UnitTask,
        bindings: dict[str, Any],
        scratchpad: dict[str, Any],
        turn_context: dict[str, Any],
    ) -> dict[str, Any]:
        policies: dict[str, Any] = self.config.get("policies") or {}
        trace = bool(self.config.get("trace", False))
        evaluated: list[dict[str, Any]] = []

        if not policies:
            out: dict[str, Any] = {"status": "allow"}
            if trace:
                out["evaluated"] = []
            return out

        for name in sorted(policies.keys()):
            rule = policies[name]
            if not isinstance(rule, dict):
                continue
            applies_to = rule.get("applies_to")
            applies: bool
            if callable(applies_to):
                try:
                    applies = bool(applies_to(tool_name, task, bindings))
                except Exception as e:
                    self.log.warning(
                        "PolicyConstraints: applies_to callable raised for %s: %s",
                        name,
                        e,
                    )
                    applies = False
            elif isinstance(applies_to, (list, tuple)):
                applies = tool_name in applies_to
            else:
                applies = False

            if not applies:
                if trace:
                    evaluated.append(
                        {"policy": name, "applies": False, "result": "skip"}
                    )
                continue

            predicate = rule.get("predicate")
            if not callable(predicate):
                self.log.warning(
                    "PolicyConstraints: rule %r has no callable predicate; skipping",
                    name,
                )
                continue

            try:
                passed = bool(predicate(bindings, scratchpad, turn_context))
            except Exception as e:
                self.log.warning(
                    "PolicyConstraints: predicate for %r raised %s; treating as deny",
                    name,
                    e,
                )
                passed = False

            if passed:
                if trace:
                    evaluated.append(
                        {"policy": name, "applies": True, "result": "allow"}
                    )
                continue

            # Denial — short-circuit.
            if trace:
                evaluated.append({"policy": name, "applies": True, "result": "deny"})
            reason = str(rule.get("reason") or f"policy {name!r} denied the call")
            self.log.info(
                "PolicyConstraints: DENY tool=%s policy=%s reason=%s",
                tool_name,
                name,
                reason,
            )
            out = {
                "status": "deny",
                "policy": name,
                "reason": reason,
            }
            if trace:
                out["evaluated"] = evaluated
            return out

        out = {"status": "allow"}
        if trace:
            out["evaluated"] = evaluated
        return out


# ── Demo / tutorial hybrid primitives ──────────────────────────────────


class Embedder(HybridPrimitive):
    """Map arbitrary inputs into a fixed-dimension vector space.

    Config:
        dim (int): Embedding dimension.  Default 64.
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.dim: int = config.get("dim", 64)
        self._weights: dict[str, list[float]] = {}

    def initialize(self) -> None:
        super().initialize()
        self._weights = {}

    def _random_vec(self, seed: int) -> list[float]:
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self.dim)]
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / (norm + 1e-9) for v in vec]

    def forward(self, inputs: Any) -> list[float]:
        key = str(inputs)
        if key not in self._weights:
            self._weights[key] = self._random_vec(hash(key) & 0xFFFFFFFF)
        return self._weights[key]


class Classifier(HybridPrimitive):
    """Classify an embedding vector into one of *classes*.

    Config:
        classes (list[str]): Class labels.
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.classes: list[str] = config.get("classes", [])
        self._centroids: dict[str, list[float]] = {}

    def initialize(self) -> None:
        super().initialize()
        dim = 64
        for cls in self.classes:
            rng = random.Random(hash(cls) & 0xFFFFFFFF)
            self._centroids[cls] = [rng.gauss(0, 1) for _ in range(dim)]

    @staticmethod
    def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb + 1e-9)

    def forward(self, embedding: list[float]) -> dict[str, Any]:
        scores = {cls: self._cosine(embedding, c) for cls, c in self._centroids.items()}
        best = max(scores, key=scores.__getitem__)
        return {"label": best, "confidence": scores[best], "scores": scores}


class Similarity(HybridPrimitive):
    """Compute pairwise similarity between two embedding vectors.

    Config:
        metric (str): "cosine" or "euclidean".  Default "cosine".
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.metric: str = config.get("metric", "cosine")

    def forward(self, a: list[float], b: list[float]) -> float:
        if self.metric == "cosine":
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(x * x for x in b))
            return dot / (na * nb + 1e-9)
        elif self.metric == "euclidean":
            return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
        raise ValueError(f"Unknown metric: {self.metric}")


class ScoredRule(HybridPrimitive):
    """A rule whose condition is evaluated by a neural scorer.

    Config:
        scorer (callable):   (working_memory) -> float
        action (callable):   (working_memory) -> None
        threshold (float):   Firing threshold (default 0.5).
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.scorer: Callable[[dict[str, Any]], float] = config["scorer"]
        self.action: Callable[[dict[str, Any]], Any] = config["action"]
        self.threshold: float = config.get("threshold", 0.5)

    def forward(self, working_memory: dict[str, Any]) -> dict[str, Any]:
        score = self.scorer(working_memory)
        wm = dict(working_memory)
        wm.setdefault("_neural_scores", {})[self.name] = score
        if score >= self.threshold:
            self.action(wm)
            wm.setdefault("_fired_rules", []).append(self.name)
        return wm


class GuidedAttention(HybridPrimitive):
    """Re-weight an embedding using symbolic relevance signals.

    Config:
        focus_predicates (list[str]):  Predicate names to attend to.
        boost (float):  Multiplier for attended dimensions (default 2.0).
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.focus_predicates: list[str] = config.get("focus_predicates", [])
        self.boost: float = config.get("boost", 2.0)

    def forward(
        self,
        embedding: list[float],
        active_facts: list[Fact],
    ) -> list[float]:
        active_preds = {f.predicate for f in active_facts}
        overlap = len(active_preds & set(self.focus_predicates))
        if overlap == 0:
            return list(embedding)

        dim = len(embedding)
        chunk = max(1, dim // max(len(self.focus_predicates), 1))
        boosted_end = min(dim, overlap * chunk)

        result = list(embedding)
        for i in range(boosted_end):
            result[i] *= self.boost

        norm = math.sqrt(sum(v * v for v in result))
        return [v / (norm + 1e-9) for v in result]
