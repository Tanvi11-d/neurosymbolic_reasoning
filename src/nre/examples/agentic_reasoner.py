"""Agentic tool-calling turn — the canonical NRE internal DSL domain.

This module is two things in one file:

1. **The DSL domain** (:func:`register_agent_turn`). The full turn-level kernel pipeline
   (``DECOMPOSE → INIT-SCRATCHPAD → GET-ORDER → CHECK-PREREQUISITES → MATCH-FUNCS-AND-PARAMS``)
   is expressed as an :class:`nre.dsl.Engine` domain written in the internal DSL —
   ``ctx.let``, ``<<``, ``ctx.when`` / ``otherwise``, ``ctx.each``, ``ctx.returns``. It
   replaces the former procedural ``nre.dsl.run_turn`` orchestrator.

2. **Engine factories** — :func:`build_engine` (OpenRouter-backed, used by the FastAPI
   server and the CLI) and :func:`build_turn_engine` (injectable primitives, used by
   tests). Both install :func:`register_agent_turn` so the caller executes via the DSL
   with ``engine.run("agent_turn", …)``.

Run the CLI example::

    python -m nre.examples.agentic_reasoner
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from typing import Any

from nre.dsl import DomainContext, Engine
from nre.llm import OpenRouterClient, OpenRouterSettings
from nre.primitives.deterministic import (
    InitScratchpad,
    UnitTask,
    apply_init_scratchpad,
)
from nre.primitives.hybrid import (
    CheckPrerequisites,
    DecomposeInput,
    MatchFuncsAndParams,
    OrderUnitTasks,
)


# ── Loggers (preserved from the former nre.dsl.run_turn module) ──────

_log_kernel = logging.getLogger("nre.kernel")
_log_decompose = logging.getLogger("nre.decompose")
_log_init_scratchpad = logging.getLogger("nre.init_scratchpad")
_log_order_tasks = logging.getLogger("nre.order_tasks")
_log_check_prereqs = logging.getLogger("nre.check_prereqs")
_log_match_tools = logging.getLogger("nre.match_tools")


# ── Small helpers extracted from the agent_turn loop ─────────────────


def _thread_multi_turn_context(
    ctx: DomainContext,
    conversation_summary: str | None,
    prior_turns: Any,
) -> None:
    """Plumb optional multi-turn context into the LLM-backed primitives.

    DECOMPOSE and GET-ORDER read these from their ``config`` dict; we set or
    clear them here so each invocation of the domain sees fresh state and
    doesn't accidentally inherit values from a previous run.
    """
    if conversation_summary is not None:
        ctx.decompose.config["conversation_summary"] = conversation_summary
        ctx.order_tasks.config["conversation_summary"] = conversation_summary
    else:
        ctx.decompose.config.pop("conversation_summary", None)
        ctx.order_tasks.config.pop("conversation_summary", None)

    if isinstance(prior_turns, list):
        ctx.decompose.config["prior_turns"] = prior_turns
    else:
        ctx.decompose.config.pop("prior_turns", None)


def _queue_remediation(
    task_id: str,
    prereq_check: dict[str, Any],
    tasks_by_id: dict[str, UnitTask],
    queue: deque,
    seen_remediations: set[str],
) -> None:
    """Re-queue the failed task behind a remediation, if a fresh one was suggested.

    A remediation is "fresh" only when we haven't already attempted it on this
    turn — that prevents a stubborn producer from causing infinite retry churn
    (spec INV-T finiteness).
    """
    remediation = prereq_check.get("remediation_task")
    if not isinstance(remediation, UnitTask):
        return
    if remediation.id in seen_remediations:
        return
    seen_remediations.add(remediation.id)
    tasks_by_id[remediation.id] = remediation
    # Remediation must run before the retry of the original task.
    queue.appendleft(task_id)
    queue.appendleft(remediation.id)


def _policy_denial_step(
    ctx: DomainContext,
    task: UnitTask,
    match_result: dict[str, Any],
    scratchpad: dict[str, Any],
    task_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Run the optional ``policy_constraints`` gate and return a deny-step if any.

    When no policy primitive is registered, or the candidate isn't a successful
    bind, we return ``None`` and let the caller record the regular match step.
    Denials become ``status=denied`` steps; the task is abandoned (spec INV-4)
    and ``produces`` keys are NOT written so downstream consumers don't see
    phantom state from a refused mutation.
    """
    status = match_result.get("status")
    function_name = match_result.get("function")
    if "policy_constraints" not in ctx._engine.registry:
        return None
    if status not in ("success", "bound") or not function_name:
        return None

    decision = ctx.policy_constraints(
        function_name,
        task,
        match_result.get("bindings") or {},
        scratchpad,
        task_context,
    )
    if decision.get("status") != "deny":
        return None

    denied_step: dict[str, Any] = {
        "task_id": task.id,
        "phase": "match",
        "status": "denied",
        "function": function_name,
        "policy": decision.get("policy"),
        "reason": decision.get("reason"),
        "bindings": match_result.get("bindings") or {},
    }
    if "binding_sources" in match_result:
        denied_step["binding_sources"] = match_result["binding_sources"]
    return denied_step


def _write_produces_keys(
    scratchpad: dict[str, Any],
    match_result: dict[str, Any],
    merged_tools: dict[str, Any],
    task_id: str,
) -> None:
    """Write the tool's ``produces`` keys into the scratchpad after MATCH.

    - ``status=success`` → write the tool's real ``result`` (kernel is the
      executor, so S(produced_key) is ground truth).
    - ``status=bound`` → write a pending sentinel ``{"__nre_pending__": id,
      "tool": fn}``. The base LLM will run the tool later; ``Satisfied`` can
      recognise the sentinel as "pending executor completion" so the retry
      loop doesn't re-loop on the same prereq forever.
    """
    status = match_result.get("status")
    function_name = match_result.get("function")
    if status not in ("success", "bound") or not function_name:
        return

    tool_spec = merged_tools.get(function_name) or {}
    if not isinstance(tool_spec, dict):
        return
    produces_keys = tool_spec.get("produces")
    if not isinstance(produces_keys, (list, tuple)):
        return

    if status == "success":
        value: Any = match_result.get("result")
    else:
        value = {"__nre_pending__": task_id, "tool": function_name}
    for key in produces_keys:
        scratchpad[str(key)] = value


# ── DSL domain registration ──────────────────────────────────────────


def register_agent_turn(engine: Engine) -> None:
    """Install the ``agent_turn`` domain on *engine*.

    The engine's registry must already hold these five primitives:

    =====================  ===================================
    Registered name        Class
    =====================  ===================================
    ``decompose``          :class:`DecomposeInput`
    ``init_scratchpad``    :class:`InitScratchpad`
    ``order_tasks``        :class:`OrderUnitTasks`
    ``check_prereqs``      :class:`CheckPrerequisites`
    ``match_tools``        :class:`MatchFuncsAndParams`
    =====================  ===================================

    Inputs accepted by ``engine.run("agent_turn", …)``:

    - ``iota`` (str, required) — user utterance for this turn.
    - ``tools`` (dict, required) — tool registry Ω.
    - ``gamma`` (dict, optional) — initial config γ.
    - ``scratchpad_carry`` (dict | None, optional) — prior scratchpad state to mutate.
    - ``gamma_seeds`` (dict, optional) — seeds layered in before the decompose seed.
    - ``turn_context_by_task_id`` (dict, optional) — τ map per task id.
    - ``conversation_summary`` (str | None, optional) — threaded into the LLM-backed
      ``decompose`` and ``order_tasks`` primitives.
    - ``prior_turns`` (list[dict] | None, optional) — OpenAI-style messages for turns
      before ``iota``; threaded into ``decompose`` via ``config`` only (``forward`` args
      unchanged).

    Return dict keys: ``scratchpad``, ``ordered_task_ids``, ``steps``, ``cycle``,
    ``cycles``, ``decompose`` (plus ``_trace`` contributed by ``ctx.returns``).
    """

    @engine.domain("agent_turn")
    def agent_turn(ctx: DomainContext) -> dict[str, Any]:
        """One tool-calling turn, expressed in the NRE internal DSL."""
        # ── Read turn-level inputs ──────────────────────────────
        iota = ctx._inputs["iota"]
        gamma = ctx._inputs.get("gamma") or {}
        tools = ctx._inputs["tools"]
        carry = ctx._inputs.get("scratchpad_carry")
        seeds = ctx._inputs.get("gamma_seeds")
        turn_context_by_task_id = ctx._inputs.get("turn_context_by_task_id") or {}
        conversation_summary = ctx._inputs.get("conversation_summary")
        prior_turns = ctx._inputs.get("prior_turns")

        _thread_multi_turn_context(ctx, conversation_summary, prior_turns)

        # Expose the tool registry to CheckPrerequisites so its registry-driven
        # defaults (spec Sub-process 4 GET-PREREQS / Satisfied / TASK-TO-CHECK)
        # can consult ``requires`` / ``param_types`` / ``produces`` per turn.
        # The merged registry (tools + any DECOMPOSE-added tools) is set after
        # DECOMPOSE runs; this pre-set gives CHECK at least the base Ω.
        ctx.check_prereqs.config["tools_registry"] = tools

        _log_kernel.info(
            "agent_turn: begin iota_chars=%s gamma_keys=%s tools=%s prior_turns_msgs=%s",
            len(iota),
            sorted(gamma.keys()),
            sorted(tools.keys()) if isinstance(tools, dict) else [],
            len(prior_turns) if isinstance(prior_turns, list) else 0,
        )

        # ── DECOMPOSE ───────────────────────────────────────────
        decompose_result = ctx.decompose(iota, gamma, tools)
        turns = ctx.let[list] << list(decompose_result["turns"])
        decompose_config = dict(decompose_result.get("config") or {})
        merged_tools = {**tools, **(decompose_result.get("tools") or {})}
        # Any DECOMPOSE-added tools become visible to CheckPrerequisites too
        # (they may declare their own requires/produces for CHECK to use).
        ctx.check_prereqs.config["tools_registry"] = merged_tools
        _log_decompose.info(
            "DECOMPOSE → %s tasks ids=%s config_keys=%s",
            len(turns.val),
            [t.id for t in turns.val],
            sorted(decompose_config.keys()),
        )

        # ── INIT-SCRATCHPAD ─────────────────────────────────────
        # Spec Sub-process 1 ``Extract(γ)``: γ is deep-flattened into dot-path
        # keys by InitScratchpad so MATCH's LOOKUP has a populated S.
        # Decompose-authored ``config`` still wins on key collisions.
        seed = ctx.init_scratchpad(decompose_config, gamma=gamma)
        scratchpad = apply_init_scratchpad(seed, carry=carry, gamma_seeds=seeds)
        _log_init_scratchpad.info(
            "INIT-SCRATCHPAD → keys=%s (after merge)",
            sorted(scratchpad.keys()),
        )

        # ── GET-ORDER ───────────────────────────────────────────
        order = ctx.order_tasks(turns.val)

        with ctx.when(order["has_cycle"]) as has_cycle:
            if has_cycle:
                _log_order_tasks.warning(
                    "ORDER → cycle detected cycles=%s",
                    order.get("cycles"),
                )
                return ctx.returns(
                    scratchpad=scratchpad,
                    ordered_task_ids=[],
                    steps=[],
                    cycle=True,
                    cycles=order["cycles"],
                    decompose=decompose_result,
                )

        _log_order_tasks.info(
            "ORDER → linear order=%s llm_sibling=%s",
            order.get("ordered"),
            order.get("used_llm_sibling_order"),
        )

        # ── CHECK-PREREQUISITES + MATCH-FUNCS-AND-PARAMS ────────
        # Spec Sub-process 4: CHECK may return ``Blocked(t')`` or ``Absent`` with
        # a remediation task. We use a fuel-bounded worklist so pathological
        # remediation cycles cannot wedge the engine (spec INV-T finiteness).
        tasks_by_id: dict[str, UnitTask] = {t.id: t for t in turns.val}
        steps = ctx.let[list] << []
        seen_remediations: set[str] = set()
        queue: deque[str] = deque(order["ordered"])
        remaining_attempts = max(16, len(queue) * 4)

        while queue and remaining_attempts > 0:
            task_id = queue.popleft()
            remaining_attempts -= 1
            task = tasks_by_id.get(task_id)
            if task is None:
                continue

            prereq_check = ctx.check_prereqs(task, scratchpad)
            _log_check_prereqs.info(
                "CHECK-PREREQUISITES task_id=%s → status=%s key=%s",
                task_id,
                prereq_check["status"],
                prereq_check.get("key"),
            )
            if prereq_check["status"] != "ready":
                steps.val.append({"task_id": task_id, "phase": "prereq", **prereq_check})
                _queue_remediation(
                    task_id, prereq_check, tasks_by_id, queue, seen_remediations
                )
                continue

            task_context = dict(turn_context_by_task_id.get(task_id, {}))
            match_result = ctx.match_tools(task, scratchpad, task_context, merged_tools)
            _log_match_tools.info(
                "MATCH-FUNCS-AND-PARAMS task_id=%s → status=%s function=%s",
                task_id,
                match_result.get("status"),
                match_result.get("function"),
            )

            denied = _policy_denial_step(
                ctx, task, match_result, scratchpad, task_context
            )
            if denied is not None:
                steps.val.append(denied)
                continue

            steps.val.append({"task_id": task_id, "phase": "match", **match_result})
            _write_produces_keys(scratchpad, match_result, merged_tools, task_id)

        _log_kernel.info(
            "agent_turn: end steps=%s ordered=%s",
            len(steps.val),
            order.get("ordered"),
        )
        return ctx.returns(
            scratchpad=scratchpad,
            ordered_task_ids=list(order["ordered"]),
            steps=steps.val,
            cycle=False,
            cycles=[],
            decompose=decompose_result,
        )


# ── Engine factories ─────────────────────────────────────────────────


def build_turn_engine(
    *,
    decompose_input: DecomposeInput,
    init_scratchpad: InitScratchpad,
    order_unit_tasks: OrderUnitTasks,
    check_prerequisites: CheckPrerequisites,
    match_funcs: MatchFuncsAndParams,
) -> Engine:
    """Assemble an :class:`Engine` from pre-built primitive instances and install ``agent_turn``.

    Use this when you want to inject custom (e.g. stubbed) primitives — typically from
    tests that bypass the LLM. For the OpenRouter-backed production wiring, use
    :func:`build_engine` instead.
    """
    engine = Engine()
    engine.register(decompose_input, name="decompose")
    engine.register(init_scratchpad, name="init_scratchpad")
    engine.register(order_unit_tasks, name="order_tasks")
    engine.register(check_prerequisites, name="check_prereqs")
    engine.register(match_funcs, name="match_tools")
    register_agent_turn(engine)
    return engine


# ── Example scenario (BFCL-flavored) ─────────────────────────────────

IOTA_1 = "Create a directory called 'archive' in workspace and move report.txt into it."
IOTA_2 = "Search the moved report for any mention of 'revenue'."


def _make_turn1_tools(
    scratchpad: dict[str, Any],
    log: list[dict[str, Any]],
) -> dict[str, Any]:
    def cd(folder: str):
        log.append({"name": "cd", "kwargs": {"folder": folder}})
        scratchpad["cwd"] = "/workspace"
        scratchpad["cwd_is_workspace"] = True
        return {}

    def mkdir(dir_name: str):
        log.append({"name": "mkdir", "kwargs": {"dir_name": dir_name}})
        scratchpad["archive_exists"] = True
        return {}

    def mv(source: str, destination: str):
        log.append(
            {"name": "mv", "kwargs": {"source": source, "destination": destination}}
        )
        return {}

    return {
        "cd": {"description": "Tool cd", "params": ["folder"], "execute": cd},
        "mkdir": {
            "description": "Tool mkdir",
            "params": ["dir_name"],
            "execute": mkdir,
        },
        "mv": {
            "description": "Tool mv",
            "params": ["source", "destination"],
            "execute": mv,
        },
    }


def build_engine() -> Engine:
    """Build an :class:`Engine` with OpenRouter-backed primitives and ``agent_turn`` installed.

    Used by :mod:`nre.server.app` and the CLI (``python -m nre.examples.agentic_reasoner``).
    Requires ``OPENROUTER_API_KEY`` in the environment.
    """
    engine = Engine()
    client = OpenRouterClient(OpenRouterSettings())

    engine.register(
        DecomposeInput,
        name="decompose",
        llm_client=client,
        conversation_summary="",
    )
    engine.register(
        OrderUnitTasks,
        name="order_tasks",
        llm_client=client,
        conversation_summary="",
    )
    engine.register(
        CheckPrerequisites,
        name="check_prereqs",
        llm_client=client,
        tools_registry_for_llm={},
    )
    engine.register(
        MatchFuncsAndParams,
        name="match_tools",
        llm_client=client,
        llm_fill_missing_params=True,
    )
    engine.register(InitScratchpad, name="init_scratchpad")

    register_agent_turn(engine)
    return engine


# ── CLI entrypoint ───────────────────────────────────────────────────


def _print_turn(label: str, result: dict[str, Any]) -> None:
    print(f"\n=== {label} ===")
    if result.get("cycle"):
        print("Cycle detected:", result.get("cycles"))
        return
    for step in result.get("steps") or []:
        if step.get("phase") != "match":
            print(
                f"  prereq {step.get('task_id')}: {step.get('status')} {step.get('key', '')}"
            )
            continue
        status = step.get("status")
        function_name = step.get("function")
        if status == "success":
            print(f"  exec {step.get('task_id')}: {function_name} {step.get('bindings')}")
        else:
            print(f"  match {step.get('task_id')}: {status} {step.get('reason', '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic reasoner Engine example")
    _ = parser.parse_args()

    if not OpenRouterSettings().api_key:
        print(
            "OPENROUTER_API_KEY is not set; kernel primitives require OpenRouter.",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = build_engine()
    print("=== Agentic reasoner (Engine + five kernel primitives) ===")
    print("Engine:", engine)

    log: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    tools = _make_turn1_tools(state, log)

    print("\nDomain: agent_turn (OpenRouter DECOMPOSE + kernel)")
    print("IOTA:", IOTA_1[:60], "...")

    out = engine.run(
        "agent_turn",
        iota=IOTA_1,
        gamma={},
        tools=tools,
    )

    _print_turn("LLM single turn", out)
    print("\nScratchpad:", out.get("scratchpad"))
    print("Raw execution log (demo tools):", log)


if __name__ == "__main__":
    main()
