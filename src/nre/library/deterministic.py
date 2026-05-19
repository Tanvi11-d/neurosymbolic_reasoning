"""Deterministic primitive library — all concrete implementations."""

from __future__ import annotations

import bisect
from typing import Any

from ..primitives.deterministic import Constraint, DeterministicPrimitive, Fact, Rule


# ── Kernel deterministic primitives ───────────────────────────────────


class InitScratchpad(DeterministicPrimitive):
    """Sub-process 2: seed ``S`` from decompose ``config`` (flat key–value),
    optionally augmented by a deep-flattened ``γ``.

    Spec Sub-process 1 says ``config ← Extract(γ)``. The LLM-backed DECOMPOSE
    almost never emits useful ``config_keys``, so if we only honored ``config``
    the scratchpad would be empty for every turn and the LOOKUP chain in MATCH
    (τ → S → task.parameters) would collapse to its fallback.

    When ``gamma`` is provided, we deep-walk it into dot-path / ``[index]`` keys
    (e.g. ``VehicleControlAPI.doorStatus.driver`` or ``tweets[0].id``) and
    layer them **underneath** ``config`` (so a DECOMPOSE-authored ``config``
    entry always wins on collisions). Only leaves are emitted — empty inner
    dicts and lists contribute no keys.

    INV-S is preserved: this primitive only adds keys to the returned seed.
    Full turn merge (carry + ``γ_seeds`` + this seed) remains
    :func:`nre.primitives.deterministic.apply_init_scratchpad`.
    """

    def forward(
        self,
        config: dict[str, Any] | None = None,
        *,
        gamma: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        c = dict(config or {})
        gamma_flat = _flatten_gamma(gamma) if gamma else {}
        # Config wins on collision (spec: DECOMPOSE output is authoritative).
        seed: dict[str, Any] = {**gamma_flat, **c}
        keys = sorted(seed.keys())
        self.log.info(
            "InitScratchpad: seed count=%s config_keys=%s gamma_keys=%s",
            len(keys),
            sorted(c.keys()),
            len(gamma_flat),
        )
        if self.log_bodies_enabled():
            self.log.debug("InitScratchpad: seed repr: %r", seed)
        return seed


def _flatten_gamma(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Deep-walk ``obj`` into ``{dot.path[idx]: leaf_value}``.

    Dicts descend via ``f"{prefix}.{key}"`` (or bare ``key`` at the top level).
    Lists descend via ``f"{prefix}[{i}]"``.
    Non-dict, non-list values are emitted as leaves under the current prefix.
    Empty inner dicts / lists produce no keys.
    """
    out: dict[str, Any] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, (dict, list)):
                out.update(_flatten_gamma(v, key))
            else:
                out[key] = v
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            if isinstance(v, (dict, list)):
                out.update(_flatten_gamma(v, key))
            else:
                out[key] = v
    else:
        # Scalar at top level (no prefix) — caller provided a non-container γ;
        # nothing sensible to emit without a key.
        if prefix:
            out[prefix] = obj
    return out


# ── Core primitives ──────────────────────────────────────────────────


class RuleEngine(DeterministicPrimitive):
    """Forward-chaining rule engine.

    Accepts a list of :class:`Rule` objects (via config ``rules``)
    and fires all whose condition matches the current working memory,
    in priority order, until quiescence or *max_iterations*.
    """

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.rules: list[Rule] = list(config.get("rules", []))
        self.max_iterations: int = config.get("max_iterations", 100)

    def add_rule(self, rule: Rule) -> None:
        self.rules.append(rule)

    def forward(self, working_memory: dict[str, Any]) -> dict[str, Any]:
        wm = dict(working_memory)
        fired: list[str] = []
        for _ in range(self.max_iterations):
            progress = False
            for rule in sorted(self.rules, key=lambda r: -r.priority):
                if rule.name in fired:
                    continue
                if rule.condition(wm):
                    rule.action(wm)
                    fired.append(rule.name)
                    progress = True
            if not progress:
                break
        wm["_fired_rules"] = fired
        return wm


class Unifier(DeterministicPrimitive):
    """Simple first-order unification over :class:`Fact` objects.

    Variable positions in a pattern are indicated by strings starting
    with ``?`` (e.g. ``Fact("parent", ("?X", "alice"))``).
    """

    def forward(
        self,
        pattern: Fact,
        facts: list[Fact],
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for fact in facts:
            if fact.predicate != pattern.predicate:
                continue
            if len(fact.args) != len(pattern.args):
                continue
            bindings: dict[str, Any] = {}
            ok = True
            for p_arg, f_arg in zip(pattern.args, fact.args):
                if isinstance(p_arg, str) and p_arg.startswith("?"):
                    if p_arg in bindings and bindings[p_arg] != f_arg:
                        ok = False
                        break
                    bindings[p_arg] = f_arg
                elif p_arg != f_arg:
                    ok = False
                    break
            if ok:
                results.append(bindings)
        return results


class ConstraintSolver(DeterministicPrimitive):
    """Backtracking constraint solver over finite domains."""

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.constraints: list[Constraint] = list(config.get("constraints", []))

    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)

    def forward(self, domains: dict[str, list[Any]]) -> list[dict[str, Any]]:
        variables = list(domains.keys())
        solutions: list[dict[str, Any]] = []
        self._backtrack(variables, domains, {}, solutions)
        return solutions

    def _backtrack(
        self,
        variables: list[str],
        domains: dict[str, list[Any]],
        assignment: dict[str, Any],
        solutions: list[dict[str, Any]],
    ) -> None:
        if len(assignment) == len(variables):
            if all(c.check(assignment) for c in self.constraints):
                solutions.append(dict(assignment))
            return
        var = variables[len(assignment)]
        for val in domains[var]:
            assignment[var] = val
            consistent = True
            for c in self.constraints:
                if all(v in assignment for v in c.variables):
                    if not c.check(assignment):
                        consistent = False
                        break
            if consistent:
                self._backtrack(variables, domains, assignment, solutions)
            del assignment[var]


class CycleDetect(DeterministicPrimitive):
    """Detect cycles in a directed graph via DFS back-edge detection.

    Input: adjacency list ``dict[str, list[str]]``.
    Output: ``{"has_cycle": bool, "cycles": list[list[str]]}``.
    """

    def forward(self, graph: dict[str, list[str]]) -> dict[str, Any]:
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {v: WHITE for v in graph}
        parent: dict[str, str | None] = {v: None for v in graph}
        cycles: list[list[str]] = []

        def dfs(u: str) -> None:
            color[u] = GRAY
            for v in graph.get(u, []):
                if v not in color:
                    color[v] = WHITE
                if color[v] == GRAY:
                    cycle = [v, u]
                    node = u
                    while node != v:
                        node = parent[node]  # type: ignore[assignment]
                        if node is None:
                            break
                        cycle.append(node)
                    cycle.reverse()
                    cycles.append(cycle)
                elif color[v] == WHITE:
                    parent[v] = u
                    dfs(v)
            color[u] = BLACK

        for v in graph:
            if color[v] == WHITE:
                dfs(v)

        return {"has_cycle": len(cycles) > 0, "cycles": cycles}


class TopologicalSort(DeterministicPrimitive):
    """Topological sort of a DAG (Kahn's algorithm).

    Input: adjacency list ``dict[str, list[str]]``.
    Output: ``{"sorted": list[str], "is_dag": bool}``.
    """

    def forward(self, graph: dict[str, list[str]]) -> dict[str, Any]:
        all_nodes: set[str] = set(graph.keys())
        for targets in graph.values():
            all_nodes.update(targets)

        in_degree: dict[str, int] = {v: 0 for v in all_nodes}
        for u, targets in graph.items():
            for v in targets:
                in_degree[v] = in_degree.get(v, 0) + 1

        queue = sorted([v for v, d in in_degree.items() if d == 0])
        result: list[str] = []

        while queue:
            node = queue.pop(0)
            result.append(node)
            for v in graph.get(node, []):
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    bisect.insort(queue, v)

        return {
            "sorted": result,
            "is_dag": len(result) == len(all_nodes),
        }

    @staticmethod
    def levels(graph: dict[str, list[str]]) -> list[list[str]]:
        """Partition nodes into Kahn *waves*: within each batch, no two tasks depend on each other.

        Caller must ensure ``graph`` is a DAG (e.g. after :class:`CycleDetect`).
        Raises ``RuntimeError`` if a cycle is present.
        """
        all_nodes: set[str] = set(graph.keys())
        for targets in graph.values():
            all_nodes.update(targets)

        in_degree: dict[str, int] = {v: 0 for v in all_nodes}
        for u, targets in graph.items():
            for v in targets:
                in_degree[v] = in_degree.get(v, 0) + 1

        remaining = set(all_nodes)
        waves: list[list[str]] = []
        while remaining:
            ready = sorted([v for v in remaining if in_degree[v] == 0])
            if not ready:
                raise RuntimeError("TopologicalSort.levels: graph has a cycle")
            waves.append(list(ready))
            for v in ready:
                remaining.remove(v)
                for w in graph.get(v, []):
                    in_degree[w] -= 1
        return waves


topological_levels = TopologicalSort.levels


class KnowledgeBase(DeterministicPrimitive):
    """Fact store with assertion, retraction, and querying."""

    def __init__(self, name: str | None = None, **config: Any) -> None:
        super().__init__(name, **config)
        self.facts: set[Fact] = set(config.get("facts", set()))

    def assert_fact(self, fact: Fact) -> None:
        self.facts.add(fact)

    def retract(self, fact: Fact) -> None:
        self.facts.discard(fact)

    def query(self, predicate: str | None = None) -> list[Fact]:
        if predicate is None:
            return list(self.facts)
        return [f for f in self.facts if f.predicate == predicate]

    def forward(self, query_predicate: str | None = None) -> list[Fact]:
        return self.query(query_predicate)


class InitScratchpadStateManagement(InitScratchpad):
    pass
