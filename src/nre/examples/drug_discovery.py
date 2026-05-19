"""Drug-discovery screening logic for candidate selection logic that removes toxic chemicals using core CoreThink primitives.

Demonstrates a simplified virtual screen using only the core library:

1. KnowledgeBase stores targets, compounds, properties, and known bindings.
2. Embedder + Similarity score compound-target affinity.
3. RuleEngine enforces Lipinski rule-of-five safety filters.
4. Unifier queries the KB for properties and known interactions.
5. Survivors are ranked and returned.

Run directly::

    python -m nre.examples.drug_discovery
"""

from __future__ import annotations

from nre.dsl import Engine
from nre.library import (
    Embedder,
    Similarity,
    KnowledgeBase,
    RuleEngine,
    Unifier,
)
from nre.primitives import Fact, Rule


def build_engine() -> Engine:
    engine = Engine()

    # ── Register core primitives ──────────────────────────────────
    engine.register(Embedder, name="embed", dim=32)
    engine.register(Similarity, name="similarity")
    engine.register(KnowledgeBase, name="kb")
    engine.register(RuleEngine, name="rules")
    engine.register(Unifier, name="unify")

    # ── Populate knowledge base ───────────────────────────────────
    kb: KnowledgeBase = engine.registry.get("kb")  # type: ignore[assignment]

    kb.assert_fact(Fact("target", ("EGFR",)))

    compounds = {
        "erlotinib":  {"mw": 393.4, "logp": 3.3, "hbd": 1, "hba": 7, "toxic": False},
        "gefitinib":  {"mw": 446.9, "logp": 4.1, "hbd": 1, "hba": 7, "toxic": False},
        "afatinib":   {"mw": 485.9, "logp": 3.8, "hbd": 2, "hba": 8, "toxic": False},
        "compound_x": {"mw": 620.0, "logp": 6.2, "hbd": 6, "hba": 3, "toxic": True},
        "compound_y": {"mw": 280.3, "logp": 1.9, "hbd": 2, "hba": 5, "toxic": False},
    }
    for name, props in compounds.items():
        kb.assert_fact(Fact("compound", (name,)))
        for prop, val in props.items():
            kb.assert_fact(Fact(prop, (name, val)))

    kb.assert_fact(Fact("binds", ("erlotinib", "EGFR")))
    kb.assert_fact(Fact("binds", ("gefitinib", "EGFR")))

    # ── Safety rules (Lipinski + toxicity) ────────────────────────
    rule_engine: RuleEngine = engine.registry.get("rules")  # type: ignore[assignment]

    def _reject(reason: str):
        def action(wm: dict) -> None:
            wm["reject"] = True
            wm.setdefault("reject_reasons", []).append(reason)
        return action

    rule_engine.add_rule(Rule("reject_toxic",
        condition=lambda wm: wm.get("toxic") is True,
        action=_reject("toxic"), priority=10))
    rule_engine.add_rule(Rule("lipinski_mw",
        condition=lambda wm: wm.get("mw", 0) > 500,
        action=_reject("MW > 500"), priority=5))
    rule_engine.add_rule(Rule("lipinski_logp",
        condition=lambda wm: wm.get("logp", 0) > 5,
        action=_reject("LogP > 5"), priority=5))
    rule_engine.add_rule(Rule("lipinski_hbd",
        condition=lambda wm: wm.get("hbd", 0) > 5,
        action=_reject("HBD > 5"), priority=5))
    rule_engine.add_rule(Rule("lipinski_hba",
        condition=lambda wm: wm.get("hba", 0) > 10,
        action=_reject("HBA > 10"), priority=5))

    # ── Domain logic ──────────────────────────────────────────────

    @engine.domain("screen")
    def screen(ctx):
        target  = ctx.let[str]
        hits    = ctx.let[list] << []
        rejects = ctx.let[list] << []

        target << ctx._inputs["target"]

        # Embed the target
        target_emb = target >> ctx.embed

        # Get all compounds from KB
        compound_facts = ctx.unify(Fact("compound", ("?name",)), list(ctx.kb.facts))

        # Screen each compound
        with ctx.each(compound_facts) as loop:
            for binding in loop:
                name = binding["?name"]

                # Look up properties via unification
                def _prop(pred: str):
                    m = ctx.unify(Fact(pred, (name, "?val")), list(ctx.kb.facts))
                    return m[0]["?val"] if m else None

                # Run safety rules
                wm = ctx.rules({
                    "compound": name,
                    "mw": _prop("mw"), "logp": _prop("logp"),
                    "hbd": _prop("hbd"), "hba": _prop("hba"),
                    "toxic": _prop("toxic"),
                })

                with ctx.when(wm.get("reject", False)) as rejected:
                    if rejected:
                        rejects.val.append({
                            "compound": name,
                            "reasons": wm.get("reject_reasons", []),
                        })

                with ctx.otherwise as passed:
                    if passed:
                        # Score affinity via embedding similarity
                        compound_emb = ctx.embed(name)
                        affinity = ctx.similarity(target_emb.value, compound_emb)

                        # Bonus for known binders
                        known = ctx.unify(
                            Fact("binds", (name, target.val)),
                            list(ctx.kb.facts),
                        )
                        with ctx.when(len(known) > 0) as is_known:
                            if is_known:
                                affinity = min(affinity + 0.2, 1.0)

                        hits.val.append({
                            "compound": name,
                            "affinity": round(affinity, 4),
                            "known_binder": len(known) > 0,
                        })

        # Rank hits by affinity
        hits << sorted(hits.val, key=lambda h: -h["affinity"])

        return ctx.returns(
            target=target,
            hits=hits,
            rejects=rejects,
        )

    return engine


def main() -> None:
    engine = build_engine()

    print("=== CoreThink Drug Discovery Screen ===\n")

    result = engine.run("screen", target="EGFR")

    print(f"Target: {result['target']}\n")

    print("Hits (ranked by affinity):")
    for h in result["hits"]:
        known = " [known binder]" if h["known_binder"] else ""
        print(f"  {h['compound']:15s}  affinity={h['affinity']:.4f}{known}")

    print(f"\nRejected ({len(result['rejects'])}):")
    for r in result["rejects"]:
        reasons = ", ".join(r["reasons"])
        print(f"  {r['compound']:15s}  reasons: {reasons}")

    print()


if __name__ == "__main__":
    main()
