"""§6.4: choose the gold-annotation set and pre-fill it with pipeline output.

§6.4 asks for ≥150 cases annotated by a legal reviewer, with a lightweight tool pre-filled with
model output for correction. **Which 150 is the decision that matters**, because annotation time is
the scarcest resource on this project and a uniform random sample would spend most of it confirming
things we already believe.

The set is therefore built from four strata, each buying something different:

  `conflict`   (40) cases where the rules and LLM labellers disagreed and the label is UNRESOLVED.
               These resolve §5.1's outstanding human check AND return 40 cases to the usable pool.
               Highest information per annotation: we currently have no label at all for them.
  `stratified` (60) stratified across decade and claim family, the backbone needed for §6.1's
               extraction F1 and §6.2's canonical-label accuracy to be measurable at all.
  `error`      (30) cases the best model got wrong but that trip none of §10.3's deterministic
               signals. §10.3 could not explain these and an LLM's opinion on them was discounted;
               a reviewer can say whether the facts actually determine the outcome.
  `agreement`  (30) a deliberate overlap block, flagged for a second annotator so §6.4's
               inter-annotator agreement can be computed. Drawn from `stratified` so agreement is
               measured on ordinary cases rather than hard ones.

Everything a reviewer needs is pre-filled from the pipeline so the task is **correction, not
authoring**: extracted facts with their spans and canonical labels, claims with their families, the
outcome label with the sentence it came from, and the statutes found. Each field carries what the
pipeline thinks and an empty slot for what the reviewer decides, so disagreement is recorded rather
than overwritten.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from src import paths

SEED = 573
OUT = paths.GOLD / "gold_tasks.jsonl"
PLAN = paths.GOLD / "gold_plan.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-conflict", type=int, default=40)
    ap.add_argument("--n-stratified", type=int, default=80)
    ap.add_argument("--n-error", type=int, default=30)
    ap.add_argument("--n-agreement", type=int, default=30)
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)

    final = {r["doc_id"]: r for r in
             (json.loads(l) for l in open(paths.INTERIM / "outcome_labels_final.jsonl"))}
    reg = {r["doc_id"]: r for r in (json.loads(l) for l in open(paths.CASE_REGISTRY))}
    masked = {json.loads(l)["doc_id"]: json.loads(l)
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    facts = defaultdict(list)
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        facts[r["case_id"]].append(r)
    canon = {}
    for line in open(paths.INTERIM / "canonical_facts.jsonl"):
        r = json.loads(line)
        canon[r["fact_id"]] = r
    claims = defaultdict(list)
    for line in open(paths.INTERIM / "claims.jsonl"):
        r = json.loads(line)
        claims[r["case_id"]].append(r)
    fam = json.loads((paths.INTERIM / "claim_families.json").read_text())["membership"]
    merge = json.loads((paths.INTERIM / "family_merge.json").read_text())["family_to_merged"]
    statutes = {r["doc_id"]: r for r in
                (json.loads(l) for l in open(paths.INTERIM / "statute_targets.jsonl"))}
    conflicts = [r["doc_id"] for r in
                 (json.loads(l) for l in open(paths.INTERIM / "label_conflicts.jsonl"))]
    screen = {r["doc_id"]: r for r in
              (json.loads(l) for l in open(paths.INTERIM / "case_screen.jsonl"))}

    # errors §10.3 could not explain
    ea_path = paths.EXPERIMENTS / "paper3" / "error_analysis_forum_heldout.json"
    unexplained = []
    if ea_path.exists():
        ea = json.loads(ea_path.read_text())
        unexplained = [c for c, d in ea["per_case"].items()
                       if not any(d["signals"].values())]

    def pick(pool, k, label, need_facts: bool = True):
        # The `conflict` stratum has NO extracted facts: extraction eligibility required a binary
        # outcome label, and these cases are UNRESOLVED precisely because the labellers disagreed.
        # That is fine for their purpose -- the reviewer resolves the outcome from the order
        # evidence, which does not need facts -- so they are admitted without the facts filter and
        # their task carries no fact-review block.
        pool = [c for c in pool if masked.get(c) and (not need_facts or c in facts)]
        if len(pool) <= k:
            return pool
        idx = rng.choice(len(pool), k, replace=False)
        return [pool[i] for i in sorted(idx)]

    chosen: dict[str, str] = {}

    for c in pick(conflicts, args.n_conflict, "conflict", need_facts=False):
        chosen[c] = "conflict"

    # stratified: spread over decade x merged family
    eligible = [c for c in facts
                if screen.get(c, {}).get("is_property")
                and final.get(c, {}).get("outcome") in ("WIN", "LOSE")
                and c not in chosen]
    cells = defaultdict(list)
    for c in eligible:
        dec = (reg[c]["year"] // 10) * 10
        prim = fam.get(c, {}).get("primary")
        cells[(dec, merge.get(prim, prim) or "_none")].append(c)
    keys = sorted(cells)
    per = max(1, args.n_stratified // max(1, len(keys)))
    strat = []
    for k in keys:
        strat += pick(cells[k], per, "stratified")
    rng.shuffle(strat)
    strat = strat[:args.n_stratified]
    # Per-cell allocation under-fills when there are more cells than slots (integer division gives
    # 1 per cell and some cells are empty), so top up from the remaining pool to hit the target.
    if len(strat) < args.n_stratified:
        rest = [c for c in eligible if c not in chosen and c not in set(strat)]
        strat += pick(rest, args.n_stratified - len(strat), "stratified")
    for c in strat:
        chosen[c] = "stratified"

    for c in pick([c for c in unexplained if c not in chosen], args.n_error, "error"):
        chosen[c] = "error"

    # agreement block: a second pass over ordinary cases already in the set
    strat_in = [c for c, s in chosen.items() if s == "stratified"]
    agree = set(pick(strat_in, min(args.n_agreement, len(strat_in)), "agreement"))

    # --- write tasks
    n = 0
    with OUT.open("w") as out:
        for c, stratum in sorted(chosen.items()):
            fl = facts[c]
            prim = fam.get(c, {}).get("primary")
            task = {
                "case_id": c,
                "stratum": stratum,
                "second_annotator": c in agree,
                "title": reg[c]["title"],
                "year": reg[c]["year"],
                "url": reg[c]["url"],
                "masked_text": masked[c]["masked_text"],
                "prior_court_text": masked[c].get("prior_court_text", ""),

                # --- what the pipeline believes, each paired with an empty reviewer slot
                "outcome": {
                    "pipeline": final[c]["outcome"],
                    "source": final[c]["source"],
                    "evidence": final[c].get("order_evidence"),
                    "rules_said": final[c].get("rules_outcome"),
                    "llm_said": final[c].get("llm_outcome"),
                    "reviewer": None,            # WIN | LOSE | PARTIAL | REMAND | OTHER | UNCLEAR
                    "reviewer_note": None,
                },
                "claim_family": {
                    "pipeline_primary": merge.get(prim, prim),
                    "pipeline_all": sorted({merge.get(f, f)
                                            for f in fam.get(c, {}).get("families", [])}),
                    "reviewer": None,
                    "reviewer_note": None,
                },
                "claims": [
                    {"text": x["text"], "kind": x["kind"], "raised_by": x["raised_by"],
                     "pipeline_family": x["family"], "relief_sought": x.get("relief_sought"),
                     "reviewer_correct": None,   # true | false
                     "reviewer_family": None}
                    for x in claims.get(c, [])[:12]
                ],
                # A conflict-stratum case has no extracted facts; its task is outcome-only.
                "facts": [] if stratum == "conflict" else [
                    {"fact_id": f["fact_id"],
                     "text": f["text"],
                     "quote": f["quote"],
                     "source_span": f["source_span"],
                     "pipeline_asserted_by": f["asserted_by"],
                     "pipeline_disputed_status": f["disputed_status"],
                     "pipeline_label": canon.get(f["fact_id"], {}).get("label"),
                     "pipeline_label_is_new": canon.get(f["fact_id"], {}).get("is_new"),
                     # §6.1 asks for extraction P/R/F1, so a reviewer must be able to say a fact is
                     # wrong AND to add one the extractor missed (see `missed_facts` below)
                     "reviewer_is_a_fact": None,        # true | false
                     "reviewer_asserted_by": None,
                     "reviewer_label": None,
                     "reviewer_note": None}
                    for f in fl[:40]
                ],
                "task_scope": ("outcome only (no facts extracted: this case was UNRESOLVED, so it "
                               "never entered the extraction pool)" if stratum == "conflict"
                               else "full (outcome, family, claims, facts, statutes)"),
                "missed_facts": [],   # reviewer adds facts the extractor failed to find
                "statutes": {
                    "pipeline_all": statutes.get(c, {}).get("labels_all", [])[:20],
                    "pipeline_novel": statutes.get(c, {}).get("labels_novel", [])[:20],
                    "reviewer_correct": None,
                    "reviewer_missing": [],
                },
                "annotation_status": "pending",
            }
            out.write(json.dumps(task) + "\n")
            n += 1

    plan = {
        "n_tasks": n,
        "strata": dict(Counter(chosen.values())),
        "second_annotator_block": len(agree),
        "facts_to_review": sum(min(40, len(facts.get(c, []))) for c in chosen),
        "outcome_only_tasks": sum(1 for s in chosen.values() if s == "conflict"),
        "why_these_cases": {
            "conflict": "rules and LLM labellers disagreed; no usable label exists. Resolves §5.1's "
                        "human check and returns these cases to the eligible pool.",
            "stratified": "decade x merged claim family. The backbone for §6.1 extraction F1 and "
                          "§6.2 canonical-label accuracy.",
            "error": "best model wrong AND tripping none of §10.3's deterministic signals. §10.3 "
                     "could not explain these; a reviewer can say whether the facts determine the "
                     "outcome at all.",
            "agreement": "overlap block for §6.4's inter-annotator agreement, drawn from the "
                         "ordinary stratum so agreement is measured on typical cases.",
        },
        "meets_6_4_minimum": n >= 150,
    }
    PLAN.write_text(json.dumps(plan, indent=2))

    print(f"gold tasks: {n} cases -> {OUT}")
    print(f"  strata: {plan['strata']}")
    print(f"  second-annotator overlap block: {len(agree)}")
    print(f"  fact judgements to make: {plan['facts_to_review']:,}")
    print(f"  §6.4 minimum of 150 cases: {'met' if plan['meets_6_4_minimum'] else 'NOT met'}")
    print(f"  -> {PLAN}")


if __name__ == "__main__":
    main()
