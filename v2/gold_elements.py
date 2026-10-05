"""Build an element-satisfaction annotation set from the 150 prepared gold cases.

S4 scores 0.521 as a feature layer and fails its acceptance test, but its value was never
predictive -- it is the only layer in either version of this project whose output is *checkable*.
What has never been measured is whether its verdicts are **right**. That needs human labels, and
blocker B4's 150 cases were prepared for a different task (extraction), so they are repurposed here.

## Design decisions, each costing reviewer time if got wrong

**Grouped by case, not by item.** The expensive part of this task is reading a case's facts, not
deciding one element. A reviewer judging nine elements of one case reads the facts once; nine items
drawn from nine cases means nine readings. So the unit of work is a case with all of its sampled
elements.

**UNCLEAR is subsampled, the decisive classes are not.** 63% of S4's verdicts are UNCLEAR and they
are the least informative to check -- confirming "the facts do not address this" nine times teaches
little. Every SATISFIED and every NOT_SATISFIED on a selected case is included; UNCLEAR is capped per
case. NOT_SATISFIED is only 3.7% of the corpus and is the verdict the architecture's "or defeat" half
depends on, so cases carrying one are selected first.

**Blind.** The reviewer never sees S4's verdict before giving their own. Anchoring would turn an
agreement measurement into a confirmation exercise. The annotation app reveals the model's answer
only after submission, and the stored record keeps both.

**Quotes are selected, not typed.** The reviewer picks the fact NUMBER that supports their verdict.
One click instead of a transcription, and the quote is exact by construction, so the same mechanical
gate that scores S4 scores the human.

## The number this produces, and what to compare it against

Per-verdict agreement between reviewer and S4, plus precision and recall for SATISFIED and
NOT_SATISFIED separately.

Compare the disagreement rate against **3.1%**, which is S4's measured disagreement *with itself*
across two independent runs at temperature 0 (see ARCHITECTURE.md §6b). A human-model disagreement
near 3% would mean the model agrees with the reviewer about as well as it agrees with itself, and no
amount of further prompting would close it. That floor is why the accidental double-run was worth
keeping.
"""
from __future__ import annotations

import argparse
import collections
import json
import random

from v2 import paths
from v2.element_catalogue import load_claims

SEED = 573
GOLD = paths.ROOT.parent / "Data" / "gold" / "gold_tasks.jsonl"
VERDICTS = paths.INTERIM / "element_verdicts_element_satisfy_v2_facts.jsonl"
OUT = paths.ROOT.parent / "Data" / "gold" / "element_gold_tasks.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=60)
    ap.add_argument("--unclear-per-case", type=int, default=2)
    ap.add_argument("--satisfied-per-case", type=int, default=3,
                    help="cap SATISFIED per case too. 208 of them is more precision than the "
                         "estimate needs and the reviewer time is the scarce resource.")
    args = ap.parse_args()

    claims = {c.claim_id: c for c in load_claims()}
    el = {e.element_id: e for c in claims.values() for e in c.elements}

    gold = {g["case_id"]: g for g in (json.loads(l) for l in open(GOLD))}
    s4 = collections.defaultdict(list)
    for line in open(VERDICTS):
        r = json.loads(line)
        if r["case_id"] in gold:
            s4[r["case_id"]].append(r)

    # cases carrying a NOT_SATISFIED first -- it is the rarest and the most load-bearing
    def n_neg(c):
        return sum(1 for r in s4[c] for d in r["verdicts"] if d["verdict"] == "NOT_SATISFIED")
    rng = random.Random(SEED)
    order = sorted(s4, key=lambda c: (-n_neg(c), rng.random()))
    picked = order[: args.cases]

    tasks, counts = [], collections.Counter()
    for c in picked:
        g = gold[c]
        facts = [f["text"] if isinstance(f, dict) else str(f) for f in (g.get("facts") or [])]
        if not facts:
            continue
        items = []
        kept = collections.Counter()
        for r in s4[c]:
            for d in r["verdicts"]:
                cap = {"UNCLEAR": args.unclear_per_case,
                       "SATISFIED": args.satisfied_per_case}.get(d["verdict"])
                if cap is not None:
                    if kept[d["verdict"]] >= cap:
                        continue
                    kept[d["verdict"]] += 1
                e = el.get(d["element_id"])
                if e is None:
                    continue
                items.append({
                    "element_id": e.element_id, "element_name": e.name,
                    "element_definition": e.definition,
                    "claim_id": e.claim_id, "claim_name": e.claim_name,
                    "claim_issue": claims[e.claim_id].issue,
                    "model_verdict": d["verdict"], "model_quote": d.get("quote", ""),
                    "model_favours": d.get("favours", "neither")})
                counts[d["verdict"]] += 1
        if items:
            tasks.append({"case_id": c, "stratum": g.get("stratum"), "title": g.get("title"),
                          "year": g.get("year"), "url": g.get("url"),
                          "outcome": g.get("outcome"), "facts": facts, "items": items,
                          "annotation_status": "pending"})

    OUT.write_text("".join(json.dumps(t) + "\n" for t in tasks))
    n_items = sum(len(t["items"]) for t in tasks)
    print(f"-> {OUT}")
    print(f"  {len(tasks)} cases, {n_items} element judgements")
    print(f"  by model verdict: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print(f"  facts to read: {sum(len(t['facts']) for t in tasks):,} across {len(tasks)} cases "
          f"(median {sorted(len(t['facts']) for t in tasks)[len(tasks)//2]} per case)")
    mins = (n_items * 15 + len(tasks) * 60) / 60
    print(f"\n  reviewer budget at ~15 s per judgement plus ~60 s reading each case: "
          f"**{mins:.0f} minutes**")
    import math
    for cls, n in counts.most_common():
        if n:
            hw = 1.96 * math.sqrt(0.8 * 0.2 / n)
            print(f"    {cls:14s} n={n:3d}  ->  95% CI half-width ~±{hw:.3f} on a precision of 0.80")
    print("  The app writes after every case, so a partial pass is still scoreable --")
    print("  stop whenever and run the scorer on what exists.")
    print(f"\n  annotate with:  .venv/bin/streamlit run v2/annotate_elements.py")
    print(f"  then score with: .venv/bin/python -m v2.score_gold_elements")


if __name__ == "__main__":
    main()
