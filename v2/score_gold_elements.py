"""Score S4's element verdicts against the human annotations.

The headline is not a single accuracy. S4's verdict classes are wildly unbalanced (64% UNCLEAR), so
an overall agreement rate is dominated by the easy class. What matters:

  precision on SATISFIED      of the elements S4 called satisfied, how many does the reviewer agree
                              are satisfied? This is the number a user of the system depends on.
  recall on SATISFIED         of the elements the reviewer calls satisfied, how many did S4 find?
  the same pair for NOT_SATISFIED, which the architecture's "or defeat" half rests on.
  quote agreement             when both say SATISFIED, did they point at the same fact? A right
                              verdict from the wrong evidence is a different failure from a wrong
                              verdict, and only this separates them.

**The comparison that interprets all of it:** S4's disagreement with ITSELF across two independent
runs at temperature 0 is **3.1%** (ARCHITECTURE.md §6b). If human-model disagreement approaches that
floor, the model agrees with the reviewer about as well as it agrees with itself, and prompt work
cannot close the remainder. If it is far above, there is real headroom.
"""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

ANN = Path("Data/gold/element_gold_annotations.jsonl")
SELF_DISAGREEMENT = 0.031
V = ("SATISFIED", "NOT_SATISFIED", "UNCLEAR")


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    if not ANN.exists():
        raise SystemExit(f"{ANN} missing — nothing annotated yet.\n"
                         f"Run: .venv/bin/streamlit run v2/annotate_elements.py")
    latest = {}
    for line in open(ANN):
        r = json.loads(line)
        latest[(r["case_id"], r.get("annotator", ""))] = r

    js = [j for r in latest.values() for j in r["judgements"]]
    annots = sorted({a for (_, a) in latest})
    print(f"=== S4 element verdicts vs human annotation ===")
    print(f"annotators: {', '.join(annots)}   cases: {len(latest)}   judgements: {len(js)}\n")
    if not js:
        return

    agree = sum(1 for j in js if j["human_verdict"] == j["model_verdict"])
    lo, hi = wilson(agree, len(js))
    print(f"  overall agreement {agree}/{len(js)} = {agree/len(js):.1%}  "
          f"95% CI [{lo:.1%}, {hi:.1%}]")
    print(f"  (dominated by the UNCLEAR class -- read the per-class rows below)\n")

    print("  confusion, rows = human, cols = model:")
    M = collections.Counter((j["human_verdict"], j["model_verdict"]) for j in js)
    print(f"    {'':16s}" + "".join(f"{c[:9]:>16s}" for c in V))
    for h in V:
        print(f"    {h:16s}" + "".join(f"{M[(h,c)]:>16d}" for c in V))

    print("\n  per class:")
    for c in V:
        tp = M[(c, c)]
        mp = sum(M[(h, c)] for h in V)
        hp = sum(M[(c, m)] for m in V)
        if not mp and not hp:
            continue
        pl, ph = wilson(tp, mp) if mp else (0, 0)
        rl, rh = wilson(tp, hp) if hp else (0, 0)
        print(f"    {c:14s} precision {tp}/{mp} = {tp/mp if mp else 0:.1%} [{pl:.0%},{ph:.0%}]"
              f"   recall {tp}/{hp} = {tp/hp if hp else 0:.1%} [{rl:.0%},{rh:.0%}]")

    both = [j for j in js if j["human_verdict"] == j["model_verdict"] != "UNCLEAR"]
    if both:
        same = sum(1 for j in both
                   if j["human_quote"] and j["human_quote"].strip()[:60].lower()
                   in (j["model_quote"] or "").lower()
                   or (j["model_quote"] or "").strip()[:60].lower()
                   in j["human_quote"].lower())
        print(f"\n  same evidence when both agree on a non-UNCLEAR verdict: "
              f"{same}/{len(both)} = {same/len(both):.1%}")
        print("    (a right verdict from the wrong fact is a different failure from a wrong verdict)")

    dis = 1 - agree / len(js)
    print(f"\n  INTERPRETATION")
    print(f"    human-model disagreement      {dis:.1%}")
    print(f"    S4's disagreement with ITSELF {SELF_DISAGREEMENT:.1%}  (two runs, temperature 0)")
    if dis <= SELF_DISAGREEMENT * 1.5:
        print("    -> at the self-consistency floor. The model agrees with the reviewer about as")
        print("       well as it agrees with itself; prompt work cannot close the remainder.")
    else:
        print(f"    -> {dis/SELF_DISAGREEMENT:.1f}x the floor, so there is real headroom that is")
        print("       not explained by the model's own instability.")

    out = Path("v2/experiments/s4_gold_agreement.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": "S4 vs human gold", "annotators": annots, "n_cases": len(latest),
        "n_judgements": len(js), "overall_agreement": round(agree / len(js), 4),
        "overall_ci95": [round(lo, 4), round(hi, 4)],
        "confusion": {f"{h}|{m}": n for (h, m), n in M.items()},
        "per_class": {c: {"precision_n": sum(M[(h, c)] for h in V),
                          "recall_n": sum(M[(c, m)] for m in V),
                          "tp": M[(c, c)]} for c in V},
        "self_disagreement_reference": SELF_DISAGREEMENT}, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
