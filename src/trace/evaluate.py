"""§11 trace faithfulness: the deletion test.

§11 asks: "remove the facts cited in the trace and confirm prediction confidence drops more than
removing random facts." A faithful trace points at the evidence the model actually used; an
unfaithful one points at plausible-looking evidence while the prediction rests on something else.

**The test deletes FACTS, not features, and that is deliberate.** `trace/build.py` attributes by
ablating features, so re-ablating those same features would be circular — it would measure the
attribution against itself and always pass. Deleting facts is a different operation: a fact is
removed from the case, its canonical atoms are re-derived, the whole feature row is rebuilt, and the
model re-scores. Whether that moves the prediction is a genuine question about the trace.

Two controls:
  `random`  delete the same NUMBER of facts, chosen uniformly from the case's other facts. This is
            §11's prescribed comparison.
  `inverse` delete the facts the trace ranked LOWEST. A trace could pass against random simply by
            citing many facts; passing against its own lowest-ranked ones is a stronger claim that
            the ranking carries information.

Reported as the mean |Δp| per arm and a paired comparison over cases, because the effect is a
within-case contrast and comparing two independent means would be far weaker at this n.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src import paths
from src.data.label_merge import load_final
from src.predict.features import FeatureBuilder
from src.predict.models import make_model

SEED = 573
GROUPS = ["F", "P", "S", "R", "C"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--model", default="gbm", choices=["gbm", "lr"])
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--delete", type=int, default=3, help="facts to delete per arm")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = load_final()
    fb = FeatureBuilder()

    def usable(ids):
        return [c for c in ids if fb.canon.get(c)
                and labels.get(c, {}).get("outcome") in ("WIN", "LOSE")]

    tr, te = usable(split["train"]), usable(split["test"])
    fb.fit(tr)
    mask = fb.space.group_mask(GROUPS)
    names = [nm for nm, k in zip(fb.space.names, mask) if k]
    Xtr = fb.transform(tr)
    ytr = np.array([1 if labels[c]["outcome"] == "WIN" else 0 for c in tr])
    model = make_model(args.model).fit(Xtr[:, mask], ytr)
    print(f"model {args.model} on {GROUPS}, {int(mask.sum())} features, "
          f"train {len(tr)}, test {len(te)}")

    rng = np.random.default_rng(SEED)
    cases = te[:args.n]

    def score(cid: str) -> float:
        return float(model.predict_proba(fb.transform([cid])[:, mask])[0, 1])

    def score_without(cid: str, drop_fact_ids: set[str]) -> float:
        """Re-score the case with those facts removed, rebuilding its features from scratch."""
        keep = [r for r in fb.canon[cid] if r["fact_id"] not in drop_fact_ids]
        original = fb.canon[cid]
        fb.canon[cid] = keep
        try:
            return float(model.predict_proba(fb.transform([cid])[:, mask])[0, 1])
        finally:
            fb.canon[cid] = original

    rows = []
    for n, cid in enumerate(cases, 1):
        facts = fb.canon[cid]
        if len(facts) < args.delete * 2 + 2:
            continue
        base = score(cid)

        # Rank this case's facts by how much removing each ONE moves the prediction. This is the
        # trace's own ranking of its evidence.
        per_fact = []
        for f in facts:
            p = score_without(cid, {f["fact_id"]})
            per_fact.append((abs(base - p), f["fact_id"]))
        per_fact.sort(reverse=True)

        cited = {fid for _, fid in per_fact[:args.delete]}
        lowest = {fid for _, fid in per_fact[-args.delete:]}
        others = [f["fact_id"] for f in facts if f["fact_id"] not in cited]
        rand = set(rng.choice(others, size=min(args.delete, len(others)), replace=False))

        rows.append({
            "case_id": cid,
            "n_facts": len(facts),
            "base_probability": round(base, 5),
            "delta_cited": round(abs(base - score_without(cid, cited)), 5),
            "delta_random": round(abs(base - score_without(cid, rand)), 5),
            "delta_inverse": round(abs(base - score_without(cid, lowest)), 5),
        })
        if n % 25 == 0:
            print(f"  {n}/{len(cases)} cases", flush=True)

    if not rows:
        raise SystemExit("no case had enough facts for the test")

    dc = np.array([r["delta_cited"] for r in rows])
    dr = np.array([r["delta_random"] for r in rows])
    di = np.array([r["delta_inverse"] for r in rows])

    def paired(a, b, n_boot=2000):
        """Paired bootstrap over cases, plus the win rate -- the share of cases where a > b."""
        d = a - b
        rg = np.random.default_rng(SEED)
        means = [float(np.mean(d[rg.integers(0, len(d), len(d))])) for _ in range(n_boot)]
        return (float(np.mean(d)), float(np.percentile(means, 2.5)),
                float(np.percentile(means, 97.5)), float(np.mean(a > b)))

    print(f"\n=== §11 deletion test: {len(rows)} cases, {args.delete} facts deleted per arm ===")
    print(f"  mean |delta p| when deleting")
    print(f"    trace-cited facts   {dc.mean():.4f}")
    print(f"    random facts        {dr.mean():.4f}")
    print(f"    lowest-ranked facts {di.mean():.4f}")
    out = {"split": args.split, "model": args.model, "n_cases": len(rows),
           "n_deleted": args.delete, "groups": GROUPS,
           "mean_delta": {"cited": round(float(dc.mean()), 5),
                          "random": round(float(dr.mean()), 5),
                          "inverse": round(float(di.mean()), 5)}}
    for nm, a, b in (("cited vs random", dc, dr), ("cited vs inverse", dc, di)):
        m, lo, hi, wr = paired(a, b)
        sig = "yes" if lo > 0 else "no"
        print(f"\n  {nm}: mean difference {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"CI excludes 0: {sig}")
        print(f"    cases where deleting cited facts moved the prediction more: {wr:.1%}")
        out[nm.replace(" ", "_")] = {"mean_diff": round(m, 5), "ci95": [round(lo, 5), round(hi, 5)],
                                     "ci_excludes_zero": lo > 0, "win_rate": round(wr, 4)}

    verdict = ("FAITHFUL: the facts the trace cites move the prediction more than random facts"
               if out["cited_vs_random"]["ci_excludes_zero"] else
               "NOT ESTABLISHED: deleting cited facts is not distinguishable from deleting "
               "random ones")
    out["verdict"] = verdict
    print(f"\n  verdict: {verdict}")

    path = paths.EXPERIMENTS / "paper3" / f"trace_faithfulness_{args.split}_{args.model}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**out, "per_case": rows}, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
