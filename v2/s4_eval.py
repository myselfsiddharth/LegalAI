"""The acceptance test for v2: does the ELEMENT layer earn its place?

The architecture's central claim is that elements are the right intermediate representation between
facts and claims. That is falsifiable without any gold annotation:

    Do S4 element features predict the outcome better than the S2 soft fact representation
    they are derived FROM?

S2 is the honest baseline, not v1's canonical atoms. S2 scores 0.633 on this split and is itself
derived from the same facts, so a gain here is attributable to the element abstraction rather than to
better preprocessing. Three outcomes and what each means:

  element > S2          the abstraction ADDS something -- the layer earns its place.
  element ~= S2         the layer is a lossless relabelling: defensible as explanation, but it is
                        not buying predictive power and should not be sold as if it were.
  element < S2          the layer DESTROYS information, exactly as v1's hard canonicalisation did,
                        and the architecture's middle is a bottleneck rather than a benefit.

A fourth arm settles whether any gain is the abstraction or merely more features:

  `S2 + element`        if the combination beats both, the two carry complementary signal.

## Features

Per case, over all 66 catalogue elements: +1 satisfied, -1 not satisfied, 0 unclear or not asked.
Plus four summary counts (satisfied, not satisfied, unclear, elements asked). Only GATED verdicts
count -- a verdict whose quote could not be located never reaches the feature matrix, so the model
cannot be rewarded for fabricated support.

## What this cannot show

Outcome is a weak yardstick and v1 established why: masked text reaches only 0.656-0.723 across four
splits, and every structured model sits at or below a per-decade majority baseline. So a null result
here is evidence the element layer adds no OUTCOME signal; it is not evidence the layer is wrong. An
element layer that is legally correct and outcome-neutral would still be the useful product --
"here are the elements your facts satisfy, with the quote for each" is checkable, which outcome
never is. That reading is stated here so a null is not over-interpreted later.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.label_merge import load_final
from src.eval import metrics
from v2 import paths
from v2.element_catalogue import all_elements, load_claims

SEED = 573


def lr():
    return LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=SEED)


def score(Atr, ytr, Ate, yte, name):
    sc = StandardScaler().fit(Atr)
    m = lr().fit(sc.transform(Atr), ytr)
    pr = m.predict_proba(sc.transform(Ate))[:, 1]
    r = metrics.evaluate(yte, m.predict(sc.transform(Ate)), pr, name=name)
    r["n_features"] = int(Atr.shape[1])
    return r, pr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--verdicts", default=None)
    args = ap.parse_args()

    vp = (paths.INTERIM / args.verdicts if args.verdicts
          else paths.INTERIM / "element_verdicts_element_satisfy_v2_facts.jsonl")
    if not vp.exists():
        raise SystemExit(f"{vp} missing -- run `python -m v2.element_layer` first")

    elements = all_elements(load_claims())
    eid = {e.element_id: i for i, e in enumerate(elements)}
    n_el = len(elements)

    V: dict[str, np.ndarray] = {}
    C: dict[str, np.ndarray] = {}
    asked: dict[str, set[str]] = {}
    for line in open(vp):
        r = json.loads(line)
        c = r["case_id"]
        v = V.setdefault(c, np.zeros(n_el, dtype=np.float32))
        cnt = C.setdefault(c, np.zeros(4, dtype=np.float32))
        s = asked.setdefault(c, set())
        for d in r["verdicts"]:
            i = eid.get(d["element_id"])
            if i is None:
                continue
            s.add(d["element_id"])
            if d["verdict"] == "SATISFIED" and d.get("gated"):
                v[i] = 1.0
                cnt[0] += 1
            elif d["verdict"] == "NOT_SATISFIED" and d.get("gated"):
                v[i] = -1.0
                cnt[1] += 1
            else:
                cnt[2] += 1
        cnt[3] = len(s)

    npz = np.load(paths.INTERIM / f"soft_families_{args.split}.npz", allow_pickle=True)
    s2 = {str(d): npz["train"][i] for i, d in enumerate(npz["train_ids"])}
    s2.update({str(d): npz["test"][i] for i, d in enumerate(npz["test_ids"])})

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    def ok(c):
        return c in ylab and c in V and c in s2
    tr = [c for c in split["train"] if ok(c)]
    te = [c for c in split["test"] if ok(c)]
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])

    print(f"=== S4 acceptance test: does the element layer earn its place? ===")
    print(f"cases with verdicts: {len(V):,}   usable in all arms: train {len(tr):,}, test {len(te)}")
    print(f"test WIN {yte.mean():.3f}")
    tot = sum(C[c][:3].sum() for c in V)
    sat = sum(C[c][0] for c in V)
    nsat = sum(C[c][1] for c in V)
    print(f"gated verdicts: {int(sat):,} SATISFIED, {int(nsat):,} NOT_SATISFIED, "
          f"{int(tot-sat-nsat):,} UNCLEAR  (NOT_SATISFIED share {nsat/max(1,tot):.1%})\n")

    E = lambda cs: np.vstack([np.concatenate([V[c], C[c]]) for c in cs])
    S = lambda cs: np.vstack([s2[c] for c in cs])

    results, probs = [], {}
    for name, f in (("S2 soft facts", S), ("S4 elements", E),
                    ("S2 + S4", lambda cs: np.hstack([S(cs), E(cs)]))):
        r, pr = score(f(tr), ytr, f(te), yte, name)
        results.append(r)
        probs[name] = pr
        print(f"  {name:16s} AUROC={r['auroc']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"({r['n_features']:,} features)")

    def paired(a, b, n_boot=2000):
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[i], probs[a][i]) - roc_auc_score(yte[i], probs[b][i]))
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    print("\n  paired bootstrap:")
    comps = {}
    for a, b in (("S4 elements", "S2 soft facts"), ("S2 + S4", "S2 soft facts"),
                 ("S2 + S4", "S4 elements")):
        m, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:34s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    e_vs_s2 = comps["S4 elements - S2 soft facts"]
    verdict = ("EARNS ITS PLACE: the element abstraction adds outcome signal over the facts it is "
               "derived from" if e_vs_s2["ci_excludes_zero"] else
               "LOSSY: the element layer scores BELOW the facts it is derived from, the same failure "
               "mode as v1's hard canonicalisation"
               if e_vs_s2["ci95"][1] < 0 else
               "NEUTRAL: indistinguishable from the facts it is derived from. Defensible as a "
               "checkable explanation layer; not a predictive gain, and must not be sold as one")
    print(f"\n  VERDICT: {verdict}")

    out = paths.EXPERIMENTS / f"s4_acceptance_{args.split}.json"
    out.write_text(json.dumps({
        "stage": "S4 acceptance test", "split": args.split,
        "n_train": len(tr), "n_test": len(te), "n_elements": n_el,
        "gated_satisfied": int(sat), "gated_not_satisfied": int(nsat), "gated_unclear": int(tot-sat-nsat),
        "not_satisfied_share": round(float(nsat) / max(1.0, float(tot)), 4),
        "results": results, "paired": comps, "verdict": verdict,
        "caveat": "outcome is a weak yardstick (v1: masked text 0.656-0.723, every structured model "
                  "at or below a per-decade majority baseline). A null here means the layer adds no "
                  "OUTCOME signal, not that it is legally wrong."}, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
