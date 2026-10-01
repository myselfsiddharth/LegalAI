"""§10.1 outcome models, and the ablation runner for §10.2.

Model choices:
  `lr`   logistic regression -- the interpretable reference §10.1 asks for.
  `gbm`  sklearn `HistGradientBoostingClassifier`. §10.1 names LightGBM/XGBoost; LightGBM is
         installed here but cannot load (`libomp.dylib` missing on this machine), and installing
         a system OpenMP runtime is not worth a dependency on the whole pipeline. HistGB is the
         same family of algorithm, bundled with sklearn, and needs no OpenMP. Swap it back if
         LightGBM becomes loadable -- the interface is the same.

Two things the runner enforces, both of which would otherwise quietly inflate results:

**The feature space is fit on train only.** See `features.py`; selecting atoms by document
frequency over all data leaks the test distribution into the feature space.

**Class weights are balanced, and the floor is a PER-GROUP majority.** The WIN rate runs 39%→64%
across the temporal split, so a global majority baseline is beaten by any model that notices drift.
Reporting against it would flatter everything.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import paths
from src.eval import metrics
from src.predict.features import FeatureBuilder

SEED = 573


def make_model(kind: str):
    if kind == "lr":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                               random_state=SEED))
    if kind == "gbm":
        return HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.06, max_depth=None, min_samples_leaf=15,
            l2_regularization=1.0, class_weight="balanced", random_state=SEED)
    raise ValueError(kind)


# §10.2 ablation 1 is cumulative: F -> F+P -> F+P+E -> +C -> +Q.
#
# An arm whose ADDED group has no features is not a result, it is the previous arm under a new
# name. That must be stated rather than printed as a separate row: the first run of this had
# "F", "F+P" and "F+P+E" reporting byte-identical numbers because §8 found zero BH-significant
# patterns (P empty) and the burden metadata was unfilled (E empty). Reading those as "patterns
# and elements add nothing" would be right by accident and wrong in reasoning.
# §10.2 ablation 1, cumulative in the order PROJECT.md specifies:
#   F -> F+P -> F+P+E -> +S -> +R -> +C, then the shortcut Q last and alone.
ABLATIONS = [
    ("F", ["F"], None),
    ("F+P", ["F", "P"], "P"),
    ("F+P+E", ["F", "P", "E"], "E"),
    ("F+P+E+S", ["F", "P", "E", "S"], "S"),
    ("F+P+E+S+R", ["F", "P", "E", "S", "R"], "R"),
    ("F+P+E+S+R+C", ["F", "P", "E", "S", "R", "C"], "C"),
    ("all+Q", ["F", "P", "E", "S", "R", "C", "Q"], "Q"),
    ("C only", ["C"], None),
    ("S only", ["S"], None),
    ("R only", ["R"], None),
    ("C+S+R", ["C", "S", "R"], None),
    ("Q only", ["Q"], None),                          # the shortcut alone
]


def run(split_name: str, model_kinds=("lr", "gbm"), min_atom_cases: int = 10,
        require_human_burden: bool = False) -> dict:
    split = json.loads((paths.SPLITS / f"{split_name}.json").read_text())
    fb = FeatureBuilder(min_atom_cases=min_atom_cases,
                        require_human_burden=require_human_burden)

    # Only cases we actually have features for can be used, and that must be stated, not
    # silently intersected -- a model evaluated on 200 of 900 test cases is not comparable to
    # one evaluated on all 900.
    def usable(ids):
        return [c for c in ids if fb.canon.get(c) and c in fb.labels
                and fb.labels[c]["outcome"] in ("WIN", "LOSE")]

    tr, dv, te = usable(split["train"]), usable(split["dev"]), usable(split["test"])
    cov = {"train": fb.coverage(tr), "dev": fb.coverage(dv), "test": fb.coverage(te),
           "split_train_total": len(split["train"]), "split_test_total": len(split["test"])}
    if len(tr) < 50 or len(te) < 20:
        return {"error": f"too few cases with features (train={len(tr)}, test={len(te)}); "
                         f"run §6 canonicalisation on more cases",
                "coverage": cov}

    fb.fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    ytr = np.array([1 if fb.labels[c]["outcome"] == "WIN" else 0 for c in tr])
    yte = np.array([1 if fb.labels[c]["outcome"] == "WIN" else 0 for c in te])

    groups_te = {
        "family": [(fb.membership.get(c, {}).get("primary") or "none") for c in te],
        "decade": [str((fb.labels[c]["year"] // 10) * 10) for c in te],
    }
    groups_te["family"] = [fb.fam_to_super.get(f, f) for f in groups_te["family"]]

    out = {"split": split_name, "split_hash": split.get("hash"), "coverage": cov,
           "n_features": len(fb.space.names),
           "features_per_group": dict(Counter(fb.space.group_of)),
           "burden_provenance": dict(fb.burden_provenance),
           "results": []}

    # --- floors
    maj = metrics.majority_baseline(
        ytr, yte, per_group={"train": [str(fb.labels[c]["year"] // 10 * 10) for c in tr],
                             "test": groups_te["decade"]})
    out["results"].append(metrics.evaluate(yte, maj["global"], name="majority (global)",
                                           groups=groups_te))
    out["results"].append(metrics.evaluate(yte, maj["per_group"],
                                           name="majority (per decade)", groups=groups_te))

    # --- ablations x models
    # which groups actually carry features, and why an empty one is empty
    per_group = Counter(fb.space.group_of)
    empty = {g: per_group.get(g, 0) for g in ("F", "P", "E", "S", "R", "C", "Q")
             if not per_group.get(g)}
    out["empty_groups"] = {}
    for g in empty:
        out["empty_groups"][g] = {
            "P": "no pattern survived BOTH bootstrap stability and Benjamini-Hochberg in §8 "
                 "(351 patterns tested, 0 BH-significant). This is a result, not a missing input.",
            "E": "no element carries burden metadata yet (burden_provenance is unset), so the "
                 "element features cannot be computed.",
            "S": "no predicted statutes on disk -- run `python -m src.predict.side_inputs`.",
            "R": "no retrieved precedents on disk -- run `python -m src.predict.side_inputs`.",
        }.get(g, "no features in this group")

    preds_for_mcnemar = {}
    for kind in model_kinds:
        for label, groups, added in ABLATIONS:
            mask = fb.space.group_mask(groups)
            if mask.sum() == 0:
                continue
            if added and added in empty:
                # the added group contributes nothing; skip so the table cannot imply otherwise
                continue
            m = make_model(kind)
            m.fit(Xtr[:, mask], ytr)
            pred = m.predict(Xte[:, mask])
            prob = (m.predict_proba(Xte[:, mask])[:, 1]
                    if hasattr(m, "predict_proba") else None)
            nm = f"{kind} {label}"
            res = metrics.evaluate(yte, pred, prob, name=nm, groups=groups_te)
            res["feature_groups"] = [g for g in groups if per_group.get(g)]
            res["n_features_used"] = int(mask.sum())
            out["results"].append(res)
            preds_for_mcnemar[nm] = pred

    # --- paired tests: does +Q (the shortcut) change anything? (§10.2 ablation)
    out["mcnemar"] = {}
    for kind in model_kinds:
        a, b = next((f"{kind} {lb}" for lb, _g, ad in reversed(ABLATIONS)
                     if ad not in empty and "Q" not in lb and "only" not in lb),
                    f"{kind} F"), f"{kind} F+P+E+C+Q"
        if a in preds_for_mcnemar and b in preds_for_mcnemar:
            out["mcnemar"][f"{b} vs {a}"] = metrics.mcnemar(
                yte, preds_for_mcnemar[b], preds_for_mcnemar[a])
        c = f"{kind} F"
        if c in preds_for_mcnemar and a in preds_for_mcnemar:
            out["mcnemar"][f"{a} vs {c}"] = metrics.mcnemar(
                yte, preds_for_mcnemar[a], preds_for_mcnemar[c])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--models", nargs="+", default=["lr", "gbm"])
    ap.add_argument("--min-atom-cases", type=int, default=10)
    ap.add_argument("--require-human-burden", action="store_true",
                    help="use only burden metadata a human has confirmed (E group will be empty "
                         "until that review happens)")
    args = ap.parse_args()

    out = run(args.split, tuple(args.models), args.min_atom_cases, args.require_human_burden)
    if "error" in out:
        print(f"ERROR: {out['error']}")
        print(f"coverage: {json.dumps(out['coverage'], indent=1)}")
        raise SystemExit(1)

    print(f"=== §10 outcome prediction, split={out['split']} (hash {out['split_hash']}) ===")
    c = out["coverage"]
    print(f"  cases with features: train {c['train']['with_canonical_facts']}"
          f"/{c['split_train_total']}, test {c['test']['with_canonical_facts']}"
          f"/{c['split_test_total']}")
    print(f"  features: {out['n_features']} -> {out['features_per_group']}")
    for g, why in out.get("empty_groups", {}).items():
        print(f"  GROUP {g} IS EMPTY, so no ablation arm adds it: {why}")
    prov = out["burden_provenance"]
    if prov.get("llm_draft"):
        print(f"  NOTE the E group rests on {prov['llm_draft']} LLM-DRAFTED burden values "
              f"(unreviewed); see ontology/CHANGELOG.md")
    print()
    for r in out["results"]:
        print("  " + metrics.format_row(r))
    if out["mcnemar"]:
        print("\n  paired tests (McNemar, exact):")
        for k, v in out["mcnemar"].items():
            print(f"    {k}: {v}")

    path = paths.EXPERIMENTS / "paper3" / f"outcome_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
