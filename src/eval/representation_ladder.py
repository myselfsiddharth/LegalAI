"""Where in the pipeline does the predictive signal go?

Each stage of §6 replaces one representation with a more structured one. This runs the SAME
classifier, the SAME split, the SAME labels and the SAME case set at every rung, so a drop can
only be attributed to the representation change itself:

  1. `masked_text`        the input a predictor is allowed to see (§5.2)
  2. extracted fact text  the atomic propositions of §6.1, as free text
  3. canonical atoms      those facts mapped to the controlled vocabulary of §6.2

Rung 1 -> 2 asks whether extraction throws information away. Rung 2 -> 3 asks whether the
vocabulary can carry it. The case set is restricted to cases present at every rung, because a
comparison across different case sets measures the cases, not the representation.

This is the diagnostic that explains §8's null result: FP-Growth mines the atoms of rung 3, so if
rung 3 is at chance then no pattern over those atoms can be significant, and the absence of
patterns is a property of the vocabulary rather than of the law.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src import paths
from src.data.label_merge import load_final
from src.eval import metrics
from src.predict.features import FeatureBuilder

SEED = 573


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    fact_text: dict[str, list[str]] = {}
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        fact_text.setdefault(r["case_id"], []).append(r["text"])

    fb = FeatureBuilder()

    def usable(ids):
        return [c for c in ids
                if fb.canon.get(c) and c in labels and masked.get(c) and fact_text.get(c)]

    tr, te = usable(split["train"]), usable(split["test"])
    ytr = np.array([labels[c] for c in tr])
    yte = np.array([labels[c] for c in te])
    if len(tr) < 50 or len(te) < 20:
        raise SystemExit(f"too few cases at every rung (train={len(tr)}, test={len(te)})")

    def lr():
        return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                                  random_state=SEED)

    results = []

    def tfidf_rung(name, texts):
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(texts[c] for c in tr)
        Xte = vec.transform(texts[c] for c in te)
        m = lr().fit(Xtr, ytr)
        results.append(metrics.evaluate(yte, m.predict(Xte), m.predict_proba(Xte)[:, 1],
                                        name=name))
        results[-1]["n_features"] = Xtr.shape[1]

    tfidf_rung("1. masked text", masked)
    tfidf_rung("2. extracted fact text", {c: " ".join(fact_text.get(c, [])) for c in tr + te})

    fb.fit(tr)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    mask = fb.space.group_mask(["F"])
    m = lr().fit(Xtr[:, mask], ytr)
    results.append(metrics.evaluate(yte, m.predict(Xte[:, mask]),
                                    m.predict_proba(Xte[:, mask])[:, 1],
                                    name="3. canonical atoms"))
    results[-1]["n_features"] = int(mask.sum())

    maj = metrics.majority_baseline(ytr, yte)
    results.insert(0, metrics.evaluate(yte, maj["global"], name="0. majority"))

    print(f"=== representation ladder, split={args.split} ===")
    print(f"identical case set at every rung: train {len(tr)}, test {len(te)}; "
          f"test WIN rate {yte.mean():.3f}\n")
    for r in results:
        print("  " + metrics.format_row(r) +
              (f"  feats={r['n_features']}" if "n_features" in r else ""))

    by = {r["name"]: r.get("auroc") for r in results if r.get("auroc")}
    print(f"\n  AUROC: " + "  ->  ".join(f"{k.split('. ')[1]} {v:.3f}" for k, v in by.items()))
    print("  chance 0.500")
    t, f_, a = (by.get("1. masked text"), by.get("2. extracted fact text"),
                by.get("3. canonical atoms"))
    if t and f_ and a:
        print(f"\n  extraction (text -> facts):        {t:.3f} -> {f_:.3f}  "
              f"({'preserves' if abs(t-f_) < 0.03 else 'loses'} the signal)")
        print(f"  canonicalisation (facts -> atoms): {f_:.3f} -> {a:.3f}  "
              f"({'preserves' if abs(f_-a) < 0.03 else 'DESTROYS'} the signal)")

    out = paths.EXPERIMENTS / "paper1" / f"representation_ladder_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "split_hash": split.get("hash"),
                               "n_train": len(tr), "n_test": len(te),
                               "results": results}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
