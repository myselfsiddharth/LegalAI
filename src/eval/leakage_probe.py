"""Stage 1 acceptance gate (§5.2): does `masked_text` still reveal the outcome?

§5.2 asks for a TF-IDF + logistic regression probe on masked text and warns to tighten
masking if it comes out "suspiciously high". Suspiciously high *relative to what* is the whole
question, so the probe runs four arms on identical splits, folds and metrics:

  full        unmasked judgment      -- the ceiling. The outcome is literally written here, so
                                        a number near 1.0 confirms the probe can detect leakage
                                        at all. A LOW number here would mean the probe is too
                                        weak to trust anywhere.
  masked      `masked_text`          -- the arm under test.
  prior_court `prior_court_text`     -- the procedural recital alone: how far can you get
                                        knowing ONLY what the court below did? This is the
                                        shortcut, measured on its own.
  majority    no features            -- the floor.

Interpretation:
  masked ~ full      -> masking failed, tighten it.
  masked ~ majority  -> the facts carry no signal (or the mask removed too much).
  in between          -> the honest operating range, and `prior_court` says how much of it is
                        the shortcut rather than the law.

Reported with bootstrap 95% CIs per §14.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import paths

SEED = 573


def load_arm_texts():
    masked, prior, full, order = {}, {}, {}, {}
    order_start = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]
        prior[r["doc_id"]] = r["prior_court_text"]
        order_start[r["doc_id"]] = r.get("order_region_start")
    for line in open(paths.CASE_TEXT):
        r = json.loads(line)
        full[r["doc_id"]] = r["text"]
        # `order_only` is the probe's SENSITIVITY control. The operative order is one
        # sentence in a ~35k-char judgment, so TF-IDF over the full text dilutes it and the
        # `full` arm understates how detectable the outcome is. Feeding the probe the order
        # region alone establishes what it scores when the answer is definitely present; only
        # against that can a low `masked` score be read as masking working rather than as the
        # probe being blind.
        s = order_start.get(r["doc_id"])
        order[r["doc_id"]] = r["text"][s:] if s is not None else r["text"][-3000:]
    return {"full": full, "masked": masked, "prior_court": prior, "order_only": order}


def load_labels():
    from src.data.label_merge import load_final
    return {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}


def bootstrap_ci(y, p, metric, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        try:
            vals.append(metric(y[i], p[i]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def run_arm(name, texts, split, labels, max_features=50_000):
    tr = [d for d in split["train"] if d in labels and texts.get(d, "").strip()]
    te = [d for d in split["test"] if d in labels and texts.get(d, "").strip()]
    if len(tr) < 50 or len(te) < 20:
        return {"arm": name, "error": f"too few usable cases (train={len(tr)}, test={len(te)})"}

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                          min_df=3, sublinear_tf=True, strip_accents="unicode")
    Xtr = vec.fit_transform(texts[d] for d in tr)
    Xte = vec.transform(texts[d] for d in te)
    ytr = [labels[d] for d in tr]
    yte = [labels[d] for d in te]

    clf = LogisticRegression(max_iter=2000, C=1.0, random_state=SEED, class_weight="balanced")
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    prob = clf.predict_proba(Xte)[:, 1]

    acc, mf1 = accuracy_score(yte, pred), f1_score(yte, pred, average="macro")
    auc = roc_auc_score(yte, prob) if len(set(yte)) > 1 else float("nan")
    return {
        "arm": name, "n_train": len(tr), "n_test": len(te),
        "accuracy": round(acc, 4), "macro_f1": round(mf1, 4), "auroc": round(auc, 4),
        "macro_f1_ci": [round(x, 4) for x in bootstrap_ci(yte, pred, lambda a, b: f1_score(a, b, average="macro"))],
        "auroc_ci": [round(x, 4) for x in bootstrap_ci(yte, prob, roc_auc_score)],
        "test_win_rate": round(float(np.mean(yte)), 4),
        "n_features": Xtr.shape[1],
    }


def majority_arm(split, labels):
    tr = [labels[d] for d in split["train"] if d in labels]
    te = [labels[d] for d in split["test"] if d in labels]
    if not tr or not te:
        return {"arm": "majority", "error": "empty"}
    maj = Counter(tr).most_common(1)[0][0]
    pred = [maj] * len(te)
    return {"arm": "majority", "n_train": len(tr), "n_test": len(te),
            "accuracy": round(accuracy_score(te, pred), 4),
            "macro_f1": round(f1_score(te, pred, average="macro"), 4),
            "auroc": 0.5,
            "macro_f1_ci": [round(x, 4) for x in bootstrap_ci(te, pred, lambda a, b: f1_score(a, b, average="macro"))],
            "test_win_rate": round(float(np.mean(te)), 4),
            "train_win_rate": round(float(np.mean(tr)), 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="temporal_2005_2013")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = load_labels()
    arms = load_arm_texts()

    results = [majority_arm(split, labels)]
    for name in ("prior_court", "masked", "full", "order_only"):
        results.append(run_arm(name, arms[name], split, labels))

    print(f"=== leakage probe: TF-IDF(1,2) + LR, split={args.split} ===")
    print(f"train n={results[0].get('n_train')}  test n={results[0].get('n_test')}  "
          f"train WIN={results[0].get('train_win_rate')}  test WIN={results[0].get('test_win_rate')}\n")
    hdr = f"{'arm':12s} {'n_tr':>5s} {'n_te':>5s} {'acc':>6s} {'macroF1':>8s} {'95% CI':>16s} {'AUROC':>7s} {'95% CI':>16s}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        if "error" in r:
            print(f"{r['arm']:12s} {r['error']}")
            continue
        ci = f"[{r['macro_f1_ci'][0]:.3f},{r['macro_f1_ci'][1]:.3f}]"
        aci = f"[{r['auroc_ci'][0]:.3f},{r['auroc_ci'][1]:.3f}]" if "auroc_ci" in r else "-"
        print(f"{r['arm']:12s} {r['n_train']:5d} {r['n_test']:5d} {r['accuracy']:6.3f} "
              f"{r['macro_f1']:8.3f} {ci:>16s} {r['auroc']:7.3f} {aci:>16s}")

    out = paths.EXPERIMENTS / "paper3" / f"leakage_probe_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "hash": split.get("hash"),
                               "results": results}, indent=1))
    print(f"\n-> {out}")

    by = {r["arm"]: r for r in results if "macro_f1" in r}
    if "order_only" in by:
        print(f"\n  probe sensitivity check -- order region alone: "
              f"{by['order_only']['macro_f1']:.3f} macro-F1, "
              f"AUROC {by['order_only']['auroc']:.3f}")
        print("  (this arm SHOULD be high: it is the text masking removes. If it were low, "
              "the probe could not detect leakage and no other arm would be interpretable.)")
    if {"masked", "full", "majority"} <= set(by):
        m, fu, mj = by["masked"]["macro_f1"], by["full"]["macro_f1"], by["majority"]["macro_f1"]
        span = fu - mj
        print(f"\ninterpretation: majority {mj:.3f} -> masked {m:.3f} -> full {fu:.3f}")
        if span > 0:
            print(f"  masked sits {100*(m-mj)/span:.0f}% of the way from the floor to the "
                  f"unmasked ceiling")
        if "prior_court" in by:
            print(f"  prior-court recital alone: {by['prior_court']['macro_f1']:.3f} macro-F1 "
                  f"(AUROC {by['prior_court']['auroc']:.3f})")


if __name__ == "__main__":
    main()
