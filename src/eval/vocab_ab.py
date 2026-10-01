"""Ontology vocabulary vs an INDUCED vocabulary, at equal granularity.

Both reduce the same facts to ~345 discrete atoms. One was authored top-down from ontology enums;
the other was clustered bottom-up from the corpus with no ontology input at all. Both are scored on
the same cases with the same classifier, against the fact text they each replace.

How to read the result:

  induced ~ text, ontology << text  -> the ONTOLOGY's vocabulary is the problem; induce it instead.
  induced ~ ontology << text        -> ANY reduction to a few hundred symbols loses the signal, so
                                       §8's itemisation is mis-specified whatever the vocabulary is.
  induced << ontology               -> clustering is worse than curation; the loss is elsewhere.

The third reading matters as much as the first two: if itemisation itself is what destroys the
signal, then no amount of vocabulary work rescues FP-Growth over these facts, and that is a finding
about pattern mining on legal text rather than a bug to fix.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import paths
from src.data.label_merge import load_final

SEED = 573


def lr():
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)


def load_atoms(path, key="label"):
    by_case = defaultdict(set)
    txt = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_case[r["case_id"]].add(r[key])
            txt[r["case_id"]].append(r["fact_text"])
    return by_case, txt


def score_atoms(atoms, tr, te, ytr, yte, min_cases=2):
    """Binary atom indicators. The atom vocabulary is fit on TRAIN only -- selecting atoms by
    document frequency over all data would leak the test distribution into the representation."""
    df = Counter(a for c in tr for a in atoms[c])
    vocab = sorted(a for a, n in df.items() if n >= min_cases)
    idx = {a: i for i, a in enumerate(vocab)}

    def mat(ids):
        X = np.zeros((len(ids), len(vocab)), dtype=np.float32)
        for i, c in enumerate(ids):
            for a in atoms[c]:
                j = idx.get(a)
                if j is not None:
                    X[i, j] = 1.0
        return X

    m = lr().fit(mat(tr), ytr)
    p = m.predict_proba(mat(te))[:, 1]
    return roc_auc_score(yte, p), p, len(vocab)


def score_text(txt, tr, te, ytr, yte):
    vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          strip_accents="unicode")
    Xtr = vec.fit_transform(" ".join(txt[c]) for c in tr)
    Xte = vec.transform(" ".join(txt[c]) for c in te)
    m = lr().fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), p, Xtr.shape[1]


def paired(yte, pa, pb, n=2000):
    """Paired bootstrap over test CASES. Comparing two independent CIs would be far too weak at
    this n; resampling the same cases for both systems is what makes a difference testable."""
    rng = np.random.default_rng(SEED)
    d = []
    for _ in range(n):
        i = rng.integers(0, len(yte), len(yte))
        if len(set(yte[i].tolist())) < 2:
            continue
        d.append(roc_auc_score(yte[i], pa[i]) - roc_auc_score(yte[i], pb[i]))
    return (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(np.mean([x <= 0 for x in d])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    ont_atoms, _ont_txt = load_atoms(paths.INTERIM / "canonical_facts.jsonl")
    ind_atoms, ind_txt = load_atoms(paths.INTERIM / "induced_facts.jsonl")

    def both(ids):
        return [c for c in ids if c in labels and ont_atoms.get(c) and ind_atoms.get(c)]

    tr, te = both(split["train"]), both(split["test"])
    ytr = np.array([labels[c] for c in tr])
    yte = np.array([labels[c] for c in te])
    print(f"=== vocabulary A/B, split={args.split} ===")
    print(f"identical case set: train {len(tr)}, test {len(te)}; test WIN {yte.mean():.3f}\n")

    t_auc, t_p, t_f = score_text(ind_txt, tr, te, ytr, yte)
    i_auc, i_p, i_f = score_atoms(ind_atoms, tr, te, ytr, yte)
    o_auc, o_p, o_f = score_atoms(ont_atoms, tr, te, ytr, yte)

    print(f"  {'representation':30s} {'feats':>6s} {'AUROC':>7s}")
    print("  " + "-" * 46)
    print(f"  {'fact text (tf-idf)':30s} {t_f:6d} {t_auc:7.3f}")
    print(f"  {'INDUCED atoms (clustered)':30s} {i_f:6d} {i_auc:7.3f}")
    print(f"  {'ontology atoms (§6.2)':30s} {o_f:6d} {o_auc:7.3f}")
    print(f"  {'chance':30s} {'':6s} {0.5:7.3f}")

    print("\n  paired bootstrap on the same test cases:")
    rows = {}
    for nm, a, b in (("text - induced", t_p, i_p), ("text - ontology", t_p, o_p),
                     ("induced - ontology", i_p, o_p)):
        lo, hi, p = paired(yte, a, b)
        diff = roc_auc_score(yte, a) - roc_auc_score(yte, b)
        rows[nm] = {"diff": round(float(diff), 4), "ci95": [round(lo, 4), round(hi, 4)],
                    "p_le_0": round(p, 4)}
        print(f"    {nm:20s} {diff:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(<=0) = {p:.3f}")

    if i_auc - o_auc > 0.04:
        verdict = "the ONTOLOGY vocabulary is the problem -- induce it instead"
    elif t_auc - i_auc > 0.04:
        verdict = ("ANY reduction to ~345 symbols loses the signal -- §8's itemisation is "
                   "mis-specified regardless of vocabulary")
    else:
        verdict = "induced and ontology behave alike and both track the text; loss is elsewhere"
    print(f"\n  reading: {verdict}")

    out = paths.EXPERIMENTS / "paper1" / f"vocab_ab_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_train": len(tr), "n_test": len(te),
        "text": {"auroc": round(t_auc, 4), "features": t_f},
        "induced": {"auroc": round(i_auc, 4), "features": i_f},
        "ontology": {"auroc": round(o_auc, 4), "features": o_f},
        "paired": rows, "verdict": verdict}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
