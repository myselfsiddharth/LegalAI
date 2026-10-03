"""S2 — fact families by SOFT assignment over embeddings. The step v1 built in its lossiest form.

v1 mapped each fact onto one of 345 authored labels by argmax. Measured on identical cases and
identical granularity, that argmax costs **+0.084 AUROC** -- 56% of the entire text-to-atoms loss
(`../experiments/paper1/fact_embedding_rung_forum_heldout.json`). This module is the architecture's
step 2 as actually specified: "classification model run on all embeddings of the facts", with the
classification kept soft.

## What it produces

For every case, a `k`-dimensional vector: the mean over the case's facts of a softmax-weighted
cosine similarity to `k` centroids. Centroids are fitted on TRAIN facts only, and test facts are
assigned, never fitted -- the same discipline v1 used for its induced-vocabulary control.

The output is written to `v2/interim/soft_families_<split>.npz` so S4 and S5 consume one
representation rather than each recomputing it.

## Why a soft histogram and not the mean embedding

Both were measured. The 4,096-dim mean scores 0.558; the 345-dim soft histogram scores 0.601.
Averaging seventeen fact vectors washes out the one decisive fact, and a per-centroid histogram
preserves which *kinds* of fact are present. Reported in the same results file.

## Temperature

`--temp` scales the cosine before the softmax: 0 is a uniform blur, large values approach the argmax
v1 used. It is swept rather than assumed, because the whole point of this module is that the hard
limit is the wrong choice -- so the sweep has to show where the optimum actually sits.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.label_merge import load_final
from src.eval import metrics
from src.extract.induced_vocab import load_fact_texts
from v2 import paths

SEED = 573


def soft_histogram(V: np.ndarray, Cn: np.ndarray, temp: float) -> np.ndarray:
    """One case's facts -> a k-vector. temp=0 uniform; temp->inf approaches the v1 argmax."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    s = Vn @ Cn.T
    if temp <= 0:
        W = np.full_like(s, 1.0 / s.shape[1])
    else:
        z = s * temp
        z -= z.max(1, keepdims=True)
        W = np.exp(z)
        W /= W.sum(1, keepdims=True)
    return W.sum(0) / len(V)


def hard_histogram(V: np.ndarray, Cn: np.ndarray) -> np.ndarray:
    """The v1 operation, for reference: argmax onto one centroid per fact."""
    Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    idx = (Vn @ Cn.T).argmax(1)
    h = np.zeros(Cn.shape[0], dtype=np.float32)
    for i in idx:
        h[i] += 1.0
    return h / len(V)


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
    ap.add_argument("--k", type=int, default=345,
                    help="centroids; defaults to the authored vocabulary's size so the "
                         "soft-vs-hard comparison is at equal granularity")
    ap.add_argument("--temps", type=float, nargs="+", default=[0, 2, 6, 12, 24, 48])
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    facts, X = load_fact_texts()
    by_case: dict[str, list[int]] = {}
    for i, f in enumerate(facts):
        by_case.setdefault(f["case_id"], []).append(i)
    print(f"S2: {len(facts):,} facts with cached embeddings, dim {X.shape[1]}, "
          f"{len(by_case):,} cases")

    tr = [c for c in split["train"] if c in by_case and c in ylab]
    te = [c for c in split["test"] if c in by_case and c in ylab]
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])
    print(f"train {len(tr):,}  test {len(te)}  test WIN {yte.mean():.3f}")

    rows = np.array([i for c in tr for i in by_case[c]])
    print(f"fitting {args.k} centroids on {len(rows):,} TRAIN facts "
          f"(test facts assigned, never fitted)", flush=True)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED, n_init=5,
                         batch_size=2048).fit(X[rows])
    C = km.cluster_centers_
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

    results, probs, best = [], {}, None
    print("\n  temperature sweep (temp->inf is the v1 argmax):")
    for t in args.temps:
        Atr = np.vstack([soft_histogram(X[by_case[c]], Cn, t) for c in tr])
        Ate = np.vstack([soft_histogram(X[by_case[c]], Cn, t) for c in te])
        r, pr = score(Atr, ytr, Ate, yte, f"soft_temp{t:g}")
        r["temp"] = t
        results.append(r)
        probs[r["name"]] = pr
        print(f"    temp={t:<4g} AUROC={r['auroc']:.3f}  mF1={r['macro_f1']:.3f}")
        if best is None or r["auroc"] > best[0]:
            best = (r["auroc"], t, Atr, Ate)

    print("\n  references at the same k:")
    Htr = np.vstack([hard_histogram(X[by_case[c]], Cn) for c in tr])
    Hte = np.vstack([hard_histogram(X[by_case[c]], Cn) for c in te])
    r_hard, probs["hard_argmax"] = score(Htr, ytr, Hte, yte, "hard_argmax")
    results.append(r_hard)
    print(f"    hard_argmax   AUROC={r_hard['auroc']:.3f}   <- the v1 operation")
    Mtr = np.vstack([X[by_case[c]].mean(0) for c in tr])
    Mte = np.vstack([X[by_case[c]].mean(0) for c in te])
    r_mean, probs["emb_mean"] = score(Mtr, ytr, Mte, yte, "emb_mean")
    results.append(r_mean)
    print(f"    emb_mean      AUROC={r_mean['auroc']:.3f}   ({X.shape[1]:,} dims)")

    def paired(a, b, n_boot=2000):
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[i], probs[a][i]) - roc_auc_score(yte[i], probs[b][i]))
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    bauroc, btemp, Atr, Ate = best
    bname = f"soft_temp{btemp:g}"
    m, lo, hi = paired(bname, "hard_argmax")
    print(f"\n  {bname} - hard_argmax: {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"{'  significant' if lo > 0 else ''}")

    out_npz = paths.INTERIM / f"soft_families_{args.split}.npz"
    np.savez_compressed(out_npz, train_ids=np.array(tr), test_ids=np.array(te),
                        train=Atr.astype(np.float32), test=Ate.astype(np.float32),
                        centroids=C.astype(np.float32), temp=btemp, k=args.k)
    print(f"  -> {out_npz}  (S4/S5 consume this, not a recomputation)")

    out = paths.EXPERIMENTS / f"s2_soft_families_{args.split}.json"
    out.write_text(json.dumps({
        "stage": "S2 soft fact families", "split": args.split, "k": args.k,
        "n_facts": len(facts), "n_train": len(tr), "n_test": len(te),
        "best_temp": btemp, "best_auroc": round(bauroc, 4),
        "hard_argmax_auroc": r_hard["auroc"], "emb_mean_auroc": r_mean["auroc"],
        "soft_minus_hard": {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                            "ci_excludes_zero": lo > 0},
        "v1_reference": {"canonical_atoms": 0.517, "extracted_fact_text": 0.629,
                         "masked_text": 0.666,
                         "note": "v1 ladder + fact_embedding_rung, same split"},
        "results": results}, indent=1))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
