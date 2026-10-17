"""Three-seed summary of the fine-tuned headline arm, and a paired bootstrap against TF-IDF.

ft.train overwrites ft_masked_InLegalBERT.json on every invocation, so the per-seed record lives
here. Seed 0's per-case probabilities were never saved (fixed in ft/train.py afterwards), so its
scores come from commit 63eb0c0 and it is excluded from the paired test; seeds 1 and 2 are paired.
TF-IDF is refit on exactly the ft case set (same split, same masked_text, same 901 test cases), so
the comparison does not inherit a case-set difference from v1's 0.666.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

from src import paths
from src.data.label_merge import load_final

EXP = paths.ROOT / "ft" / "experiments"
SEED = 573
SEED0 = {"auroc": 0.7010, "acc": 0.6648, "source": "commit 63eb0c0; probabilities not saved"}


def main() -> None:
    split = json.loads((paths.SPLITS / "forum_heldout.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    text = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        if (r.get("masked_text") or "").strip():
            text[r["doc_id"]] = r["masked_text"]

    ft = {s: np.load(EXP / f"probs_masked_InLegalBERT_seed{s}.npz") for s in (1, 2)}
    te = [str(c) for c in ft[1]["case_ids"]]
    assert list(ft[2]["case_ids"]) == list(ft[1]["case_ids"]), "seeds disagree on case order"
    tr = [c for c in split["train"] if c in ylab and c in text]
    yte = np.array([ylab[c] for c in te]); ytr = np.array([ylab[c] for c in tr])

    v = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                        strip_accents="unicode")
    m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)
    m.fit(v.fit_transform(text[c] for c in tr), ytr)
    p_tf = m.predict_proba(v.transform(text[c] for c in te))[:, 1]
    tf = {"auroc": float(roc_auc_score(yte, p_tf)), "acc": float(accuracy_score(yte, p_tf > .5))}

    probs = {s: ft[s]["probs"].astype(float) for s in ft}
    probs["mean12"] = (probs[1] + probs[2]) / 2
    rng = np.random.default_rng(SEED)
    idx = [rng.integers(0, len(yte), len(yte)) for _ in range(2000)]
    idx = [i for i in idx if len(set(yte[i].tolist())) == 2]

    def paired(p, q):
        d = np.array([roc_auc_score(yte[i], p[i]) - roc_auc_score(yte[i], q[i]) for i in idx])
        return {"diff": float(roc_auc_score(yte, p) - roc_auc_score(yte, q)),
                "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                "p_le_0": float((d <= 0).mean())}

    runs = {"seed0": SEED0}
    for s in (1, 2):
        runs[f"seed{s}"] = {"auroc": float(roc_auc_score(yte, probs[s])),
                            "acc": float(accuracy_score(yte, probs[s] > .5))}
    au = [runs[k]["auroc"] for k in runs]; ac = [runs[k]["acc"] for k in runs]
    out = {"n_test": len(te), "n_train_tfidf": len(tr),
           "finetuned_runs": runs,
           "finetuned_mean": {"auroc": float(np.mean(au)), "auroc_sd": float(np.std(au, ddof=1)),
                              "acc": float(np.mean(ac)), "acc_sd": float(np.std(ac, ddof=1))},
           "tfidf_refit_same_cases": tf,
           "paired_vs_tfidf": {"seed1": paired(probs[1], p_tf), "seed2": paired(probs[2], p_tf),
                               "mean_of_seeds_1_2": paired(probs["mean12"], p_tf)},
           "preregistered_boundary": 0.69}
    (EXP / "ft_masked_summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
