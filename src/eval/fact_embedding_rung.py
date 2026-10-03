"""The ladder rung that was never built: facts kept CONTINUOUS instead of discretised.

`src/eval/representation_ladder.py` has three rungs -- masked text, extracted fact text, and
canonical atoms -- and concludes that discretising facts into a label vocabulary costs 0.097 AUROC.
That conclusion is measured only against DISCRETE vocabularies: the authored 345-entry ontology one
and, as a control, a k-means-induced one of equal granularity.

The architecture this project set out to build does not actually require a discrete vocabulary at
step 2. It says "fact family -- classification model run on all embeddings of the facts". Keeping the
embeddings and clustering softly is a different operation from mapping each fact onto one of 345
authored labels, and it has never been tested here. Every fact already carries a cached embedding
(canonicalisation embedded them), so the rung costs nothing but a few minutes of CPU.

Three arms, all on the ladder's identical case set and classifier:

  `fact_emb_mean`        per case, the mean of its fact embeddings. The plainest continuous
                         fact-structured representation.
  `fact_emb_mean_max`    mean concatenated with element-wise max, so a single decisive fact can
                         survive averaging.
  `fact_emb_softcluster` soft assignment to k centroids (distances to each, softmaxed and summed
                         over the case's facts) -- the user's "classification model on embeddings"
                         read literally, but WITHOUT the hard argmax that makes it discrete.

This is the experiment that decides whether the architecture's foundation is sound:

  if continuous facts land near MASKED TEXT (0.657) -> discretisation is the whole problem, and the
     pipeline works provided step 2 never hardens into labels. The architecture is vindicated.
  if they land near CANONICAL ATOMS (0.517)        -> the information was lost at EXTRACTION, before
     any vocabulary was involved. The architecture's foundation is the problem, not its symbols.
  if they land near EXTRACTED FACT TEXT (0.614)    -> the bottleneck is which facts get selected,
     not how they are represented.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src import paths
from src.data.label_merge import load_final
from src.eval import metrics
from src.extract.induced_vocab import load_fact_texts

SEED = 573


def lr():
    return LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--k", type=int, default=345, help="soft-cluster centroids")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    facts, X = load_fact_texts()
    print(f"facts with cached embeddings: {len(facts):,}  dim={X.shape[1]}")

    by_case: dict[str, list[int]] = {}
    for i, f in enumerate(facts):
        by_case.setdefault(f["case_id"], []).append(i)

    # the ladder's rungs 1 and 2, rebuilt here so the comparison is on one case set
    masked = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]
    fact_text = {c: " ".join(facts[i]["text"] for i in idx) for c, idx in by_case.items()}

    cases = [c for c in by_case if c in ylab and masked.get(c, "").strip()]
    tr = [c for c in split["train"] if c in cases]
    te = [c for c in split["test"] if c in cases]
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])
    print(f"identical case set: train {len(tr):,}, test {len(te)}; test WIN {yte.mean():.3f}\n")

    # soft clustering, fitted on TRAIN facts only
    train_rows = np.array([i for c in tr for i in by_case[c]])
    km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED, n_init=5,
                         batch_size=2048).fit(X[train_rows])
    C = km.cluster_centers_
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

    def pool(cs):
        mean, meanmax, soft = [], [], []
        for c in cs:
            V = X[by_case[c]]
            m = V.mean(0)
            mean.append(m)
            meanmax.append(np.concatenate([m, V.max(0)]))
            Vn = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
            s = Vn @ Cn.T                      # cosine to each centroid
            s = np.exp(s * 12.0)               # soften, no argmax
            s = s / s.sum(1, keepdims=True)
            soft.append(s.sum(0) / len(V))
        return np.vstack(mean), np.vstack(meanmax), np.vstack(soft)

    Mtr, MXtr, Str = pool(tr)
    Mte, MXte, Ste = pool(te)

    results, probs = [], {}

    def dense_arm(name, Atr, Ate):
        sc = StandardScaler().fit(Atr)
        m = lr().fit(sc.transform(Atr), ytr)
        pr = m.predict_proba(sc.transform(Ate))[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, m.predict(sc.transform(Ate)), pr, name=name)
        r["n_features"] = Atr.shape[1]
        results.append(r)
        print(f"  {name:26s} AUROC={r['auroc']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"({Atr.shape[1]:,} dims)")

    def tfidf_arm(name, texts):
        v = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                            sublinear_tf=True, strip_accents="unicode")
        Xtr = v.fit_transform(texts[c] for c in tr)
        Xte = v.transform(texts[c] for c in te)
        m = lr().fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, m.predict(Xte), pr, name=name)
        r["n_features"] = int(Xtr.shape[1])
        results.append(r)
        print(f"  {name:26s} AUROC={r['auroc']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"({Xtr.shape[1]:,} features)")

    print("  reference rungs (same cases):")
    tfidf_arm("masked text", masked)
    tfidf_arm("extracted fact text", fact_text)
    print("\n  the new rung -- facts kept continuous:")
    dense_arm("fact_emb_mean", Mtr, Mte)
    dense_arm("fact_emb_mean_max", MXtr, MXte)
    dense_arm("fact_emb_softcluster", Str, Ste)

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
    for a, b in (("masked text", "fact_emb_mean"), ("fact_emb_mean", "extracted fact text"),
                 ("fact_emb_mean_max", "fact_emb_mean"),
                 ("fact_emb_softcluster", "fact_emb_mean"),
                 ("masked text", "extracted fact text")):
        m, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:48s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    by = {r["name"]: r["auroc"] for r in results}
    best = max(("fact_emb_mean", "fact_emb_mean_max", "fact_emb_softcluster"), key=lambda n: by[n])
    ATOMS = 0.517            # ladder rung 3, same split
    gap_to_text = by["masked text"] - by[best]
    verdict = (
        "DISCRETISATION WAS THE PROBLEM: continuous facts recover most of the text signal, so the "
        "architecture is sound provided step 2 never hardens into a label vocabulary"
        if gap_to_text < 0.02 else
        "EXTRACTION IS THE PROBLEM: continuous facts score near the discrete atoms, so the "
        "information was lost before any vocabulary was involved"
        if by[best] - ATOMS < 0.03 else
        "FACT SELECTION IS THE BOTTLENECK: continuous facts land between the atoms and the full "
        "text, so what is lost is which facts get extracted, not how they are represented")
    print(f"\n  best continuous arm: {best} at {by[best]:.3f}")
    print(f"  reference: masked text {by['masked text']:.3f}, "
          f"extracted fact text {by['extracted fact text']:.3f}, canonical atoms {ATOMS:.3f}")
    print(f"  VERDICT: {verdict}")

    out = paths.EXPERIMENTS / "paper1" / f"fact_embedding_rung_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_train": len(tr), "n_test": len(te),
        "n_facts_with_embeddings": len(facts), "embedding_dim": int(X.shape[1]), "k": args.k,
        "question": "the ladder measured discretisation against DISCRETE vocabularies only. Does a "
                    "continuous fact representation -- what the intended architecture's step 2 "
                    "actually describes -- preserve the signal that discretisation destroys?",
        "canonical_atoms_reference": ATOMS,
        "results": results, "paired": comps, "best_continuous": best, "verdict": verdict},
        indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
