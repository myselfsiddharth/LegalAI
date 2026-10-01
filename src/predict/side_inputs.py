"""Generate §10.1's `S` and `R` feature inputs: predicted statutes and retrieved precedents.

Both groups are specified by §10.1 and have never had data. Both are also the places where §5.2's
leakage rule bites hardest, so the construction matters more than the modelling:

**`S` — predicted statutes, never court-cited ones.** §5.2: "statutes cited in the analysis section
are chosen by the court knowing the outcome ... Court-cited authorities are targets, never ordinary
inputs." So the statute model is fit on TRAIN cases only, predicting the provisions a court would
introduce, and its probabilities are written for every case. A test case's features therefore come
from a model that never saw it.

**`R` — retrieved precedents and their outcomes.** Retrieval is time-respecting (a precedent must
predate the query), and the feature consumer counts only precedents whose outcome label was
available at training time. A looser reading of "earlier precedents" would let one test case's
outcome inform another's prediction; restricting to train labels forecloses that.

Retrieval uses the best system measured in §9.3: RRF over BM25 and dense, both on masked text.
Dense over extracted *facts* was 4x weaker and is not used.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from src import paths
from src.authorities import statute_predict as SP
from src.authorities.retrieval import BM25Index, scrub_citations
from src.authorities.precedent_retrieve import rrf
from src.data.label_merge import load_final
from src.llm import client

SEED = 573
STATUTES_OUT = paths.INTERIM / "predicted_statutes.jsonl"
PRECEDENTS_OUT = paths.INTERIM / "retrieved_precedents.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--min-train-cases", type=int, default=30)
    ap.add_argument("--skip-statutes", action="store_true")
    ap.add_argument("--skip-precedents", action="store_true")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    train = set(split["train"])
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    reg = {r["doc_id"]: r for r in (json.loads(l) for l in open(paths.CASE_REGISTRY))}
    years = {d: reg[d]["year"] for d in masked if reg.get(d, {}).get("year")}
    ids = sorted(d for d in masked if masked[d] and d in years)
    print(f"{len(ids):,} cases with masked text and a year")

    # ---------------------------------------------------------------- S
    if not args.skip_statutes:
        if not SP.TARGETS_PATH.exists():
            SP.build_targets()
        targets = SP.load_targets("labels_novel")
        tr = [d for d in ids if d in train and targets.get(d)]
        df = Counter(l for d in tr for l in targets[d])
        labels = sorted(l for l, n in df.items() if n >= args.min_train_cases)
        print(f"\nS: fitting statute model on {len(tr):,} TRAIN cases, "
              f"{len(labels)} provisions")
        Y = np.zeros((len(tr), len(labels)), dtype=np.int8)
        lidx = {l: i for i, l in enumerate(labels)}
        for i, d in enumerate(tr):
            for l in targets[d]:
                if l in lidx:
                    Y[i, lidx[l]] = 1
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(masked[d] for d in tr)
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=SEED), n_jobs=-1).fit(Xtr, Y)
        P = clf.predict_proba(vec.transform(masked[d] for d in ids))
        with STATUTES_OUT.open("w") as out:
            for i, d in enumerate(ids):
                # keep only the provisions the model actually leans toward; a dense row of near-zero
                # probabilities would be 46 noise features per case
                scores = {labels[j]: round(float(P[i, j]), 4)
                          for j in np.argsort(-P[i])[:10] if P[i, j] >= 0.10}
                out.write(json.dumps({"doc_id": d, "scores": scores}) + "\n")
        nz = sum(1 for i in range(len(ids)) if (P[i] >= 0.10).any())
        print(f"   -> {STATUTES_OUT}  ({nz:,} of {len(ids):,} cases have >=1 provision at p>=0.10)")

    # ---------------------------------------------------------------- R
    if not args.skip_precedents:
        print(f"\nR: retrieving top-{args.top_k} earlier precedents per case")
        scrubbed = {d: scrub_citations(masked[d]) for d in ids}
        idx = BM25Index()
        idx.build(lambda: ((d, years[d], scrubbed[d]) for d in ids), verbose=False)

        T = client.embed([scrubbed[d][:8000] for d in ids], batch_size=32, workers=24)
        ok = T.any(axis=1)
        if not ok.all():
            print(f"   {int((~ok).sum())} text embeddings never arrived; "
                  f"those cases get BM25 only")
        T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        tpos = {d: i for i, d in enumerate(ids)}
        yarr = np.array([years[d] for d in ids])

        written = 0
        with PRECEDENTS_OUT.open("w") as out:
            for n, d in enumerate(ids, 1):
                yq = years[d]
                b = [x for x, _ in idx.search(scrubbed[d], args.top_k, before_year=yq - 1,
                                              exclude={d})]
                if ok[tpos[d]]:
                    sim = T @ T[tpos[d]]
                    sim = np.where((yarr < yq) & ok, sim, -np.inf)
                    top = np.argsort(-sim)[:args.top_k + 1]
                    dn = [ids[i] for i in top if ids[i] != d and np.isfinite(sim[i])][:args.top_k]
                else:
                    dn = []
                fused = rrf([b, dn])[:args.top_k] if dn else b
                # similarity for the weighting: reciprocal rank, which is comparable across arms
                hits = [{"doc_id": h, "score": round(1.0 / (1 + i), 4)}
                        for i, h in enumerate(fused)]
                # time-respecting, asserted
                assert all(years[h["doc_id"]] < yq for h in hits), f"time leak for {d}"
                out.write(json.dumps({"doc_id": d, "hits": hits}) + "\n")
                written += 1
                if n % 1000 == 0:
                    print(f"   {n}/{len(ids)}", flush=True)
        print(f"   -> {PRECEDENTS_OUT}  ({written:,} cases)")
        print("   time-respecting assertion passed for every case")


if __name__ == "__main__":
    main()
