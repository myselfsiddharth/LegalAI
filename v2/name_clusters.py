"""Stage 3d — name the induced clusters, and ask what KIND of fact carries the outcome signal.

The finding that motivates this: at matched granularity (66 dimensions, soft, same embeddings, same
cases) a **data-driven** basis scores 0.632 and the **authored doctrinal** basis scores 0.524. So the
outcome-predictive content of these facts is largely orthogonal to legal element structure
(ARCHITECTURE.md §6d). That leaves an obvious question nobody has asked: *what are the predictive
clusters actually about?*

Three possible answers, and they have very different consequences:

  **legal substance** -- the clusters are doctrinal after all, just not the doctrine the authored
      ontology chose. Then this produces a better, data-grounded ontology, which is the best case.
  **procedural** -- the signal is forum, stage, appeal route, remand history. Then the pipeline is
      learning court machinery, and v1 already showed metadata alone (year + forum) reaches 0.582,
      so this would be consistent and deflationary rather than surprising.
  **party or quantum** -- the signal is who the parties are, or dates and sums. Deflationary too,
      and a fairness concern if the model is keying on the State being a litigant.

The prompt is written to resist the flattering answer: it says in terms that an honest `procedural`
label is worth more here than a legal-sounding one. That matters because a model asked to describe
legal text will reach for legal language by default.

## How clusters are ranked

Univariate AUROC of each cluster's soft-histogram column, computed **on TRAIN only**, so the ranking
never looks at test. Strength is |AUROC - 0.5| and direction is its sign. Reporting the top clusters
by a train-only statistic and then quoting the model's test AUROC separately keeps selection and
evaluation apart.

## The comparison that makes this a finding rather than a word cloud

Every named cluster is matched to its nearest authored element by descriptor cosine. If the clusters
that predict best sit FAR from every element in the 66-element ontology, the orthogonality result
gets a content-level explanation and not merely an AUROC gap.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from pydantic import BaseModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.label_merge import load_final
from src.extract.induced_vocab import load_fact_texts
from src.llm import client
from v2 import paths
from v2.canon_soft import soft_histogram
from v2.element_catalogue import all_elements, load_claims

SEED = 573
PROMPT_ID = "name_fact_cluster.v1"
MODEL = "qwen3-235b-a22b-instruct-2507"
TEMP = 0.25


class ClusterName(BaseModel):
    name: str
    definition: str = ""
    kind: Literal["legal_substance", "procedural", "party_entity",
                  "temporal_quantum", "incoherent"]
    legal_element: str | None = None
    confidence: float = 0.5


def name_one(members: list[str]) -> ClusterName | None:
    body = "\n".join(f"- {m}" for m in members)
    res = client.complete(
        [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
         {"role": "user", "content": f"## STATEMENTS IN THIS GROUP\n\n{body}"}],
        model=MODEL, json_schema=ClusterName, prompt_id=PROMPT_ID, max_tokens=600)
    return res.parsed if res.ok else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--k", type=int, default=66,
                    help="66 matches the authored ontology's element count, so the two bases are "
                         "compared at identical granularity")
    ap.add_argument("--members", type=int, default=12, help="representative facts shown per cluster")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    facts, X = load_fact_texts()
    by_case: dict[str, list[int]] = {}
    for i, f in enumerate(facts):
        by_case.setdefault(f["case_id"], []).append(i)

    tr = [c for c in split["train"] if c in by_case and c in ylab]
    te = [c for c in split["test"] if c in by_case and c in ylab]
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])

    rows = np.array([i for c in tr for i in by_case[c]])
    print(f"fitting k={args.k} on {len(rows):,} TRAIN facts", flush=True)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED, n_init=5,
                         batch_size=2048).fit(X[rows])
    C = km.cluster_centers_
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)

    Htr = np.vstack([soft_histogram(X[by_case[c]], Cn, TEMP) for c in tr])
    Hte = np.vstack([soft_histogram(X[by_case[c]], Cn, TEMP) for c in te])

    sc = StandardScaler().fit(Htr)
    m = LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced",
                           random_state=SEED).fit(sc.transform(Htr), ytr)
    test_auroc = roc_auc_score(yte, m.predict_proba(sc.transform(Hte))[:, 1])
    coefs = m.coef_[0]
    print(f"model test AUROC {test_auroc:.3f}  (selection below uses TRAIN only)\n")

    # univariate strength, TRAIN only
    uni = np.array([roc_auc_score(ytr, Htr[:, j]) for j in range(args.k)])
    strength = np.abs(uni - 0.5)
    order = list(np.argsort(-strength))

    # representative facts per cluster: nearest to the centroid, among TRAIN facts
    Xn = X[rows] / (np.linalg.norm(X[rows], axis=1, keepdims=True) + 1e-9)
    sims = Xn @ Cn.T
    members = {}
    for j in range(args.k):
        top = np.argsort(-sims[:, j])[: args.members]
        members[j] = [facts[rows[t]]["text"][:220] for t in top]

    print(f"naming {args.k} clusters with {MODEL} ({args.workers} workers)", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        named = list(pool.map(lambda j: name_one(members[j]), range(args.k)))
    got = sum(1 for n in named if n is not None)
    print(f"  named {got}/{args.k}\n")

    # nearest authored element, by descriptor cosine
    els = all_elements(load_claims())
    etexts = [f"{e.claim_name} — {e.name}. {e.definition}".strip() for e in els]
    ctexts = [f"{(named[j].name if named[j] else 'unnamed')}. "
              f"{(named[j].definition if named[j] else '')}" for j in range(args.k)]
    E = np.asarray(client.embed(etexts, verbose=False), dtype=np.float32)
    Cc = np.asarray(client.embed(ctexts, verbose=False), dtype=np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    Cc /= (np.linalg.norm(Cc, axis=1, keepdims=True) + 1e-9)
    near = Cc @ E.T

    out_rows = []
    for j in range(args.k):
        n = named[j]
        bi = int(near[j].argmax())
        out_rows.append({
            "cluster": j, "name": n.name if n else None,
            "kind": n.kind if n else None,
            "legal_element": n.legal_element if n else None,
            "definition": n.definition if n else None,
            "confidence": round(n.confidence, 2) if n else None,
            "train_univariate_auroc": round(float(uni[j]), 4),
            "strength": round(float(strength[j]), 4),
            "favours": "WIN" if uni[j] > 0.5 else "LOSE",
            "lr_coef": round(float(coefs[j]), 3),
            "nearest_authored_element": els[bi].element_id,
            "nearest_authored_name": f"{els[bi].claim_name} — {els[bi].name}",
            "nearest_similarity": round(float(near[j].max()), 3),
            "members_sample": members[j][:3]})

    print(f"  === top {args.top} clusters by TRAIN univariate strength ===")
    print(f"  {'name':34s} {'kind':16s} {'favours':8s} {'uAUROC':>7s} {'~elem':>6s}")
    for r in [out_rows[j] for j in order[: args.top]]:
        print(f"  {str(r['name'])[:33]:34s} {str(r['kind'])[:15]:16s} {r['favours']:8s} "
              f"{r['train_univariate_auroc']:.3f}   {r['nearest_similarity']:.2f}")

    import collections
    kinds_all = collections.Counter(r["kind"] for r in out_rows if r["kind"])
    kinds_top = collections.Counter(out_rows[j]["kind"] for j in order[: args.top]
                                    if out_rows[j]["kind"])
    print(f"\n  kind mix, all {args.k}: " + ", ".join(f"{k} {v}" for k, v in kinds_all.most_common()))
    print(f"  kind mix, top {args.top}:  " + ", ".join(f"{k} {v}" for k, v in kinds_top.most_common()))
    sub = kinds_top.get("legal_substance", 0)
    print(f"\n  >>> of the {args.top} most predictive clusters, {sub} "
          f"({sub/args.top:.0%}) are legal substance <<<")
    sim_top = float(np.mean([out_rows[j]["nearest_similarity"] for j in order[: args.top]]))
    sim_all = float(np.mean([r["nearest_similarity"] for r in out_rows]))
    print(f"  mean similarity to the nearest authored element: "
          f"top {args.top} = {sim_top:.3f}, all = {sim_all:.3f}")

    out = paths.EXPERIMENTS / f"named_clusters_k{args.k}_{args.split}.json"
    out.write_text(json.dumps({
        "stage": "3d name the induced clusters", "split": args.split, "k": args.k,
        "model": MODEL, "prompt_id": PROMPT_ID, "temp": TEMP,
        "model_test_auroc": round(float(test_auroc), 4),
        "selection": "TRAIN-only univariate AUROC; test never used for ranking",
        "kind_mix_all": dict(kinds_all), "kind_mix_top": dict(kinds_top),
        "legal_substance_share_top": round(sub / args.top, 4),
        "mean_nearest_similarity_top": round(sim_top, 4),
        "mean_nearest_similarity_all": round(sim_all, 4),
        "clusters": out_rows, "top_order": [int(j) for j in order]}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
