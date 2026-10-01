"""Stage 3 (§7.2-7.3): cluster claim texts and check the ontology's 15 families against them.

The point is not to produce clusters. It is to find out whether the ontology's taxonomy carves
this corpus at its joints. §7.3 asks for three specific answers and this module reports all three:

  * how well unsupervised clusters agree with the assigned families (NMI, purity, ARI);
  * clusters that match no module -- candidate families the ontology is missing;
  * modules that match no cluster -- families the ontology asserts but the data does not support.

Either mismatch is a result. A module with no cluster is not automatically wrong -- it may be a
real but rare claim type -- so §7's 30-case floor is reported alongside, and the §6.3 change log
is where a merge or split gets justified.

Two clusterers, as §7.2 requires: HDBSCAN (density-based, finds its own k, labels outliers -1)
and agglomerative with k chosen by silhouette. If they disagree sharply, the structure is weak
and the alignment numbers should not be over-read.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score)

from src import paths

# Choosing k by silhouette is O(n^2) in both the clustering fit and the score, and it is run once
# per candidate k. At 7,441 claims x 4096 dims that measured at 99.5% CPU for 19 minutes with no
# end in sight -- roughly 21 fits plus 21 scores, each pairwise over 7,441 points.
#
# The sweep only has to RANK candidate k values, which a subsample does just as well, so k is chosen
# on a sample and the final clustering is then fit on every point. `silhouette_score` also takes its
# own `sample_size`, which bounds the scoring independently of the fit.
SWEEP_SAMPLE = 2500
SIL_SAMPLE = 2000
from src.llm import client
from src.extract.claims import FAMILIES

CLAIMS_PATH = paths.INTERIM / "claims.jsonl"
OUT_PATH = paths.INTERIM / "claim_families.json"


def load_claims() -> list[dict]:
    return [json.loads(l) for l in open(CLAIMS_PATH) if json.loads(l)["kind"] == "claim"]


def purity(clusters: np.ndarray, labels: list[str]) -> float:
    """Fraction of claims whose cluster's majority family is their own family. Outliers excluded."""
    tot = hit = 0
    for c in set(clusters):
        if c == -1:
            continue
        members = [labels[i] for i in range(len(labels)) if clusters[i] == c]
        if not members:
            continue
        tot += len(members)
        hit += Counter(members).most_common(1)[0][1]
    return hit / tot if tot else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-cluster-size", type=int, default=8)
    ap.add_argument("--max-k", type=int, default=24)
    args = ap.parse_args()

    claims = load_claims()
    if len(claims) < 40:
        raise SystemExit(f"only {len(claims)} claims -- run `python -m src.extract.claims` first")
    texts = [c["text"] for c in claims]
    fams = [c["family"] for c in claims]

    print(f"embedding {len(texts)} claim texts...", flush=True)
    X = client.embed(texts, verbose=False)
    keep = X.any(axis=1)
    if not keep.all():
        print(f"  {int((~keep).sum())} embeddings never arrived; excluded (NOT treated as "
              f"unclusterable)")
        X, texts, fams = X[keep], [t for t, k in zip(texts, keep) if k], \
                         [f for f, k in zip(fams, keep) if k]
        claims = [c for c, k in zip(claims, keep) if k]

    results = {"n_claims": len(texts), "n_assigned_families": len(set(fams))}

    # --- HDBSCAN: density-based, chooses its own k, marks outliers -1
    hdb = HDBSCAN(min_cluster_size=args.min_cluster_size, metric="euclidean")
    h = hdb.fit_predict(X)
    n_h = len({c for c in h if c != -1})
    outlier = float((h == -1).mean())
    results["hdbscan"] = {
        "n_clusters": n_h, "outlier_frac": round(outlier, 4),
        "nmi": round(normalized_mutual_info_score(fams, h), 4),
        "ari": round(adjusted_rand_score(fams, h), 4),
        "purity": round(purity(h, fams), 4)}

    # --- agglomerative with k by silhouette
    rng = np.random.default_rng(573)
    if len(X) > SWEEP_SAMPLE:
        sel = rng.choice(len(X), SWEEP_SAMPLE, replace=False)
        Xs = X[sel]
        print(f"  choosing k on a {SWEEP_SAMPLE}-point subsample "
              f"(the sweep only has to rank k; the final fit uses all {len(X)})", flush=True)
    else:
        Xs = X
    best = (None, -1.0)
    sil_curve = {}
    for k in range(4, min(args.max_k, len(Xs) // 4) + 1):
        a = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        if len(set(a)) < 2:
            continue
        s = silhouette_score(Xs, a, sample_size=min(SIL_SAMPLE, len(Xs)), random_state=573)
        sil_curve[k] = round(float(s), 4)
        if s > best[1]:
            best = (k, s)
    k_best = best[0]
    print(f"  k={k_best} by silhouette; fitting on all {len(X)} points", flush=True)
    agg = AgglomerativeClustering(n_clusters=k_best).fit_predict(X)
    results["agglomerative"] = {
        "k_by_silhouette": k_best, "silhouette": round(float(best[1]), 4),
        "silhouette_curve": sil_curve,
        "nmi": round(normalized_mutual_info_score(fams, agg), 4),
        "ari": round(adjusted_rand_score(fams, agg), 4),
        "purity": round(purity(agg, fams), 4)}

    # --- which modules and clusters fail to line up (§7.3)
    #
    # A cluster is "matched" to the family of its majority member when that family holds a real
    # plurality; below that the cluster is mixed and names no family.
    MAJORITY = 0.5
    cl_to_fam, mixed = {}, []
    for c in sorted({x for x in agg}):
        members = [fams[i] for i in range(len(fams)) if agg[i] == c]
        top, n = Counter(members).most_common(1)[0]
        (cl_to_fam.__setitem__(c, top) if n / len(members) >= MAJORITY
         else mixed.append({"cluster": int(c), "size": len(members),
                            "composition": dict(Counter(members).most_common(4))}))
    matched = set(cl_to_fam.values())
    assigned = Counter(fams)
    cases_per_fam = Counter()
    for pair in {(c["case_id"], c["family"]) for c in claims}:
        cases_per_fam[pair[1]] += 1

    results["modules_with_no_cluster"] = [
        {"family": f, "n_claims": assigned[f], "n_cases": cases_per_fam[f]}
        for f in sorted(assigned) if f not in matched]
    results["mixed_clusters"] = mixed
    results["families_below_30_case_floor"] = [
        {"family": f, "n_cases": cases_per_fam[f]}
        for f in sorted(assigned) if cases_per_fam[f] < 30]
    results["ontology_families_never_assigned"] = [f for f in FAMILIES if f not in assigned]
    results["new_family_proposals"] = dict(
        Counter(c["family"] for c in claims if c.get("family_is_new")).most_common(15))

    # --- per-case family membership: many-to-many, plus a primary (§7.4)
    by_case = defaultdict(Counter)
    for c in claims:
        by_case[c["case_id"]][c["family"]] += 1
    membership = {case: {"families": sorted(cnt), "primary": cnt.most_common(1)[0][0]}
                  for case, cnt in by_case.items()}
    results["n_cases"] = len(membership)
    results["cases_with_multiple_families"] = sum(1 for m in membership.values()
                                                  if len(m["families"]) > 1)
    OUT_PATH.write_text(json.dumps({"summary": results, "membership": membership}, indent=1))

    # --- report
    print(f"\n=== claim families: {len(texts)} claims, {len(membership)} cases ===")
    for name in ("hdbscan", "agglomerative"):
        r = results[name]
        extra = (f"outliers {r['outlier_frac']:.1%}" if name == "hdbscan"
                 else f"silhouette {r['silhouette']:.3f}")
        k = r.get("n_clusters", r.get("k_by_silhouette"))
        print(f"  {name:14s} k={k:3d}  NMI={r['nmi']:.3f}  ARI={r['ari']:.3f}  "
              f"purity={r['purity']:.3f}  {extra}")
    print(f"\n  cases in more than one family: {results['cases_with_multiple_families']} "
          f"({100*results['cases_with_multiple_families']/max(1,len(membership)):.0f}%)")

    print(f"\n  assigned families ({len(assigned)}):")
    for f, n in assigned.most_common():
        flag = "  <- under §7's 30-case floor" if cases_per_fam[f] < 30 else ""
        print(f"    {f:30s} {n:4d} claims  {cases_per_fam[f]:4d} cases{flag}")
    if results["ontology_families_never_assigned"]:
        print(f"\n  ontology families never assigned to any claim: "
              f"{results['ontology_families_never_assigned']}")
    if results["modules_with_no_cluster"]:
        print(f"  families that no cluster's majority names (candidates to merge): "
              f"{[m['family'] for m in results['modules_with_no_cluster']]}")
    if mixed:
        print(f"  mixed clusters naming no family: {len(mixed)}")
        for m in mixed[:4]:
            print(f"    cluster {m['cluster']} (n={m['size']}): {m['composition']}")
    if results["new_family_proposals"]:
        print(f"  NEW family proposals: {results['new_family_proposals']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
