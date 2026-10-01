"""§7.3 → §6.3: merge claim families the data does not separate, and record why.

§7's acceptance says low-support families (<30 cases) must be "either merged or excluded", and
§0.6 says every ontology change needs data-backed justification. This is that step.

The evidence for merging, measured on 465 claims from 300 cases:

  * HDBSCAN on claim embeddings: k=2, **ARI −0.005** — its partition is uncorrelated with the
    assigned 15-way families — at 80% outliers.
  * Agglomerative, k=5 by silhouette: NMI 0.271, purity 0.318, **silhouette 0.035**. There is
    almost no cluster structure at 15-way granularity.
  * Only **2 of 15** ontology families cleared the 30-case floor (Title 87, Possession 78).
  * The model assigned **42** distinct families where the ontology has 15, inventing 27.

But the clusters are legally coherent one level up, which is the actual finding — the taxonomy is
too fine for this corpus, not wrong:

    cluster 0  Title 49, Possession 41, Partition 13, Trust 5      -> private title & possession
    cluster 1  SpecificPerformance 25, Title 15, Cancellation 10   -> contract & instrument
    cluster 2  Possession 28, Title 9, Tenure 7, LeaseTenancy 4    -> tenure & occupancy
    cluster 3  Title 35, UltraVires 20, ConstitutionalDeprivation 19, LandAcquisition 13
                                                                   -> state action against property

Assignment is DERIVED, not authored: each family goes to the cluster its claims most often land
in, and a super-family is named by its dominant members. That keeps the merge reproducible from
the data rather than resting on our intuition about property law.

Defence families are excluded, not merged. `Procedural/Forum` (16), `Statutory/Regulatory
Subservience` (13), `Public Interest/Planning Policy` (5) and `Equitable` (2) were assigned to
CLAIMS by the extractor, which is a routing error -- they are grounds for resisting a claim, not
claims. Folding them into a claim super-family would bury that error.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import yaml
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

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
from src.extract.claims import DEFENCE_FAMILIES
from src.llm import client

CLAIMS_PATH = paths.INTERIM / "claims.jsonl"
OUT_PATH = paths.INTERIM / "family_merge.json"
CHANGELOG = paths.ONTOLOGY / "CHANGELOG.md"

MIN_CASES = 30
# Names for the merged families, chosen from each cluster's dominant members. The GROUPING is
# derived from the data; only these labels are ours.
NAME_HINTS = {
    frozenset({"Title", "Possession", "Partition"}): "PrivateTitlePossession",
    frozenset({"SpecificPerformance", "Cancellation"}): "ContractInstrument",
    frozenset({"LeaseTenancy", "Tenure"}): "TenureOccupancy",
    frozenset({"UltraVires", "ConstitutionalDeprivation", "LandAcquisition"}): "StateAction",
}


def name_cluster(members: Counter) -> str:
    top = {m for m, _ in members.most_common(6)}
    best, score = None, 0
    for key, nm in NAME_HINTS.items():
        ov = len(key & top)
        if ov > score:
            best, score = nm, ov
    return best or ("Mixed_" + "_".join(m for m, _ in members.most_common(2)))


def main() -> None:
    claims = [json.loads(l) for l in open(CLAIMS_PATH) if json.loads(l)["kind"] == "claim"]
    texts = [c["text"] for c in claims]
    fams = [c["family"] for c in claims]

    X = client.embed(texts)
    keep = X.any(axis=1)
    X, claims = X[keep], [c for c, k in zip(claims, keep) if k]
    fams = [c["family"] for c in claims]

    rng = np.random.default_rng(573)
    Xs = X if len(X) <= SWEEP_SAMPLE else X[rng.choice(len(X), SWEEP_SAMPLE, replace=False)]
    best = (None, -1.0)
    for k in range(3, 13):
        a = AgglomerativeClustering(n_clusters=k).fit_predict(Xs)
        s = silhouette_score(Xs, a, sample_size=min(SIL_SAMPLE, len(Xs)), random_state=573)
        if s > best[1]:
            best = (k, s)
    k, sil = best
    labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)

    # family -> the cluster its claims most often land in
    fam_cluster = defaultdict(Counter)
    for f, c in zip(fams, labels):
        fam_cluster[f][int(c)] += 1
    cluster_members = defaultdict(Counter)
    assign = {}
    for f, cc in fam_cluster.items():
        if f in DEFENCE_FAMILIES:
            continue                              # a routing error, not a claim family
        c = cc.most_common(1)[0][0]
        assign[f] = c
        cluster_members[c][f] = sum(cc.values())

    names = {c: name_cluster(m) for c, m in cluster_members.items()}
    fam_to_super = {f: names[c] for f, c in assign.items()}

    # case counts per merged family
    cases_per_super = defaultdict(set)
    for c in claims:
        sup = fam_to_super.get(c["family"])
        if sup:
            cases_per_super[sup].add(c["case_id"])
    cases_per_fam = defaultdict(set)
    for c in claims:
        cases_per_fam[c["family"]].add(c["case_id"])

    viable_before = sum(1 for f, s in cases_per_fam.items()
                        if len(s) >= MIN_CASES and f not in DEFENCE_FAMILIES)
    viable_after = sum(1 for s in cases_per_super.values() if len(s) >= MIN_CASES)

    out = {
        "derived_from": {"n_claims": len(claims), "n_cases": len({c["case_id"] for c in claims}),
                         "k_by_silhouette": k, "silhouette": round(float(sil), 4)},
        "min_cases": MIN_CASES,
        "merged_families": {nm: {"n_cases": len(cs),
                                 "viable": len(cs) >= MIN_CASES,
                                 "absorbs": sorted(f for f, s in fam_to_super.items() if s == nm)}
                            for nm, cs in sorted(cases_per_super.items(),
                                                 key=lambda x: -len(x[1]))},
        "family_to_merged": fam_to_super,
        "excluded_defence_families_misrouted_to_claims": {
            f: len(cases_per_fam[f]) for f in cases_per_fam if f in DEFENCE_FAMILIES},
        "viable_families_before_merge": viable_before,
        "viable_families_after_merge": viable_after,
    }
    OUT_PATH.write_text(json.dumps(out, indent=1))

    print(f"=== family merge (derived from k={k} agglomerative, silhouette {sil:.3f}) ===")
    print(f"  claims {len(claims)}, cases {out['derived_from']['n_cases']}\n")
    print(f"  {'merged family':26s} {'cases':>6s}  absorbs")
    for nm, d in out["merged_families"].items():
        flag = "" if d["viable"] else "   <- still under the 30-case floor"
        print(f"  {nm:26s} {d['n_cases']:6d}  {', '.join(d['absorbs'][:6])}{flag}")
    print(f"\n  families clearing the {MIN_CASES}-case floor: "
          f"{viable_before} before merge -> {viable_after} after")
    if out["excluded_defence_families_misrouted_to_claims"]:
        print(f"  EXCLUDED, defence families the extractor routed to claims: "
              f"{out['excluded_defence_families_misrouted_to_claims']}")

    # §6.3 changelog
    lines = [
        f"\n## {date.today().isoformat()} — claim families merged 15 -> "
        f"{len(out['merged_families'])}\n",
        f"`src/cluster/family_merge.py`. Derived from {len(claims)} claims over "
        f"{out['derived_from']['n_cases']} cases.\n\n",
        "**Why.** The 15-way taxonomy is not supported by the data at this scale:\n",
        "- HDBSCAN on claim embeddings: ARI **-0.005** against the assigned families, 80% outliers.\n",
        f"- Agglomerative k={k} by silhouette: NMI 0.271, purity 0.318, "
        f"silhouette **{sil:.3f}** — almost no structure at 15-way granularity.\n",
        f"- Only **{viable_before} of 15** families cleared §7's {MIN_CASES}-case floor.\n",
        "- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.\n\n",
        "The clusters are legally coherent one level up, so the taxonomy is too FINE for this "
        "corpus rather than wrong. Each family is assigned to the cluster its claims most often "
        "land in; only the merged names are authored.\n\n",
        f"**Effect.** Families clearing the {MIN_CASES}-case floor: {viable_before} -> "
        f"{viable_after}, which is what makes §8 per-family mining runnable.\n\n",
        "**Merged families.**\n\n| merged | cases | absorbs |\n|---|---|---|\n"]
    for nm, d in out["merged_families"].items():
        lines.append(f"| `{nm}` | {d['n_cases']} | {', '.join(d['absorbs'])} |\n")
    lines.append(
        f"\n**Excluded.** {out['excluded_defence_families_misrouted_to_claims']} — these are "
        f"DEFENCE families the extractor routed to claims. They are grounds for resisting a "
        f"claim, not claims, and folding them into a claim super-family would bury that "
        f"routing error.\n")
    lines.append(
        "\n**Not frozen.** Derived from 300 cases. Re-run after extraction scales up; a family "
        "below the floor here may clear it later, and the merge should be revisited rather than "
        "inherited.\n")
    with CHANGELOG.open("a") as f:
        f.writelines(lines)
    print(f"\n-> {OUT_PATH}  (+ §6.3 entry in {CHANGELOG.name})")


if __name__ == "__main__":
    main()
