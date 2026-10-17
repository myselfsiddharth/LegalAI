"""Attach a good-law record to every retrieved precedent, and test whether it helps ranking.

For a query decided in year Y, a precedent's record counts only treatments by cases decided BEFORE
Y -- what a lawyer researching at that time could have known. The record:
  followed / distinguished / doubted / overruled   counts, from treatments.jsonl
  status    OVERRULED if any confirmed overruling, DOUBTED if doubted, else GOOD (or UNKNOWN when
            no later court is recorded treating it at all)

Two measurements on the 93 held-out client narratives:
  coverage     how many top-10 precedents carry any recorded treatment, and how many are flagged
  re-ranking   tie-break the reranker's 0-3 scores by followed count instead of first-stage rank;
               and demote OVERRULED. Does recall move?
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import numpy as np

from src import paths

P = paths.ROOT / "product"
EXP = P / "experiments" / "good_law.json"
SEED = 573




def main() -> None:
    reg = {json.loads(l)["doc_id"]: json.loads(l) for l in open(paths.CASE_REGISTRY)}
    tr = [json.loads(l) for l in open(P / "data" / "treatments.jsonl")]
    by_cited = defaultdict(list)
    for r in tr:
        if r.get("treatment"):
            by_cited[r["cited"]].append((r["citing_year"], r["treatment"]))

    def record(doc: str, year_limit: int) -> dict:
        c = Counter(t for y, t in by_cited.get(doc, []) if y and y < year_limit)
        status = ("OVERRULED" if c["OVERRULED"] else "DOUBTED" if c["DOUBTED"]
                  else "GOOD" if sum(c.values()) else "UNKNOWN")
        return {"followed": c["FOLLOWED"], "distinguished": c["DISTINGUISHED"],
                "doubted": c["DOUBTED"], "overruled": c["OVERRULED"], "referred": c["REFERRED"],
                "status": status}

    d = json.loads((P / "data" / "candidates_top50.json").read_text())
    scores = json.loads((P / "data" / "rerank_scores.json").read_text())
    targets = {c: set(v) for c, v in d["targets"].items()}
    cands = {c: v["memo"] for c, v in d["candidates"].items()}

    cov = Counter()
    arms = defaultdict(list)
    for c, lst in cands.items():
        yq = reg[c]["year"]
        sc = scores.get(c, {})
        recs = {x: record(x, yq) for x in lst}
        rr = sorted(lst, key=lambda x: (-sc.get(x, 1), lst.index(x)))
        for x in rr[:10]:
            cov[recs[x]["status"]] += 1
            cov["any_treatment"] += recs[x]["status"] != "UNKNOWN"
            cov["followed_at_least_once"] += recs[x]["followed"] > 0
        by_follow = sorted(lst, key=lambda x: (-sc.get(x, 1), -recs[x]["followed"], lst.index(x)))
        demote = sorted(rr, key=lambda x: recs[x]["status"] == "OVERRULED")
        for name, ranked in (("reranked", rr), ("rerank_then_followed", by_follow),
                             ("rerank_demote_overruled", demote)):
            arms[name].append(len(set(ranked[:10]) & targets[c]) / len(targets[c]))

    rng = np.random.default_rng(SEED)
    base = np.array(arms["reranked"])
    out = {"n": len(cands), "top10_coverage": dict(cov), "r10": {}}
    for k, v in arms.items():
        v = np.array(v)
        bs = [(v - base)[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
        out["r10"][k] = {"mean": round(float(v.mean()), 4),
                         "vs_reranked": round(float((v - base).mean()), 4),
                         "ci95": [round(float(np.percentile(bs, 2.5)), 4),
                                  round(float(np.percentile(bs, 97.5)), 4)]}
    EXP.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
