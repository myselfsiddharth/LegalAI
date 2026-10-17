"""Rerank the top-50 precedents against the lawyer's issue memo, using each candidate's case card.

The first stage (narrative + memo, RRF of full-text BM25 and dense) reaches R@50 0.363 but R@10
only 0.173: many right precedents are retrieved and ranked too low. An LLM reads the memo's issues
and ten precedent cards per call and scores each 0-3; the top 50 are re-sorted by score with the
first-stage rank as tie-break.

Missing cards (a card call that failed) keep a neutral score of 1 and are counted, so a card outage
cannot read as the reranker judging a case irrelevant.

Scored on the same 93 held-out cases against the precedents the court actually cited (reachable
set), with a paired bootstrap against the first stage.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from src import paths
from src.llm import client
from product.case_cards import OUT as CARDS
from product.issue_memo import OUT as MEMOS

PROMPT_ID = "rerank.v1"
CANDS = paths.ROOT / "product" / "data" / "candidates_top50.json"
NAR = paths.ROOT / "product" / "data" / "client_narratives.jsonl"
OUT = paths.ROOT / "product" / "experiments" / "rerank.json"
BATCH = 10
SEED = 573


class S(BaseModel):
    i: int
    score: int


class Reply(BaseModel):
    scores: list[S]


def render_card(n: int, title: str, c: dict | None) -> str:
    if c is None:
        return f"{n}. {title}\n   (no card available)"
    return (f"{n}. {title}\n   Dispute: {c['dispute']}\n   Issues decided: {'; '.join(c['issues_decided'])}"
            f"\n   Holdings: {'; '.join(c['holdings'])}\n   Provisions: {'; '.join(c['provisions'])}")


def score_batch(query: str, items: list[tuple[str, str, dict | None]], model: str) -> dict:
    body = query + "\n\n## CANDIDATE PRECEDENTS\n\n" + "\n\n".join(
        render_card(n, t, c) for n, (_, t, c) in enumerate(items, 1))
    res = client.complete([{"role": "system", "content": client.load_prompt(PROMPT_ID)},
                           {"role": "user", "content": body}],
                          model=model, json_schema=Reply, prompt_id=PROMPT_ID, max_tokens=400)
    out = {}
    if res.ok and res.parsed is not None:
        for s in res.parsed.scores:
            if 1 <= s.i <= len(items):
                out[items[s.i - 1][0]] = max(0, min(3, s.score))
    return out


def metrics(ranked, tgt):
    r10 = len(set(ranked[:10]) & tgt) / len(tgt)
    m = next((1 / i for i, d in enumerate(ranked, 1) if d in tgt), 0.0)
    return r10, m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    ap.add_argument("--lists", default="memo",
                    help="'memo' (RRF of full-text BM25 + dense), 'memo_all' (+ card retrievers), "
                         "or 'union' (both top-50s, ordered by memo_all)")
    args = ap.parse_args()

    d = json.loads(CANDS.read_text())
    targets = {c: set(v) for c, v in d["targets"].items()}
    if args.lists == "union":
        cands = {c: list(dict.fromkeys(v["memo_all"] + [x for x in v["memo"] if x not in v["memo_all"]]))
                 for c, v in d["candidates"].items() if "memo_all" in v}
    else:
        cands = {c: v[args.lists] for c, v in d["candidates"].items() if args.lists in v}
    memo = {r["case_id"]: r["memo"] for r in map(json.loads, open(MEMOS)) if r["status"] == "ok"}
    nar = {r["case_id"]: r["narrative"] for r in map(json.loads, open(NAR)) if r["status"] == "ok"}
    cards = {r["doc_id"]: r["card"] for r in map(json.loads, open(CARDS)) if r["status"] == "ok"}
    titles = {r["doc_id"]: r["title"] for r in map(json.loads, open(paths.CASE_REGISTRY))}

    jobs = []
    for c, lst in cands.items():
        m = memo[c]
        query = (f"## CLIENT'S SITUATION\n\n{nar[c]}\n\n## ISSUE MEMO\n\nIssues: {'; '.join(m['issues'])}"
                 f"\nClaims: {'; '.join(m['claims'])}\nDoctrines: {'; '.join(m['doctrines'])}")
        items = [(x, titles.get(x, x), cards.get(x)) for x in lst]
        for i in range(0, len(items), BATCH):
            jobs.append((c, query, items[i:i + BATCH]))
    print(f"{len(cands)} queries, {len(jobs)} rerank calls; "
          f"cards missing for {sum(1 for lst in cands.values() for x in lst if x not in cards)} slots")

    with ThreadPoolExecutor(max_workers=32) as pool:
        res = list(pool.map(lambda j: (j[0], score_batch(j[1], j[2], args.model)), jobs))
    scores: dict[str, dict] = {}
    for c, sc in res:
        scores.setdefault(c, {}).update(sc)

    (paths.ROOT / "product" / "data" / f"rerank_scores_{args.lists}.json").write_text(json.dumps(scores))
    if args.lists == "memo":
        (paths.ROOT / "product" / "data" / "rerank_scores.json").write_text(json.dumps(scores))
    base_r10, base_mrr, rr_r10, rr_mrr = [], [], [], []
    unscored = 0
    dist = Counter()
    for c, lst in cands.items():
        sc = scores.get(c, {})
        unscored += sum(1 for x in lst if x not in sc)
        for x in lst:
            dist[sc.get(x, "missing")] += 1
        reranked = sorted(lst, key=lambda x: (-sc.get(x, 1), lst.index(x)))
        a, b = metrics(lst, targets[c]); base_r10.append(a); base_mrr.append(b)
        a, b = metrics(reranked, targets[c]); rr_r10.append(a); rr_mrr.append(b)
    diff = np.array(rr_r10) - np.array(base_r10)
    rng = np.random.default_rng(SEED)
    bs = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(2000)]
    out = {"n": len(cands), "first_stage": {"r10": round(float(np.mean(base_r10)), 4),
                                            "mrr": round(float(np.mean(base_mrr)), 4)},
           "reranked": {"r10": round(float(np.mean(rr_r10)), 4), "mrr": round(float(np.mean(rr_mrr)), 4)},
           "r10_gain": round(float(diff.mean()), 4),
           "r10_gain_ci95": [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
           "unscored_slots": unscored, "score_distribution": {str(k): v for k, v in dist.items()}}
    out["lists"] = args.lists
    OUT.with_name(f"rerank_{args.lists}.json").write_text(json.dumps(out, indent=1))
    if args.lists == "memo":
        OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
