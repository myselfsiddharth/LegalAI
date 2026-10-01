"""Stage 5 (§9.3): retrieve the precedents a court cited, from the facts alone.

Given a case, rank earlier cases by how likely the court was to cite them. Targets are the
precedents the court actually cited **that exist in this corpus** — §9.3's "court-cited precedents
present in the DB".

Three things govern whether the number means anything:

**1. Reachability caps recall, and the cap must be reported.** Of 46,904 case-to-case citation
edges, only 10,360 (22.1%) point at a case this corpus contains — it is Supreme Court land/property
only, while judgments cite High Courts, the Privy Council and English decisions. Recall is therefore
reported against *reachable* targets, with the raw figure beside it. Quoting recall against all
cited authorities would make any retriever look hopeless; quoting it against reachable ones without
saying so would make it look better than it is.

**2. The query must not contain its own answer.** A judgment names the authority it relies on in the
same sentence that reasons about it, so queries are passed through `scrub_citations`, which drops
every sentence carrying a case connector or a reporter abbreviation. Queries are built from
`masked_text` (§5.2), so the court's analysis is already gone.

**3. A `mention` baseline is mandatory.** This is the lesson from §9.2, where a control that simply
copied statutes out of the input beat every learned model. The same trap exists here: a precedent
discussed in the parties' arguments is named in `masked_text`. `mention` ranks exactly the pool
cases whose party names appear in the query, so the margin over it is the part that is actually
retrieved rather than read off.

Time-respecting retrieval is enforced by `before_year` and **asserted** after the fact, per §9.3.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict

import numpy as np

from src import paths
from src.authorities.retrieval import BM25Index, scrub_citations, tokenize
from src.llm import client

SEED = 573
MIN_TARGETS = 3
STOP_PARTY = {"state", "of", "the", "and", "ors", "anr", "others", "another", "union", "india",
              "govt", "government", "vs", "v", "etc", "ltd", "co", "shri", "sri", "smt", "m",
              "s", "in", "re", "all", "ex", "parte"}


def party_tokens(title: str) -> set[str]:
    """Distinctive surname tokens from a case title. Institutional words are dropped because
    `State of A.P.` and `Union of India` appear in hundreds of unrelated cases."""
    t = re.sub(r"\s+on\s+\d{1,2}\s+\w+\s+\d{4}\s*$", "", title or "", flags=re.I)
    return {w for w in re.findall(r"[A-Za-z]{4,}", t.lower()) if w not in STOP_PARTY}


# --- metrics ------------------------------------------------------------------------------
def recall_at_k(ranked, targets, k):
    return len(set(ranked[:k]) & targets) / len(targets) if targets else float("nan")


def mrr(ranked, targets):
    for i, d in enumerate(ranked, 1):
        if d in targets:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, targets, k):
    dcg = sum(1.0 / math.log2(i + 1) for i, d in enumerate(ranked[:k], 1) if d in targets)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(targets), k) + 1))
    return dcg / ideal if ideal else float("nan")


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion (§9.3's hybrid). Rank-based, so it needs no score calibration
    between retrievers that produce incomparable scales."""
    s: dict[str, float] = defaultdict(float)
    for r in rankings:
        for i, d in enumerate(r, 1):
            s[d] += 1.0 / (k + i)
    return [d for d, _ in sorted(s.items(), key=lambda kv: -kv[1])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--n-queries", type=int, default=400)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--text-dense", action="store_true",
                    help="also embed masked text directly (costs API calls); separates 'dense "
                         "fails here' from 'fact embeddings carry no precedent signal'")
    args = ap.parse_args()

    # --- pool and targets
    reg = {r["doc_id"]: r for r in (json.loads(l) for l in open(paths.CASE_REGISTRY))}
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    pool_ids = [d for d in masked if masked[d] and reg.get(d, {}).get("year")]
    years = {d: reg[d]["year"] for d in pool_ids}
    titles = {d: reg[d]["title"] for d in pool_ids}
    pool = set(pool_ids)
    print(f"pool: {len(pool):,} cases with masked text and a year")

    cited: dict[str, set[str]] = defaultdict(set)
    all_case_edges: dict[str, int] = Counter()
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        if r.get("edge_type") != "case":
            continue
        a, b = r["citing_doc_id"], r.get("cited_doc_id")
        all_case_edges[a] += 1
        # time-respecting by construction: a precedent must predate the citing case
        if b in pool and a in pool and years[b] < years[a]:
            cited[a].add(b)

    queries = [d for d in pool if len(cited.get(d, ())) >= MIN_TARGETS]
    queries.sort()
    rng = np.random.default_rng(SEED)
    if args.n_queries and len(queries) > args.n_queries:
        queries = [queries[i] for i in sorted(rng.choice(len(queries), args.n_queries, replace=False))]
    print(f"queries with >={MIN_TARGETS} reachable precedents: {len(queries):,}")
    reach = sum(len(cited[d]) for d in queries)
    raw = sum(all_case_edges[d] for d in queries)
    print(f"  targets: {reach:,} reachable of {raw:,} case citations "
          f"({100*reach/max(1,raw):.1f}% reachable -- recall below is against the reachable set)")

    # --- build BM25 over the pool, on scrubbed masked text
    print("\nbuilding BM25 over scrubbed masked text...", flush=True)
    scrubbed = {d: scrub_citations(masked[d]) for d in pool_ids}
    idx = BM25Index()
    idx.build(lambda: ((d, years[d], scrubbed[d]) for d in pool_ids), verbose=False)

    # --- dense arms, both from cached fact embeddings (no API calls)
    #
    # Two aggregations, because the first one measured badly and the reason was diagnosable rather
    # than fundamental:
    #
    #   MEAN-POOL  one vector per case, the mean of its ~17 fact embeddings. Scored R@10 0.053
    #              against BM25's 0.207. Averaging 17 facts washes out exactly the specifics that
    #              identify a precedent -- a case about a 1923 partition and one about a mortgage
    #              redemption both average toward "generic land dispute".
    #   MAX-POOL   keep every fact vector and score a candidate case by its single best
    #              fact-to-fact match against the query's facts. One sharply matching fact is
    #              what makes a precedent relevant, and max-pooling is what preserves it.
    #
    # Both are reported so the fix is demonstrated rather than asserted.
    print("building fact-level dense index from cached embeddings...", flush=True)
    cache = client._get_cache()
    emodel = client.DEFAULT_EMBED_MODEL
    fvecs: list[np.ndarray] = []
    fcase: list[str] = []
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        if r["case_id"] not in pool:
            continue
        hit = cache.get(client._key("embed", emodel, r["text"], {}))
        if hit is None:
            continue
        fvecs.append(np.array(json.loads(hit["response"]), dtype=np.float32))
        fcase.append(r["case_id"])
    F = np.vstack(fvecs)
    F /= (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    del fvecs
    dense_ids = sorted(set(fcase))
    cpos = {d: i for i, d in enumerate(dense_ids)}
    fcase_idx = np.fromiter((cpos[c] for c in fcase), dtype=np.int32, count=len(fcase))
    fyear = np.fromiter((years[c] for c in fcase), dtype=np.int32, count=len(fcase))
    # rows of F belonging to each case, for building the query side
    rows_of: dict[str, np.ndarray] = {}
    for i, c in enumerate(fcase):
        rows_of.setdefault(c, []).append(i)
    rows_of = {c: np.array(v, dtype=np.int32) for c, v in rows_of.items()}
    # mean-pooled case matrix, for the comparison arm
    D = np.zeros((len(dense_ids), F.shape[1]), dtype=np.float32)
    for c, rr in rows_of.items():
        D[cpos[c]] = F[rr].mean(axis=0)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    dyear = np.array([years[d] for d in dense_ids])
    dpos = cpos
    print(f"  {F.shape[0]:,} fact vectors over {len(dense_ids):,} cases "
          f"({F.nbytes/1e9:.2f} GB)")

    # --- atom sets for the overlap arm
    atoms: dict[str, set[str]] = defaultdict(set)
    for line in open(paths.INTERIM / "canonical_facts.jsonl"):
        r = json.loads(line)
        atoms[r["case_id"]].add(r["label"])
    adf = Counter(a for d in pool_ids for a in atoms.get(d, ()))
    N = len(pool_ids)
    aidf = {a: math.log(1 + N / c) for a, c in adf.items()}

    # --- popularity prior: how often a pool case is cited at all
    pop = Counter()
    for a, bs in cited.items():
        for b in bs:
            pop[b] += 1
    pop_order = [d for d, _ in pop.most_common()]

    # --- dense over the MASKED TEXT itself, to separate two very different diagnoses.
    #
    # BM25 on masked text works (R@10 0.207) while dense over fact embeddings does not, under either
    # pooling. That is either (a) dense retrieval failing on this task, or (b) the extracted FACTS
    # not carrying precedent-relevance signal even though the text does. Embedding the masked text
    # directly discriminates: if it works, the facts are the problem; if it fails too, dense is.
    text_dense = None
    if args.text_dense:
        print("embedding masked text for the text-dense arm...", flush=True)
        ids = [d for d in pool_ids]
        T = client.embed([scrubbed[d][:8000] for d in ids], batch_size=32, workers=24)
        ok = T.any(axis=1)
        if not ok.all():
            print(f"  {int((~ok).sum())} text embeddings never arrived; those cases are excluded")
        tids = [d for d, k in zip(ids, ok) if k]
        T = T[ok]
        T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
        text_dense = (tids, {d: i for i, d in enumerate(tids)}, T,
                      np.array([years[d] for d in tids]))

    systems = ["mention", "popularity", "bm25", "dense_meanpool", "dense_maxpool",
               "dense_top3", "atom_overlap", "rrf(bm25+dense_meanpool)"]
    if text_dense:
        systems += ["dense_masked_text", "rrf(bm25+dense_masked_text)"]
    scores = {s: defaultdict(list) for s in systems}
    violations = 0

    for qi, q in enumerate(queries, 1):
        tgt = cited[q]
        yq = years[q]
        qtext = scrubbed[q]

        # mention: pool cases whose party names appear in the query text
        qtok = set(tokenize(masked[q]))
        men = []
        for d in pool_ids:
            if d == q or years[d] >= yq:
                continue
            pt = party_tokens(titles[d])
            if pt and len(pt & qtok) >= max(2, len(pt) // 2):
                men.append(d)
        ranked = {"mention": men}

        ranked["popularity"] = [d for d in pop_order if d != q and years[d] < yq][:args.k]
        ranked["bm25"] = [d for d, _ in idx.search(qtext, args.k, before_year=yq - 1,
                                                   exclude={q})]

        if q in dpos:
            sim = D @ D[dpos[q]]
            sim = np.where(dyear < yq, sim, -np.inf)
            top = np.argsort(-sim)[:args.k + 1]
            ranked["dense_meanpool"] = [dense_ids[i] for i in top
                                        if dense_ids[i] != q and np.isfinite(sim[i])][:args.k]
        else:
            ranked["dense_meanpool"] = []

        qrows = rows_of.get(q)
        if qrows is not None and len(qrows):
            # (n_query_facts x n_pool_facts) similarities, reduced to one score per candidate case
            # by taking the single best fact-to-fact match.
            S = F[qrows] @ F.T                                  # (nq, nF)
            best_per_fact = S.max(axis=0)                       # (nF,)
            best_per_fact[fyear >= yq] = -np.inf                # time-respecting
            case_best = np.full(len(dense_ids), -np.inf, dtype=np.float32)
            np.maximum.at(case_best, fcase_idx, best_per_fact)
            case_best[cpos[q]] = -np.inf
            top = np.argsort(-case_best)[:args.k]
            ranked["dense_maxpool"] = [dense_ids[i] for i in top if np.isfinite(case_best[i])]

            # top-3 mean: a middle ground. Max rewards a single boilerplate match ("the appellant
            # filed an appeal"); the full mean washes out the distinctive fact. Averaging the best
            # few needs several matching facts without requiring all of them to match.
            finite = np.isfinite(best_per_fact)
            top3 = defaultdict(list)
            for fi in np.nonzero(finite)[0]:
                top3[fcase_idx[fi]].append(best_per_fact[fi])
            sc3 = [(dense_ids[ci], float(np.mean(sorted(v, reverse=True)[:3])))
                   for ci, v in top3.items() if dense_ids[ci] != q]
            sc3.sort(key=lambda kv: -kv[1])
            ranked["dense_top3"] = [d for d, _ in sc3[:args.k]]
            del S
        else:
            ranked["dense_maxpool"] = []
            ranked["dense_top3"] = []

        if text_dense:
            tids, tpos, T, tyear = text_dense
            if q in tpos:
                sim = T @ T[tpos[q]]
                sim = np.where(tyear < yq, sim, -np.inf)
                top = np.argsort(-sim)[:args.k + 1]
                ranked["dense_masked_text"] = [tids[i] for i in top
                                               if tids[i] != q and np.isfinite(sim[i])][:args.k]
            else:
                ranked["dense_masked_text"] = []
            ranked["rrf(bm25+dense_masked_text)"] = rrf(
                [ranked["bm25"], ranked["dense_masked_text"]])[:args.k]

        qa = atoms.get(q, set())
        if qa:
            qw = sum(aidf.get(a, 0.0) for a in qa)
            sc = []
            for d in pool_ids:
                if d == q or years[d] >= yq:
                    continue
                da = atoms.get(d)
                if not da:
                    continue
                inter = sum(aidf.get(a, 0.0) for a in (qa & da))
                if inter <= 0:
                    continue
                union = qw + sum(aidf.get(a, 0.0) for a in da) - inter
                sc.append((d, inter / union if union else 0.0))
            sc.sort(key=lambda kv: -kv[1])
            ranked["atom_overlap"] = [d for d, _ in sc[:args.k]]
        else:
            ranked["atom_overlap"] = []

        ranked["rrf(bm25+dense_meanpool)"] = rrf([ranked["bm25"],
                                                  ranked["dense_meanpool"]])[:args.k]

        for s, r in ranked.items():
            # §9.3: time-respecting retrieval, asserted rather than assumed
            violations += sum(1 for d in r if years[d] >= yq)
            scores[s]["r@10"].append(recall_at_k(r, tgt, 10))
            scores[s]["r@50"].append(recall_at_k(r, tgt, 50))
            scores[s]["mrr"].append(mrr(r, tgt))
            scores[s]["ndcg@10"].append(ndcg_at_k(r, tgt, 10))
        if qi % 100 == 0:
            print(f"  {qi}/{len(queries)} queries", flush=True)

    assert violations == 0, f"TIME LEAK: {violations} retrieved cases do not predate their query"
    print(f"\ntime-respecting assertion passed: 0 of the retrieved cases post-date their query")

    hdr = f"{'system':20s} {'R@10':>7s} {'R@50':>7s} {'MRR':>7s} {'nDCG@10':>8s}"
    print("\n" + hdr); print("-" * len(hdr))
    rows = []
    for s in systems:
        m = {k: float(np.nanmean(v)) for k, v in scores[s].items()}
        rows.append({"system": s, **{k: round(v, 4) for k, v in m.items()}})
        print(f"{s:20s} {m['r@10']:7.3f} {m['r@50']:7.3f} {m['mrr']:7.3f} {m['ndcg@10']:8.3f}")

    out = paths.EXPERIMENTS / "paper2" / f"precedent_retrieval_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_queries": len(queries),
        "targets_reachable": reach, "targets_raw": raw,
        "reachable_frac": round(reach / max(1, raw), 4),
        "note": "recall is against the REACHABLE target set; see targets_reachable/targets_raw",
        "results": rows}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
