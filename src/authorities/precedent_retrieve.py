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

    # --- case vectors for the dense arm: mean-pooled fact embeddings, from cache (no API)
    print("building case vectors from cached fact embeddings...", flush=True)
    cache = client._get_cache()
    emodel = client.DEFAULT_EMBED_MODEL
    sums: dict[str, np.ndarray] = {}
    cnt: Counter = Counter()
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        hit = cache.get(client._key("embed", emodel, r["text"], {}))
        if hit is None:
            continue
        v = np.array(json.loads(hit["response"]), dtype=np.float32)
        c = r["case_id"]
        sums[c] = v if c not in sums else sums[c] + v
        cnt[c] += 1
    dense_ids = [d for d in pool_ids if cnt[d]]
    D = np.vstack([sums[d] / cnt[d] for d in dense_ids])
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    dpos = {d: i for i, d in enumerate(dense_ids)}
    dyear = np.array([years[d] for d in dense_ids])
    print(f"  {len(dense_ids):,} cases have a dense vector")

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

    systems = ["mention", "popularity", "bm25", "dense", "atom_overlap", "rrf(bm25+dense)"]
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
            ok = (dyear < yq)
            sim = np.where(ok, sim, -np.inf)
            top = np.argsort(-sim)[:args.k]
            ranked["dense"] = [dense_ids[i] for i in top
                               if dense_ids[i] != q and np.isfinite(sim[i])]
        else:
            ranked["dense"] = []

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

        ranked["rrf(bm25+dense)"] = rrf([ranked["bm25"], ranked["dense"]])[:args.k]

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
