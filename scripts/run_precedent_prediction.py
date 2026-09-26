"""Precedent prediction: given the facts of a dispute, retrieve the authorities that apply.

This is the inverted pipeline. Instead of extracting citations from a judgment that already
contains them, a judgment's own citations are held out and used as labels, and only its
scrubbed text is given to the retriever. Labels are free -- the citation metadata already
records what each case actually cited.

Two controls, because "our retriever found something" means nothing on its own:
  - popularity: always return the most-cited cases in the corpus. Citation graphs are
    extremely skewed, so this is a deceptively strong baseline and any real retriever must
    beat it.
  - the reachability ceiling: only ~26% of citation edges point at a case that is in this
    corpus at all (it is Supreme Court only, and judgments cite High Courts, the Privy
    Council and English cases). Recall is reported against the REACHABLE authorities, with
    the raw figure alongside so the ceiling stays visible.

Usage:
  python3 scripts/run_precedent_prediction.py --n 100 --min-year 2000
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from ode_lib import ROOT
from retrieval_lib import BM25Index, DenseIndex, load_corpus, scrub_citations

CITATIONS = ROOT / "Data" / "processed" / "citations_classified.jsonl"
MAP_PATH = ROOT / "Data" / "processed" / "doc_id_to_pdf.json"
OUT_PATH = ROOT / "Data" / "processed" / "precedent_prediction.json"


def load_edges():
    """citing_doc_id -> set of cited case doc_ids, plus year lookup."""
    cites = defaultdict(set)
    years = {}
    with CITATIONS.open() as f:
        for line in f:
            e = json.loads(line)
            y = e.get("citing_year")
            years[e["citing_doc_id"]] = int(y) if str(y).isdigit() else 0
            if e["edge_type"] == "case":
                cites[e["citing_doc_id"]].add(e["cited_doc_id"])
    return cites, years


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="query cases to evaluate")
    ap.add_argument("--min-year", type=int, default=2000,
                    help="only query cases from this year on -- pre-2000 headnotes list the "
                         "authorities up front and leak the answer")
    ap.add_argument("--min-authorities", type=int, default=3,
                    help="a query case must cite this many REACHABLE authorities")
    ap.add_argument("--ks", type=int, nargs="+", default=[10, 50, 100])
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--embed-model", default="qwen3-embedding-8b")
    ap.add_argument("--facts-chars", type=int, nargs="+", default=[0],
                    help="query-length arms, in characters of scrubbed text from the start of "
                         "the judgment; 0 means the whole scrubbed text. Several values are "
                         "evaluated against one index build.")
    args = ap.parse_args()

    cites, years = load_edges()
    haystack = set(json.loads(MAP_PATH.read_text()))
    print(f"haystack: {len(haystack):,} mappable cases")

    corpus = load_corpus()
    queries = [d for d in haystack
               if years.get(d, 0) >= args.min_year
               and len(cites.get(d, set()) & haystack) >= args.min_authorities]
    queries.sort(key=lambda d: (years[d], d))
    queries = queries[: args.n]
    print(f"query cases: {len(queries)} (year >= {args.min_year}, "
          f">= {args.min_authorities} reachable authorities)")

    wanted = set(queries)
    texts = {d: t for d, _y, t in corpus() if d in wanted}


    if args.retriever == "dense":
        from ode_lib import get_client
        index = DenseIndex(get_client(), args.embed_model, years)
        covered = set(index.doc_ids)
        missing = len(haystack - covered)
        print(f"dense index: {len(index):,} vectors; {missing:,} haystack cases not yet "
              f"embedded and therefore unreachable")
        haystack = haystack & covered
        queries = [q for q in queries if q in covered]
        print(f"queries retained: {len(queries)}")
    else:
        print("building BM25 index...", flush=True)
        index = BM25Index()
        index.build(corpus)

    # popularity control: in-degree over the haystack
    indeg = Counter()
    for citing, targets in cites.items():
        for t in targets:
            if t in haystack:
                indeg[t] += 1
    popular = [d for d, _ in indeg.most_common()]

    maxk = max(args.ks)
    reachable_total = raw_total = 0
    pop_hits = {k: 0 for k in args.ks}
    for qid in queries:
        truth = cites[qid] & haystack
        reachable_total += len(truth)
        raw_total += len(cites[qid])
        pop = [d for d in popular if d != qid and years.get(d, 0) <= years[qid]][:maxk]
        for k in args.ks:
            pop_hits[k] += len(truth & set(pop[:k]))

    # The whole scrubbed judgment still contains the court's reasoning, which nobody has
    # before the case is decided. Shorter arms approximate what a litigant actually holds:
    # the opening of a judgment is its statement of facts and procedural history.
    arms = {}
    for chars in args.facts_chars:
        label = "full" if not chars else f"{chars}c"
        hits = {k: 0 for k in args.ks}
        per_case = []
        for n, qid in enumerate(queries, 1):
            truth = cites[qid] & haystack
            q = scrub_citations(texts[qid])
            if chars:
                q = q[:chars]
            ranked = [d for d, _ in index.search(q, maxk, before_year=years[qid],
                                                 exclude={qid})]
            rec = {}
            for k in args.ks:
                h = len(truth & set(ranked[:k]))
                hits[k] += h
                rec[f"hits@{k}"] = h
            per_case.append({"doc_id": qid, "year": years[qid],
                             "reachable_authorities": len(truth),
                             "all_authorities": len(cites[qid]), **rec})
        arms[label] = {
            "query_chars": chars or None,
            "recall": {f"@{k}": round(hits[k] / max(reachable_total, 1), 4) for k in args.ks},
            "precision": {f"@{k}": round(hits[k] / (k * len(queries)), 4) for k in args.ks},
            "hits": {f"@{k}": hits[k] for k in args.ks},
            "per_case": per_case,
        }
        print(f"  arm {label}: recall@{maxk} "
              f"{hits[maxk] / max(reachable_total, 1):.1%}", flush=True)

    print(f"\n=== PRECEDENT PREDICTION ({len(queries)} cases, year >= {args.min_year}) ===")
    print(f"reachable authorities: {reachable_total}   all cited: {raw_total}   "
          f"reachability {reachable_total / max(raw_total, 1):.1%}")
    head = "".join(f"{lbl:>14}" for lbl in arms)
    print(f"\n{'k':>5}{head}{'popularity':>14}")
    for k in args.ks:
        row = "".join(f"{arms[l]['recall'][f'@{k}']:>13.1%} " for l in arms)
        print(f"{k:>5}{row}{pop_hits[k] / max(reachable_total, 1):>13.1%}")
    print("\nprecision@k (share of returned candidates that were really cited)")
    for k in args.ks:
        row = "".join(f"{arms[l]['precision'][f'@{k}']:>13.1%} " for l in arms)
        print(f"{k:>5}{row}")

    OUT_PATH.write_text(json.dumps({
        "queries": len(queries), "min_year": args.min_year, "haystack": len(haystack),
        "reachable_authorities": reachable_total, "all_cited": raw_total,
        "popularity_recall": {f"@{k}": round(pop_hits[k] / max(reachable_total, 1), 4)
                              for k in args.ks},
        "arms": arms,
    }, indent=2))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
