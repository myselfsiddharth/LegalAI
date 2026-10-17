"""How good is the authority layer on a client's story rather than a finished judgment?

Same held-out cases, two inputs:
  narrative   the client_narratives.py retelling: no court history, no statute numbers, no case
              names -- what a lawyer actually receives
  masked      the judgment's own facts-and-pleadings text -- what every earlier experiment used,
              and the reference the narrative is measured against

Precedents: BM25 and dense over the scrubbed masked text of every corpus case, time-respecting (an
authority must predate the query), RRF of the two; scored against the precedents the court actually
cited that exist in the corpus (recall against REACHABLE targets, as everywhere in this project).
Statutes: kNN label voting -- the 25 most similar TRAIN-split cases by dense embedding vote the
provisions their courts relied on (`labels_all`), applied to both inputs, against a global-prior
control. On a narrative `labels_all` is
the honest target: the client never names a section, so nothing can be copied.

The narrative minus masked gap, with a paired bootstrap, is the number this module exists for.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np

from src import paths
from src.authorities.precedent_retrieve import rrf
from src.authorities.retrieval import BM25Index, scrub_citations
from src.authorities import normalize as N
from src.llm import client
from product.issue_memo import OUT as MEMOS, memo_text

NAR = paths.ROOT / "product" / "data" / "client_narratives.jsonl"
OUT = paths.ROOT / "product" / "experiments" / "narrative_vs_masked.json"
SEED = 573
MIN_TRAIN = 10
K_NN = 25


def recall_at(ranked, tgt, k):
    return len(set(ranked[:k]) & tgt) / len(tgt)


def mrr(ranked, tgt):
    for i, d in enumerate(ranked, 1):
        if d in tgt:
            return 1 / i
    return 0.0


def boot(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    rng = np.random.default_rng(SEED)
    d = [(a[i] - b[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def main() -> None:
    nar = {r["case_id"]: r["narrative"] for r in map(json.loads, open(NAR)) if r["status"] == "ok"}
    memo = {}
    if MEMOS.exists():
        memo = {r["case_id"]: r["memo"] for r in map(json.loads, open(MEMOS)) if r["status"] == "ok"}
    # narrative + the lawyer's issue memo, the input the product would actually retrieve with
    nm = {c: nar[c] + "\n\n" + memo_text(memo[c]) for c in nar if c in memo}
    INPUTS = ("narrative", "memo", "masked") if nm else ("narrative", "masked")
    reg = {r["doc_id"]: r for r in map(json.loads, open(paths.CASE_REGISTRY))}
    masked = {r["doc_id"]: r["masked_text"] for r in map(json.loads, open(paths.INTERIM / "masked_text.jsonl"))}
    years = {d: reg[d]["year"] for d in masked if masked[d] and reg.get(d, {}).get("year")}
    pool = sorted(years)
    scrubbed = {d: scrub_citations(masked[d]) for d in pool}
    cited: dict[str, set] = {}
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        a, b = r["citing_doc_id"], r.get("cited_doc_id")
        if r.get("edge_type") == "case" and a in years and b in years and years[b] < years[a]:
            cited.setdefault(a, set()).add(b)
    qs = sorted(c for c in nar if cited.get(c))
    print(f"{len(qs)} narrative cases with reachable precedents; pool {len(pool):,}")

    # ---------------- precedents
    idx = BM25Index()
    idx.build(lambda: ((d, years[d], scrubbed[d]) for d in pool), verbose=False)
    # Candidates are EARLIER, already-decided cases, so a product may read their whole judgment --
    # holdings included. Only the query side must be pre-decision. Index the full text too.
    def full_records():
        for line in open(paths.INTERIM / "case_text.jsonl"):
            r = json.loads(line)
            if r["doc_id"] in years:
                yield r["doc_id"], years[r["doc_id"]], scrub_citations(r["text"])
    idx_full = BM25Index()
    idx_full.build(full_records, verbose=False)
    # Case cards (what each judgment DECIDED) as a third representation, when the corpus is carded.
    cards_path = paths.ROOT / "product" / "data" / "case_cards.jsonl"
    card_txt = {}
    if cards_path.exists():
        from product.case_cards import card_text
        for line in open(cards_path):
            r = json.loads(line)
            if r["status"] == "ok" and r["doc_id"] in years:
                card_txt[r["doc_id"]] = card_text(r["card"])
    use_cards = len(card_txt) >= 0.95 * len(pool)
    print(f"case cards for {len(card_txt):,}/{len(pool):,} pool cases -> card arms "
          f"{'ON' if use_cards else 'OFF'}")
    if use_cards:
        idx_card = BM25Index()
        idx_card.build(lambda: ((d, years[d], card_txt[d]) for d in pool if d in card_txt), verbose=False)
        Cm = client.embed([card_txt.get(d, scrubbed[d][:2000]) for d in pool], batch_size=32, workers=24)
        okc = Cm.any(axis=1)
        Cm = Cm / (np.linalg.norm(Cm, axis=1, keepdims=True) + 1e-9)
    print("embedding pool (cached where available) and queries...", flush=True)
    P = client.embed([scrubbed[d][:8000] for d in pool], batch_size=32, workers=24)
    okp = P.any(axis=1)
    P = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    pyear = np.array([years[d] for d in pool])
    Qn = client.embed([nar[c] for c in qs], batch_size=32, workers=24)
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-9)
    Qm = None
    if nm:
        Qm = client.embed([nm[c] for c in qs], batch_size=32, workers=24)
        Qm = Qm / (np.linalg.norm(Qm, axis=1, keepdims=True) + 1e-9)
    pos = {d: i for i, d in enumerate(pool)}

    def dense(qvec, yq, self_id, k=200):
        s = P @ qvec
        s[(pyear >= yq) | ~okp] = -9
        s[pos[self_id]] = -9
        return [pool[i] for i in np.argsort(-s)[:k]]

    per = {f"{inp}_{sysm}": {"r10": [], "r50": [], "mrr": []}
           for inp in INPUTS for sysm in ("bm25", "dense", "rrf", "bm25full", "rrf_full",
                                          "bm25card", "densecard", "rrf_all")}
    cands: dict[str, dict] = {}
    for qi, c in enumerate(qs):
        yq, tgt = years[c], cited[c]
        arms = [("narrative", nar[c], Qn[qi]), ("masked", scrubbed[c], P[pos[c]])]
        if nm:
            arms.append(("memo", nm[c], Qm[qi]))
        for inp, text, qv in arms:
            b = [d for d, _ in idx.search(text, 200, before_year=yq - 1, exclude={c})]
            dn = dense(qv, yq, c)
            bf = [d for d, _ in idx_full.search(text, 200, before_year=yq - 1, exclude={c})]
            systems = [("bm25", b), ("dense", dn), ("rrf", rrf([b, dn])),
                       ("bm25full", bf), ("rrf_full", rrf([bf, dn]))]
            if use_cards:
                bc = [d for d, _ in idx_card.search(text, 200, before_year=yq - 1, exclude={c})]
                sc_ = Cm @ qv
                sc_[(pyear >= yq) | ~okc] = -9
                sc_[pos[c]] = -9
                dc = [pool[i] for i in np.argsort(-sc_)[:200]]
                systems += [("bm25card", bc), ("densecard", dc), ("rrf_all", rrf([bf, dn, bc, dc]))]
            if inp in ("memo", "narrative"):
                sd = dict(systems)
                cands.setdefault(c, {})[inp] = sd["rrf_full"][:50]
                if "rrf_all" in sd:
                    cands[c][inp + "_all"] = sd["rrf_all"][:50]
            for sysm, ranked in systems:
                m = per[f"{inp}_{sysm}"]
                m["r10"].append(recall_at(ranked, tgt, 10))
                m["r50"].append(recall_at(ranked, tgt, 50))
                m["mrr"].append(mrr(ranked, tgt))
    (paths.ROOT / "product" / "data" / "candidates_top50.json").write_text(json.dumps(
        {"targets": {c: sorted(cited[c]) for c in qs}, "candidates": cands}))
    prec = {k: {mk: round(float(np.mean(v)), 4) for mk, v in m.items()} for k, m in per.items() if m["r10"]}
    print("\n  precedents (recall vs reachable cited precedents):")
    for k, v in prec.items():
        print(f"    {k:18s} R@10 {v['r10']:.3f}  R@50 {v['r50']:.3f}  MRR {v['mrr']:.3f}")
    pgap = {s: {"r10_diff": round(prec[f'narrative_{s}']['r10'] - prec[f'masked_{s}']['r10'], 4),
                "r10_ci95": boot(per[f"narrative_{s}"]["r10"], per[f"masked_{s}"]["r10"])}
            for s in ("bm25", "dense", "rrf")}
    if nm:
        for s in ("bm25", "dense", "rrf"):
            pgap[s]["memo_minus_narrative_r10"] = round(prec[f"memo_{s}"]["r10"] - prec[f"narrative_{s}"]["r10"], 4)
            pgap[s]["memo_minus_narrative_ci95"] = boot(per[f"memo_{s}"]["r10"], per[f"narrative_{s}"]["r10"])

    # ---------------- statutes
    split = json.loads((paths.SPLITS / "forum_heldout.json").read_text())
    tg = {r["doc_id"]: r["labels_all"] for r in map(json.loads, open(paths.INTERIM / "statute_targets.jsonl"))}
    tr = [c for c in split["train"] if tg.get(c) and masked.get(c)]
    df = Counter(l for c in tr for l in tg[c])
    labels = sorted(l for l, n in df.items() if n >= MIN_TRAIN)
    li = {l: i for i, l in enumerate(labels)}
    # kNN label voting over dense embeddings: find the most similar TRAINING cases and vote the
    # provisions their courts relied on, weighted by similarity. A first attempt with 474
    # one-vs-rest logistic regressions ranked WORSE than the global prior (R@10 0.043 vs 0.197 on
    # masked text) -- many tiny per-label models whose probabilities are not comparable -- so it was
    # replaced, and the prior is kept as the control any method must beat.
    trp = np.array([pos[c] for c in tr if c in pos and okp[pos[c]]])
    trc = [pool[i] for i in trp]
    T = P[trp]
    sq = [c for c in nar if set(tg.get(c) or ()) & set(labels)]
    true = [set(tg[c]) & set(labels) for c in sq]
    prior = [l for l, _ in df.most_common() if l in li]
    Qs = client.embed([nar[c] for c in sq], batch_size=32, workers=24)
    Qs = Qs / (np.linalg.norm(Qs, axis=1, keepdims=True) + 1e-9)
    stat, sper, saved_stat = {}, {}, {}
    qarms = [("narrative", Qs), ("masked", np.stack([P[pos[c]] for c in sq]))]
    named = {}
    if nm:
        Qms = client.embed([nm[c] for c in sq], batch_size=32, workers=24)
        Qms = Qms / (np.linalg.norm(Qms, axis=1, keepdims=True) + 1e-9)
        qarms.append(("memo", Qms))
        # provisions the memo NAMES, normalised the same way the court's targets were
        named = {c: list(dict.fromkeys(x.label for x in N.extract(" ; ".join(memo[c]["provisions"]))))
                 for c in sq}
        qarms.append(("memo_named_then_knn", Qms))
    for inp, Q in qarms:
        sims = Q @ T.T
        ranked, p5, r10 = [], [], []
        for row, t in zip(sims, true):
            top = np.argsort(-row)[:K_NN]
            votes = Counter()
            for j in top:
                for l in tg[trc[j]]:
                    if l in li:
                        votes[l] += float(row[j])
            rk = [l for l, _ in votes.most_common()] + [l for l in prior if l not in votes]
            if inp == "memo":
                saved_stat[sq[len(p5)]] = rk[:20]
            if inp == "memo_named_then_knn":
                first = [l for l in named[sq[len(p5)]] if l in li]
                rk = first + [l for l in rk if l not in first]
            p5.append(len(set(rk[:5]) & t) / 5)
            r10.append(len(set(rk[:10]) & t) / len(t))
        stat[inp] = {"p@5": round(float(np.mean(p5)), 4), "r@10": round(float(np.mean(r10)), 4)}
        sper[inp] = r10
    if saved_stat:
        (paths.ROOT / "product" / "data" / "statute_predictions.json").write_text(json.dumps(saved_stat))
    if named:
        # how trustworthy is a memo-named provision on its own? judged against EVERY provision the
        # court relied on, not just the trainable label space
        hit = tot = 0
        for c in sq:
            for l in named[c]:
                tot += 1
                hit += l in set(tg[c])
        stat["memo_named_precision"] = {"correct": hit, "named": tot,
                                        "precision": round(hit / max(1, tot), 4)}
    pr10 = [len(set(prior[:10]) & t) / len(t) for t in true]
    stat["prior_top10"] = {"r@10": round(float(np.mean(pr10)), 4)}
    print(f"\n  statutes ({len(sq)} cases, {len(labels)} provisions with >= {MIN_TRAIN} train cases):")
    for k, v in stat.items():
        print(f"    {k:12s} {v}")

    out = {"n_precedent_queries": len(qs), "n_statute_cases": len(sq), "precedents": prec,
           "precedent_gap_narrative_minus_masked": pgap, "statutes": stat,
           "statute_gap_r10": {"diff": round(stat['narrative']['r@10'] - stat['masked']['r@10'], 4),
                               "ci95": boot(sper["narrative"], sper["masked"])},
           "notes": ["recall against reachable cited precedents only",
                     "narratives generated by qwen3-235b from masked text; leakage-gated"]}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\n  gaps: {json.dumps(pgap)}\n  statutes r@10 gap {out['statute_gap_r10']}\n-> {OUT}")


if __name__ == "__main__":
    main()
