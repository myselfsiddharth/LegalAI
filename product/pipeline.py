"""LexGraph for lawyers: one call from a client's story to a certified research brief.

    lg = LexGraph()                      # loads indexes once (cached under product/index/)
    out = lg.analyse(narrative)          # memo, statutes, precedents, good law, elements, brief

Every step is the one measured in product/experiments/ on 93 held-out client narratives:

  1. issue memo          issue_memo.v1 (no case names)
  2. precedents          RRF of full-text BM25 and dense over facts. Card retrievers were tested and
                         dropped: they lift first-stage R@10 (0.173 -> 0.191) but defeat the reranker,
                         which judges by the same cards (final 0.192 vs 0.215) -- correlated errors.
  3. rerank              rerank.v1 over the top 50, ten cards per call
  4. good law            treatments by later courts (direction-checked), as of the query date
  5. statutes            kNN vote over the 25 most similar decided cases
  6. elements            S4 (element_satisfy.v2) on the narrative's sentences
  7. brief               brief.v1, citing only P1..P8 and the predicted provisions
  8. certification       every precedent id, quoted fact and provision checked mechanically

`as_of_year` restricts every authority to cases decided before that year (evaluation); `exclude`
removes specific judgments (the held-out case itself). Live use leaves both unset.
"""
from __future__ import annotations

import json
import pickle
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src import paths
from src.authorities.precedent_retrieve import rrf
from src.authorities.retrieval import BM25Index, scrub_citations
from src.llm import client
from product.brief import Brief, norm
from product.elements_narrative import sentences
from product.issue_memo import memo_text, one as issue_memo
from product.rerank import score_batch

P = paths.ROOT / "product"
IDX = P / "index"
MODEL = "qwen3-235b-a22b-instruct-2507"
TOP_RERANK, TOP_BRIEF, TOP_STAT, K_NN = 50, 8, 10, 25


class LexGraph:
    def __init__(self, verbose: bool = True):
        t0 = time.time()
        IDX.mkdir(parents=True, exist_ok=True)
        reg = {r["doc_id"]: r for r in map(json.loads, open(paths.CASE_REGISTRY))}
        masked = {r["doc_id"]: r["masked_text"] for r in map(json.loads, open(paths.INTERIM / "masked_text.jsonl"))}
        self.years = {d: reg[d]["year"] for d in masked if masked[d] and reg.get(d, {}).get("year")}
        self.titles = {d: reg[d]["title"] for d in self.years}
        self.urls = {d: reg[d].get("url") for d in self.years}
        self.pool = sorted(self.years)
        self.pos = {d: i for i, d in enumerate(self.pool)}
        self.pyear = np.array([self.years[d] for d in self.pool])
        self.cards = {r["doc_id"]: r["card"] for r in map(json.loads, open(P / "data" / "case_cards.jsonl"))
                      if r["status"] == "ok"}

        bm = IDX / "bm25_full.pkl"
        if bm.exists():
            self.bm_full = pickle.loads(bm.read_bytes())
        else:
            def full():
                for line in open(paths.INTERIM / "case_text.jsonl"):
                    r = json.loads(line)
                    if r["doc_id"] in self.years:
                        yield r["doc_id"], self.years[r["doc_id"]], scrub_citations(r["text"])
            self.bm_full = BM25Index()
            self.bm_full.build(full, verbose=False)
            bm.write_bytes(pickle.dumps(self.bm_full))

        emb = IDX / "dense_facts.npy"
        if emb.exists():
            self.D = np.load(emb)
        else:
            self.D = client.embed([scrub_citations(masked[d])[:8000] for d in self.pool], batch_size=32, workers=24)
            np.save(emb, self.D)
        self.D = self.D / (np.linalg.norm(self.D, axis=1, keepdims=True) + 1e-9)

        self.treat: dict[str, list] = {}
        for r in map(json.loads, open(P / "data" / "treatments.jsonl")):
            if r.get("treatment"):
                self.treat.setdefault(r["cited"], []).append((r["citing_year"], r["treatment"], r.get("quote", "")))
        self.stat_labels = {r["doc_id"]: r["labels_all"] for r in map(json.loads, open(paths.INTERIM / "statute_targets.jsonl"))}
        from v2.element_catalogue import load_claims
        self.claims = load_claims()
        if verbose:
            print(f"LexGraph ready in {time.time()-t0:.0f}s: {len(self.pool):,} judgments, "
                  f"{len(self.cards):,} case cards, {sum(len(v) for v in self.treat.values()):,} treatments")

    # ---------------------------------------------------------------- pieces
    def good_law(self, doc: str, before: int | None) -> dict:
        rows = [(y, t, q) for y, t, q in self.treat.get(doc, []) if before is None or (y and y < before)]
        c = Counter(t for _, t, _ in rows)
        status = ("OVERRULED" if c["OVERRULED"] else "DOUBTED" if c["DOUBTED"]
                  else "GOOD" if rows else "UNKNOWN")
        neg = [q for _, t, q in rows if t in ("OVERRULED", "DOUBTED", "DISTINGUISHED") and q][:3]
        return {"status": status, "followed": c["FOLLOWED"], "distinguished": c["DISTINGUISHED"],
                "doubted": c["DOUBTED"], "overruled": c["OVERRULED"], "negative_quotes": neg}

    def retrieve(self, text: str, qvec: np.ndarray, as_of: int | None, exclude: set) -> list[str]:
        before = None if as_of is None else as_of - 1
        bf = [d for d, _ in self.bm_full.search(text, 200, before_year=before, exclude=exclude)]
        s = self.D @ qvec
        if as_of is not None:
            s[self.pyear >= as_of] = -9
        for d in exclude:
            if d in self.pos:
                s[self.pos[d]] = -9
        dn = [self.pool[i] for i in np.argsort(-s)[:200]]
        return rrf([bf, dn])

    def statutes(self, qvec: np.ndarray, as_of: int | None, exclude: set) -> list[str]:
        s = self.D @ qvec
        if as_of is not None:
            s[self.pyear >= as_of] = -9
        for d in exclude:
            if d in self.pos:
                s[self.pos[d]] = -9
        votes = Counter()
        for i in np.argsort(-s)[:K_NN]:
            for l in self.stat_labels.get(self.pool[i], []):
                votes[l] += float(s[i])
        return [l for l, _ in votes.most_common(TOP_STAT)]

    def elements(self, narrative: str) -> list[dict]:
        from v2 import element_layer as S4
        from v2.element_catalogue import select_claims
        S4.PROMPT_ID = "element_satisfy.v2"
        sents = sentences(narrative)
        if not sents:
            return []
        facts = [{"text": s, "asserted_by": "client", "disputed_status": "contested"} for s in sents]
        V = np.asarray(client.embed(sents, verbose=False), dtype=np.float32)
        out = []
        for ci in select_claims(self.claims, V, 2):
            r = S4.judge("live", facts, self.claims[ci], MODEL)
            el = {e.element_id: e for e in self.claims[ci].elements}
            out.append({"claim": self.claims[ci].name,
                        "elements": [{"element": el[v["element_id"]].name, "verdict": v["verdict"],
                                      "quote": v["quote"]} for v in r["verdicts"] if v["element_id"] in el],
                        "refused": r.get("refused_quote_absent", 0) + r.get("refused_quote_short", 0)})
        return out

    # ---------------------------------------------------------------- the whole thing
    def analyse(self, narrative: str, as_of_year: int | None = None, exclude: tuple = ()) -> dict:
        t = {}
        exclude = set(exclude)
        t0 = time.time()
        memo = issue_memo("live", narrative, MODEL)
        if memo.get("status") != "ok":
            return {"error": f"issue memo failed: {memo.get('status')}"}
        m = memo["memo"]
        query = narrative + "\n\n" + memo_text(m)
        qvec = np.asarray(client.embed([query], verbose=False)[0], dtype=np.float32)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-9)
        t["memo"] = time.time() - t0

        t0 = time.time()
        cands = self.retrieve(query, qvec, as_of_year, exclude)[:TOP_RERANK]
        rq = (f"## CLIENT'S SITUATION\n\n{narrative}\n\n## ISSUE MEMO\n\nIssues: {'; '.join(m['issues'])}"
              f"\nClaims: {'; '.join(m['claims'])}\nDoctrines: {'; '.join(m['doctrines'])}")
        items = [(d, self.titles[d], self.cards.get(d)) for d in cands]
        with ThreadPoolExecutor(max_workers=8) as pool:
            parts = list(pool.map(lambda i: score_batch(rq, items[i:i + 10], MODEL), range(0, len(items), 10)))
        sc = {k: v for p in parts for k, v in p.items()}
        ranked = sorted(cands, key=lambda d: (-sc.get(d, 1), cands.index(d)))
        t["retrieve+rerank"] = time.time() - t0

        prec = [{"id": f"P{i}", "doc_id": d, "title": self.titles[d], "year": self.years[d],
                 "url": self.urls.get(d), "relevance": sc.get(d), "card": self.cards.get(d),
                 "good_law": self.good_law(d, as_of_year)}
                for i, d in enumerate(ranked[:TOP_BRIEF], 1)]
        # Articles 136 and 142 are the Supreme Court's own appeal and remedial jurisdiction: present in
        # most judgments, uninformative about the merits. Hidden from the lawyer, kept in evaluation.
        provisions = [l for l in self.statutes(qvec, as_of_year, exclude)
                      if l not in ("CONST:Art136", "CONST:Art142")]

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:
            fe = pool.submit(self.elements, narrative)
            fb = pool.submit(self._brief, narrative, m, prec, provisions)
            elements, (brief, cert) = fe.result(), fb.result()
        t["elements+brief"] = time.time() - t0
        return {"memo": m, "precedents": prec, "provisions": provisions, "elements": elements,
                "brief": brief, "certification": cert, "timings_s": {k: round(v, 1) for k, v in t.items()},
                "more_precedents": [{"doc_id": d, "title": self.titles[d], "year": self.years[d],
                                     "relevance": sc.get(d)} for d in ranked[TOP_BRIEF:20]]}

    def _brief(self, narrative: str, m: dict, prec: list, provisions: list) -> tuple[dict | None, dict]:
        def status_line(g):
            if g["status"] == "UNKNOWN":
                return "no later treatment recorded"
            return f"{g['status']} (followed {g['followed']}x)" if g["status"] == "GOOD" else g["status"]
        plist = "\n\n".join(
            f"{p['id']}. {p['title']}\n   Treatment by later courts: {status_line(p['good_law'])}\n"
            f"   Decided: {'; '.join(p['card']['issues_decided'])}\n   Holdings: {'; '.join(p['card']['holdings'])}"
            for p in prec if p["card"])
        body = (f"## CLIENT'S STORY\n\n{narrative}\n\n## ISSUE MEMO\n\nIssues: {'; '.join(m['issues'])}\n"
                f"Claims: {'; '.join(m['claims'])}\n\n## PRECEDENTS\n\n{plist}\n\n## PROVISIONS\n\n"
                + ", ".join(provisions))
        res = client.complete([{"role": "system", "content": client.load_prompt("brief.v1")},
                               {"role": "user", "content": body}],
                              model=MODEL, json_schema=Brief, prompt_id="brief.v1", max_tokens=2500)
        if not res.ok or res.parsed is None:
            return None, {"error": "brief failed"}
        b = res.parsed.model_dump()
        ids = {p["id"] for p in prec}
        nn = norm(narrative)
        cert = Counter()
        for iss in b["issues"]:
            iss["precedents_checked"] = [{"ref": x, "certified": x.strip() in ids} for x in iss["precedents"]]
            iss["facts_checked"] = [{"quote": f, "verbatim": len(norm(f)) > 15 and norm(f) in nn} for f in iss["facts"]]
            iss["provisions_checked"] = [{"label": s, "predicted": s.strip() in provisions} for s in iss["provisions"]]
            for x in iss["precedents_checked"]:
                cert["precedent_ok" if x["certified"] else "precedent_refused"] += 1
            for x in iss["facts_checked"]:
                cert["fact_ok" if x["verbatim"] else "fact_refused"] += 1
            for x in iss["provisions_checked"]:
                cert["provision_ok" if x["predicted"] else "provision_unverified"] += 1
        return b, dict(cert)


if __name__ == "__main__":
    import sys
    lg = LexGraph()
    text = open(sys.argv[1]).read() if len(sys.argv) > 1 else sys.stdin.read()
    print(json.dumps(lg.analyse(text), indent=1)[:6000])
