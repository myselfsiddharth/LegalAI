"""The element layer on a client's story instead of a finished judgment.

S4 was only ever run on facts extracted from judgments, and half its NOT_SATISFIED verdicts quoted a
court's finding (s4_verdicts_by_fact_type.json). A client's narrative contains no findings by
construction, so this measures the element layer on the input it will actually see.

Facts: each narrative sentence, asserted_by=client, status=contested -- the client's account is
exactly that, an account. Claims: the same `select_claims` rule as S4 (top 2 by max fact-embedding
cosine). Model, prompt (element_satisfy.v2) and quote gate unchanged.

No gold exists for narratives, so three measurements:
  mix           SATISFIED / NOT_SATISFIED / UNCLEAR, against S4 on the same cases' judgment facts
  fabrication   quotes not found in the narrative
  agreement     per (case, element) asked in both runs, how often the narrative verdict matches the
                judgment-facts verdict -- the client's story should support what the record does
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src import paths
from src.llm import client
from v2 import element_layer as S4
from v2.element_catalogue import load_claims, select_claims

NAR = paths.ROOT / "product" / "data" / "client_narratives.jsonl"
OUT = paths.ROOT / "product" / "data" / "elements_narrative.jsonl"
EXP = paths.ROOT / "product" / "experiments" / "elements_narrative.json"
BASE = paths.ROOT / "v2" / "interim" / "element_verdicts_element_satisfy_v2_facts.jsonl"
MODEL = "qwen3-235b-a22b-instruct-2507"


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", " ".join(text.split()))
    return [p.strip() for p in parts if len(p.strip()) > 20]


def main() -> None:
    S4.PROMPT_ID = "element_satisfy.v2"
    claims = load_claims()
    rows = [r for r in map(json.loads, open(NAR)) if r["status"] == "ok"]
    jobs = []
    for r in rows:
        sents = sentences(r["narrative"])
        facts = [{"text": s, "asserted_by": "client", "disputed_status": "contested"} for s in sents]
        V = np.asarray(client.embed(sents, verbose=False), dtype=np.float32)
        for ci in select_claims(claims, V, 2):
            jobs.append((r["case_id"], facts, claims[ci]))
    print(f"{len(rows)} narratives -> {len(jobs)} (case, claim) calls", flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        out = list(pool.map(lambda j: S4.judge(j[0], j[1], j[2], MODEL), jobs))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in out))

    def mix(rs):
        c = Counter(v["verdict"] for r in rs for v in r["verdicts"])
        refused = sum(r.get("refused_quote_absent", 0) + r.get("refused_quote_short", 0) for r in rs)
        tot = sum(c.values())
        return {k: c[k] for k in ("SATISFIED", "NOT_SATISFIED", "UNCLEAR")} | {
            "refused": refused, "fabrication": round(refused / max(1, tot + refused), 4)}

    cases = {r["case_id"] for r in rows}
    base = [r for r in map(json.loads, open(BASE)) if r["case_id"] in cases]
    bv = {(r["case_id"], v["element_id"]): v["verdict"] for r in base for v in r["verdicts"]}
    nv = {(r["case_id"], v["element_id"]): v["verdict"] for r in out for v in r["verdicts"]}
    both = sorted(set(bv) & set(nv))
    M = Counter((bv[k], nv[k]) for k in both)
    agree = sum(M[(x, x)] for x in ("SATISFIED", "NOT_SATISFIED", "UNCLEAR"))
    res = {"n_narratives": len(rows), "narrative": mix(out), "judgment_facts_same_cases": mix(base),
           "pairs_asked_in_both": len(both), "agreement": round(agree / max(1, len(both)), 4),
           "confusion_rows_judgment_cols_narrative": {f"{a}|{b}": n for (a, b), n in M.items()},
           "claims_overlap": round(len({(r['case_id'], r['claim_id']) for r in out}
                                       & {(r['case_id'], r['claim_id']) for r in base})
                                   / max(1, len(out)), 4)}
    EXP.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
