"""Good-law signal: how did later courts treat each precedent? Rebuilds the signed citation graph.

The original signed graph (HEADNOTE regex pre-2000 + an LLM classifier post-2000) was deleted with the
legacy pipeline on 2026-10-03; only the raw edges in citations_classified.jsonl survive. Each case
edge carries `cited_text`, the citation exactly as written in the citing judgment, so every mention
can be located directly and classified from its surrounding passage.

Scope: every case-to-case edge whose CITED case is in the land-dispute pool (the cases the product
retrieves). Citing text comes from case_text.jsonl when the citing case is in the pool, otherwise
from corpus_wide's order-removed text (treatment language lives in the reasoning, which that text
keeps).

Gates, each counted separately:
  not_located   the anchor text could not be found in the citing judgment
  quote_absent  a non-REFERRED label whose quote is not in the passage -> downgraded to REFERRED
  no_answer / unparsed / skipped   call-level failures, never read as a treatment
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from src import paths
from src.llm import client

PROMPT_ID = "treatment.v2"  # v1 inverted direction on ~half of OVERRULED (spot check 15/30)
OUT = paths.ROOT / "product" / "data" / "treatments.jsonl"
BEFORE, AFTER = 700, 900
BATCH = 8


class L(BaseModel):
    i: int
    treatment: Literal["FOLLOWED", "DISTINGUISHED", "OVERRULED", "DOUBTED", "REFERRED"]
    quote: str = ""


class Reply(BaseModel):
    labels: list[L]


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def locate(text: str, anchor: str) -> str | None:
    """Passage around the first mention of `anchor`, matched whitespace- and case-insensitively."""
    a = norm(anchor)
    a = re.sub(r"^[\d\W]+", "", a)[:45]
    if len(a) < 8:
        return None
    flat = re.sub(r"\s+", " ", text)
    i = flat.lower().find(a)
    if i < 0:
        return None
    return flat[max(0, i - BEFORE): i + len(a) + AFTER]


def classify(batch: list[dict], model: str) -> list[dict]:
    body = "\n\n".join(f"{n}. CITED: {e['cited_text'][:120]}\nPASSAGE: {e['passage']}"
                       for n, e in enumerate(batch, 1))
    res = client.complete([{"role": "system", "content": client.load_prompt(PROMPT_ID)},
                           {"role": "user", "content": body}],
                          model=model, json_schema=Reply, prompt_id=PROMPT_ID,
                          max_tokens=120 + 90 * len(batch))
    why = "ok" if res.ok and res.parsed is not None else ("no_answer" if not res.ok else "unparsed")
    got = {l.i: l for l in res.parsed.labels} if why == "ok" else {}
    out = []
    for n, e in enumerate(batch, 1):
        base = {k: e[k] for k in ("citing", "cited", "citing_year")}
        l = got.get(n)
        if l is None:
            out.append({**base, "treatment": None, "status": why if why != "ok" else "skipped"})
            continue
        t, q, st = l.treatment, l.quote.strip(), "ok"
        if t != "REFERRED" and (len(q) < 15 or norm(q) not in norm(e["passage"])):
            t, st = "REFERRED", "quote_absent"
        out.append({**base, "treatment": t, "quote": q, "status": st})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    full = {}
    for line in open(paths.INTERIM / "case_text.jsonl"):
        r = json.loads(line)
        full[r["doc_id"]] = r["text"]
    pool = set(full)
    edges = []
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        if r.get("edge_type") == "case" and r.get("cited_doc_id") in pool:
            edges.append({"citing": r["citing_doc_id"], "cited": r["cited_doc_id"],
                          "citing_year": int(r.get("citing_year") or 0),
                          "cited_text": r.get("cited_text") or ""})
    need = {e["citing"] for e in edges} - pool
    other = {}
    for line in open(paths.INTERIM / "corpus_wide.jsonl"):
        r = json.loads(line)
        if r["doc_id"] in need:
            other[r["doc_id"]] = r.get("ildc_style_text") or ""
    jobs, st = [], Counter()
    for e in edges:
        text = full.get(e["citing"]) or other.get(e["citing"]) or ""
        p = locate(text, e["cited_text"]) if text else None
        if p is None:
            st["not_located" if text else "no_text"] += 1
            continue
        jobs.append({**e, "passage": p})
    print(f"edges citing a pool case: {len(edges):,}; located {len(jobs):,}; {dict(st)}", flush=True)

    batches = [jobs[i:i + BATCH] for i in range(0, len(jobs), BATCH)]
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool_:
        for n, out in enumerate(pool_.map(lambda b: classify(b, args.model), batches), 1):
            rows += out
            if n % 200 == 0:
                print(f"  {n:,}/{len(batches):,}", flush=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"-> {OUT}")
    print("  treatments: " + json.dumps(Counter(r["treatment"] for r in rows)))
    print("  status:     " + json.dumps(Counter(r["status"] for r in rows)))


if __name__ == "__main__":
    main()
