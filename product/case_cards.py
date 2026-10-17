"""Precedent cards: what each candidate case decided, written from its FULL judgment.

Candidates are earlier, already-decided cases, so reading their holdings is legitimate -- the query
case stays facts-only throughout. A card (dispute, issues decided, holdings, provisions, doctrines)
is what the reranker compares against the lawyer's issue memo, and what the product shows a lawyer
next to each retrieved precedent.

Input to the card model: the first 3,000 and last 9,000 characters of the judgment, since Indian
judgments state the dispute up front and the holding near the end.

  --ids-from FILE   card only the candidates listed in a candidates file (default: the evaluation's
                    top-50 lists); omit to card the whole corpus for the product index.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src import paths
from src.llm import client

PROMPT_ID = "case_card.v1"
OUT = paths.ROOT / "product" / "data" / "case_cards.jsonl"
CANDS = paths.ROOT / "product" / "data" / "candidates_top50.json"


class Card(BaseModel):
    dispute: str
    issues_decided: list[str]
    holdings: list[str]
    provisions: list[str]
    doctrines: list[str]


def card_text(c: dict) -> str:
    return " ".join([c.get("dispute", ""), *c.get("issues_decided", []), *c.get("holdings", []),
                     *c.get("provisions", []), *c.get("doctrines", [])])


def one(doc_id: str, text: str, model: str) -> dict:
    excerpt = text[:3000] + "\n\n[...]\n\n" + text[-9000:] if len(text) > 12500 else text
    res = client.complete([{"role": "system", "content": client.load_prompt(PROMPT_ID)},
                           {"role": "user", "content": excerpt}],
                          model=model, json_schema=Card, prompt_id=PROMPT_ID, max_tokens=900)
    if not res.ok:
        return {"doc_id": doc_id, "status": "no_answer"}
    if res.parsed is None:
        return {"doc_id": doc_id, "status": "unparsed"}
    return {"doc_id": doc_id, "status": "ok", "card": res.parsed.model_dump()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="card every case in the corpus")
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    want = None
    if not args.all:
        d = json.loads(CANDS.read_text())
        want = {x for v in d["candidates"].values() for k in v.values() for x in k}
    done = set()
    if OUT.exists():
        done = {r["doc_id"] for r in map(json.loads, open(OUT)) if r["status"] == "ok"}
    jobs = []
    for line in open(paths.INTERIM / "case_text.jsonl"):
        r = json.loads(line)
        if (want is None or r["doc_id"] in want) and r["doc_id"] not in done:
            jobs.append((r["doc_id"], r["text"]))
    print(f"{len(jobs):,} cards to write ({len(done):,} already done)", flush=True)
    from collections import Counter
    st = Counter()
    with OUT.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, r in enumerate(pool.map(lambda j: one(j[0], j[1], args.model), jobs), 1):
            fh.write(json.dumps(r) + "\n")
            st[r["status"]] += 1
            if n % 250 == 0:
                fh.flush()
                print(f"  {n:,}/{len(jobs):,} {dict(st)}", flush=True)
    print(f"-> {OUT}  {dict(st)}")


if __name__ == "__main__":
    main()
