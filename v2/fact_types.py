"""Label every fact as PRIMARY / PROCEDURAL / FINDING / LAW / ASSERTION, reading its source sentence.

Why: S4's defeat verdicts are weak, and the attribution-cue detector (attribution_audit.py) reaches
only 82 of the 1,481 NOT_SATISFIED verdicts it was meant to explain. Hand classification of the
gold set found the larger problem is conclusions with no cue at all ("the respondent terminated the
tenancy by means of a valid notice") -- a court's characterisation dressed as a primary fact. A cue
regex cannot see those; a model reading the source sentence can.

Facts are batched per case (default 15 per call) because neighbouring facts share context and the
corpus is 96,623 facts. Each fact is shown with its source sentence and the extracted span marked.

  --pilot      only the 40 gold cases, so the labels can be checked against the hand labels in
               s4_cited_fact_types.json and the 51 hand-checked detector samples before the full
               run spends ~6,500 calls.

Unlabelled facts (a call that never returned, or an entry the model skipped) are counted
separately from labelled ones -- an outage must never read as a finding (gotcha #14).
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from src.llm import client
from v2 import paths
from v2.attribution_audit import source_prefix

PROMPT_ID = "fact_type.v1"
TYPES = ("PRIMARY", "PROCEDURAL", "FINDING", "LAW", "ASSERTION")
GOLD = paths.ROOT.parent / "Data" / "gold" / "element_gold_tasks.jsonl"


class Label(BaseModel):
    i: int
    type: Literal["PRIMARY", "PROCEDURAL", "FINDING", "LAW", "ASSERTION"]


class Reply(BaseModel):
    labels: list[Label]


def out_path(pilot: bool):
    return paths.INTERIM / f"fact_types{'_pilot' if pilot else ''}.jsonl"


def render(batch: list[dict], masked: str) -> str:
    lines = []
    for n, f in enumerate(batch, 1):
        sp = f["source_span"]
        a, b = json.loads(sp) if isinstance(sp, str) else sp
        src = (source_prefix(masked, a)[-260:] + "⟪" + masked[a:b][:260] + "⟫")
        src = " ".join(src.split())
        lines.append(f"{n}. FACT: {f['text']}\n   SOURCE: ...{src}")
    return "\n".join(lines)


def label_batch(batch: list[dict], masked: str, model: str) -> list[dict]:
    res = client.complete(
        [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
         {"role": "user", "content": render(batch, masked)}],
        model=model, json_schema=Reply, prompt_id=PROMPT_ID, max_tokens=60 + 25 * len(batch))
    got = {}
    if res.ok and res.parsed is not None:
        got = {l.i: l.type for l in res.parsed.labels if 1 <= l.i <= len(batch)}
    why = "ok" if res.ok and res.parsed is not None else ("no_answer" if not res.ok else "unparsed")
    return [{"fact_id": f["fact_id"], "case_id": f["case_id"],
             "type": got.get(n), "status": "ok" if n in got else why if why != "ok" else "skipped"}
            for n, f in enumerate(batch, 1)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    args = ap.parse_args()

    masked = {}
    for line in open(paths.MASKED):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]
    facts = [json.loads(l) for l in open(paths.FACTS)]
    if args.pilot:
        keep = {json.loads(l)["case_id"] for l in open(GOLD)}
        facts = [f for f in facts if f["case_id"] in keep]

    out = out_path(args.pilot)
    done = set()
    if out.exists():
        for line in open(out):
            r = json.loads(line)
            if r["status"] == "ok":
                done.add(r["fact_id"])
    by_case = collections.defaultdict(list)
    for f in facts:
        if f["fact_id"] not in done and f["case_id"] in masked:
            by_case[f["case_id"]].append(f)
    jobs = [(c, fs[i:i + args.batch]) for c, fs in by_case.items()
            for i in range(0, len(fs), args.batch)]
    print(f"facts {len(facts):,}; already labelled {len(done):,}; {len(jobs):,} calls "
          f"(batch {args.batch}, model {args.model})", flush=True)

    t0, status, types = time.time(), collections.Counter(), collections.Counter()
    with out.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(label_batch, b, masked[c], args.model) for c, b in jobs]
        for n, fu in enumerate(futs, 1):
            for r in fu.result():
                fh.write(json.dumps(r) + "\n")
                status[r["status"]] += 1
                if r["type"]:
                    types[r["type"]] += 1
            if n % 200 == 0:
                fh.flush()
                el = time.time() - t0
                print(f"  {n:,}/{len(futs):,} ({n/el:.1f}/s, eta {(len(futs)-n)/(n/el)/60:.0f} min)",
                      flush=True)
    tot = sum(types.values())
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out}")
    print("  labelled: " + "  ".join(f"{t} {types[t]:,} ({types[t]/max(1,tot):.1%})" for t in TYPES))
    print(f"  NOT labelled (outage/skip, not an opinion): "
          + ", ".join(f"{k} {v:,}" for k, v in status.items() if k != "ok"))


if __name__ == "__main__":
    main()
