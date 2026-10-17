"""Do S4's NOT_SATISFIED verdicts survive once prior-court findings are labelled or removed?

attribution_audit.py found that ~1.5% of facts are a court's finding with the attribution stripped,
and s4_cited_fact_types.json that those facts back 20 of S4's 43 NOT_SATISFIED verdicts in the gold
set. This re-runs S4 on every (case, claim) pair that produced a NOT_SATISFIED in the full run, plus
every pair in the 40 gold cases, under two arms:

  exclude   the flagged facts are removed; prompt unchanged (element_satisfy.v2)
  tag       the flagged facts stay, marked PRIOR COURT FINDING; element_satisfy.v3 says such a fact
            may not be quoted as the basis for a verdict

`exclude` answers the question directly -- what is left on primary evidence. `tag` tests whether the
cheap fix (label, don't drop) is enough, and counts how often the model quotes a tagged fact anyway.
The detector is ~75% precise, so `exclude` also removes some genuine primary facts; `tag` lets the
model see them, which is one reason to run both.

Same model, same gate, same claim selection as the full run; only the fact list or its labels
change. Resumable per arm.
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from concurrent.futures import ThreadPoolExecutor

from src.extract.induced_vocab import load_fact_texts
from v2 import element_layer as S4
from v2 import paths
from v2.attribution_audit import classify
from v2.element_catalogue import load_claims

BASE = paths.INTERIM / "element_verdicts_element_satisfy_v2_facts.jsonl"
GOLD = paths.ROOT.parent / "Data" / "gold" / "element_gold_tasks.jsonl"
TAGS = paths.INTERIM / "prior_court_findings.json"
ARMS = {"exclude": "element_satisfy.v2", "tag": "element_satisfy.v3"}


def out_path(arm: str):
    return paths.INTERIM / f"element_verdicts_findings_{arm}.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    args = ap.parse_args()

    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"] for l in open(paths.MASKED)}
    facts, _ = load_fact_texts()
    flagged = set()
    for f in facts:
        m = masked.get(f["case_id"])
        if m is not None and classify(f, m)[0] == "stripped":
            flagged.add(f["fact_id"])
    TAGS.write_text(json.dumps(sorted(flagged)))
    by_case = collections.defaultdict(list)
    for f in facts:
        by_case[f["case_id"]].append(f)
    print(f"facts {len(facts):,}; flagged as prior-court findings {len(flagged):,} -> {TAGS.name}")

    base = [json.loads(l) for l in open(BASE)]
    gold_cases = {json.loads(l)["case_id"] for l in open(GOLD)}
    pairs = sorted({(r["case_id"], r["claim_id"]) for r in base
                    if r["case_id"] in gold_cases
                    or any(v["verdict"] == "NOT_SATISFIED" for v in r["verdicts"])})
    claims = {c.claim_id: c for c in load_claims()}
    print(f"pairs to re-run: {len(pairs):,} (NOT_SATISFIED-bearing or in the 40 gold cases)")

    for arm in args.arms:
        S4.PROMPT_ID = ARMS[arm]
        out = out_path(arm)
        done = set()
        if out.exists():
            for line in open(out):
                try:
                    r = json.loads(line)
                    done.add((r["case_id"], r["claim_id"]))
                except json.JSONDecodeError:
                    pass
        jobs = []
        for c, cl in pairs:
            if (c, cl) in done:
                continue
            fs = by_case[c]
            if arm == "exclude":
                fs = [f for f in fs if f["fact_id"] not in flagged]
            else:
                fs = [dict(f, prior_court_finding=f["fact_id"] in flagged) for f in fs]
            jobs.append((c, fs, claims[cl]))
        print(f"\n[{arm}] prompt {ARMS[arm]}: {len(jobs):,} calls ({len(done):,} already done)",
              flush=True)
        t0 = time.time()
        with out.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(S4.judge, c, fs, cl, args.model) for c, fs, cl in jobs]
            for n, fu in enumerate(futs, 1):
                r = fu.result()
                r["arm"] = arm
                fh.write(json.dumps(r) + "\n")
                if n % 100 == 0:
                    fh.flush()
                    el = time.time() - t0
                    print(f"  {n:,}/{len(futs):,} ({n/el:.1f}/s)", flush=True)
        print(f"[{arm}] done in {(time.time()-t0)/60:.1f} min -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
