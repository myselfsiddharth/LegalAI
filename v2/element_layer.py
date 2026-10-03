"""S4 — do the facts SATISFY or DEFEAT each element? The untested heart of the architecture.

v1 never built this. Its feature group `E` is empty because the element metadata it wanted
(`burden_on`, `burden_standard`) does not exist for any of the 66 elements (blocker B2), so the
layer was skipped entirely rather than built unweighted. That left the architecture's central claim
-- that elements are the right intermediate representation between facts and claims -- with no
measurement either way.

## The gate, and why this task is worth doing before the gold set exists

Every `SATISFIED` or `NOT_SATISFIED` verdict must carry a quote copied from the facts shown. The
quote is checked mechanically (whitespace and case normalised, minimum length enforced), and a
verdict whose quote cannot be located is **discarded and counted as a refusal**.

That yields a fabrication metric requiring **no ground truth**: it catches manufactured support, as
distinct from wrong reasoning. A model that invents a quote is caught; a model that reasons badly
from a real quote is not, and that limit is stated rather than hidden.

Three distinct non-answers are counted separately, because conflating them is how v1 turned an
outage into a finding twice:

  `refused_quote_absent`   the model gave a quote that is not in the facts -> fabricated support
  `refused_quote_short`    a fragment too short to be evidence (< 25 chars)
  `dropped_no_answer`      the call never returned (`ok=False`) -- an outage, NOT a model opinion

## Scope and honesty constraints, both inherited from v1

**Party attribution is 19-20%.** Deciding whether a fact *defeats* an element properly requires
knowing who asserted it, and for four facts in five v1 cannot say. The `favours` field is therefore
collected but reported with its coverage, and no claim is made that it is reliable.

**Burden metadata is 0 of 66** (B2). Elements are treated as unweighted. A real implementation would
weight by who bears the burden; this one cannot, and says so.

## The model matters here, and it is not v1's

v1 chose `llama4-scout-17b` everywhere, correctly: on the extraction benchmark it matched models
6.8x slower. On THIS task it is degenerate. Measured on identical cases:

| model | SATISFIED | NOT_SATISFIED | UNCLEAR | fabricated |
|---|---|---|---|---|
| `llama4-scout-17b` | 15 | **0** | 150 | 0.6% |
| `llama4-scout-17b`, prompt rewritten to solicit negatives | 23 | **0** | 151 | 0.6% |
| `llama4-scout-17b`, given the court's REASONING too (leaky) | 21 | **0** | 149 | 2.9% |
| **`qwen3-235b-a22b-instruct-2507`** | 27 | **4** | 56 | **0.0%** |

Scout emits zero `NOT_SATISFIED` under every condition tried, including a prompt that calls the
verdict "common in contested litigation" and lists five examples of it. Two hypotheses were tested
and refuted before the third was confirmed:

1. *prompt over-correction* -- refuted: rewriting the prompt to actively solicit negatives changed
   nothing (0 -> 0).
2. *defeats live in the court's reasoning, which §5.2 masking removes* -- refuted: showing the model
   the order-removed text, reasoning included, still produced 0.
3. *it is the model* -- confirmed. A larger model uses all three verdicts and fabricates less.

A layer that cannot say `NOT_SATISFIED` cannot express the "or defeat" half of the architecture, so
this was a blocking defect rather than a quality preference. The cost is throughput: qwen3-235b is
several times slower than scout, which is why S4 runs detached.

## Elements are scored per (case, claim), not per (case, 66 elements)

A case about adverse possession has nothing to say about easement elements. Asking anyway produces
sixty confident `UNCLEAR`s that dilute every metric. `element_catalogue.select_claims` picks the top
claims by max fact-embedding cosine, so each call covers 4-6 elements that are plausibly in play.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from pydantic import BaseModel

from src.extract.induced_vocab import load_fact_texts
from src.llm import client
from v2 import paths
from v2.element_catalogue import Claim, all_elements, load_claims, select_claims

PROMPT_ID = os.environ.get("S4_PROMPT", "element_satisfy.v1")
MIN_QUOTE = 25
_EV = os.environ.get("S4_EVIDENCE", "facts")
OUT = paths.INTERIM / f"element_verdicts_{PROMPT_ID.replace('.','_')}_{_EV}.jsonl"


class Verdict(BaseModel):
    element_id: str
    verdict: Literal["SATISFIED", "NOT_SATISFIED", "UNCLEAR"]
    quote: str = ""
    favours: Literal["claimant", "respondent", "neither"] = "neither"


class Reply(BaseModel):
    verdicts: list[Verdict]


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def build_user(facts: list[dict], claim: Claim, raw: str | None = None) -> str:
    """`raw`, when given, replaces the fact list with order-removed judgment text.

    That arm is LEAKY by construction -- it retains the court's reasoning, which v1 showed carries
    outcome signal. It exists only as a diagnostic for one question: NOT_SATISFIED never appears
    from extracted facts even under a prompt that actively solicits it, and the hypothesis is that
    affirmative contradiction of a legal element is a JUDICIAL FINDING rather than a narrated fact.
    If NOT_SATISFIED appears once the reasoning is visible, that hypothesis is confirmed. Never use
    this arm to build features.
    """
    if raw is not None:
        lines = ["## CASE TEXT (order removed)", "", raw[:14000]]
    else:
        lines = ["## FACTS", ""]
        for i, f in enumerate(facts, 1):
            who = f.get("asserted_by") or "unattributed"
            st = f.get("disputed_status") or "unknown"
            lines.append(f"{i}. {f['text']}   [asserted_by: {who}; status: {st}]")
    lines += ["", f"## CLAIM — {claim.name}", "", claim.issue, "", "## ELEMENTS", ""]
    for e in claim.elements:
        d = f" — {e.definition}" if e.definition else ""
        lines.append(f"- `{e.element_id}` **{e.name}**{d}")
    return "\n".join(lines)


def judge(case_id: str, facts: list[dict], claim: Claim, model: str | None,
          raw: str | None = None) -> dict:
    """One (case, claim). Returns the gated verdicts plus per-reason refusal counts."""
    sysmsg = client.load_prompt(PROMPT_ID)
    res = client.complete(
        [{"role": "system", "content": sysmsg},
         {"role": "user", "content": build_user(facts, claim, raw)}],
        model=model, json_schema=Reply, prompt_id=PROMPT_ID, max_tokens=1600)

    if not res.ok:
        return {"case_id": case_id, "claim_id": claim.claim_id, "verdicts": [],
                "dropped_no_answer": len(claim.elements), "refused_quote_absent": 0,
                "refused_quote_short": 0, "unparsed": 0}
    if res.parsed is None:
        return {"case_id": case_id, "claim_id": claim.claim_id, "verdicts": [],
                "dropped_no_answer": 0, "refused_quote_absent": 0,
                "refused_quote_short": 0, "unparsed": len(claim.elements)}

    haystack = _norm(raw[:14000] if raw is not None else " ".join(f["text"] for f in facts))
    valid_ids = {e.element_id for e in claim.elements}
    kept, absent, short = [], 0, 0
    for v in res.parsed.verdicts:
        if v.element_id not in valid_ids:
            continue
        if v.verdict == "UNCLEAR":
            kept.append({"element_id": v.element_id, "verdict": "UNCLEAR",
                         "quote": "", "favours": v.favours, "gated": False})
            continue
        q = _norm(v.quote)
        if len(q) < MIN_QUOTE:
            short += 1
            continue
        if q not in haystack:
            absent += 1
            continue
        kept.append({"element_id": v.element_id, "verdict": v.verdict,
                     "quote": v.quote.strip(), "favours": v.favours, "gated": True})
    return {"case_id": case_id, "claim_id": claim.claim_id, "verdicts": kept,
            "dropped_no_answer": 0, "refused_quote_absent": absent,
            "refused_quote_short": short, "unparsed": 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="cases to run; 0 = all")
    ap.add_argument("--top-claims", type=int, default=2)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507",
                    help="NOT the v1 default. llama4-scout-17b emits ZERO NOT_SATISFIED on this "
                         "task -- see the module docstring.")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--evidence", choices=("facts", "ildc_style"), default="facts",
                    help="'ildc_style' is a LEAKY diagnostic (retains the court's reasoning); "
                         "never use it to build features")
    args = ap.parse_args()

    claims = load_claims()
    print(f"catalogue: {len(claims)} claims, {len(all_elements(claims))} elements")

    facts, X = load_fact_texts()
    by_case: dict[str, list[int]] = {}
    for i, f in enumerate(facts):
        by_case.setdefault(f["case_id"], []).append(i)
    cases = sorted(by_case)
    if args.n:
        cases = cases[:: max(1, len(cases) // args.n)][: args.n]

    done: set[tuple[str, str]] = set()
    if args.resume and OUT.exists():
        for line in open(OUT):
            try:
                r = json.loads(line)
                done.add((r["case_id"], r["claim_id"]))
            except json.JSONDecodeError:
                pass

    raws: dict[str, str] = {}
    if args.evidence == "ildc_style":
        print("!! LEAKY DIAGNOSTIC ARM: evidence includes the court's reasoning. "
              "Do not build features from this.\n")
        for line in open(paths.MASKED):
            r = json.loads(line)
            raws[r["doc_id"]] = r["ildc_style_text"]

    jobs = []
    for c in cases:
        idx = by_case[c]
        for ci in select_claims(claims, X[idx], args.top_claims):
            if (c, claims[ci].claim_id) not in done:
                jobs.append((c, [facts[i] for i in idx], claims[ci], raws.get(c)))
    print(f"{len(cases):,} cases x top-{args.top_claims} claims -> {len(jobs):,} calls "
          f"({len(done):,} already done)", flush=True)

    attributed = sum(1 for c in cases for i in by_case[c]
                     if (facts[i].get("asserted_by") or "court_narrative") != "court_narrative")
    total_f = sum(len(by_case[c]) for c in cases)
    print(f"party attribution coverage on these cases: {attributed/max(1,total_f):.1%} "
          f"(v1 constraint -- `favours` is reported with this caveat)\n")

    t0 = time.time()
    agg = {"SATISFIED": 0, "NOT_SATISFIED": 0, "UNCLEAR": 0}
    ref = {"refused_quote_absent": 0, "refused_quote_short": 0, "dropped_no_answer": 0,
           "unparsed": 0}
    with OUT.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(judge, c, f, cl, args.model, rw) for c, f, cl, rw in jobs]
        for n, fu in enumerate(futs, 1):
            r = fu.result()
            out.write(json.dumps(r) + "\n")
            for v in r["verdicts"]:
                agg[v["verdict"]] += 1
            for k in ref:
                ref[k] += r.get(k, 0)
            if n % 200 == 0:
                out.flush()
                el = time.time() - t0
                print(f"  {n:,}/{len(futs):,}  ({n/el:.1f}/s, "
                      f"eta {(len(futs)-n)/(n/el)/60:.1f} min)", flush=True)

    tot_v = sum(agg.values())
    proposed = tot_v + ref["refused_quote_absent"] + ref["refused_quote_short"]
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
    print(f"  verdicts kept: {tot_v:,}   " + "  ".join(f"{k} {v:,}" for k, v in agg.items()))
    print(f"  REFUSED (fabricated support): absent {ref['refused_quote_absent']:,}, "
          f"too short {ref['refused_quote_short']:,}  "
          f"= {(ref['refused_quote_absent']+ref['refused_quote_short'])/max(1,proposed):.2%} "
          f"of {proposed:,} proposed")
    print(f"  NEVER ANSWERED (outage, not an opinion): {ref['dropped_no_answer']:,}")
    print(f"  schema-unparseable: {ref['unparsed']:,}")
    print(f"-> {OUT}")

    (paths.EXPERIMENTS / f"s4_element_layer_{PROMPT_ID.replace('.','_')}_{_EV}.json").write_text(json.dumps({
        "stage": "S4 element layer", "prompt_id": PROMPT_ID,
        "model": args.model or client.DEFAULT_MODEL,
        "n_cases": len(cases), "top_claims": args.top_claims, "n_calls": len(jobs),
        "verdicts": agg, "refusals": ref, "n_proposed": proposed,
        "fabrication_rate": round((ref["refused_quote_absent"] + ref["refused_quote_short"])
                                  / max(1, proposed), 4),
        "attribution_coverage": round(attributed / max(1, total_f), 4),
        "min_quote_chars": MIN_QUOTE,
        "caveats": ["burden metadata is 0 of 66 (B2), so elements are unweighted",
                    "`favours` is unreliable at this attribution coverage",
                    "the gate catches fabricated SUPPORT, not wrong reasoning"]}, indent=1))


if __name__ == "__main__":
    main()
