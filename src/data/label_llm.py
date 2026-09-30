"""Stage 1 (§5.1): the LLM labelling pass, and the agreement number that justifies trusting it.

§5.1 prescribes rules first, LLM for residuals, then a human check. We run the LLM over
*every* case rather than only the residuals, because the rules-vs-LLM agreement rate on the
cases where rules DID fire is the only cheap estimate of label quality we can get, and the
cache makes the extra calls free on every re-run.

Two guards, both learned the hard way on this project:

  * The model must quote the operative sentence **verbatim**, and we check the quote really
    occurs in the judgment. A quote that does not occur is not evidence, so the label is
    marked unverified rather than accepted. (This is the check that previously caught a model
    presenting an elided reconstruction as a quotation.)
  * A dropped call is counted separately from a model that answered "I don't know".
    Conflating them once turned an API outage into a false substantive finding, so the
    summary reports `dropped` on its own line and never folds it into disagreement.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from pydantic import BaseModel, Field

from src import paths
from src.llm import client

PROMPT_ID = "label_outcome.v1"
OUT_PATH = paths.INTERIM / "outcome_labels_llm.jsonl"

OPENING_CHARS = 3500
ORDER_CHARS = 3500
MIN_QUOTE_CHARS = 12          # "Appeal dismissed." is a real operative line, not a stub
# A quote must actually state a disposition. This replaces a longer length floor, which
# rejected the genuine one-line dispositions used in pre-1970 law-report style.
DISPOSITION_TERM = re.compile(
    r"\b(allow\w*|dismiss\w*|decree\w*|set aside|remand\w*|remit\w*|quash\w*|reject\w*"
    r"|succeed\w*|fail\w*|disposed|withdrawn|modif\w*|restor\w*|direct\w*|grant\w*)\b", re.I)


class OutcomeJSON(BaseModel):
    initiator_outcome: str
    operative_quote: str = ""
    initiator_role: str | None = None
    initiator_was_original_plaintiff: bool | None = None
    original_plaintiff_outcome: str | None = None
    prior_court_outcome: str | None = None
    confidence: float = Field(default=0.5)


def _norm(s: str) -> str:
    """Normalise for comparison, absorbing two artifacts of PDF extraction.

    Whitespace: extraction inserts line breaks mid-sentence, so a correct quote differs from
    the source in whitespace alone.

    Hyphenation: judgments before roughly 1980 are typeset with hyphenated line wraps, which
    survive extraction as "execut- ing", "plain- tiff's", "sanc- tioned". A model quoting the
    sentence silently repairs these -- which is the correct reading of the page, but does not
    match the raw text. Measured on a 60-case pilot, this single artifact was rejecting 9 of
    25 otherwise-correct quotes. Repair is applied to BOTH sides, so it cannot manufacture a
    match that the words themselves do not support.
    """
    s = re.sub(r"\s+", " ", s or "").strip().lower()
    s = re.sub(r"(\w)[-\u00ad]\s+(\w)", r"\1\2", s)      # execut- ing -> executing
    return s


def quote_is_grounded(quote: str, text: str) -> bool:
    q = _norm(quote)
    if len(q) < MIN_QUOTE_CHARS:
        return False
    if "..." in quote or "…" in quote:      # an elision is a reconstruction, not a quotation
        return False
    if not DISPOSITION_TERM.search(q):      # not a disposition, so not the operative sentence
        return False
    return q in _norm(text)


def build_messages(rec: dict) -> list[dict]:
    text = rec["text"]
    opening = text[:OPENING_CHARS]
    order = text[-ORDER_CHARS:]
    return [
        {"role": "system", "content": client.load_prompt(PROMPT_ID)},
        {"role": "user", "content": f"OPENING:\n{opening}\n\n---\n\nORDER:\n{order}"},
    ]


def label_one(rec: dict, model: str) -> dict:
    res = client.complete(build_messages(rec), model=model, json_schema=OutcomeJSON,
                          prompt_id=PROMPT_ID, max_tokens=500)
    row = {"doc_id": rec["doc_id"], "year": rec["year"], "model": model,
           "cached": res.cached, "ok": res.ok}
    if not res.ok:
        row["status"] = "dropped"           # never got an answer -- NOT a disagreement
        row["error"] = res.error
        return row
    if res.parsed is None:
        row["status"] = "unparseable"       # the model answered, but not in the schema
        row["raw"] = res.text[:300]
        return row
    p = res.parsed
    row.update({
        "status": "ok",
        "initiator_outcome": p.initiator_outcome,
        "operative_quote": p.operative_quote,
        "quote_grounded": quote_is_grounded(p.operative_quote, rec["text"]),
        "initiator_role": p.initiator_role,
        "initiator_was_original_plaintiff": p.initiator_was_original_plaintiff,
        "original_plaintiff_outcome": p.original_plaintiff_outcome,
        "prior_court_outcome": p.prior_court_outcome,
        "confidence": p.confidence,
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=0, help="0 = all cases")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only-unknown", action="store_true",
                    help="label only the cases the rules could not decide (§5.1 literal reading)")
    args = ap.parse_args()

    rules = {json.loads(l)["doc_id"]: json.loads(l)
             for l in open(paths.INTERIM / "outcome_labels.jsonl")}
    cases = [json.loads(l) for l in open(paths.CASE_TEXT)]
    if args.only_unknown:
        cases = [c for c in cases
                 if rules.get(c["doc_id"], {}).get("initiator_outcome") == "UNKNOWN"]
    if args.n:
        cases = cases[:args.n]

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok":
                        done.add(r["doc_id"])
                except json.JSONDecodeError:
                    pass
    todo = [c for c in cases if c["doc_id"] not in done]
    print(f"{len(cases)} target cases, {len(done)} already labelled, {len(todo)} to do "
          f"(model={args.model}, workers={args.workers})", flush=True)

    rows = []
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(lambda c: label_one(c, args.model), todo), 1):
            out.write(json.dumps(row) + "\n")
            rows.append(row)
            if i % 250 == 0:
                out.flush()
                u = client.usage()
                print(f"  {i}/{len(todo)}  cache_hit_rate={u['cache_hit_rate']:.2f} "
                      f"dropped={u['dropped']}", flush=True)

    report(args.model)


def report(model: str | None = None) -> None:
    rules = {json.loads(l)["doc_id"]: json.loads(l)
             for l in open(paths.INTERIM / "outcome_labels.jsonl")}
    llm = {}
    with OUT_PATH.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            llm[r["doc_id"]] = r                      # later rows win

    status = Counter(r.get("status") for r in llm.values())
    ok = {d: r for d, r in llm.items() if r.get("status") == "ok"}
    grounded = sum(1 for r in ok.values() if r["quote_grounded"])

    print(f"\n=== LLM labelling: {len(llm)} cases attempted ===")
    for k, v in status.most_common():
        print(f"  {k:12s} {v:5d}  {100*v/len(llm):5.1f}%")
    print(f"  operative quote verifiably in the judgment: {grounded}/{len(ok)} "
          f"({100*grounded/len(ok):.1f}%)" if ok else "")

    # agreement, restricted to cases where BOTH decided and the quote checks out
    both = [(rules[d]["initiator_outcome"], r["initiator_outcome"])
            for d, r in ok.items()
            if d in rules and rules[d]["initiator_outcome"] != "UNKNOWN" and r["quote_grounded"]]
    if both:
        agree = sum(1 for a, b in both if a == b)
        print(f"\n  rules vs LLM, both decided & quote grounded (n={len(both)}): "
              f"{100*agree/len(both):.1f}% agreement")
        conf = Counter((a, b) for a, b in both if a != b)
        if conf:
            print("  top disagreements (rules -> llm):")
            for (a, b), c in conf.most_common(8):
                print(f"    {a:8s} -> {b:8s} {c:4d}")

    # what the LLM recovers from the rules' residual
    recovered = Counter(r["initiator_outcome"] for d, r in ok.items()
                        if rules.get(d, {}).get("initiator_outcome") == "UNKNOWN"
                        and r["quote_grounded"])
    if recovered:
        print(f"\n  recovered from rules-UNKNOWN (quote grounded): {sum(recovered.values())}")
        for k, v in recovered.most_common():
            print(f"    {k:8s} {v:5d}")
    print(f"\n  usage: {client.usage()}")


if __name__ == "__main__":
    main()
