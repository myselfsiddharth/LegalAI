"""How much of the fact stream is a court's finding with its attribution stripped off?

Motivation (s4_cited_fact_types.json): 20 of S4's 43 NOT_SATISFIED verdicts rest on a court's
conclusion rather than primary evidence, and the usual mechanism is extraction dropping the
attribution -- "it was concluded that ... the plaintiff did not acquire any good or valid title"
becomes the bare fact "The plaintiff did not acquire valid title". Masking cannot catch this: it is a
PRIOR court's view, not this court's order, so it is legitimately inside `masked_text`. What is lost
is only the label saying whose view it is.

Method, deliberately mechanical so it can be audited: each fact carries a character span into
`masked_text`. Take the sentence that contains the span, up to where the quote starts, and look for a
court-attribution cue there (a forum + a finding verb, or an impersonal "it was held/found/...").
A fact is
  stripped    cue in the source sentence, no attribution in the fact text
  attributed  the fact text itself names the court and the finding
  plain       no cue in the source sentence

`--sample N` prints N random facts from each of the first two classes with their source sentence, so
the detector's precision can be read off by eye before anyone trusts the rate (gotcha #9).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re

from v2 import paths

FORUM = (r"(trial court|trial judge|high court|learned (?:single |trial |district |civil )?judge|"
         r"division bench|full bench|appellate court|first appellate|lower (?:appellate )?court|"
         r"courts? below|both the courts|the courts|munsi(?:f|ff)|subordinate judge|sub-judge|"
         r"district judge|civil judge|tribunal|collector|commissioner|board of revenue|magistrate|"
         r"rent controller|arbitrator|the court|this court|single judge|bench)")
VERB = (r"(held|holds|found|finds|holding|finding|findings|concluded|conclusion|observed|opined|"
        r"recorded|decided|took the view|was of the (?:view|opinion)|came to the conclusion|"
        r"negatived|repelled|rejected|accepted|upheld|affirmed|confirmed)")
CUE = re.compile(rf"\b{FORUM}\b[^.;]{{0,120}}?\b{VERB}\b|\bit (?:was|has been|is) "
                 rf"(?:held|found|concluded|observed|recorded|decided)\b|\bon the (?:above |said )?"
                 rf"findings?\b", re.I)
IN_FACT = re.compile(rf"\b{FORUM}\b[^.;]{{0,120}}?\b{VERB}\b|\b(?:held|found|concluded|opined) that\b",
                     re.I)
SENT_END = re.compile(r"[.;:]\s")


def source_prefix(text: str, start: int, look: int = 400) -> str:
    """From the start of the sentence containing `start` up to `start`."""
    lo = max(0, start - look)
    window = text[lo:start]
    ends = [m.end() for m in SENT_END.finditer(window)]
    return window[ends[-1]:] if ends else window


def classify(fact: dict, masked: str) -> tuple[str, str]:
    sp = fact["source_span"]
    a, b = json.loads(sp) if isinstance(sp, str) else sp
    prefix = source_prefix(masked, a)
    # A fact that names a forum at all ("The trial court acquitted...", "the High Court allowed
    # the appeal") keeps its attribution even without a finding verb; the hand-checked sample
    # showed these were the commonest false "stripped" hits.
    if IN_FACT.search(fact["text"]) or re.search(rf"\b{FORUM}\b", fact["text"], re.I):
        return "attributed", prefix
    if CUE.search(prefix + " " + masked[a:b][:80]):
        return "stripped", prefix
    return "plain", prefix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    masked = {}
    for line in open(paths.MASKED):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]

    facts = [json.loads(l) for l in open(paths.FACTS)]
    out, by_class, per_case = [], collections.Counter(), collections.defaultdict(collections.Counter)
    by_who = collections.defaultdict(collections.Counter)
    for f in facts:
        m = masked.get(f["case_id"])
        if m is None:
            continue
        cls, prefix = classify(f, m)
        by_class[cls] += 1
        per_case[f["case_id"]][cls] += 1
        by_who[f.get("asserted_by") or "unknown"][cls] += 1
        out.append((cls, f, prefix))

    n = sum(by_class.values())
    print(f"=== attribution audit over {n:,} facts, {len(per_case):,} cases ===")
    for c in ("plain", "attributed", "stripped"):
        print(f"  {c:11s} {by_class[c]:7,d}  {by_class[c]/n:6.1%}")
    hit = sum(1 for c in per_case.values() if c["stripped"])
    print(f"  cases with >=1 stripped finding: {hit:,}/{len(per_case):,} = {hit/len(per_case):.1%}")
    print("\n  by asserted_by (stripped rate):")
    for who, c in sorted(by_who.items(), key=lambda kv: -sum(kv[1].values())):
        t = sum(c.values())
        print(f"    {who:16s} {c['stripped']:6,d}/{t:6,d} = {c['stripped']/t:.1%}")

    if args.sample:
        rng = random.Random(573)
        for cls in ("stripped", "attributed", "plain"):
            pool = [o for o in out if o[0] == cls]
            print(f"\n----- sample: {cls} -----")
            for i, (_, f, prefix) in enumerate(rng.sample(pool, min(args.sample, len(pool)))):
                print(f"[{cls[0]}{i}] FACT: {f['text'][:160]}")
                print(f"      SRC : ...{prefix[-170:]}")

    (paths.EXPERIMENTS / "attribution_audit.json").write_text(json.dumps({
        "n_facts": n, "n_cases": len(per_case), "classes": dict(by_class),
        "cases_with_stripped": hit,
        "by_asserted_by": {k: dict(v) for k, v in by_who.items()},
        "caveat": "mechanical cue detector; precision from the hand-checked sample is reported "
                  "separately. Does not catch this court's own reasoning left in masked_text."},
        indent=1))


if __name__ == "__main__":
    main()
