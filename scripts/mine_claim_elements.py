"""Build a catalogue of the ELEMENTS a land-dispute claim decomposes into.

Step 2 of the pipeline maps facts onto the constituent requirements of a claim ("did the
possession run continuously for the statutory period?"). Those requirement lists have to
come from somewhere, and writing them from memory would be precisely the fabrication this
project exists to prevent -- an invented element is worse than an invented citation, since
nothing downstream would ever catch it.

So the catalogue is mined from what courts actually said. Two stages, the same cheap-filter
-then-constrained-model shape used elsewhere in this pipeline:

  1. A sentence must name the claim AND carry a requirement trigger, in the SAME sentence.
     Window-based matching was tried first and failed: enumeration markers like "(i)" in
     Indian judgments overwhelmingly mark numbered findings and statutory sub-clauses, not
     requirement lists, so windows returned narrative with the right words near it.
  2. The model reads one such sentence and returns only the requirement phrases it states,
     or nothing. It never invents an element: a sentence that merely mentions the claim
     yields [].

Every element keeps the judgments and sentences that stated it, so a catalogue entry can be
traced to its source exactly like a citation can.

Usage: python3 scripts/mine_claim_elements.py --per-claim 60
"""
import argparse
import json
import re
import time
from collections import Counter, defaultdict

from ode_lib import ROOT, get_client

CORPUS = ROOT / "Data" / "processed" / "corpus_text.jsonl"
OUT = ROOT / "Data" / "processed" / "claim_elements.json"
SENT_CACHE = ROOT / "Data" / "processed" / "element_sentences.json"
CACHE_CAP = 200  # mine generously once; each run slices what it needs

CLAIMS = {
    "adverse possession":   r"adverse possession",
    "specific performance": r"specific performance",
    "easement":             r"easement",
    "injunction":           r"injunction",
    "partition":            r"partition",
    "land acquisition":     r"land acquisition|acquisition of land",
}
TRIGGER = re.compile(
    r"\b(?:essential\s+)?(?:ingredients?|requisites?|elements?|requirements?)\b"
    r"|\bin order to (?:establish|succeed|constitute|prove|claim|get|obtain)\b"
    r"|\b(?:must|has to|have to|needs? to)\s+(?:prove|establish|show|satisfy|plead)\b"
    r"|\bto (?:establish|constitute|succeed in|make out|sustain) (?:a |an |the )?(?:plea |claim |case )?of\b"
    r"|\bsine qua non\b|\bburden (?:of proof )?(?:lies|is) (?:up)?on\b", re.I)
SENT = re.compile(r"(?<=[.;:!?])\s+(?=[A-Z(])")

SYSTEM = ("You extract legal requirements from Indian Supreme Court judgments. You never "
          "add a requirement the sentence does not state.")
PROMPT = """Claim: {claim}

Sentence from a judgment:
"{sentence}"

Does this sentence state what a party must prove to establish {claim}? If it does, list \
only the requirements it actually states, each as a short noun phrase of at most six words \
(for example "continuous possession", "hostile to the true owner", "readiness and \
willingness"). If the sentence merely mentions {claim} without stating a requirement, \
return an empty list.

Return ONLY JSON: {{"elements": ["...", "..."]}}"""


def norm(s: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    s = re.sub(r"\b(?:the|a|an|of|to|be|is|was|that|his|her|its|their)\b", " ", s)
    return " ".join(s.split())


def load_sentences(per_claim: int, refresh: bool) -> dict[str, list[dict]]:
    """Candidate sentences, mined once and cached.

    The mining pass reads 320 MB and sentence-splits every judgment that mentions a claim,
    which dominates the runtime; the model calls that follow are comparatively cheap. Paying
    it on every run made iteration painful, so it is cached and sliced.
    """
    # Validity is about the cap the cache was mined at, NOT about per-claim counts: some
    # claims genuinely have few candidates in the whole corpus (easement has 7), so requiring
    # per_claim sentences for every claim invalidates a perfectly good cache forever and
    # re-reads 320 MB on every run.
    if SENT_CACHE.exists() and not refresh:
        cached = json.loads(SENT_CACHE.read_text())
        if cached.get("_cap", 0) >= per_claim and set(cached.get("claims", {})) >= set(CLAIMS):
            print(f"using cached sentences ({SENT_CACHE.name}, "
                  f"mined at cap {cached['_cap']})")
            return {c: cached["claims"][c][:per_claim] for c in CLAIMS}
    mined = mine_sentences(CACHE_CAP)
    SENT_CACHE.write_text(json.dumps({"_cap": CACHE_CAP, "claims": mined}, indent=1))
    print(f"mined and cached -> {SENT_CACHE.name}")
    return {c: mined.get(c, [])[:per_claim] for c in CLAIMS}


def mine_sentences(per_claim: int) -> dict[str, list[dict]]:
    rx = {k: re.compile(v, re.I) for k, v in CLAIMS.items()}
    out = {k: [] for k in CLAIMS}
    seen = defaultdict(set)
    # Cheap gates first: most judgments mention no land claim at all, and splitting a
    # 60k-character document into sentences is far more expensive than a substring test.
    words = ["adverse possession", "specific performance", "easement", "injunction",
             "partition", "acquisition of land", "land acquisition"]
    with CORPUS.open() as f:
        for line in f:
            if all(len(v) >= per_claim for v in out.values()):
                break
            d = json.loads(line)
            flat = re.sub(r"\s+", " ", d["text"])
            low = flat.lower()
            if not any(w in low for w in words):
                continue
            for s in SENT.split(flat):
                if not (60 < len(s) < 700) or not TRIGGER.search(s):
                    continue
                for claim, r in rx.items():
                    if len(out[claim]) >= per_claim or not r.search(s):
                        continue
                    key = norm(s)[:140]
                    if key in seen[claim]:
                        continue
                    seen[claim].add(key)
                    out[claim].append({"doc_id": d["doc_id"], "year": d["year"],
                                       "sentence": s.strip()})
    return out


def parse(raw: str) -> list[str]:
    body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    a, b = body.find("{"), body.rfind("}")
    if a == -1:
        return []
    try:
        got = json.loads(body[a:b + 1]).get("elements") or []
    except json.JSONDecodeError:
        return []
    return [e.strip() for e in got
            if isinstance(e, str) and 3 <= len(e.strip()) <= 70]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-claim", type=int, default=60)
    ap.add_argument("--min-support", type=int, default=2,
                    help="distinct judgments that must state an element for it to be kept")
    ap.add_argument("--model", default="llama4-scout-17b")
    ap.add_argument("--refresh", action="store_true", help="re-mine sentences from the corpus")
    args = ap.parse_args()

    cands = load_sentences(args.per_claim, args.refresh)
    for c, v in cands.items():
        print(f"  {c:22} {len(v):3} sentences")

    client = get_client()
    catalogue = {}
    for claim, rows in cands.items():
        by_element = defaultdict(lambda: {"support_docs": set(), "sources": []})
        empties = dropped = 0
        for i, row in enumerate(rows, 1):
            # A transient endpoint outage once dropped 150 of 253 sentences silently, which
            # looked like "the model found no elements" rather than "we never asked".
            # Retry, and only give up on a sentence after the endpoint stays down.
            els, failed = None, None
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=300,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": PROMPT.format(
                                      claim=claim, sentence=row["sentence"])}])
                    els = parse(resp.choices[0].message.content or "")
                    break
                except Exception as e:
                    failed = type(e).__name__
                    if attempt < 3:
                        time.sleep(2 * (attempt + 1) ** 2)
            if els is None:
                dropped += 1
                print(f"    DROPPED after retries ({failed})", flush=True)
                continue
            if not els:
                empties += 1
            for e in els:
                k = norm(e)
                if not k:
                    continue
                by_element[k]["support_docs"].add(row["doc_id"])
                by_element[k]["sources"].append({"doc_id": row["doc_id"], "year": row["year"],
                                                 "phrase": e, "sentence": row["sentence"][:280]})
        kept = {k: v for k, v in by_element.items()
                if len(v["support_docs"]) >= args.min_support}
        catalogue[claim] = sorted(
            ({"element": max(Counter(s["phrase"] for s in v["sources"]).items(),
                             key=lambda kv: kv[1])[0],
              "key": k,
              "support": len(v["support_docs"]),
              "sources": v["sources"][:4]} for k, v in kept.items()),
            key=lambda e: -e["support"])
        print(f"  {claim:22} {len(rows)} sentences -> {len(by_element)} raw, "
              f"{len(kept)} with support >= {args.min_support} "
              f"({empties} stated none, {dropped} dropped)",
              flush=True)

    OUT.write_text(json.dumps(catalogue, indent=2))
    print(f"\nWrote {OUT}")
    for claim, els in catalogue.items():
        if not els:
            continue
        print(f"\n### {claim}")
        for e in els[:8]:
            print(f"  [{e['support']:2}] {e['element']}")


if __name__ == "__main__":
    main()
