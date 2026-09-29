"""Step 2: map the facts of a dispute onto the elements a claim requires.

Given facts and a claim, decide element by element whether the facts satisfy it. The
interesting problem is not getting a verdict -- a model will happily produce one -- but
making the verdict checkable. So a "satisfied" verdict must quote the fact that satisfies
the element, verbatim, and that quote is then checked against the input text. A quote that
does not occur in the facts means the model manufactured its own support, and the verdict is
refused rather than reported.

This gives a fabrication measure that needs no ground truth: the share of satisfied verdicts
whose supporting quote cannot be found in the input the model was given.

Element lists come from claim_elements.json, mined from judgments that stated them
(mine_claim_elements.py), so neither the elements nor their support are invented here.

Usage:
  python3 scripts/map_facts_to_elements.py --claim "adverse possession" --n 10
"""
import argparse
import json
import re
import time
from collections import Counter

from ode_lib import ROOT, get_client
from retrieval_lib import scrub_citations
from run_precedent_prediction import load_edges

ELEMENTS = ROOT / "Data" / "processed" / "claim_elements.json"
CORPUS = ROOT / "Data" / "processed" / "corpus_text.jsonl"
MAP_PATH = ROOT / "Data" / "processed" / "doc_id_to_pdf.json"
OUT = ROOT / "Data" / "processed" / "element_mapping.json"

SYSTEM = ("You analyse the facts of an Indian land dispute. You only ever support a "
          "conclusion by quoting the facts you were given, word for word.")
PROMPT = """Facts of the dispute:
\"\"\"
{facts}
\"\"\"

The claim is: {claim}
One requirement of that claim is: {element}

Do these facts satisfy that requirement? Answer SATISFIED, NOT_SATISFIED or UNCLEAR.

If SATISFIED, you must quote the exact words from the facts above that establish it — \
copied character for character, not paraphrased, and not from your own knowledge. If no \
passage in the facts establishes it, answer UNCLEAR with an empty quote.

Return ONLY JSON: {{"verdict": "...", "quote": "...", "reason": "<one short sentence>"}}"""


def norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", s.lower()).split())


def parse(raw: str) -> dict:
    body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    a, b = body.find("{"), body.rfind("}")
    if a == -1:
        return {}
    try:
        got = json.loads(body[a:b + 1])
    except json.JSONDecodeError:
        return {}
    return got if isinstance(got, dict) else {}


def check_quote(quote: str, facts_norm: str) -> tuple[bool, str]:
    """Does the quote actually occur in the facts the model was shown?

    Normalized substring rather than exact string match, because whitespace and punctuation
    are mangled by PDF extraction and a faithful copy can still differ on those. Short
    quotes are rejected outright: a handful of words occurs in almost any judgment and would
    certify nothing.
    """
    q = norm(quote)
    if len(q) < 25:
        return False, "quote too short to be evidence"
    if q in facts_norm:
        return True, "verbatim in facts"
    return False, "quote does not occur in the facts provided"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claim", default="adverse possession")
    ap.add_argument("--n", type=int, default=10, help="query cases")
    ap.add_argument("--facts-chars", type=int, default=6000)
    ap.add_argument("--min-support", type=int, default=3,
                    help="only test elements at least this many judgments stated")
    ap.add_argument("--model", default="llama4-scout-17b")
    args = ap.parse_args()

    catalogue = json.loads(ELEMENTS.read_text())
    if args.claim not in catalogue:
        raise SystemExit(f"no elements for {args.claim!r}; have: {sorted(catalogue)}")
    elements = [e for e in catalogue[args.claim] if e["support"] >= args.min_support]
    if not elements:
        raise SystemExit(f"no elements for {args.claim!r} at support >= {args.min_support}")
    print(f"{args.claim}: testing {len(elements)} elements "
          f"(support >= {args.min_support})")
    for e in elements:
        print(f"    [{e['support']}] {e['element']}")

    # cases that actually plead this claim, so the mapping is not being asked nonsense
    haystack = set(json.loads(MAP_PATH.read_text()))
    _, years = load_edges()
    term = re.compile(re.escape(args.claim.split()[0]), re.I)
    cases = []
    with CORPUS.open() as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] not in haystack or years.get(d["doc_id"], 0) < 2000:
                continue
            if not re.search(re.escape(args.claim), d["text"], re.I):
                continue
            cases.append(d)
            if len(cases) >= args.n:
                break
    print(f"\n{len(cases)} query cases mentioning the claim\n")

    client = get_client()
    status = Counter()
    rows = []
    for n, d in enumerate(cases, 1):
        facts = scrub_citations(d["text"])[: args.facts_chars]
        fnorm = norm(facts)
        per_case = []
        for e in elements:
            # Retry, for the same reason mine_claim_elements.py does: a flaky endpoint
            # otherwise silently converts "we never asked" into a missing verdict, and an
            # earlier run lost 23 of 80 calls that way.
            got, failed = None, None
            for attempt in range(4):
                try:
                    resp = client.chat.completions.create(
                        model=args.model, temperature=0, max_tokens=400,
                        messages=[{"role": "system", "content": SYSTEM},
                                  {"role": "user", "content": PROMPT.format(
                                      facts=facts, claim=args.claim, element=e["element"])}])
                    got = parse(resp.choices[0].message.content or "")
                    break
                except Exception as ex:
                    failed = type(ex).__name__
                    if attempt < 3:
                        time.sleep(2 * (attempt + 1) ** 2)
            if got is None:
                status["DROPPED"] += 1
                print(f"    DROPPED after retries ({failed})", flush=True)
                continue
            verdict = str(got.get("verdict", "")).upper()
            quote = str(got.get("quote") or "")
            rec = {"element": e["element"], "verdict": verdict,
                   "reason": str(got.get("reason") or "")[:160]}
            if verdict == "SATISFIED":
                ok, why = check_quote(quote, fnorm)
                rec.update({"quote": quote[:220], "grounded": ok, "check": why})
                status["SATISFIED_grounded" if ok else "SATISFIED_refused"] += 1
            else:
                status[verdict or "UNPARSED"] += 1
            per_case.append(rec)
        rows.append({"doc_id": d["doc_id"], "year": d["year"], "elements": per_case})
        print(f"  [{n}/{len(cases)}] {d['doc_id']}", flush=True)

    sat = status["SATISFIED_grounded"] + status["SATISFIED_refused"]
    print(f"\n=== ELEMENT MAPPING: {args.claim} ===")
    for k, v in status.most_common():
        print(f"  {k:22} {v}")
    if sat:
        print(f"\nsatisfied verdicts: {sat}")
        print(f"  grounded in a real quote : {status['SATISFIED_grounded']} "
              f"({status['SATISFIED_grounded']/sat:.1%})")
        print(f"  refused, quote not found : {status['SATISFIED_refused']} "
              f"({status['SATISFIED_refused']/sat:.1%})  <-- manufactured support")
    OUT.write_text(json.dumps({"claim": args.claim, "elements_tested": len(elements),
                               "cases": len(cases), "status": dict(status),
                               "per_case": rows}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
