"""End-to-end proof-carrying answer: facts in, IRAC out, every authority certified.

Retrieval alone cannot fabricate -- every candidate is a real corpus case with a real
doc_id -- so the verifier does nothing useful bolted straight onto it. The fabrication risk
appears one step later, when a model drafts the answer and can cite something that was
never in its brief, or garble a name into a case that does not exist. That is what gets
checked here:

  facts -> retrieve candidates -> model drafts IRAC citing authorities by name
        -> resolve each name against the candidate set by party matching
        -> CERTIFIED or REFUSED, never silently included

Two arms, because a verifier that never refuses anything proves nothing:
  grounded     - the model is given the retrieved candidates and told to cite only those
  closed_book  - the same model, same facts, no candidates at all

Each certified authority is then scored against the judgment's real citations, separating
"the model stayed inside its brief" (groundedness) from "the brief was right" (correctness).

Usage: python3 scripts/answer_from_facts.py --n 25 --k 20
"""
import argparse
import json
import re
from collections import Counter

from ode_lib import ROOT, get_client
from phase3_verify import normalize_parties, party_match_score, PARTY_MATCH_THRESHOLD
from retrieval_lib import BM25Index, load_corpus, scrub_citations
from run_precedent_prediction import load_edges

MAP_PATH = ROOT / "Data" / "processed" / "doc_id_to_pdf.json"
CORPUS = ROOT / "Data" / "processed" / "corpus_text.jsonl"
OUT = ROOT / "Data" / "processed" / "proof_carrying_answers.json"

SYSTEM = ("You are a legal research assistant working on Indian Supreme Court land and "
          "property disputes. You never cite an authority you cannot support.")

GROUNDED_PROMPT = """These are the facts of a dispute:

{facts}

These are the only prior decisions you may rely on:
{candidates}

Answer in IRAC form and cite the authorities that apply. Cite ONLY from the list above, by \
case name exactly as written there. If none of them apply, return an empty authorities list.

Return ONLY a JSON object:
{{"issue": "...", "rule": "...", "application": "...", "conclusion": "...",
  "authorities": ["<case name from the list>", ...]}}"""

CLOSED_PROMPT = """These are the facts of a dispute:

{facts}

Answer in IRAC form and cite the prior Indian Supreme Court decisions that apply.

Return ONLY a JSON object:
{{"issue": "...", "rule": "...", "application": "...", "conclusion": "...",
  "authorities": ["<case name as 'A v. B'>", ...]}}"""


def case_name(title: str) -> str:
    """"X vs Y on 30 May, 2022" -> "X vs Y" -- the caption as a lawyer would cite it."""
    return re.split(r"\s+on\s+\d", title)[0].strip()


def parse_json(raw: str) -> dict:
    body = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    a, b = body.find("{"), body.rfind("}")
    if a == -1 or b == -1:
        return {}
    try:
        out = json.loads(body[a:b + 1])
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.lower()).split())


def certify(name: str, candidates: dict[str, str]) -> dict:
    """Resolve a cited name to a retrieved candidate, verbatim first then by party names.

    Two stages on purpose. The model is told to cite exactly as written in its brief, so a
    faithful copy -- including a truncated one -- should certify without any fuzziness. Only
    then does party matching run, using the same matcher and threshold as the citation
    verifier so both stages agree on what counts as the same case.

    Keeping the stages apart matters for what the numbers mean. The party matcher is
    deliberately strict: MIN_COVERAGE refuses a match carried by one shared generic word, so
    "State of Bombay v. Advani" does NOT resolve to "Province Of Bombay vs Kusaldas S Advani"
    (gotcha #9). That is correct, but a refusal for sloppy paraphrase would otherwise be
    indistinguishable from a refusal for outright fabrication, which is the thing we are
    actually trying to measure. `match` records which stage fired, and REFUSED carries the
    best party score it could find, so near-misses stay visible.
    """
    cited_norm = _norm(case_name(name))
    if len(cited_norm) >= 12:
        for doc_id, title in candidates.items():
            cand_norm = _norm(case_name(title))
            if cited_norm in cand_norm or cand_norm in cited_norm:
                return {"status": "CERTIFIED", "doc_id": doc_id, "match": "verbatim",
                        "score": 1.0}

    cited = normalize_parties(case_name(name))
    best_id, best_score = None, 0.0
    for doc_id, title in candidates.items():
        score = party_match_score(cited, normalize_parties(case_name(title)))
        if score > best_score:
            best_id, best_score = doc_id, score
    if best_score >= PARTY_MATCH_THRESHOLD:
        return {"status": "CERTIFIED", "doc_id": best_id, "match": "parties",
                "score": round(best_score, 2)}
    return {"status": "REFUSED", "reason": "does not resolve to any retrieved candidate",
            "score": round(best_score, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--k", type=int, default=20, help="candidates put in front of the model")
    ap.add_argument("--facts-chars", type=int, default=2000)
    ap.add_argument("--min-year", type=int, default=2000)
    ap.add_argument("--model", default="llama4-scout-17b")
    args = ap.parse_args()

    cites, years = load_edges()
    haystack = set(json.loads(MAP_PATH.read_text()))
    titles, texts = {}, {}
    queries = [d for d in haystack if years.get(d, 0) >= args.min_year
               and len(cites.get(d, set()) & haystack) >= 3]
    queries.sort(key=lambda d: (years[d], d))
    queries = queries[: args.n]
    wanted = set(queries)
    with CORPUS.open() as f:
        for line in f:
            d = json.loads(line)
            titles[d["doc_id"]] = d["title"]
            if d["doc_id"] in wanted:
                texts[d["doc_id"]] = d["text"]
    all_titles = dict(titles)
    print(f"{len(queries)} query cases, {len(all_titles):,} corpus titles; "
          f"building BM25 index...", flush=True)
    index = BM25Index()
    index.build(load_corpus(), verbose=False)
    client = get_client()

    arms = {"grounded": GROUNDED_PROMPT, "closed_book": CLOSED_PROMPT}
    results = {a: [] for a in arms}

    for n, qid in enumerate(queries, 1):
        facts = scrub_citations(texts[qid])[: args.facts_chars]
        truth = cites[qid] & haystack
        ranked = [d for d, _ in index.search(facts, args.k, before_year=years[qid],
                                             exclude={qid})]
        cand = {d: titles.get(d, "") for d in ranked}
        listing = "\n".join(f"- {case_name(titles.get(d, ''))}" for d in ranked)

        for arm, template in arms.items():
            prompt = template.format(facts=facts, candidates=listing)
            try:
                resp = client.chat.completions.create(
                    model=args.model, temperature=0, max_tokens=1200,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": prompt}])
                parsed = parse_json(resp.choices[0].message.content or "")
            except Exception as e:
                print(f"  [{n}] {arm} ERROR {type(e).__name__}", flush=True)
                continue
            checked = []
            for name in (parsed.get("authorities") or [])[:20]:
                if not isinstance(name, str) or not name.strip():
                    continue
                v = certify(name, cand)
                if v["status"] == "CERTIFIED":
                    v["really_cited"] = v["doc_id"] in truth
                else:
                    # A refusal has two very different meanings and reporting them together
                    # would be dishonest (gotcha #7): the citation may name a real case that
                    # simply was not in this brief, or it may name nothing that exists. Only
                    # the second is a fabrication signal. Re-resolve against every title in
                    # the corpus to tell them apart.
                    wide = certify(name, all_titles)
                    v["resolves_in_corpus"] = wide["status"] == "CERTIFIED"
                    if wide["status"] == "CERTIFIED":
                        v["corpus_doc_id"] = wide["doc_id"]
                        v["really_cited"] = wide["doc_id"] in truth
                checked.append({"cited": name, **v})
            results[arm].append({"doc_id": qid, "year": years[qid],
                                 "n_truth": len(truth), "authorities": checked,
                                 "irac_present": [k for k in ("issue", "rule", "application",
                                                              "conclusion") if parsed.get(k)]})
        if n % 5 == 0:
            print(f"  [{n}/{len(queries)}]", flush=True)

    print(f"\n=== PROOF-CARRYING ANSWERS ({len(queries)} cases, k={args.k}) ===")
    print(f"{'arm':13} {'proposed':>9} {'CERTIFIED':>10} {'REFUSED':>8} "
          f"{'refused-but-real':>17} {'refused-nonexistent':>20} {'grounded':>9} "
          f"{'cert. really cited':>19}")
    summary = {}
    for arm, rows in results.items():
        c = Counter()
        cert_correct = real_but_unbriefed = nonexistent = 0
        for r in rows:
            for a in r["authorities"]:
                c[a["status"]] += 1
                if a["status"] == "CERTIFIED":
                    cert_correct += bool(a.get("really_cited"))
                elif a.get("resolves_in_corpus"):
                    real_but_unbriefed += 1
                else:
                    nonexistent += 1
        total = sum(c.values())
        cert = c["CERTIFIED"]
        summary[arm] = {
            "proposed": total, "certified": cert, "refused": c["REFUSED"],
            "refused_but_real_case": real_but_unbriefed,
            "refused_resolves_to_nothing": nonexistent,
            "groundedness": round(cert / total, 4) if total else None,
            "certified_really_cited": cert_correct,
            "precision_of_certified": round(cert_correct / cert, 4) if cert else None,
            "fabrication_rate": round(nonexistent / total, 4) if total else None,
        }
        print(f"{arm:13} {total:>9} {cert:>10} {c['REFUSED']:>8} "
              f"{real_but_unbriefed:>17} {nonexistent:>20} "
              f"{(cert/total if total else 0):>8.1%} "
              f"{(cert_correct/cert if cert else 0):>18.1%}")

    OUT.write_text(json.dumps({"queries": len(queries), "k": args.k, "model": args.model,
                               "facts_chars": args.facts_chars, "summary": summary,
                               "per_case": results}, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
