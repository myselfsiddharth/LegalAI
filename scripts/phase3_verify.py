"""Phase III -- Deterministic verification + IRAC output.

Runs Phase I extraction (GROUND + rule-matching) across a real judgment, then verifies
every AuthorityCited object against the citation ground truth (citations_classified.jsonl
-- the metadata provided for this project, not something the LLM produced) before
assembling an IRAC-style (Issue, Rule, Application, Conclusion) report. This is the
"proof-carrying" step the whole project is about: an authority that isn't actually a
real outbound citation of this document gets flagged UNVERIFIED rather than silently
included, which is a direct, checkable proxy for the fabricated-authority failure mode.

Running GROUND on every sentence of a full judgment (100-300+ sentences) would be slow
and mostly wasted, since our current rule set only fires on a handful of trigger
patterns. So this pre-filters to sentences containing at least one ontology terminology
keyword before spending an LLM call on them -- a cheap candidate-generation pass in
front of the expensive step, not a shortcut around it.

Usage: python3 scripts/phase3_verify.py --doc-id 168057026
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from pypdf import PdfReader

from ode_lib import ROOT, get_client, load_ontology, load_rules, extract_concepts

SAMPLE_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"
CITATIONS_PATH = ROOT / "Data" / "processed" / "citations_classified.jsonl"
PDF_DIR = ROOT / "Data" / "raw_pdfs"


def find_pdf_for_doc_id(doc_id: str) -> Path:
    with SAMPLE_PATH.open() as f:
        for row in csv.DictReader(f):
            if row["doc_id"] == doc_id:
                return PDF_DIR / row["year"] / row["filename"]
    raise SystemExit(f"doc_id {doc_id} not found in {SAMPLE_PATH}")


def load_real_outbound_edges(doc_id: str) -> list[dict]:
    """The actual case-citation edges this document has, per the provided metadata --
    the ground truth Phase III checks proposed authorities against."""
    edges = []
    with CITATIONS_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            if e["citing_doc_id"] == doc_id and e["edge_type"] == "case":
                edges.append(e)
    return edges


def candidate_sentences(text: str, ontology: dict) -> list[str]:
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", re.sub(r"\s+", " ", text))
    keywords = [k for k in ontology["terminology"] if not k.startswith("_")]
    keyword_pattern = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE)
    return [s.strip() for s in raw_sentences if 30 < len(s) < 400 and keyword_pattern.search(s)]


# Found via manual audit on Kedar Nath Yadav: "approved" is in the ontology as a
# TreatmentAction trigger for judicial endorsement ("X v. Y ... approved"), but the same
# word also means ordinary administrative sign-off ("approved by the Chief Minister").
# Grounding doesn't disambiguate by context, so it fired on "the Cabinet", "TML", "Rs. 90
# lakhs per annum" -- none of which are case citations. Rather than fix the grounding
# prompt (which needs more contrastive gold examples to do properly), filter cheaply here:
# an AuthorityCited candidate that doesn't look like a case name isn't a case citation,
# full stop, regardless of what triggered it.
#
# A second, different failure showed up at batch scale: when a headnote sentence lists
# several citations before a shared verb ("X v. Y (cite1), A v. B (cite2) relied on."),
# grounding sometimes drops the first party's name and the "v." connector, emitting just
# "Gilbert Pinto (I.L.R. 42 Mad. 654)" instead of "Krishna Shetti v. Gilbert Pinto (...)".
# That's still clearly a case reference (it has a reporter citation), just missing "v.",
# so the filter accepts either signal -- a v./vs. connector, OR a law-reporter citation.
#
# Two further bugs found auditing the first 50-case batch (all 1950s):
#  - Indian reporter citations usually sit OUTSIDE parentheses -- "Ayyappa Reddy, (1913)
#    I.L.R. 38 Mad. 738", "[1954] S.C.R. 177" -- so a parenthetical-only check rejected
#    ~25 genuine case citations as REJECTED_NOT_A_CASE. Search the whole string instead.
#  - The old pattern ran under IGNORECASE, which turned "[A-Z]{2,}" into "any two
#    letters" and accepted any parenthetical: "clause (iv)", "(Act XVIII of 1937)". Those
#    statute fragments then landed in UNVERIFIED and inflated the unverified count.
#    Reporter abbreviations are matched case-sensitively now.
CASE_CONNECTOR = re.compile(r"(?:\bv\.?|\bV\.|\b[Vv][Ss]\.?|\bversus)(?=\s|$)")
# Dotted capitals ("I.L.R.", "S.C.R.", "L.R. ... I.A.", "K.B.") or a known undotted modern
# reporter ("(2005) 3 SCC 123", "AIR 1950 SC 27"). The undotted list is explicit rather
# than "any capitals + digit", which would also accept "Section 302 IPC".
REPORTER = re.compile(r"(?:\b[A-Z]\.\s?){2,}|\b(?:SCC|SCR|AIR|SCALE|JT|ITR|MLJ|SCJ|KB|QB|AC|WLR)\b")


def looks_like_case_name(text: str) -> bool:
    if len(text) < 8:
        return False
    return bool(CASE_CONNECTOR.search(text)) or (
        bool(REPORTER.search(text)) and bool(re.search(r"\d", text)))


# Minimum normalized length for a party name to count as evidence. Shorter fragments
# ("Das", "Velu", "Davey") match too many unrelated parties to verify anything.
MIN_PARTY_CHARS = 6
PARTY_MATCH_THRESHOLD = 0.8  # per-token; "kisho"/"kesho" = 0.8, "bam"/"ram" = 0.67

_PARTY_SPLIT = re.compile(r"\s(?:v|vs|versus)\.?(?:\s|$)")
_NOISE_SUFFIX = re.compile(r"\s(?:and|&)\s(?:others?|anr|another|ors)$|\s(?:ors|anr)$")
# Reporter/abbreviation tokens left dangling after cutting at the first volume number:
# "Ayyappa Reddy, I.L.R." / "Charusila Dasi, I.L.R. I Cal." / "Sahi, SUPP."
_TRAILING_REPORTER = re.compile(
    r"(?:[,\s&]*\b(?:[A-Z]\.|[A-Z][a-z]{1,4}\.|[IVX]+\b|SUPP\.?|SCC\b|SCR\b|AIR\b|SC\b))+[,\s]*$")


def normalize_parties(text: str) -> list[str]:
    """Reduce a citation string to its party names, e.g.
    "Ayyappa Reddy, (1913) I.L.R. 38 Mad. 738" -> ["ayyappa reddy"],
    "1) and Amarendra Mansingh v. Sanatan Singh" -> ["amarendra mansingh", "sanatan singh"].
    Both extracted fillers and metadata edge texts go through this, so they're compared
    like-for-like -- the metadata has its own noise (footnote markers, "Vide", OCR)."""
    t = re.sub(r"^\s*(?:\d+\)\s*,?\s*)+(?:and\s+)?", "", text)   # "1) and ...", "2), (2) ..."
    t = re.sub(r"[\(\[][^)\]]*[\)\]]|[\(\[][^)\]]*$", " ", t)    # (1913), [1954], unclosed "(Civil..."
    t = re.split(r"\s\d", " " + t, maxsplit=1)[0]                # cut at reporter volume/page
    t = _TRAILING_REPORTER.sub("", t)
    t = re.sub(r"^\s*(?:(?:vide|in re|the)\s+)+", "", t.lower())
    parties = []
    for p in _PARTY_SPLIT.split(t):
        p = " ".join(re.sub(r"[^a-z0-9 ]", " ", p).split())
        p = re.sub(r"^the\s+", "", _NOISE_SUFFIX.sub("", p)).strip()
        if p:
            parties.append(p)
    return parties


# Connectives plus honorifics, which pad one side's party name but not the other's
# ("Tendolkar" vs "Shri Justice S. R. Tendolkar").
_STOPWORDS = {"of", "and", "the", "in", "re", "vide",
              "shri", "sri", "smt", "mst", "justice", "maharaja", "raja", "rajah"}
# Matched tokens must cover at least this share of the longer party's characters. Without
# it a one-word party is "contained" in any longer name sharing a word: "Pannalal v.
# Naraini" verified "Bhubneshwar Prasad Narain Singh" via naraini~narain alone.
MIN_COVERAGE = 0.5
JOINED_THRESHOLD = 0.9  # whole-string fallback for spacing variants ("Horilal"/"Hori Lal")


def _content_tokens(party: str) -> list[str]:
    return [t for t in party.split() if len(t) > 1 and t not in _STOPWORDS]  # drops initials


def _token_sim(a: str, b: str) -> float:
    if a == b:
        return 1.0
    # Line-break truncation on either side: "Mukho-" vs "Mukhopadhya", "Hori" vs "Horilal"
    if min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a)):
        return 1.0
    return SequenceMatcher(None, a, b).ratio()  # OCR variants: "Kisho"/"Kesho", "Bhusan"/"Bhu8an"


def _party_sim(p: str, q: str) -> float:
    """Every content token of the shorter party must have a close match in the longer one.
    Token-level (not a character ratio over the whole string) so that a shared generic
    prefix can't carry a mismatch: "Commissioner of Income-tax, Madras" must NOT match
    "Commissioner of Income-tax, Bihar and Orissa"."""
    short, long_ = sorted((_content_tokens(p), _content_tokens(q)), key=lambda t: len("".join(t)))
    if len("".join(short)) < MIN_PARTY_CHARS or not long_:
        return 0.0
    joined = SequenceMatcher(None, "".join(short), "".join(long_)).ratio()
    covered = sum(len(t) for t in long_
                  if max(_token_sim(s, t) for s in short) >= PARTY_MATCH_THRESHOLD)
    if covered / len("".join(long_)) < MIN_COVERAGE:
        return joined if joined >= JOINED_THRESHOLD else 0.0
    return max(min(max(_token_sim(s, t) for t in long_) for s in short),
               joined if joined >= JOINED_THRESHOLD else 0.0)


def party_match_score(cited_parties: list[str], edge_parties: list[str]) -> float:
    """Every usable cited party must match some edge party; score is the weakest link."""
    usable = [p for p in cited_parties if len("".join(_content_tokens(p))) >= MIN_PARTY_CHARS]
    if not usable or not edge_parties:
        return 0.0
    return min(max(_party_sim(p, e) for e in edge_parties) for p in usable)


def title_from_filename(filename: str) -> str:
    """"The_State_Of_X_vs_Y_on_17_December_1953_1.PDF" -> "The State Of X vs Y"."""
    return re.split(r"_on_\d", filename.replace(".PDF", "").replace(".pdf", ""))[0].replace("_", " ")


def verify_authority(cited_text: str, real_edges: list[dict], own_title: str | None = None) -> dict:
    """Check a proposed AuthorityCited against this document's real outbound edges.

    Matching is party-name based: the extracted string often has only one party (the
    multi-citation-list truncation above) or an OCR variant, and the metadata carries
    party names but no reporter citations. A match means "this judgment really does cite
    a case with this party", which is necessary but not sufficient for the specific case
    being the right one -- hence `n_matching_edges` so ambiguous matches are visible."""
    if not looks_like_case_name(cited_text):
        return {"status": "REJECTED_NOT_A_CASE",
                "reason": "no v./vs. connector or law-reporter citation -- likely a "
                          "grounding false positive, not a genuine citation candidate"}

    cited_parties = normalize_parties(cited_text)
    scored = sorted(((party_match_score(cited_parties, normalize_parties(e["cited_text"])), e)
                     for e in real_edges), key=lambda x: -x[0])
    matches = [(s, e) for s, e in scored if s >= PARTY_MATCH_THRESHOLD]
    if matches:
        best_score, best_edge = matches[0]
        return {"status": "VERIFIED", "match_score": round(best_score, 2),
                "matched_cited_doc_id": best_edge["cited_doc_id"],
                "matched_cited_text": best_edge["cited_text"],
                "n_matching_edges": len({e["cited_doc_id"] for _, e in matches})}

    # The judgment's own title (usually from a page header) is not a citation of it. Needs
    # both parties: a lone "State of Bombay" could just as well be an unmatched citation.
    if (own_title and len(cited_parties) >= 2 and
            party_match_score(cited_parties, normalize_parties(own_title)) >= PARTY_MATCH_THRESHOLD):
        return {"status": "SELF_REFERENCE",
                "reason": "matches the citing judgment's own title, not an outbound citation"}

    best_score = scored[0][0] if scored else 0.0
    return {"status": "UNVERIFIED", "match_score": round(best_score, 2),
            "reason": "no matching outbound edge in citation metadata for this document"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc-id", required=True)
    parser.add_argument("--limit", type=int, default=60, help="max candidate sentences to ground (cost control)")
    parser.add_argument("--model", default="llama4-scout-17b")
    args = parser.parse_args()

    pdf_path = find_pdf_for_doc_id(args.doc_id)
    real_edges = load_real_outbound_edges(args.doc_id)
    print(f"Case: {pdf_path.name}")
    print(f"Real outbound case citations in metadata: {len(real_edges)}")

    ontology = load_ontology()
    rule_lookup = load_rules()
    client = get_client()

    reader = PdfReader(str(pdf_path))
    text = "\n".join((p.extract_text() or "") for p in reader.pages)
    candidates = candidate_sentences(text, ontology)[: args.limit]
    print(f"Candidate sentences (keyword-filtered, capped at {args.limit}): {len(candidates)}")

    objects_by_concept = defaultdict(list)
    for sentence in candidates:
        for obj in extract_concepts(client, args.model, sentence, ontology, rule_lookup):
            objects_by_concept[obj["concept"]].append(obj)

    print("\nExtracted objects by concept:")
    for concept, objs in objects_by_concept.items():
        print(f"  {concept}: {len(objs)}")

    # Verify every AuthorityCited object against the real citation graph
    for obj in objects_by_concept.get("AuthorityCited", []):
        obj["verification"] = verify_authority(obj["filler"], real_edges,
                                               own_title=title_from_filename(pdf_path.name))

    report = {
        "case": pdf_path.name,
        "doc_id": args.doc_id,
        "Issue": [o["filler"] for o in objects_by_concept.get("Issue", [])],
        "Rule": [o["filler"] for o in objects_by_concept.get("RuleCited", [])],
        "Authority": [
            {"cited": o["filler"], "verification": o["verification"], "source_sentence": o["source_sentence"]}
            for o in objects_by_concept.get("AuthorityCited", [])
        ],
        "Application": {
            "Facts": [o["filler"] for o in objects_by_concept.get("Fact", [])],
            "Claims": [o["filler"] for o in objects_by_concept.get("Claim", [])],
        },
        "Conclusion": {
            "Holdings": [o["filler"] for o in objects_by_concept.get("Holding", [])],
            "Outcome": [o["filler"] for o in objects_by_concept.get("Outcome", [])],
        },
    }

    out_path = ROOT / "Data" / "processed" / f"irac_{args.doc_id}.json"
    out_path.write_text(json.dumps(report, indent=2))

    print("\n=== IRAC REPORT ===")
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")

    n_verified = sum(1 for a in report["Authority"] if a["verification"]["status"] == "VERIFIED")
    n_unverified = len(report["Authority"]) - n_verified
    print(f"\nAuthority verification: {n_verified} VERIFIED, {n_unverified} UNVERIFIED "
          f"(flagged, not silently dropped)")


if __name__ == "__main__":
    main()
