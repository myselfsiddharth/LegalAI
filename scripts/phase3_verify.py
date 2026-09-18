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

MATCH_THRESHOLD = 0.6  # same threshold used in build_signed_graph.py for consistency


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
# so the filter accepts either signal -- a v./vs. pattern, OR a citation-reporter-style
# parenthetical (multiple capital letters, with or without periods, e.g. "I.L.R.", "SCC").
CASE_NAME_OR_CITATION_PATTERN = re.compile(
    r"\bv\.?\s|\bvs\.?\s|\bversus\b|\([^)]*(?:(?:[A-Z]\.){2,}|[A-Z]{2,})[^)]*\)",
    re.IGNORECASE,
)


def looks_like_case_name(text: str) -> bool:
    return bool(CASE_NAME_OR_CITATION_PATTERN.search(text)) and len(text) >= 8


def verify_authority(cited_text: str, real_edges: list[dict]) -> dict:
    """Check a proposed AuthorityCited against this document's real outbound edges."""
    if not looks_like_case_name(cited_text):
        return {"status": "REJECTED_NOT_A_CASE",
                "reason": "filler doesn't look like a case name (no v./vs.) -- likely a "
                          "grounding false positive, not a genuine citation candidate"}

    # Strip a trailing citation parenthetical for substring comparison, so a truncated
    # extraction like "Gilbert Pinto (I.L.R. 42 Mad. 654)" -- missing "Krishna Shetti v."
    # from the multi-citation-list bug -- still matches "Krishna Shetti v. Gilbert Pinto"
    # in the ground truth via plain containment, not just fuzzy ratio.
    core = re.sub(r"\([^)]*\)", "", cited_text).strip().lower()

    best_edge, best_score = None, 0.0
    for edge in real_edges:
        edge_text = edge["cited_text"].lower()
        score = SequenceMatcher(None, cited_text.lower(), edge_text).ratio()
        if core and len(core) >= 6 and (core in edge_text or edge_text in cited_text.lower()):
            score = max(score, 0.75)
        if score > best_score:
            best_edge, best_score = edge, score
    if best_edge and best_score >= MATCH_THRESHOLD:
        return {"status": "VERIFIED", "match_score": round(best_score, 2),
                "matched_cited_doc_id": best_edge["cited_doc_id"],
                "matched_cited_text": best_edge["cited_text"]}
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
        obj["verification"] = verify_authority(obj["filler"], real_edges)

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
