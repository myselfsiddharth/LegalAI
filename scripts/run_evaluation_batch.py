"""Run Phase I extraction + Phase III verification across many cases and aggregate real
numbers -- this is what turns "we built a verification mechanism" into an actual metric
for the proposal's Evaluation Plan (fabricated/unverifiable-authority rate).

Cost control: each case grounds up to --per-case-limit candidate sentences (already
keyword-pre-filtered, see phase3_verify.candidate_sentences), so total LLM calls scale
as roughly N cases x per-case-limit, not full document length.

Usage:
  python3 scripts/run_evaluation_batch.py --n 50              # first 50 sample cases
  python3 scripts/run_evaluation_batch.py --n 50 --per-case-limit 30
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from pypdf import PdfReader

from ode_lib import ROOT, get_client, load_ontology, load_rules, extract_concepts
from phase3_verify import (
    load_real_outbound_edges,
    candidate_sentences,
    verify_authority,
    PDF_DIR,
)

SAMPLE_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"
OUT_DIR = ROOT / "Data" / "processed" / "irac_batch"
SUMMARY_PATH = ROOT / "Data" / "processed" / "evaluation_summary.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50, help="number of cases to process")
    parser.add_argument("--per-case-limit", type=int, default=30, help="max candidate sentences per case")
    parser.add_argument("--model", default="llama4-scout-17b")
    parser.add_argument("--seed-offset", type=int, default=0, help="skip this many rows first (for a different slice)")
    args = parser.parse_args()

    with SAMPLE_PATH.open() as f:
        rows = list(csv.DictReader(f))
    rows = rows[args.seed_offset: args.seed_offset + args.n]

    ontology = load_ontology()
    rule_lookup = load_rules()
    client = get_client()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    concept_totals = Counter()
    authority_status_totals = Counter()
    per_case_summaries = []
    unverified_samples = []  # keep a few for manual spot-check
    rejected_samples = []

    for i, row in enumerate(rows):
        doc_id, filename, year = row["doc_id"], row["filename"], row["year"]
        pdf_path = PDF_DIR / year / filename
        if not pdf_path.exists():
            print(f"[{i+1}/{len(rows)}] SKIP (no PDF) {filename}")
            continue

        real_edges = load_real_outbound_edges(doc_id)
        text = "\n".join((p.extract_text() or "") for p in PdfReader(str(pdf_path)).pages)
        candidates = candidate_sentences(text, ontology)[: args.per_case_limit]

        objects_by_concept = defaultdict(list)
        for sentence in candidates:
            for obj in extract_concepts(client, args.model, sentence, ontology, rule_lookup):
                objects_by_concept[obj["concept"]].append(obj)

        for concept, objs in objects_by_concept.items():
            concept_totals[concept] += len(objs)

        case_authority_results = []
        for obj in objects_by_concept.get("AuthorityCited", []):
            verification = verify_authority(obj["filler"], real_edges)
            authority_status_totals[verification["status"]] += 1
            case_authority_results.append({"cited": obj["filler"], "verification": verification})
            if verification["status"] == "UNVERIFIED" and len(unverified_samples) < 15:
                unverified_samples.append({"case": filename, "cited": obj["filler"],
                                            "source_sentence": obj["source_sentence"]})
            if verification["status"] == "REJECTED_NOT_A_CASE" and len(rejected_samples) < 15:
                rejected_samples.append({"case": filename, "cited": obj["filler"]})

        case_summary = {
            "doc_id": doc_id, "filename": filename, "year": year,
            "candidate_sentences": len(candidates),
            "objects_by_concept": {k: len(v) for k, v in objects_by_concept.items()},
            "authority_results": case_authority_results,
        }
        per_case_summaries.append(case_summary)
        (OUT_DIR / f"{doc_id}.json").write_text(json.dumps(case_summary, indent=2))

        n_verified = sum(1 for a in case_authority_results if a["verification"]["status"] == "VERIFIED")
        print(f"[{i+1}/{len(rows)}] {filename[:50]:50s} "
              f"candidates={len(candidates):3d} authorities={len(case_authority_results):2d} verified={n_verified}")

    total_authorities = sum(authority_status_totals.values())
    summary = {
        "cases_processed": len(per_case_summaries),
        "concept_totals": dict(concept_totals),
        "authority_status_totals": dict(authority_status_totals),
        "authority_verification_rate_among_plausible_case_names": (
            round(authority_status_totals["VERIFIED"] /
                  max(1, authority_status_totals["VERIFIED"] + authority_status_totals["UNVERIFIED"]), 3)
        ),
        "unverified_samples": unverified_samples,
        "rejected_not_a_case_samples": rejected_samples,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(f"\n=== SUMMARY ({len(per_case_summaries)} cases) ===")
    print("Objects extracted by concept:", dict(concept_totals))
    print("Authority verification status:", dict(authority_status_totals))
    print(f"Verification rate among plausible case-name candidates: "
          f"{summary['authority_verification_rate_among_plausible_case_names']:.1%}")
    print(f"\nWrote {SUMMARY_PATH}")
    print(f"Wrote {len(per_case_summaries)} per-case reports to {OUT_DIR}/")


if __name__ == "__main__":
    main()
