"""Run Phase I extraction + Phase III verification across many cases and aggregate real
numbers -- this is what turns "we built a verification mechanism" into an actual metric
for the proposal's Evaluation Plan (fabricated/unverifiable-authority rate).

Cost control: each case grounds up to --per-case-limit candidate sentences (already
keyword-pre-filtered, see phase3_verify.candidate_sentences), so total LLM calls scale
as roughly N cases x per-case-limit, not full document length.

Verification is re-run over every case's saved AuthorityCited fillers on each run, so a
verifier fix never needs new LLM calls: --rescore-only re-scores existing outputs.

Usage:
  python3 scripts/run_evaluation_batch.py --n 50                   # first 50 rows (all 1950s!)
  python3 scripts/run_evaluation_batch.py --per-decade 6 --resume  # 6 per decade, skip done cases
  python3 scripts/run_evaluation_batch.py --n 50 --rescore-only --summary Data/processed/evaluation_summary_1950s.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from ode_lib import ROOT, get_client, load_ontology, load_rules, extract_concepts
from phase3_verify import load_real_outbound_edges, candidate_sentences
from eval_lib import (add_selection_args, select_cases, selection_description, judgment_text,
                      score_authorities, authority_metrics, metrics_by_decade)

OUT_DIR = ROOT / "Data" / "processed" / "irac_batch"
SUMMARY_PATH = ROOT / "Data" / "processed" / "evaluation_summary.json"


def run_case(row, text, client, model, ontology, rule_lookup, per_case_limit) -> dict:
    candidates = candidate_sentences(text, ontology)[:per_case_limit]
    objects_by_concept = defaultdict(list)
    for sentence in candidates:
        for obj in extract_concepts(client, model, sentence, ontology, rule_lookup):
            objects_by_concept[obj["concept"]].append(obj)
    return {
        "doc_id": row["doc_id"], "filename": row["filename"], "year": row["year"],
        "candidate_sentences": len(candidates),
        "objects_by_concept": {k: len(v) for k, v in objects_by_concept.items()},
        "authority_results": [{"cited": o["filler"], "source_sentence": o["source_sentence"]}
                              for o in objects_by_concept.get("AuthorityCited", [])],
    }


def main():
    parser = argparse.ArgumentParser()
    add_selection_args(parser)
    parser.add_argument("--per-case-limit", type=int, default=30, help="max candidate sentences per case")
    parser.add_argument("--model", default="llama4-scout-17b")
    parser.add_argument("--resume", action="store_true",
                        help="reuse existing per-case outputs instead of re-extracting them")
    parser.add_argument("--rescore-only", action="store_true",
                        help="no LLM calls: re-verify existing per-case outputs, skip cases without one")
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    rows = select_cases(args)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = ontology = rule_lookup = None
    if not args.rescore_only:
        ontology, rule_lookup, client = load_ontology(), load_rules(), get_client()

    concept_totals = Counter()
    per_case, failed = [], []
    for i, row in enumerate(rows):
        doc_id, filename = row["doc_id"], row["filename"]
        out_path = OUT_DIR / f"{doc_id}.json"
        text = judgment_text(row)
        if text is None:
            print(f"[{i+1}/{len(rows)}] SKIP (no PDF) {filename}")
            continue

        if out_path.exists() and (args.resume or args.rescore_only):
            case, how = json.loads(out_path.read_text()), "cached"
        elif args.rescore_only:
            print(f"[{i+1}/{len(rows)}] SKIP (not run yet) {filename}")
            continue
        else:
            try:
                case, how = run_case(row, text, client, args.model, ontology, rule_lookup,
                                     args.per_case_limit), "extracted"
            except Exception as e:  # one hung/failed request shouldn't cost the whole batch
                print(f"[{i+1}/{len(rows)}] FAILED {filename}: {e!r} -- re-run with --resume to retry")
                failed.append(filename)
                continue

        real_edges = load_real_outbound_edges(doc_id)
        case["decade"] = row["decade"]
        case["n_real_edges"] = len(real_edges)
        case["authority_results"] = score_authorities(case["authority_results"], row, text, real_edges)
        out_path.write_text(json.dumps(case, indent=2))
        per_case.append(case)
        concept_totals.update(case["objects_by_concept"])

        n_verified = sum(a["verification"]["status"] == "VERIFIED" for a in case["authority_results"])
        print(f"[{i+1}/{len(rows)}] {how:9s} {filename[:50]:50s} "
              f"authorities={len(case['authority_results']):2d} verified={n_verified}", flush=True)

    samples = defaultdict(list)  # a few of each non-verified status, for manual spot-checks
    for case in per_case:
        for a in case["authority_results"]:
            status = a["verification"]["status"]
            if status != "VERIFIED" and len(samples[status]) < 25:
                samples[status].append({"case": case["filename"], "cited": a["cited"],
                                        "in_judgment_text": a.get("in_judgment_text"),
                                        "source_sentence": a.get("source_sentence")})

    summary = {
        "selection": selection_description(args),
        "model": args.model,
        "per_case_limit": args.per_case_limit,
        "failed_cases": failed,
        "concept_totals": dict(concept_totals),
        "authority_metrics": authority_metrics(per_case),
        "authority_metrics_by_decade": metrics_by_decade(per_case),
        "samples": dict(samples),
    }
    args.summary.write_text(json.dumps(summary, indent=2))

    m = summary["authority_metrics"]
    print(f"\n=== SUMMARY ({m['cases']} cases{', ' + str(len(failed)) + ' FAILED' if failed else ''}) ===")
    print("Objects extracted by concept:", dict(concept_totals))
    print("Authority status:", m["status_counts"])
    print(f"Verified rate among case-shaped citations: {m['verified_rate']}  "
          f"edge recall: {m['edge_recall']}")
    print(f"\nWrote {args.summary}")


if __name__ == "__main__":
    main()
