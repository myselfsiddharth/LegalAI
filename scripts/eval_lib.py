"""Shared evaluation code: case selection, judgment loading, authority scoring, and metric
aggregation. Factored out so the LexGraph pipeline (run_evaluation_batch.py) and the
baselines (run_baselines.py) are selected and scored by exactly the same code -- a
baseline comparison is only meaningful if every system's authorities go through the same
verify_authority() and the same metric definitions."""

import csv
import re
from collections import Counter, defaultdict

from pypdf import PdfReader

from ode_lib import ROOT
from phase3_verify import (PDF_DIR, load_real_outbound_edges, normalize_parties,
                           title_from_filename, verify_authority, _content_tokens,
                           MIN_PARTY_CHARS)

SAMPLE_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"


def add_selection_args(parser) -> None:
    parser.add_argument("--n", type=int, default=50,
                        help="number of cases, taken in file order (NB: sample_cases.csv is sorted "
                             "by decade, so --n 50 is all 1950s -- use --per-decade for coverage)")
    parser.add_argument("--per-decade", type=int, default=None,
                        help="instead of --n: take the first K cases of every decade")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="skip this many rows (per decade with --per-decade) first")


def select_cases(args) -> list[dict]:
    with SAMPLE_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if args.per_decade is None:
        return rows[args.seed_offset: args.seed_offset + args.n]
    by_decade = defaultdict(list)
    for row in rows:
        by_decade[row["decade"]].append(row)
    return [row for decade in sorted(by_decade)
            for row in by_decade[decade][args.seed_offset: args.seed_offset + args.per_decade]]


def selection_description(args) -> dict:
    if args.per_decade is None:
        return {"mode": "first_n", "n": args.n, "offset": args.seed_offset}
    return {"mode": "per_decade", "per_decade": args.per_decade, "offset": args.seed_offset}


def pdf_path(row: dict):
    return PDF_DIR / row["year"] / row["filename"]


def judgment_text(row: dict) -> str | None:
    path = pdf_path(row)
    if not path.exists():
        return None
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _normalize_text(text: str) -> str:
    return " " + " ".join(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split()) + " "


def appears_in_text(cited: str, normalized_text: str) -> bool | None:
    """Does any usable party name of the citation occur verbatim (after normalization) in
    the judgment? Separates the two very different meanings of UNVERIFIED (gotcha #7):
    named in the judgment but missing from the metadata, vs. not in the judgment at all,
    which for a generating system is the fabrication signal. "Any party" is deliberately
    lenient -- it undercounts fabrication rather than overcounting it. None when no party
    name is long enough to check ("Das", "Ltd.") -- that's not evidence either way."""
    usable = [p for p in normalize_parties(cited) if len("".join(_content_tokens(p))) >= MIN_PARTY_CHARS]
    if not usable:
        return None
    return any(f" {party} " in normalized_text for party in usable)


def score_authorities(authorities: list[dict], row: dict, text: str | None,
                      real_edges: list[dict] | None = None) -> list[dict]:
    """(Re-)verify a list of {"cited": ..., ...} dicts with the current verifier. Other keys
    (e.g. source_sentence) are kept, so this is safe to run over saved results."""
    edges = real_edges if real_edges is not None else load_real_outbound_edges(row["doc_id"])
    own_title = title_from_filename(row["filename"])
    normalized = _normalize_text(text) if text else None
    scored = []
    for a in authorities:
        a = dict(a)
        a["verification"] = verify_authority(a["cited"], edges, own_title=own_title)
        if normalized is not None:
            a["in_judgment_text"] = appears_in_text(a["cited"], normalized)
        scored.append(a)
    return scored


STATUSES = ["VERIFIED", "UNVERIFIED", "SELF_REFERENCE", "REJECTED_NOT_A_CASE"]


def authority_metrics(cases: list[dict]) -> dict:
    """Metrics over per-case records ({"n_real_edges", "authority_results"}).

    - verified_rate = VERIFIED / (VERIFIED + UNVERIFIED): among outputs that look like case
      citations, how many this judgment really cites. Not a hallucination rate by itself --
      see unverified_* for the breakdown.
    - edge_recall: distinct real outbound case edges recovered / all real edges. Measures
      coverage, which precision alone hides (citing nothing scores a perfect 0 fabrications).
    """
    status = Counter()
    ambiguous = unverified_in_text = unverified_not_in_text = unverified_uncheckable = 0
    matched_edges = real_edges = cases_with_authorities = 0
    for case in cases:
        results = case["authority_results"]
        cases_with_authorities += bool(results)
        real_edges += case["n_real_edges"]
        matched = set()
        for a in results:
            v = a["verification"]
            status[v["status"]] += 1
            if v["status"] == "VERIFIED":
                matched.add(v["matched_cited_doc_id"])
                ambiguous += v.get("n_matching_edges", 1) > 1
            elif v["status"] == "UNVERIFIED" and "in_judgment_text" in a:
                if a["in_judgment_text"] is None:
                    unverified_uncheckable += 1
                elif a["in_judgment_text"]:
                    unverified_in_text += 1
                else:
                    unverified_not_in_text += 1
        matched_edges += len(matched)
    v, u = status["VERIFIED"], status["UNVERIFIED"]
    return {
        "cases": len(cases),
        "cases_with_any_authority": cases_with_authorities,
        "authority_outputs": sum(status.values()),
        "status_counts": {s: status[s] for s in STATUSES},
        "case_shaped_citations": v + u,
        "verified_rate": round(v / (v + u), 3) if v + u else None,
        "verified_ambiguous_party_match": ambiguous,
        "unverified_named_in_judgment": unverified_in_text,
        "unverified_not_in_judgment": unverified_not_in_text,
        "unverified_name_too_short_to_check": unverified_uncheckable,
        "real_edges_total": real_edges,
        "real_edges_recovered": matched_edges,
        "edge_recall": round(matched_edges / real_edges, 3) if real_edges else None,
    }


def metrics_by_decade(cases: list[dict]) -> dict:
    by_decade = defaultdict(list)
    for case in cases:
        by_decade[case["decade"]].append(case)
    return {f"{d}s": authority_metrics(c) for d, c in sorted(by_decade.items())}
