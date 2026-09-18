"""Split citation-graph edges into case-to-case citations vs statute/article references.

The provided citations_*.json files list every outbound Indian Kanoon link found in a
judgment under "precedents". That mixes citations to other judgments with citations to
statutes, articles, and sections (Indian Kanoon links both to the same URL scheme), so
before this becomes the signed legal knowledge graph (Phase II), we split the two edge
types apart. This is a first-pass heuristic classifier; expected to be refined later via
the residual-driven refinement loop (misclassified samples reveal missing patterns).

Usage: python3 scripts/classify_citations.py
Output: Data/processed/citations_classified.jsonl (one edge per line)
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CITATIONS_DIR = ROOT / "Data" / "Legal AI Dataset" / "citations"
OUT_DIR = ROOT / "Data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Trailing \b after an optional "." breaks when the abbreviation is followed by a space
# ("S. 4" -> the "." then " " has no word boundary between them), and plurals ("articles")
# don't match a singular-only alternative. Both are fixed with s? + a non-word lookahead.
# "s"/"S." is handled separately and requires a following digit, since a bare single letter
# followed by anything else is usually a person's initial ("S. Radhakrishnan"), not "Section".
_STATUTE_LONG = (
    r"\b(art|article|sec|section|clause|cl|rule|order|regulation|reg|schedule|para|"
    r"paragraph|chapter|ch|act|constitution|ordinance|code)s?\.?(?=\s|$|[0-9(),])"
)
_STATUTE_S_SHORTHAND = r"\bs\.?\s*(?=[0-9])"
STATUTE_PATTERN = re.compile(f"(?:{_STATUTE_LONG})|(?:{_STATUTE_S_SHORTHAND})", re.IGNORECASE)
CASE_PATTERN = re.compile(r"\bv(?:s)?\.?\s", re.IGNORECASE)


def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("\n", " ").replace("\t", " ")).strip()


def doc_id_from_url(url: str) -> str:
    if "/doc/" not in url:
        return ""
    return url.rstrip("/").split("/doc/")[-1]


def classify(name: str) -> str:
    has_case_marker = bool(CASE_PATTERN.search(name))
    has_statute_marker = bool(STATUTE_PATTERN.search(name))
    if has_case_marker and len(name) >= 15 and not (has_statute_marker and len(name) < 25):
        return "case"
    if has_statute_marker:
        return "statute"
    return "other"


def main():
    files = sorted(CITATIONS_DIR.glob("citations*.json"))
    if not files:
        raise SystemExit(f"No citation files found under {CITATIONS_DIR}")

    edges = []
    counts = {"case": 0, "statute": 0, "other": 0}
    seen_citing = set()

    for f in files:
        records = json.loads(f.read_text())
        for rec in records:
            citing_case = clean_text(rec.get("case", ""))
            citing_url = rec.get("url", "")
            citing_year = rec.get("year", "")
            seen_citing.add(doc_id_from_url(citing_url))
            for p in rec.get("precedents", []):
                cited_text = clean_text(p.get("case", ""))
                cited_url = p.get("url", "")
                edge_type = classify(cited_text)
                counts[edge_type] += 1
                edges.append(
                    {
                        "citing_doc_id": doc_id_from_url(citing_url),
                        "citing_case": citing_case,
                        "citing_url": citing_url,
                        "citing_year": citing_year,
                        "cited_doc_id": doc_id_from_url(cited_url),
                        "cited_text": cited_text,
                        "cited_url": cited_url,
                        "edge_type": edge_type,
                    }
                )

    out_path = OUT_DIR / "citations_classified.jsonl"
    with out_path.open("w") as out:
        for e in edges:
            out.write(json.dumps(e) + "\n")

    total = sum(counts.values())
    print(f"Processed {len(files)} citation files, {len(seen_citing)} unique citing cases")
    print(f"Total edges: {total}")
    for k, v in counts.items():
        print(f"  {k:8s}: {v:6d} ({100 * v / total:.1f}%)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
