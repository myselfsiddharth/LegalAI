"""Map Indian Kanoon doc_ids to judgment PDFs on disk, by normalizing case titles.

The corpus is 26,684 PDFs named after their case title; the citation metadata keys
everything by doc_id and carries the same title as free text. Nothing in the dataset joins
the two for the full corpus (the land-disputes spreadsheet covers only its own 6,954 rows),
so the join is done here on a normalized title key.

Two normalization traps, both found by calibrating against the 500 verified doc_id/filename
pairs in sample_cases.csv:
  - filenames carry a copy marker after the year ("..._on_23_May_1957_1"), titles do not,
    so stripping "trailing digits" from both silently removes the YEAR from titles and
    every pair then mismatches by exactly that year;
  - punctuation differs freely ("Madras.Union" vs "Madras_Union"), so both sides collapse
    to alphanumeric tokens.

Calibration is asserted at run time: if the rate against those 500 pairs drops, the
normalizer has regressed and the join is not trustworthy.

Usage: python3 scripts/build_doc_id_map.py
"""
import csv
import json
import re
import sys
from pathlib import Path

from ode_lib import ROOT

PDF_DIR = ROOT / "Data" / "raw_pdfs"
CITATIONS = ROOT / "Data" / "processed" / "citations_classified.jsonl"
SAMPLE = ROOT / "Data" / "processed" / "sample_cases.csv"
OUT = ROOT / "Data" / "processed" / "doc_id_to_pdf.json"

MIN_CALIBRATION = 0.90


def title_key(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"((?:19|20)\d{2})\s+\d{1,2}$", r"\1", s)


def load_titles() -> dict[str, str]:
    titles = {}
    with CITATIONS.open() as f:
        for line in f:
            e = json.loads(line)
            titles.setdefault(e["citing_doc_id"], e["citing_case"])
    return titles


def main() -> None:
    titles = load_titles()
    pdfs: dict[str, Path] = {}
    for p in PDF_DIR.rglob("*.PDF"):
        pdfs.setdefault(title_key(p.stem), p)

    with SAMPLE.open() as f:
        truth = {r["doc_id"]: r["filename"] for r in csv.DictReader(f)}
    checkable = [(d, fn) for d, fn in truth.items() if d in titles]
    hits = sum(1 for d, fn in checkable
               if title_key(fn.replace(".PDF", "")) == title_key(titles[d]))
    rate = hits / len(checkable) if checkable else 0.0
    print(f"calibration against verified pairs: {hits}/{len(checkable)} ({rate:.1%})")
    if rate < MIN_CALIBRATION:
        sys.exit(f"normalizer regressed below {MIN_CALIBRATION:.0%} -- refusing to write a "
                 f"join nothing downstream could trust")

    mapping = {d: str(pdfs[title_key(t)].relative_to(ROOT))
               for d, t in titles.items() if title_key(t) in pdfs}
    OUT.write_text(json.dumps(mapping, indent=0))
    print(f"{len(titles)} titles in metadata, {len(mapping)} located on disk "
          f"({len(mapping)/len(titles):.1%}) -> {OUT}")


if __name__ == "__main__":
    main()
