"""Build a stratified sample of land-dispute cases to actually work with this semester.

Combines the land-dispute index (6,954 cases) with the classified citation graph
(scripts/classify_citations.py output) to select cases that are BOTH confirmed land
disputes AND have real case-to-case citation data (needed for the signed graph in
Phase II and the verification ground truth in Phase III). Samples are drawn per decade,
preferring cases with more outbound case citations first (richer graph signal), so the
sample isn't accidentally dominated by one era.

Usage: python3 scripts/build_sample.py [--target 500]
Requires: scripts/classify_citations.py has already been run.
Output: Data/processed/sample_cases.csv
"""

import argparse
import csv
import json
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / "Data" / "Legal AI Dataset" / "filtered_land_disputes.xlsx"
CLASSIFIED_PATH = ROOT / "Data" / "processed" / "citations_classified.jsonl"
OUT_PATH = ROOT / "Data" / "processed" / "sample_cases.csv"

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def read_land_disputes_xlsx(path: Path):
    """Minimal .xlsx reader (no openpyxl available): walks the raw sheet XML."""
    z = zipfile.ZipFile(path)
    with z.open("xl/worksheets/sheet1.xml") as f:
        root = ET.parse(f).getroot()
    rows = root.find("s:sheetData", NS).findall("s:row", NS)

    def cell_text(c):
        if c.get("t") == "inlineStr":
            is_el = c.find("s:is", NS)
            t_el = is_el.find("s:t", NS) if is_el is not None else None
            return t_el.text if t_el is not None else ""
        v = c.find("s:v", NS)
        return v.text if v is not None else ""

    records = []
    for row in rows[1:]:  # skip header
        cells = {c.get("r")[0]: cell_text(c) for c in row.findall("s:c", NS)}
        if not cells.get("E"):
            continue
        records.append(
            {
                "filename": cells.get("A", ""),
                "year": cells.get("B", ""),
                "drive_path": cells.get("C", ""),
                "indiankanoon_url": cells.get("D", ""),
                "doc_id": cells.get("E", ""),
            }
        )
    return records


def load_case_citation_counts(path: Path) -> Counter:
    """doc_id -> number of outbound edges classified as real case citations."""
    counts = Counter()
    with path.open() as f:
        for line in f:
            e = json.loads(line)
            if e["edge_type"] == "case" and e["citing_doc_id"]:
                counts[e["citing_doc_id"]] += 1
    return counts


def decade_of(year_str: str) -> int:
    return int(year_str) // 10 * 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=500, help="total sample size")
    args = parser.parse_args()

    if not CLASSIFIED_PATH.exists():
        raise SystemExit(f"Missing {CLASSIFIED_PATH} -- run scripts/classify_citations.py first")

    land_disputes = read_land_disputes_xlsx(XLSX_PATH)
    case_citation_counts = load_case_citation_counts(CLASSIFIED_PATH)

    eligible = []
    for rec in land_disputes:
        n_citations = case_citation_counts.get(rec["doc_id"], 0)
        if n_citations > 0:
            rec["case_citation_count"] = n_citations
            rec["decade"] = decade_of(rec["year"])
            eligible.append(rec)

    print(f"Land-dispute cases total: {len(land_disputes)}")
    print(f"Eligible (land dispute + has case citations): {len(eligible)}")

    by_decade = defaultdict(list)
    for rec in eligible:
        by_decade[rec["decade"]].append(rec)
    for decade in by_decade:
        by_decade[decade].sort(key=lambda r: r["case_citation_count"], reverse=True)

    decades = sorted(by_decade)
    n_decades = len(decades)
    base_quota = args.target // n_decades
    sample = []
    for decade in decades:
        pool = by_decade[decade]
        take = min(base_quota, len(pool))
        sample.extend(pool[:take])

    # Any shortfall (a decade had fewer eligible cases than its quota) gets filled from
    # whichever decades still have leftover cases, richest-cited first.
    shortfall = args.target - len(sample)
    if shortfall > 0:
        leftovers = []
        for decade in decades:
            already_taken = min(base_quota, len(by_decade[decade]))
            leftovers.extend(by_decade[decade][already_taken:])
        leftovers.sort(key=lambda r: r["case_citation_count"], reverse=True)
        sample.extend(leftovers[:shortfall])

    with OUT_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "year",
                "decade",
                "doc_id",
                "drive_path",
                "indiankanoon_url",
                "case_citation_count",
            ],
        )
        writer.writeheader()
        for rec in sorted(sample, key=lambda r: (r["decade"], -r["case_citation_count"])):
            writer.writerow(rec)

    print(f"\nSample size: {len(sample)}")
    print("By decade:")
    sample_by_decade = Counter(r["decade"] for r in sample)
    for decade in decades:
        print(f"  {decade}s: {sample_by_decade.get(decade, 0):4d} (of {len(by_decade[decade])} eligible)")
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
