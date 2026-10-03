"""Stage 0 (§4): discover what the dataset actually contains and emit one case registry.

The dataset arrives as three disjoint pieces that nothing joins for us:

  1. `filtered_land_disputes.xlsx` -- 6,954 land/property cases with a `filename` column.
  2. `Data/raw_pdfs/<year>/*.PDF`  -- all 26,688 judgment PDFs.
  3. `citations/*.json`            -- citation metadata for 7,497 cases, keyed by
                                      Indian Kanoon URL, listing cited "precedents".

The join that matters is (1) -> (2). The spreadsheet names the file directly, so this is
an EXACT match on a normalised filename -- not the fuzzy title match that
the legacy `build_doc_id_map.py` needed for the general corpus (and which silently produced a
0.6% join rate once). We assert the rate here so a regression cannot pass unnoticed.

Note that citation "precedents" mix real case citations with statute and article
references -- only about a third are case-to-case. That is Stage 5's problem (§9.1); this
module just records the raw list and a count.

Output: Data/interim/case_registry.jsonl, one CaseRecord per land-dispute case.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

from src import paths

# The spreadsheet is the authority on which cases are land/property disputes.
# A join rate below this means the filename convention changed; fail rather than
# silently emit a registry nothing downstream can trust.
MIN_PDF_JOIN_RATE = 0.99

_DOC_ID_RE = re.compile(r"/doc/(\d+)")


def _norm_filename(name: str) -> str:
    return unicodedata.normalize("NFC", str(name).strip())


def doc_id_from_url(url: str | None) -> str | None:
    """Indian Kanoon URLs carry the doc_id: .../doc/1780900/ -> '1780900'."""
    if not url:
        return None
    m = _DOC_ID_RE.search(str(url))
    return m.group(1) if m else None


@dataclass
class CaseRecord:
    doc_id: str
    title: str            # from the filename, which is the only title the xlsx carries
    year: int
    pdf_path: str         # relative to repo root
    url: str
    # citation metadata, absent for most cases
    cited_raw: list[dict] = field(default_factory=list)
    n_cited_raw: int = 0
    has_citation_metadata: bool = False


def _index_pdfs_on_disk() -> dict[str, Path]:
    """Map normalised filename -> path. Includes a lowercased key as fallback, because
    the spreadsheet and the archive disagree on .PDF vs .pdf for some entries."""
    index: dict[str, Path] = {}
    for p in paths.RAW_PDFS.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".pdf":
            n = _norm_filename(p.name)
            index.setdefault(n, p)
            index.setdefault(n.lower(), p)
    return index


def _load_land_rows() -> list[dict]:
    import openpyxl

    wb = openpyxl.load_workbook(paths.LAND_XLSX, read_only=True)
    ws = wb.active
    rows = ws.iter_rows(values_only=True)
    header = [str(h).strip() if h is not None else "" for h in next(rows)]
    out = []
    for r in rows:
        d = dict(zip(header, r))
        if d.get("filename"):
            out.append(d)
    wb.close()
    return out


def _load_citation_metadata() -> dict[str, list[dict]]:
    """doc_id -> raw precedent list, merged across the 10 citation JSON files."""
    cites: dict[str, list[dict]] = {}
    for jf in sorted(paths.CITATIONS_DIR.glob("*.json")):
        try:
            entries = json.loads(jf.read_text())
        except json.JSONDecodeError:
            continue
        for e in entries if isinstance(entries, list) else []:
            did = doc_id_from_url(e.get("url"))
            if did and did not in cites:
                cites[did] = e.get("precedents") or []
    return cites


def _title_from_filename(fn: str) -> str:
    """'Abdulla_Ahmed_vs_Animendra_Kissen_Mitter_on_14_March_1950_1.PDF' ->
    'Abdulla Ahmed vs Animendra Kissen Mitter on 14 March 1950'.

    The trailing '_1' is a copy marker, not part of the title; stripping it is only safe
    AFTER the date, which is why the year is matched explicitly (CLAUDE.md gotcha #15
    is exactly the bug of stripping trailing digits without anchoring on the year)."""
    stem = re.sub(r"\.pdf$", "", str(fn), flags=re.I)
    stem = re.sub(r"(_on_\d{1,2}_[A-Za-z]+_\d{4})_\d+$", r"\1", stem)
    return stem.replace("_", " ").strip()


def build(strict: bool = True) -> list[CaseRecord]:
    land = _load_land_rows()
    disk = _index_pdfs_on_disk()
    cites = _load_citation_metadata()

    records: list[CaseRecord] = []
    stats = Counter()
    unresolved: list[str] = []
    seen: set[str] = set()

    for row in land:
        fn = _norm_filename(row["filename"])
        p = disk.get(fn) or disk.get(fn.lower())
        if p is None:
            stats["pdf_unresolved"] += 1
            unresolved.append(fn)
            continue
        stats["pdf_resolved"] += 1

        doc_id = str(row.get("doc id") or "").strip()
        if doc_id.endswith(".0"):
            doc_id = doc_id[:-2]
        if not doc_id:
            doc_id = doc_id_from_url(row.get("url")) or ""
        if not doc_id:
            stats["no_doc_id"] += 1
            continue
        if doc_id in seen:
            stats["duplicate_doc_id"] += 1
            continue
        seen.add(doc_id)

        year_raw = str(row.get("year") or "").strip()
        year = int(year_raw) if year_raw.isdigit() else 0
        if not year:
            stats["no_year"] += 1

        cited = cites.get(doc_id, [])
        if cited:
            stats["with_citation_metadata"] += 1

        records.append(CaseRecord(
            doc_id=doc_id,
            title=_title_from_filename(fn),
            year=year,
            pdf_path=str(p.relative_to(paths.ROOT)),
            url=str(row.get("url") or ""),
            cited_raw=cited,
            n_cited_raw=len(cited),
            has_citation_metadata=bool(cited),
        ))

    total = stats["pdf_resolved"] + stats["pdf_unresolved"]
    rate = stats["pdf_resolved"] / total if total else 0.0
    if strict and rate < MIN_PDF_JOIN_RATE:
        raise AssertionError(
            f"filename->PDF join rate {rate:.3%} is below the {MIN_PDF_JOIN_RATE:.0%} floor "
            f"({stats['pdf_unresolved']} of {total} unresolved). The archive or the "
            f"spreadsheet's filename convention changed; fix the join before trusting "
            f"anything downstream. First unresolved: {unresolved[:3]}"
        )
    return records, stats, rate


def load() -> list[CaseRecord]:
    """Read the registry back. Everything downstream uses this, not the xlsx."""
    out = []
    with paths.CASE_REGISTRY.open() as f:
        for line in f:
            out.append(CaseRecord(**json.loads(line)))
    return out


def main() -> None:
    records, stats, rate = build()
    with paths.CASE_REGISTRY.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")

    by_decade = Counter(r.year // 10 * 10 for r in records if r.year)
    with_cites = sum(1 for r in records if r.has_citation_metadata)
    print(f"case registry: {len(records)} land/property cases -> {paths.CASE_REGISTRY}")
    print(f"  filename->PDF join rate: {rate:.2%} (floor {MIN_PDF_JOIN_RATE:.0%})")
    print(f"  with citation metadata:  {with_cites} ({100*with_cites/len(records):.1f}%)")
    print(f"  raw cited entries total: {sum(r.n_cited_raw for r in records):,}")
    print("  by decade: " + ", ".join(f"{d}s={n}" for d, n in sorted(by_decade.items())))
    odd = {k: v for k, v in stats.items() if k not in ("pdf_resolved", "with_citation_metadata")}
    if odd:
        print(f"  anomalies: {dict(odd)}")


if __name__ == "__main__":
    main()
