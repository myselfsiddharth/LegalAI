"""Extract signed citation edges directly from the HEADNOTE of each judgment PDF.

Indian Supreme Court judgments (as scraped from Indian Kanoon) include a HEADNOTE
section that explicitly states how the court treated each authority it cites: grouped
lists of "Case v. Case (Citation)" followed by a verb such as "relied on", "followed",
"distinguished", "disapproved", "overruled", "approved", "explained", or "referred to".
This is a much stronger and more direct signal for the signed legal knowledge graph
(Phase II) than inferring polarity with an LLM classifier -- it's already in the text.

This is a first-pass rule-based extractor (the "Lift" step of the ODE MVP recipe).
Known limitations to refine later via residual review:
  - PDF page breaks inject repeated header/footer text mid-sentence; the cleanup here
    strips the exact case title, but a citation split across a page boundary right at
    the title's own line-wrap point can still get mangled and silently dropped.
  - The verb-to-sign mapping is a coarse heuristic (see SIGN_POLARITY below).

Usage: python3 scripts/extract_headnote_signs.py
Output: Data/processed/headnote_signed_citations.jsonl
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "Data" / "raw_pdfs"
OUT_PATH = ROOT / "Data" / "processed" / "headnote_signed_citations.jsonl"

BOILERPLATE = re.compile(r"Indian Kanoon - http://indiankanoon\.org/doc/\d+/\s*\d*")

SIGN_VERBS = [
    "disapproved", "distinguished", "relied on", "referred to", "referred",
    "followed", "approved", "overruled", "explained", "doubted",
]
VERB_PATTERN = re.compile(r"\b(" + "|".join(SIGN_VERBS) + r")\b\.?", re.IGNORECASE)

CASE_CITE_PATTERN = re.compile(
    r"([A-Z][A-Za-z.&'-]*(?:\s+[A-Za-z.&'-]+)*\s+v\.?\s+[A-Za-z.&'-]+(?:\s+[A-Za-z.&'-]+)*)"
    r"\s*\(([^)]+)\)"
)

# Coarse polarity for the signed graph. "referred to" is treated as neutral/citing-only
# (mentioned, not endorsed or rejected); "explained"/"doubted" sit in between and are
# flagged rather than forced into a bucket.
SIGN_POLARITY = {
    "relied on": "positive",
    "followed": "positive",
    "approved": "positive",
    "distinguished": "negative",
    "disapproved": "negative",
    "overruled": "negative",
    "referred to": "neutral",
    "referred": "neutral",
    "explained": "neutral",
    "doubted": "negative",
}


def clean(text: str, case_title: str = "") -> str:
    text = BOILERPLATE.sub(" ", text)
    if case_title:
        text = text.replace(case_title, " ")
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # de-hyphenate line-wraps
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_headnote(raw_text: str) -> str | None:
    hn_idx = raw_text.find("HEADNOTE:")
    if hn_idx == -1:
        return None
    jm_indices = [m.start() for m in re.finditer(r"JUDGMENT:", raw_text)]
    after = [i for i in jm_indices if i > hn_idx]
    end_idx = after[0] if after else len(raw_text)
    return raw_text[hn_idx:end_idx]


def process_pdf(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    raw = "\n".join((p.extract_text() or "") for p in reader.pages)
    title = raw.split("\n")[0].strip()
    doc_id_match = re.search(r"doc/(\d+)/", raw)
    citing_doc_id = doc_id_match.group(1) if doc_id_match else None

    headnote = extract_headnote(raw)
    if not headnote:
        return []
    headnote = clean(headnote, case_title=title)

    edges = []
    last_end = 0
    for m in VERB_PATTERN.finditer(headnote):
        segment = headnote[last_end:m.start()]
        verb = m.group(1).lower()
        for name, cite in CASE_CITE_PATTERN.findall(segment):
            edges.append(
                {
                    "citing_case": title,
                    "citing_doc_id": citing_doc_id,
                    "cited_case": re.sub(r"\s+", " ", name).strip(),
                    "cited_citation": re.sub(r"\s+", " ", cite).strip(),
                    "verb": verb,
                    "polarity": SIGN_POLARITY.get(verb, "neutral"),
                }
            )
        last_end = m.end()
    return edges


def main():
    pdf_paths = sorted(PDF_DIR.rglob("*.PDF"))
    if not pdf_paths:
        raise SystemExit(f"No PDFs found under {PDF_DIR} -- run fetch_sample_pdfs.py first")

    all_edges = []
    docs_with_headnote_signs = 0
    for pdf_path in pdf_paths:
        edges = process_pdf(pdf_path)
        if edges:
            docs_with_headnote_signs += 1
        all_edges.extend(edges)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for e in all_edges:
            f.write(json.dumps(e) + "\n")

    from collections import Counter

    polarity_counts = Counter(e["polarity"] for e in all_edges)
    print(f"PDFs processed: {len(pdf_paths)}")
    print(f"PDFs with >=1 extracted headnote sign: {docs_with_headnote_signs} "
          f"({100 * docs_with_headnote_signs / len(pdf_paths):.0f}%)")
    print(f"Total signed edges extracted: {len(all_edges)}")
    for polarity, count in polarity_counts.items():
        print(f"  {polarity:8s}: {count}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
