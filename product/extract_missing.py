"""Text for the citing judgments the good-law layer could not read.

treatment.py located only 5,077 of 10,360 citations of pool cases: 3,683 edges come from 1,218 citing
judgments with no text in case_text.jsonl or corpus_wide.jsonl. Their PDFs are on disk under
Data/raw_pdfs/<year>/. This joins each citing title to a filename and extracts the text.

The join: title "X vs Y on 9 December, 1954" and filename "X_vs_Y_on_9_December_1954_1.PDF" are
both reduced to lowercase alphanumerics, with the filename's trailing copy marker ("_1") removed
BEFORE normalisation -- stripping trailing digits after normalisation removes the year and silently
breaks every join (gotcha #15). The join rate is asserted; below the floor the module refuses to
write.
"""
from __future__ import annotations

import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

from src import paths

OUT = paths.ROOT / "product" / "data" / "extra_text.jsonl"
MIN_JOIN = 0.80


def key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def extract(path: str) -> str:
    from pypdf import PdfReader
    try:
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    except Exception:
        return ""


def main() -> None:
    have = set()
    for fn in ("case_text.jsonl", "corpus_wide.jsonl"):
        for line in open(paths.INTERIM / fn):
            have.add(json.loads(line)["doc_id"])
    pool = {json.loads(l)["doc_id"] for l in open(paths.INTERIM / "case_text.jsonl")}
    miss = {}
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        if r.get("edge_type") == "case" and r.get("cited_doc_id") in pool and r["citing_doc_id"] not in have:
            miss[r["citing_doc_id"]] = (r["citing_case"], str(r.get("citing_year") or ""))

    files = {}
    for root, _, fs in os.walk("Data/raw_pdfs"):
        for f in fs:
            stem = re.sub(r"_\d+$", "", os.path.splitext(f)[0])     # copy marker, before normalising
            files.setdefault((os.path.basename(root), key(stem)), os.path.join(root, f))
    joined = {}
    for doc, (title, year) in miss.items():
        p = files.get((year, key(title)))
        if p is None:   # Indian Kanoon titles truncate long party lists with "..."; try the prefix
            pre = key(title.split("...")[0])
            cands = [v for (y, k), v in files.items() if y == year and k.startswith(pre) and len(pre) > 12]
            p = cands[0] if len(cands) == 1 else None
        if p:
            joined[doc] = p
    rate = len(joined) / max(1, len(miss))
    print(f"{len(miss):,} citing judgments without text; joined {len(joined):,} to a PDF ({rate:.1%})")
    if rate < MIN_JOIN:
        raise SystemExit(f"join rate {rate:.1%} below the {MIN_JOIN:.0%} floor -- refusing to write")

    docs = list(joined)
    with ProcessPoolExecutor(max_workers=8) as pool_:
        texts = list(pool_.map(extract, [joined[d] for d in docs], chunksize=8))
    ok = sum(1 for t in texts if len(t) > 1000)
    with OUT.open("w") as fh:
        for d, t in zip(docs, texts):
            if len(t) > 1000:
                fh.write(json.dumps({"doc_id": d, "text": t}) + "\n")
    print(f"extracted {ok:,}/{len(docs):,} with usable text -> {OUT}")


if __name__ == "__main__":
    main()
