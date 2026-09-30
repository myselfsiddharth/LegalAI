"""Stage 0 (§4): cache judgment text for every registry case, once.

Everything downstream reads text repeatedly -- masking, section classification, fact
extraction, retrieval indexing -- and re-parsing 6,954 PDFs each time dominates runtime.

Two things this keeps that a naive `"".join(page.extract_text())` throws away, both
required by §6.1's `source_span` contract:

  * `page_offsets` -- char offset where each page starts, so a span can name its page.
  * `para_offsets` -- char offset of each numbered paragraph marker ("12. Thereafter...").
    Indian judgments have no section headers (measured: `FACTS` 29.7%, `ORDER` 3.7%), but
    post-2000 judgments carry a median of 32 numbered paragraphs and 85% have >= 5. The
    numbered paragraph is therefore the segmentation unit for masking (§5.2) and the
    addressable unit for a trace (§11).

Resumable: doc_ids already in the output are skipped, so a killed run loses at most the
record in flight. Safe to re-run.
"""
from __future__ import annotations

import json
import re
import sys
import time

from pypdf import PdfReader

from src import paths
from src.data import adapter

# A numbered paragraph marker at the start of a line: "12. The appellant ..."
# Requires the following char to be uppercase/quote so it does not fire on "12. 5 acres".
PARA_MARKER = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]+(?=[\"'A-Z(])", re.M)


def extract_one(pdf_path) -> tuple[str, list[int], list[dict]]:
    reader = PdfReader(pdf_path)
    chunks, page_offsets, pos = [], [], 0
    for page in reader.pages:
        page_offsets.append(pos)
        t = page.extract_text() or ""
        chunks.append(t)
        pos += len(t) + 1  # +1 for the "\n" join below
    text = "\n".join(chunks)
    paras = [{"num": int(m.group(1)), "start": m.start()} for m in PARA_MARKER.finditer(text)]
    return text, page_offsets, paras


def main() -> None:
    records = adapter.load()
    done: set[str] = set()
    if paths.CASE_TEXT.exists():
        with paths.CASE_TEXT.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except json.JSONDecodeError:
                    pass

    todo = [r for r in records if r.doc_id not in done]
    print(f"{len(records)} registry cases, {len(done)} already cached, {len(todo)} to do",
          flush=True)

    t0 = time.time()
    written = failed = empty = 0
    with paths.CASE_TEXT.open("a") as out:
        for i, rec in enumerate(todo, 1):
            try:
                text, page_offsets, paras = extract_one(paths.ROOT / rec.pdf_path)
            except Exception as e:                      # a corrupt PDF must not kill the run
                failed += 1
                print(f"  FAIL {rec.doc_id}: {type(e).__name__}: {e}", flush=True)
                continue
            if len(text.strip()) < 500:                 # scanned/image-only judgment
                empty += 1
            out.write(json.dumps({
                "doc_id": rec.doc_id,
                "title": rec.title,
                "year": rec.year,
                "n_chars": len(text),
                "n_pages": len(page_offsets),
                "page_offsets": page_offsets,
                "para_offsets": paras,
                "text": text,
            }) + "\n")
            written += 1
            if written % 200 == 0:
                out.flush()
                el = time.time() - t0
                rate = written / el
                print(f"  {written}/{len(todo)} cached  ({rate:.1f}/s, "
                      f"eta {(len(todo)-written)/rate/60:.1f} min)", flush=True)

    print(f"done: wrote {written}, failed {failed}, near-empty {empty} -> {paths.CASE_TEXT}",
          flush=True)


if __name__ == "__main__":
    main()
