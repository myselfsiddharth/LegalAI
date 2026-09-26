"""Extract and cache judgment text for every case we can link to a doc_id.

Everything downstream of the facts-to-authorities inversion reads text repeatedly --
fact segmentation, citation scrubbing, building the retrieval index -- and re-parsing
PDFs each time dominates the runtime. doc_id_to_pdf.json is built by
build_doc_id_map.py; this turns it into one JSONL of {doc_id, title, year, text}.

Resumable: existing doc_ids in the output are skipped, so an interrupted run continues.
"""
import json
import sys
from pathlib import Path

from pypdf import PdfReader

from ode_lib import ROOT

MAP_PATH = ROOT / "Data" / "processed" / "doc_id_to_pdf.json"
OUT_PATH = ROOT / "Data" / "processed" / "corpus_text.jsonl"
TITLES_PATH = ROOT / "Data" / "processed" / "citations_classified.jsonl"


def main() -> None:
    mapping = json.loads(MAP_PATH.read_text())
    titles = {}
    with TITLES_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            titles.setdefault(e["citing_doc_id"], (e["citing_case"], e["citing_year"]))

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except json.JSONDecodeError:
                    pass
    print(f"{len(mapping)} mapped cases, {len(done)} already cached", flush=True)

    written = failed = 0
    with OUT_PATH.open("a") as out:
        for i, (doc_id, path) in enumerate(sorted(mapping.items()), 1):
            if doc_id in done:
                continue
            try:
                text = "\n".join((p.extract_text() or "") for p in PdfReader(ROOT / path).pages)
            except Exception as e:
                failed += 1
                print(f"  FAIL {doc_id}: {type(e).__name__}", flush=True)
                continue
            title, year = titles.get(doc_id, ("", ""))
            out.write(json.dumps({"doc_id": doc_id, "title": title, "year": year,
                                  "n_chars": len(text), "text": text}) + "\n")
            written += 1
            if written % 250 == 0:
                out.flush()
                print(f"  cached {written} (at {i}/{len(mapping)})", flush=True)
    print(f"done: wrote {written}, failed {failed} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
