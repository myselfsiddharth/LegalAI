"""Build text, labels and masks for ALL 26,688 Supreme Court judgments, not just the land subset.

## Why this exists

The leakage finding (`src/eval/leakage_setback.py`) is measured on 6,954 land/property disputes.
The obvious strengthening step was to replicate it on ILDC itself -- 34,816 Indian Supreme Court
proceedings, the corpus the field actually cites. That release is gated, the licence request is
sitting in a review queue, and it may never be approved.

It is also not necessary. ILDC is drawn from Indian Kanoon's Supreme Court judgments; so is this
project's corpus, and all 26,688 of them are already on disk. The land/property subset is a
restriction WE imposed via the provided spreadsheet, not a limit of the data. Dropping it gives:

  - **~4x the scale**, comparable to ILDC's 34,816;
  - **every case type** -- criminal, tax, constitutional, service, company -- instead of one.
    This is a STRONGER generalisation test than ILDC in one respect, because it asks whether the
    contamination is a property of Indian judicial writing or an artifact of land-dispute style;
  - **no licence, no gate, no dependency on anyone answering an email.**

What ILDC uniquely offers is its own labels and its own deletion boundary, which this cannot
reproduce. So this is a generalisation test, not a replication, and it is labelled as such.

## What it does, and what it costs

Three deterministic, API-free stages per judgment, run in parallel:

  1. text      `src/data/text.extract_one` -- the same extractor stage 0 uses
  2. label     `src/data/labels.label_by_rules` -- no LLM. See the caveat below.
  3. mask      `src/data/mask.mask_case` -- the same §5.2 masking

Output is deliberately compact: only what the setback analysis reads (label, year,
`masked_text`, `ildc_style_text`), not the full judgment text, or the file would be gigabytes.

## Two caveats, both stated rather than buried

**Labels are rules-only.** The main corpus merges a rules labeller with an LLM labeller and holds
conflicts as UNRESOLVED; agreement is 87.0%. Running the LLM pass over 26,688 cases is possible but
slow, and it is not needed here: **label noise attenuates measured effects.** A contamination gap
found despite noisier labels is a conservative estimate, not an inflated one. Cases the rules
labeller cannot decide are dropped rather than guessed.

**The split is temporal, and cannot be group-aware.** Without the spreadsheet there is no case-group
metadata, so connected matters (an appeal and its later review) may straddle the split. This inflates
every arm equally, and the quantity measured here is the DIFFERENCE between arms at matched word
counts, which is robust to a shared inflation. Absolute AUROCs from this file should not be compared
against the main corpus's.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

from src import paths
from src.data.labels import label_by_rules
from src.data.mask import mask_case
from src.data.text import extract_one

OUT = paths.INTERIM / "corpus_wide.jsonl"
MIN_CHARS = 500
KEEP = {"WIN", "LOSE"}


def _year_from_dir(p: Path) -> int:
    try:
        return int(p.parent.name)
    except ValueError:
        return 0


def one(pdf: str) -> dict | None:
    """Extract, label and mask a single judgment. Returns None for anything unusable."""
    p = Path(pdf)
    try:
        text, _, _ = extract_one(p)
    except Exception:
        return {"skip": "unreadable_pdf"}
    if len(text.strip()) < MIN_CHARS:
        return {"skip": "near_empty"}                 # scanned / image-only
    year = _year_from_dir(p)
    lab = label_by_rules(text)
    if lab.initiator_outcome not in KEEP:
        return {"skip": f"label_{lab.initiator_outcome}"}
    try:
        mc = mask_case({"doc_id": p.stem, "year": year, "text": text, "n_chars": len(text)})
    except Exception:
        return {"skip": "mask_failed"}
    if not mc.masked_text.strip() or not mc.ildc_style_text.strip():
        return {"skip": "empty_after_mask"}
    return {"doc_id": p.stem, "year": year, "label": 1 if lab.initiator_outcome == "WIN" else 0,
            "tier": lab.disposition_tier, "ambiguous": lab.ambiguous,
            "masked_text": mc.masked_text, "ildc_style_text": mc.ildc_style_text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="for a quick smoke run")
    args = ap.parse_args()

    pdfs = sorted(str(p) for p in (paths.DATA / "raw_pdfs").rglob("*")
                  if p.suffix.lower() == ".pdf")
    if args.limit:
        pdfs = pdfs[::max(1, len(pdfs) // args.limit)][: args.limit]   # spread across years

    done: set[str] = set()
    if OUT.exists():
        with OUT.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["doc_id"])
                except json.JSONDecodeError:
                    pass
    todo = [p for p in pdfs if Path(p).stem not in done]
    print(f"{len(pdfs):,} judgments on disk, {len(done):,} already built, {len(todo):,} to do, "
          f"{args.workers} workers", flush=True)

    t0, kept, skips = time.time(), 0, {}
    with OUT.open("a") as out, Pool(args.workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(one, todo, chunksize=8), 1):
            if rec is None or "skip" in (rec or {}):
                k = (rec or {}).get("skip", "none")
                skips[k] = skips.get(k, 0) + 1
            else:
                out.write(json.dumps(rec) + "\n")
                kept += 1
            if i % 1000 == 0:
                out.flush()
                el = time.time() - t0
                print(f"  {i:,}/{len(todo):,}  kept {kept:,}  ({i/el:.1f}/s, "
                      f"eta {(len(todo)-i)/(i/el)/60:.0f} min)", flush=True)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min: kept {kept:,} usable WIN/LOSE cases")
    for k, v in sorted(skips.items(), key=lambda x: -x[1]):
        print(f"  skipped {k}: {v:,}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
