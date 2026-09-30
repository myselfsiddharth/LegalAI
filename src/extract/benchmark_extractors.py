"""Choose the extraction model by measurement, not by size or reputation.

There is no gold fact annotation yet (§6.4 wants >=150 hand-annotated cases), so this scores each
model on properties that can be checked **without ground truth** and that each correspond to a
failure we have actually seen:

  `grounding_rate`     share of proposed facts whose quote verifies verbatim in the excerpt.
                       This is the fabrication measure: an unverifiable quote is not evidence.
  `legal_leak_rate`    share rejected as statements of law or about an authority. The prompt
                       forbids these and scout emitted them anyway; they would make FP-Growth
                       encode the statute book rather than the case.
  `attribution_rate`   share attributed to plaintiff or defendant rather than left as
                       `court_narrative`. §8.1 builds party-oriented views from this, and scout
                       attributed only 19%.
  `admitted_rate`      share marked admitted in `disputed_status`, which is what actually
                       carries admittedness (see fact_filters).
  `facts_per_case`     recall proxy. High is not automatically better -- a model that emits 60
                       near-duplicates scores well here and badly on `distinct_atom_rate`.
  `dup_rate`           share deduplicated across the chunk overlap: a model repeating itself.
  `unparseable` / `dropped`  kept separate, always. A dropped call is an outage, not an answer.

`--ladder` then runs the decisive test on the winner candidates: does the extracted fact TEXT
still predict the outcome? That is the only measure here tied to downstream utility, and the
representation ladder showed it is exactly where the pipeline can silently lose everything.

All models see the SAME cases, so differences are the model.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from src import paths
from src.extract.facts import extract_case
from src.llm import client

OUT = paths.EXPERIMENTS / "paper1" / "extractor_benchmark.json"


def bench_model(model: str, cases: list[dict], titles: dict, workers: int) -> dict:
    t0 = time.time()
    all_facts, total = [], Counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for facts, st in pool.map(
                lambda c: extract_case(c, model, titles.get(c["doc_id"], "")), cases):
            all_facts.extend(facts)
            total.update(st)
    elapsed = time.time() - t0

    prop = total["proposed"] or 1
    n = len(all_facts) or 1
    ab = Counter(f.asserted_by for f in all_facts)
    ds = Counter(f.disputed_status for f in all_facts)
    per_case = Counter(f.case_id for f in all_facts)
    return {
        "model": model,
        "n_cases": len(cases),
        "proposed": total["proposed"],
        "kept": total["kept"],
        "grounding_rate": round(1 - total["discarded_quote_not_located"] / prop, 4),
        "legal_leak_rate": round(total["discarded_legal_statement"] / prop, 4),
        "dup_rate": round(total["deduped"] / prop, 4),
        "facts_per_case": round(len(all_facts) / max(1, len(cases)), 2),
        "cases_with_zero_facts": sum(1 for c in cases if per_case[c["doc_id"]] == 0),
        "attribution_rate": round((ab["plaintiff"] + ab["defendant"]) / n, 4),
        "admitted_rate": round(ds["admitted"] / n, 4),
        "contested_rate": round(ds["contested"] / n, 4),
        "distinct_atom_rate": round(len({f.text.lower() for f in all_facts}) / n, 4),
        "with_event_date": round(sum(1 for f in all_facts if f.event_date) / n, 4),
        "chunks_ok": total["chunks_ok"],
        "chunks_empty": total["chunks_empty"],
        "chunks_unparseable": total["chunks_unparseable"],
        "chunks_DROPPED": total["chunks_dropped"],
        "seconds": round(elapsed, 1),
        "sec_per_case": round(elapsed / max(1, len(cases)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    from src.data.label_merge import load_final
    labels = load_final()
    screen = {json.loads(l)["doc_id"]: json.loads(l)
              for l in open(paths.INTERIM / "case_screen.jsonl")}
    titles = {json.loads(l)["doc_id"]: json.loads(l)["title"]
              for l in open(paths.CASE_REGISTRY)}
    eligible = {d for d, r in labels.items()
                if r["outcome"] in ("WIN", "LOSE")
                and screen.get(d, {}).get("is_property", False)}

    cases = [json.loads(l) for l in open(paths.INTERIM / "masked_text.jsonl")]
    cases = [c for c in cases if c["doc_id"] in eligible and c["n_chars_masked"] >= 1000]
    # Stratify by decade so one model is not judged on a different era's drafting style.
    cases.sort(key=lambda r: (r["year"], r["doc_id"]))
    step = max(1, len(cases) // args.n)
    cases = cases[::step][:args.n]
    print(f"benchmark set: {len(cases)} cases, "
          f"years {min(c['year'] for c in cases)}-{max(c['year'] for c in cases)}\n", flush=True)

    rows = []
    for m in args.models:
        print(f"-- {m}", flush=True)
        try:
            r = bench_model(m, cases, titles, args.workers)
        except Exception as e:                                     # noqa: BLE001
            print(f"   EXCEPTION {type(e).__name__}: {e}", flush=True)
            continue
        rows.append(r)
        print(f"   kept {r['kept']:4d}/{r['proposed']:4d}  ground {r['grounding_rate']:.2f}  "
              f"leak {r['legal_leak_rate']:.2f}  attrib {r['attribution_rate']:.2f}  "
              f"f/case {r['facts_per_case']:5.1f}  {r['sec_per_case']:.1f}s/case  "
              f"dropped {r['chunks_DROPPED']}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"n_cases": len(cases), "results": rows}, indent=1))

    if rows:
        hdr = (f"\n{'model':32s} {'kept':>5s} {'ground':>7s} {'leak':>6s} {'attrib':>7s} "
               f"{'adm':>5s} {'f/case':>7s} {'distinct':>9s} {'dates':>6s} {'s/case':>7s} "
               f"{'drop':>5s}")
        print(hdr); print("-" * len(hdr))
        for r in sorted(rows, key=lambda x: -x["grounding_rate"]):
            print(f"{r['model']:32s} {r['kept']:5d} {r['grounding_rate']:7.3f} "
                  f"{r['legal_leak_rate']:6.3f} {r['attribution_rate']:7.3f} "
                  f"{r['admitted_rate']:5.2f} {r['facts_per_case']:7.1f} "
                  f"{r['distinct_atom_rate']:9.3f} {r['with_event_date']:6.2f} "
                  f"{r['sec_per_case']:7.2f} {r['chunks_DROPPED']:5d}")
        print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
