"""Audit the extracted fact set before anything is built on it.

A 20-case benchmark says a model works on 20 cases. This checks the properties that must hold
across all of them, and reports them **by decade**, because the corpus spans 75 years of changing
drafting conventions and a model can be fine on modern judgments and poor on 1950s ones without any
aggregate number moving.

Checks, in order of how badly a failure would matter:

1. **Every `source_span` round-trips.** `masked_text[start:end]` must match the stored quote under
   the grounding normalisation. A span that does not index its own source breaks §11's trace and
   would be invisible in any accuracy metric.
2. **Split coverage.** Which test/dev/train cases have facts. This is the number that limited every
   §10 result to n=143, so it is reported per split, not in aggregate.
3. **Attribution and admittedness by decade.** §8.1 discards a fact that reaches neither party
   view; if that rate is era-dependent, the temporal split inherits an artefact.
4. **Zero-fact and thin cases.** A case with no facts is either genuinely fact-free or a call that
   never returned; `facts.py --only-missing` re-runs them and the difference is the answer.
5. **Duplicate quotes within a case**, which would inflate atom support even though transactions
   are sets.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict

from src import grounding, paths
from src.extract.fact_filters import classify_content, orientation_coverage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-spans", type=int, default=4000,
                    help="how many spans to round-trip check (0 = all)")
    args = ap.parse_args()

    facts = [json.loads(l) for l in open(paths.INTERIM / "facts.jsonl")]
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    by_case = defaultdict(list)
    for f in facts:
        by_case[f["case_id"]].append(f)

    print(f"=== fact set QA: {len(facts):,} facts over {len(by_case):,} cases ===\n")

    # 1. span integrity
    check = facts if not args.sample_spans else facts[::max(1, len(facts) // args.sample_spans)]
    bad = []
    for f in check:
        src = masked.get(f["case_id"])
        if src is None:
            bad.append((f["fact_id"], "case has no masked text"))
            continue
        s, e = f["source_span"]
        if grounding.normalize(src[s:e]) != grounding.normalize(f["quote"]):
            bad.append((f["fact_id"], "span does not match its quote"))
    print(f"1. source_span round-trip: {len(check) - len(bad):,}/{len(check):,} exact")
    if bad:
        print(f"   FAILURES: {len(bad)}")
        for fid, why in bad[:5]:
            print(f"     {fid}: {why}")
    else:
        print("   no failures")

    # 2. split coverage -- the constraint behind n=143
    print("\n2. split coverage (cases with >=1 fact):")
    for sp in sorted(paths.SPLITS.glob("*.json")):
        d = json.loads(sp.read_text())
        print(f"   {sp.stem}")
        for part in ("train", "dev", "test"):
            ids = d.get(part, [])
            have = sum(1 for c in ids if by_case.get(c))
            bar = "#" * int(30 * have / max(1, len(ids)))
            print(f"     {part:5s} {have:5d}/{len(ids):5d}  {100*have/max(1,len(ids)):5.1f}%  {bar}")

    # 3. by decade
    print("\n3. by decade:")
    years = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        years[r["doc_id"]] = r["year"]
    dec = defaultdict(list)
    for cid, fs in by_case.items():
        dec[(years.get(cid, 0) // 10) * 10].append(fs)
    hdr = (f"   {'decade':8s} {'cases':>6s} {'facts':>7s} {'f/case':>7s} {'attrib':>7s} "
           f"{'admitted':>9s} {'orphan':>7s} {'law leak':>9s} {'dates':>6s}")
    print(hdr); print("   " + "-" * (len(hdr) - 3))
    for d in sorted(dec):
        groups = dec[d]
        flat = [f for g in groups for f in g]
        cov = orientation_coverage(flat)
        leak = sum(1 for f in flat if classify_content(f["text"]) != "fact") / max(1, len(flat))
        dates = sum(1 for f in flat if f.get("event_date")) / max(1, len(flat))
        print(f"   {d}s{'':3s} {len(groups):6d} {len(flat):7d} "
              f"{len(flat)/max(1,len(groups)):7.1f} {cov['attributed_frac']:7.2f} "
              f"{cov['admitted_frac']:9.2f} {cov['orphan_frac']:7.2f} {leak:9.3f} {dates:6.2f}")

    allcov = orientation_coverage(facts)
    print(f"\n   overall: attributed {allcov['attributed_frac']:.2f}, "
          f"admitted {allcov['admitted_frac']:.2f}, "
          f"usable in a party view {allcov['oriented_frac']:.2f}, "
          f"orphaned {allcov['orphan_frac']:.2f}")

    # 4. thin and zero-fact cases
    counts = [len(v) for v in by_case.values()]
    eligible_with_text = sum(1 for c in masked if masked[c] and len(masked[c]) >= 1000)
    print(f"\n4. facts per case: median {statistics.median(counts):.0f}, "
          f"mean {statistics.fmean(counts):.1f}, min {min(counts)}, max {max(counts)}")
    thin = sum(1 for c in counts if c <= 2)
    print(f"   cases with <=2 facts: {thin} ({100*thin/len(counts):.1f}%) "
          f"-- re-run with `facts.py --only-missing` to tell fact-free from never-answered")

    # 5. duplicate quotes inside one case
    dup_cases = 0
    dup_facts = 0
    for cid, fs in by_case.items():
        qs = Counter(grounding.normalize(f["quote"]) for f in fs)
        d = sum(v - 1 for v in qs.values() if v > 1)
        if d:
            dup_cases += 1
            dup_facts += d
    print(f"\n5. duplicate quotes within a case: {dup_facts} facts across {dup_cases} cases "
          f"({100*dup_facts/len(facts):.2f}% of all facts)")

    ok = not bad
    print(f"\nQA verdict: {'PASS' if ok else 'FAIL -- spans do not index their source'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
