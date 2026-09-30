"""Stage 4 (§8.1): turn each case into party-oriented transactions of canonical labels.

A transaction is the set of canonical atoms for one case, and FP-Growth mines sets. Two
properties of this step decide whether §8.2 can find anything at all:

**Support is per CASE, not per fact.** An atom appearing 20 times in one judgment has support 1,
not 20. So transactions are SETS -- deduplicating within a case is not an optimisation, it is the
definition. Getting this wrong would let a verbose judgment manufacture support on its own.

**Two views per case, because the same fact cuts differently by party.** §8.1:

    T_plaintiff(c) = facts asserted by the plaintiff, or admitted
    T_defendant(c) = facts asserted by the defendant, or admitted

Admitted facts appear in BOTH. What distinguishes the views is the contested, attributed
material. Note the orientation reads `disputed_status` as well as `asserted_by` -- see
`fact_filters` for why reading `asserted_by` alone discarded ~80% of facts.

Atoms are suffixed with the view (`@p` / `@d`) when a combined transaction is built, so a rule can
say "the PLAINTIFF asserted continuous possession" rather than merely "continuous possession
appears somewhere in this case". §10.2 ablation 3 turns that suffixing off to measure what party
orientation is worth.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

from src import paths
from src.extract.fact_filters import in_defendant_view, in_plaintiff_view

CANON_PATH = paths.INTERIM / "canonical_facts.jsonl"
FAMILIES_PATH = paths.INTERIM / "claim_families.json"


def load_canonical() -> dict[str, list[dict]]:
    by_case: dict[str, list[dict]] = defaultdict(list)
    if not CANON_PATH.exists():
        return by_case
    with CANON_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            by_case[r["case_id"]].append(r)
    return by_case


def load_family_membership() -> dict[str, dict]:
    """case_id -> {families: [...], primary: str}. Empty if §7 has not run."""
    if not FAMILIES_PATH.exists():
        return {}
    return json.loads(FAMILIES_PATH.read_text()).get("membership", {})


def transactions_for_case(facts: list[dict], oriented: bool = True) -> dict[str, set[str]]:
    """Returns {'plaintiff': set, 'defendant': set, 'combined': set}."""
    p, d = set(), set()
    for f in facts:
        lab = f["label"]
        if in_plaintiff_view(f):
            p.add(lab)
        if in_defendant_view(f):
            d.add(lab)
    if oriented:
        combined = {f"{a}@p" for a in p} | {f"{a}@d" for a in d}
    else:
        combined = p | d
    return {"plaintiff": p, "defendant": d, "combined": combined}


def build(oriented: bool = True, min_atoms: int = 2):
    """-> (by_family, stats). by_family[family] = list of {case_id, plaintiff, defendant, combined}."""
    canon = load_canonical()
    members = load_family_membership()
    stats = Counter()
    by_family: dict[str, list[dict]] = defaultdict(list)

    for case_id, facts in canon.items():
        tx = transactions_for_case(facts, oriented=oriented)
        if len(tx["combined"]) < min_atoms:
            stats["skipped_too_few_atoms"] += 1
            continue
        rec = {"case_id": case_id, **{k: sorted(v) for k, v in tx.items()}}
        fams = members.get(case_id, {}).get("families")
        if not fams:
            stats["no_family"] += 1
            fams = ["_UNASSIGNED"]
        for fam in fams:
            by_family[fam].append(rec)
        stats["cases"] += 1
    return by_family, stats


def main() -> None:
    by_family, stats = build()
    out = paths.PATTERNS / "transactions.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in by_family.items()}, indent=1))

    print(f"transactions: {stats['cases']} cases -> {out}")
    if stats["skipped_too_few_atoms"]:
        print(f"  skipped, fewer than 2 atoms: {stats['skipped_too_few_atoms']}")
    if stats["no_family"]:
        print(f"  cases with no claim family (run §7 to assign): {stats['no_family']}")

    print(f"\n  {'family':30s} {'cases':>6s} {'atoms':>7s} {'median|T|':>10s}  viable?")
    import statistics
    for fam, rows in sorted(by_family.items(), key=lambda x: -len(x[1])):
        atoms = {a for r in rows for a in r["combined"]}
        med = statistics.median(len(r["combined"]) for r in rows)
        # §8.2 starts at min_support 0.05 and wants >=50 itemsets; §7 sets a 30-case floor.
        viable = "yes" if len(rows) >= 30 else "NO (under §7's 30-case floor)"
        print(f"  {fam:30s} {len(rows):6d} {len(atoms):7d} {med:10.0f}  {viable}")

    # The free calculation that predicts whether §8.2 can find anything.
    print(f"\n  atom recurrence across cases (support is per CASE, so this is the ceiling):")
    for fam, rows in sorted(by_family.items(), key=lambda x: -len(x[1]))[:6]:
        n = len(rows)
        df = Counter(a for r in rows for a in set(r["combined"]))
        for thr in (0.05, 0.10, 0.20):
            k = sum(1 for c in df.values() if c / n >= thr)
            print(f"    {fam:26s} n={n:4d}  atoms with support >= {thr:.2f}: {k:4d}", end="")
            if thr == 0.05:
                print(f"   <- §8.2's starting min_support")
            else:
                print()


if __name__ == "__main__":
    main()
