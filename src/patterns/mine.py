"""Stage 4 (§8.2): FP-Growth fact patterns per claim family, with the controls that make them
mean something.

Four things here are not optional, and each guards a specific way pattern mining goes wrong:

**1. Outcome rules are computed on the TRAINING split only.** A pattern→outcome rule fitted on
test cases is a leak dressed as a feature, and it would be invisible in the final accuracy because
the rule itself carries the answer. Family membership and support come from all available cases;
anything that touches the label is restricted to train.

**2. Multiple testing is corrected.** Mining thousands of itemsets and keeping those whose outcome
distribution differs from the family base rate at p<0.05 manufactures roughly 5% of that count as
"discoveries". Benjamini–Hochberg on the chi-square p-values is what stops that, and it changes
the count enough to matter.

**3. Stability is checked by bootstrap.** A pattern that appears in one resample of the training
set and not in the next is describing this sample, not the law. §8.2 asks for ≥80% of 20
resamples; we report the distribution so the threshold is a choice, not a hidden default.

**4. Closed itemsets, not all itemsets.** {possession, hostile} and {possession} with identical
support say one thing, and reporting both inflates every count. We keep closed sets -- those with
no superset of equal support.

Base rates are per family, never global: with outcome drift running 39%→64% across the temporal
split (see the report's fact-patterns section), a global base rate would make a pattern look discriminative purely because
its family skews late.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from src import paths
from src.patterns.transactions import build as build_transactions

OUT_DIR = paths.PATTERNS
MIN_SUPPORT_START = 0.05
MIN_ITEMSETS, MAX_ITEMSETS = 50, 5_000
MAX_LEN = 4
N_BOOTSTRAP = 20
STABILITY_THRESHOLD = 0.8
MIN_FAMILY_CASES = 30


def _fpgrowth(transactions: list[set[str]], min_support: float, max_len: int):
    """FP-Growth over label sets. Uses mlxtend when available, else a self-contained
    Apriori-style fallback so the stage is runnable without the optional dependency."""
    try:
        import pandas as pd
        from mlxtend.frequent_patterns import fpgrowth
        from mlxtend.preprocessing import TransactionEncoder
        te = TransactionEncoder()
        arr = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(arr, columns=te.columns_)
        res = fpgrowth(df, min_support=min_support, use_colnames=True, max_len=max_len)
        return [(frozenset(r.itemsets), float(r.support)) for r in res.itertuples()]
    except ImportError:
        return _apriori_fallback(transactions, min_support, max_len)


def _apriori_fallback(transactions: list[set[str]], min_support: float, max_len: int):
    n = len(transactions)
    min_count = max(1, int(np.ceil(min_support * n)))
    df = Counter()
    for t in transactions:
        for a in t:
            df[a] += 1
    current = {frozenset([a]): c for a, c in df.items() if c >= min_count}
    out = [(k, v / n) for k, v in current.items()]
    k = 1
    while current and k < max_len:
        cands: dict[frozenset, int] = defaultdict(int)
        prev = list(current)
        for i in range(len(prev)):
            for j in range(i + 1, len(prev)):
                u = prev[i] | prev[j]
                if len(u) != k + 1:
                    continue
                if u in cands:
                    continue
                c = sum(1 for t in transactions if u <= t)
                if c >= min_count:
                    cands[u] = c
        current = dict(cands)
        out.extend((s, c / n) for s, c in current.items())
        k += 1
    return out


def closed_itemsets(itemsets: list[tuple[frozenset, float]]) -> list[tuple[frozenset, float]]:
    """Drop any itemset that has a superset with the same support -- it adds no information."""
    by_len = defaultdict(list)
    for s, sup in itemsets:
        by_len[len(s)].append((s, sup))
    keep = []
    for ln, items in by_len.items():
        bigger = [x for L, v in by_len.items() if L > ln for x in v]
        for s, sup in items:
            if any(s < s2 and abs(sup - sup2) < 1e-9 for s2, sup2 in bigger):
                continue
            keep.append((s, sup))
    return keep


def tune_support(transactions: list[set[str]], max_len: int):
    """Walk min_support down until the itemset count lands in [MIN_ITEMSETS, MAX_ITEMSETS]."""
    tried = []
    for ms in (0.30, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02):
        sets = _fpgrowth(transactions, ms, max_len)
        tried.append((ms, len(sets)))
        if MIN_ITEMSETS <= len(sets) <= MAX_ITEMSETS:
            return ms, sets, tried
        if len(sets) > MAX_ITEMSETS:
            return ms, sets[:MAX_ITEMSETS], tried            # too many: stop descending
    ms, sets = tried[-1][0], _fpgrowth(transactions, tried[-1][0], max_len)
    return ms, sets, tried


def chi2_and_p(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """2x2 chi-square with Yates correction. a,b = pattern present WIN/LOSE; c,d = absent."""
    from math import erfc, sqrt
    n = a + b + c + d
    if n == 0 or (a + b) == 0 or (c + d) == 0 or (a + c) == 0 or (b + d) == 0:
        return 0.0, 1.0
    num = abs(a * d - b * c) - n / 2
    if num < 0:
        num = 0.0
    chi2 = n * num * num / ((a + b) * (c + d) * (a + c) * (b + d))
    p = erfc(sqrt(chi2 / 2)) if chi2 > 0 else 1.0        # 1 dof survival function
    return chi2, p


def benjamini_hochberg(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    keep = [False] * m
    kmax = -1
    for rank, i in enumerate(order, 1):
        if pvals[i] <= alpha * rank / m:
            kmax = rank
    for rank, i in enumerate(order, 1):
        if rank <= kmax:
            keep[i] = True
    return keep


def mine_family(fam: str, rows: list[dict], labels: dict[str, int], train: set[str],
                view: str = "combined") -> dict:
    tx_all = [set(r[view]) for r in rows]
    ms, itemsets, tried = tune_support(tx_all, MAX_LEN)
    closed = closed_itemsets(itemsets)

    # --- anything touching the label uses TRAIN ONLY
    tr_rows = [r for r in rows if r["case_id"] in train and r["case_id"] in labels]
    tr_tx = [(set(r[view]), labels[r["case_id"]]) for r in tr_rows]
    n_tr = len(tr_tx)
    base_win = (sum(y for _, y in tr_tx) / n_tr) if n_tr else float("nan")

    rules, pvals = [], []
    for s, sup in closed:
        a = sum(1 for t, y in tr_tx if s <= t and y == 1)
        b = sum(1 for t, y in tr_tx if s <= t and y == 0)
        c = sum(1 for t, y in tr_tx if not s <= t and y == 1)
        d = sum(1 for t, y in tr_tx if not s <= t and y == 0)
        if a + b < 3:
            continue                                   # too rare in train to say anything
        conf = a / (a + b)
        lift = (conf / base_win) if base_win else float("nan")
        # conviction: how much more often the rule would be wrong if pattern and outcome were
        # independent. Infinite when the rule has no counterexample.
        conv = ((1 - base_win) / (1 - conf)) if conf < 1 else float("inf")
        chi2, p = chi2_and_p(a, b, c, d)
        rules.append({"pattern": sorted(s), "len": len(s), "support_all": round(sup, 4),
                      "n_train_with_pattern": a + b, "confidence_win": round(conf, 4),
                      "lift": round(lift, 3) if lift == lift else None,
                      "conviction": (round(conv, 3) if conv != float("inf") else "inf"),
                      "chi2": round(chi2, 3), "p": p})
        pvals.append(p)

    sig = benjamini_hochberg(pvals) if pvals else []
    for r, k in zip(rules, sig):
        r["discriminative_bh"] = bool(k)

    # --- bootstrap stability on the training transactions
    rng = np.random.default_rng(573)
    pat_keys = [tuple(r["pattern"]) for r in rules]
    seen = Counter()
    if n_tr >= 10 and pat_keys:
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_tr, n_tr)
            boot = [tr_tx[i][0] for i in idx]
            present = _fpgrowth(boot, ms, MAX_LEN)
            pset = {frozenset(s) for s, _ in present}
            for key in pat_keys:
                if frozenset(key) in pset:
                    seen[key] += 1
    for r in rules:
        r["stability"] = round(seen[tuple(r["pattern"])] / N_BOOTSTRAP, 3) if pat_keys else 0.0
        r["stable"] = r["stability"] >= STABILITY_THRESHOLD

    rules.sort(key=lambda r: (-abs(r["confidence_win"] - (base_win if base_win == base_win else 0.5)),
                              -r["support_all"]))
    return {
        "family": fam, "view": view, "n_cases": len(rows), "n_train_cases": n_tr,
        "min_support": ms, "support_search": tried,
        "n_itemsets": len(itemsets), "n_closed": len(closed), "n_rules": len(rules),
        "train_base_win_rate": round(base_win, 4) if base_win == base_win else None,
        "n_discriminative_bh": sum(1 for r in rules if r.get("discriminative_bh")),
        "n_stable": sum(1 for r in rules if r["stable"]),
        "n_stable_and_discriminative": sum(1 for r in rules
                                           if r["stable"] and r.get("discriminative_bh")),
        "rules": rules,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--view", default="combined", choices=["combined", "plaintiff", "defendant"])
    ap.add_argument("--oriented", type=int, default=1,
                    help="0 drops the @p/@d suffix -- §10.2 ablation 3")
    args = ap.parse_args()

    from src.data.label_merge import load_final
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    train = set(split["train"])

    by_family, stats = build_transactions(oriented=bool(args.oriented))
    if not by_family:
        raise SystemExit("no transactions -- run §6 canonicalisation (and §7 for families) first")

    out = {"split": args.split, "split_hash": split.get("hash"), "view": args.view,
           "oriented": bool(args.oriented), "families": {}}
    print(f"=== FP-Growth patterns, split={args.split}, view={args.view}, "
          f"oriented={bool(args.oriented)} ===\n")
    hdr = (f"{'family':26s} {'cases':>5s} {'train':>5s} {'minsup':>7s} {'sets':>5s} "
           f"{'closed':>6s} {'rules':>5s} {'BH sig':>7s} {'stable':>6s} {'both':>5s} {'base':>6s}")
    print(hdr); print("-" * len(hdr))

    for fam, rows in sorted(by_family.items(), key=lambda x: -len(x[1])):
        if len(rows) < MIN_FAMILY_CASES:
            print(f"{fam:26s} {len(rows):5d}  -- under §7's {MIN_FAMILY_CASES}-case floor, skipped")
            continue
        r = mine_family(fam, rows, labels, train, view=args.view)
        out["families"][fam] = r
        base = r["train_base_win_rate"]
        print(f"{fam:26s} {r['n_cases']:5d} {r['n_train_cases']:5d} {r['min_support']:7.2f} "
              f"{r['n_itemsets']:5d} {r['n_closed']:6d} {r['n_rules']:5d} "
              f"{r['n_discriminative_bh']:7d} {r['n_stable']:6d} "
              f"{r['n_stable_and_discriminative']:5d} "
              f"{(f'{base:.2f}' if base is not None else '-'):>6s}")

    path = OUT_DIR / f"patterns_{args.split}_{args.view}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n-> {path}")

    for fam, r in list(out["families"].items())[:3]:
        keep = [x for x in r["rules"] if x["stable"] and x.get("discriminative_bh")][:6]
        if not keep:
            continue
        print(f"\n  top stable + BH-significant patterns for {fam} "
              f"(train base WIN {r['train_base_win_rate']}):")
        for x in keep:
            print(f"    conf_WIN={x['confidence_win']:.2f} lift={x['lift']} "
                  f"n={x['n_train_with_pattern']:3d} stab={x['stability']:.2f}  "
                  f"{{{', '.join(x['pattern'])}}}")


if __name__ == "__main__":
    main()
