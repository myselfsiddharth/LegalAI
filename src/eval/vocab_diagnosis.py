"""Which part of canonicalisation loses the signal: the unnameable tail, or the mapping itself?

The representation ladder shows canonical atoms sit at chance (AUROC 0.502) while the fact text they
replace reaches 0.577, and a sweep of the atom-frequency threshold rules out filtering as the cause.
That leaves two very different diagnoses, with opposite remedies:

  (a) **Coverage.** 22.7% of facts fall outside the vocabulary. If the signal lives mostly in those,
      the fix is §6.2's refinement loop -- add labels, re-canonicalise, done.

  (b) **The mapping.** If facts the vocabulary CAN name also lose their signal when named, then
      expanding coverage cannot help: the vocabulary's granularity or semantics is wrong, and
      `vocab_refine` would add 100 more labels that lose signal just as efficiently.

This separates them. For the mapped facts and the out-of-vocabulary facts, it scores the same cases
with the atoms and with the underlying text:

    mapped text   vs  mapped atoms        -> does naming a nameable fact destroy it?
    unmapped text vs  unmapped proposals  -> is the tail where the signal was?

Same classifier, same split, same case set per comparison, so only the representation varies.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import paths
from src.data.label_merge import load_final
from src.eval import metrics

SEED = 573


def lr():
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--min-atom-cases", type=int, default=2)
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    mapped_atoms, mapped_text = defaultdict(set), defaultdict(list)
    new_atoms, new_text = defaultdict(set), defaultdict(list)
    n_mapped = n_new = 0
    with (paths.INTERIM / "canonical_facts.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            cid = r["case_id"]
            if r.get("is_new"):
                new_atoms[cid].add(r["label"])
                new_text[cid].append(r["fact_text"])
                n_new += 1
            else:
                mapped_atoms[cid].add(r["label"])
                mapped_text[cid].append(r["fact_text"])
                n_mapped += 1

    print(f"canonical facts: {n_mapped + n_new:,}  "
          f"mapped {n_mapped:,} ({100*n_mapped/(n_mapped+n_new):.1f}%)  "
          f"out of vocabulary {n_new:,} ({100*n_new/(n_mapped+n_new):.1f}%)\n")

    def arms(name, atoms, text):
        """Score atoms and text on exactly the cases that have BOTH, so the pair is comparable."""
        tr = [c for c in split["train"] if c in labels and atoms.get(c) and text.get(c)]
        te = [c for c in split["test"] if c in labels and atoms.get(c) and text.get(c)]
        if len(tr) < 50 or len(te) < 20:
            print(f"{name}: too few cases (train={len(tr)}, test={len(te)})")
            return None
        ytr = np.array([labels[c] for c in tr])
        yte = np.array([labels[c] for c in te])

        # atoms: binary indicators, vocabulary fit on TRAIN only
        from collections import Counter
        df = Counter(a for c in tr for a in atoms[c])
        vocab = sorted(a for a, n in df.items() if n >= args.min_atom_cases)
        idx = {a: i for i, a in enumerate(vocab)}
        def mat(ids):
            X = np.zeros((len(ids), len(vocab)), dtype=np.float32)
            for i, c in enumerate(ids):
                for a in atoms[c]:
                    j = idx.get(a)
                    if j is not None:
                        X[i, j] = 1.0
            return X
        m = lr().fit(mat(tr), ytr)
        a_prob = m.predict_proba(mat(te))[:, 1]
        a_auc = roc_auc_score(yte, a_prob)

        # text: tf-idf over the same facts
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(" ".join(text[c]) for c in tr)
        Xte = vec.transform(" ".join(text[c]) for c in te)
        m2 = lr().fit(Xtr, ytr)
        t_prob = m2.predict_proba(Xte)[:, 1]
        t_auc = roc_auc_score(yte, t_prob)

        # Paired bootstrap on the SAME test cases: resampling cases rather than comparing two
        # independent CIs is what makes the difference testable at this n.
        rng = np.random.default_rng(SEED)
        diffs = []
        for _ in range(2000):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            diffs.append(roc_auc_score(yte[i], t_prob[i]) - roc_auc_score(yte[i], a_prob[i]))
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        p_gt0 = float(np.mean([d <= 0 for d in diffs]))

        print(f"{name}  (train {len(tr)}, test {len(te)})")
        print(f"    TEXT  of these facts : AUROC {t_auc:.3f}   ({Xtr.shape[1]} features)")
        print(f"    ATOMS of these facts : AUROC {a_auc:.3f}   ({len(vocab)} features)")
        print(f"    -> naming them costs {t_auc - a_auc:+.3f}  "
              f"paired 95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(no cost) = {p_gt0:.3f}")
        return {"name": name, "n_train": len(tr), "n_test": len(te),
                "paired_diff_ci95": [round(lo, 4), round(hi, 4)],
                "p_no_cost": round(p_gt0, 4),
                "text_auroc": round(float(t_auc), 4), "atom_auroc": round(float(a_auc), 4),
                "cost_of_naming": round(float(t_auc - a_auc), 4),
                "n_text_features": int(Xtr.shape[1]), "n_atom_features": len(vocab)}

    out = []
    r1 = arms("A. facts the vocabulary CAN name", mapped_atoms, mapped_text)
    print()
    r2 = arms("B. facts it CANNOT name (proposals as atoms)", new_atoms, new_text)
    out = [x for x in (r1, r2) if x]

    if r1 and r2:
        print("\nDIAGNOSIS")
        if r1["cost_of_naming"] > 0.04:
            print(f"  Naming a NAMEABLE fact costs {r1['cost_of_naming']:+.3f} AUROC. The loss is in")
            print(f"  the MAPPING, not in coverage -- expanding the vocabulary adds labels that")
            print(f"  would lose signal the same way. The granularity or semantics is wrong.")
        else:
            print(f"  Mapped facts survive naming ({r1['cost_of_naming']:+.3f}). The loss is")
            print(f"  concentrated in the {100*n_new/(n_mapped+n_new):.1f}% the vocabulary cannot")
            print(f"  name, so §6.2's refinement loop is the right fix.")

    path = paths.EXPERIMENTS / "paper1" / f"vocab_diagnosis_{args.split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"split": args.split, "n_mapped": n_mapped, "n_new": n_new,
                                "arms": out}, indent=1))
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
