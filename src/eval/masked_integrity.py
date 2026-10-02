"""Is `masked_text` -- the facts-only baseline -- itself contaminated?

`src/eval/leakage_setback.py` established that deleting the operative order leaves ~128 words of
outcome-telegraphing text behind, which accounts for 85% of the apparent advantage of ILDC's chosen
input. That result makes a second question urgent, because the mistake it corrected was assuming an
arm was clean without testing it:

    `masked` is the baseline EVERY comparison in this project runs against, and the text
    `src/extract/facts.py` reads. If it is leaky, every "facts-only" number is inflated.

`src/data/mask.py` already scrubs at sentence level: it drops sentences matching `OUTCOME_CUE` and
diverts sentences matching `PRIOR_COURT_DISPOSITION` into a separate channel. So the question is not
whether scrubbing happens but whether it is COMPLETE -- and there is a concrete reason to think it
is not. The two detectors in this repository disagree by construction:

  mask.py OUTCOME_CUE   requires the proceeding noun ADJACENT to the disposition verb:
                        `(appeal|petition|suit|slp) (is|are|stands) (hereby)? (allowed|dismissed)`
  labels.py ALLOWED/    allows up to 90 characters between them (`_GAP`), covers `we hereby
  DISMISSED             dismiss the appeal`, `the appeal fails`, `is devoid of merit`, and a much
                        wider proceeding vocabulary (`special leave petitions`, `revisions`,
                        `applications`, `proceedings`).

So a sentence like "This appeal, filed by the State against the judgment dated 12 March 1998, is
dismissed." is a disposition that labels.py reads as one and mask.py's scrubber does not match.
This module finds every such sentence, PRINTS THEM for reading (gotcha #9: a false clean is worse
than a false dirty), and measures what removing them costs.

Arms:
  `masked`              as built, the current baseline
  `masked_strict`       with every escaped disposition sentence removed
  `masked_minus128`     with the final 128 words removed -- the boundary test that found the
  `masked_minus512`     contamination in the first place, applied to this arm

If `masked_strict` scores materially below `masked`, the facts-only baseline is leaky and every
comparison against it understates the gap. If it does not, the baseline is sound and the
disagreement between the two detectors is labels.py over-matching recitals.
"""
from __future__ import annotations

import argparse
import json
import random
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import paths
from src.data import labels as L
from src.data import mask as M
from src.data.label_merge import load_final
from src.eval import metrics

SEED = 573
SENT_SPLIT = re.compile(r"(?<=[.;])\s+|\n")

# Tier-aware, matching how labels.py actually decides: a tier-1 disposition governs.
TIERED = {"allowed_t1": L.ALLOWED_T1, "dismissed_t1": L.DISMISSED_T1,
          "allowed_t2": L.ALLOWED_T2, "dismissed_t2": L.DISMISSED_T2,
          "partial": L.PARTIAL, "remand": L.REMAND}


def escaped(sent: str) -> list[str]:
    """Disposition patterns labels.py fires on that mask.py's own scrubbers do NOT."""
    if M.OUTCOME_CUE.search(sent) or M.PRIOR_COURT_DISPOSITION.search(sent):
        return []                                    # mask.py would have handled it
    return [k for k, p in TIERED.items() if p.search(sent)]


def strict_scrub(text: str) -> str:
    return "\n".join(s for s in SENT_SPLIT.split(text) if s.strip() and not escaped(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--show", type=int, default=30)
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    masked = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]

    arms = {"masked": masked, "masked_strict": {d: strict_scrub(t) for d, t in masked.items()}}
    for cut in (128, 512):
        arms[f"masked_minus{cut}"] = {
            d: " ".join(t.split()[:-cut]) if len(t.split()) > cut + 50 else ""
            for d, t in masked.items()}
    # Volume-controlled version of the same question. `masked_minus*` shrinks the text, so a loss
    # there is confounded with ordinary information loss. These two arms are both exactly 512 words,
    # so a gap between them is positional -- the same design that found the contamination in w0.
    for i in (0, 1):
        arms[f"masked_w{i}"] = {}
        for d, t in masked.items():
            w = t.split()
            end = len(w) - i * 512
            arms[f"masked_w{i}"][d] = " ".join(w[max(0, end - 512):end]) if end > 0 else ""

    def ok(c):
        return c in ylab and all(arms[a].get(c, "").strip() for a in arms)

    tr = [c for c in split["train"] if ok(c)]
    te = [c for c in split["test"] if ok(c)]
    ytr, yte = np.array([ylab[c] for c in tr]), np.array([ylab[c] for c in te])

    print(f"=== is `masked` itself contaminated? split={args.split} ===")
    print(f"identical case set: train {len(tr):,}, test {len(te)}; test WIN {yte.mean():.3f}\n")

    # ---- what escaped, and what does it look like? ------------------------------------------
    hits, by_pat = [], {k: 0 for k in TIERED}
    for c in te:
        for s in SENT_SPLIT.split(masked[c]):
            pats = escaped(s)
            if pats:
                hits.append((c, s.strip(), pats))
                for p in pats:
                    by_pat[p] += 1
    cases_hit = len({c for c, _, _ in hits})
    print(f"  escaped disposition sentences: {len(hits)} in {cases_hit}/{len(te)} test cases "
          f"({cases_hit/len(te):.1%})")
    print(f"  by pattern: " + ", ".join(f"{k}={v}" for k, v in by_pat.items() if v))
    removed = int(np.mean([len(masked[c].split()) - len(arms['masked_strict'][c].split())
                           for c in te]))
    print(f"  mean words removed per case by the strict scrub: {removed} "
          f"of {int(np.mean([len(masked[c].split()) for c in te]))}\n")

    print(f"  --- a random sample of {args.show}, for reading (gotcha #9) ---")
    rng = random.Random(SEED)
    for c, s, pats in rng.sample(hits, min(args.show, len(hits))):
        lab = "WIN " if ylab[c] else "LOSE"
        print(f"   [{lab} {','.join(pats):24s}] {s[:160]}")

    # ---- what does removing them cost? ------------------------------------------------------
    print()
    results, probs = [], {}
    for name in ("masked", "masked_strict", "masked_minus128", "masked_minus512",
                 "masked_w0", "masked_w1"):
        t = arms[name]
        v = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                            sublinear_tf=True, strip_accents="unicode")
        m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                               random_state=SEED).fit(v.fit_transform(t[c] for c in tr), ytr)
        Xte = v.transform(t[c] for c in te)
        pr = m.predict_proba(Xte)[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, m.predict(Xte), pr, name=name)
        r["mean_words"] = int(np.mean([len(t[c].split()) for c in te]))
        results.append(r)
        print(f"  {name:18s} acc={r['accuracy']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"AUROC={r['auroc']:.3f}  ({r['mean_words']:,} words)")

    def paired(a, b, n_boot=2000):
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[i], probs[a][i]) - roc_auc_score(yte[i], probs[b][i]))
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    print("\n  paired bootstrap (positive = `masked` is higher, i.e. removing it cost signal):")
    comps = {}
    for b in ("masked_strict", "masked_minus128", "masked_minus512"):
        m, lo, hi = paired("masked", b)
        comps[f"masked - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                  "ci_excludes_zero": lo > 0}
        print(f"    masked - {b:18s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  SIGNIFICANT' if lo > 0 else ''}")

    mw, mlo, mhi = paired("masked_w0", "masked_w1")
    comps["masked_w0 - masked_w1"] = {"mean_diff": round(mw, 4),
                                      "ci95": [round(mlo, 4), round(mhi, 4)],
                                      "ci_excludes_zero": mlo > 0}
    print(f"\n  volume-controlled (both arms exactly 512 words -- positional, not volume):")
    print(f"    masked_w0 - masked_w1      {mw:+.4f}  95% CI [{mlo:+.4f}, {mhi:+.4f}]"
          f"{'  SIGNIFICANT -> masked tail is enriched' if mlo > 0 else '  not significant -> masked tail is NOT enriched'}")
    for nm in ("masked_w0", "masked_w1"):
        cr = sum(1 for c in te if any(p.search(arms[nm][c]) for p in TIERED.values())) / len(te)
        print(f"    {nm} disposition-cue rate: {cr:.3f}")

    leaky = comps["masked - masked_strict"]["ci_excludes_zero"]
    verdict = ("LEAKY: removing the escaped disposition sentences costs real signal, so the "
               "facts-only baseline is inflated and every comparison against it understates the gap"
               if leaky else
               "SOUND: removing them costs nothing measurable, so the residual matches are "
               "labels.py over-matching recitals rather than dispositions mask.py missed")
    print(f"\n  VERDICT: {verdict}")

    out = paths.EXPERIMENTS / "paper1" / f"masked_integrity_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_train": len(tr), "n_test": len(te),
        "question": "is masked_text -- the facts-only baseline and the input to fact extraction -- "
                    "itself contaminated by dispositions that mask.py's OUTCOME_CUE misses but "
                    "labels.py's disposition patterns catch?",
        "escaped_sentences": len(hits), "cases_affected": cases_hit,
        "cases_affected_frac": round(cases_hit / len(te), 4), "by_pattern": by_pat,
        "mean_words_removed": removed,
        "examples": [{"doc_id": c, "outcome": "WIN" if ylab[c] else "LOSE",
                      "patterns": p, "sentence": s[:400]} for c, s, p in hits[:60]],
        "results": results, "paired": comps, "verdict": verdict}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
