"""The setback curve at corpus scale: every case type, ~3x the cases, no licence required.

`src/eval/leakage_setback.py` measured, on 6,954 land/property disputes, that deleting the operative
order leaves roughly 128 words of outcome-telegraphing text behind, and that this residue accounts
for essentially all of the apparent advantage of a last-512-token input. It replicated across two
splits of that corpus.

Two objections remain that neither split can answer:

  1. **It might be a land-dispute artifact.** Land judgments have their own conventions. If the
     contamination is a property of how Indian courts write *land* decisions, the result says little
     about the benchmark the field cites.
  2. **It might be a small-sample artifact.** 842 and 722 test cases is not many.

The intended answer to both was a replication on ILDC (34,816 Indian Supreme Court proceedings).
That release is gated behind a review queue that may never be processed. This module answers both
objections without it, using data already on disk: all **26,688** Supreme Court judgments,
1950-2025, every case type -- criminal, tax, constitutional, service, company, land. ILDC is drawn
from the same population (Indian Kanoon's Supreme Court judgments); the land restriction was ours,
imposed by a provided spreadsheet, not a limit of the data.

This is a **generalisation test, not a replication.** It cannot reproduce ILDC's labels or ILDC's
deletion boundary. What it can establish is whether the phenomenon is a property of Indian judicial
writing at scale, or of one dispute type at small n.

Arms are identical to `leakage_setback` so the numbers are directly comparable, and the helpers are
imported from it rather than reimplemented.

Splits:
  `temporal`  train <= 2004, test > 2013 -- the primary, matching the main corpus's temporal split.
              No future-to-past leakage.
  `random`    a seeded random split whose test set spans every decade, used ONLY for the by-era
              breakdown, which asks whether the contamination is concentrated in the pre-2000
              HEADNOTE era or stable across 75 years. Group-aware splitting is impossible without
              the spreadsheet's case groups, so absolute AUROCs here are optimistic; the quantity
              reported is the DIFFERENCE between word-count-matched arms, which a shared inflation
              does not move.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import paths
from src.eval import metrics
from src.eval.leakage_setback import cues_in, scrub_cues, window_at

SEED = 573
WINDOW = 512
N_WINDOWS = 4
SRC = paths.INTERIM / "corpus_wide.jsonl"


def lr():
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)


def vec():
    return TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                           sublinear_tf=True, strip_accents="unicode")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("temporal", "random"), default="temporal")
    ap.add_argument("--max-train", type=int, default=0, help="0 = use all")
    args = ap.parse_args()

    if not SRC.exists():
        raise SystemExit(f"{SRC} missing -- run `python -m src.data.corpus_wide` first")
    rows = [json.loads(l) for l in open(SRC)]

    arms = {"masked": {}, "ildc_style": {}}
    for i in range(N_WINDOWS):
        arms[f"w{i}"] = {}
    arms["w0_scrubbed"], arms["ildc_minus128"] = {}, {}
    meta = {}
    for r in rows:
        d, t = r["doc_id"], r["ildc_style_text"]
        arms["masked"][d] = r["masked_text"]
        arms["ildc_style"][d] = t
        for i in range(N_WINDOWS):
            arms[f"w{i}"][d] = window_at(t, i * WINDOW)
        arms["w0_scrubbed"][d] = scrub_cues(arms["w0"][d])
        w = t.split()
        arms["ildc_minus128"][d] = " ".join(w[:-128]) if len(w) > 178 else ""
        meta[d] = {"year": r["year"], "label": r["label"]}

    usable = [d for d in meta if all(arms[a].get(d, "").strip() for a in arms)]
    if args.split == "temporal":
        tr = [d for d in usable if 0 < meta[d]["year"] <= 2004]
        te = [d for d in usable if meta[d]["year"] > 2013]
    else:
        rng = np.random.default_rng(SEED)
        sh = list(usable)
        rng.shuffle(sh)
        cut = int(len(sh) * 0.8)
        tr, te = sh[:cut], sh[cut:]
    if args.max_train:
        tr = tr[: args.max_train]
    ytr = np.array([meta[d]["label"] for d in tr])
    yte = np.array([meta[d]["label"] for d in te])

    print(f"=== setback at corpus scale: every case type, split={args.split} ===")
    print(f"{len(rows):,} rules-labelled WIN/LOSE judgments built from 26,688 PDFs")
    print(f"usable in all arms: {len(usable):,}")
    print(f"train {len(tr):,}  test {len(te):,}  test WIN {yte.mean():.3f}\n")

    audit = {}
    for name in ("masked", "w0", "w1", "w2", "w3"):
        n_any = sum(1 for d in te if cues_in(arms[name][d]))
        audit[name] = {"cases_with_cue": n_any, "rate": round(n_any / len(te), 4)}
    base = audit["masked"]["rate"]
    print("  cue audit (corroboration only -- it did not replicate across the land corpus's splits):")
    for name in ("masked", "w0", "w1", "w2", "w3"):
        a = audit[name]
        tag = "  <- BASE RATE" if name == "masked" else f"  excess {a['rate'] - base:+.3f}"
        print(f"    {name:12s} {a['cases_with_cue']:5d}/{len(te)} = {a['rate']:.3f}{tag}")

    results, probs = [], {}
    order = ["masked", "w0", "w1", "w2", "w3", "w0_scrubbed", "ildc_style", "ildc_minus128"]
    print()
    for name in order:
        t = arms[name]
        v = vec()
        m = lr().fit(v.fit_transform(t[d] for d in tr), ytr)
        pr = m.predict_proba(v.transform(t[d] for d in te))[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, m.predict(v.transform(t[d] for d in te)), pr, name=name)
        r["mean_words"] = int(np.mean([len(t[d].split()) for d in te]))
        results.append(r)
        print(f"  {name:14s} acc={r['accuracy']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"AUROC={r['auroc']:.3f}  ({r['mean_words']:,} words)")

    def paired(a, b, n_boot=2000, idx=None):
        ii = np.arange(len(yte)) if idx is None else np.asarray(idx)
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            s = ii[rng.integers(0, len(ii), len(ii))]
            if len(set(yte[s].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[s], probs[a][s]) - roc_auc_score(yte[s], probs[b][s]))
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    print("\n  paired bootstrap:")
    comps = {}
    for a, b in (("w0", "masked"), ("w1", "masked"), ("w2", "masked"), ("w3", "masked"),
                 ("w0_scrubbed", "masked"), ("w0", "w1"),
                 ("ildc_style", "masked"), ("ildc_minus128", "masked"),
                 ("ildc_style", "ildc_minus128")):
        m, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:26s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    print("\n  contamination depth:")
    sweep = []
    for off in (0, 32, 64, 128, 256, 512):
        t = {d: window_at(arms["ildc_style"][d], off) for d in tr + te}
        if any(not t[d].strip() for d in tr + te):
            continue
        v = vec()
        m = lr().fit(v.fit_transform(t[d] for d in tr), ytr)
        pr = m.predict_proba(v.transform(t[d] for d in te))[:, 1]
        probs[f"off{off}"] = pr
        au = float(roc_auc_score(yte, pr))
        cue = sum(1 for d in te if cues_in(t[d])) / len(te)
        dd, lo, hi = paired(f"off{off}", "masked")
        sweep.append({"offset_words": off, "auroc": round(au, 4), "cue_rate": round(cue, 4),
                      "vs_masked": round(dd, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "ci_excludes_zero": lo > 0})
        print(f"    offset {off:4d}  AUROC={au:.3f}  cue={cue:.3f}  "
              f"vs masked {dd:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")
    clean_at = next((r["offset_words"] for r in sweep if not r["ci_excludes_zero"]), None)
    print(f"    -> advantage gone at {clean_at} words   (land corpus: 128)")

    # is the contamination an artifact of one era? only meaningful when the test set spans decades
    eras = {}
    decades = sorted({(meta[d]["year"] // 10) * 10 for d in te if meta[d]["year"]})
    if len(decades) > 2:
        print("\n  by era -- is this a pre-2000 reporting-style artifact?")
        for dec in decades:
            idx = [i for i, d in enumerate(te) if (meta[d]["year"] // 10) * 10 == dec]
            if len(idx) < 60:
                continue
            m, lo, hi = paired("w0", "masked", n_boot=1000, idx=idx)
            eras[str(dec)] = {"n": len(idx), "w0_minus_masked": round(m, 4),
                              "ci95": [round(lo, 4), round(hi, 4)], "ci_excludes_zero": lo > 0}
            print(f"    {dec}s  n={len(idx):4d}  w0-masked {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  significant' if lo > 0 else ''}")

    w = [next(r for r in results if r["name"] == f"w{i}")["auroc"] for i in range(N_WINDOWS)]
    ms = next(r for r in results if r["name"] == "masked")["auroc"]
    above = [i for i in range(N_WINDOWS) if comps[f"w{i} - masked"]["ci_excludes_zero"]]
    verdict = ("RESIDUAL LEAKAGE: the cliff reproduces at corpus scale across every case type"
               if above == [0] else
               f"DIFFERENT SHAPE AT SCALE: windows above masked = {above}")
    print(f"\n  setback curve: " + "  ".join(f"w{i}={w[i]:.3f}" for i in range(N_WINDOWS))
          + f"   (masked={ms:.3f})")
    print(f"  VERDICT: {verdict}")

    out = paths.EXPERIMENTS / "paper1" / f"corpus_scale_setback_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_rows_built": len(rows), "n_usable": len(usable),
        "n_train": len(tr), "n_test": len(te),
        "scope": "all 26,688 Supreme Court judgments 1950-2025, every case type, rules-only labels",
        "caveats": ["labels are rules-only (no LLM pass); label noise ATTENUATES effects, so this "
                    "is a conservative estimate",
                    "no case-group metadata, so the random split may leak connected matters; the "
                    "reported quantity is the difference between word-count-matched arms, which a "
                    "shared inflation does not move",
                    "this is a GENERALISATION test, not an ILDC replication: it reproduces neither "
                    "ILDC's labels nor ILDC's deletion boundary"],
        "cue_audit": audit, "results": results, "paired": comps,
        "contamination_depth_sweep": sweep, "advantage_gone_at_offset_words": clean_at,
        "land_corpus_boundary_words": 128, "by_era": eras, "verdict": verdict}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
