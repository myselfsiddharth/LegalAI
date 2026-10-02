"""Does `last_512_tokens` win because of the court's REASONING, or residual order leakage?

`src/eval/reasoning_ablation.py` reports that the final 512 tokens of an order-removed judgment
reach AUROC 0.845 while facts-and-pleadings-only reach 0.668 (+0.176, CI [+0.148, +0.206]). Two
objections to that result are already answered by measurement:

  "the masked arm just has less text"  -- no: the WINNING arm has 511 words against masked's 2,501.
  "the order was never removed"        -- no: a model trained on the removed span alone scores 0.976.

The objection NOT yet answered is narrower and fairer. The last 512 tokens sit exactly where the
order used to be, so an imperfect boundary would contaminate that arm specifically, and no other.
This module settles it three ways.

## 1. A setback curve -- the decisive test

Four windows of exactly 512 words each, stepping backwards away from the end of the
order-removed document:

    w0 = [-512:]        w1 = [-1024:-512]        w2 = [-1536:-1024]        w3 = [-2048:-1536]

Every arm holds word count constant by construction, so volume cannot explain any difference
between them. The two hypotheses predict different shapes, stated before running:

  RESIDUAL LEAKAGE  w0 >> w1 ~= w2 ~= w3 ~= masked. Contamination lives next to the boundary and
                    cannot reach back 512 words, so the curve falls off a cliff after w0.
  REASONING         w0 > w1 > w2 > w3, all above masked. A court's analysis occupies the last
                    several thousand words, so signal decays gradually with distance.

## 2. A cue audit, against the right base rate

The disposition patterns from `src/data/labels.py` -- the same ones that produced this project's
outcome labels -- are run over w0 to count cases containing an explicit disposition statement.

A raw count would be uninterpretable on its own. `masked` text is full of sentences like "the High
Court dismissed the suit", which recite what a LOWER court did and leak nothing about this court's
decision; the regexes fire on those too. So the audit reports the cue rate in `masked` beside it as
the base rate. Only an EXCESS over that base rate is evidence of contamination.

## 3. A cue-scrubbed arm

Every sentence in w0 matching any disposition pattern is deleted and the arm is re-scored. If the
result survives removal of every outcome-shaped sentence, the signal is not carried by them.

Also reported: `w0_cue_free`, the same model evaluated only on test cases where no cue was found at
all, which avoids the scrub's own risk of deleting genuine reasoning along with the cues.
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src import paths
from src.data import labels as L
from src.data.label_merge import load_final
from src.eval import metrics

SEED = 573
WINDOW = 512
N_WINDOWS = 4

# Deliberately generous: every disposition shape the label rules know about, WITHOUT the negator
# check they apply. For an audit, over-counting a cue is the conservative error -- it inflates the
# measured contamination and strengthens the scrub, so a result that survives is not an artifact of
# a lenient detector.
CUE_PATTERNS = {
    "allowed": L.ALLOWED, "dismissed": L.DISMISSED, "partial": L.PARTIAL,
    "remand": L.REMAND, "other": L.OTHER,
}
SENT_SPLIT = re.compile(r"(?<=[.;])\s+")


def lr():
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)


def vec():
    return TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                           sublinear_tf=True, strip_accents="unicode")


def window_at(text: str, offset: int, size: int = WINDOW) -> str:
    """A `size`-word window ending `offset` words before the end of `text`."""
    w = text.split()
    end = len(w) - offset
    if end <= 0:
        return ""
    return " ".join(w[max(0, end - size):end])


def window(text: str, i: int, size: int = WINDOW) -> str:
    """The i-th non-overlapping 512-word window counting BACK from the end. i=0 is final."""
    return window_at(text, i * size, size)


def cues_in(text: str) -> list[str]:
    return [name for name, pat in CUE_PATTERNS.items() if pat.search(text)]


def scrub_cues(text: str) -> str:
    """Delete every sentence containing any disposition cue."""
    return " ".join(s for s in SENT_SPLIT.split(text) if not cues_in(s))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    masked, ildc = {}, {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]
        ildc[r["doc_id"]] = r["ildc_style_text"]

    arms: dict[str, dict[str, str]] = {"masked": masked}
    for i in range(N_WINDOWS):
        arms[f"w{i}"] = {d: window(t, i) for d, t in ildc.items()}
    arms["w0_scrubbed"] = {d: scrub_cues(t) for d, t in arms["w0"].items()}
    # The whole-document arm CONTAINS the contaminated tail. If the reasoning genuinely adds
    # signal, `ildc_style` must still beat `masked` once that tail is cut off -- otherwise its
    # advantage is the same residue, merely diluted across 5,000 words.
    arms["ildc_style"] = dict(ildc)
    for cut in (128, 512):
        arms[f"ildc_minus{cut}"] = {d: " ".join(t.split()[:-cut]) if len(t.split()) > cut else ""
                                    for d, t in ildc.items()}

    # one case set for every arm, so only the input construction varies
    def ok(c):
        return c in ylab and all(arms[a].get(c, "").strip() for a in arms)

    tr = [c for c in split["train"] if ok(c)]
    te = [c for c in split["test"] if ok(c)]
    lost = len([c for c in split["test"] if c in ylab]) - len(te)
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])

    print(f"=== setback curve: reasoning or residual leakage? split={args.split} ===")
    print(f"identical case set: train {len(tr):,}, test {len(te)}; test WIN {yte.mean():.3f}")
    print(f"{lost} labelled test cases dropped for being shorter than "
          f"{N_WINDOWS * WINDOW:,} words in the order-removed text\n")

    # ---- 2. the cue audit, with masked as the base rate -------------------------------------
    audit = {}
    for name in ("masked", "w0", "w1", "w2", "w3"):
        hits = [cues_in(arms[name][c]) for c in te]
        n_any = sum(1 for h in hits if h)
        audit[name] = {"cases_with_cue": n_any, "rate": round(n_any / len(te), 4),
                       "by_cue": {k: sum(1 for h in hits if k in h) for k in CUE_PATTERNS}}
    base = audit["masked"]["rate"]
    print("  cue audit (disposition patterns from src/data/labels.py):")
    for name in ("masked", "w0", "w1", "w2", "w3"):
        a = audit[name]
        tag = "  <- BASE RATE (recitals of the court below)" if name == "masked" else \
              f"  excess over base {a['rate'] - base:+.3f}"
        print(f"    {name:12s} {a['cases_with_cue']:4d}/{len(te)} = {a['rate']:.3f}{tag}")

    cue_free = [i for i, c in enumerate(te) if not cues_in(arms["w0"][c])]
    print(f"    w0 cue-free test cases: {len(cue_free)}/{len(te)}")
    scrub_words = int(np.mean([len(arms["w0_scrubbed"][c].split()) for c in te]))
    print(f"    w0 after scrubbing cue sentences: {scrub_words} words (from {WINDOW})\n")

    # ---- 1 & 3. score every arm --------------------------------------------------------------
    results, probs = [], {}
    order = ["masked", "w0", "w1", "w2", "w3", "w0_scrubbed",
             "ildc_style", "ildc_minus128", "ildc_minus512"]
    for name in order:
        t = arms[name]
        v = vec()
        Xtr = v.fit_transform(t[c] for c in tr)
        Xte = v.transform(t[c] for c in te)
        m = lr().fit(Xtr, ytr)
        pr = m.predict_proba(Xte)[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, m.predict(Xte), pr, name=name)
        r["mean_words"] = int(np.mean([len(t[c].split()) for c in te]))
        results.append(r)
        print(f"  {name:14s} acc={r['accuracy']:.3f}  mF1={r['macro_f1']:.3f}  "
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

    # the same w0 model, evaluated only where no cue was detected
    if len(cue_free) >= 100:
        sub_auroc = roc_auc_score(yte[cue_free], probs["w0"][cue_free])
        mk = yte[cue_free]
        sub_masked = roc_auc_score(mk, probs["masked"][cue_free])
        results.append({"name": "w0_cue_free_subset", "auroc": round(float(sub_auroc), 4),
                        "n": len(cue_free), "masked_auroc_same_subset": round(float(sub_masked), 4)})
        print(f"  {'w0_cue_free':14s} AUROC={sub_auroc:.3f}  (n={len(cue_free)}; "
              f"masked on the same cases {sub_masked:.3f})")

    # ---- paired comparisons ------------------------------------------------------------------
    print("\n  paired bootstrap on the same test cases:")
    comps = {}
    for a, b in (("w0", "masked"), ("w1", "masked"), ("w2", "masked"), ("w3", "masked"),
                 ("w0_scrubbed", "masked"), ("w0", "w0_scrubbed"), ("w0", "w1"), ("w1", "w3"),
                 ("ildc_style", "masked"), ("ildc_minus128", "masked"),
                 ("ildc_minus512", "masked"), ("ildc_style", "ildc_minus128")):
        m, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:26s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    # ---- how DEEP does the contamination reach? ----------------------------------------------
    # The setback curve answers the question at 512-word granularity. If the advantage is
    # disposition language bleeding past the order boundary, it should decay over a much shorter
    # distance than that -- so sweep the same 512-word window backwards in fine steps and find
    # where it stops beating `masked`. The answer bounds how much text after the order boundary a
    # benchmark would have to discard to be clean.
    print("\n  contamination depth -- a 512-word window set back in fine steps:")
    sweep = []
    for off in (0, 32, 64, 128, 256, 512):
        t = {d: window_at(x, off) for d, x in ildc.items()}
        if any(not t.get(c, "").strip() for c in tr + te):
            continue
        v = vec()
        m = lr().fit(v.fit_transform(t[c] for c in tr), ytr)
        pr = m.predict_proba(v.transform(t[c] for c in te))[:, 1]
        au = roc_auc_score(yte, pr)
        cue = sum(1 for c in te if cues_in(t[c])) / len(te)
        probs[f"off{off}"] = pr
        d, lo, hi = paired(f"off{off}", "masked")
        sweep.append({"offset_words": off, "auroc": round(float(au), 4), "cue_rate": round(cue, 4),
                      "vs_masked": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                      "ci_excludes_zero": lo > 0})
        print(f"    offset {off:4d} words  AUROC={au:.3f}  cue_rate={cue:.3f}  "
              f"vs masked {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")
    clean_at = next((r["offset_words"] for r in sweep if not r["ci_excludes_zero"]), None)
    if clean_at is not None:
        print(f"    -> the advantage is gone once the window is set back {clean_at} words")

    # ---- which shape did we get? ------------------------------------------------------------
    by = {r["name"]: r for r in results}
    ms = by["masked"]["auroc"]
    w = [by[f"w{i}"]["auroc"] for i in range(N_WINDOWS)]
    above = [i for i in range(N_WINDOWS) if comps[f"w{i} - masked"]["ci_excludes_zero"]]
    cliff = (w[0] - w[1]) > (w[1] - w[3]) and not comps["w1 - masked"]["ci_excludes_zero"]
    verdict = ("RESIDUAL LEAKAGE: only the window adjacent to the removed order beats masked"
               if cliff else
               "REASONING: windows set back from the order boundary also beat masked, "
               "so the signal is not contamination from the disposition")
    print(f"\n  setback curve: " + "  ".join(f"w{i}={w[i]:.3f}" for i in range(N_WINDOWS))
          + f"   (masked={ms:.3f})")
    print(f"  windows significantly above masked: {above or 'none'}")
    print(f"  VERDICT: {verdict}")
    im = comps["ildc_minus128 - masked"]
    print(f"\n  does the WHOLE-document advantage survive cutting the contaminated tail?")
    print(f"    ildc_style    - masked  {comps['ildc_style - masked']['mean_diff']:+.4f} "
          f"{comps['ildc_style - masked']['ci95']}")
    print(f"    ildc_minus128 - masked  {im['mean_diff']:+.4f} {im['ci95']}"
          f"{'  STILL SIGNIFICANT -> the reasoning itself adds signal' if im['ci_excludes_zero'] else '  NOT significant -> the whole advantage was the tail'}")

    out = paths.EXPERIMENTS / "paper1" / f"leakage_setback_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "hash": split.get("hash"),
        "n_train": len(tr), "n_test": len(te), "n_test_dropped_too_short": lost,
        "window_words": WINDOW, "n_windows": N_WINDOWS,
        "question": "does last_512_tokens beat masked because of the court's reasoning, or because "
                    "residual order text leaked into the final window?",
        "cue_audit": audit, "cue_base_rate_is": "masked -- recitals of the court below",
        "results": results, "paired": comps, "contamination_depth_sweep": sweep,
        "advantage_gone_at_offset_words": clean_at, "verdict": verdict}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
