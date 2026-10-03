"""Run the reasoning ablation on ILDC's own data and labels.

`src/eval/reasoning_ablation.py` measured, on OUR corpus, that an ILDC-style input (everything but
the disposition) reaches AUROC 0.718 while facts-and-pleadings-only reaches 0.654, and that ILDC's
actual best input — the last 512 tokens — reaches 0.782. That is a *comparable construction*, not a
measurement on the benchmark itself, which is the obvious weakness.

This closes that gap. It takes ILDC's released text and labels and asks one question:

    Their input already has the disposition removed. If we ALSO remove the court's reasoning,
    how much accuracy remains?

Arms, all on ILDC's own test split with ILDC's own labels:

  `ildc_full`        the released text, unchanged. Reproduces their setup.
  `ildc_last512`     the final 512 tokens, which their paper reports as its best input.
  `ildc_facts_only`  the released text with the court's reasoning removed by this project's §5.2
                     masking. This is the arm that does not exist in their paper.
  `ildc_first_third` a positional control: the opening third of the document. If `ildc_facts_only`
                     scores like this, the masking is just a crude truncation and the result says
                     nothing about reasoning specifically.
  `ildc_w0..w3`      THE SETBACK ARMS, added 2026-10-03 and the reason this module must not be run
                     without them. Four windows of exactly 512 words stepping back from the end of
                     ILDC's released text. See below.

## Why the setback arms are mandatory here

`src/eval/leakage_setback.py` established on our own corpus that deleting the operative order leaves
roughly **128 words** of outcome-telegraphing text behind -- Indian judgments announce the conclusion
one sentence BEFORE the operative order ("For the foregoing reasons we find no merit in the appeal.
The appeal is dismissed with costs."), so a boundary drawn correctly at the second sentence leaves
the first. On two splits, essentially the whole apparent advantage of a last-512 input was that
residue: a 512-word window set back past it scored no better than facts-only (+0.013 and +0.012,
both CIs spanning zero).

ILDC deletes "the end section(s) directly stating the decision" and its best model reads the **last
512 tokens** -- which is precisely the window our result says is contaminated. So running
`ildc_full` minus `ildc_facts_only` alone would inherit exactly the artifact this module exists to
measure. The setback arms are what distinguish "ILDC retains informative reasoning" from "ILDC
retains the sentence before the order".

The helpers are imported from `leakage_setback` rather than reimplemented, deliberately: this
project has already been bitten by two detectors in one repository disagreeing by construction
(see `src/eval/masked_integrity.py`), and one implementation cannot drift from itself.

## One sensitivity the smoke test exposed -- read this before quoting "128 words"

The depth sweep locates the boundary by asking where the window stops beating a facts-only arm, so
the answer depends on WHICH facts-only arm. Against `masked` (our §5.2 masking of the raw judgment)
`leakage_setback` put it at 128 words on both splits. Against `ildc_facts_only` (the same masking
applied to an already-order-removed document, which is what this module must use) the smoke-test
substitute put it at 512, with offsets 128 and 256 only marginally significant --
[+0.0118, +0.0646] and [+0.0030, +0.0584].

So **128 words is a lower bound under one baseline, not a constant.** Report the sweep table rather
than the single number, and say which baseline it is measured against.

## Running it

    python -m src.eval.ildc_ablation --smoke-test     # verify the code path, no gated data needed
    python -m src.eval.ildc_ablation                  # the real thing, once Data/ildc/ is populated

## Getting the data

The dataset is GATED on Hugging Face and its licence restricts use to academic research. That
acceptance is the user's to give, so this module does not attempt to fetch it. Two routes:

  1. Accept the terms at https://huggingface.co/datasets/Exploration-Lab/IL-TUR , then either place
     the parquet files in `Data/ildc/` or export `HF_TOKEN` and re-run with `--download`.
  2. Download the CSVs from the authors' original release and place them in `Data/ildc/` as
     `ILDC_multi.csv` (columns: text, label, split, name).

Either layout is detected automatically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import paths
from src.data.mask import mask_case
from src.eval import metrics
# One implementation of the setback machinery, shared with the measurement that motivated it.
from src.eval.leakage_setback import CUE_PATTERNS, cues_in, window_at

WINDOW = 512
N_WINDOWS = 4

SEED = 573
ILDC_DIR = paths.DATA / "ildc"
HF_REPO = "Exploration-Lab/IL-TUR"
HF_FILES = ["cjpe/test-00000-of-00001.parquet",
            "cjpe/multi_dev-00000-of-00001.parquet",
            "cjpe/multi_train-00000-of-00002.parquet",
            "cjpe/multi_train-00001-of-00002.parquet"]


def download() -> None:
    """Fetch the gated parquet files using the user's own HF token."""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not tok:
        raise SystemExit(
            "HF_TOKEN is not set. The ILDC release is a GATED dataset: accept the licence at\n"
            "  https://huggingface.co/datasets/Exploration-Lab/IL-TUR\n"
            "then set HF_TOKEN to a read token, or place the CSV/parquet files in Data/ildc/\n"
            "manually. This module will not route around the gate.")
    import urllib.request
    ILDC_DIR.mkdir(parents=True, exist_ok=True)
    for f in HF_FILES:
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{f}"
        dest = ILDC_DIR / f.split("/")[-1]
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {tok}"})
        with urllib.request.urlopen(req) as r, dest.open("wb") as out:
            out.write(r.read())
        print(f"  {dest.name}  {dest.stat().st_size/1e6:.1f} MB")


def load_substitute() -> list[dict]:
    """Stand in for the gated release using THIS project's own corpus, for a smoke test only.

    The ILDC licence acceptance is the user's to give, so this module cannot be run end to end
    until it is. That leaves a real risk: a code path that fails only on the real data, discovered
    an hour into a run nobody can repeat cheaply. This builds an ILDC-shaped input from our own
    `ildc_style_text` (order removed, reasoning retained -- the same construction ILDC uses) with
    our own WIN/LOSE labels, so every path in `main` executes against real judgment text.

    It is NOT a result. `--smoke-test` writes to a `_smoketest` filename and the output says so.
    """
    from src.data.label_merge import load_final
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    split = json.loads((paths.SPLITS / "forum_heldout.json").read_text())
    where = {c: "train" for c in split["train"]}
    where.update({c: "test" for c in split["test"]})
    rows = []
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        d = r["doc_id"]
        if d in ylab and d in where:
            rows.append({"text": r["ildc_style_text"], "label": ylab[d],
                         "split": where[d], "name": d})
    return rows


def load_ildc() -> list[dict]:
    """Rows of {text, label, split, name}, from parquet or CSV, whichever is present."""
    rows: list[dict] = []
    pq_files = sorted(glob.glob(str(ILDC_DIR / "*.parquet")))
    if pq_files:
        import pyarrow.parquet as pq
        for f in pq_files:
            t = pq.read_table(f).to_pylist()
            # the released split column is authoritative; infer from the filename if absent
            default = ("test" if "test" in f else "dev" if "dev" in f else "train")
            for r in t:
                rows.append({"text": r.get("text") or r.get("Text") or "",
                             "label": int(r.get("label", r.get("Label", -1))),
                             "split": r.get("split") or default,
                             "name": r.get("name") or r.get("Name") or ""})
        return rows
    csvs = sorted(glob.glob(str(ILDC_DIR / "*.csv")))
    if csvs:
        import csv as _csv
        for f in csvs:
            with open(f, newline="", encoding="utf-8", errors="replace") as fh:
                for r in _csv.DictReader(fh):
                    rows.append({"text": r.get("text", ""), "label": int(r.get("label", -1)),
                                 "split": r.get("split", "train"), "name": r.get("name", "")})
        return rows
    raise SystemExit(
        f"no ILDC data in {ILDC_DIR}. See this module's docstring: the dataset is gated and its\n"
        f"licence acceptance is yours to give. Place the files there, or set HF_TOKEN and pass\n"
        f"--download.")


def facts_only(text: str) -> str:
    """Apply this project's §5.2 masking to an ILDC document.

    ILDC text has already had the disposition deleted, so what the mask removes here is the
    court's REASONING plus the caption and reporting apparatus -- exactly the difference between
    the two papers' task definitions.
    """
    mc = mask_case({"doc_id": "ildc", "year": 0, "text": text, "n_chars": len(text)},
                   order_already_removed=True)
    return mc.masked_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--smoke-test", action="store_true",
                    help="exercise every code path using this project's own corpus as an "
                         "ILDC-shaped stand-in. Not a result; writes a _smoketest file.")
    ap.add_argument("--max-train", type=int, default=8000,
                    help="cap training rows; the full 32k is unnecessary for a TF-IDF model and "
                         "masking 32k documents is the slow part")
    args = ap.parse_args()

    if args.download:
        download()

    if args.smoke_test:
        print("=== SMOKE TEST: substituting this project's corpus for the gated release ===")
        print("=== the numbers below are NOT an ILDC result ===\n")
        rows = load_substitute()
    else:
        rows = load_ildc()
    by_split: dict[str, list[dict]] = {}
    for r in rows:
        if r["label"] in (0, 1) and len(r["text"] or "") > 500:
            by_split.setdefault(r["split"], []).append(r)
    print(f"ILDC loaded: " + ", ".join(f"{k} {len(v):,}" for k, v in sorted(by_split.items())))

    tr = by_split.get("train", [])[: args.max_train]
    te = by_split.get("test", [])
    if not tr or not te:
        raise SystemExit(f"need both train and test rows; got {len(tr)} / {len(te)}")
    ytr = np.array([r["label"] for r in tr])
    yte = np.array([r["label"] for r in te])
    print(f"using train {len(tr):,}, test {len(te):,}; test positive rate {yte.mean():.3f}")

    print("applying §5.2 masking to remove the court's reasoning (slow)...", flush=True)
    arms = {
        "ildc_full": {"tr": [r["text"] for r in tr], "te": [r["text"] for r in te]},
        "ildc_last512": {"tr": [" ".join(r["text"].split()[-512:]) for r in tr],
                         "te": [" ".join(r["text"].split()[-512:]) for r in te]},
        "ildc_facts_only": {"tr": [facts_only(r["text"]) for r in tr],
                            "te": [facts_only(r["text"]) for r in te]},
    }
    # positional control: the opening third, so a crude-truncation explanation can be ruled out
    arms["ildc_first_third"] = {
        "tr": [" ".join(r["text"].split()[: max(1, len(r["text"].split()) // 3)]) for r in tr],
        "te": [" ".join(r["text"].split()[: max(1, len(r["text"].split()) // 3)]) for r in te]}

    # THE SETBACK ARMS. Each is exactly 512 words, so volume cannot explain any difference between
    # them; only distance from the end of ILDC's released text varies. w0 IS `ildc_last512`, i.e.
    # the input ILDC's own best model reads.
    for i in range(N_WINDOWS):
        arms[f"ildc_w{i}"] = {"tr": [window_at(r["text"], i * WINDOW) for r in tr],
                              "te": [window_at(r["text"], i * WINDOW) for r in te]}
    # and the whole document with the suspect tail cut off
    arms["ildc_minus128"] = {
        k: [" ".join(r["text"].split()[:-128]) if len(r["text"].split()) > 178 else ""
            for r in v] for k, v in (("tr", tr), ("te", te))}

    results, probs = [], {}
    scored = ("ildc_full", "ildc_last512", "ildc_first_third", "ildc_facts_only",
              "ildc_w0", "ildc_w1", "ildc_w2", "ildc_w3", "ildc_minus128")
    for name in scored:
        a = arms[name]
        keep_tr = [i for i, t in enumerate(a["tr"]) if t.strip()]
        keep_te = [i for i, t in enumerate(a["te"]) if t.strip()]
        if len(keep_te) < 100:
            print(f"  {name}: only {len(keep_te)} usable test docs, skipped")
            continue
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(a["tr"][i] for i in keep_tr)
        Xte = vec.transform(a["te"][i] for i in keep_te)
        m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                               random_state=SEED).fit(Xtr, ytr[keep_tr])
        p = m.predict(Xte)
        pr = m.predict_proba(Xte)[:, 1]
        y = yte[keep_te]
        probs[name] = (keep_te, pr)
        r = metrics.evaluate(y, p, pr, name=name)
        r["mean_words"] = int(np.mean([len(a["te"][i].split()) for i in keep_te]))
        r["n_usable_test"] = len(keep_te)
        results.append(r)
        print(f"  {name:20s} acc={r['accuracy']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"AUROC={r['auroc']:.3f}  ({r['mean_words']:,} words, n={len(keep_te)})")

    # paired comparison on the cases both arms could score
    def paired(a, b, n_boot=2000):
        ia, pa = probs[a]
        ib, pb = probs[b]
        common = sorted(set(ia) & set(ib))
        ma = {j: k for k, j in enumerate(ia)}
        mb = {j: k for k, j in enumerate(ib)}
        y = yte[common]
        va = np.array([pa[ma[j]] for j in common])
        vb = np.array([pb[mb[j]] for j in common])
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            if len(set(y[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(y[i], va[i]) - roc_auc_score(y[i], vb[i]))
        return (float(np.mean(d)), float(np.percentile(d, 2.5)),
                float(np.percentile(d, 97.5)), len(common))

    print("\n  paired bootstrap:")
    comps = {}
    for a, b in (("ildc_full", "ildc_facts_only"), ("ildc_last512", "ildc_facts_only"),
                 ("ildc_facts_only", "ildc_first_third"),
                 ("ildc_w0", "ildc_facts_only"), ("ildc_w1", "ildc_facts_only"),
                 ("ildc_w2", "ildc_facts_only"), ("ildc_w3", "ildc_facts_only"),
                 ("ildc_minus128", "ildc_facts_only"), ("ildc_w0", "ildc_w1"),
                 ("ildc_full", "ildc_minus128")):
        if a in probs and b in probs:
            m, lo, hi, n = paired(a, b)
            comps[f"{a} - {b}"] = {"mean_diff": round(m, 4),
                                   "ci95": [round(lo, 4), round(hi, 4)],
                                   "ci_excludes_zero": lo > 0, "n": n}
            print(f"    {a + ' - ' + b:40s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={n})")

    # ---- cue audit, against the facts-only arm as the base rate ------------------------------
    # A raw cue count is uninterpretable: judgment text recites what LOWER courts did, and the
    # patterns match those too. Only an EXCESS over that base rate is evidence of contamination.
    # Note this statistic did NOT replicate across our own two splits while the setback curve did,
    # so it is reported as corroboration, never as the finding.
    audit = {}
    base_texts = arms["ildc_facts_only"]["te"]
    for name in ("ildc_facts_only", "ildc_w0", "ildc_w1", "ildc_w2", "ildc_w3"):
        t = arms[name]["te"]
        n_any = sum(1 for x in t if x.strip() and cues_in(x))
        n_use = sum(1 for x in t if x.strip())
        audit[name] = {"cases_with_cue": n_any, "n": n_use,
                       "rate": round(n_any / max(1, n_use), 4)}
    base = audit["ildc_facts_only"]["rate"]
    print("\n  cue audit (disposition patterns from src/data/labels.py):")
    for name in ("ildc_facts_only", "ildc_w0", "ildc_w1", "ildc_w2", "ildc_w3"):
        a = audit[name]
        tag = "  <- BASE RATE" if name == "ildc_facts_only" else \
              f"  excess over base {a['rate'] - base:+.3f}"
        print(f"    {name:18s} {a['cases_with_cue']:5d}/{a['n']} = {a['rate']:.3f}{tag}")

    # ---- how deep does the contamination reach? ----------------------------------------------
    sweep, clean_at = [], None
    if "ildc_facts_only" in probs:
        print("\n  contamination depth -- a 512-word window set back in fine steps:")
        for off in (0, 32, 64, 128, 256, 512):
            ttr = [window_at(r["text"], off) for r in tr]
            tte = [window_at(r["text"], off) for r in te]
            ktr = [i for i, x in enumerate(ttr) if x.strip()]
            kte = [i for i, x in enumerate(tte) if x.strip()]
            if len(kte) < 100:
                continue
            v = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                                sublinear_tf=True, strip_accents="unicode")
            m = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced",
                                   random_state=SEED).fit(
                v.fit_transform(ttr[i] for i in ktr), ytr[ktr])
            pr = m.predict_proba(v.transform(tte[i] for i in kte))[:, 1]
            au = float(roc_auc_score(yte[kte], pr))
            cue = sum(1 for i in kte if cues_in(tte[i])) / len(kte)
            probs[f"off{off}"] = (kte, pr)
            d, lo, hi, n = paired(f"off{off}", "ildc_facts_only")
            sweep.append({"offset_words": off, "auroc": round(au, 4), "cue_rate": round(cue, 4),
                          "vs_facts_only": round(d, 4), "ci95": [round(lo, 4), round(hi, 4)],
                          "ci_excludes_zero": lo > 0, "n": n})
            print(f"    offset {off:4d} words  AUROC={au:.3f}  cue_rate={cue:.3f}  "
                  f"vs facts_only {d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"{'  significant' if lo > 0 else ''}")
        clean_at = next((r["offset_words"] for r in sweep if not r["ci_excludes_zero"]), None)
        if clean_at is not None:
            print(f"    -> the advantage is gone once the window is set back {clean_at} words"
                  f"   (our corpus: 128)")

    # ---- which shape? ------------------------------------------------------------------------
    by = {r["name"]: r for r in results}
    verdict_setback = "not computed"
    if all(f"ildc_w{i}" in by for i in range(N_WINDOWS)) and "ildc_facts_only" in by:
        w = [by[f"ildc_w{i}"]["auroc"] for i in range(N_WINDOWS)]
        above = [i for i in range(N_WINDOWS)
                 if comps.get(f"ildc_w{i} - ildc_facts_only", {}).get("ci_excludes_zero")]
        verdict_setback = (
            "RESIDUAL LEAKAGE: only the window adjacent to the deleted decision beats facts-only, "
            "so ILDC's best input is contaminated the same way ours was"
            if above == [0] else
            "REASONING: windows set back from the boundary also beat facts-only, so ILDC retains "
            "informative reasoning rather than the sentence before the order"
            if len(above) > 1 else
            f"INCONCLUSIVE: windows above facts-only = {above}")
        print(f"\n  setback curve: " + "  ".join(f"w{i}={w[i]:.3f}" for i in range(N_WINDOWS))
              + f"   (facts_only={by['ildc_facts_only']['auroc']:.3f})")
        print(f"  windows significantly above facts_only: {above or 'none'}")
        print(f"  VERDICT: {verdict_setback}")
        if "ildc_minus128" in by:
            c = comps.get("ildc_minus128 - ildc_facts_only", {})
            if c:
                print(f"\n  whole document minus the suspect tail: {c['mean_diff']:+.4f} "
                      f"{c['ci95']}"
                      f"{'  significant -> the reasoning itself adds signal' if c.get('ci_excludes_zero') else '  NOT significant -> the advantage was the tail'}")

    tag = "_smoketest" if args.smoke_test else ""
    out = paths.EXPERIMENTS / "paper1" / f"ildc_ablation{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": ("SUBSTITUTE: this project's own corpus, ILDC-shaped (smoke test)"
                    if args.smoke_test
                    else "ILDC (Malik et al., ACL 2021) via Exploration-Lab/IL-TUR"),
        "n_train_used": len(tr), "n_test": len(te),
        "note": "ILDC text already has the disposition deleted by its authors. The facts_only arm "
                "additionally removes the court's reasoning using this project's §5.2 masking, "
                "which is the difference between the two papers' task definitions.",
        "smoke_test": args.smoke_test,
        "NOT_A_RESULT" if args.smoke_test else "_": (
            "this ran on THIS PROJECT'S corpus as an ILDC-shaped stand-in, to exercise the code "
            "path while the real release is gated. It is not an ILDC measurement."
            if args.smoke_test else None),
        "cue_audit": audit, "contamination_depth_sweep": sweep,
        "advantage_gone_at_offset_words": clean_at,
        "our_corpus_boundary_words": 128,
        "verdict_setback": verdict_setback,
        "results": results, "paired": comps}, indent=1))
    print(f"\n-> {out}")
    if "ildc_full" in probs and "ildc_facts_only" in probs:
        bf = next(r for r in results if r["name"] == "ildc_full")
        fo = next(r for r in results if r["name"] == "ildc_facts_only")
        src = ("the SUBSTITUTE corpus (smoke test -- NOT an ILDC result)" if args.smoke_test
               else "ILDC's own data and labels")
        print(f"\n  On {src}: removing the court's reasoning moves accuracy "
              f"{bf['accuracy']:.3f} -> {fo['accuracy']:.3f} "
              f"and AUROC {bf['auroc']:.3f} -> {fo['auroc']:.3f}.")


if __name__ == "__main__":
    main()
