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

The comparison of interest is `ildc_full` minus `ildc_facts_only`, on identical cases.

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
    ap.add_argument("--max-train", type=int, default=8000,
                    help="cap training rows; the full 32k is unnecessary for a TF-IDF model and "
                         "masking 32k documents is the slow part")
    args = ap.parse_args()

    if args.download:
        download()

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

    results, probs = [], {}
    for name in ("ildc_full", "ildc_last512", "ildc_first_third", "ildc_facts_only"):
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
                 ("ildc_facts_only", "ildc_first_third")):
        if a in probs and b in probs:
            m, lo, hi, n = paired(a, b)
            comps[f"{a} - {b}"] = {"mean_diff": round(m, 4),
                                   "ci95": [round(lo, 4), round(hi, 4)],
                                   "ci_excludes_zero": lo > 0, "n": n}
            print(f"    {a + ' - ' + b:40s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  (n={n})")

    out = paths.EXPERIMENTS / "paper1" / "ildc_ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": "ILDC (Malik et al., ACL 2021) via Exploration-Lab/IL-TUR",
        "n_train_used": len(tr), "n_test": len(te),
        "note": "ILDC text already has the disposition deleted by its authors. The facts_only arm "
                "additionally removes the court's reasoning using this project's §5.2 masking, "
                "which is the difference between the two papers' task definitions.",
        "results": results, "paired": comps}, indent=1))
    print(f"\n-> {out}")
    if "ildc_full" in probs and "ildc_facts_only" in probs:
        bf = next(r for r in results if r["name"] == "ildc_full")
        fo = next(r for r in results if r["name"] == "ildc_facts_only")
        print(f"\n  On ILDC's own data and labels: removing the court's reasoning moves accuracy "
              f"{bf['accuracy']:.3f} -> {fo['accuracy']:.3f} "
              f"and AUROC {bf['auroc']:.3f} -> {fo['auroc']:.3f}.")


if __name__ == "__main__":
    main()
