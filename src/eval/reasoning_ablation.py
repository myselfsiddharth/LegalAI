"""How much of reported legal-judgment-prediction accuracy comes from the court's reasoning?

Malik et al. (ACL 2021) build ILDC from 34,816 Indian Supreme Court proceedings and report a best
accuracy of **78%** against 94% for human legal experts. They are explicit and careful about the
decision itself: the "end section(s) directly stating the decision have been deleted from the
documents in ILDC since that is what we aim to predict." So ILDC does **not** leak the operative
order, and any claim that it does would be wrong.

What ILDC *retains* is the court's reasoning: "we consider (along with the facts) the entire case
(except the judgment), and we predict the judgment only." Their best model reads the **last 512
tokens**, chosen empirically, and justified because "the last parts of case proceedings usually
contain the main information about the case and the rationale behind the judgment."

§5.2 of this project takes the stricter line: the analysis section is excluded too, because a
court's reasoning is written *knowing* the outcome, so a model reading it is reading an argument
constructed to support a conclusion already reached.

That is a difference in task definition, not a defect in either paper — and it is measurable. This
runs both input constructions on **identical cases, splits and classifier**:

  `masked`                 facts, pleadings and arguments only (this project, §5.2)
  `ildc_style`             everything except the operative order (ILDC's construction)
  `ildc_style_no_headnote` the same, minus the pre-2000 HEADNOTE, which lists the outcome and the
                           authorities outright and is an artifact of this corpus rather than of
                           ILDC's method
  `last_512_tokens`        ILDC's actual best input: the final 512 tokens of the order-removed
                           document
  `full`                   the unmasked judgment, as a ceiling
  `order_only`             the removed disposition, as the probe's sensitivity control

The gap between `masked` and `ildc_style` is the quantity of interest: the share of achievable
accuracy that comes from the court's reasoning rather than from the facts of the dispute.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import paths
from src.data.label_merge import load_final
from src.eval import metrics

SEED = 573


def lr():
    return LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced", random_state=SEED)


def last_tokens(text: str, n: int = 512) -> str:
    w = text.split()
    return " ".join(w[-n:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    arms: dict[str, dict[str, str]] = {k: {} for k in
                                       ("masked", "ildc_style", "ildc_style_no_headnote",
                                        "last_512_tokens", "full", "order_only")}
    order_start = {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        d = r["doc_id"]
        arms["masked"][d] = r["masked_text"]
        arms["ildc_style"][d] = r["ildc_style_text"]
        arms["ildc_style_no_headnote"][d] = r["ildc_style_no_headnote"]
        arms["last_512_tokens"][d] = last_tokens(r["ildc_style_text"], 512)
        order_start[d] = r.get("order_region_start")
    for line in open(paths.CASE_TEXT):
        r = json.loads(line)
        d = r["doc_id"]
        arms["full"][d] = r["text"]
        s = order_start.get(d)
        arms["order_only"][d] = r["text"][s:] if s is not None else r["text"][-3000:]

    # identical case set across every arm, so only the input construction varies
    def ok(c):
        return c in labels and all(arms[a].get(c, "").strip() for a in arms)

    tr = [c for c in split["train"] if ok(c)]
    te = [c for c in split["test"] if ok(c)]
    ytr = np.array([labels[c] for c in tr])
    yte = np.array([labels[c] for c in te])

    print(f"=== how much comes from the court's reasoning? split={args.split} ===")
    print(f"identical case set: train {len(tr):,}, test {len(te)}; test WIN {yte.mean():.3f}\n")

    results, probs = [], {}
    for name in ("order_only", "full", "ildc_style", "ildc_style_no_headnote",
                 "last_512_tokens", "masked"):
        texts = arms[name]
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(texts[c] for c in tr)
        Xte = vec.transform(texts[c] for c in te)
        m = lr().fit(Xtr, ytr)
        p = m.predict(Xte)
        pr = m.predict_proba(Xte)[:, 1]
        probs[name] = pr
        r = metrics.evaluate(yte, p, pr, name=name)
        r["n_features"] = int(Xtr.shape[1])
        mean_words = int(np.mean([len(texts[c].split()) for c in te]))
        r["mean_words"] = mean_words
        results.append(r)
        print(f"  {name:24s} acc={r['accuracy']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"AUROC={r['auroc']:.3f}  ({mean_words:,} words, {Xtr.shape[1]:,} feats)")

    by = {r["name"]: r for r in results}

    def paired(a, b, n_boot=2000):
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[i], probs[a][i]) - roc_auc_score(yte[i], probs[b][i]))
        return (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))

    print("\n  paired bootstrap on the same test cases:")
    comps = {}
    for a, b in (("ildc_style", "masked"), ("ildc_style_no_headnote", "masked"),
                 ("last_512_tokens", "masked"), ("full", "ildc_style")):
        m, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:42s} {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    ms, ic = by["masked"]["auroc"], by["ildc_style"]["auroc"]
    icn, ceil = by["ildc_style_no_headnote"]["auroc"], by["order_only"]["auroc"]
    span = ceil - 0.5
    print(f"\n  interpretation")
    print(f"    facts and pleadings alone        AUROC {ms:.3f}")
    print(f"    + the court's reasoning          AUROC {ic:.3f}  "
          f"({ic - ms:+.3f})")
    print(f"    + reasoning, no HEADNOTE         AUROC {icn:.3f}  ({icn - ms:+.3f})")
    print(f"    the removed disposition itself   AUROC {ceil:.3f}")
    if span > 0:
        print(f"\n    Of the signal available above chance, the facts carry "
              f"{100*(ms-0.5)/span:.0f}% and")
        print(f"    adding the court's reasoning carries {100*(icn-0.5)/span:.0f}% "
              f"(HEADNOTE excluded).")

    out = paths.EXPERIMENTS / "paper1" / f"reasoning_ablation_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "split": args.split, "n_train": len(tr), "n_test": len(te),
        "reference": "Malik et al., ILDC for CJPE, ACL 2021: 34,816 Indian Supreme Court "
                     "proceedings, best model 78% accuracy, human experts 94%. ILDC deletes the "
                     "end sections stating the decision but retains the court's reasoning, and "
                     "its best model reads the last 512 tokens.",
        "results": results, "paired": comps}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
