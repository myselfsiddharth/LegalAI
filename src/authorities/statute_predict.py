"""Stage 5 (§9.2): predict the statutory provisions a court relies on, from the facts.

This is the task that tells us whether the extracted facts are useful for **anything**. Outcome
prediction sits near chance on this corpus (see writeup/LexGraph_Report.tex, the outcome-models section), but which statute governs
a dispute should follow from its subject matter far more directly than who wins does — an adverse
possession case reaches the Limitation Act whatever the result. If the facts cannot predict the
statute either, the extraction has no demonstrated value; if they can, §6.1's output is worth
something even though §8's itemisation is not.

**The leakage rule (§5.2) is what makes this well-posed.** Statutes the court cites in its analysis
are chosen knowing the outcome, so they are legitimate **targets** but must never be inputs:

  targets  <- provisions normalised from the FULL judgment (what the court relied on)
  inputs   <- `masked_text` only, or representations derived from it

Multi-label over provisions that clear a support floor in TRAIN. Reported with micro/macro-F1 and
P@k/R@k per §9.2, against two baselines that are easy to beat by accident:

  `prior`        always predict the k globally most frequent provisions. On a skewed label set this
                 scores deceptively well, which is exactly why it is here.
  `family_prior` the most frequent provisions within the case's claim family -- a model that only
                 learns "land acquisition cases cite the Land Acquisition Act" should not be
                 mistaken for one that reads the facts.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier

from src import paths
from src.authorities import normalize as N
from src.data.label_merge import load_final

SEED = 573
TARGETS_PATH = paths.INTERIM / "statute_targets.jsonl"
MIN_TRAIN_CASES = 30


def build_targets(limit: int = 0) -> None:
    """Two target sets per case, and the distinction is the whole task.

    `labels_all`   every provision resolved anywhere in the judgment.
    `labels_novel` provisions the COURT introduced -- resolved in the judgment but **absent from
                   `masked_text`**, i.e. not already named in the facts, pleadings or arguments.

    Why `labels_novel` is the real target. A statute pleaded by a party sits in the predictor's
    INPUT as well as its target, so predicting `labels_all` is mostly string matching: a control
    that simply copies every provision out of `masked_text` scores micro-F1 **0.713** against the
    best learned model's 0.447. That is not a prediction task, it is an extraction task the §9.1
    normaliser already solves.

    §5.2 anticipates exactly this -- "only use authorities cited in pleadings/arguments, or
    predicted authorities; court-cited authorities are targets" -- so the well-posed question is
    which provisions the court reaches for that nobody pleaded.
    """
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    n = kept = kept_novel = 0
    with TARGETS_PATH.open("w") as out:
        for line in open(paths.CASE_TEXT):
            r = json.loads(line)
            n += 1
            if limit and n > limit:
                break
            doc = r["doc_id"]
            all_lab = {c.label for c in N.extract(r["text"]) if c.code != "UNRESOLVED"}
            in_input = {c.label for c in N.extract(masked.get(doc, ""))
                        if c.code != "UNRESOLVED"}
            novel = sorted(all_lab - in_input)
            methods = Counter(c.resolved_by for c in N.extract(r["text"])
                              if c.code != "UNRESOLVED")
            out.write(json.dumps({"doc_id": doc, "year": r["year"],
                                  "labels_all": sorted(all_lab),
                                  "labels_in_input": sorted(in_input),
                                  "labels_novel": novel,
                                  "by_method": dict(methods)}) + "\n")
            kept += bool(all_lab)
            kept_novel += bool(novel)
    print(f"statute targets: {n} judgments; {kept} with >=1 resolved provision, "
          f"{kept_novel} with >=1 provision the court introduced -> {TARGETS_PATH}")


def load_targets(which: str = "labels_novel") -> dict[str, list[str]]:
    return {r["doc_id"]: r[which]
            for r in (json.loads(l) for l in open(TARGETS_PATH))}


def _pk_rk(true_sets, ranked, k):
    """P@k and R@k, macro-averaged over cases. Cases with no true label are skipped: precision is
    undefined there and counting them as 0 would penalise the model for the labeller's gaps."""
    ps, rs = [], []
    for t, r in zip(true_sets, ranked):
        if not t:
            continue
        top = r[:k]
        hit = len(set(top) & t)
        ps.append(hit / k)
        rs.append(hit / len(t))
    return float(np.mean(ps)), float(np.mean(rs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--rebuild-targets", action="store_true")
    ap.add_argument("--min-train-cases", type=int, default=MIN_TRAIN_CASES)
    ap.add_argument("--targets", default="labels_novel",
                    choices=["labels_novel", "labels_all"],
                    help="'labels_novel' (default) are provisions the court introduced that are "
                         "absent from masked_text -- the well-posed task. 'labels_all' is kept for "
                         "comparison and is mostly solvable by copying from the input.")
    args = ap.parse_args()

    if args.rebuild_targets or not TARGETS_PATH.exists():
        build_targets()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    targets = load_targets(args.targets)
    print(f"target set: {args.targets}")
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    fact_txt: dict[str, list[str]] = defaultdict(list)
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        fact_txt[r["case_id"]].append(r["text"])
    atoms: dict[str, set[str]] = defaultdict(set)
    for line in open(paths.INTERIM / "canonical_facts.jsonl"):
        r = json.loads(line)
        atoms[r["case_id"]].add(r["label"])

    fam_path = paths.INTERIM / "claim_families.json"
    membership = (json.loads(fam_path.read_text()).get("membership", {})
                  if fam_path.exists() else {})
    merge_path = paths.INTERIM / "family_merge.json"
    fam_map = (json.loads(merge_path.read_text()).get("family_to_merged", {})
               if merge_path.exists() else {})

    def usable(ids):
        return [c for c in ids if targets.get(c) and masked.get(c) and fact_txt.get(c)]

    tr, te = usable(split["train"]), usable(split["test"])
    print(f"\n=== §9.2 statute prediction, split={args.split} ===")
    print(f"cases with targets, masked text and facts: train {len(tr)}, test {len(te)}")

    # label space: provisions with enough TRAIN support to be learnable
    df = Counter(l for c in tr for l in targets[c])
    labels = sorted(l for l, n in df.items() if n >= args.min_train_cases)
    if not labels:
        raise SystemExit("no provision clears the train support floor")
    lidx = {l: i for i, l in enumerate(labels)}
    print(f"label space: {len(labels)} provisions with >={args.min_train_cases} train cases "
          f"(of {len(df)} seen)")

    def Y(ids):
        M = np.zeros((len(ids), len(labels)), dtype=np.int8)
        for i, c in enumerate(ids):
            for l in targets[c]:
                j = lidx.get(l)
                if j is not None:
                    M[i, j] = 1
        return M

    Ytr, Yte = Y(tr), Y(te)
    true_sets = [set(targets[c]) & set(labels) for c in te]
    print(f"mean provisions per case (in label space): train {Ytr.sum(1).mean():.1f}, "
          f"test {Yte.sum(1).mean():.1f}")

    results = []

    def record(name, ranked, pred):
        mic = f1_score(Yte, pred, average="micro", zero_division=0)
        mac = f1_score(Yte, pred, average="macro", zero_division=0)
        row = {"system": name, "micro_f1": round(float(mic), 4), "macro_f1": round(float(mac), 4)}
        for k in (1, 5, 10):
            p, r = _pk_rk(true_sets, ranked, k)
            row[f"p@{k}"] = round(p, 4)
            row[f"r@{k}"] = round(r, 4)
        results.append(row)

    # --- baseline 1: global prior
    order = [l for l, _ in Counter(l for c in tr for l in targets[c]).most_common()
             if l in lidx]
    ranked = [order for _ in te]
    pred = np.zeros_like(Yte)
    for j in [lidx[l] for l in order[:5]]:
        pred[:, j] = 1
    record("prior (top-5 global)", ranked, pred)

    # --- baseline 2: per-family prior
    def fam_of(c):
        p = membership.get(c, {}).get("primary")
        return fam_map.get(p, p) or "_none"
    fam_counts = defaultdict(Counter)
    for c in tr:
        for l in targets[c]:
            if l in lidx:
                fam_counts[fam_of(c)][l] += 1
    fam_order = {f: [l for l, _ in cc.most_common()] for f, cc in fam_counts.items()}
    ranked_f = [fam_order.get(fam_of(c), order) or order for c in te]
    pred_f = np.zeros_like(Yte)
    for i, c in enumerate(te):
        for l in (fam_order.get(fam_of(c), order) or order)[:5]:
            pred_f[i, lidx[l]] = 1
    record("family prior (top-5)", ranked_f, pred_f)

    # --- baseline 3: COPY. Predict exactly the provisions that appear in `masked_text` itself.
    #
    # This is the control that decides whether the task is inference or string matching. A statute
    # named in the facts or pleadings is present in the INPUT as well as the target, so a model can
    # score well by copying. If `copy` approaches the learned models, they have learned little
    # beyond extraction; the gap over `copy` is the part that is actually predicted.
    copy_ranked, copy_pred = [], np.zeros_like(Yte)
    for i, c in enumerate(te):
        present = [x.label for x in N.extract(masked[c]) if x.label in lidx]
        seen = Counter(present)
        rk = [l for l, _ in seen.most_common()] + [l for l in order if l not in seen]
        copy_ranked.append(rk)
        for l in seen:
            copy_pred[i, lidx[l]] = 1
    record("copy from masked text", copy_ranked, copy_pred)

    # --- models over three input representations, all derived from masked text only
    def run_text(name, texts):
        vec = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                              sublinear_tf=True, strip_accents="unicode")
        Xtr = vec.fit_transform(texts[c] for c in tr)
        Xte = vec.transform(texts[c] for c in te)
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=SEED), n_jobs=-1).fit(Xtr, Ytr)
        prob = clf.predict_proba(Xte)
        pred = (prob >= 0.5).astype(np.int8)
        rk = [[labels[j] for j in np.argsort(-row)] for row in prob]
        record(name, rk, pred)

    run_text("masked text", {c: masked[c] for c in tr + te})
    run_text("extracted fact text", {c: " ".join(fact_txt[c]) for c in tr + te})

    # atoms as a sparse indicator matrix
    av = sorted({a for c in tr for a in atoms.get(c, ())})
    aidx = {a: i for i, a in enumerate(av)}
    def A(ids):
        M = np.zeros((len(ids), len(av)), dtype=np.float32)
        for i, c in enumerate(ids):
            for a in atoms.get(c, ()):
                j = aidx.get(a)
                if j is not None:
                    M[i, j] = 1.0
        return M
    if av:
        clf = OneVsRestClassifier(
            LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced",
                               random_state=SEED), n_jobs=-1).fit(A(tr), Ytr)
        prob = clf.predict_proba(A(te))
        record("canonical atoms", [[labels[j] for j in np.argsort(-row)] for row in prob],
               (prob >= 0.5).astype(np.int8))

    hdr = (f"{'system':24s} {'microF1':>8s} {'macroF1':>8s} {'P@1':>6s} {'P@5':>6s} "
           f"{'R@5':>6s} {'R@10':>6s}")
    print("\n" + hdr); print("-" * len(hdr))
    for r in results:
        print(f"{r['system']:24s} {r['micro_f1']:8.3f} {r['macro_f1']:8.3f} "
              f"{r['p@1']:6.3f} {r['p@5']:6.3f} {r['r@5']:6.3f} {r['r@10']:6.3f}")

    out = paths.EXPERIMENTS / "paper2" / f"statute_prediction_{args.split}_{args.targets}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "targets": args.targets,
                               "split_hash": split.get("hash"),
                               "n_train": len(tr), "n_test": len(te),
                               "n_labels": len(labels), "labels": labels,
                               "results": results}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
