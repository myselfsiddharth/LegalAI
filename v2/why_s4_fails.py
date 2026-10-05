"""Why does S4 lose to S2, and is it the element ABSTRACTION or our DISCRETISATION of it?

S4 scores 0.521 against S2's 0.633. Three explanations were on the table; two are testable for free
because every fact already carries a cached embedding.

## H1 — an encoding bug, not a limit

`s4_eval.py` encodes an element as +1 satisfied, -1 not satisfied, **0 for UNCLEAR *and* 0 for
never-asked**. Those are different facts about the world. "We asked whether the possession was
hostile and the facts are silent" is informative -- it means the pleadings never established
hostility, which is often exactly why a claim fails. "We never asked" is not informative. Collapsing
both to zero throws away the distinction.

Fix: separate indicators. Free.

## H2 — we repeated S2's own mistake one level up

S2's entire finding was that **soft beats hard at identical granularity**: a 345-dim soft histogram
scores 0.633 where a 345-label argmax scores 0.593, and v1's authored-label version 0.517. Then S4
collapses a rich judgement into one of three labels. That is the same hard discretisation, applied to
elements instead of facts, and nobody noticed because the labels are legally meaningful.

The analogue of S2's winning move is an **element-similarity histogram**: for each of the 66
catalogue elements, a continuous score from the cosine between that element's descriptor and the
case's facts. 66 dense continuous dimensions instead of ~3 non-zero ternary ones, no LLM call, no
claim selection, nothing to be wrong about. If that recovers the gap, the element abstraction was
never the problem -- our quantisation of it was.

## H3 — the ceiling, and a flaw in the acceptance test I designed

Masked text reaches 0.666 on this split; S2 reaches 0.633. **So any fact-derived representation has
at most 0.033 of headroom above S2.** The acceptance test asked S4 to beat S2, which was close to
unwinnable: it demanded the element layer capture something the source text itself barely contains.
That does not make S4's 0.521 acceptable -- it is far BELOW S2, not merely failing to exceed it --
but it does mean "beats S2" was the wrong bar, and this module reports the headroom explicitly so the
bar is visible.
"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.data.label_merge import load_final
from src.eval import metrics
from src.extract.induced_vocab import load_fact_texts
from src.llm import client
from v2 import paths
from v2.element_catalogue import all_elements, load_claims

SEED = 573


def lr():
    return LogisticRegression(max_iter=5000, C=0.5, class_weight="balanced", random_state=SEED)


def dense(Atr, ytr, Ate, yte, name):
    sc = StandardScaler().fit(Atr)
    m = lr().fit(sc.transform(Atr), ytr)
    pr = m.predict_proba(sc.transform(Ate))[:, 1]
    r = metrics.evaluate(yte, m.predict(sc.transform(Ate)), pr, name=name)
    r["n_features"] = int(Atr.shape[1])
    return r, pr


def element_descriptor_emb(els) -> np.ndarray:
    """Embed each element once: name + definition + its claim's name."""
    cache = paths.INTERIM / "element_descriptor_emb.npy"
    if cache.exists():
        D = np.load(cache)
    else:
        texts = [f"{e.claim_name} — {e.name}. {e.definition}".strip() for e in els]
        D = np.asarray(client.embed(texts, verbose=False), dtype=np.float32)
        np.save(cache, D)
    return D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    args = ap.parse_args()

    els = all_elements(load_claims())
    eid = {e.element_id: i for i, e in enumerate(els)}
    n_el = len(els)

    facts, X = load_fact_texts()
    by_case: dict[str, list[int]] = {}
    for i, f in enumerate(facts):
        by_case.setdefault(f["case_id"], []).append(i)

    # --- S4 verdicts, in two encodings ---------------------------------------------------------
    old = collections.defaultdict(lambda: np.zeros(n_el, dtype=np.float32))   # the buggy one
    sat = collections.defaultdict(lambda: np.zeros(n_el, dtype=np.float32))
    neg = collections.defaultdict(lambda: np.zeros(n_el, dtype=np.float32))
    unc = collections.defaultdict(lambda: np.zeros(n_el, dtype=np.float32))
    ask = collections.defaultdict(lambda: np.zeros(n_el, dtype=np.float32))
    vp = paths.INTERIM / "element_verdicts_element_satisfy_v2_facts.jsonl"
    for line in open(vp):
        r = json.loads(line)
        c = r["case_id"]
        for d in r["verdicts"]:
            i = eid.get(d["element_id"])
            if i is None:
                continue
            ask[c][i] = 1.0
            if d["verdict"] == "SATISFIED" and d.get("gated"):
                old[c][i] = 1.0
                sat[c][i] = 1.0
            elif d["verdict"] == "NOT_SATISFIED" and d.get("gated"):
                old[c][i] = -1.0
                neg[c][i] = 1.0
            else:
                unc[c][i] = 1.0

    # --- the soft element representation (H2) ---------------------------------------------------
    D = element_descriptor_emb(els)
    softel: dict[str, np.ndarray] = {}
    for c, idx in by_case.items():
        V = X[idx]
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        S = V @ D.T                                   # facts x elements
        softel[c] = S.max(0)                          # max over facts, per element

    npz = np.load(paths.INTERIM / f"soft_families_{args.split}.npz", allow_pickle=True)
    s2 = {str(d): npz["train"][i] for i, d in enumerate(npz["train_ids"])}
    s2.update({str(d): npz["test"][i] for i, d in enumerate(npz["test_ids"])})

    masked = {}
    for line in open(paths.MASKED):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    ylab = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}

    def ok(c):
        return c in ylab and c in old and c in s2 and c in softel and masked.get(c, "").strip()
    tr = [c for c in split["train"] if ok(c)]
    te = [c for c in split["test"] if ok(c)]
    ytr = np.array([ylab[c] for c in tr])
    yte = np.array([ylab[c] for c in te])
    print(f"=== why does S4 fail? split={args.split}  train {len(tr):,} test {len(te)} ===\n")

    results, probs = [], {}

    # the ceiling, so the bar is visible (H3)
    v = TfidfVectorizer(max_features=50_000, ngram_range=(1, 2), min_df=3,
                        sublinear_tf=True, strip_accents="unicode")
    Xtr = v.fit_transform(masked[c] for c in tr)
    m = lr().fit(Xtr, ytr)
    Xte = v.transform(masked[c] for c in te)
    pr = m.predict_proba(Xte)[:, 1]
    probs["masked text (ceiling)"] = pr
    r = metrics.evaluate(yte, m.predict(Xte), pr, name="masked text (ceiling)")
    r["n_features"] = int(Xtr.shape[1])
    results.append(r)

    arms = {
        "S2 soft facts": lambda cs: np.vstack([s2[c] for c in cs]),
        "S4 old encoding": lambda cs: np.vstack([old[c] for c in cs]),
        "S4 H1 split encoding": lambda cs: np.vstack(
            [np.concatenate([sat[c], neg[c], unc[c], ask[c]]) for c in cs]),
        "S4 H2 soft elements": lambda cs: np.vstack([softel[c] for c in cs]),
        "S2 + H2 soft elements": lambda cs: np.vstack(
            [np.concatenate([s2[c], softel[c]]) for c in cs]),
    }
    for name, f in arms.items():
        r, pr = dense(f(tr), ytr, f(te), yte, name)
        results.append(r)
        probs[name] = pr

    for r in results:
        print(f"  {r['name']:24s} AUROC={r['auroc']:.3f}  mF1={r['macro_f1']:.3f}  "
              f"({r['n_features']:,} features)")

    def paired(a, b, n_boot=2000):
        rng = np.random.default_rng(SEED)
        d = []
        for _ in range(n_boot):
            i = rng.integers(0, len(yte), len(yte))
            if len(set(yte[i].tolist())) < 2:
                continue
            d.append(roc_auc_score(yte[i], probs[a][i]) - roc_auc_score(yte[i], probs[b][i]))
        return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    print("\n  paired bootstrap:")
    comps = {}
    for a, b in (("S4 H1 split encoding", "S4 old encoding"),
                 ("S4 H2 soft elements", "S4 old encoding"),
                 ("S4 H2 soft elements", "S2 soft facts"),
                 ("S2 + H2 soft elements", "S2 soft facts"),
                 ("masked text (ceiling)", "S2 soft facts")):
        m_, lo, hi = paired(a, b)
        comps[f"{a} - {b}"] = {"mean_diff": round(m_, 4), "ci95": [round(lo, 4), round(hi, 4)],
                               "ci_excludes_zero": lo > 0}
        print(f"    {a + ' - ' + b:48s} {m_:+.4f}  [{lo:+.4f}, {hi:+.4f}]"
              f"{'  significant' if lo > 0 else ''}")

    by = {r["name"]: r["auroc"] for r in results}
    head = by["masked text (ceiling)"] - by["S2 soft facts"]
    print(f"\n  HEADROOM: masked text {by['masked text (ceiling)']:.3f} - S2 "
          f"{by['S2 soft facts']:.3f} = {head:+.3f}")
    print(f"    Any fact-derived layer has at most {head:.3f} to gain over S2. 'Beat S2' was a")
    print(f"    near-unwinnable bar, and the acceptance test should have said so.")
    print(f"\n  H1 encoding fix:   {by['S4 H1 split encoding'] - by['S4 old encoding']:+.3f}")
    print(f"  H2 soft elements:  {by['S4 H2 soft elements'] - by['S4 old encoding']:+.3f} "
          f"over the old encoding, and {by['S4 H2 soft elements'] - by['S2 soft facts']:+.3f} vs S2")

    out = paths.EXPERIMENTS / f"why_s4_fails_{args.split}.json"
    out.write_text(json.dumps({"split": args.split, "n_train": len(tr), "n_test": len(te),
                               "headroom_text_minus_s2": round(head, 4),
                               "results": results, "paired": comps}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
