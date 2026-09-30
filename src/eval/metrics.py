"""Metrics for §10, reported the way §14 requires: with n, split, and 95% CIs.

Three of these exist because the obvious choice would mislead on this dataset:

**Per-period / per-family majority, not global majority.** The WIN rate runs 39%→64% across the
temporal split. A global majority baseline is beaten by any model that merely notices the drift, so
it flatters everything. §10.1 asks for a per-family majority and the temporal drift makes a
per-period one necessary too.

**ECE with a reliability table.** §10.1 asks for calibration, and it is not decoration here: the
LLM-0 probe predicted WIN 13% of the time where the truth was 43%. A model can rank well and still
be unusable for a decision, and accuracy hides that completely.

**McNemar, not a comparison of two accuracies.** §10.1 asks for a paired test against LLM-0.
Two models at 62% and 65% on the same 900 cases may disagree on 40 cases or on 400; only the paired
disagreement counts say whether the gap is real.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def bootstrap_ci(y, pred, metric, n: int = 1000, seed: int = 573):
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        try:
            vals.append(metric(y[i], pred[i]))
        except ValueError:
            continue
    if not vals:
        return (float("nan"), float("nan"))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def expected_calibration_error(y, prob, n_bins: int = 10):
    """ECE plus the per-bin table, so a bad number can be read rather than just quoted."""
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, table = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob > lo) & (prob <= hi) if lo > 0 else (prob >= lo) & (prob <= hi)
        if not m.any():
            table.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": 0,
                          "mean_pred": None, "observed": None})
            continue
        mp, obs = float(prob[m].mean()), float(y[m].mean())
        ece += (m.sum() / len(y)) * abs(mp - obs)
        table.append({"bin": f"({lo:.1f},{hi:.1f}]", "n": int(m.sum()),
                      "mean_pred": round(mp, 4), "observed": round(obs, 4),
                      "gap": round(mp - obs, 4)})
    return round(float(ece), 4), table


def mcnemar(y, pred_a, pred_b):
    """Paired test on the same cases. Returns the discordant counts and an exact two-sided p.

    b = a right, b wrong; c = a wrong, b right. The binomial exact form is used rather than the
    chi-square approximation, which is unreliable when b+c is small -- and b+c is often small
    precisely when two systems are close, which is when the test matters most.
    """
    from math import comb
    y, a, b_ = np.asarray(y), np.asarray(pred_a), np.asarray(pred_b)
    a_ok, b_ok = (a == y), (b_ == y)
    b = int((a_ok & ~b_ok).sum())
    c = int((~a_ok & b_ok).sum())
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "n_discordant": 0, "p": 1.0,
                "note": "the two systems are right and wrong on exactly the same cases"}
    k = min(b, c)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return {"b_a_right_b_wrong": b, "c_a_wrong_b_right": c, "n_discordant": n,
            "p": round(float(p), 6)}


def evaluate(y, pred, prob=None, *, name: str = "", groups: dict | None = None,
             n_boot: int = 1000) -> dict:
    """One system on one test set. `groups` maps a group name (family, decade, forum) to a list
    parallel to y, for the per-group breakdown §10.1 asks for."""
    y = np.asarray(y)
    pred = np.asarray(pred)
    out = {
        "name": name, "n": int(len(y)),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "test_win_rate": round(float(y.mean()), 4),
        "pred_win_rate": round(float(pred.mean()), 4),
    }
    lo, hi = bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average="macro"), n=n_boot)
    out["macro_f1_ci95"] = [round(lo, 4), round(hi, 4)]
    if prob is not None:
        prob = np.asarray(prob, dtype=float)
        if len(set(y.tolist())) > 1:
            out["auroc"] = round(float(roc_auc_score(y, prob)), 4)
            alo, ahi = bootstrap_ci(y, prob, roc_auc_score, n=n_boot)
            out["auroc_ci95"] = [round(alo, 4), round(ahi, 4)]
        ece, table = expected_calibration_error(y, prob)
        out["ece"] = ece
        out["reliability"] = table
    if groups:
        out["by_group"] = {}
        for gname, gvals in groups.items():
            gv = np.asarray(gvals, dtype=object)
            per = {}
            for g in sorted({str(x) for x in gv}):
                m = gv == g
                if m.sum() < 10:
                    continue                          # too few to report
                per[g] = {"n": int(m.sum()),
                          "accuracy": round(float(accuracy_score(y[m], pred[m])), 4),
                          "macro_f1": round(float(f1_score(y[m], pred[m], average="macro")), 4),
                          "win_rate": round(float(y[m].mean()), 4)}
            out["by_group"][gname] = per
    return out


def majority_baseline(y_train, y_test, *, per_group: dict | None = None):
    """Global majority, and -- when groups are given -- a per-group majority, which is the
    honest floor on this dataset (see module docstring)."""
    maj = Counter(np.asarray(y_train).tolist()).most_common(1)[0][0]
    out = {"global": np.full(len(y_test), maj)}
    if per_group:
        gtr, gte = per_group["train"], per_group["test"]
        by = defaultdict(list)
        for g, v in zip(gtr, y_train):
            by[str(g)].append(v)
        gmaj = {g: Counter(v).most_common(1)[0][0] for g, v in by.items()}
        out["per_group"] = np.array([gmaj.get(str(g), maj) for g in gte])
    return out


def format_row(res: dict) -> str:
    ci = res.get("macro_f1_ci95", [float("nan")] * 2)
    auc = res.get("auroc")
    aci = res.get("auroc_ci95", [float("nan")] * 2)
    return (f"{res['name']:26s} n={res['n']:4d}  acc={res['accuracy']:.3f}  "
            f"mF1={res['macro_f1']:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  "
            f"AUROC={(f'{auc:.3f}' if auc is not None else '  -  ')}"
            f"{f' [{aci[0]:.3f},{aci[1]:.3f}]' if auc is not None else ''}  "
            f"ECE={res.get('ece', float('nan')):.3f}  predWIN={res['pred_win_rate']:.2f}")
