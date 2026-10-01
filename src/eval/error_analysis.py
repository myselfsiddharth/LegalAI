"""§10.3 error analysis: where the best model fails, and whether the LLM fails there too.

§10.3 asks for 50 errors from the best model and from an LLM baseline, categorised, with the overlap
between them reported. The categories it names are: extraction miss, wrong canonical label, family
misassignment, missing pattern, statute mismatch, genuinely discretionary outcome, label noise.

**Deterministic signals lead, the LLM is only asked about the residual.** Most of those categories
are checkable from the pipeline's own records, and a checkable signal beats a model's opinion about
its own failure:

  `label_noise`          the case's outcome label came from a rules/LLM CONFLICT, or from the LLM
                         alone with an unverifiable quote. We already hold 562 such conflicts.
  `extraction_thin`      the case yielded few facts, so there was little to predict from.
  `family_unassigned`    no claim family, so the family-conditioned features are blank.
  `statute_mismatch`     the predicted provisions and the ones the court actually introduced are
                         disjoint, so §9.2's contribution was wrong for this case.
  `atoms_only_generic`   every atom the case has is common across the corpus, so `F` carried nothing
                         specific.
  `discretionary`        the claim family is one the ontology marks as discretionary relief
                         (specific performance is the clear case: §10.4 flags exactly this).

Only cases that trip none of these go to the LLM, which is asked to choose between "the facts
genuinely do not determine this" and "the facts point the other way". That keeps the LLM's role to
the one judgement the records cannot supply.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from pydantic import BaseModel

from src import paths
from src.authorities import statute_predict as SP
from src.data.label_merge import load_final
from src.predict.features import FeatureBuilder
from src.predict.models import make_model
from src.llm import client

SEED = 573
GROUPS = ["F", "P", "E", "S", "R", "C"]
PROMPT_ID = "error_category.v1"
DISCRETIONARY = {"SpecificPerformance", "ContractInstrument"}


class Cat(BaseModel):
    category: str
    reason: str = ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--n-llm", type=int, default=50)
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    final = load_final()
    y_of = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in final.items() if r["outcome"] in ("WIN", "LOSE")}
    fb = FeatureBuilder()

    def usable(ids):
        return [c for c in ids if c in y_of and fb.canon.get(c)]

    tr, te = usable(split["train"]), usable(split["test"])
    fb.fit(tr)
    mask = fb.space.group_mask(GROUPS)
    Xtr, Xte = fb.transform(tr), fb.transform(te)
    ytr = np.array([y_of[c] for c in tr])
    yte = np.array([y_of[c] for c in te])
    model = make_model("gbm").fit(Xtr[:, mask], ytr)
    pred = model.predict(Xte[:, mask])
    prob = model.predict_proba(Xte[:, mask])[:, 1]
    acc = float((pred == yte).mean())
    print(f"best structured model: gbm {GROUPS}, test {len(te)}, accuracy {acc:.3f}")

    # --- LLM-CoT predictions on whatever overlap exists
    llm_path = paths.EXPERIMENTS / "paper3" / "llm_baselines_forum_heldout.json"
    llm_pred: dict[str, int] = {}
    if llm_path.exists():
        # re-derive from the per-arm cache: rerun is free, the cache holds every answer
        from src.predict.llm_baselines import main as _unused  # noqa: F401
    # cheaper: read the saved per-case answers if present, else skip the overlap section
    cot_path = paths.INTERIM / "llm_cot_predictions.jsonl"
    if cot_path.exists():
        for r in (json.loads(l) for l in open(cot_path)):
            if r.get("status") == "ok":
                llm_pred[r["doc_id"]] = 1 if r["pred"] == "WIN" else 0

    # --- deterministic signals
    atoms_df = Counter()
    for d in tr:
        for r in fb.canon[d]:
            atoms_df[r["label"]] += 1
    common = {a for a, n in atoms_df.items() if n >= 0.10 * len(tr)}

    novel_targets = SP.load_targets("labels_novel") if SP.TARGETS_PATH.exists() else {}

    errors = [c for c, p, y in zip(te, pred, yte) if p != y]
    correct = [c for c, p, y in zip(te, pred, yte) if p == y]
    print(f"errors: {len(errors)} of {len(te)} ({100*len(errors)/len(te):.1f}%)")

    cats: dict[str, str] = {}
    detail: dict[str, dict] = {}
    all_sig: dict[str, dict] = {}
    for c in te:
        rec = final[c]
        facts = fb.canon[c]
        labs = {r["label"] for r in facts}
        fams = [fb.fam_to_super.get(f, f) for f in fb.membership.get(c, {}).get("families", [])]
        ps = set(fb.pred_statutes.get(c, {}))
        ts = set(novel_targets.get(c, []))
        sig = {
            "label_noise": rec.get("source") in ("conflict", "none")
                           or (rec.get("source") == "llm" and not rec.get("llm_quote_grounded")),
            "extraction_thin": len(facts) <= 5,
            "family_unassigned": not fams,
            "statute_mismatch": bool(ps) and bool(ts) and not (ps & ts),
            "atoms_only_generic": bool(labs) and labs <= common,
            "discretionary": any(f in DISCRETIONARY for f in fams),
        }
        all_sig[c] = sig
        detail[c] = {"signals": sig, "n_facts": len(facts), "families": fams,
                     "probability": round(float(prob[te.index(c)]), 4)}
        hit = [k for k, v in sig.items() if v]
        cats[c] = hit[0] if hit else "UNEXPLAINED"

    counts = Counter(cats[c] for c in errors)
    print(f"\n  deterministic categorisation of {len(errors)} errors:")
    for k, v in counts.most_common():
        print(f"    {k:22s} {v:4d}  {100*v/len(errors):5.1f}%")

    # --- the control that makes a signal diagnostic rather than merely common.
    #
    # A signal present in 40% of errors tells us nothing if it is also present in 40% of correct
    # predictions. The lift is error-rate-among-flagged over the base error rate; a lift near 1.0
    # means the signal does not discriminate and must not be reported as an explanation.
    base_err = len(errors) / len(te)
    print(f"\n  signal diagnosticity (base error rate {base_err:.3f}):")
    hdr = (f"    {'signal':22s} {'in errors':>10s} {'in correct':>11s} "
           f"{'err|flagged':>12s} {'lift':>6s}")
    print(hdr); print("    " + "-" * (len(hdr) - 4))
    diag = {}
    for k in ("label_noise", "extraction_thin", "family_unassigned", "statute_mismatch",
              "atoms_only_generic", "discretionary"):
        ne = sum(1 for c in errors if all_sig[c][k])
        nc = sum(1 for c in correct if all_sig[c][k])
        flagged = ne + nc
        if not flagged:
            print(f"    {k:22s} {'0':>10s} {'0':>11s} {'n/a':>12s} {'n/a':>6s}"
                  f"   <- never fires; vacuous as implemented")
            diag[k] = {"n_errors": 0, "n_correct": 0, "lift": None,
                       "note": "never fires on this test set"}
            continue
        p_err = ne / flagged
        lift = p_err / base_err
        flag = "" if lift > 1.15 else "   <- not diagnostic"
        print(f"    {k:22s} {ne:10d} {nc:11d} {p_err:12.3f} {lift:6.2f}{flag}")
        diag[k] = {"n_errors": ne, "n_correct": nc, "p_error_given_flag": round(p_err, 4),
                   "lift": round(lift, 3)}

    # --- LLM on the residual only
    unexplained = [c for c in errors if cats[c] == "UNEXPLAINED"][:args.n_llm]
    print(f"\n  {sum(1 for c in errors if cats[c] == 'UNEXPLAINED')} errors trip no signal")
    if unexplained:
        masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
                  for l in open(paths.INTERIM / "masked_text.jsonl")}
        sysmsg = client.load_prompt(PROMPT_ID)
        from concurrent.futures import ThreadPoolExecutor

        def ask(c):
            i = te.index(c)
            res = client.complete(
                [{"role": "system", "content": sysmsg},
                 {"role": "user", "content":
                  f"MODEL PREDICTED: {'WIN' if pred[i] == 1 else 'LOSE'} "
                  f"(p={prob[i]:.2f})\nACTUAL OUTCOME: {final[c]['outcome']}\n\n"
                  f"CASE (facts and pleadings only):\n{masked[c][:12000]}"}],
                model=args.model, json_schema=Cat, prompt_id=PROMPT_ID, max_tokens=200)
            if not res.ok or res.parsed is None:
                return c, None
            return c, res.parsed

        with ThreadPoolExecutor(max_workers=24) as pool:
            llm_cats = dict(pool.map(ask, unexplained))
        lc = Counter(v.category for v in llm_cats.values() if v)
        print(f"\n  LLM categorisation of {len(unexplained)} unexplained errors:")
        for k, v in lc.most_common():
            print(f"    {k:28s} {v:4d}")
        for c, v in llm_cats.items():
            if v:
                detail[c]["llm_category"] = v.category
                detail[c]["llm_reason"] = v.reason[:200]

    # --- overlap with the LLM baseline
    overlap = None
    if llm_pred:
        both = [c for c in te if c in llm_pred]
        if both:
            ym = np.array([y_of[c] for c in both])
            pm = np.array([pred[te.index(c)] for c in both])
            pl = np.array([llm_pred[c] for c in both])
            me, le = (pm != ym), (pl != ym)
            overlap = {"n": len(both),
                       "model_errors": int(me.sum()), "llm_errors": int(le.sum()),
                       "both_wrong": int((me & le).sum()),
                       "only_model_wrong": int((me & ~le).sum()),
                       "only_llm_wrong": int((~me & le).sum()),
                       "jaccard": round(float((me & le).sum() / max(1, (me | le).sum())), 4)}
            print(f"\n  error overlap with LLM-CoT (n={len(both)}):")
            print(f"    both wrong {overlap['both_wrong']}, only model {overlap['only_model_wrong']}, "
                  f"only LLM {overlap['only_llm_wrong']}, Jaccard {overlap['jaccard']}")

    out = paths.EXPERIMENTS / "paper3" / f"error_analysis_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "accuracy": round(acc, 4),
                               "n_test": len(te), "n_errors": len(errors),
                               "deterministic_categories": dict(counts),
                               "signal_diagnosticity": diag,
                               "base_error_rate": round(base_err, 4),
                               "overlap_with_llm_cot": overlap,
                               "per_case": detail}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
