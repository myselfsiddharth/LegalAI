"""§11: emit a trace from a prediction back to the evidence it rests on.

§0.5 makes traceability a property of the pipeline rather than a reporting afterthought, and §11
fixes the output contract. This builds it from what the pipeline actually produces, and — this is
the part that matters — **says what it cannot produce** rather than filling the shape with plausible
values.

What is real here:
  * `prediction` from the §10 model, with its probability.
  * `facts` with `source_span` offsets into `masked_text`, every one verified to round-trip against
    its own quote (`facts_qa.py`: 4,184/4,184 exact).
  * `authorities.statutes` from §9.2's train-only model — predictions, never the court's own
    citations, which §5.2 makes targets.
  * `authorities.precedents` from §9.3, each predating the case, with the outcome of the analogous
    party where that label was available at training time.
  * `top_feature_contributions`, computed by ablation rather than asserted.

What is **not** real, and is emitted as an explicit gap:
  * `issues` / `elements`. §11's contract wants element status with `burden_on` and the patterns
    that satisfy or defeat each one. No element carries burden metadata (BLOCKER B2), and §8 found
    no pattern surviving significance correction, so there is nothing to populate them with. A trace
    that invented element statuses would be worse than one that reports them missing — it would read
    as legal reasoning while being decoration.

`top_feature_contributions` uses **ablation**, not coefficients: set a feature to zero, re-score,
record the change in predicted probability. That is model-agnostic, works for the gradient-boosted
model as well as the linear one, and measures what the model actually does with the feature rather
than what its weight suggests. It is also exactly the quantity §11's deletion test checks, so the
trace and its own evaluation cannot disagree by construction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from src import paths
from src.data.label_merge import load_final
from src.predict.features import FeatureBuilder
from src.predict.models import make_model

SEED = 573
DEFAULT_GROUPS = ["F", "P", "S", "R", "C"]


def _contributions(model, x: np.ndarray, names: list[str], top: int = 10) -> list[dict]:
    """Ablation attribution: zero one active feature at a time and re-score.

    Only features that are actually active (non-zero) for this case can be ablated, so the trace
    cites evidence the case has rather than evidence it lacks.
    """
    base = float(model.predict_proba(x.reshape(1, -1))[0, 1])
    active = np.nonzero(x)[0]
    out = []
    for j in active:
        xx = x.copy()
        xx[j] = 0.0
        p = float(model.predict_proba(xx.reshape(1, -1))[0, 1])
        out.append({"feature": names[j], "index": int(j),
                    "delta_probability": round(base - p, 5)})
    out.sort(key=lambda d: -abs(d["delta_probability"]))
    return out[:top]


def build_traces(split_name: str, kind: str = "gbm", groups=None, n: int = 25,
                 top: int = 10) -> list[dict]:
    groups = groups or DEFAULT_GROUPS
    split = json.loads((paths.SPLITS / f"{split_name}.json").read_text())
    fb = FeatureBuilder()
    labels = load_final()

    def usable(ids):
        return [c for c in ids if fb.canon.get(c)
                and labels.get(c, {}).get("outcome") in ("WIN", "LOSE")]

    tr, te = usable(split["train"]), usable(split["test"])
    fb.fit(tr)
    Xtr = fb.transform(tr)
    ytr = np.array([1 if labels[c]["outcome"] == "WIN" else 0 for c in tr])
    mask = fb.space.group_mask(groups)
    names = [nm for nm, keep in zip(fb.space.names, mask) if keep]
    model = make_model(kind).fit(Xtr[:, mask], ytr)

    Xte = fb.transform(te)
    facts_by_case: dict[str, list[dict]] = defaultdict(list)
    for line in open(paths.INTERIM / "facts.jsonl"):
        r = json.loads(line)
        facts_by_case[r["case_id"]].append(r)
    reg = {r["doc_id"]: r for r in (json.loads(l) for l in open(paths.CASE_REGISTRY))}

    traces = []
    for i, c in enumerate(te[:n]):
        x = Xte[i, mask]
        prob = float(model.predict_proba(x.reshape(1, -1))[0, 1])
        contribs = _contributions(model, x, names, top=top)

        # the facts behind the cited features, where a feature names an atom
        cited_atoms = {f["feature"].split("=", 1)[1] for f in contribs
                       if f["feature"].startswith("F:atom")}
        supporting = []
        for fact in facts_by_case.get(c, []):
            supporting.append(fact)
        statutes = fb.pred_statutes.get(c, {})
        precedents = []
        for d, s in fb.retrieved.get(c, [])[:5]:
            precedents.append({
                "case_id": d,
                "title": reg.get(d, {}).get("title"),
                "year": reg.get(d, {}).get("year"),
                "similarity": s,
                # the outcome of the analogous party, only where we were entitled to know it
                "analogous_party_outcome": (
                    "WIN" if fb._r_label.get(d) == 1 else
                    "LOSE" if fb._r_label.get(d) == 0 else None),
            })

        fams = fb.membership.get(c, {}).get("families", [])
        traces.append({
            "case_id": c,
            "title": reg.get(c, {}).get("title"),
            "year": reg.get(c, {}).get("year"),
            "prediction": {"label": "WIN" if prob >= 0.5 else "LOSE",
                           "probability": round(prob, 4)},
            "actual": labels[c]["outcome"],
            "claim_families": [fb.fam_to_super.get(f, f) for f in fams],
            "authorities": {
                "statutes_predicted": [{"provision": k, "probability": v}
                                       for k, v in sorted(statutes.items(),
                                                          key=lambda kv: -kv[1])],
                "precedents_retrieved": precedents,
            },
            "evidence": {
                "n_facts": len(supporting),
                "facts_cited_by_features": sorted(cited_atoms),
                # a sample of the fact spans, each verified to index its own quote
                "fact_spans": [{"fact_id": f["fact_id"], "text": f["text"][:160],
                                "source_span": f["source_span"],
                                "asserted_by": f["asserted_by"],
                                "disputed_status": f["disputed_status"]}
                               for f in supporting[:8]],
            },
            "top_feature_contributions": contribs,
            "gaps": {
                "issues_elements": (
                    "EMPTY. §11's contract wants per-element status with burden_on and the "
                    "patterns satisfying or defeating each. No element carries burden metadata "
                    "(BLOCKER B2) and §8 found no pattern surviving Benjamini-Hochberg, so there "
                    "is nothing to populate this with. Inventing element statuses would read as "
                    "legal reasoning while being decoration."),
                "defaults_assumed": (
                    "EMPTY. §10.4's default/burden-shift modelling is not built, so no conclusion "
                    "here depends on a stated default."),
            },
            "provenance": {
                "model": kind, "feature_groups": groups,
                "split": split_name, "split_hash": split.get("hash"),
                "attribution": "ablation (zero the feature, re-score); not coefficients",
                "spans_index": "masked_text, offsets verified to round-trip against the quote",
            },
        })
    return traces, (model, fb, mask, names, te, Xte, labels)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--model", default="gbm", choices=["gbm", "lr"])
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    traces, _ = build_traces(args.split, args.model, n=args.n)
    out = paths.EXPERIMENTS / "paper3" / f"traces_{args.split}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(traces, indent=1))

    correct = sum(1 for t in traces if t["prediction"]["label"] == t["actual"])
    print(f"traces: {len(traces)} cases -> {out}")
    print(f"  prediction correct on {correct}/{len(traces)}")
    nz = [len(t['top_feature_contributions']) for t in traces]
    print(f"  cited features per trace: median {int(np.median(nz))}")
    print(f"  with >=1 predicted statute: "
          f"{sum(1 for t in traces if t['authorities']['statutes_predicted'])}")
    print(f"  with >=1 retrieved precedent: "
          f"{sum(1 for t in traces if t['authorities']['precedents_retrieved'])}")
    print(f"  with a labelled analogous-party outcome: "
          f"{sum(1 for t in traces if any(p['analogous_party_outcome'] for p in t['authorities']['precedents_retrieved']))}")

    t0 = traces[0]
    print(f"\n  example — {t0['title'][:60]} ({t0['year']})")
    print(f"    predicted {t0['prediction']['label']} p={t0['prediction']['probability']}, "
          f"actual {t0['actual']}")
    print(f"    families: {t0['claim_families']}")
    print(f"    statutes: {[s['provision'] for s in t0['authorities']['statutes_predicted'][:4]]}")
    print(f"    top features:")
    for f in t0["top_feature_contributions"][:5]:
        print(f"      {f['delta_probability']:+.4f}  {f['feature'][:66]}")


if __name__ == "__main__":
    main()
