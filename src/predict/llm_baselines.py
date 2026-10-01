"""§3.3's LLM baseline suite, on the same split, cases and metrics as §10.

Five arms, each isolating a different question:

  `LLM-0`     zero-shot on masked text. Already run by `src/eval/llm_probe.py`; repeated here so
              every arm shares one scoring path.
  `LLM-FS`    few-shot with k retrieved training examples. Retrieval is **time-respecting**: an
              example must predate the case, so the prompt cannot carry the future.
  `LLM-CoT`   element-by-element reasoning, with the ontology's elements for the case's claim
              families supplied.
  `LLM-CF`    our canonical facts instead of raw text. This isolates the value of §6's extraction.
  `LLM-CF+A`  canonical facts plus predicted statutes and retrieved precedents. This isolates the
              value of §9's grounding.

**`LLM-CF` is a falsifiable prediction, not a formality.** The representation ladder showed canonical
atoms sit near chance (AUROC 0.517) while the masked text they replace reaches 0.657. If that is a
property of the representation rather than of the classifier, then handing an LLM the atoms instead
of the text should make it *worse*, not better — `LLM-CF` below `LLM-0`. A different result would
mean the atoms carry signal that linear and tree models cannot reach, which would change what §6 is
worth.

Few-shot examples use TRAIN labels only. Precedent outcomes in `LLM-CF+A` likewise come from the
train-restricted label map, so no test case's outcome can inform another's prediction.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel

from src import paths
from src.authorities.retrieval import BM25Index, scrub_citations
from src.data.label_merge import load_final
from src.eval import metrics
from src.llm import client

SEED = 573
MAX_CHARS = 18_000
ARMS = ("LLM-0", "LLM-FS", "LLM-CoT", "LLM-CF", "LLM-CF+A")
PROMPTS = {"LLM-0": "probe_outcome.v1", "LLM-FS": "baseline_fs.v1",
           "LLM-CoT": "baseline_cot.v1", "LLM-CF": "baseline_cf.v1",
           "LLM-CF+A": "baseline_cfa.v1"}


class Verdict(BaseModel):
    outcome: str
    confidence: float = 0.5


class VerdictCoT(BaseModel):
    outcome: str
    confidence: float = 0.5
    claim: str | None = None
    elements: list[dict] = []


def fact_bundle(facts: list[dict], limit: int = 60) -> str:
    """Canonical labels with their provenance. This is all `LLM-CF` sees."""
    lines = []
    for r in facts[:limit]:
        who = r.get("asserted_by", "unknown")
        st = r.get("disputed_status", "unclear")
        lines.append(f"- {r['label']}  (asserted_by={who}, {st})")
    return "\n".join(lines) if lines else "(no canonical facts extracted for this case)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k-shot", type=int, default=4)
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    labels = load_final()
    y_of = {d: (1 if r["outcome"] == "WIN" else 0)
            for d, r in labels.items() if r["outcome"] in ("WIN", "LOSE")}
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    reg = {r["doc_id"]: r for r in (json.loads(l) for l in open(paths.CASE_REGISTRY))}
    years = {d: reg[d]["year"] for d in reg}

    canon: dict[str, list[dict]] = defaultdict(list)
    for line in open(paths.INTERIM / "canonical_facts.jsonl"):
        r = json.loads(line)
        canon[r["case_id"]].append(r)
    pred_stat = {r["doc_id"]: r["scores"]
                 for r in (json.loads(l) for l in open(paths.INTERIM / "predicted_statutes.jsonl"))}
    retrieved = {r["doc_id"]: r["hits"]
                 for r in (json.loads(l) for l in
                           open(paths.INTERIM / "retrieved_precedents.jsonl"))}

    fam_path = paths.INTERIM / "claim_families.json"
    membership = (json.loads(fam_path.read_text()).get("membership", {})
                  if fam_path.exists() else {})

    import yaml
    ont = yaml.safe_load((paths.ONTOLOGY / "ontology_v1.yaml").read_text())
    elements_by_claim = {c["name"]: [e["name"] for e in c["elements"]] for c in ont["claims"]}

    def usable(ids):
        return [c for c in ids if c in y_of and masked.get(c) and canon.get(c)]

    tr, te = usable(split["train"]), usable(split["test"])
    rng = np.random.default_rng(SEED)
    if args.n and len(te) > args.n:
        te = [te[i] for i in sorted(rng.choice(len(te), args.n, replace=False))]
    yte = np.array([y_of[c] for c in te])
    print(f"=== §3.3 LLM baselines, split={args.split}, model={args.model} ===")
    print(f"train pool {len(tr)}, test {len(te)}, test WIN {yte.mean():.3f}")

    # --- few-shot retrieval index over TRAIN cases only, time-respecting at query time
    fs_idx = None
    if "LLM-FS" in args.arms:
        print("building few-shot retrieval index over train cases...", flush=True)
        scrub_tr = {d: scrub_citations(masked[d]) for d in tr}
        fs_idx = BM25Index()
        fs_idx.build(lambda: ((d, years[d], scrub_tr[d]) for d in tr), verbose=False)

    # --- element list for CoT, from the case's own claim families
    def elements_for(cid: str) -> str:
        fams = membership.get(cid, {}).get("families", [])
        names: list[str] = []
        for c in ont["claims"]:
            if any(f.lower() in c["name"].lower() or c["slug"].startswith(str(f).lower())
                   for f in fams):
                names += elements_by_claim[c["name"]]
        if not names:    # no family assigned: offer the most common claims' elements
            for c in ont["claims"][:3]:
                names += elements_by_claim[c["name"]]
        seen, out = set(), []
        for n in names:
            if n not in seen:
                seen.add(n); out.append(n)
        return "\n".join(f"- {n}" for n in out[:14])

    def build_msgs(arm: str, cid: str):
        sysmsg = client.load_prompt(PROMPTS[arm])
        if arm == "LLM-0":
            return [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": masked[cid][:MAX_CHARS]}]
        if arm == "LLM-FS":
            yq = years[cid]
            hits = fs_idx.search(scrub_citations(masked[cid]), args.k_shot * 3,
                                 before_year=yq - 1, exclude={cid})
            ex = []
            for d, _s in hits[:args.k_shot]:
                ex.append(f"EXAMPLE ({years[d]}) — outcome: "
                          f"{'WIN' if y_of[d] == 1 else 'LOSE'}\n"
                          f"{masked[d][:2500]}")
            body = ("\n\n---\n\n".join(ex) if ex else "(no earlier example available)")
            return [{"role": "system", "content": sysmsg},
                    {"role": "user", "content":
                     f"{body}\n\n===\n\nCASE TO PREDICT:\n{masked[cid][:9000]}"}]
        if arm == "LLM-CoT":
            return [{"role": "system", "content": sysmsg},
                    {"role": "user", "content":
                     f"ELEMENTS THAT MAY APPLY:\n{elements_for(cid)}\n\n"
                     f"CASE:\n{masked[cid][:MAX_CHARS]}"}]
        if arm == "LLM-CF":
            return [{"role": "system", "content": sysmsg},
                    {"role": "user", "content": f"CANONICAL FACTS:\n{fact_bundle(canon[cid])}"}]
        # LLM-CF+A
        st = sorted(pred_stat.get(cid, {}).items(), key=lambda kv: -kv[1])[:8]
        st_txt = "\n".join(f"- {k}  (p={v})" for k, v in st) or "(none predicted)"
        pr = []
        for h in retrieved.get(cid, [])[:5]:
            d = h["doc_id"]
            o = y_of.get(d) if d in set(tr) else None
            pr.append(f"- {reg.get(d, {}).get('title', d)[:70]} ({years.get(d)}), "
                      f"similarity {h['score']}, analogous party: "
                      f"{'WIN' if o == 1 else 'LOSE' if o == 0 else 'unknown'}")
        pr_txt = "\n".join(pr) or "(none retrieved)"
        return [{"role": "system", "content": sysmsg},
                {"role": "user", "content":
                 f"CANONICAL FACTS:\n{fact_bundle(canon[cid], 40)}\n\n"
                 f"PREDICTED STATUTES:\n{st_txt}\n\n"
                 f"RETRIEVED PRECEDENTS:\n{pr_txt}"}]

    def run_one(arm: str, cid: str) -> dict:
        schema = VerdictCoT if arm == "LLM-CoT" else Verdict
        res = client.complete(build_msgs(arm, cid), model=args.model, json_schema=schema,
                              prompt_id=PROMPTS[arm],
                              max_tokens=900 if arm == "LLM-CoT" else 80)
        if not res.ok:
            return {"doc_id": cid, "status": "dropped"}
        if res.parsed is None or res.parsed.outcome not in ("WIN", "LOSE"):
            return {"doc_id": cid, "status": "unusable"}
        return {"doc_id": cid, "status": "ok", "pred": res.parsed.outcome,
                "confidence": float(res.parsed.confidence)}

    results = []
    preds: dict[str, np.ndarray] = {}
    for arm in args.arms:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda c: run_one(arm, c), te))
        ok = [r for r in rows if r["status"] == "ok"]
        dropped = sum(1 for r in rows if r["status"] == "dropped")
        unusable = sum(1 for r in rows if r["status"] == "unusable")
        if not ok:
            print(f"{arm}: no usable predictions (dropped {dropped}, unusable {unusable})")
            continue
        idx = {r["doc_id"]: r for r in ok}
        sub = [c for c in te if c in idx]
        y = np.array([y_of[c] for c in sub])
        p = np.array([1 if idx[c]["pred"] == "WIN" else 0 for c in sub])
        # confidence-weighted score, so AUROC is not computed over two tied values
        s = np.array([idx[c]["confidence"] if idx[c]["pred"] == "WIN"
                      else 1 - idx[c]["confidence"] for c in sub])
        r = metrics.evaluate(y, p, s, name=arm)
        r["dropped"] = dropped
        r["unusable"] = unusable
        r["n_scored"] = len(sub)
        results.append(r)
        preds[arm] = (sub, y, p)
        print("  " + metrics.format_row(r) + f"  dropped={dropped} unusable={unusable}")

    # --- paired tests between arms that scored the same cases
    pairs = {}
    for a in ("LLM-FS", "LLM-CoT", "LLM-CF", "LLM-CF+A"):
        if a in preds and "LLM-0" in preds:
            sa, ya, pa = preds[a]
            sb, _yb, pb = preds["LLM-0"]
            common = [c for c in sa if c in set(sb)]
            if len(common) < 30:
                continue
            ia = {c: i for i, c in enumerate(sa)}
            ib = {c: i for i, c in enumerate(sb)}
            yy = np.array([y_of[c] for c in common])
            pairs[f"{a} vs LLM-0"] = metrics.mcnemar(
                yy, np.array([pa[ia[c]] for c in common]),
                np.array([pb[ib[c]] for c in common]))
    if pairs:
        print("\n  paired tests vs LLM-0 (McNemar, exact):")
        for k, v in pairs.items():
            print(f"    {k}: {v}")

    out = paths.EXPERIMENTS / "paper3" / f"llm_baselines_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"split": args.split, "model": args.model,
                               "prompts": PROMPTS, "n_test": len(te),
                               "k_shot": args.k_shot,
                               "results": results, "mcnemar": pairs,
                               "usage": client.usage()}, indent=1))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
