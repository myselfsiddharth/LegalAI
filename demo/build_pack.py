"""Build a self-contained data pack so the demo loads instantly and cannot fail live.

A demo that trains a model on launch, or reads a 517 MB file, is a demo that hangs in front of an
audience. Everything shown is already computed and on disk, so this bundles it: curated cases with
their facts, element verdicts, predicted statutes and retrieved precedents; every headline number
pulled from the 49 result files; and a BM25 index over all 6,954 land cases so the
``paste your own facts'' panel can retrieve live without any model.

Nothing here recomputes a result. If a number is in the pack it came from a file under
``experiments/'', ``v2/experiments/'' or ``ft/experiments/''.
"""
from __future__ import annotations

import json
import pickle
import random
from collections import defaultdict

from src import paths as v1
from src.authorities.retrieval import BM25Index
from src.data.label_merge import load_final

PACK = v1.ROOT / "demo" / "pack"
PACK.mkdir(parents=True, exist_ok=True)
SEED = 573
N_CASES = 60


def jl(p):
    with open(p) as f:
        for line in f:
            yield json.loads(line)


def build_findings() -> dict:
    """Every number the demo displays, read from result files. No recomputation."""
    def g(p):
        try:
            return json.load(open(v1.ROOT / p))
        except Exception:
            return None

    out = {}
    ra = g("experiments/paper1/reasoning_ablation_forum_heldout.json")
    if ra:
        out["reasoning_ablation"] = {r["name"]: {"auroc": r["auroc"], "acc": r["accuracy"],
                                                 "words": r["mean_words"]} for r in ra["results"]}
        out["reasoning_paired"] = ra["paired"]

    out["setback"] = {}
    for name, p in (("land_forum", "experiments/paper1/leakage_setback_forum_heldout.json"),
                    ("land_temporal", "experiments/paper1/leakage_setback_temporal_2004_2013.json"),
                    ("corpus_temporal", "experiments/paper1/corpus_scale_setback_temporal.json"),
                    ("corpus_random", "experiments/paper1/corpus_scale_setback_random.json")):
        d = g(p)
        if d:
            out["setback"][name] = {
                "n_test": d["n_test"],
                "results": {r["name"]: r["auroc"] for r in d["results"]},
                "sweep": d.get("contamination_depth_sweep", []),
                "clean_at": d.get("advantage_gone_at_offset_words"),
                "cue_audit": d.get("cue_audit", {}),
                "paired": d.get("paired", {}),
                "by_era": d.get("by_era", {})}

    ld = g("experiments/paper1/representation_ladder_forum_heldout.json")
    if ld:
        out["ladder"] = [{"name": r.get("name"), "auroc": r.get("auroc"),
                          "feats": r.get("n_features")} for r in ld["results"]]
    vb = g("experiments/paper1/vocab_ab_forum_heldout.json")
    if vb:
        out["vocab_ab"] = vb.get("paired", {})
    s2 = g("v2/experiments/s2_soft_families_forum_heldout.json")
    if s2:
        out["s2"] = {k: s2[k] for k in ("best_temp", "best_auroc", "hard_argmax_auroc",
                                        "emb_mean_auroc", "v1_reference")}
    ws = g("v2/experiments/why_s4_fails_forum_heldout.json")
    if ws:
        out["why_s4"] = {"results": {r["name"]: r["auroc"] for r in ws["results"]},
                         "paired": ws["paired"],
                         "headroom": ws["headroom_text_minus_s2"]}
    nc = g("v2/experiments/named_clusters_k66_forum_heldout.json")
    if nc:
        out["clusters"] = {"kind_all": nc["kind_mix_all"], "kind_top": nc["kind_mix_top"],
                           "top": [nc["clusters"][j] for j in nc["top_order"][:20]]}
    sa = g("v2/experiments/s4_acceptance_forum_heldout.json")
    if sa:
        out["s4_acceptance"] = {"results": {r["name"]: r["auroc"] for r in sa["results"]},
                                "paired": sa["paired"],
                                "satisfied": sa["gated_satisfied"],
                                "not_satisfied": sa["gated_not_satisfied"]}
    el = g("v2/experiments/s4_element_layer_element_satisfy_v2_facts.json")
    if el:
        out["s4_layer"] = {"verdicts": el["verdicts"], "fabrication": el["fabrication_rate"],
                           "n_proposed": el["n_proposed"]}
    # three-seed summary first: ft_masked_InLegalBERT.json is overwritten per seed and holds
    # only the last one, so reading it would silently report a single run
    fs = g("ft/experiments/ft_masked_summary.json")
    if fs:
        out["ft_masked"] = {"auroc": fs["finetuned_mean"]["auroc"],
                            "acc": fs["finetuned_mean"]["acc"],
                            "sd": fs["finetuned_mean"]["auroc_sd"],
                            "seeds": sorted(fs["finetuned_runs"]), "control": False,
                            "bar": {"auroc": fs["tfidf_refit_same_cases"]["auroc"],
                                    "acc": fs["tfidf_refit_same_cases"]["acc"]},
                            "paired": fs["paired_vs_tfidf"]}
    ea = g("v2/experiments/s4_ai_reviewer_agreement.json")
    fr = g("v2/experiments/s4_findings_rerun.json")
    if ea and fr:
        out["element_quality"] = {
            "reviewer": "AI reviewer (claude-opus-5-5), blind; not a human",
            "n": ea["n_judgements"], "agreement_base": ea["overall_agreement"],
            "gold_by_arm": fr["gold_vs_ai_reviewer"]}
    for nm, p in (("ft_masked", "ft/experiments/ft_masked_InLegalBERT.json"),
                  ("ft_last512", "ft/experiments/ft_last512_InLegalBERT.json")):
        if nm in out:
            continue
        d = g(p)
        if d:
            out[nm] = {"auroc": d["test_case_auroc_mean"], "acc": d["test_case_acc_mean"],
                       "sd": d["test_case_auroc_sd"], "seeds": d["seeds"],
                       "control": d["is_capability_control"], "bar": d["v1_bar"]}
    st = g("experiments/paper2/statute_prediction_forum_heldout_labels_novel.json")
    if st:
        out["statutes"] = {r["system"]: r["micro_f1"] for r in st["results"]}
    pr = g("experiments/paper2/precedent_retrieval_forum_heldout.json")
    if pr:
        out["precedents"] = {r["system"]: {"r10": r["r@10"], "mrr": r["mrr"]}
                             for r in pr["results"]}
    oc = g("experiments/paper3/outcome_forum_heldout.json")
    if oc:
        keep = ("majority (per decade)", "gbm F+P+E+S+R+C", "lr S only", "gbm C only", "lr F")
        out["outcome"] = {r["name"]: {"acc": r["accuracy"], "mf1": r["macro_f1"],
                                      "auroc": r.get("auroc")}
                          for r in oc["results"] if r["name"] in keep}
    return out


def main() -> None:
    print("findings...", flush=True)
    findings = build_findings()
    (PACK / "findings.json").write_text(json.dumps(findings, indent=1))
    print(f"  {len(findings)} blocks")

    reg = {r["doc_id"]: r for r in jl(v1.INTERIM / "case_registry.jsonl")}
    labels = load_final()
    stat = {r["doc_id"]: r["scores"] for r in jl(v1.INTERIM / "predicted_statutes.jsonl")}
    prec = {r["doc_id"]: r["hits"] for r in jl(v1.INTERIM / "retrieved_precedents.jsonl")}

    facts = defaultdict(list)
    for r in jl(v1.INTERIM / "facts.jsonl"):
        facts[r["case_id"]].append(r)

    # Element verdicts: the TAGGED re-run (prior-court findings marked, prompt v3) wherever it
    # exists -- it is the better arm on the gold set -- and the original full run elsewhere.
    vdir = v1.ROOT / "v2" / "interim"
    pair = {}
    for name, src in (("element_verdicts_element_satisfy_v2_facts.jsonl", "full run"),
                      ("element_verdicts_findings_tag.jsonl", "tagged re-run")):
        if (vdir / name).exists():
            for r in jl(vdir / name):
                pair[(r["case_id"], r["claim_id"])] = dict(r, source=src)
    verd = defaultdict(list)
    for (c, _), r in pair.items():
        verd[c].append(r)
    pcf = vdir / "prior_court_findings.json"
    flagged = set(json.loads(pcf.read_text())) if pcf.exists() else set()
    ftypes = {}
    if (vdir / "fact_types.jsonl").exists():
        for r in jl(vdir / "fact_types.jsonl"):
            if r.get("type"):
                ftypes[r["fact_id"]] = r["type"]

    from v2.element_catalogue import all_elements, load_claims
    els = {e.element_id: {"name": e.name, "definition": e.definition,
                          "claim": e.claim_name} for e in all_elements(load_claims())}
    (PACK / "elements.json").write_text(json.dumps(els, indent=1))

    # curate: cases rich enough to be worth showing, spread across decades, and preferring
    # those with a NOT_SATISFIED verdict since that is the interesting half of the element layer
    def n_neg(c):
        return sum(1 for r in verd[c] for d in r["verdicts"]
                   if d["verdict"] == "NOT_SATISFIED" and d.get("gated"))
    cand = [c for c in facts
            if c in reg and c in stat and c in prec and verd.get(c) and len(facts[c]) >= 6]
    rng = random.Random(SEED)
    by_dec = defaultdict(list)
    for c in cand:
        by_dec[(reg[c]["year"] // 10) * 10].append(c)
    picked = []
    for dec in sorted(by_dec):
        pool = sorted(by_dec[dec], key=lambda c: (-n_neg(c), rng.random()))
        picked += pool[: max(2, N_CASES // max(1, len(by_dec)))]
    picked = picked[:N_CASES]

    cases = []
    for c in picked:
        lab = labels.get(c, {})
        cases.append({
            "doc_id": c, "title": reg[c]["title"], "year": reg[c]["year"],
            "url": reg[c].get("url"), "outcome": lab.get("outcome"),
            "facts": [{"text": f["text"], "asserted_by": f.get("asserted_by"),
                       "disputed_status": f.get("disputed_status"),
                       "prior_court_finding": f["fact_id"] in flagged,
                       "type": ftypes.get(f["fact_id"])} for f in facts[c][:40]],
            "statutes": sorted(stat[c].items(), key=lambda kv: -kv[1])[:10],
            "precedents": [{"doc_id": h["doc_id"], "score": h["score"],
                            "title": (reg.get(h["doc_id"]) or {}).get("title"),
                            "year": (reg.get(h["doc_id"]) or {}).get("year")}
                           for h in prec[c][:10]],
            "elements": [{"element_id": d["element_id"], "verdict": d["verdict"],
                          "quote": d.get("quote", ""), "favours": d.get("favours"),
                          "gated": d.get("gated", False), "source": r.get("source")}
                         for r in verd[c] for d in r["verdicts"]]})
    (PACK / "cases.json").write_text(json.dumps(cases, indent=1))
    print(f"  {len(cases)} curated cases across "
          f"{len({(c['year']//10)*10 for c in cases})} decades, "
          f"{sum(1 for c in cases for e in c['elements'] if e['verdict']=='NOT_SATISFIED')} "
          f"NOT_SATISFIED verdicts")

    print("BM25 over all land cases (for the paste-your-own-facts panel)...", flush=True)
    texts = {}
    for r in jl(v1.INTERIM / "masked_text.jsonl"):
        t = r.get("masked_text") or ""
        if t.strip():
            texts[r["doc_id"]] = t
    def records():
        for d, t in texts.items():
            yield d, (reg.get(d) or {}).get("year", 0), t
    idx = BM25Index()
    idx.build(records, verbose=False)
    titles = {d: {"title": (reg.get(d) or {}).get("title"),
                  "year": (reg.get(d) or {}).get("year"),
                  "url": (reg.get(d) or {}).get("url")} for d in texts}
    with (PACK / "bm25.pkl").open("wb") as f:
        pickle.dump({"index": idx, "titles": titles}, f)
    print(f"  indexed {len(texts):,} cases")
    print(f"\n-> {PACK}")
    for p in sorted(PACK.iterdir()):
        print(f"   {p.name:18s} {p.stat().st_size/1e6:6.1f} MB")


if __name__ == "__main__":
    main()
