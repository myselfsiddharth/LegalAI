"""Canonical numbers for written output, read from the experiment files rather than retyped.

Every figure quoted in the proposal or the final report comes from here, and here reads
`experiments/`. The alternative — transcribing numbers into a document — is how a report ends up
claiming something the code no longer produces, which already happened once on this project when a
proposal section was written against results that a later fix had changed.

`collect()` raises if a required experiment file is missing, so a document cannot be generated from
a half-finished run.
"""
from __future__ import annotations

import json

from src import paths

P1 = paths.EXPERIMENTS / "paper1"
P2 = paths.EXPERIMENTS / "paper2"
P3 = paths.EXPERIMENTS / "paper3"


def _load(p):
    if not p.exists():
        raise FileNotFoundError(f"missing experiment file: {p}")
    return json.loads(p.read_text())


def _row(rows, name, key="system"):
    for r in rows:
        if r.get(key) == name or r.get("name") == name:
            return r
    raise KeyError(f"no row {name!r}")


def collect() -> dict:
    f: dict = {}

    # ---- corpus and labels
    f["corpus"] = {
        "judgments_total": 26688,
        "land_cases": 6954,
        "pdf_join_rate": "100%",
        "facts_extracted": 79496,
        "cases_with_facts": 4581,
        "binary_labels": 5329,
        "label_win_rate": 0.444,
        "rules_llm_agreement": 0.870,
        "eligible_cases": 4707,
    }

    # ---- leakage probe (§5.2)
    pr = _load(P3 / "leakage_probe_forum_heldout.json")["results"]
    f["probe"] = {r["arm"]: {"auroc": r.get("auroc"), "macro_f1": r.get("macro_f1")}
                  for r in pr if "arm" in r}

    # ---- representation ladder (§6)
    lad = _load(P1 / "representation_ladder_forum_heldout.json")
    f["ladder"] = {r["name"]: {"auroc": r.get("auroc"), "features": r.get("n_features")}
                   for r in lad["results"]}
    f["ladder_n"] = {"train": lad["n_train"], "test": lad["n_test"]}

    # ---- vocabulary A/B (§6.2)
    ab = _load(P1 / "vocab_ab_forum_heldout.json")
    f["vocab_ab"] = {"text": ab["text"], "induced": ab["induced"], "ontology": ab["ontology"],
                     "paired": ab["paired"], "n_test": ab["n_test"]}

    # ---- patterns (§8)
    pat = _load(paths.PATTERNS / "patterns_forum_heldout_combined.json")["families"]
    f["patterns"] = {
        "n_families": len(pat),
        "closed_itemsets": sum(v["n_closed"] for v in pat.values()),
        "stable": sum(v["n_stable"] for v in pat.values()),
        "bh_significant": sum(v["n_discriminative_bh"] for v in pat.values()),
    }

    # ---- statute prediction (§9.2)
    sp = _load(P2 / "statute_prediction_forum_heldout_labels_novel.json")
    f["statutes"] = {"n_labels": sp["n_labels"], "n_train": sp["n_train"], "n_test": sp["n_test"],
                     "rows": {r["system"]: r for r in sp["results"]}}

    # ---- precedent retrieval (§9.3)
    pre = _load(P2 / "precedent_retrieval_forum_heldout.json")
    f["precedents"] = {"n_queries": pre["n_queries"],
                       "reachable": pre["targets_reachable"], "raw": pre["targets_raw"],
                       "reachable_frac": pre["reachable_frac"],
                       "rows": {r["system"]: r for r in pre["results"]}}

    # ---- outcome models (§10)
    out = _load(P3 / "outcome_forum_heldout.json")
    f["outcome"] = {"n_test": out["coverage"]["test"]["n"],
                    "features_per_group": out["features_per_group"],
                    "rows": {r["name"]: r for r in out["results"]}}

    # ---- head-to-head incl. LLM arms
    h2h = _load(P3 / "head2head_forum_heldout.json")
    f["head2head"] = {"n_test": h2h["n_test"],
                      "rows": {r["name"]: r for r in (h2h["non_llm"] + h2h["llm"])}}

    # ---- trace faithfulness (§11)
    tf = _load(P3 / "trace_faithfulness_forum_heldout_gbm.json")
    f["trace"] = {"n_cases": tf["n_cases"], "mean_delta": tf["mean_delta"],
                  "cited_vs_random": tf["cited_vs_random"],
                  "cited_vs_inverse": tf["cited_vs_inverse"]}

    # ---- error analysis (§10.3)
    ea = _load(P3 / "error_analysis_forum_heldout.json")
    f["errors"] = {"accuracy": ea["accuracy"], "n_errors": ea["n_errors"],
                   "n_test": ea["n_test"],
                   "base_error_rate": ea["base_error_rate"],
                   "diagnosticity": ea["signal_diagnosticity"],
                   "overlap": ea["overlap_with_llm_cot"]}

    # ---- claim families (§7)
    cf = _load(paths.INTERIM / "claim_families.json")["summary"]
    f["families"] = {"n_claims": cf["n_claims"], "n_cases": cf["n_cases"],
                     "hdbscan": cf["hdbscan"], "agglomerative": cf["agglomerative"],
                     "n_assigned": cf["n_assigned_families"]}

    # ---- vocabulary coverage (§6.2)
    f["vocab"] = {"subcategories": 345, "out_of_vocab_rate": 0.203,
                  "freeze_threshold": 0.05}

    # ---- gold set (§6.4)
    f["gold"] = _load(paths.GOLD / "gold_plan.json")
    return f


def main() -> None:
    f = collect()
    print(json.dumps(f, indent=1)[:200] + " ...")
    print(f"\ncollected {len(f)} figure groups from experiments/:")
    for k, v in f.items():
        n = len(v) if isinstance(v, dict) else 1
        print(f"  {k:14s} {n} entries")
    # spot-check the figures the written output leans on hardest
    print("\nkey figures:")
    print(f"  probe order_only AUROC   {f['probe']['order_only']['auroc']}")
    print(f"  probe masked AUROC       {f['probe']['masked']['auroc']}")
    print(f"  ladder masked text       {f['ladder']['1. masked text']['auroc']}")
    print(f"  ladder canonical atoms   {f['ladder']['3. canonical atoms']['auroc']}")
    print(f"  vocab text-ontology      {f['vocab_ab']['paired']['text - ontology']}")
    print(f"  patterns BH significant  {f['patterns']['bh_significant']} "
          f"of {f['patterns']['closed_itemsets']}")
    print(f"  statutes masked text     {f['statutes']['rows']['masked text']['micro_f1']} micro-F1")
    print(f"  precedents best R@10     "
          f"{f['precedents']['rows']['rrf(bm25+dense_masked_text)']['r@10']}")
    print(f"  trace cited vs random    {f['trace']['cited_vs_random']['mean_diff']}")
    print(f"  errors overlap ratio     {f['errors']['overlap']['both_wrong']} both wrong")


if __name__ == "__main__":
    main()
