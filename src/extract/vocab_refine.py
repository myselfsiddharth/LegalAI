"""§6.2 steps 3-4: cluster the out-of-vocabulary proposals, name them, and propose `vocab_v2`.

§6.2 prescribes a loop: label a sample allowing a `NEW:` escape, cluster the proposals, have the
LLM name the clusters, human-review, merge, and repeat until the NEW rate is under 5% -- then
freeze. This is the machinery for one turn of that loop.

Measured going in: the out-of-vocabulary rate is **~18.8%**, far above the 5% freeze threshold.
Note that this is the rate after correcting the canonicaliser's own accounting -- it reported 0.7%
because it only counted proposals it had explicitly flagged `NEW:`, while a sixth of its answers
were labels absent from the vocabulary that it had not flagged. Trusting the flag would have frozen
a vocabulary that could not name a sixth of the facts.

Nothing here edits the vocabulary directly. It writes a **proposal** file plus a CHANGELOG entry,
because §0.6 requires a data-backed justification per change and §6.2 step 3 requires human review
before a merge. `--apply` writes `vocab_v2.yaml`, and even then every added entry is tagged with
its provenance and the cluster size behind it.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import date

import numpy as np
import yaml
from pydantic import BaseModel
from sklearn.cluster import HDBSCAN

from src import paths
from src.extract.vocab import slug
from src.llm import client

PROMPT_ID = "name_vocab_cluster.v1"
CANON = paths.INTERIM / "canonical_facts.jsonl"
PROPOSAL_OUT = paths.ONTOLOGY / "vocab_v2_proposal.json"
VOCAB_V1 = paths.ONTOLOGY / "vocab_v1.yaml"
VOCAB_V2 = paths.ONTOLOGY / "vocab_v2.yaml"
CHANGELOG = paths.ONTOLOGY / "CHANGELOG.md"

MIN_CLUSTER = 5
FREEZE_THRESHOLD = 0.05


class NameJSON(BaseModel):
    category: str
    subcategory: str
    definition: str = ""
    merge_into: str | None = None
    confidence: float = 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--apply", action="store_true",
                    help="write vocab_v2.yaml (default: propose only, for review)")
    ap.add_argument("--min-cluster", type=int, default=MIN_CLUSTER)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(CANON)]
    total = len(rows)
    new_rows = [r for r in rows if r.get("is_new")]
    rate = len(new_rows) / max(1, total)
    print(f"canonical facts: {total:,}   out of vocabulary: {len(new_rows):,} ({rate:.1%})")
    print(f"  §6.2 freezes the vocabulary below {FREEZE_THRESHOLD:.0%}\n")
    if not new_rows:
        print("nothing to refine.")
        return

    vocab = yaml.safe_load(VOCAB_V1.read_text())
    existing = {f"{c}.{s}" for c, cc in vocab["categories"].items()
                for s in cc["subcategories"]}

    # Cluster the proposals by meaning, using the FACT TEXT rather than the proposed label:
    # two labellers inventing different words for the same thing produce different labels but
    # similar facts, and it is the concept we want to group.
    texts = [f"{r.get('new_proposal') or r['subcategory']} -- {r['fact_text'][:200]}"
             for r in new_rows]
    print(f"embedding {len(texts):,} proposals...", flush=True)
    X = client.embed(texts, batch_size=32)
    keep = X.any(axis=1)
    if not keep.all():
        print(f"  {int((~keep).sum())} embeddings never arrived; excluded")
        X = X[keep]
        new_rows = [r for r, k in zip(new_rows, keep) if k]

    lab = HDBSCAN(min_cluster_size=args.min_cluster, metric="euclidean").fit_predict(X)
    clusters = defaultdict(list)
    for r, c in zip(new_rows, lab):
        clusters[int(c)].append(r)
    n_out = len(clusters.pop(-1, []))
    print(f"  {len(clusters)} clusters; {n_out} proposals left as outliers "
          f"({100*n_out/len(new_rows):.1f}%)\n")

    cat_list = ", ".join(sorted(vocab["categories"]))
    proposals = []
    for cid, members in sorted(clusters.items(), key=lambda x: -len(x[1])):
        sample = [f"- {m.get('new_proposal') or m['subcategory']}  <- {m['fact_text'][:110]}"
                  for m in members[:12]]
        res = client.complete(
            [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
             {"role": "user", "content":
              f"EXISTING CATEGORIES: {cat_list}\n\n"
              f"PROPOSALS IN THIS GROUP ({len(members)} facts):\n" + "\n".join(sample)}],
            model=args.model, json_schema=NameJSON, prompt_id=PROMPT_ID, max_tokens=300)
        if not res.ok or res.parsed is None:
            continue
        p = res.parsed
        key = f"{p.category}.{slug(p.subcategory)}"
        proposals.append({
            "cluster": cid, "n_facts": len(members),
            "category": p.category, "subcategory": slug(p.subcategory),
            "key": key, "definition": p.definition,
            "merge_into": p.merge_into if (p.merge_into in existing) else None,
            "merge_into_raw": p.merge_into,
            "confidence": round(float(p.confidence), 3),
            "already_in_vocab": key in existing,
            "is_not_a_fact": p.category.strip().upper().startswith("NEW:NOTAFACT"),
            "examples": [m["fact_text"][:120] for m in members[:4]],
        })

    add = [p for p in proposals
           if not p["merge_into"] and not p["already_in_vocab"] and not p["is_not_a_fact"]
           and p["confidence"] >= 0.5]
    merge = [p for p in proposals if p["merge_into"]]
    reject = [p for p in proposals if p["is_not_a_fact"]]
    lowconf = [p for p in proposals if p["confidence"] < 0.5 and not p["merge_into"]]
    covered = sum(p["n_facts"] for p in add) + sum(p["n_facts"] for p in merge)
    projected = (len(new_rows) - covered) / max(1, total)

    out = {"measured": {"n_facts": total, "n_out_of_vocab": len(new_rows),
                        "out_of_vocab_rate": round(rate, 4),
                        "n_clusters": len(clusters), "n_outliers": n_out},
           "proposed_additions": add, "proposed_merges": merge,
           "rejected_not_a_fact": reject, "low_confidence_needs_review": lowconf,
           "projected_out_of_vocab_rate": round(projected, 4),
           "freeze_threshold": FREEZE_THRESHOLD}
    PROPOSAL_OUT.write_text(json.dumps(out, indent=1))

    print(f"  {'':3s} {'n':>5s}  {'conf':>5s}  key / action")
    for p in proposals[:28]:
        action = ("MERGE -> " + p["merge_into"] if p["merge_into"]
                  else "reject: not a fact" if p["is_not_a_fact"]
                  else "already present" if p["already_in_vocab"]
                  else "review (low conf)" if p["confidence"] < 0.5
                  else "ADD")
        print(f"      {p['n_facts']:5d}  {p['confidence']:5.2f}  {p['key'][:44]:44s} {action}")
    print(f"\n  additions {len(add)} · merges {len(merge)} · rejected as non-facts "
          f"{len(reject)} · needs review {len(lowconf)}")
    print(f"  out-of-vocabulary rate: {rate:.1%} -> projected {projected:.1%} "
          f"({'below' if projected < FREEZE_THRESHOLD else 'STILL ABOVE'} "
          f"the {FREEZE_THRESHOLD:.0%} freeze threshold)")
    print(f"\n-> {PROPOSAL_OUT}")

    if not args.apply:
        print("  (proposal only; §6.2 step 3 wants human review. Re-run with --apply to write "
              "vocab_v2.yaml.)")
        return

    for p in add:
        cat = p["category"] if p["category"] in vocab["categories"] else p["category"].lstrip("NEW:")
        vocab["categories"].setdefault(cat, {"n": 0, "subcategories": {}})
        vocab["categories"][cat]["subcategories"][p["subcategory"]] = {
            "name": p["subcategory"].replace("_", " "), "definition": p["definition"],
            "source": f"vocab_refine:cluster_{p['cluster']}_n{p['n_facts']}",
            "provenance": "llm_named_unreviewed"}
    for c in vocab["categories"].values():
        c["n"] = len(c["subcategories"])
    vocab["version"] = 2
    vocab["note"] = (f"v2: {len(add)} subcategories added from {len(clusters)} clusters of "
                     f"out-of-vocabulary proposals. Added entries are tagged "
                     f"provenance=llm_named_unreviewed.")
    VOCAB_V2.write_text(yaml.safe_dump(vocab, sort_keys=False, allow_unicode=True, width=100))

    with CHANGELOG.open("a") as f:
        f.write(f"\n## {date.today().isoformat()} — vocabulary v1 -> v2\n")
        f.write(f"`src/extract/vocab_refine.py`, prompt `{PROMPT_ID}`, model `{args.model}`.\n\n")
        f.write(f"Out-of-vocabulary rate measured at **{rate:.1%}** over {total:,} canonical "
                f"facts, against §6.2's {FREEZE_THRESHOLD:.0%} freeze threshold.\n\n")
        f.write(f"- {len(new_rows):,} out-of-vocabulary facts clustered into {len(clusters)} "
                f"groups ({n_out} outliers)\n")
        f.write(f"- **{len(add)} subcategories added**, tagged `llm_named_unreviewed`\n")
        f.write(f"- {len(merge)} clusters folded into existing labels\n")
        f.write(f"- {len(reject)} clusters rejected as not facts (statutes, cited cases, legal "
                f"principles)\n")
        f.write(f"- {len(lowconf)} clusters flagged for review (confidence < 0.5)\n")
        f.write(f"- projected out-of-vocabulary rate: **{projected:.1%}**\n\n")
        f.write("Added entries are **not reviewed**. §6.2 step 3 requires human review before a "
                "vocabulary is frozen, and nothing here should be described as frozen.\n")
    print(f"-> {VOCAB_V2} (+ §6.3 entry)")


if __name__ == "__main__":
    main()
