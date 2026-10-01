"""An INDUCED fact vocabulary, to isolate what canonicalisation actually costs.

The representation ladder shows canonical atoms at chance while the fact text they replace carries
signal, and `vocab_diagnosis` locates the loss in facts the ontology CAN name (-0.09 AUROC) rather
than in the unnameable tail (-0.00). That is consistent with two different claims, and they have
opposite consequences for the project:

  (i)  **This ontology's vocabulary is wrong for the task.** Its 345 labels were authored top-down
       from ontology enums; they may carve facts along distinctions that do not track outcomes.
       Remedy: induce the vocabulary from the corpus instead.

  (ii) **Any reduction to a few hundred symbols loses the signal.** 8,000 tf-idf features collapsing
       to ~350 discrete atoms may simply destroy the information, whatever the symbols are.
       Remedy: none at this granularity -- the itemisation that §8 requires is itself the problem,
       and FP-Growth over any such vocabulary is doomed.

This builds the control that separates them: cluster the FACT EMBEDDINGS bottom-up into the same
number of atoms the ontology has, with no ontology input at all. Compared on the ladder:

    fact text  ->  INDUCED atoms  ->  ontology atoms

If induced atoms keep the signal, the ontology is at fault (i) and a learned vocabulary is the fix.
If they lose it too, (ii) holds and §8's whole approach is mis-specified -- which is a finding about
pattern mining over legal facts, not a bug.

Embeddings come from the cache (canonicalisation already embedded every fact text), so this costs
no API calls. Clustering is fit on TRAIN cases only; test facts are assigned to the nearest centroid,
because fitting the vocabulary on all data would leak the test distribution into the representation.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src import paths
from src.llm import client

OUT_PATH = paths.INTERIM / "induced_facts.jsonl"
SEED = 573


def load_fact_texts() -> tuple[list[dict], np.ndarray]:
    """Facts plus their cached embeddings. A fact whose embedding is not cached is skipped rather
    than embedded now, so this stays an offline, zero-cost step."""
    facts = [json.loads(l) for l in open(paths.INTERIM / "facts.jsonl")]
    cache = client._get_cache()
    model = client.DEFAULT_EMBED_MODEL
    vecs, keep = [], []
    for f in facts:
        hit = cache.get(client._key("embed", model, f["text"], {}))
        if hit is None:
            continue
        vecs.append(np.array(json.loads(hit["response"]), dtype=np.float32))
        keep.append(f)
    if not vecs:
        raise SystemExit("no cached fact embeddings -- run canonicalisation first")
    return keep, np.vstack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--k", type=int, default=345,
                    help="number of induced atoms; defaults to the ontology vocabulary's size so "
                         "the comparison is at equal granularity")
    args = ap.parse_args()

    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    train_cases = set(split["train"])

    facts, X = load_fact_texts()
    print(f"facts with cached embeddings: {len(facts):,}")

    is_train = np.array([f["case_id"] in train_cases for f in facts])
    print(f"  fitting {args.k} clusters on {int(is_train.sum()):,} train facts "
          f"(test facts are assigned, never fitted)")
    km = MiniBatchKMeans(n_clusters=args.k, random_state=SEED, n_init=5, batch_size=2048)
    km.fit(X[is_train])
    assign = km.predict(X)

    # Name each cluster by its most frequent content words, for interpretability only. The AUROC
    # test does not depend on the names.
    import re
    STOP = set("the a an of to in and or is are was were be been for on by with that this it as at "
               "from not no such any which who whom whose his her their its had has have".split())
    words = defaultdict(Counter)
    for f, c in zip(facts, assign):
        for w in re.findall(r"[a-z]{4,}", f["text"].lower()):
            if w not in STOP:
                words[int(c)][w] += 1
    names = {c: "_".join(w for w, _ in words[c].most_common(3)) or f"c{c}"
             for c in range(args.k)}

    with OUT_PATH.open("w") as out:
        for f, c in zip(facts, assign):
            out.write(json.dumps({
                "fact_id": f["fact_id"], "case_id": f["case_id"],
                "label": f"ind.{int(c):03d}.{names[int(c)]}",
                "cluster": int(c),
                "asserted_by": f["asserted_by"], "disputed_status": f["disputed_status"],
                "fact_text": f["text"],
            }) + "\n")

    sizes = Counter(int(c) for c in assign)
    print(f"  cluster sizes: median {sorted(sizes.values())[len(sizes)//2]}, "
          f"min {min(sizes.values())}, max {max(sizes.values())}")
    print(f"  singleton clusters: {sum(1 for v in sizes.values() if v == 1)}")
    print(f"\n  sample induced atoms:")
    for c, _n in sizes.most_common(8):
        ex = next(f["text"] for f, a in zip(facts, assign) if a == c)
        print(f"    ind.{c:03d}.{names[c][:30]:30s} n={_n:5d}  e.g. {ex[:62]}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
