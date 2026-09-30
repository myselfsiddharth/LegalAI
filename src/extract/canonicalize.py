"""Stage 2 (§6.2): free-text facts -> canonical labels drawn from a controlled vocabulary.

Why a vocabulary at all: FP-Growth (§8) mines sets of labels, so "he bought the land under a
registered sale deed" and "a sale deed was executed and registered in his favour" must reduce to
the same atom or no pattern can ever reach support. Free text cannot be itemised.

Two design decisions that matter downstream:

* **Candidates are retrieved, not dumped.** The seed vocabulary has 345 subcategories across 15
  categories; putting all of them in every prompt would be long, expensive and would push the
  model toward whatever appears early in the list. We embed the vocabulary once and retrieve the
  nearest candidates per fact, so the choice is over a short, relevant menu.

* **`NEW:` is a first-class answer.** §6.2's loop depends on it: label a sample, cluster the NEW
  proposals, review, merge, repeat until the NEW rate is under 5%, then freeze. A prompt that
  forces a choice from the seed list would hide exactly the signal that loop needs, and the
  resulting vocabulary would look complete while quietly mislabelling everything it lacks a word
  for.

Polarity and `proof_status` are properties rather than separate labels, but the distinction they
carry is required: §10.4 insists "not proved" and "proved false" are different facts, so a
burden-shift rule that cannot tell them apart is wrong.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field

import numpy as np
import yaml
from pydantic import BaseModel

from src import paths
from src.llm import client

PROMPT_ID = "canonicalize_fact.v1"
VOCAB_PATH = paths.ONTOLOGY / "vocab_v1.yaml"
OUT_PATH = paths.INTERIM / "canonical_facts.jsonl"
EMB_CACHE = paths.CACHE / "vocab_embeddings.npz"

N_CANDIDATES = 24
PROPERTY_KEYS = ("polarity", "proof_status", "actor",
                 "duration_years_bin", "amount_bin", "delay_years_bin")


class CanonJSON(BaseModel):
    category: str
    subcategory: str
    properties: dict = {}
    confidence: float = 0.5


@dataclass
class CanonicalFact:
    fact_id: str
    case_id: str
    category: str
    subcategory: str
    label: str                      # the mined atom, e.g. "possession.continuous.gt12y"
    properties: dict = field(default_factory=dict)
    confidence: float = 0.0
    is_new: bool = False            # the model proposed a label outside the vocabulary
    new_proposal: str | None = None
    fact_text: str = ""
    asserted_by: str = ""
    disputed_status: str = ""


def load_vocab() -> tuple[list[str], list[str], dict]:
    v = yaml.safe_load(VOCAB_PATH.read_text())
    keys, descs = [], []
    for cat, c in v["categories"].items():
        for sub, meta in c["subcategories"].items():
            keys.append(f"{cat}.{sub}")
            d = meta.get("definition") or ""
            descs.append(f"{cat}: {meta['name']}" + (f" -- {d}" if d else ""))
    return keys, descs, v


def vocab_matrix(keys: list[str], descs: list[str]) -> np.ndarray:
    """Embed the vocabulary once and cache it on disk; it changes only when the vocab does."""
    sig = str(len(keys)) + ":" + str(hash(tuple(keys)) & 0xFFFFFFFF)
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=False)
        if str(z.get("sig", np.array("")).item() if "sig" in z else "") == sig:
            return z["M"]
    M = client.embed(descs, verbose=True)
    np.savez(EMB_CACHE, M=M, sig=np.array(sig))
    return M


def candidates_for(fact_text: str, keys: list[str], M: np.ndarray, qvec: np.ndarray,
                   k: int = N_CANDIDATES) -> list[str]:
    # Voyager returns L2-normalised vectors, so a dot product IS cosine similarity.
    sims = M @ qvec
    idx = np.argsort(-sims)[:k]
    return [keys[i] for i in idx]


def make_label(category: str, subcategory: str, props: dict) -> str:
    """The atom FP-Growth will mine. Polarity and the strongest numeric bin are folded IN, because
    a pattern must distinguish "notice served" from "notice not served" and "possession over 12
    years" from "possession under 3" -- those are different legal facts, not the same fact with
    metadata."""
    parts = [category.lower(), re.sub(r"[^\w]+", "_", subcategory.lower()).strip("_")]
    if props.get("polarity") == "neg":
        parts.append("not")
    for k in ("duration_years_bin", "delay_years_bin", "amount_bin"):
        v = props.get(k)
        if v and v != "unknown":
            parts.append(v)
            break
    ps = props.get("proof_status")
    if ps in ("not_shown", "proved_false"):
        parts.append(ps)
    return ".".join(parts)


def canonicalize_one(fact: dict, keys: list[str], M: np.ndarray, model: str,
                     vocab_keys: set[str] | None = None) -> CanonicalFact | None:
    qvec = client.embed([fact["text"]])[0]
    if not qvec.any():
        return None                                  # embedding never arrived; NOT a null label
    cands = candidates_for(fact["text"], keys, M, qvec)
    menu = "\n".join(f"- {c}" for c in cands)
    msgs = [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
            {"role": "user", "content":
             f"FACT: {fact['text']}\n\nCONTEXT (how it was asserted): "
             f"asserted_by={fact['asserted_by']}, disputed_status={fact['disputed_status']}"
             f"{', event_date=' + fact['event_date'] if fact.get('event_date') else ''}\n\n"
             f"CANDIDATE LABELS:\n{menu}"}]
    res = client.complete(msgs, model=model, json_schema=CanonJSON, prompt_id=PROMPT_ID,
                          max_tokens=350)
    if not res.ok or res.parsed is None:
        return None
    p = res.parsed
    props = {k: v for k, v in (p.properties or {}).items() if k in PROPERTY_KEYS and v}

    sub_raw = (p.subcategory or "").strip()
    is_new = sub_raw.upper().startswith("NEW") or p.category.strip().upper() == "NEW"
    proposal = None
    if is_new:
        proposal = re.sub(r"^NEW\s*:?\s*", "", sub_raw, flags=re.I).strip() or sub_raw
        category = p.category.strip() if p.category.strip().upper() != "NEW" else "NEW"
        subcategory = proposal
    else:
        # the model may answer "Category.subcategory" in the subcategory field
        category = p.category.strip()
        subcategory = sub_raw.split(".")[-1] if "." in sub_raw else sub_raw

    # A label the model invented under a VALID category name is still a label outside the
    # vocabulary, and §6.2's freeze decision turns on that rate. Measured on a 150-fact pilot the
    # model marked only 0.7% as NEW while 16.7% of its answers were not in the vocabulary at all
    # -- so trusting the NEW flag alone would have frozen a vocabulary that could not name a sixth
    # of the facts. Out-of-vocabulary is decided here, not by the model.
    if vocab_keys is not None and not is_new:
        if f"{category}.{subcategory}" not in vocab_keys:
            is_new = True
            proposal = f"{category}.{subcategory}"

    return CanonicalFact(
        fact_id=fact["fact_id"], case_id=fact["case_id"],
        category=category, subcategory=subcategory,
        label=make_label(category, subcategory, props),
        properties=props, confidence=float(p.confidence),
        is_new=is_new, new_proposal=proposal,
        fact_text=fact["text"], asserted_by=fact["asserted_by"],
        disputed_status=fact["disputed_status"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of facts to label (0 = all)")
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    keys, descs, _ = load_vocab()
    vocab_keys = set(keys)
    print(f"vocabulary: {len(keys)} labels; embedding (cached after first run)...", flush=True)
    M = vocab_matrix(keys, descs)

    facts = [json.loads(l) for l in open(paths.INTERIM / "facts.jsonl")]
    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["fact_id"])
                except json.JSONDecodeError:
                    pass
    todo = [f for f in facts if f["fact_id"] not in done]
    if args.n:
        todo = todo[:args.n]
    print(f"{len(facts)} facts, {len(done)} already labelled, {len(todo)} to do", flush=True)

    stats = Counter()
    rows = []
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, cf in enumerate(
                pool.map(lambda f: canonicalize_one(f, keys, M, args.model, vocab_keys),
                         todo), 1):
            if cf is None:
                stats["dropped"] += 1           # never got an answer -- not a null label
                continue
            out.write(json.dumps(asdict(cf)) + "\n")
            rows.append(cf)
            stats["new" if cf.is_new else "mapped"] += 1
            if i % 100 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}  NEW rate "
                      f"{stats['new']/max(1,stats['new']+stats['mapped']):.1%}", flush=True)
    report(rows, stats, keys)


def report(rows, stats, keys) -> None:
    n = len(rows) or 1
    valid = set(keys)
    print(f"\n=== canonicalization: {len(rows)} facts labelled ===")
    print(f"  mapped to vocabulary: {stats['mapped']}  ({100*stats['mapped']/n:.1f}%)")
    print(f"  OUT OF VOCABULARY:    {stats['new']}  ({100*stats['new']/n:.1f}%)   "
          f"<- §6.2 freezes the vocabulary when this is under 5%")
    flagged = sum(1 for r in rows if r.is_new and r.new_proposal
                  and not r.new_proposal.count("."))
    print(f"      of which the model itself marked NEW: {flagged}; the rest were detected "
          f"by checking its answer against the vocabulary")
    if stats["dropped"]:
        print(f"  DROPPED (no answer):  {stats['dropped']}  -- not evidence of an unlabelable fact")
    off = [r for r in rows if not r.is_new and f"{r.category}.{r.subcategory}" not in valid]
    assert not off, f"BUG: {len(off)} out-of-vocab labels escaped the is_new check"
    cats = Counter(r.category for r in rows)
    print(f"\n  categories used ({len(cats)}):")
    for k, v in cats.most_common(12):
        print(f"    {k:18s} {v:4d}  {100*v/n:5.1f}%")
    atoms = Counter(r.label for r in rows)
    print(f"\n  distinct atoms: {len(atoms)} over {len(rows)} facts "
          f"(mean {len(rows)/max(1,len(atoms)):.1f} facts per atom)")
    for k, v in atoms.most_common(12):
        print(f"    {v:4d}  {k}")
    props = Counter(k for r in rows for k in r.properties)
    print(f"\n  property coverage: " + ", ".join(f"{k}={v}" for k, v in props.most_common()))
    if stats["new"]:
        print(f"\n  sample NEW proposals (input to the §6.2 clustering step):")
        for r in [x for x in rows if x.is_new][:10]:
            print(f"    {r.new_proposal[:60]:60s} <- {r.fact_text[:60]}")
    print(f"\n  usage: {client.usage()}")


if __name__ == "__main__":
    main()
