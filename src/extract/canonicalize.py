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
from pydantic import BaseModel, RootModel

from src import paths
from src.llm import client

PROMPT_ID = "canonicalize_fact.v2"
BATCH_FACTS = 8
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


class CanonItem(CanonJSON):
    i: int


class CanonBatch(RootModel):
    root: list[CanonItem]


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
                     vocab_keys: set[str] | None = None,
                     qvec: np.ndarray | None = None) -> CanonicalFact | None:
    # `qvec` is supplied by the caller, which embeds all facts in batches first. Embedding one
    # fact per call left the endpoint's batch unused; since Voyager is token-throughput-limited
    # the win is in issuing fewer requests, not in more concurrency.
    if qvec is None:
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


def _finish(fact: dict, p, vocab_keys: set[str] | None) -> CanonicalFact:
    """Turn one validated model answer into a CanonicalFact. Shared by the batched and
    single-fact paths so they cannot drift apart."""
    props = {k: v for k, v in (p.properties or {}).items() if k in PROPERTY_KEYS and v}
    sub_raw = (p.subcategory or "").strip()
    is_new = sub_raw.upper().startswith("NEW") or p.category.strip().upper() == "NEW"
    proposal = None
    if is_new:
        proposal = re.sub(r"^NEW\s*:?\s*", "", sub_raw, flags=re.I).strip() or sub_raw
        category = p.category.strip() if p.category.strip().upper() != "NEW" else "NEW"
        subcategory = proposal
    else:
        category = p.category.strip()
        subcategory = sub_raw.split(".")[-1] if "." in sub_raw else sub_raw
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


def canonicalize_batch(batch: list[tuple[dict, "np.ndarray"]], keys: list[str], M: "np.ndarray",
                       model: str, vocab_keys: set[str] | None
                       ) -> tuple[list[CanonicalFact], list[dict], Counter]:
    """Label up to BATCH_FACTS facts in ONE call.

    Why: one call per fact put canonicalisation at ~85,000 calls and 5.3 hours, against
    extraction's ~12,500. Batching is the whole difference between a 5-hour stage and a 40-minute
    one, and the task is per-fact independent so nothing is lost by sharing a call.

    The candidate menu is the UNION of each fact's own top-k, so every fact still sees its own
    shortlist -- a shared menu built any other way could crowd out the label a fact needs.

    Returns (labelled, needs_retry, stats). A fact the model skipped, duplicated or indexed out of
    range is NOT guessed at: it goes to `needs_retry` for a single-fact call. Silently dropping it
    would look like a fact the vocabulary could not name.
    """
    stats = Counter()
    facts = [f for f, _ in batch]
    menu: list[str] = []
    seen = set()
    for _f, q in batch:
        for c in candidates_for("", keys, M, q):
            if c not in seen:
                seen.add(c)
                menu.append(c)

    lines = []
    for i, f in enumerate(facts, 1):
        ctx = f"asserted_by={f['asserted_by']}, disputed_status={f['disputed_status']}"
        if f.get("event_date"):
            ctx += f", event_date={f['event_date']}"
        lines.append(f"{i}. {f['text']}\n   ({ctx})")
    msgs = [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
            {"role": "user", "content":
             "FACTS:\n" + "\n".join(lines) +
             "\n\nCANDIDATE LABELS:\n" + "\n".join(f"- {c}" for c in menu)}]

    res = client.complete(msgs, model=model, json_schema=CanonBatch, prompt_id=PROMPT_ID,
                          max_tokens=180 * len(facts) + 200)
    if not res.ok:
        stats["batch_dropped"] += 1
        return [], facts, stats                      # never answered: retry, do not discard
    if res.parsed is None:
        stats["batch_unparseable"] += 1
        return [], facts, stats

    out, used = [], set()
    for item in res.parsed.root:
        idx = item.i - 1
        if not (0 <= idx < len(facts)):
            stats["bad_index"] += 1
            continue
        if idx in used:
            stats["duplicate_index"] += 1
            continue
        used.add(idx)
        out.append(_finish(facts[idx], item, vocab_keys))
    missing = [f for i, f in enumerate(facts) if i not in used]
    stats["missing_from_batch"] += len(missing)
    stats["batch_ok"] += 1
    return out, missing, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400, help="number of facts to label (0 = all)")
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--batch", type=int, default=BATCH_FACTS,
                    help="facts per LLM call; 1 restores the one-call-per-fact behaviour")
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

    # Pre-embed every fact text in batches; a row of zeros means that text's embedding never
    # arrived, and those facts are reported as dropped rather than labelled from a zero vector.
    print(f"embedding {len(todo)} fact texts...", flush=True)
    Q = client.embed([f["text"] for f in todo], batch_size=32, verbose=False)
    n_missing = int((~Q.any(axis=1)).sum())
    if n_missing:
        print(f"  {n_missing} fact embeddings never arrived; they will be reported as dropped",
              flush=True)

    stats = Counter()
    rows = []
    pairs = list(zip(todo, Q))
    batches = [pairs[i:i + args.batch] for i in range(0, len(pairs), args.batch)]
    print(f"  {len(batches)} batches of up to {args.batch} facts "
          f"({len(todo)} facts / {args.batch} per call)", flush=True)

    retry: list[tuple[dict, object]] = []
    done_n = 0
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for labelled, missing, st in pool.map(
                lambda b: canonicalize_batch(b, keys, M, args.model, vocab_keys), batches):
            stats.update(st)
            for cf in labelled:
                out.write(json.dumps(asdict(cf)) + "\n")
                rows.append(cf)
                stats["new" if cf.is_new else "mapped"] += 1
            qmap = {f["fact_id"]: q for f, q in pairs}
            retry.extend((f, qmap[f["fact_id"]]) for f in missing)
            done_n += len(labelled) + len(missing)
            if done_n // 2000 != (done_n - len(labelled) - len(missing)) // 2000:
                out.flush()
                print(f"  {done_n}/{len(todo)}  NEW rate "
                      f"{stats['new']/max(1,stats['new']+stats['mapped']):.1%}  "
                      f"awaiting retry {len(retry)}", flush=True)

        # Facts the batch path skipped get their own call, so a batching failure never looks like
        # an unlabelable fact.
        if retry:
            print(f"  retrying {len(retry)} facts individually", flush=True)
            for cf in pool.map(
                    lambda iq: canonicalize_one(iq[0], keys, M, args.model, vocab_keys,
                                                qvec=iq[1]), retry):
                if cf is None:
                    stats["dropped"] += 1       # never got an answer -- not a null label
                    continue
                out.write(json.dumps(asdict(cf)) + "\n")
                rows.append(cf)
                stats["new" if cf.is_new else "mapped"] += 1
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
    if stats["batch_ok"] or stats["batch_dropped"]:
        print(f"  batching: {stats['batch_ok']} ok, {stats['batch_dropped']} dropped, "
              f"{stats['batch_unparseable']} unparseable; "
              f"{stats['missing_from_batch']} facts fell back to a single call "
              f"({stats['bad_index']} bad index, {stats['duplicate_index']} duplicate index)")
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
