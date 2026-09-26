"""Dense document index over the cached corpus, for comparison against BM25.

BM25 indexes every word of a judgment; a single embedding of the first few hundred words
would be a straw man. So each document is represented by CHUNKS_PER_DOC chunks sampled
evenly across its whole length, embedded and mean-pooled into one L2-normalised vector.
That keeps the document's full span in view at a fraction of the cost of embedding every
chunk (~55k calls instead of ~273k).

Embeddings come from ASU's Voyager endpoint, which is free for this project, so call volume
is not the constraint -- wall clock is. Resumable: vectors already on disk are skipped.

Usage: python3 scripts/build_dense_index.py
"""
import argparse
import json

import numpy as np

from ode_lib import ROOT, get_client
from run_baselines import chunk_text, embed

CORPUS = ROOT / "Data" / "processed" / "corpus_text.jsonl"
VECS = ROOT / "Data" / "processed" / "dense_index.npy"
META = ROOT / "Data" / "processed" / "dense_index_meta.json"

CHUNKS_PER_DOC = 10


def doc_chunks(text: str, n: int) -> list[str]:
    chunks = chunk_text(text)
    if len(chunks) <= n:
        return chunks
    step = len(chunks) / n
    return [chunks[int(i * step)] for i in range(n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-embedding-8b")
    ap.add_argument("--chunks", type=int, default=CHUNKS_PER_DOC)
    ap.add_argument("--limit", type=int, default=0, help="stop after N docs (smoke test)")
    ap.add_argument("--group", type=int, default=24,
                    help="documents whose chunks share one embed() call sequence")
    args = ap.parse_args()

    done_ids, done_vecs = [], None
    if META.exists() and VECS.exists():
        done_ids = json.loads(META.read_text())["doc_ids"]
        done_vecs = np.load(VECS)
        print(f"resuming: {len(done_ids)} vectors already built")
    done = set(done_ids)

    client = get_client()
    ids, rows = list(done_ids), ([done_vecs] if done_vecs is not None else [])
    pending = []
    with CORPUS.open() as f:
        for line in f:
            d = json.loads(line)
            if d["doc_id"] in done:
                continue
            pending.append((d["doc_id"], d["title"], d["text"]))
            if args.limit and len(pending) >= args.limit:
                break
    print(f"{len(pending)} documents to embed, {args.chunks} chunks each", flush=True)

    # One API call per document wastes most of each request: embed() batches at 32, so
    # flattening several documents' chunks into one call cuts the number of round trips
    # (the dominant cost) by roughly the number of documents per group.
    batch_ids, batch_vecs = [], []
    done_n = 0
    for start in range(0, len(pending), args.group):
        group = pending[start: start + args.group]
        flat, spans = [], []
        for doc_id, title, text in group:
            pieces = doc_chunks(text, args.chunks)
            if not pieces:
                spans.append((doc_id, 0))
                continue
            # the title carries the parties and is the one line a human would skim first
            pieces = [f"{title}\n{pieces[0]}"] + pieces[1:]
            spans.append((doc_id, len(pieces)))
            flat.extend(pieces)
        try:
            vecs = np.asarray(embed(client, args.model, flat), dtype=np.float32)
        except Exception as e:
            print(f"  FAIL group at {start}: {type(e).__name__}: {e}", flush=True)
            continue
        off = 0
        for doc_id, count in spans:
            if not count:
                continue
            v = vecs[off: off + count].mean(axis=0)
            off += count
            v = v / (np.linalg.norm(v) or 1.0)
            batch_ids.append(doc_id)
            batch_vecs.append(v)
        done_n += len(group)

        if batch_vecs and (done_n % (args.group * 4) == 0 or start + args.group >= len(pending)):
            ids.extend(batch_ids)
            rows.append(np.vstack(batch_vecs))
            batch_ids, batch_vecs = [], []
            np.save(VECS, np.vstack(rows))
            META.write_text(json.dumps({"doc_ids": ids, "model": args.model,
                                        "chunks_per_doc": args.chunks}))
            print(f"  {done_n}/{len(pending)} embedded ({len(ids)} on disk)", flush=True)

    print(f"done: {len(ids)} vectors -> {VECS}")


if __name__ == "__main__":
    main()
