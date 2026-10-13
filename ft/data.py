"""Build the chunked datasets. Reads v1's artifacts; writes only inside ft/.

Masked text averages ~2,500 words against BERT's 512-token limit, so each case is tokenised once and
split into non-overlapping 512-token windows. Every chunk carries its case's label -- weak
supervision, since many chunks will not contain whatever signal exists, which is equally true of the
published hierarchical approach. At inference the chunk probabilities are averaged per case, so the
reported metric is always case-level.

Two arms, both built here:

  `masked`    facts and pleadings only. THE HEADLINE ARM.
  `last512`   the final 512 tokens of the order-removed text. Leaky by construction -- v1 showed
              that window is ~85% residual disposition language. It exists ONLY as a capability
              control (PLAN.md §3) and its number is never reported as a result.

The split is v1's hashed `forum_heldout`, so every number stays comparable, and the test set is used
for nothing but final evaluation.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from src import paths as v1
from src.data.label_merge import load_final

OUT = v1.ROOT / "ft" / "cache"
OUT.mkdir(parents=True, exist_ok=True)
MAXLEN = 512


def load_arm(arm: str) -> dict[str, str]:
    field = {"masked": "masked_text", "last512": "ildc_style_text"}[arm]
    out = {}
    for line in open(v1.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        t = r.get(field) or ""
        if arm == "last512":
            t = " ".join(t.split()[-512:])
        if t.strip():
            out[r["doc_id"]] = t
    return out


def build(arm: str, model_name: str) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    texts = load_arm(arm)
    labels = {d: (1 if r["outcome"] == "WIN" else 0)
              for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    split = json.loads((v1.SPLITS / "forum_heldout.json").read_text())
    where = {}
    for part in ("train", "dev", "test"):
        for c in split.get(part, []):
            where[c] = part

    rows = {"train": [], "dev": [], "test": []}
    n_cases = {"train": 0, "dev": 0, "test": 0}
    for d, t in texts.items():
        if d not in labels or d not in where:
            continue
        part = where[d]
        ids = tok(t, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        n_cases[part] += 1
        for s in range(0, len(ids), MAXLEN - 2):
            win = ids[s:s + MAXLEN - 2]
            if len(win) < 32:                      # a stub carries nothing; drop it
                continue
            rows[part].append({"case_id": d, "label": labels[d],
                               "input_ids": [tok.cls_token_id] + win + [tok.sep_token_id]})

    meta = {"arm": arm, "model": model_name, "max_len": MAXLEN,
            "cases": n_cases, "chunks": {k: len(v) for k, v in rows.items()}}
    for part, v in rows.items():
        np.save(OUT / f"{arm}_{part}_labels.npy",
                np.array([r["label"] for r in v], dtype=np.int64))
        with (OUT / f"{arm}_{part}.jsonl").open("w") as f:
            for r in v:
                f.write(json.dumps(r) + "\n")
    (OUT / f"{arm}_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"  {arm}: cases {n_cases}  chunks {meta['chunks']}")
    ch = rows["train"]
    if ch:
        lens = [len(r["input_ids"]) for r in ch]
        per = {}
        for r in ch:
            per[r["case_id"]] = per.get(r["case_id"], 0) + 1
        print(f"    chunk length mean {np.mean(lens):.0f}, chunks/case median "
              f"{int(np.median(list(per.values())))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="law-ai/InLegalBERT")
    ap.add_argument("--arms", nargs="+", default=["masked", "last512"])
    args = ap.parse_args()
    print(f"tokenising with {args.model}")
    for a in args.arms:
        build(a, args.model)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
