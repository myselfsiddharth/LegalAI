"""Stage 1 (§5.2), second half: the `LLM-0` outcome-identification probe.

§5.2 asks for two leakage probes, and they fail differently. TF-IDF + logistic regression can
only exploit lexical overlap it saw in training, so it will miss a hint that requires reading
-- "the appellant has been in possession since the decree in his favour" gives the game away
to a reader and almost nothing to a bag of words. An LLM asked to name the outcome with no
reasoning catches that class.

The comparison that makes the number meaningful is the same one the TF-IDF probe uses: the
identical model and prompt are run on `masked_text` and on the ORDER REGION. The order arm is
the sensitivity control -- if the model cannot name the outcome when handed the operative
order, its failure on masked text says nothing.

Run on a split's test set so it is directly comparable to `leakage_probe.py`.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src import paths
from src.llm import client

PROMPT_ID = "probe_outcome.v1"
MAX_CHARS = 24_000          # keep well inside context; masked median is ~11k


class ProbeJSON(BaseModel):
    outcome: str
    confidence: float = 0.5


def _texts(arm: str) -> dict[str, str]:
    masked, order_start = {}, {}
    for line in open(paths.INTERIM / "masked_text.jsonl"):
        r = json.loads(line)
        masked[r["doc_id"]] = r["masked_text"]
        order_start[r["doc_id"]] = r.get("order_region_start")
    if arm == "masked":
        return masked
    out = {}
    for line in open(paths.CASE_TEXT):
        r = json.loads(line)
        if arm == "full":
            out[r["doc_id"]] = r["text"]
        else:                                        # order_only
            s = order_start.get(r["doc_id"])
            out[r["doc_id"]] = r["text"][s:] if s is not None else r["text"][-3000:]
    return out


def probe_one(doc_id: str, text: str, model: str) -> dict:
    msgs = [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
            {"role": "user", "content": text[:MAX_CHARS]}]
    res = client.complete(msgs, model=model, json_schema=ProbeJSON,
                          prompt_id=PROMPT_ID, max_tokens=60)
    if not res.ok:
        return {"doc_id": doc_id, "status": "dropped", "error": res.error}
    if res.parsed is None or res.parsed.outcome not in ("WIN", "LOSE"):
        return {"doc_id": doc_id, "status": "unusable", "raw": res.text[:120]}
    return {"doc_id": doc_id, "status": "ok", "pred": res.parsed.outcome,
            "confidence": float(res.parsed.confidence), "cached": res.cached}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="forum_heldout")
    ap.add_argument("--arms", nargs="+", default=["masked", "order_only"])
    ap.add_argument("--n", type=int, default=0, help="0 = whole test set")
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    from src.data.label_merge import load_final
    labels = {d: r["outcome"] for d, r in load_final().items() if r["outcome"] in ("WIN", "LOSE")}
    split = json.loads((paths.SPLITS / f"{args.split}.json").read_text())
    test = [d for d in split["test"] if d in labels]
    if args.n:
        test = test[:args.n]

    summary = {"split": args.split, "hash": split.get("hash"), "model": args.model,
               "prompt_id": PROMPT_ID, "n_test": len(test), "arms": {}}
    print(f"=== LLM-0 outcome-identification probe ===")
    print(f"split={args.split}  model={args.model}  prompt={PROMPT_ID}  n={len(test)}\n")

    for arm in args.arms:
        texts = _texts(arm)
        items = [(d, texts.get(d, "")) for d in test]
        items = [(d, t) for d, t in items if t.strip()]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda it: probe_one(it[0], it[1], args.model), items))

        status = Counter(r["status"] for r in rows)
        ok = [r for r in rows if r["status"] == "ok"]
        if not ok:
            print(f"{arm:12s} no usable predictions: {dict(status)}")
            continue
        y = np.array([1 if labels[r["doc_id"]] == "WIN" else 0 for r in ok])
        p = np.array([1 if r["pred"] == "WIN" else 0 for r in ok])
        # a confidence-weighted score, so AUROC is not computed over two tied values only
        s = np.array([(c if q == 1 else 1 - c)
                      for q, c in ((r["pred"] == "WIN", r["confidence"]) for r in ok)])
        acc = accuracy_score(y, p)
        mf1 = f1_score(y, p, average="macro")
        auc = roc_auc_score(y, s) if len(set(y)) > 1 else float("nan")
        res = {"n": len(ok), "accuracy": round(acc, 4), "macro_f1": round(mf1, 4),
               "auroc": round(float(auc), 4),
               "pred_win_rate": round(float(p.mean()), 4),
               "true_win_rate": round(float(y.mean()), 4),
               "dropped": status["dropped"], "unusable": status["unusable"]}
        summary["arms"][arm] = res
        print(f"{arm:12s} n={res['n']:4d}  acc={acc:.3f}  macroF1={mf1:.3f}  AUROC={auc:.3f}  "
              f"pred_WIN={res['pred_win_rate']:.2f} (true {res['true_win_rate']:.2f})  "
              f"dropped={res['dropped']} unusable={res['unusable']}")

    out = paths.EXPERIMENTS / "paper3" / f"llm_probe_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary["usage"] = client.usage()
    out.write_text(json.dumps(summary, indent=1))

    a = summary["arms"]
    if "order_only" in a and "masked" in a:
        print(f"\ninterpretation:")
        print(f"  given the operative order, the model names the outcome at "
              f"{a['order_only']['accuracy']:.1%} accuracy -- the probe is sensitive.")
        print(f"  given masked text, it reaches {a['masked']['accuracy']:.1%} "
              f"(AUROC {a['masked']['auroc']:.3f}).")
        print(f"  compare the TF-IDF probe's masked AUROC in "
              f"experiments/paper3/leakage_probe_{args.split}.json")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
