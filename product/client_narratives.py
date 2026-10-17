"""Simulated client input: each held-out case retold as a client's first meeting with a lawyer.

Every experiment so far fed the pipeline facts extracted from a FINISHED judgment. A lawyer's real
input is a client's story: no court history, no statute numbers, no case names, no conclusions.
That gap was the largest untested assumption in the project, so this module builds the input the
product will actually face and the evaluation in `eval_narratives.py` runs on it.

Cases: `forum_heldout` TEST cases (never seen in training) with >= 3 reachable cited precedents and a
statute target, sampled with a fixed seed.

Leakage gate, because a narrative that names a section or a ruling would quietly hand the retriever
its answer:
  - statute / provision references ("Section 4", "Act, 1894", "Article 300A", "Order 21")
  - court or proceeding words ("High Court", "decree", "appeal", "petition", "judgment", "held")
  - case-name patterns ("X v. Y", "X vs Y")
A narrative that trips the gate is regenerated once with the offending phrases named; if it still
fails it is EXCLUDED and counted, never silently kept.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src import paths
from src.llm import client

PROMPT_ID = "client_narrative.v1"
OUT = paths.ROOT / "product" / "data" / "client_narratives.jsonl"
MIN_TARGETS = 3
SEED = 573

GATE = {
    "provision": re.compile(r"\b(?:sections?|sec\.|ss?\.|articles?|art\.|order|rule|clause)\s*"
                            r"\d+[a-z]?\b|\bact,?\s+(?:no\.?\s*)?\d{4}\b|\b\d{4}\s+act\b", re.I),
    "court": re.compile(r"\b(?:high court|supreme court|trial court|district court|tribunal|"
                        r"munsi(?:f|ff)|decree|appeal|appellant|respondent|petition|petitioner|"
                        r"judgment|writ|hon'?ble|the court|learned|was held|have held|it held)\b",
                        re.I),
    "case_name": re.compile(r"\b[A-Z][a-z]+\s+(?:v\.|vs\.?|versus)\s+[A-Z]", re.M),
}


def gate(text: str) -> list[str]:
    hits = []
    for name, rx in GATE.items():
        for m in rx.finditer(text):
            hits.append(f"{name}:{m.group(0)}")
    return hits


def select_cases(n: int) -> list[str]:
    split = json.loads((paths.SPLITS / "forum_heldout.json").read_text())
    reg = {json.loads(l)["doc_id"]: json.loads(l) for l in open(paths.CASE_REGISTRY)}
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    years = {d: reg[d]["year"] for d in masked if masked[d] and reg.get(d, {}).get("year")}
    cited: dict[str, set] = {}
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        a, b = r["citing_doc_id"], r.get("cited_doc_id")
        if r.get("edge_type") == "case" and a in years and b in years and years[b] < years[a]:
            cited.setdefault(a, set()).add(b)
    stat = {json.loads(l)["doc_id"] for l in open(paths.INTERIM / "statute_targets.jsonl")
            if json.loads(l).get("labels_all")}
    cand = sorted(c for c in split["test"]
                  if len(cited.get(c, ())) >= MIN_TARGETS and c in stat and len(masked.get(c) or "") > 1500)
    rng = np.random.default_rng(SEED)
    if len(cand) > n:
        cand = [cand[i] for i in sorted(rng.choice(len(cand), n, replace=False))]
    return cand


def narrate(case_id: str, text: str, model: str) -> dict:
    sysmsg = client.load_prompt(PROMPT_ID)
    msgs = [{"role": "system", "content": sysmsg},
            {"role": "user", "content": text[:14000]}]
    res = client.complete(msgs, model=model, prompt_id=PROMPT_ID, max_tokens=900)
    if not res.ok:
        return {"case_id": case_id, "status": "no_answer"}
    nar, hits, attempt = res.text.strip(), gate(res.text), 1
    if hits:
        note = ("Your narrative mentions things the client could not know before going to court, "
                "or names a law or a case: " + "; ".join(sorted(set(h.split(':', 1)[1] for h in hits)))
                + ". Rewrite it without any of them, following every rule above.")
        res2 = client.complete(msgs + [{"role": "assistant", "content": nar},
                                       {"role": "user", "content": note}],
                               model=model, prompt_id=PROMPT_ID, max_tokens=900)
        attempt = 2
        if res2.ok:
            nar, hits = res2.text.strip(), gate(res2.text)
    return {"case_id": case_id, "narrative": nar, "words": len(nar.split()),
            "gate_hits": hits, "attempts": attempt,
            "status": "ok" if not hits else "excluded_leak"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    cases = select_cases(args.n)
    masked = {json.loads(l)["doc_id"]: json.loads(l)["masked_text"]
              for l in open(paths.INTERIM / "masked_text.jsonl")}
    print(f"{len(cases)} held-out cases selected (>= {MIN_TARGETS} reachable precedents, statute target)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda c: narrate(c, masked[c], args.model), cases))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    st = Counter(r["status"] for r in rows)
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"-> {OUT}")
    print(f"  {dict(st)}; retried {sum(1 for r in rows if r.get('attempts') == 2)}; "
          f"median words {sorted(r['words'] for r in ok)[len(ok)//2] if ok else 0}")


if __name__ == "__main__":
    main()
