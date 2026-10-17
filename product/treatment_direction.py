"""Second check on the rare, high-stakes labels: is the CITED case really the one overruled?

treatment.v1 and v2 both inverted direction on about half of OVERRULED (spot checks 15/30, 13/24):
a passage like "X was overruled in Y" got Y marked overruled. A good-law flag that marks a landmark
overruling judgment as bad law is worse than no flag, so every OVERRULED and DOUBTED label is put to
a single yes/no question; anything not confirmed is downgraded to REFERRED and counted.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src import paths
from src.llm import client
from product.treatment import OUT, locate

PROMPT_ID = "treatment_direction.v1"


class A(BaseModel):
    answer: str


def main() -> None:
    rows = [json.loads(l) for l in open(OUT)]
    reg = {json.loads(l)["doc_id"]: json.loads(l)["title"] for l in open(paths.CASE_REGISTRY)}
    anchors = {}
    for line in open("Data/processed/citations_classified.jsonl"):
        r = json.loads(line)
        if r.get("edge_type") == "case":
            anchors[(r["citing_doc_id"], r.get("cited_doc_id"))] = r.get("cited_text") or ""
    texts = {}
    need = {r["citing"] for r in rows if r.get("treatment") in ("OVERRULED", "DOUBTED")}
    for fn, key in (("case_text.jsonl", "text"), ("corpus_wide.jsonl", "ildc_style_text")):
        for line in open(paths.INTERIM / fn):
            r = json.loads(line)
            if r["doc_id"] in need and r["doc_id"] not in texts:
                texts[r["doc_id"]] = r.get(key) or ""
    extra = paths.ROOT / "product" / "data" / "extra_text.jsonl"
    if extra.exists():
        for line in open(extra):
            r = json.loads(line)
            if r["doc_id"] in need:
                texts.setdefault(r["doc_id"], r["text"])
    todo = [i for i, r in enumerate(rows) if r.get("treatment") in ("OVERRULED", "DOUBTED")]

    def ask(i):
        r = rows[i]
        anchor = anchors.get((r["citing"], r["cited"]), "")
        target = re.sub(r"\s+on \d.*$", "", reg.get(r["cited"], anchor))
        passage = locate(texts.get(r["citing"], ""), anchor) or r.get("quote", "")
        res = client.complete([{"role": "system", "content": client.load_prompt(PROMPT_ID)},
                               {"role": "user", "content": f"TARGET: {target}\n(as cited: {anchor})\n\nPASSAGE: {passage}"}],
                              model="qwen3-235b-a22b-instruct-2507", json_schema=A,
                              prompt_id=PROMPT_ID, max_tokens=20)
        return i, (res.parsed.answer.strip().upper() if res.ok and res.parsed else None)

    with ThreadPoolExecutor(max_workers=16) as pool:
        ans = dict(pool.map(ask, todo))
    st = Counter()
    for i, a in ans.items():
        r = rows[i]
        r["direction_check"] = a
        if a != "YES":
            st[f"{r['treatment']}->REFERRED ({a})"] += 1
            r["treatment_before_check"], r["treatment"] = r["treatment"], "REFERRED"
        else:
            st[f"{r['treatment']} confirmed"] += 1
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(dict(st))
    print("treatments now: " + json.dumps(Counter(r["treatment"] for r in rows)))


if __name__ == "__main__":
    main()
