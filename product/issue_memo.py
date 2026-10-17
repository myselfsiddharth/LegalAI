"""Issue spotting: turn a client's narrative into a lawyer's research memo before retrieval.

eval_narratives.py showed the precedent layer loses ~45% of its recall on a client's story (R@10
0.233 -> 0.131). The hypothesis: narratives speak in events, judgments in doctrine and statute
names, and a lawyer bridges the two by spotting issues before searching. This module produces that
memo; `eval_narratives.py --memo` retrieves with narrative + memo and measures whether it closes
the gap.

Memorisation guard: the prompt forbids case names, and every memo is scanned for case-name patterns
(`X v. Y`). A model that recognises a famous dispute and names its precedents would inflate the
evaluation without helping a real user, so offending memos are counted and their case-name spans
stripped before use.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src import paths
from src.llm import client

PROMPT_ID = "issue_memo.v1"
NAR = paths.ROOT / "product" / "data" / "client_narratives.jsonl"
OUT = paths.ROOT / "product" / "data" / "issue_memos.jsonl"
CASE_NAME = re.compile(r"\b[A-Z][\w.&' ]{1,60}?\s+(?:v\.|vs\.?|versus)\s+[A-Z][\w.&' ]{1,60}")


class Memo(BaseModel):
    issues: list[str]
    claims: list[str]
    provisions: list[str]
    doctrines: list[str]
    search_terms: str


def memo_text(m: dict) -> str:
    return " ".join([*m.get("issues", []), *m.get("claims", []), *m.get("provisions", []),
                     *m.get("doctrines", []), m.get("search_terms", "")])


def one(case_id: str, narrative: str, model: str) -> dict:
    res = client.complete([{"role": "system", "content": client.load_prompt(PROMPT_ID)},
                           {"role": "user", "content": narrative}],
                          model=model, json_schema=Memo, prompt_id=PROMPT_ID, max_tokens=900)
    if not res.ok:
        return {"case_id": case_id, "status": "no_answer"}
    if res.parsed is None:
        return {"case_id": case_id, "status": "unparsed"}
    m = res.parsed.model_dump()
    blob = json.dumps(m)
    names = CASE_NAME.findall(blob)
    if names:
        for k in ("issues", "claims", "provisions", "doctrines"):
            m[k] = [CASE_NAME.sub("", x) for x in m[k]]
        m["search_terms"] = CASE_NAME.sub("", m["search_terms"])
    return {"case_id": case_id, "status": "ok", "memo": m, "case_names_stripped": names}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-235b-a22b-instruct-2507")
    args = ap.parse_args()
    rows = [r for r in map(json.loads, open(NAR)) if r["status"] == "ok"]
    with ThreadPoolExecutor(max_workers=32) as pool:
        out = list(pool.map(lambda r: one(r["case_id"], r["narrative"], args.model), rows))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in out))
    from collections import Counter
    print(f"-> {OUT}  {dict(Counter(r['status'] for r in out))}; memos naming a case (stripped): "
          f"{sum(1 for r in out if r.get('case_names_stripped'))}")


if __name__ == "__main__":
    main()
