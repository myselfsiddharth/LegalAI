"""Step 9: the research brief -- issues, both sides' arguments, certified authorities.

Grounded arm: the model sees the client's story, the issue memo, the top-8 reranked precedents (each
as P1..P8 with its case card and good-law status) and the top-10 predicted provisions, and may cite
ONLY those identifiers. Closed-book arm: the same story and memo, no lists; the model names cases
from its own knowledge.

Certification, mechanical and per item:
  precedent   grounded: must be one of the P-ids shown. closed-book: the name must resolve to a
              real corpus judgment decided before the query (party-name match against the registry)
              -- else FABRICATED / UNRESOLVED.
  fact        must occur in the client's story (normalised)
  provision   grounded: must be in the list shown

Then, against what the court actually did: the share of certified precedents the court really cited
(a proxy for relevance -- the court's list is not the only right answer, and the report says so).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, field_validator

from src import paths
from src.llm import client

P = paths.ROOT / "product"
OUT = P / "data" / "briefs.jsonl"
EXP = P / "experiments" / "brief.json"
MODEL = "qwen3-235b-a22b-instruct-2507"
TOP_P, TOP_S = 8, 10


class Issue(BaseModel):
    issue: str
    client_argument: str
    opponent_argument: str
    precedents: list[str]
    provisions: list[str]
    facts: list[str]
    assessment: str

    @field_validator("precedents", "provisions", "facts", mode="before")
    @classmethod
    def _listify(cls, v):
        # the model sometimes returns one string where a list is asked for; that is a format slip,
        # not a refusal, and rejecting the whole brief for it cost 53 of 186 briefs on the first run
        return [v] if isinstance(v, str) else v


class Brief(BaseModel):
    issues: list[Issue]
    gaps: list[str]

    @field_validator("gaps", mode="before")
    @classmethod
    def _listify(cls, v):
        return [v] if isinstance(v, str) else v


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", (s or "").lower())).strip()


STOP = {"the", "of", "and", "vs", "v", "state", "union", "india", "ors", "anr", "others", "another",
        "ltd", "pvt", "co", "smt", "shri", "sri", "dr", "m", "s", "etc", "the", "on", "in", "by", "through"}


GENERIC = {"tamil", "nadu", "west", "bengal", "uttar", "pradesh", "madhya", "andhra", "himachal",
           "haryana", "punjab", "karnataka", "kerala", "maharashtra", "gujarat", "rajasthan", "bihar",
           "orissa", "odisha", "assam", "delhi", "bombay", "madras", "calcutta", "mysore", "goa",
           "municipal", "corporation", "development", "authority", "commissioner", "collector",
           "government", "secretary", "board", "trust", "society", "company", "bank", "singh",
           "kumar", "lal", "ram", "das", "devi", "prasad", "nath", "chand", "rao", "reddy"}


def sides(name: str) -> list[set[str]]:
    """Distinctive tokens of each side of a case name ('A v. B' / 'A vs B')."""
    name = re.sub(r"\s+on \d{1,2} \w+ \d{4}.*$", "", name)
    name = name.split(",")[0]                                     # "A v. B, AIR 1978 SC 597"
    name = re.sub(r"\b(?:AIR|SCC|SCR|SCALE|ILR|JT|SCJ)\b.*$|[\(\[]?\d{4}[\)\]]?.*$", "", name)
    name = re.sub(r"\((?:dead|d)\)[^)]*$|\(dead\)|by lrs?\.?", " ", name, flags=re.I)
    parts = re.split(r"\s+(?:v\.?|vs\.?|versus)\s+", name, maxsplit=1, flags=re.I)
    return [{w for w in norm(p).split() if len(w) > 2 and w not in STOP and w not in GENERIC}
            for p in parts]


def resolves(name: str, title: str) -> bool:
    """Both sides must match, in either order. A side made only of generic words (State of X,
    Union of India) cannot carry a match on its own -- the failure the first matcher showed."""
    a, b = sides(name), sides(title)
    if len(a) < 2 or len(b) < 2 or not (a[0] or a[1]):
        return False

    def ok(x, y):
        return (not x or len(x & y) / len(x) >= 0.5)
    straight = ok(a[0], b[0]) and ok(a[1], b[1])
    crossed = ok(a[0], b[1]) and ok(a[1], b[0])
    # at least one side must carry a distinctive token that actually matched
    hit = (a[0] & (b[0] | b[1])) or (a[1] & (b[0] | b[1]))
    return bool(hit) and (straight or crossed)


def party_tokens(title: str) -> set[str]:
    t = re.sub(r"\s+on \d{1,2} \w+ \d{4}.*$", "", title)
    return {w for w in norm(t).split() if len(w) > 2 and w not in STOP}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["grounded", "closed"])
    args = ap.parse_args()

    reg = {r["doc_id"]: r for r in map(json.loads, open(paths.CASE_REGISTRY))}
    nar = {r["case_id"]: r["narrative"] for r in map(json.loads, open(P / "data" / "client_narratives.jsonl")) if r["status"] == "ok"}
    memo = {r["case_id"]: r["memo"] for r in map(json.loads, open(P / "data" / "issue_memos.jsonl")) if r["status"] == "ok"}
    cards = {r["doc_id"]: r["card"] for r in map(json.loads, open(P / "data" / "case_cards.jsonl")) if r["status"] == "ok"}
    d = json.loads((P / "data" / "candidates_top50.json").read_text())
    scores = json.loads((P / "data" / "rerank_scores.json").read_text())
    targets = {c: set(v) for c, v in d["targets"].items()}
    statutes = json.loads((P / "data" / "statute_predictions.json").read_text())
    tr = [json.loads(l) for l in open(P / "data" / "treatments.jsonl")]
    by_cited: dict[str, list] = {}
    for r in tr:
        if r.get("treatment"):
            by_cited.setdefault(r["cited"], []).append((r["citing_year"], r["treatment"]))

    def status(doc, yq):
        c = Counter(t for y, t in by_cited.get(doc, []) if y and y < yq)
        if c["OVERRULED"]:
            return "OVERRULED"
        if c["DOUBTED"]:
            return "DOUBTED"
        return f"GOOD (followed {c['FOLLOWED']}x)" if sum(c.values()) else "no later treatment recorded"

    cases = sorted(c for c in d["candidates"] if c in nar and c in memo and c in statutes)
    tokens = {doc: party_tokens(r["title"]) for doc, r in reg.items() if r.get("title")}

    def build(c):
        yq = reg[c]["year"]
        lst = d["candidates"][c]["memo"]
        sc = scores.get(c, {})
        top = sorted(lst, key=lambda x: (-sc.get(x, 1), lst.index(x)))[:TOP_P]
        m = memo[c]
        head = (f"## CLIENT'S STORY\n\n{nar[c]}\n\n## ISSUE MEMO\n\nIssues: {'; '.join(m['issues'])}\n"
                f"Claims: {'; '.join(m['claims'])}")
        prec = "\n\n".join(
            f"P{i}. {reg[x]['title']}\n   Treatment by later courts: {status(x, yq)}\n"
            f"   Decided: {'; '.join(cards[x]['issues_decided'])}\n   Holdings: {'; '.join(cards[x]['holdings'])}"
            for i, x in enumerate(top, 1) if x in cards)
        return yq, top, head, prec, statutes[c][:TOP_S]

    def run(c, arm):
        yq, top, head, prec, stat = build(c)
        if arm == "grounded":
            body = head + f"\n\n## PRECEDENTS\n\n{prec}\n\n## PROVISIONS\n\n" + ", ".join(stat)
            pid = "brief.v1"
        else:
            body, pid = head, "brief_closed.v1"
        res = client.complete([{"role": "system", "content": client.load_prompt(pid)},
                               {"role": "user", "content": body}],
                              model=MODEL, json_schema=Brief, prompt_id=pid, max_tokens=2500)
        if not res.ok or res.parsed is None:
            return {"case_id": c, "arm": arm, "status": "no_answer" if not res.ok else "unparsed"}
        b = res.parsed.model_dump()
        nn = norm(nar[c])
        cert = Counter()
        cited_ok = []
        for iss in b["issues"]:
            for p in iss["precedents"]:
                if arm == "grounded":
                    mt = re.fullmatch(r"\s*P(\d+)\s*", p)
                    if mt and 1 <= int(mt.group(1)) <= len(top):
                        cert["prec_certified"] += 1
                        cited_ok.append(top[int(mt.group(1)) - 1])
                    else:
                        cert["prec_refused"] += 1
                else:
                    pt = party_tokens(p)
                    hits = [doc for doc, tk in tokens.items()
                            if tk and reg[doc]["year"] < yq and len(pt & tk) >= 1
                            and resolves(p, reg[doc]["title"])]
                    if len(hits) == 1:
                        cert["prec_certified"] += 1
                        cited_ok.append(hits[0])
                    elif hits:
                        cert["prec_ambiguous"] += 1
                    else:
                        cert["prec_unresolved"] += 1
            for f in iss["facts"]:
                cert["fact_ok" if len(norm(f)) > 15 and norm(f) in nn else "fact_refused"] += 1
            if arm == "grounded":
                for s in iss["provisions"]:
                    cert["prov_ok" if s.strip() in stat else "prov_refused"] += 1
            cert["both_sides"] += bool(iss["client_argument"].strip() and iss["opponent_argument"].strip())
            cert["issues"] += 1
        uniq = set(cited_ok)
        return {"case_id": c, "arm": arm, "status": "ok", "brief": b, "cert": dict(cert),
                "certified_docs": sorted(uniq),
                "court_cited_hits": len(uniq & targets[c]), "n_certified_unique": len(uniq)}

    jobs = [(c, a) for c in cases for a in args.arms]
    with ThreadPoolExecutor(max_workers=24) as pool:
        rows = list(pool.map(lambda j: run(*j), jobs))
    OUT.write_text("".join(json.dumps(r) + "\n" for r in rows))

    summary = {}
    for arm in args.arms:
        rs = [r for r in rows if r["arm"] == arm and r["status"] == "ok"]
        c = Counter()
        for r in rs:
            c.update(r["cert"])
        hits = sum(r["court_cited_hits"] for r in rs)
        uniq = sum(r["n_certified_unique"] for r in rs)
        proposed = c["prec_certified"] + c["prec_refused"] + c["prec_unresolved"] + c["prec_ambiguous"]
        summary[arm] = {
            "briefs": len(rs), "failed": sum(1 for r in rows if r["arm"] == arm and r["status"] != "ok"),
            "issues": c["issues"], "both_sides": c["both_sides"],
            "precedents_proposed": proposed, "certified": c["prec_certified"],
            "refused_or_unresolved": c["prec_refused"] + c["prec_unresolved"],
            "ambiguous": c["prec_ambiguous"],
            "certified_rate": round(c["prec_certified"] / max(1, proposed), 4),
            "facts_ok": c["fact_ok"], "facts_refused": c["fact_refused"],
            "provisions_ok": c["prov_ok"], "provisions_refused": c["prov_refused"],
            "unique_certified_precedents": uniq, "of_which_court_cited": hits,
            "court_cited_share": round(hits / max(1, uniq), 4)}
    EXP.write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
