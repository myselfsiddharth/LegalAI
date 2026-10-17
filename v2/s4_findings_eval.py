"""Score s4_findings_rerun.py: what happens to S4's defeat verdicts without prior-court findings?

Compared on the same (case, claim) pairs: the original full run (`base`), `exclude`, and `tag`.

  survival      of base's NOT_SATISFIED verdicts, the share an arm still calls NOT_SATISFIED,
                split by whether base's quote came from a flagged prior-court finding. If the
                finding-backed ones collapse and the evidence-backed ones hold, the defeat layer was
                leaning on the findings, and the fix removes exactly that.
  violations    `tag` only: verdicts that quote a fact the prompt said not to quote.
  gold          agreement with the AI reviewer pass on the 40 gold cases. That annotation was made
                with the findings VISIBLE, so it slightly favours base -- read it as a floor.
"""
from __future__ import annotations

import collections
import json
import re

from src.extract.induced_vocab import load_fact_texts
from v2 import paths

BASE = paths.INTERIM / "element_verdicts_element_satisfy_v2_facts.jsonl"
ARMS = ("exclude", "tag")
ANN = paths.ROOT.parent / "Data" / "gold" / "element_gold_annotations_ai.jsonl"


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip().lower()


def load(path) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[(r["case_id"], r["claim_id"])] = r
    return out


def main() -> None:
    flagged = set(json.loads((paths.INTERIM / "prior_court_findings.json").read_text()))
    facts, _ = load_fact_texts()
    by_case = collections.defaultdict(list)
    for f in facts:
        by_case[f["case_id"]].append((norm(f["text"]), f["fact_id"] in flagged))

    def from_finding(case_id: str, quote: str) -> bool:
        q = norm(quote)
        return bool(q) and any(q in t for t, fl in by_case[case_id] if fl)

    runs = {a: load(paths.INTERIM / f"element_verdicts_findings_{a}.jsonl") for a in ARMS}
    pairs = set.intersection(*(set(r) for r in runs.values()))
    base_all = load(BASE)
    runs = {"base": {p: base_all[p] for p in pairs}, **{a: {p: r[p] for p in pairs}
                                                       for a, r in runs.items()}}
    print(f"=== S4 without prior-court findings: {len(pairs):,} (case, claim) pairs ===\n")

    summary = {}
    print(f"  {'arm':8s} {'SAT':>6s} {'NOT':>6s} {'UNCL':>6s} {'refused':>8s} {'dropped':>8s}"
          f" {'NOT from finding':>17s}")
    for a, rr in runs.items():
        c = collections.Counter()
        for (case, _), r in rr.items():
            for v in r["verdicts"]:
                c[v["verdict"]] += 1
                if v["verdict"] == "NOT_SATISFIED" and from_finding(case, v["quote"]):
                    c["not_from_finding"] += 1
                if v["verdict"] != "UNCLEAR" and from_finding(case, v["quote"]):
                    c["quotes_finding"] += 1
            c["refused"] += r.get("refused_quote_absent", 0) + r.get("refused_quote_short", 0)
            c["dropped"] += r.get("dropped_no_answer", 0) + r.get("unparsed", 0)
        summary[a] = dict(c)
        print(f"  {a:8s} {c['SATISFIED']:6,d} {c['NOT_SATISFIED']:6,d} {c['UNCLEAR']:6,d} "
              f"{c['refused']:8,d} {c['dropped']:8,d} {c['not_from_finding']:17,d}")

    # survival of base's defeat verdicts
    print("\n  survival of base NOT_SATISFIED verdicts:")
    surv = {}
    for a in ARMS:
        s = collections.Counter()
        for p, r in runs["base"].items():
            arm_v = {v["element_id"]: v["verdict"] for v in runs[a][p]["verdicts"]}
            for v in r["verdicts"]:
                if v["verdict"] != "NOT_SATISFIED":
                    continue
                k = "finding" if from_finding(p[0], v["quote"]) else "evidence"
                s[f"{k}_n"] += 1
                s[f"{k}_{arm_v.get(v['element_id'], 'MISSING')}"] += 1
        surv[a] = dict(s)
        for k in ("finding", "evidence"):
            n = s[f"{k}_n"]
            if n:
                print(f"    {a:8s} base quote from {k:9s} n={n:5,d}  still NOT {s[k+'_NOT_SATISFIED']/n:6.1%}"
                      f"  -> UNCLEAR {s[k+'_UNCLEAR']/n:6.1%}  -> SAT {s[k+'_SATISFIED']/n:6.1%}")

    if "tag" in runs:
        q = summary["tag"].get("quotes_finding", 0)
        dec = summary["tag"].get("SATISFIED", 0) + summary["tag"].get("NOT_SATISFIED", 0)
        print(f"\n  tag arm violations: {q:,} of {dec:,} decisive verdicts quote a tagged finding "
              f"({q/max(1,dec):.1%})")

    # gold agreement (AI reviewer, made with findings visible)
    gold = {}
    if ANN.exists():
        ann = {}
        for line in open(ANN):
            r = json.loads(line)
            for j in r["judgements"]:
                ann[(r["case_id"], j["element_id"])] = j["human_verdict"]
        print("\n  agreement with the AI reviewer on the 40 gold cases (findings were visible to it):")
        for a, rr in runs.items():
            M = collections.Counter()
            for (case, _), r in rr.items():
                for v in r["verdicts"]:
                    h = ann.get((case, v["element_id"]))
                    if h is not None:
                        M[(h, v["verdict"])] += 1
            n = sum(M.values())
            agree = sum(M[(x, x)] for x in ("SATISFIED", "NOT_SATISFIED", "UNCLEAR"))
            mp = sum(M[(h, "NOT_SATISFIED")] for h in ("SATISFIED", "NOT_SATISFIED", "UNCLEAR"))
            sp = sum(M[(h, "SATISFIED")] for h in ("SATISFIED", "NOT_SATISFIED", "UNCLEAR"))
            gold[a] = {"n": n, "agree": agree, "not_prec": [M[("NOT_SATISFIED", "NOT_SATISFIED")], mp],
                       "sat_prec": [M[("SATISFIED", "SATISFIED")], sp]}
            print(f"    {a:8s} {agree}/{n} = {agree/max(1,n):.1%}   "
                  f"NOT precision {M[('NOT_SATISFIED','NOT_SATISFIED')]}/{mp}   "
                  f"SAT precision {M[('SATISFIED','SATISFIED')]}/{sp}")

    (paths.EXPERIMENTS / "s4_findings_rerun.json").write_text(json.dumps({
        "n_pairs": len(pairs), "n_flagged_facts": len(flagged), "verdicts": summary,
        "survival": surv, "gold_vs_ai_reviewer": gold,
        "caveats": ["flag detector precision ~0.75, recall unmeasured",
                    "gold reference is an AI reviewer that saw the findings",
                    "pairs are selected for having a base NOT_SATISFIED, so base's NOT count is "
                    "inflated relative to the corpus by construction"]}, indent=1))
    print(f"\n-> {paths.EXPERIMENTS / 's4_findings_rerun.json'}")


if __name__ == "__main__":
    main()
