"""Stage 3 (§7.1): claims and defences, with the family each one belongs to.

A claim is what makes the rest of the pipeline sortable. §8 mines fact patterns **per claim
family**, because the facts that decide an adverse-possession case and those that decide a
specific-performance case have nothing to do with one another and mining them together yields
patterns that describe the corpus rather than the law.

The model proposes a family here and §7.2 checks that proposal by clustering the claim texts
independently: if unsupervised clusters line up with the ontology's 15 modules, the taxonomy is
doing real work; where they do not, the mismatch is the finding (§7.3 asks for exactly that, as
NMI and purity plus the clusters with no module and modules with no cluster).

Same grounding rule as everywhere else: a claim whose quote cannot be located in the excerpt is
discarded, because the quote is the only evidence that the claim was pleaded rather than inferred.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field

from pydantic import BaseModel

from src import grounding, paths
from src.llm import client

PROMPT_ID = "extract_claims.v1"
OUT_PATH = paths.INTERIM / "claims.jsonl"

FAMILIES = ["Title", "Possession", "LeaseTenancy", "Redevelopment", "ConstitutionalDeprivation",
            "UltraVires", "CooperativeSociety", "AdversePossession", "Partition",
            "SpecificPerformance", "Cancellation", "MortgageRedemption", "Succession",
            "Benami", "LandAcquisition"]
DEFENCE_FAMILIES = ["Title/Tenure", "Public Interest/Planning Policy",
                    "Statutory/Regulatory Subservience", "Consent/Compliance",
                    "Procedural/Forum", "Equitable"]

# Claims and prayers are pleaded early; the prayer is recited near the start of the narrative.
# Reading the whole masked text would triple cost for little gain.
HEAD_CHARS = 9_000


class ClaimJSON(BaseModel):
    text: str
    quote: str = ""
    raised_by: str = "unknown"
    relief_sought: str | None = None
    statutory_hook: str | None = None
    family: str = ""


class DefenceJSON(BaseModel):
    text: str
    quote: str = ""
    raised_by: str = "unknown"
    family: str = ""


class ClaimsJSON(BaseModel):
    claims: list[ClaimJSON] = []
    defences: list[DefenceJSON] = []


@dataclass
class Claim:
    claim_id: str
    case_id: str
    kind: str                       # 'claim' | 'defence'
    text: str
    quote: str
    source_span: list[int]
    raised_by: str
    family: str
    family_is_new: bool = False
    relief_sought: str | None = None
    statutory_hook: str | None = None


def extract_case(rec: dict, model: str) -> tuple[list[Claim], Counter]:
    head = rec["masked_text"][:HEAD_CHARS]
    stats = Counter()
    res = client.complete(
        [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
         {"role": "user", "content": head}],
        model=model, json_schema=ClaimsJSON, prompt_id=PROMPT_ID, max_tokens=2000)
    if not res.ok:
        stats["dropped"] += 1
        return [], stats
    if res.parsed is None:
        stats["unparseable"] += 1
        return [], stats
    stats["ok"] += 1

    out: list[Claim] = []
    for kind, items, valid in (("claim", res.parsed.claims, FAMILIES),
                               ("defence", res.parsed.defences, DEFENCE_FAMILIES)):
        stats[f"{kind}_proposed"] += len(items)
        for it in items:
            span = grounding.locate(it.quote, head)
            if span is None:
                stats[f"{kind}_discarded_quote"] += 1
                continue
            fam = (it.family or "").strip()
            is_new = fam.upper().startswith("NEW") or fam not in valid
            if is_new:
                stats[f"{kind}_family_new"] += 1
            out.append(Claim(
                claim_id=f"{rec['doc_id']}_{kind[0]}{sum(1 for c in out if c.kind == kind)+1:02d}",
                case_id=rec["doc_id"], kind=kind, text=it.text.strip(), quote=it.quote,
                source_span=[span[0], span[1]],
                raised_by=it.raised_by if it.raised_by in ("plaintiff", "defendant") else "unknown",
                family=fam, family_is_new=is_new,
                relief_sought=getattr(it, "relief_sought", None),
                statutory_hook=getattr(it, "statutory_hook", None)))
            stats[f"{kind}_kept"] += 1
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    from src.data.label_merge import load_final
    labels = load_final()
    screen = {json.loads(l)["doc_id"]: json.loads(l)
              for l in open(paths.INTERIM / "case_screen.jsonl")}
    eligible = {d for d, r in labels.items()
                if r["outcome"] in ("WIN", "LOSE")
                and screen.get(d, {}).get("is_property", False)}

    cases = [json.loads(l) for l in open(paths.INTERIM / "masked_text.jsonl")]
    cases = [c for c in cases if c["doc_id"] in eligible and c["n_chars_masked"] >= 1000]
    cases.sort(key=lambda r: r["doc_id"])

    done = set()
    if OUT_PATH.exists():
        with OUT_PATH.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["case_id"])
                except json.JSONDecodeError:
                    pass
    todo = [c for c in cases if c["doc_id"] not in done]
    if args.n:
        todo = todo[:args.n]
    print(f"{len(cases)} eligible, {len(done)} done, {len(todo)} to do", flush=True)

    total = Counter()
    n_claims_per_case = []
    with OUT_PATH.open("a") as out, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i, (claims, st) in enumerate(
                pool.map(lambda c: extract_case(c, args.model), todo), 1):
            for c in claims:
                out.write(json.dumps(asdict(c)) + "\n")
            total.update(st)
            n_claims_per_case.append(sum(1 for c in claims if c.kind == "claim"))
            if i % 50 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}  claims {total['claim_kept']} "
                      f"defences {total['defence_kept']}", flush=True)
    report(total, n_claims_per_case, len(todo))


def report(total: Counter, per_case: list[int], n_cases: int) -> None:
    import statistics
    print(f"\n=== claim extraction: {n_cases} cases ===")
    print(f"  calls ok {total['ok']}, unparseable {total['unparseable']}, "
          f"DROPPED {total['dropped']}")
    for kind in ("claim", "defence"):
        prop = total[f"{kind}_proposed"] or 1
        print(f"  {kind}s: proposed {total[f'{kind}_proposed']} -> kept {total[f'{kind}_kept']} "
              f"({100*total[f'{kind}_kept']/prop:.1f}%); "
              f"discarded for an unlocatable quote {total[f'{kind}_discarded_quote']}; "
              f"family outside the list {total[f'{kind}_family_new']}")
    if per_case:
        print(f"  claims per case: median {statistics.median(per_case):.0f}, "
              f"cases with none {sum(1 for x in per_case if x == 0)}")

    if OUT_PATH.exists():
        rows = [json.loads(l) for l in open(OUT_PATH)]
        fam = Counter(r["family"] for r in rows if r["kind"] == "claim")
        cases_per_fam = Counter()
        for f in {(r["case_id"], r["family"]) for r in rows if r["kind"] == "claim"}:
            cases_per_fam[f[1]] += 1
        print(f"\n  claim families over {len(rows)} rows "
              f"({len({r['case_id'] for r in rows})} cases):")
        for k, v in fam.most_common():
            low = "  <- under §7's 30-case floor" if cases_per_fam[k] < 30 else ""
            print(f"    {k:28s} {v:4d} claims  {cases_per_fam[k]:4d} cases{low}")
        dfam = Counter(r["family"] for r in rows if r["kind"] == "defence")
        print(f"\n  defence families:")
        for k, v in dfam.most_common(10):
            print(f"    {k:40s} {v:4d}")
    print(f"\n  usage: {client.usage()}")


if __name__ == "__main__":
    main()
