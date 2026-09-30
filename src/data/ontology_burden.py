"""Fill element burden metadata in `ontology_v1.yaml` (§8.3, and the `E` feature group of §10.1).

The refined ontology says every element "must carry burden_on, burden_standard,
burden_shifts_when" and then supplies those values for **none** of its 66 elements; 6 of 15 claims
carry a claim-level default in a notes line. So the metadata has to be produced, and how it was
produced matters more than that it exists -- §8.3's element scores and any burden-shift rule read
these values directly, and a confidently wrong allocation is worse than a blank one because
nothing downstream would question it.

Every value therefore carries `burden_provenance`:

  `ontology_notes` -- the source document stated it for this claim; propagated to its elements.
  `llm_draft`      -- drafted here, **not reviewed**. Use it, report it as drafted, and do not
                      describe any result resting on it as validated.
  `human`          -- a reviewer confirmed or corrected it. Set by hand, never by this script.

Writes a §6.3 CHANGELOG entry, because this changes the ontology.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import date

import yaml
from pydantic import BaseModel

from src import paths
from src.llm import client

PROMPT_ID = "element_burden.v1"
ONT_PATH = paths.ONTOLOGY / "ontology_v1.yaml"
CHANGELOG = paths.ONTOLOGY / "CHANGELOG.md"

VALID_ON = {"claimant", "respondent", "authority"}

# --- degeneracy gate ---------------------------------------------------------------------
# A first run with llama4-scout-17b produced an annotation that LOOKED complete and carried
# almost no information: 92% `claimant`, only two confidence values (0.8 and 0.7), nothing
# flagged below 0.6 despite the prompt asking for it, and `clear_proof_required` used ZERO
# times -- even for adverse possession, which the prompt names explicitly and which is the
# textbook instance of a heightened standard in Indian law. `burden_shifts_when` was null for
# all four adverse-possession elements.
#
# That is the dangerous shape of failure: nothing downstream would question it, and §8.3's
# element scores and §10.1's `E` features would quietly rest on a constant. So the write is
# GATED. A degenerate annotation is refused, not saved with a caveat.
MAX_SINGLE_VALUE_SHARE = 0.85      # no one burden_on / standard may dominate this hard
MIN_DISTINCT_STANDARDS = 3
# Elements whose heightened standard is settled doctrine; if the annotator never reaches
# `clear_proof_required` for ANY of these, it is not discriminating.
EXPECT_CLEAR_PROOF_CLAIMS = {"claim_08", "claim_11", "claim_14"}   # adverse possession, fraud/
                                                                  # cancellation, benami


def degeneracy_report(ont: dict) -> tuple[bool, list[str]]:
    els = [(c["claim_id"], e) for c in ont["claims"] for e in c["elements"]]
    filled = [(cid, e) for cid, e in els if e.get("burden_on")]
    problems = []
    if not filled:
        return False, ["nothing was filled"]
    n = len(filled)

    for field in ("burden_on", "burden_standard"):
        counts = Counter(e.get(field) for _, e in filled)
        top, k = counts.most_common(1)[0]
        if k / n > MAX_SINGLE_VALUE_SHARE:
            problems.append(f"{field} is {100*k/n:.0f}% {top!r} "
                            f"(limit {100*MAX_SINGLE_VALUE_SHARE:.0f}%) -- not discriminating")
    if len({e.get("burden_standard") for _, e in filled}) < MIN_DISTINCT_STANDARDS:
        problems.append(f"fewer than {MIN_DISTINCT_STANDARDS} distinct burden_standard values")

    got_clear = {cid for cid, e in filled
                 if e.get("burden_standard") == "clear_proof_required"}
    if not (got_clear & EXPECT_CLEAR_PROOF_CLAIMS):
        problems.append("`clear_proof_required` never reached for adverse possession, "
                        "cancellation-for-fraud or benami -- the settled heightened-standard "
                        "claims the prompt names")

    confs = {e.get("burden_confidence") for _, e in filled}
    if len(confs) < 3:
        problems.append(f"only {len(confs)} distinct confidence values {sorted(confs)} -- the "
                        f"annotator is not expressing uncertainty")

    shifts = {str(e.get("burden_shifts_when")) for _, e in filled}
    if len(shifts) < 0.25 * n:
        problems.append(f"only {len(shifts)} distinct burden_shifts_when over {n} elements")
    return (not problems), problems
VALID_STD = {"prima_facie", "balance_of_probabilities", "statutory_presumption",
             "clear_proof_required", "unknown"}


class BurdenJSON(BaseModel):
    burden_on: str
    burden_standard: str = "unknown"
    burden_shifts_when: str | None = None
    reasoning: str = ""
    confidence: float = 0.5


def notes_default(notes: str | None) -> dict:
    """Parse the claim-level burden line the source document sometimes supplies, e.g.
    'Default burden_on: claimant. Burden shifts when joint nucleus ... is prima facie shown.'"""
    if not notes or "burden" not in notes.lower():
        return {}
    out = {}
    m = re.search(r"burden_on\s*:?\s*([^.;]+)", notes, re.I)
    if m:
        who = m.group(1).strip().lower()
        # the document names the real-world role; map it onto the schema's three values
        if any(w in who for w in ("claimant", "purchaser", "propounder", "heir", "mortgagor",
                                  "party alleging", "plaintiff")):
            out["burden_on"] = "claimant"
        elif any(w in who for w in ("respondent", "defendant", "mortgagee")):
            out["burden_on"] = "respondent"
        elif "authority" in who or "state" in who:
            out["burden_on"] = "authority"
        out["burden_on_raw"] = m.group(1).strip()
    m = re.search(r"shifts?\s+when\s+([^.]+)", notes, re.I)
    if m:
        out["burden_shifts_when"] = m.group(1).strip()
    return out


def draft_one(claim: dict, element: dict, model: str) -> dict | None:
    nd = notes_default(claim.get("notes"))
    guidance = claim.get("notes") or "(the ontology states no burden guidance for this claim)"
    msgs = [{"role": "system", "content": client.load_prompt(PROMPT_ID)},
            {"role": "user", "content":
             f"CLAIM: {claim['name']}\nISSUE: {claim['issue']}\n"
             f"ONTOLOGY BURDEN GUIDANCE FOR THIS CLAIM: {guidance}\n\n"
             f"ELEMENT: {element['name']}\n"
             f"DEFINITION: {element.get('definition') or '(none given)'}"}]
    res = client.complete(msgs, model=model, json_schema=BurdenJSON, prompt_id=PROMPT_ID,
                          max_tokens=300)
    if not res.ok or res.parsed is None:
        return None
    p = res.parsed
    on = p.burden_on.strip().lower()
    std = p.burden_standard.strip().lower()
    return {
        "burden_on": on if on in VALID_ON else nd.get("burden_on", "claimant"),
        "burden_standard": std if std in VALID_STD else "unknown",
        "burden_shifts_when": (p.burden_shifts_when or nd.get("burden_shifts_when") or None),
        "burden_reasoning": p.reasoning.strip()[:300],
        "burden_confidence": round(float(p.confidence), 3),
        "burden_provenance": "llm_draft",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=client.DEFAULT_MODEL)
    ap.add_argument("--force", action="store_true",
                    help="write even if the degeneracy gate fails")
    ap.add_argument("--overwrite-human", action="store_true",
                    help="also redraft elements a human has already confirmed (default: never)")
    args = ap.parse_args()

    ont = yaml.safe_load(ONT_PATH.read_text())
    stats = Counter()
    changed = []

    for claim in ont["claims"]:
        nd = notes_default(claim.get("notes"))
        if nd:
            stats["claims_with_notes_guidance"] += 1
        for el in claim["elements"]:
            if el.get("burden_provenance") == "human" and not args.overwrite_human:
                stats["kept_human"] += 1
                continue
            got = draft_one(claim, el, args.model)
            if got is None:
                stats["dropped"] += 1          # no answer: leave blank rather than invent
                continue
            # the ontology's own statement outranks the draft where they disagree
            if nd.get("burden_on") and nd["burden_on"] != got["burden_on"]:
                got["burden_on"] = nd["burden_on"]
                got["burden_provenance"] = "ontology_notes"
                stats["overridden_by_notes"] += 1
            elif nd.get("burden_on"):
                got["burden_provenance"] = "ontology_notes"
            el.update(got)
            stats[got["burden_provenance"]] += 1
            stats["filled"] += 1
            changed.append((claim["claim_id"], el["name"], got))

    ok, problems = degeneracy_report(ont)
    if not ok and not args.force:
        print(f"\nREFUSED to write {ONT_PATH.name}: the annotation is degenerate.\n")
        for p_ in problems:
            print(f"  - {p_}")
        print(f"\nA uniform annotation would be read downstream as legal metadata and nothing "
              f"would question it.\nTry a different --model, or --force to write it anyway "
              f"(it stays tagged llm_draft either way).")
        raise SystemExit(2)
    if not ok:
        print("\nWARNING writing a DEGENERATE annotation because --force was given:")
        for p_ in problems:
            print(f"  - {p_}")

    ONT_PATH.write_text(yaml.safe_dump(ont, sort_keys=False, allow_unicode=True, width=100))

    entry = [f"\n## {date.today().isoformat()} — element burden metadata filled\n",
             f"`src/data/ontology_burden.py`, prompt `{PROMPT_ID}`, model `{args.model}`.\n",
             f"The source document requires every element to carry `burden_on`, "
             f"`burden_standard` and `burden_shifts_when`, and supplies them for none of the "
             f"{sum(len(c['elements']) for c in ont['claims'])} elements; "
             f"{stats['claims_with_notes_guidance']} of 15 claims carry a claim-level default "
             f"in a notes line.\n",
             f"- filled: **{stats['filled']}** elements\n",
             f"- from the ontology's own notes: {stats['ontology_notes']}\n",
             f"- LLM-drafted and **not yet reviewed**: {stats['llm_draft']}\n",
             f"- claim-level notes overrode the draft's `burden_on`: "
             f"{stats['overridden_by_notes']}\n",
             f"- no answer, left blank: {stats['dropped']}\n",
             f"\nEvery value carries `burden_provenance`. Results that depend on an "
             f"`llm_draft` value must be reported as resting on drafted metadata, not on the "
             f"ontology. Reviewing an element means setting its provenance to `human`.\n"]
    with CHANGELOG.open("a") as f:
        if not CHANGELOG.exists() or CHANGELOG.stat().st_size == 0:
            f.write("# Ontology CHANGELOG (§6.3)\n\nEvery ontology or vocabulary change is "
                    "recorded here with its justification and the number of items affected.\n")
        f.writelines(entry)

    print(f"burden metadata -> {ONT_PATH}")
    for k in ("filled", "ontology_notes", "llm_draft", "overridden_by_notes", "kept_human",
              "dropped"):
        if stats[k]:
            print(f"  {k:22s} {stats[k]}")
    print(f"\n  {'claim':10s} {'element':40s} {'on':10s} {'standard':26s} conf")
    for cid, name, g in changed:
        print(f"  {cid:10s} {name[:40]:40s} {g['burden_on']:10s} "
              f"{g['burden_standard']:26s} {g['burden_confidence']:.2f}")
    low = [c for _, _, c in changed if c["burden_confidence"] < 0.6]
    print(f"\n  flagged low-confidence (contested or relief-dependent): {len(low)} "
          f"-- these are the ones to review first")
    ok2, probs2 = degeneracy_report(ont)
    print(f"  degeneracy gate: {'PASS' if ok2 else 'FAIL'}")
    for p_ in probs2:
        print(f"    - {p_}")


if __name__ == "__main__":
    main()
