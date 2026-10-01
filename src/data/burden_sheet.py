"""BLOCKER B2: a hand-annotation sheet for the 66 element burdens.

Two models have now failed `ontology_burden.py`'s degeneracy gate — `llama4-scout-17b` at 92%
`claimant` with `clear_proof_required` never used, and `glm-5-3` at 87%. The gate exists because a
constant masquerading as legal metadata would be read downstream as fact and nothing would question
it, so the honest remaining route is annotation by someone who can cite the doctrine.

This emits a CSV ready for that, pre-filled with what the ontology itself states and nothing more:

  * `ontology_note_default` — the claim-level default the source document supplies, for the 6 of 15
    claims that have one. Parsed, not guessed.
  * everything else blank, with the allowed values listed in the header comment.

Deliberately NOT pre-filled with a model's draft. A reviewer shown a plausible wrong answer tends to
accept it, and the whole reason this blocker is open is that plausible-but-constant answers are the
failure mode here.

The sheet is ordered so the elements that matter most come first: those belonging to claims that
clear §7's 30-case floor, since an element of a claim with 3 cases cannot affect any result.
"""
from __future__ import annotations

import csv
import json
from collections import Counter

import yaml

from src import paths
from src.data.ontology_burden import notes_default

OUT = paths.GOLD / "element_burden_sheet.csv"

HEADER_NOTE = [
    ["# BLOCKER B2 -- element burden annotation. Fill `burden_on`, `burden_standard`,"],
    ["# `burden_shifts_when`, `authority` (the provision or case you rely on), and `annotator`."],
    ["#"],
    ["# burden_on:        claimant | respondent | authority"],
    ["# burden_standard:  prima_facie | balance_of_probabilities | statutory_presumption |"],
    ["#                   clear_proof_required | unknown"],
    ["# burden_shifts_when: the TRIGGERING fact or event, not the consequence."],
    ["#                     e.g. 'registered deed produced', not 'defendant must then prove fraud'"],
    ["#"],
    ["# Leave a row blank rather than guessing. A blank is a known gap; a wrong value becomes an"],
    ["# `E` feature in the outcome model and nothing downstream questions it."],
    ["#"],
    ["# `ontology_note_default` is what the source ontology states for the whole claim, where it"],
    ["# states anything. It is the document's own guidance, not a model's draft."],
    [""],
]


def main() -> None:
    ont = yaml.safe_load((paths.ONTOLOGY / "ontology_v1.yaml").read_text())

    # claim -> number of cases, so the sheet can be ordered by what actually affects results
    fam = paths.INTERIM / "claim_families.json"
    counts: Counter = Counter()
    if fam.exists():
        members = json.loads(fam.read_text())["membership"]
        for m in members.values():
            for f in m.get("families", []):
                counts[f] += 1

    rows = []
    for c in ont["claims"]:
        nd = notes_default(c.get("notes"))
        # match the claim to an assigned family name loosely, for ordering only
        n_cases = max((v for k, v in counts.items()
                       if k.lower() in c["name"].lower() or c["slug"].startswith(k.lower())),
                      default=0)
        for e in c["elements"]:
            rows.append({
                "claim_id": c["claim_id"],
                "claim": c["name"],
                "claim_cases_in_corpus": n_cases,
                "element_id": e["element_id"],
                "element": e["name"],
                "element_definition": e.get("definition") or "",
                "claim_issue": c["issue"],
                "ontology_note_default": nd.get("burden_on_raw", ""),
                "ontology_note_shifts_when": nd.get("burden_shifts_when", ""),
                "burden_on": "",
                "burden_standard": "",
                "burden_shifts_when": "",
                "authority": "",
                "annotator": "",
                "confidence_0_to_1": "",
            })

    rows.sort(key=lambda r: (-r["claim_cases_in_corpus"], r["claim_id"], r["element_id"]))
    fields = list(rows[0])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as fh:
        w = csv.writer(fh)
        for line in HEADER_NOTE:
            w.writerow(line)
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    with_note = sum(1 for r in rows if r["ontology_note_default"])
    print(f"burden sheet: {len(rows)} elements -> {OUT}")
    print(f"  with a claim-level default from the ontology: {with_note}")
    print(f"  blank and needing a judgement: {len(rows) - with_note}")
    print(f"  ordered by claim case count, so the elements that can affect a result come first")
    print(f"\n  first 6 rows:")
    for r in rows[:6]:
        print(f"    [{r['claim_cases_in_corpus']:4d} cases] {r['claim'][:34]:34s} "
              f"{r['element'][:30]}")


if __name__ == "__main__":
    main()
