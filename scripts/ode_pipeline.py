"""Phase I -- Ontology-Directed Extraction: grounding, rule learning, and extraction.

This is the part of the pipeline that's been designed (the ontology, the gold examples)
but never actually run. It implements the two algorithms from the course's ODE method:

GROUND (the neural step): given a sentence and the ontology, the LLM outputs which
ontology classes are instantiated and which role-fillers attach to them -- NOT a free
summary, a constrained mapping onto the fixed class list. E.g. "held" grounds to
RulingAction with a Patient role filled by the holding's text.

LEARN (the symbolic step): for each gold (sentence, concept, filler) example, ground the
sentence, find which grounded fact's role-filler matches the gold filler, and record
that (class, role) pair as evidence for the concept. Pairs seen across multiple gold
examples become higher-confidence rules; pairs seen once are kept as low-confidence,
flagged for the residual-review loop rather than discarded.

EXTRACT: ground a new sentence, then apply the learned (class, role) -> concept rules to
emit typed objects.

The actual grounding/rule-matching logic lives in ode_lib.py, shared with
phase3_verify.py so both stages use exactly the same GROUND implementation.

Usage:
  python3 scripts/ode_pipeline.py ground --sentence "..."     # smoke test grounding alone
  python3 scripts/ode_pipeline.py learn                        # run LEARN over gold_examples.jsonl
  python3 scripts/ode_pipeline.py extract --input sentences.txt  # run EXTRACT on new sentences
"""

import argparse
import json
import os
import sys
from difflib import SequenceMatcher
from pathlib import Path

from ode_lib import (get_client, ground_sentence, load_ontology, extract_concepts,
                     content_guard, ROOT)

GOLD_PATH = ROOT / "scripts" / "ontology" / "gold_examples.jsonl"
RULES_PATH = ROOT / "scripts" / "ontology" / "learned_rules.json"


def best_matching_role(facts: list[dict], target_filler: str) -> tuple[dict, str, float] | None:
    """Find the (fact, role_name) whose role-filler text best matches the gold filler."""
    best = None
    best_score = 0.0
    target_norm = target_filler.lower()
    for fact in facts:
        for role_name, filler_text in fact.get("roles", {}).items():
            if not isinstance(filler_text, str):
                continue
            score = SequenceMatcher(None, target_norm, filler_text.lower()).ratio()
            if score > best_score:
                best, best_score = (fact, role_name, score), score
    return best


def cmd_ground(args, client, ontology):
    facts = ground_sentence(client, args.model, args.sentence, ontology)
    print(json.dumps(facts, indent=2))


MIN_LEARN_MATCH_SCORE = 0.45  # below this, treat as "grounding didn't really capture this
                              # concept" rather than counting a weak coincidental match as
                              # rule evidence -- found via audit: a 0.3-score match polluted
                              # the Fact rule with a spurious (TreatmentAction, Authority) pair.


def cmd_learn(args, client, ontology):
    gold_examples = [json.loads(line) for line in GOLD_PATH.read_text().splitlines() if line.strip()]
    evidence = {}       # (class, role, guard) -> {concept: count}
    evidence_any = {}   # (class, role)        -> {concept: count}, the unguarded fallback
    per_example_results = []

    for ex in gold_examples:
        facts = ground_sentence(client, args.model, ex["sentence"], ontology)
        match = best_matching_role(facts, ex["filler"])
        result = {"sentence": ex["sentence"][:80], "concept": ex["concept"], "gold_filler": ex["filler"][:60]}
        if match and match[2] >= MIN_LEARN_MATCH_SCORE:
            fact, role_name, score = match
            # The guard is read off the GROUNDED filler, not the gold one, so LEARN keys
            # rules on exactly the signal EXTRACT will see at inference time.
            guard = content_guard(fact["roles"][role_name])
            for store, key in ((evidence, (fact["class"], role_name, guard)),
                               (evidence_any, (fact["class"], role_name))):
                store.setdefault(key, {})
                store[key][ex["concept"]] = store[key].get(ex["concept"], 0) + 1
            result.update({"matched_class": fact["class"], "matched_role": role_name,
                           "guard": guard, "match_score": round(score, 2)})
        else:
            result["matched_class"] = None
            if match:
                result["weak_match_discarded"] = round(match[2], 2)
        per_example_results.append(result)
        status = "OK" if result.get("matched_class") else ("WEAK" if match else "NO MATCH")
        print(f"[{status}] {ex['concept']:15s} <- {result.get('matched_class', '?')}."
              f"{result.get('matched_role', '?')}[{result.get('guard', '-')}] "
              f"(score={result.get('match_score', result.get('weak_match_discarded', 0))}) "
              f"| {ex['sentence'][:52]}")

    # An unguarded "*" rule is a safety net for content shapes LEARN never saw, but emitting
    # one for EVERY pair defeats the point: TreatmentAction.Authority was seen only with
    # case-shaped fillers, so a blanket default is what lets statute spans keep arriving as
    # AuthorityCited (196 of 424 outputs last run). Emit the net only where the pair really
    # does appear with more than one shape, and leave single-shape pairs strict.
    guards_seen = {}
    for cls, role, guard in evidence:
        guards_seen.setdefault((cls, role), set()).add(guard)

    rules = []
    for store, guard_of in ((evidence, lambda k: k[2]), (evidence_any, lambda k: "*")):
        for key, concept_counts in store.items():
            if guard_of(key) == "*" and len(guards_seen.get(key, ())) < 2:
                continue
            best_concept = max(concept_counts, key=concept_counts.get)
            support = concept_counts[best_concept]
            total = sum(concept_counts.values())
            rules.append({
                "class": key[0],
                "role": key[1],
                "guard": guard_of(key),
                "concept": best_concept,
                "support": support,
                "confidence": round(support / total, 2),
            })
    rules.sort(key=lambda r: (r["guard"] == "*", -r["support"], r["class"], r["role"]))

    RULES_PATH.write_text(json.dumps(rules, indent=2))
    n_matched = sum(1 for r in per_example_results if r.get("matched_class"))
    print(f"\n{n_matched}/{len(gold_examples)} gold examples matched to a grounded fact")
    print(f"Learned {len(rules)} rules -> {RULES_PATH}")
    for r in rules:
        flag = "" if r["support"] >= 2 else "  (single example -- low confidence, needs more gold data)"
        cond = "" if r["guard"] == "*" else f" AND guard(X)={r['guard']}"
        print(f"  IF {r['class']}(A) AND {r['role']}(A, X){cond} THEN {r['concept']}(X) "
              f"[support={r['support']}, confidence={r['confidence']}]{flag}")


def cmd_extract(args, client, ontology):
    if not RULES_PATH.exists():
        sys.exit("No learned rules yet -- run `python3 scripts/ode_pipeline.py learn` first")
    rules = json.loads(RULES_PATH.read_text())
    rule_lookup = {(r["class"], r["role"]): r["concept"] for r in rules}

    sentences = Path(args.input).read_text().splitlines() if args.input else [args.sentence]
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        emitted = extract_concepts(client, args.model, sentence, ontology, rule_lookup)
        print(f"\nSENTENCE: {sentence}")
        if not emitted:
            print("  (no learned rule matched any grounded fact)")
        for e in emitted:
            print(f"  {e['concept']} = {e['filler']!r}  [via {e['via']}, trigger={e['trigger']!r}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ground", "learn", "extract"])
    parser.add_argument("--sentence", default=None)
    parser.add_argument("--input", default=None, help="file with one sentence per line (extract mode)")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_MODEL", "llama4-scout-17b"))
    args = parser.parse_args()

    ontology = load_ontology()
    client = get_client()

    if args.command == "ground":
        if not args.sentence:
            sys.exit("--sentence required for `ground`")
        cmd_ground(args, client, ontology)
    elif args.command == "learn":
        cmd_learn(args, client, ontology)
    elif args.command == "extract":
        if not args.input and not args.sentence:
            sys.exit("--input or --sentence required for `extract`")
        cmd_extract(args, client, ontology)


if __name__ == "__main__":
    main()
