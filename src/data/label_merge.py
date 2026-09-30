"""Stage 1 (§5.1): reconcile the rules and LLM labellers into one frozen label set.

The policy, and the reasoning for each branch:

  1. Both decided and agree            -> accept, `source=both`, highest confidence.
  2. Rules decided, LLM did not (or its quote does not verify) -> accept the rules label.
     Rules carry a matched span in the judgment, so they are evidence-backed by construction.
  3. Rules said UNKNOWN, LLM decided with a GROUNDED quote -> accept the LLM label. This is
     the recovery path for the 33% of cases no regex could reach.
  4. Both decided and disagree         -> `source=conflict`, label held as UNRESOLVED and
     queued for the §5.1 human check. A coin-flip between two disagreeing labellers would
     manufacture label noise and quietly cap every downstream number.
  5. Neither decided                   -> UNKNOWN, excluded from outcome experiments.

An ungrounded quote never carries a label on its own. The quote is the only evidence the LLM
offers, and an unverifiable quote is not evidence.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src import paths

OUT_PATH = paths.INTERIM / "outcome_labels_final.jsonl"
AUDIT_PATH = paths.INTERIM / "label_conflicts.jsonl"
DECIDED = ("WIN", "LOSE", "PARTIAL", "REMAND", "OTHER")


def _load_llm() -> dict:
    out = {}
    p = paths.INTERIM / "outcome_labels_llm.jsonl"
    if not p.exists():
        return out
    with p.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("status") == "ok":
                out[r["doc_id"]] = r          # later rows win
    return out


def merge() -> tuple[list[dict], Counter, list[dict]]:
    rules = {json.loads(l)["doc_id"]: json.loads(l)
             for l in open(paths.INTERIM / "outcome_labels.jsonl")}
    llm = _load_llm()
    rows, stats, conflicts = [], Counter(), []

    for doc_id, r in rules.items():
        rl = r["initiator_outcome"]
        m = llm.get(doc_id)
        ll = m["initiator_outcome"] if m else None
        grounded = bool(m and m.get("quote_grounded"))
        llm_usable = bool(ll in DECIDED and grounded)

        if rl in DECIDED and llm_usable:
            if rl == ll:
                label, source, conf = rl, "both", 0.95
                stats["agree"] += 1
            else:
                label, source, conf = "UNRESOLVED", "conflict", 0.0
                stats["conflict"] += 1
                conflicts.append({"doc_id": doc_id, "year": r["year"], "rules": rl, "llm": ll,
                                  "rules_evidence": r.get("order_evidence"),
                                  "llm_quote": m.get("operative_quote"),
                                  "llm_confidence": m.get("confidence")})
        elif rl in DECIDED:
            label, source, conf = rl, "rules", 0.85
            stats["rules_only"] += 1
            if m and ll in DECIDED and not grounded:
                stats["_llm_dropped_ungrounded_quote"] += 1
        elif llm_usable:
            label, source, conf = ll, "llm", 0.75
            stats["llm_recovered"] += 1
        else:
            label, source, conf = "UNKNOWN", "none", 0.0
            stats["still_unknown"] += 1

        rows.append({
            "doc_id": doc_id, "year": r["year"],
            "outcome": label, "source": source, "confidence": conf,
            "rules_outcome": rl, "llm_outcome": ll, "llm_quote_grounded": grounded,
            "prior_court_outcome": r.get("prior_court_outcome"),
            "proceeding_type": r.get("proceeding_type"),
            "order_evidence": r.get("order_evidence"),
            "initiator_was_original_plaintiff": (m or {}).get("initiator_was_original_plaintiff"),
            "original_plaintiff_outcome": (m or {}).get("original_plaintiff_outcome"),
        })
        stats[label] += 1
    return rows, stats, conflicts


def main() -> None:
    rows, stats, conflicts = merge()
    with OUT_PATH.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with AUDIT_PATH.open("w") as f:
        for c in conflicts:
            f.write(json.dumps(c) + "\n")

    n = len(rows)
    binary = stats["WIN"] + stats["LOSE"]
    print(f"merged labels: {n} cases -> {OUT_PATH}")
    print("\n  final label:")
    for k in ("WIN", "LOSE", "PARTIAL", "REMAND", "OTHER", "UNRESOLVED", "UNKNOWN"):
        print(f"    {k:11s} {stats[k]:5d}  {100*stats[k]/n:5.1f}%")
    print(f"    -> binary WIN/LOSE: {binary} ({100*binary/n:.1f}%), "
          f"{100*stats['WIN']/max(1,binary):.1f}% WIN")
    print("\n  provenance:")
    for k in ("agree", "rules_only", "llm_recovered", "conflict", "still_unknown"):
        print(f"    {k:16s} {stats[k]:5d}")
    both = stats["agree"] + stats["conflict"]
    if both:
        print(f"\n  rules vs LLM agreement where both decided and quote verified: "
              f"{100*stats['agree']/both:.1f}%  (n={both})")
    if stats["_llm_dropped_ungrounded_quote"]:
        print(f"  LLM labels refused for an unverifiable quote: "
              f"{stats['_llm_dropped_ungrounded_quote']}")
    if conflicts:
        c = Counter((x["rules"], x["llm"]) for x in conflicts)
        print(f"\n  conflicts by direction (rules -> llm), queued in {AUDIT_PATH.name}:")
        for (a, b), k in c.most_common(8):
            print(f"    {a:8s} -> {b:8s} {k:4d}")


if __name__ == "__main__":
    main()


def load_final() -> dict:
    """The frozen label set every downstream stage reads.

    Normalised to the same shape the rules labeller emitted (`initiator_outcome`) so callers
    do not care which labeller decided a given case. UNRESOLVED and UNKNOWN are kept in the
    dict, not dropped, so a caller can count what it excluded.
    """
    if not OUT_PATH.exists():
        raise FileNotFoundError(f"{OUT_PATH} missing -- run `python -m src.data.label_merge`")
    out = {}
    with OUT_PATH.open() as f:
        for line in f:
            r = json.loads(line)
            r["initiator_outcome"] = r["outcome"]
            out[r["doc_id"]] = r
    return out
