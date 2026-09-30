"""Deterministic filters over extracted facts (§6.1 quality gate).

Two jobs, both of which the prompt alone does not do reliably:

1. **Reject statements of law.** The extraction prompt forbids them and the model emits them
   anyway: "Section 13 of the said Act provided for a second appeal", "The Privy Council pointed
   out the distinction between...". These are not facts about the dispute and they poison pattern
   mining -- a rule like {section_13, second_appeal} -> LOSE would encode the statute book, not
   the case. The earlier iteration of this project hit the same wall and solved it the same way:
   a content guard reading the filler's shape cut misrouted statute spans from 196 to 0 with no
   recall cost, which is the precedent for doing it deterministically here.

2. **Flag the party-orientation problem.** §8.1 builds T_plaintiff from facts asserted by the
   plaintiff OR admitted, and T_defendant likewise. A fact marked `court_narrative` enters
   NEITHER view, so an extractor that labels most facts `court_narrative` silently empties both.
   `orientation_coverage` measures that directly, so it cannot regress unnoticed.
"""
from __future__ import annotations

import re

# --- statements of law, not facts about this dispute -------------------------------------
STATUTE_SUBJECT = re.compile(
    r"^\s*(?:under\s+)?(?:section|s\.|sec\.|article|art\.|order|rule|clause|schedule)\s*\d+"
    r"|^\s*(?:the\s+)?[A-Z][\w\s]{2,40}\s+Act(?:,?\s+\d{4})?\s+(?:provides?|prescribes?|requires?"
    r"|lays\s+down|contemplates?|defines?|bars?|permits?)"
    r"|\b(?:section|s\.|article|art\.|order|rule|clause)\s*\d+[A-Za-z()\d]*\s*(?:of\s+[^,.]{3,60})?"
    r"\s+(?:provides?|prescribes?|requires?|lays\s+down|contemplates?|defines?|bars?|permits?"
    r"|applies|stipulates?|mandates?|enables?|empowers?)", re.I)

COURT_HELD = re.compile(
    r"\b(?:the\s+)?(?:Privy\s+Council|Supreme\s+Court|House\s+of\s+Lords|Federal\s+Court"
    r"|this\s+Court|their\s+Lordships|the\s+Bench|Full\s+Bench|Constitution\s+Bench)\b"
    r"[^.]{0,60}?\b(?:held|observed|pointed\s+out|laid\s+down|ruled|opined|explained|clarified"
    r"|reiterated|distinguished|approved|overruled|considered)\b", re.I)

LEGAL_PRINCIPLE = re.compile(
    r"\b(?:it\s+is\s+(?:well\s+)?settled|the\s+law\s+(?:is|requires)|the\s+principle\s+(?:is|that)"
    r"|prima\s+facie\s+(?:resumable|liable)|as\s+a\s+matter\s+of\s+law"
    r"|the\s+legal\s+position\s+is|the\s+test\s+(?:is|laid)"
    r"|in\s+the\s+(?:former|latter)\s+case,?\s+the)\b", re.I)

# A case citation standing as the fact's subject: the fact is about an authority, not the dispute.
CITATION_SUBJECT = re.compile(
    r"^\s*(?:in\s+)?[A-Z][\w.'-]+(?:\s+[A-Z][\w.'-]+){0,4}\s+v(?:s?\.?|ersus)\s+", re.I)


def classify_content(text: str) -> str:
    """'law' | 'authority' | 'fact'. Checked in order of specificity."""
    t = (text or "").strip()
    if not t:
        return "fact"
    if COURT_HELD.search(t) or CITATION_SUBJECT.match(t):
        return "authority"
    if STATUTE_SUBJECT.search(t) or LEGAL_PRINCIPLE.search(t):
        return "law"
    return "fact"


def is_legal_statement(text: str) -> bool:
    return classify_content(text) != "fact"


# --- party orientation ------------------------------------------------------------------
# §8.1: T_plaintiff = facts asserted by the plaintiff OR admitted; T_defendant likewise.
#
# "Admitted" there is a statement about whether a fact is CONTESTED, and the extraction schema
# carries that in `disputed_status`, not in `asserted_by`. The two fields overlap, and the model
# resolves the overlap sensibly: it uses `asserted_by` for attribution and `disputed_status` for
# admittedness. Measured on a 12-case pilot, `asserted_by == "admitted"` was **0%** while
# `disputed_status == "admitted"` was 81% -- and tightening the prompt to insist on
# `asserted_by="admitted"` moved court_narrative only 84% -> 79.5%.
#
# So orientation reads both fields. Reading `asserted_by` alone would discard ~80% of facts as
# `court_narrative` and leave both party views nearly empty, which would look like an extraction
# failure when the information was present the whole time.
def is_admitted(f: dict) -> bool:
    return f.get("disputed_status") == "admitted" or f.get("asserted_by") == "admitted"


def in_plaintiff_view(f: dict) -> bool:
    return f.get("asserted_by") == "plaintiff" or is_admitted(f)


def in_defendant_view(f: dict) -> bool:
    return f.get("asserted_by") == "defendant" or is_admitted(f)


def orientation_coverage(facts: list[dict]) -> dict:
    """What fraction of facts can enter a party-oriented transaction at all."""
    n = len(facts) or 1
    plaintiff = sum(1 for f in facts if in_plaintiff_view(f))
    defendant = sum(1 for f in facts if in_defendant_view(f))
    oriented = sum(1 for f in facts if in_plaintiff_view(f) or in_defendant_view(f))
    return {
        "n_facts": len(facts),
        "oriented": oriented,
        "oriented_frac": round(oriented / n, 4),
        "t_plaintiff": plaintiff,
        "t_defendant": defendant,
        "admitted_frac": round(sum(1 for f in facts if is_admitted(f)) / n, 4),
        "attributed_frac": round(
            sum(1 for f in facts if f.get("asserted_by") in ("plaintiff", "defendant")) / n, 4),
        # facts that enter NEITHER view: contested but unattributed
        "orphan_frac": round(
            sum(1 for f in facts
                if not in_plaintiff_view(f) and not in_defendant_view(f)) / n, 4),
    }
