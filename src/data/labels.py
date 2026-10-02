"""Stage 1 (§5.1): a party-oriented outcome label, plus the shortcut feature it must be
ablated against.

Two labels come out of every judgment, and keeping them apart is the point:

  `initiator_outcome`   -- did the party who brought THIS proceeding win before the
                           Supreme Court? Read off the operative order.
  `prior_court_outcome` -- what did the court below do? Read off the procedural recital.

Why the second one matters, and why it is a field rather than a feature: measuring cue
positions across post-2000 judgments showed 19.2% of outcome-shaped language sits in the
FIRST 30% of the text. Those hits are not this court's disposition; they are the recital
("the High Court dismissed the suit", "the trial court decreed"). That makes the court
below's disposition simultaneously a legitimate historical fact AND the strongest single
predictor available, because an apex court affirms far more often than it reverses.

So a positional mask leaks, and a mask aggressive enough to remove the recital deletes real
facts. The resolution is to label it explicitly and make it an ablation axis (§10.2): the
gap between a model with it and without it is how much of any headline accuracy is the
shortcut rather than the law.

Rules run first and carry a matched span, so every label is auditable and §5.2 can verify
the mask actually removed the evidence. Residuals go to the LLM (see `label_residuals`).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

from src import paths

# ---------------------------------------------------------------------------
# The operative order sits at the end. We search a window rather than the whole text so
# that the recital's "the High Court dismissed the appeal" cannot be mistaken for it.
ORDER_WINDOW_CHARS = 4000

# "In the result", "Accordingly", "For the reasons aforesaid" open the operative order in
# Indian judgments; when present they mark a tighter, higher-precision window.
ORDER_OPENER = re.compile(
    r"\b(in the result|in the premises|accordingly|for the (foregoing |aforesaid |above )?reasons"
    r"|for these reasons|in view of the (foregoing|above|aforesaid)|we therefore|hence, we"
    r"|resultantly|consequently, the)\b", re.I)

# Two tiers of proceeding, and the distinction decides the label.
#
# TIER 1 is the proceeding actually before THIS court. TIER 2 is the underlying proceeding
# below. Indian orders routinely dispose of both in one breath:
#
#     "The appeal is allowed ... and the writ petition filed by the respondent stands dismissed."
#
# Reading the LAST disposition in that sentence gives LOSE, when the appellant plainly WON --
# the dismissal is a CONSEQUENCE of the appeal succeeding, applied to the opposing party's
# case. This was the single largest error in the rules labeller: 196 cases where the LLM said
# WIN and the rules said LOSE, and on audit the LLM was right in all 10 sampled.
#
# So a tier-1 disposition always governs; tier 2 is consulted only when no tier-1 match exists
# (e.g. a suit decided directly, or a writ petition under Article 32).
_PROC_T1 = r"(?:civil |criminal |special leave |letters patent )?(?:appeals?|slps?|special leave petitions?)"
_PROC_T2 = r"(?:writ petitions?|petitions?|suits?|revisions?|applications?|proceedings?)"
_PROC = rf"(?:{_PROC_T1}|{_PROC_T2})"

# Disposition patterns. Order matters: PARTIAL and REMAND are checked before the plain
# allowed/dismissed forms, because "partly allowed" also matches "allowed".
PARTIAL = re.compile(
    rf"\b{_PROC}\b[^.]{{0,100}}?\b(?:is|are|stands?|shall stand)\s+(?:hereby\s+)?"
    rf"(?:partly|partially|in part)\s+allowed\b"
    rf"|\ballowed\s+(?:only\s+)?in\s+part\b|\bpartly\s+allowed\b|\bpartly\s+succeeds?\b"
    rf"|\ballowed\s+to\s+the\s+(?:extent|aforesaid\s+extent|above\s+extent)\b"
    rf"|\ballowed\s+to\s+that\s+extent\b"
    rf"|\ballowed\s+in\s+part\s+only\b"
    rf"|\bsucceeds?\s+(?:only\s+)?in\s+part\b|\bsucceed\s+(?:only\s+)?in\s+part\b"
    rf"|\b(?:allowed|modified)\s+to\s+the\s+extent\s+(?:indicated|mentioned|stated|set\s+out)\b",
    re.I)

REMAND = re.compile(
    r"\bremand(?:ed)?\s+(?:the\s+)?(?:matter|case|suit|appeal|proceedings?)\b"
    r"|\b(?:matter|case|suit|appeal)\s+is\s+remand(?:ed)?\b"
    r"|\bremitted?\s+(?:back\s+)?to\s+the\s+(?:High Court|trial court|first appellate|District Judge)"
    r"|\bsent\s+back\s+to\s+the\s+(?:High Court|trial court)", re.I)


# The span between the proceeding noun and its verb must not step over ANOTHER disposition.
# Without this, "the appeal is allowed and the suit is dismissed" matches the DISMISSED
# pattern too -- "appeal" binds across "allowed and the suit" to "is dismissed" -- so both
# classes fire and the later one wins, yielding LOSE for a clear win. Forcing each proceeding
# noun to bind to its nearest disposition verb removes that whole class of error.
_GAP = r"(?:(?!\b(?:allowed|dismissed|rejected|accepted|succeeds?|fails?)\b)[^.]){0,90}?"


def _allowed(proc: str) -> re.Pattern:
    return re.compile(
        rf"\b{proc}\b{_GAP}\b(?:is|are|stands?|shall stand|must be|deserves? to be)\s+"
        rf"(?:hereby\s+)?(?:allowed|accepted)\b"
        rf"|\bwe\s+(?:hereby\s+)?allow\s+(?:the|these|this|all)\s+{proc}\b"
        rf"|\b{proc}\b{_GAP}\bsucceeds?\b", re.I)


def _dismissed(proc: str) -> re.Pattern:
    return re.compile(
        rf"\b{proc}\b{_GAP}\b(?:is|are|stands?|shall stand|must be|deserves? to be)\s+"
        rf"(?:hereby\s+)?(?:dismissed|rejected|not maintainable)\b"
        rf"|\bwe\s+(?:hereby\s+)?dismiss\s+(?:the|these|this|all)\s+{proc}\b"
        rf"|\b{proc}\b{_GAP}\b(?:fails?|is devoid of merit|has no merit)\b", re.I)


ALLOWED_T1, DISMISSED_T1 = _allowed(_PROC_T1), _dismissed(_PROC_T1)
ALLOWED_T2, DISMISSED_T2 = _allowed(_PROC_T2), _dismissed(_PROC_T2)
# kept for callers/tests that want the tier-agnostic form
ALLOWED, DISMISSED = _allowed(_PROC), _dismissed(_PROC)

OTHER = re.compile(
    r"\bdisposed of (?:as|in terms of|by consent)\b|\bwithdrawn\b|\binfructuous\b"
    r"|\bdoes not survive\b|\brendered academic\b|\bcompromise\b|\bsettled (?:out of court|amicably)\b"
    r"|\bnothing survives\b|\bconsent terms\b", re.I)

# The court below. Deliberately narrow: it must NAME the forum, so it cannot pick up this
# court's own order. "affirmed/upheld/set aside" carry the appellate relationship.
PRIOR_COURT = re.compile(
    r"\b(?:the\s+)?(High Court|trial court|trial Judge|first appellate court|lower appellate court"
    r"|District Judge|Subordinate Judge|Civil Judge|learned Single Judge|Division Bench|Tribunal"
    r"|Collector|Land Acquisition Officer|Revenue Tribunal)\b"
    r"[^.]{0,120}?\b(dismissed|allowed|decreed|rejected|set aside|upheld|affirmed|confirmed"
    r"|remanded|quashed|granted|refused|declined)\b", re.I)


# A negator that governs the disposition verb: same sentence, immediately before it.
# Guards against "we do not think it proper to again remand the matter" (a REFUSAL to
# remand) and "the rest of the appeals cannot now be finally disposed of". It must NOT
# fire on "no order as to costs", which trails the verb rather than governing it -- hence
# the check looks only at text BEFORE the match and forbids a sentence boundary.
NEGATOR = re.compile(
    r"\b(?:do(?:es)?\s+not|did\s+not|cannot|can\s+not|could\s+not|shall\s+not|will\s+not"
    r"|would\s+not|need\s+not|decline[sd]?\s+to|refus\w+\s+to|no\s+(?:ground|reason|need|occasion)"
    r"|not\s+(?:proper|necessary|inclined|possible)|unnecessary\s+to|instead\s+of)\b"
    r"[^.;]{0,60}$", re.I)


def _negated(window: str, match: re.Match) -> bool:
    """True when a negator governs this disposition verb rather than merely sitting near it."""
    return bool(NEGATOR.search(window[max(0, match.start() - 90):match.start()]))


@dataclass
class OutcomeLabel:
    doc_id: str
    year: int
    # --- this court's disposition, from the appellant/petitioner's perspective
    initiator_outcome: str            # WIN | LOSE | PARTIAL | REMAND | OTHER | UNKNOWN
    label_source: str                 # 'rules' | 'llm' | 'human'
    order_span: list[int] | None = None       # [start, end] char offsets of the matched text
    order_evidence: str | None = None         # the matched sentence, for audit
    order_window: list[int] | None = None     # [start, end] of the region searched
    ambiguous: bool = False           # both allow and dismiss fired in the order window
    negated_skipped: int = 0          # disposition verbs rejected as negated (audit trail)
    disposition_tier: int = 0         # 1 = read from the appeal/SLP, 2 = from the suit/petition
    competing: list[str] = field(default_factory=list)
    # --- the shortcut axis (§10.2 ablation 2b), NOT part of the label
    prior_court_outcome: str | None = None    # affirmed_below | reversed_below | ... | None
    prior_court_evidence: str | None = None
    # --- §5.1 asks for both; whether the SC appellant was the original plaintiff cannot be
    # read off the order, so this is filled by the LLM pass, not by rules.
    original_plaintiff_outcome: str | None = None
    proceeding_type: str | None = None        # appeal | writ | slp | suit | other


def _proceeding_type(text: str) -> str:
    head = text[:6000].lower()
    if "special leave petition" in head or re.search(r"\bslp\b", head):
        return "slp"
    if "writ petition" in head or "article 32" in head:
        return "writ"
    if re.search(r"\b(civil|criminal) appeal no", head):
        return "appeal"
    if "suit" in head:
        return "suit"
    return "other"


def find_order_window(text: str) -> tuple[int, int]:
    """The region to read the disposition from.

    Prefer the last explicit order opener ("In the result, ...") in the final third of the
    document; fall back to a fixed tail window. Restricting the search is what stops the
    procedural recital being read as the holding.
    """
    n = len(text)
    # The fallback window is a fixed 4,000 characters from the end, which on a SHORT document
    # starts at 0 and makes the whole text the "order region". Measured on this corpus: 59 cases
    # (0.85%) had order_region_start == 0 and every one of them produced unusable masked text --
    # a third of all unusable cases, dropping out of eligibility silently. The window is therefore
    # bounded below by 70% of the document, so it can never swallow the whole thing.
    tail_start = max(int(n * 0.70), n - ORDER_WINDOW_CHARS)
    last = None
    for m in ORDER_OPENER.finditer(text):
        if m.start() > n * 0.6:                     # only openers late in the document
            last = m.start()
    if last is not None:
        return max(int(n * 0.40), last - 200), n
    return tail_start, n


def label_by_rules(text: str) -> OutcomeLabel:
    lo, hi = find_order_window(text)
    window = text[lo:hi]

    # Tier 1 = the proceeding before this court; consult tier 2 only if tier 1 is silent.
    if ALLOWED_T1.search(window) or DISMISSED_T1.search(window):
        allow_rx, dismiss_rx, tier = ALLOWED_T1, DISMISSED_T1, 1
    else:
        allow_rx, dismiss_rx, tier = ALLOWED_T2, DISMISSED_T2, 2

    hits: list[tuple[str, re.Match]] = []
    for name, rx in (("PARTIAL", PARTIAL), ("REMAND", REMAND),
                     ("WIN", allow_rx), ("LOSE", dismiss_rx), ("OTHER", OTHER)):
        m = None
        for cand in rx.finditer(window):            # keep the LAST non-negated match
            if not _negated(window, cand):
                m = cand
        if m:
            hits.append((name, m))

    names = [h[0] for h in hits]
    outcome, chosen, ambiguous, competing = "UNKNOWN", None, False, []

    if hits:
        # Precedence: a partial allowance or a remand describes the disposition more
        # precisely than the bare allow/dismiss that also matched it.
        for pref in ("PARTIAL", "REMAND"):
            if pref in names:
                outcome = pref
                chosen = dict(hits)[pref]
                break
        else:
            if "WIN" in names and "LOSE" in names:
                # Genuinely mixed, or several appeals disposed of differently. The later
                # statement is usually the operative one, but flag it either way -- these
                # go to the LLM rather than being guessed.
                ambiguous, competing = True, ["WIN", "LOSE"]
                w, l = dict(hits)["WIN"], dict(hits)["LOSE"]
                outcome = "WIN" if w.start() > l.start() else "LOSE"
                chosen = w if w.start() > l.start() else l
            elif "WIN" in names:
                outcome, chosen = "WIN", dict(hits)["WIN"]
            elif "LOSE" in names:
                outcome, chosen = "LOSE", dict(hits)["LOSE"]
            elif "OTHER" in names:
                outcome, chosen = "OTHER", dict(hits)["OTHER"]

    span = evidence = None
    if chosen is not None:
        span = [lo + chosen.start(), lo + chosen.end()]
        evidence = window[max(0, chosen.start() - 60):chosen.end() + 60].replace("\n", " ").strip()

    prior = prior_ev = None
    pm = PRIOR_COURT.search(text[:int(len(text) * 0.5)] or text)
    if pm:
        verb = pm.group(2).lower()
        prior = {"dismissed": "below_dismissed", "rejected": "below_dismissed",
                 "refused": "below_dismissed", "declined": "below_dismissed",
                 "allowed": "below_allowed", "decreed": "below_allowed",
                 "granted": "below_allowed", "quashed": "below_allowed",
                 "set aside": "below_set_aside", "upheld": "below_upheld",
                 "affirmed": "below_upheld", "confirmed": "below_upheld",
                 "remanded": "below_remanded"}.get(verb)
        prior_ev = text[max(0, pm.start() - 40):pm.end() + 40].replace("\n", " ").strip()

    return OutcomeLabel(
        doc_id="", year=0, initiator_outcome=outcome, label_source="rules",
        order_span=span, order_evidence=evidence, order_window=[lo, hi],
        ambiguous=ambiguous, competing=competing,
        prior_court_outcome=prior, prior_court_evidence=prior_ev,
        proceeding_type=_proceeding_type(text), disposition_tier=tier,
    )


def iter_cases():
    with paths.CASE_TEXT.open() as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    out_path = paths.INTERIM / "outcome_labels.jsonl"
    stats = Counter()
    prior_stats = Counter()
    proc_stats = Counter()
    rows = []
    for rec in iter_cases():
        if rec["n_chars"] < 500:
            stats["skipped_no_text"] += 1
            continue
        lab = label_by_rules(rec["text"])
        lab.doc_id, lab.year = rec["doc_id"], rec["year"]
        rows.append(lab)
        stats[lab.initiator_outcome] += 1
        if lab.ambiguous:
            stats["_ambiguous"] += 1
        prior_stats[lab.prior_court_outcome or "none"] += 1
        proc_stats[lab.proceeding_type] += 1

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(asdict(r)) + "\n")

    n = len(rows)
    binary = stats["WIN"] + stats["LOSE"]
    print(f"outcome labels: {n} cases -> {out_path}")
    print("\n  initiator_outcome (rules only):")
    for k in ("WIN", "LOSE", "PARTIAL", "REMAND", "OTHER", "UNKNOWN"):
        print(f"    {k:8s} {stats[k]:5d}  {100*stats[k]/n:5.1f}%")
    print(f"    -> binary WIN/LOSE usable: {binary} ({100*binary/n:.1f}%), "
          f"balance {100*stats['WIN']/binary:.1f}% WIN" if binary else "")
    print(f"    -> ambiguous (both fired, sent to LLM): {stats['_ambiguous']} "
          f"({100*stats['_ambiguous']/n:.1f}%)")
    print(f"    -> residual UNKNOWN for the LLM pass: {stats['UNKNOWN']} "
          f"({100*stats['UNKNOWN']/n:.1f}%)")
    print("\n  prior_court_outcome (the shortcut axis, §10.2):")
    for k, v in prior_stats.most_common():
        print(f"    {k:20s} {v:5d}  {100*v/n:5.1f}%")
    print("\n  proceeding_type:")
    for k, v in proc_stats.most_common():
        print(f"    {k:8s} {v:5d}  {100*v/n:5.1f}%")
    if stats["skipped_no_text"]:
        print(f"\n  skipped (near-empty text, likely scanned): {stats['skipped_no_text']}")


if __name__ == "__main__":
    main()
