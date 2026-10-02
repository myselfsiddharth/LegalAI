"""Stage 1 (§5.2): build `masked_text` -- the only text a predictor is allowed to see.

§0.4 calls leakage the single most important correctness rule in the project, so this module
is deliberately conservative: when a unit's role is unclear it is EXCLUDED, because a false
"facts" label silently inflates every downstream number while a false "analysis" label only
costs recall.

What survives: the facts narrative, the pleadings and prayers, and each side's arguments as
summarised by the court.
What is removed: the court's analysis, the operative order, the HEADNOTE, and any sentence
carrying an outcome cue.

## The HEADNOTE

79.5% of pre-2000 judgments carry one and 0.0% of post-2000 judgments do. It states the
outcome outright and lists the authorities relied on, so it is pure leakage and is dropped
whole. Note the consequence for §5.3: because its presence is almost perfectly determined by
era, a train-on-old / test-on-new split conflates "later in time" with "structurally
different document". That confound survives the masking and has to be reported, not masked
away -- which is why the court-held-out split is run alongside.

## Prior-court disposition: separated, not deleted

19.2% of outcome-shaped cues sit in the FIRST 30% of a judgment. Those are the procedural
recital -- "the High Court dismissed the suit" -- not this court's holding. That text is
simultaneously a legitimate fact about the case's history and the strongest single predictor
available, because an apex court affirms far more often than it reverses.

Deleting it would remove real facts; keeping it silently would let a model win by reading the
court below. So it goes into its own channel, `prior_court_text`, excluded from
`masked_text`. §10.2's ablation then runs with and without it, and the gap measures how much
of any headline accuracy is the shortcut rather than the law.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict, field

from src import paths
from src.data.segment import Unit, segment

# --- outcome cues: sentence-level scrub, applied regardless of a unit's role -------------
OUTCOME_CUE = re.compile(
    r"\b(?:we\s+(?:hold|allow|dismiss|set\s+aside|quash|decree|direct|conclude|are\s+of\s+the"
    r"\s+(?:view|opinion))"
    r"|in\s+the\s+result|in\s+the\s+premises|for\s+the\s+(?:foregoing|aforesaid|above)\s+reasons"
    r"|(?:appeal|appeals|petition|petitions|suit|suits|slp)\s+(?:is|are|stands?|shall\s+stand)"
    r"\s+(?:hereby\s+)?(?:allowed|dismissed|rejected|accepted|partly\s+allowed)"
    r"|(?:is|are)\s+hereby\s+(?:allowed|dismissed|quashed|set\s+aside)"
    r"|no\s+order\s+as\s+to\s+costs|with\s+costs\s+throughout"
    r"|appeal\s+(?:allowed|dismissed)\s*\.|petition\s+(?:allowed|dismissed)\s*\."
    r"|we\s+find\s+(?:no\s+)?(?:merit|force|substance)"
    r"|cannot\s+be\s+sustained|deserves?\s+to\s+be\s+(?:allowed|dismissed))\b", re.I)

# --- the court's own reasoning: exclude the unit -----------------------------------------
ANALYSIS_CUE = re.compile(
    r"\b(?:in\s+our\s+(?:opinion|view|judgment)|we\s+are\s+(?:unable|satisfied|inclined|not)"
    r"|it\s+is\s+(?:well\s+settled|settled\s+law|trite)|the\s+law\s+is\s+well\s+settled"
    r"|we\s+(?:agree|disagree|concur)|in\s+our\s+considered\s+(?:opinion|view)"
    r"|we\s+(?:see|find)\s+no\s+(?:reason|ground|merit|infirmity)"
    r"|the\s+(?:contention|submission|argument)\s+(?:is|must)\s+(?:not\s+)?"
    r"(?:be\s+)?(?:accepted|rejected|devoid)"
    r"|there\s+is\s+(?:no\s+)?(?:merit|force|substance)\s+in|rightly\s+held"
    r"|we\s+have\s+(?:carefully\s+)?(?:considered|examined|perused))\b", re.I)

# --- a party's argument, as summarised by the court: KEEP ---------------------------------
ARGUMENT_CUE = re.compile(
    r"\b(?:learned\s+(?:counsel|senior\s+counsel|advocate|Attorney|Solicitor)"
    r"|(?:it\s+(?:is|was)\s+(?:contended|submitted|urged|argued|pleaded))"
    r"|(?:contended|submitted|urged|argued)\s+(?:that|before\s+us|on\s+behalf)"
    r"|on\s+behalf\s+of\s+the\s+(?:appellant|respondent|petitioner|plaintiff|defendant)"
    r"|the\s+(?:appellant|respondent|petitioner)'?s?\s+(?:case|contention|submission)\s+is)\b",
    re.I)

# --- pleadings and prayers: KEEP ----------------------------------------------------------
PLEADING_CUE = re.compile(
    r"\b(?:filed\s+a\s+suit|instituted\s+a\s+suit|suit\s+for\s+(?:declaration|possession|partition"
    r"|specific\s+performance|injunction|cancellation|redemption)"
    r"|prayed\s+for|prayer\s+(?:is|was|clause)|sought\s+a\s+declaration|plaint|written\s+statement"
    r"|relief\s+(?:claimed|sought)|cause\s+of\s+action)\b", re.I)

# --- the case caption / reporting apparatus: drop -----------------------------------------
CAPTION_CUE = re.compile(
    r"Equivalent\s+citations|From\s+the\s+Judgment|Indian\s+Kanoon|indiankanoon\.org"
    r"|^\s*(?:HEADNOTE|JUDGMENT|J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|ORDER)\s*:?\s*$"
    r"|For\s+the\s+(?:Appellant|Respondent|Petitioner)|Appeal\s+by\s+Special\s+Leave"
    r"|The\s+Judgment\s+of\s+the\s+Court\s+was\s+delivered", re.I | re.M)

HEADNOTE_START = re.compile(r"^\s*HEADNOTE\s*:?\s*$", re.M | re.I)
# The HEADNOTE runs until the judgment proper begins.
JUDGMENT_START = re.compile(
    r"^\s*(?:J\s*U\s*D\s*G\s*M\s*E\s*N\s*T|JUDGMENT|JUDGEMENT|ORDER)\s*:?\s*$", re.M)

PRIOR_COURT_DISPOSITION = re.compile(
    r"\b(?:the\s+)?(?:High\s+Court|trial\s+court|trial\s+Judge|First\s+Appellate\s+Court"
    r"|lower\s+appellate\s+court|District\s+Judge|Subordinate\s+Judge|Civil\s+Judge"
    r"|Division\s+Bench|learned\s+Single\s+Judge|Tribunal|Collector|Land\s+Acquisition\s+Officer)\b"
    r"[^.]{0,140}?\b(?:dismissed|allowed|decreed|rejected|set\s+aside|upheld|affirmed|confirmed"
    r"|remanded|quashed|granted|refused|declined|reversed)\b", re.I)

KEEP_ROLES = {"facts", "pleadings", "arguments"}

# An ILDC-style input, for the comparison in src/eval/reasoning_ablation.py.
#
# Malik et al. (ACL 2021) construct ILDC by deleting "end section(s) directly stating the decision
# ... since that is what we aim to predict", and state that they "consider (along with the facts)
# the entire case (except the judgment)". So the court's REASONING is retained; only the
# disposition is removed. Their best model reads the last 512 tokens, which they justify because
# "the last parts of case proceedings usually contain the main information about the case and the
# rationale behind the judgment".
#
# §5.2 of this project excludes the analysis section as well, on the ground that reasoning is
# written knowing the outcome. These two role sets make that difference measurable on identical
# cases rather than argued about.
ILDC_STYLE_ROLES = KEEP_ROLES | {"analysis", "unclear", "headnote", "caption"}
ILDC_STYLE_NO_HEADNOTE_ROLES = KEEP_ROLES | {"analysis", "unclear", "caption"}


@dataclass
class MaskedCase:
    doc_id: str
    year: int
    masked_text: str
    ildc_style_text: str                       # + the court's reasoning; order still removed
    ildc_style_no_headnote: str                # same, minus the pre-2000 HEADNOTE artifact
    prior_court_text: str                      # the shortcut channel, kept out of masked_text
    n_chars_original: int
    n_chars_masked: int
    kept_spans: list[list[int]] = field(default_factory=list)
    unit_roles: list[str] = field(default_factory=list)
    role_counts: dict = field(default_factory=dict)
    dropped_sentences_outcome_cue: int = 0
    headnote_dropped_chars: int = 0
    order_region_start: int | None = None


def _classify(unit: Unit, order_start: int, frac: float) -> str:
    """Assign a role to one unit. Conservative by construction: anything that reaches the
    court's reasoning, or sits inside the order region, is excluded."""
    t = unit.text
    if unit.start >= order_start:
        return "order"
    if CAPTION_CUE.search(t) and len(t) < 1200:
        return "caption"
    if ANALYSIS_CUE.search(t):
        return "analysis"                       # checked before arguments: a unit that both
                                                # recites a submission and rules on it is analysis
    if ARGUMENT_CUE.search(t):
        return "arguments"
    if PLEADING_CUE.search(t):
        return "pleadings"
    if frac < 0.45:
        return "facts"                          # early, no reasoning markers -> narrative
    return "unclear"                            # excluded; see module docstring


def mask_case(rec: dict) -> MaskedCase:
    text = rec["text"]
    n = len(text)

    # 1. HEADNOTE: drop whole. Pure leakage (states outcome, lists authorities).
    head_lo = head_hi = None
    hm = HEADNOTE_START.search(text)
    if hm:
        jm = JUDGMENT_START.search(text, hm.end())
        head_lo, head_hi = hm.start(), (jm.start() if jm else min(n, hm.start() + 12000))

    # 2. Operative order region: everything from here on is excluded.
    from src.data.labels import find_order_window
    order_start, _ = find_order_window(text)

    units = segment(text)
    kept: list[tuple[int, int, str]] = []
    prior_bits: list[str] = []
    roles, counts = [], Counter()
    dropped_cue = 0

    ildc_keep: list[str] = []
    ildc_keep_nh: list[str] = []
    for u in units:
        frac = u.start / max(1, n)
        role = _classify(u, order_start, frac)
        # The HEADNOTE is a span, not a unit. Marking any OVERLAPPING unit as headnote once
        # discarded whole judgments whenever segmentation was coarse, so the span is applied
        # per sentence below and only a unit lying ENTIRELY inside it is relabelled here.
        if head_lo is not None and u.start >= head_lo and u.end <= head_hi:
            role = "headnote"
        roles.append(role)
        counts[role] += 1

        # ILDC-style arms: everything except the operative order. The HEADNOTE variant exists
        # because a pre-2000 headnote lists the outcome and the authorities outright, which is an
        # artifact of this corpus rather than of ILDC's construction.
        in_headnote = head_lo is not None and u.start < head_hi and u.end > head_lo
        if role in ILDC_STYLE_ROLES:
            ildc_keep.append(u.text)
        if role in ILDC_STYLE_NO_HEADNOTE_ROLES and not in_headnote:
            ildc_keep_nh.append(u.text)

        if role not in KEEP_ROLES:
            continue

        # 3. Sentence-level scrub inside kept units.
        for s_lo, s_hi, s in u.sentences():
            if head_lo is not None and s_lo < head_hi and s_hi > head_lo:
                continue                        # sentence falls in the HEADNOTE span
            if OUTCOME_CUE.search(s):
                dropped_cue += 1
                continue
            if PRIOR_COURT_DISPOSITION.search(s):
                prior_bits.append(s.strip())    # separated, not deleted
                continue
            kept.append((s_lo, s_hi, s))

    masked = "\n".join(s for _, _, s in kept)
    return MaskedCase(
        doc_id=rec["doc_id"], year=rec["year"],
        masked_text=masked,
        ildc_style_text="\n".join(ildc_keep),
        ildc_style_no_headnote="\n".join(ildc_keep_nh),
        prior_court_text="\n".join(prior_bits),
        n_chars_original=n, n_chars_masked=len(masked),
        kept_spans=[[lo, hi] for lo, hi, _ in kept],
        unit_roles=roles, role_counts=dict(counts),
        dropped_sentences_outcome_cue=dropped_cue,
        headnote_dropped_chars=(head_hi - head_lo) if head_lo is not None else 0,
        order_region_start=order_start,
    )


def main() -> None:
    out_path = paths.INTERIM / "masked_text.jsonl"
    role_totals, stats = Counter(), Counter()
    ratios = []
    n_cases = 0
    with out_path.open("w") as out, paths.CASE_TEXT.open() as f:
        for line in f:
            rec = json.loads(line)
            if rec["n_chars"] < 500:
                stats["skipped_no_text"] += 1
                continue
            mc = mask_case(rec)
            out.write(json.dumps(asdict(mc)) + "\n")
            n_cases += 1
            for k, v in mc.role_counts.items():
                role_totals[k] += v
            stats["dropped_cue_sentences"] += mc.dropped_sentences_outcome_cue
            if mc.headnote_dropped_chars:
                stats["cases_with_headnote"] += 1
            if mc.prior_court_text:
                stats["cases_with_prior_court_text"] += 1
            if mc.n_chars_masked < 500:
                stats["masked_too_short"] += 1
            ratios.append(mc.n_chars_masked / max(1, mc.n_chars_original))

    import statistics
    print(f"masked text: {n_cases} cases -> {out_path}")
    print(f"  retained fraction of original chars: median {statistics.median(ratios):.1%}, "
          f"mean {statistics.fmean(ratios):.1%}")
    print(f"  cases whose masked text is under 500 chars (unusable): {stats['masked_too_short']} "
          f"({100*stats['masked_too_short']/n_cases:.1f}%)")
    print(f"  HEADNOTE found and dropped: {stats['cases_with_headnote']} "
          f"({100*stats['cases_with_headnote']/n_cases:.1f}%)")
    print(f"  prior-court disposition separated: {stats['cases_with_prior_court_text']} "
          f"({100*stats['cases_with_prior_court_text']/n_cases:.1f}%)")
    print(f"  outcome-cue sentences scrubbed from KEPT units: {stats['dropped_cue_sentences']:,}")
    print("\n  unit roles across all cases:")
    tot = sum(role_totals.values())
    for k, v in role_totals.most_common():
        flag = "KEEP" if k in KEEP_ROLES else "drop"
        print(f"    {k:10s} {v:7d}  {100*v/tot:5.1f}%  [{flag}]")


if __name__ == "__main__":
    main()
