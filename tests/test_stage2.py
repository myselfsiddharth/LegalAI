"""Stage 2 invariants. Each pins a defect that actually occurred during development."""
import pytest

from src.extract.canonicalize import make_label
from src.extract.fact_filters import (classify_content, in_defendant_view, in_plaintiff_view,
                                      is_legal_statement, orientation_coverage)
from src.extract.facts import chunk


# --- legal-statement filter -----------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("Section 13 of the said Act provided for a second appeal to the High Court", "law"),
    ("It is well settled that possession must be open and hostile", "law"),
    ("In the case of a grant of an office to be remunerated by the use of land, "
     "the land would be prima facie resumable", "law"),
    ("The Privy Council pointed out the distinction between the grant of an office", "authority"),
    ("Ram Gopal v. Nand Lal held that a mortgage by conditional sale is not a sale", "authority"),
    ("The appellant purchased the land under a registered sale deed in 1972", "fact"),
    ("The City Civil Court enhanced the compensation to Rs. 3,31,092.", "fact"),
    ("The jagir of Nawabag was granted by the Kings of Delhi to Shah Abdul Huq.", "fact"),
])
def test_classify_content(text, expected):
    assert classify_content(text) == expected


def test_legal_statements_are_rejected_as_facts():
    assert is_legal_statement("Section 5 of the Limitation Act provides for condonation")
    assert not is_legal_statement("The suit was filed on 3 February 1960")


# --- party orientation ----------------------------------------------------------------
def test_admitted_facts_enter_both_party_views():
    """§8.1: T_plaintiff = plaintiff-asserted OR admitted; T_defendant = defendant OR admitted."""
    f = {"asserted_by": "court_narrative", "disputed_status": "admitted"}
    assert in_plaintiff_view(f) and in_defendant_view(f)


def test_admittedness_is_read_from_disputed_status_not_only_asserted_by():
    """The model records admittedness in `disputed_status`. Reading `asserted_by` alone put 79.5%
    of facts in NEITHER view; this is the bug that fix guards."""
    facts = [{"asserted_by": "court_narrative", "disputed_status": "admitted"} for _ in range(8)]
    facts += [{"asserted_by": "plaintiff", "disputed_status": "contested"}]
    facts += [{"asserted_by": "court_narrative", "disputed_status": "contested"}]
    cov = orientation_coverage(facts)
    assert cov["oriented_frac"] == 0.9
    assert cov["orphan_frac"] == 0.1          # only the contested-and-unattributed one


def test_contested_unattributed_fact_enters_neither_view():
    f = {"asserted_by": "court_narrative", "disputed_status": "contested"}
    assert not in_plaintiff_view(f) and not in_defendant_view(f)


# --- canonical label construction -----------------------------------------------------
def test_negation_is_part_of_the_atom():
    """"notice served" and "notice not served" must not mine as the same item."""
    pos = make_label("Event", "Notice Served", {"polarity": "pos"})
    neg = make_label("Event", "Notice Served", {"polarity": "neg"})
    assert pos != neg and neg.endswith(".not")


def test_duration_bin_is_part_of_the_atom():
    short = make_label("Possession", "continuous", {"duration_years_bin": "lt3"})
    long = make_label("Possession", "continuous", {"duration_years_bin": "gt12"})
    assert short != long


def test_not_shown_and_proved_false_are_different_atoms():
    """§10.4: 'not proved' and 'proved false' are different facts, and a burden-shift rule that
    cannot tell them apart is wrong."""
    a = make_label("Event", "Notice Served", {"proof_status": "not_shown"})
    b = make_label("Event", "Notice Served", {"proof_status": "proved_false"})
    c = make_label("Event", "Notice Served", {"proof_status": "admitted"})
    assert a != b and a != c and b != c


def test_label_is_stable_and_lowercase():
    a = make_label("Instrument", "Sale Deed", {})
    b = make_label("instrument", "sale deed", {})
    assert a == b == "instrument.sale_deed"


# --- chunking -------------------------------------------------------------------------
def test_chunks_cover_the_text_and_overlap():
    text = " ".join(f"Sentence number {i} of the excerpt." for i in range(900))
    ch = chunk(text)
    assert len(ch) > 1
    for base, c in ch:
        assert text[base:base + len(c)] == c      # offsets index the source
    assert ch[1][0] < ch[0][0] + len(ch[0][1])    # overlap, so a straddling fact stays quotable


def test_short_text_is_one_chunk():
    assert len(chunk("A short excerpt.")) == 1
