"""Stage 1 invariants. These guard the rules §0.4 calls the most important in the project,
so each one encodes a bug that actually occurred rather than a hypothetical."""
import json
import re

import pytest

from src import paths
from src.data import labels as L
from src.data.label_llm import quote_is_grounded
from src.data.mask import OUTCOME_CUE, mask_case
from src.data.segment import segment
from src.data.splits import party_signature


# --- labelling ------------------------------------------------------------------
@pytest.mark.parametrize("text,expected", [
    ("In the result, the appeal is allowed and the judgment of the High Court is set aside.", "WIN"),
    ("For the reasons abovementioned, the appeal lacks in merits and the same is dismissed, "
     "with no order as to costs.", "LOSE"),
    ("Accordingly, this appeal is partly allowed.", "PARTIAL"),
    ("In the result, the impugned judgment is set aside and the matter is remanded to the "
     "trial Court for a fresh trial.", "REMAND"),
    # a negator governing the verb is a REFUSAL, not a disposition
    ("we do not think it proper to again remand the matter to the trial court. "
     "The appeal is dismissed.", "LOSE"),
    ("the appeal is allowed to that extent. No orders as to costs.", "PARTIAL"),
])
def test_disposition_rules(text, expected):
    assert L.label_by_rules(text).initiator_outcome == expected


def test_order_window_ignores_the_recital():
    """The procedural recital must not be read as this court's holding."""
    text = ("The High Court dismissed the appeal filed by the plaintiff. " + "filler. " * 400 +
            "In the result, the appeal is allowed.")
    lab = L.label_by_rules(text)
    assert lab.initiator_outcome == "WIN"
    assert lab.order_span[0] > len(text) * 0.5


# --- quote grounding ------------------------------------------------------------
def test_quote_grounding_tolerates_pdf_hyphenation():
    src = "we therefore allow the appeal and dismiss the plain- tiff's suit."
    assert quote_is_grounded("We therefore allow the appeal and dismiss the plaintiff's suit.", src)


def test_quote_grounding_accepts_short_real_dispositions():
    assert quote_is_grounded("Appeal dismissed.", "...with costs. Appeal dismissed. Agent for")


@pytest.mark.parametrize("quote,src", [
    ("We accordingly allow the appeal.", "the appeal is dismissed with costs"),   # fabricated
    ("the appeal is ... allowed", "the appeal is hereby allowed"),                # elided
    ("allowed", "the appeal is allowed"),                                        # not a sentence
    ("The parties are mother and son.", "the parties are mother and son."),      # no disposition
])
def test_quote_grounding_refuses(quote, src):
    assert not quote_is_grounded(quote, src)


# --- segmentation ---------------------------------------------------------------
def test_segmentation_never_returns_one_giant_unit():
    """Extracted text often has no blank lines; a single unit made one reasoning phrase
    discard an entire judgment (57.7% of cases, before the sentence-window tier)."""
    text = " ".join(f"This is sentence number {i} of the judgment." for i in range(60))
    assert len(segment(text)) > 3


def test_units_carry_offsets_that_index_the_source():
    text = " ".join(f"Sentence {i} here." for i in range(40))
    for u in segment(text):
        assert text[u.start:u.end] == u.text


# --- masking --------------------------------------------------------------------
def test_mask_removes_the_operative_order():
    text = ("The plaintiff filed a suit for declaration of title. " + "The land is at Bhadsa. " * 200
            + " In the result, the appeal is allowed and the decree is set aside.")
    mc = mask_case({"doc_id": "t", "year": 2010, "text": text, "n_chars": len(text)})
    assert not OUTCOME_CUE.search(mc.masked_text)
    assert "appeal is allowed" not in mc.masked_text.lower()


def test_mask_separates_prior_court_disposition_without_deleting_it():
    text = ("The plaintiff filed a suit for possession. The High Court dismissed the suit. "
            + "The property lies in the village. " * 200)
    mc = mask_case({"doc_id": "t", "year": 2010, "text": text, "n_chars": len(text)})
    assert "High Court dismissed" in mc.prior_court_text
    assert "High Court dismissed" not in mc.masked_text


# --- splits ---------------------------------------------------------------------
def test_party_signature_groups_repeat_visits_of_one_dispute():
    a = party_signature("Bishna Mahato vs State Of West Bengal on 28 October 2005")
    b = party_signature("Bishna Mahato vs State Of West Bengal on 14 March 2011")
    assert a == b and a


def test_party_signature_does_not_merge_on_institutions_alone():
    a = party_signature("Ramaswami Iyer vs State Of Madras on 1 January 1960")
    b = party_signature("Gopal Krishnan vs State Of Madras on 1 January 1960")
    assert a != b


def _split_files():
    """Discover splits rather than hardcoding names: the temporal split is named after the
    boundary years it chose, which move when the label set grows."""
    return sorted(paths.SPLITS.glob("*.json"))


@pytest.mark.skipif(not list(paths.SPLITS.glob("*.json")), reason="splits not built yet")
def test_no_party_group_straddles_a_split():
    reg = {json.loads(l)["doc_id"]: json.loads(l)["title"] for l in open(paths.CASE_REGISTRY)}
    for f in _split_files():
        name = f.stem
        d = json.loads(f.read_text())
        seen = {}
        for part in ("train", "dev", "test"):
            for doc in d[part]:
                sig = party_signature(reg.get(doc, doc)) or doc
                assert seen.get(sig, part) == part, f"{name}: group {sig!r} straddles splits"
                seen[sig] = part
@pytest.mark.skipif(not list(paths.SPLITS.glob("temporal_*.json")), reason="splits not built")
def test_temporal_split_respects_time():
    from src.data.label_merge import load_final
    labs = {doc: r["year"] for doc, r in load_final().items()}
    for f in paths.SPLITS.glob("temporal_*.json"):
        d = json.loads(f.read_text())
        assert max(labs[x] for x in d["train"]) <= d["t1"], f
        assert min(labs[x] for x in d["test"]) > d["t2"], f
    assert max(labs[x] for x in d["train"]) <= d["t1"]
    assert min(labs[x] for x in d["test"]) > d["t2"]
