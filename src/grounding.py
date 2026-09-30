"""Verbatim-quote verification, shared by every module that takes a claim from a model.

The rule this enforces: **a model's quote is the only evidence it offers, so an unverifiable
quote carries no claim.** It is what separates "the judgment says this" from "the model says
the judgment says this", and it is cheap -- a substring test.

Two extraction artifacts have to be absorbed, or the check rejects correct quotes. Both were
measured on this corpus, not guessed:

  * **Whitespace.** PDF extraction breaks lines mid-sentence, so a correct quote differs from
    the source in whitespace alone.
  * **Hyphenation.** Judgments before roughly 1980 are typeset with hyphenated line wraps that
    survive extraction as "execut- ing", "plain- tiff's", "sanc- tioned". A model quoting the
    sentence silently repairs them -- the correct reading of the page, but not a string match.
    On a 60-case sample this single artifact was rejecting 9 of 25 otherwise-correct quotes.

Repair is applied to BOTH sides, so it can never manufacture a match the words do not support.

What is NOT tolerated, because each would let a reconstruction pass as a quotation:
  * an ellipsis ("..." / "…") -- the model elided text and is presenting a summary;
  * a quote too short to identify anything;
  * a quote that does not appear at all.
"""
from __future__ import annotations

import re

MIN_QUOTE_CHARS = 12
_WS = re.compile(r"\s+")
_HYPHEN_WRAP = re.compile(r"(\w)[-­]\s+(\w)")
_ELLIPSIS = ("...", "…", ". . .")


def normalize(s: str) -> str:
    s = _WS.sub(" ", s or "").strip().lower()
    return _HYPHEN_WRAP.sub(r"\1\2", s)


def has_ellipsis(quote: str) -> bool:
    return any(e in (quote or "") for e in _ELLIPSIS)


def is_grounded(quote: str, source: str, min_chars: int = MIN_QUOTE_CHARS) -> bool:
    q = normalize(quote)
    if len(q) < min_chars or has_ellipsis(quote):
        return False
    return q in normalize(source)


def locate(quote: str, source: str) -> tuple[int, int] | None:
    """Char offsets of `quote` in `source`, or None.

    Normalisation changes string lengths, so offsets cannot be read off the normalised text.
    We find the match in normalised space, then walk the original forward counting only the
    characters that survive normalisation, which maps a normalised index back to a real one.
    """
    q = normalize(quote)
    if len(q) < MIN_QUOTE_CHARS or has_ellipsis(quote):
        return None

    keep: list[int] = []          # keep[i] = index in `source` of normalised char i
    out: list[str] = []
    i, n = 0, len(source)
    while i < n:
        ch = source[i]
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                keep.append(i)
            i += 1
            continue
        # hyphen-wrap: "t- i" collapses, dropping the hyphen and the following whitespace
        if ch in "-­" and i + 1 < n and source[i + 1].isspace():
            j = i + 1
            while j < n and source[j].isspace():
                j += 1
            if j < n and source[j].isalnum() and out and out[-1].isalnum():
                i = j
                continue
        out.append(ch.lower())
        keep.append(i)
        i += 1

    norm_src = "".join(out)
    pos = norm_src.find(q)
    if pos < 0:
        return None
    start = keep[pos]
    end_idx = min(pos + len(q) - 1, len(keep) - 1)
    return (start, keep[end_idx] + 1)
