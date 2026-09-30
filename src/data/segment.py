"""Split a judgment into addressable units (§6.1 `source_span`, §5.2 masking).

There are no section headers to lean on. Measured over 5,462 judgments: an `ORDER` header
appears in 3.7%, a `FACTS` header in 29.7%. What there IS, is numbered paragraphs -- 92.7%
of post-2000 judgments have them, median 32 per judgment, 85% have at least five.

So the unit is the numbered paragraph where one exists, and a blank-line block otherwise.
Every unit carries its char offsets, because a masking decision, a fact and a trace entry all
need to name the exact region of text they came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PARA_MARKER = re.compile(r"^[ \t]*(\d{1,3})\.[ \t]+(?=[\"'A-Z(])", re.M)
# A blank line, or a run of spaces acting as one in extracted text.
BLOCK_SPLIT = re.compile(r"\n\s*\n")
SENT_SPLIT = re.compile(r"(?<=[.;:!?])\s+(?=[\"'(A-Z0-9])")


@dataclass
class Unit:
    idx: int
    start: int
    end: int
    text: str
    para_num: int | None = None     # the judgment's own paragraph number, when numbered
    kind: str = "para"              # 'para' | 'block'

    def sentences(self) -> list[tuple[int, int, str]]:
        """(abs_start, abs_end, text) per sentence, offsets relative to the whole judgment."""
        out, pos = [], 0
        for s in SENT_SPLIT.split(self.text):
            if not s.strip():
                pos += len(s) + 1
                continue
            i = self.text.find(s, pos)
            if i < 0:
                i = pos
            out.append((self.start + i, self.start + i + len(s), s))
            pos = i + len(s)
        return out


def segment(text: str, min_numbered: int = 5) -> list[Unit]:
    marks = [(int(m.group(1)), m.start()) for m in PARA_MARKER.finditer(text)]
    # Require the numbering to be mostly ascending; a table of contents or a list of
    # statutory sub-clauses produces markers that jump around and must not drive the split.
    ascending = sum(1 for a, b in zip(marks, marks[1:]) if b[0] > a[0])
    if len(marks) >= min_numbered and ascending >= 0.7 * max(1, len(marks) - 1):
        units = []
        for i, (num, start) in enumerate(marks):
            end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
            if start > 0 and i == 0:
                units.append(Unit(0, 0, start, text[:start], None, "block"))   # preamble
            units.append(Unit(len(units), start, end, text[start:end], num, "para"))
        return units

    units, pos = [], 0
    for blk in BLOCK_SPLIT.split(text):
        i = text.find(blk, pos)
        if i < 0:
            i = pos
        if blk.strip():
            units.append(Unit(len(units), i, i + len(blk), blk, None, "block"))
        pos = i + len(blk)
    if len(units) > 3:
        return units

    # Third tier: sentence windows.
    #
    # Extracted PDF text frequently contains no blank lines at all -- a newline lands at
    # every visual line break instead -- so BLOCK_SPLIT returns the whole judgment as one
    # unit. That is not a harmless coarseness: a single unit takes a single role, so one
    # reasoning phrase anywhere in the judgment discards the entire document. Measured
    # before this tier existed, 57.7% of cases segmented to one unit and 57% of cases came
    # out of masking with under 500 usable characters.
    #
    # Grouping sentences into fixed windows gives uniform granularity with no reliance on
    # layout. The window is a classification context, not a claim about the author's
    # paragraphs, so masking still decides sentence by sentence within it.
    return sentence_windows(text)


def sentence_windows(text: str, per_window: int = 4) -> list[Unit]:
    sents, pos = [], 0
    for s in SENT_SPLIT.split(text):
        if not s.strip():
            pos += len(s) + 1
            continue
        i = text.find(s, pos)
        if i < 0:
            i = pos
        sents.append((i, i + len(s)))
        pos = i + len(s)
    units = []
    for k in range(0, len(sents), per_window):
        grp = sents[k:k + per_window]
        lo, hi = grp[0][0], grp[-1][1]
        units.append(Unit(len(units), lo, hi, text[lo:hi], None, "sentwin"))
    return units
