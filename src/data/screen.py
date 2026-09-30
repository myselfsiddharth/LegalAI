"""Screen the land-dispute set for what it actually contains (§4 / §5 data statistics).

`filtered_land_disputes.xlsx` was built with a keyword filter, and a keyword filter over
Indian judgments catches cases that merely HAPPEN on land. The clearest example found:
a 2005 criminal appeal over an assault during paddy-cutting, which mentions agricultural
land throughout and is a murder case.

We tag rather than delete. A criminal appeal about a boundary affray is genuinely a land
dispute in the ordinary sense and may carry usable possession facts, so the decision of what
to include belongs to each experiment, which reports the subset it used.

`primary_forum` is the court the appeal came from. §5.3 asks for a court-held-out split as
the generalisation control, but this corpus is Supreme Court only -- there is no second court
to hold out. Holding out by ORIGINATING High Court is the available substitute, so we recover
it here.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, asdict

from src import paths

CRIMINAL_CAPTION = re.compile(
    r"\bAppeal\s*\(\s*crl\s*\.?\s*\)|\bCriminal\s+Appeal\s+No|\bCrl\.?\s*A(?:ppeal)?\.?\s*No", re.I)
IPC_OFFENCE = re.compile(
    r"\b(?:section|s\.)\s*(?:302|304|307|323|324|325|326|376|395|396|397|398|399|411|420|427"
    r"|447|448|452|506)\b.{0,40}\b(?:IPC|Indian Penal Code|Penal Code)\b"
    r"|\bunder\s+section\s+\d+\s+of\s+the\s+Indian\s+Penal\s+Code", re.I)
CONVICTION = re.compile(
    r"\b(?:convicted|conviction|acquitted|acquittal|sentenced\s+to|rigorous\s+imprisonment"
    r"|life\s+imprisonment)\b", re.I)
PROPERTY_TERM = re.compile(
    r"\b(?:sale\s+deed|title|possession|partition|adverse\s+possession|specific\s+performance"
    r"|land\s+acquisition|tenancy|lease|mortgage|mutation|easement|7/12|khata|patta"
    r"|conveyance|allotment|redevelopment|benami|succession)\b", re.I)
TAX_TERM = re.compile(
    r"\b(?:income.tax|sales\s+tax|excise|customs\s+duty|assessment\s+year|assessee"
    r"|wealth\s+tax|agricultural\s+income)\b", re.I)

HIGH_COURT = re.compile(
    r"\bHigh\s+Court\s+(?:of\s+(?:Judicature\s+at\s+)?|at\s+|for\s+)"
    r"([A-Z][A-Za-z&.\- ]{2,40}?)(?=\s*(?:,|\.|\bin\b|\bat\b|\bdated\b|\bhas\b|\bhad\b|$))", re.I)
HIGH_COURT_ALT = re.compile(
    r"\b(Allahabad|Andhra\s+Pradesh|Bombay|Calcutta|Delhi|Gauhati|Gujarat|Himachal\s+Pradesh"
    r"|Jammu\s+(?:and|&)\s+Kashmir|Karnataka|Kerala|Madhya\s+Pradesh|Madras|Mysore|Nagpur"
    r"|Orissa|Patna|Punjab(?:\s+(?:and|&)\s+Haryana)?|Rajasthan|Sikkim|Travancore|Uttarakhand"
    r"|Chhattisgarh|Jharkhand|Manipur|Meghalaya|Tripura|Telangana)\s+High\s+Court", re.I)


@dataclass
class Screen:
    doc_id: str
    year: int
    domain: str            # property | criminal | criminal_mixed | tax_heavy
    is_property: bool
    n_property_terms: int
    n_conviction_terms: int
    primary_forum: str | None    # originating High Court, for the held-out split


def _forum(text: str) -> str | None:
    head = text[:20000]
    m = HIGH_COURT_ALT.search(head)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).title()
    m = HIGH_COURT.search(head)
    if m:
        name = re.sub(r"\s+", " ", m.group(1)).strip(" ,.").title()
        return name if 2 < len(name) < 40 else None
    return None


def screen_one(rec: dict) -> Screen:
    text = rec["text"]
    n_prop = len(PROPERTY_TERM.findall(text))
    n_conv = len(CONVICTION.findall(text))
    crim = bool(CRIMINAL_CAPTION.search(text[:4000])) or bool(IPC_OFFENCE.search(text))
    n_tax = len(TAX_TERM.findall(text))

    if crim and n_conv >= 5:
        domain = "criminal" if n_prop < 8 else "criminal_mixed"
    elif n_tax >= 10 and n_prop < 15:
        domain = "tax_heavy"
    else:
        domain = "property"

    return Screen(doc_id=rec["doc_id"], year=rec["year"], domain=domain,
                  is_property=(domain == "property"), n_property_terms=n_prop,
                  n_conviction_terms=n_conv, primary_forum=_forum(text))


def main() -> None:
    out = paths.INTERIM / "case_screen.jsonl"
    dom, forum = Counter(), Counter()
    n = 0
    with out.open("w") as f, paths.CASE_TEXT.open() as src:
        for line in src:
            s = screen_one(json.loads(line))
            f.write(json.dumps(asdict(s)) + "\n")
            dom[s.domain] += 1
            forum[s.primary_forum or "unknown"] += 1
            n += 1
    print(f"screen: {n} cases -> {out}")
    print("\n  domain:")
    for k, v in dom.most_common():
        print(f"    {k:15s} {v:5d}  {100*v/n:5.1f}%")
    print(f"\n  originating forum recovered: {n - forum['unknown']} "
          f"({100*(n-forum['unknown'])/n:.1f}%); top 12:")
    for k, v in forum.most_common(13):
        if k != "unknown":
            print(f"    {k:26s} {v:5d}  {100*v/n:5.1f}%")
    print(f"    {'(unknown)':26s} {forum['unknown']:5d}  {100*forum['unknown']/n:5.1f}%")


if __name__ == "__main__":
    main()
