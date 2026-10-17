"""Stage 5 (§9.1): normalise statute citations to canonical `CODE:provision` form.

§9.1 asks for citations reduced to forms like `TPA:53A`, `SRA:16(c)`, `LimitationAct:Art65`,
`Const:Art300A`, so that the same provision cited three different ways becomes one prediction
target. Surveying the corpus shows why this needs a real registry rather than a regex:

  * **Act names arrive truncated.** The most frequent "Act" in 1,200 sampled judgments is
    `Property Act` (644 mentions) — that is *Transfer of Property Act* with its head cut off by a
    line break or a preceding "the". `India Act` (516) is *Government of India Act*. `Abolition
    Act` is one of a dozen state zamindari-abolition statutes.
  * **Generic back-references are everywhere.** `the Act` (311), `the said Act`, `the Amending
    Act`, `the Central Act`. These are resolvable only from context, by looking back to the last
    Act actually named.
  * **The same statute has several real names across eras.** The Limitation Act of 1908 and of
    1963 renumber the same subject matter; `Code of Civil Procedure` appears as `C.P.C.`, `CPC`,
    `Civil Procedure Code`.

So: a registry of canonical codes with alias patterns, matched longest-first, plus a backward
context resolution for generic references. Everything unresolved is counted, not silently dropped —
an unresolved citation is a known gap, whereas a mis-resolved one corrupts a prediction target.

Provisions keep their sub-clauses (`16(c)`, `299(2)`) because a sub-clause is usually the operative
distinction, and §9.2 predicts provisions rather than Acts.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# --- canonical registry -------------------------------------------------------------------
# code -> (canonical name, alias regexes). Aliases are matched case-insensitively and the
# LONGEST match wins, so "Transfer of Property Act" beats the bare "Property Act" fallback.
# Seeded from the ontology's §3.4 "Governing Statutory Framework" list and extended with the
# statutes the corpus actually cites most.
REGISTRY: dict[str, tuple[str, list[str]]] = {
    "TPA":      ("Transfer of Property Act", [r"transfer\s+of\s+property\s+act", r"\bT\.?P\.?\s?Act\b", r"\bTPA\b"]),
    "LA":       ("Land Acquisition Act", [r"land\s+acquisition\s+act", r"\bL\.?A\.?\s?Act\b"]),
    "RFCTLARR": ("Right to Fair Compensation and Transparency in Land Acquisition Act 2013",
                 [r"right\s+to\s+fair\s+compensation[\w\s,]*act", r"\bRFCTLARR\b", r"\b2013\s+Act\b"]),
    "LIM":      ("Limitation Act", [r"limitation\s+act", r"indian\s+limitation\s+act"]),
    "SRA":      ("Specific Relief Act", [r"specific\s+relief\s+act", r"\bS\.?R\.?\s?Act\b"]),
    "CPC":      ("Code of Civil Procedure", [r"code\s+of\s+civil\s+procedure", r"civil\s+procedure\s+code", r"\bC\.?P\.?C\.?\b"]),
    "CRPC":     ("Code of Criminal Procedure", [r"code\s+of\s+criminal\s+procedure", r"criminal\s+procedure\s+code", r"\bCr\.?\s?P\.?C\.?\b"]),
    "EA":       ("Indian Evidence Act", [r"indian\s+evidence\s+act", r"evidence\s+act"]),
    "REG":      ("Registration Act", [r"indian\s+registration\s+act", r"registration\s+act"]),
    "HSA":      ("Hindu Succession Act", [r"hindu\s+succession\s+act"]),
    "ISA":      ("Indian Succession Act", [r"indian\s+succession\s+act"]),
    "HMA":      ("Hindu Marriage Act", [r"hindu\s+marriage\s+act"]),
    "HAMA":     ("Hindu Adoptions and Maintenance Act", [r"hindu\s+adoptions?\s+and\s+maintenance\s+act"]),
    "HMGA":     ("Hindu Minority and Guardianship Act", [r"hindu\s+minority\s+and\s+guardianship\s+act"]),
    # A bare "Constitution" in an Indian judgment is always the Constitution of India.
    "CONST":    ("Constitution of India", [r"constitution\s+of\s+india", r"\bconstitution\b"]),
    "GOI":      ("Government of India Act", [r"government\s+of\s+india\s+act", r"\bindia\s+act\b"]),
    "IPC":      ("Indian Penal Code", [r"indian\s+penal\s+code", r"penal\s+code", r"\bI\.?P\.?C\.?\b"]),
    "BENAMI":   ("Benami Transactions (Prohibition) Act", [r"benami\s+transactions?[\w\s()]*act", r"benami\s+act"]),
    "ITA":      ("Income-tax Act", [r"income[\s\-]?tax\s+act"]),
    "ESMT":     ("Indian Easements Act", [r"indian\s+easements?\s+act", r"easements?\s+act"]),
    "PART":     ("Partition Act", [r"partition\s+act"]),
    "CONTRACT": ("Indian Contract Act", [r"indian\s+contract\s+act", r"contract\s+act"]),
    "TRUSTS":   ("Indian Trusts Act", [r"indian\s+trusts?\s+act", r"trusts?\s+act"]),
    "STAMP":    ("Indian Stamp Act", [r"indian\s+stamp\s+act", r"stamp\s+act"]),
    "ARB":      ("Arbitration Act", [r"arbitration\s+(?:and\s+conciliation\s+)?act"]),
    "COOP":     ("Co-operative Societies Act", [r"co[\s\-]?operative\s+societies\s+act"]),
    "RENT":     ("Rent Control Act", [r"rent\s+control\s+act", r"rents?\s+act", r"rent\s+restriction\s+act"]),
    "TENANCY":  ("Tenancy Act", [r"tenancy\s+(?:and\s+agricultural\s+lands\s+)?act"]),
    "CEILING":  ("Land Ceiling Act", [r"ceiling\s+(?:on\s+land\s+holdings?\s+)?act", r"land\s+ceiling\s+act"]),
    "ABOLITION": ("Zamindari / Estates Abolition Act",
                  [r"(?:zamindari|estates?|inams?|jagirs?)\s+abolition\s+act", r"abolition\s+act"]),
    "REVENUE":  ("Land Revenue Code", [r"land\s+revenue\s+code", r"land\s+revenue\s+act"]),
    "SLUM":     ("Slum Areas Act", [r"slum\s+(?:areas?|improvement|clearance)[\w\s]*act"]),
    "TOWNPLAN": ("Town Planning Act", [r"town\s+planning\s+act", r"regional\s+and\s+town\s+planning\s+act"]),
    "MUNI":     ("Municipal Act", [r"municipal\s+(?:corporations?\s+)?act", r"municipalities\s+act"]),
    "WAKF":     ("Wakf Act", [r"wakf\s+act", r"waqf\s+act"]),
    "RELIGIOUS": ("Religious / Charitable Endowments Act",
                  [r"(?:hindu\s+)?religious\s+(?:and\s+charitable\s+)?endowments?\s+act",
                   r"charitable\s+endowments?\s+act"]),
    "GENCL":    ("General Clauses Act", [r"general\s+clauses\s+act"]),
    "SARFAESI": ("SARFAESI Act", [r"securitisation[\w\s]*act", r"\bSARFAESI\b"]),
    "RERA":     ("Real Estate (Regulation and Development) Act", [r"real\s+estate\s*\([^)]*\)\s*act", r"\bRERA\b"]),
    "IBC":      ("Insolvency and Bankruptcy Code", [r"insolvency\s+and\s+bankruptcy\s+code", r"\bIBC\b"]),
}

# Longest alias first, so a specific name is never shadowed by a shorter fallback.
_ALIASES: list[tuple[str, re.Pattern]] = sorted(
    ((code, re.compile(a, re.I)) for code, (_n, al) in REGISTRY.items() for a in al),
    key=lambda x: -len(x[1].pattern))

# --- truncation fallbacks ------------------------------------------------------------------
# Act names arrive clipped by line breaks: `Property Act` for *Transfer of Property Act* is the
# single most frequent "Act" in the corpus (644 mentions in 1,200 judgments). These are tried ONLY
# after every full-name alias has failed, and resolve with `resolved_by="truncated"` so a guess is
# never reported as a full-name match -- the distinction matters when a §9.2 target turns out wrong.
TRUNCATED: list[tuple[str, re.Pattern]] = [
    ("TPA",       re.compile(r"\bproperty\s+act\b", re.I)),
    ("GOI",       re.compile(r"\bindia\s+act\b", re.I)),
    ("ABOLITION", re.compile(r"\babolition\s+act\b", re.I)),
    ("CPC",       re.compile(r"\bprocedure\s+code\b", re.I)),
    ("LA",        re.compile(r"\bacquisition\s+act\b", re.I)),
    ("RENT",      re.compile(r"\bcontrol\s+act\b", re.I)),
    ("COOP",      re.compile(r"\bsocieties\s+act\b", re.I)),
    ("REVENUE",   re.compile(r"\brevenue\s+code\b", re.I)),
]

# Generic back-references: resolvable only from context.
GENERIC_ACT = re.compile(
    r"\bthe\s+(?:said\s+|aforesaid\s+|present\s+|amending\s+|principal\s+|parent\s+|new\s+|old\s+)?Act\b", re.I)

# The same test applied to the CAPTURED Act span, where the leading "the" has already been
# consumed by PROVISION. Getting this wrong silently reclassified "of the said Act" as a named
# statute the registry did not know, so legitimate back-references stopped resolving.
GENERIC_ACT_SPAN = re.compile(
    r"^(?:said|aforesaid|present|amending|principal|parent|new|old|central|local|state|"
    r"impugned|relevant|above|aforementioned)?\s*(?:Act|Code)$", re.I)

# "Section 53A of the Transfer of Property Act", "Art. 300A", "Order 21 Rule 97 CPC"
PROVISION = re.compile(
    r"\b(?P<kind>[Ss]ections?|[Ss]ecs?\.|[Ss]\.|[Aa]rticles?|[Aa]rts?\.|[Oo]rders?|[Rr]ules?|"
    r"[Cc]lauses?|[Ss]chedules?)\s*"
    r"(?P<num>\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*)"
    r"(?P<tail>(?:\s*(?:and|,|to)\s*\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*)*)"
    # The Act span is captured generously -- up to 70 characters that do not cross a sentence
    # end, ending at the first Act-like keyword -- and `resolve_act` then alias-matches INSIDE it.
    # Anchoring the keyword in this regex instead was wrong: the lazy word repeat swallowed the
    # space before "Act", so "Section 53A of the Transfer of Property Act" captured nothing.
    r"(?:\s*(?:of|under|in)\s+(?:the\s+)?"
    r"(?P<act>[^.;:\n]{0,70}?(?:Act|Code|Constitution|Regulations?|Rules))\b)?")

ART_ONLY = re.compile(r"\b[Aa]rt(?:icle)?s?\.?\s*(\d+[A-Za-z]?(?:\(\d+\))*)")

# "Order XXI Rule 90", "Order 21, Rule 97", "O. 39 R. 1" -- CPC orders are usually ROMAN, which
# PROVISION's digit-only number cannot see. Before this, "Order XXI Rule 90" matched only "Rule 90"
# and became CPC:90, colliding with Section 90; and "Order 21 Rule 97" became CPC:21 + CPC:97,
# colliding with Sections 21 and 97. Orders and rules now carry their own prefix (O21, O21R97).
ROMAN = r"(?:[IVXLC]+)"
ORDER_RULE = re.compile(
    rf"\b(?:[Oo]rder|O\.)\s*(?P<o>{ROMAN}|\d+)\b(?:\s*,?\s*(?:[Rr]ules?|[Rr]r?\.)\s*(?P<r>\d+[A-Za-z]?))?")
_RV = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


def _roman(s: str) -> int:
    if s.isdigit():
        return int(s)
    total, prev = 0, 0
    for ch in reversed(s.upper()):
        v = _RV.get(ch, 0)
        total, prev = (total - v, prev) if v < prev else (total + v, v)
    return total


@dataclass
class StatuteCite:
    code: str                 # canonical registry code, or 'UNRESOLVED'
    provision: str            # e.g. '53A', '16(c)', '300A'
    label: str                # 'TPA:53A' -- the §9.2 prediction target
    kind: str                 # section | article | order | rule | clause | schedule
    raw: str
    act_text: str | None = None
    resolved_by: str = "named"   # named | context | kind_default | unresolved


def _norm_provision(num: str) -> str:
    return re.sub(r"\s+", "", num)


def _kind_of(k: str) -> str:
    k = k.lower().rstrip(".")
    if k.startswith(("sec", "s")):
        return "section"
    if k.startswith(("art",)):
        return "article"
    if k.startswith("order"):
        return "order"
    if k.startswith("rule"):
        return "rule"
    if k.startswith("clause"):
        return "clause"
    return "schedule"


def resolve_act(act_text: str | None) -> tuple[str | None, str]:
    """Map a raw Act string to a canonical code. Longest alias wins."""
    if not act_text:
        return None, "unresolved"
    for code, rx in _ALIASES:
        if rx.search(act_text):
            return code, "named"
    for code, rx in TRUNCATED:
        if rx.search(act_text):
            return code, "truncated"
    return None, "unresolved"


def extract(text: str, context_window: int = 4000) -> list[StatuteCite]:
    """All statute citations in `text`, normalised.

    An Act named earlier in the judgment resolves later generic references ("the said Act"), which
    is how Indian judgments are actually written: the statute is named once in full and referred to
    generically thereafter. We track the most recent named Act and only apply it within
    `context_window` characters, so a reference does not inherit an Act from pages away.
    """
    out: list[StatuteCite] = []
    last_code: str | None = None
    last_pos = -10 ** 9

    # pre-scan for named Acts so a generic reference can look backwards
    named: list[tuple[int, str]] = []
    for code, rx in _ALIASES:
        for m in rx.finditer(text):
            named.append((m.start(), code))
    named.sort()

    def act_before(pos: int) -> str | None:
        best = None
        for p, code in named:
            if p > pos:
                break
            if pos - p <= context_window:
                best = code
        return best

    # CPC orders and rules first, then mask them so PROVISION cannot re-read their numbers.
    order_ends: list[tuple[int, int]] = []       # (end position, order number)
    for m in ORDER_RULE.finditer(text):
        o = _roman(m.group("o"))
        if not 1 <= o <= 51:                 # the CPC has 51 Orders; anything else is not one
            continue
        order_ends.append((m.end(), o))
        tail = text[m.end(): m.end() + 80]
        act = re.match(r"\s*(?:of|under|in)\s+(?:the\s+)?([^.;:\n]{0,70}?(?:Act|Code))\b", tail)
        code, how = resolve_act(act.group(1)) if act else (None, "unresolved")
        if code is None:
            code, how = "CPC", ("named" if act else "kind_default")
        rules = [m.group("r")] if m.group("r") else [None]
        more = re.match(r"(?:\s*(?:,|and|&)\s*(\d+[A-Za-z]?))+", tail)
        if m.group("r") and more:
            rules += re.findall(r"\d+[A-Za-z]?", more.group(0))
        for rr in rules:
            prov = f"O{o}" + (f"R{rr}" if rr else "")
            out.append(StatuteCite(code=code, provision=prov, label=f"{code}:{prov}",
                                   kind="order", raw=m.group(0)[:120],
                                   act_text=act.group(1) if act else None, resolved_by=how))
    masked = ORDER_RULE.sub(lambda mm: " " * len(mm.group(0))
                            if 1 <= _roman(mm.group("o")) <= 51 else mm.group(0), text)

    def named_just_before(pos: int) -> str | None:
        """An Act named immediately BEFORE the provision -- "Limitation Act, 1963, Article 65",
        "Transfer of Property Act, Section 53A". Only the last 90 characters, so this cannot reach
        back across a sentence the way context resolution does."""
        seg = text[max(0, pos - 90): pos]
        seg = seg[max(seg.rfind("."), seg.rfind(";")) + 1:]
        best = None
        for code, rx in _ALIASES:
            for mm in rx.finditer(seg):
                if best is None or mm.end() > best[0]:
                    best = (mm.end(), code)
        return best[1] if best else None

    for m in PROVISION.finditer(masked):
        kind = _kind_of(m.group("kind"))
        if kind in ("order",):
            continue                          # handled above
        act0 = m.group("act")
        if kind == "rule" and (not act0 or GENERIC_ACT_SPAN.match(act0.strip())
                               or resolve_act(act0)[0] == "CPC"):
            # "Order XXII Rule 3 and Rule 4": a rule shortly after an Order belongs to that Order
            near = [o for e, o in order_ends if 0 <= m.start() - e <= 60]
            if near:
                nums = [m.group("num")] + re.findall(r"\d+[A-Za-z]?", m.group("tail") or "")
                for num in nums:
                    prov = f"O{near[-1]}R{_norm_provision(num)}"
                    out.append(StatuteCite(code="CPC", provision=prov, label=f"CPC:{prov}",
                                           kind="order", raw=m.group(0)[:120], resolved_by="context"))
                continue
        act_text = m.group("act")
        code, how = resolve_act(act_text)
        if code is None and not act_text:
            nb = named_just_before(m.start())
            if nb and not (kind == "section" and nb == "CONST"):
                code, how = nb, "named_before"

        if code is None:
            # An explicitly NAMED Act that the registry does not know stays UNRESOLVED. It must
            # never fall through to context resolution, which would reassign it to whatever Act
            # happens to be mentioned nearby -- observed producing `CONST:6` for "section 6 of the
            # Constituent Assembly Act", and `CONST:3` for the Extra-Provincial Jurisdiction Act.
            # A citation attributed to the wrong statute is worse than one left unresolved,
            # because §9.2 would train on it as a target.
            named_but_unknown = bool(act_text) and not GENERIC_ACT_SPAN.match(act_text.strip())
            if named_but_unknown:
                pass                                    # stays UNRESOLVED
            elif kind == "article":
                code, how = "CONST", "kind_default"     # an Article is the Constitution
            elif kind == "rule":
                # A bare "Rule N" is as often a rule of some statutory scheme ("Rule 3 will have to
                # be read in the light of...", a tax schedule) as a CPC rule; with no Order and no
                # named Code it is left UNRESOLVED rather than guessed into CPC.
                pass
            else:
                # No Act named at all, or a generic back-reference ("the said Act"): look back.
                # A SECTION is never attributed to the Constitution, which is cited by Article.
                ctx = act_before(m.start())
                if ctx and not (kind == "section" and ctx == "CONST"):
                    code, how = ctx, "context"

        nums = [m.group("num")] + re.findall(r"\d+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*",
                                             m.group("tail") or "")
        for num in nums:
            prov = _norm_provision(num)
            c = code or "UNRESOLVED"
            pre = {"article": "Art", "rule": "R"}.get(kind, "")
            out.append(StatuteCite(
                code=c, provision=prov, label=f"{c}:{pre}{prov}", kind=kind,
                raw=m.group(0)[:120], act_text=act_text, resolved_by=how if code else "unresolved"))
        if code:
            last_code, last_pos = code, m.start()
    return out


def coverage(cites: list[StatuteCite]) -> dict:
    by = Counter(c.resolved_by for c in cites)
    n = len(cites) or 1
    return {"n": len(cites),
            "resolved": sum(v for k, v in by.items() if k != "unresolved"),
            "truncated_guesses": by["truncated"],
            "resolved_frac": round(1 - by["unresolved"] / n, 4),
            "by_method": dict(by),
            "distinct_labels": len({c.label for c in cites if c.code != "UNRESOLVED"})}
