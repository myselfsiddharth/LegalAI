"""Convert the refined ontology .docx into ontology/ontology_v1.yaml (§6.3).

We parse the document rather than retyping it, so the YAML stays traceable to its source
and a corrected .docx can be re-converted instead of hand-merged.

Document shape the parser relies on (verified against the actual file):
  Heading 1  -> numbered top-level section ("6. Claims")
  Heading 3  -> "Claim N: <name>"
  Normal     -> "Issue: ...", "Notes: ..." under a claim
  List Bullet 2 under "Elements:" -> one element, "Name: definition"
  List Bullet under "Facts:"/"Evidence:" -> one fact/evidence line
  Tables     -> two-column "Field / Type,Values,Purpose" schema definitions

Elements are the payload: §8.3 maps mined patterns onto them and §10.1's `E` feature group
scores them, so element names must be stable identifiers. We slugify each name and keep the
prose definition alongside.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import yaml

from src import paths

SRC_DOCX = paths.ROOT / "Docs" / "ontology_source" / "Land_Property_Dispute_India_Ontology_Refined.docx"
OUT_YAML = paths.ONTOLOGY / "ontology_v1.yaml"

CLAIM_RE = re.compile(r"^Claim\s+(\d+)\s*:\s*(.+)$")
LABELLED_RE = re.compile(r"^(Issue|Rule & Authority|Notes|Elements|Facts|Evidence)\s*:\s*(.*)$")


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^\w\s/-]", "", s).strip().lower()
    s = re.sub(r"[\s/-]+", "_", s)
    return s.strip("_")


def _iter_blocks(doc):
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    for child in doc.element.body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            p = Paragraph(child, doc)
            if p.text.strip():
                yield "p", p.style.name, p.text.strip()
        elif tag == "tbl":
            t = Table(child, doc)
            rows = [[c.text.strip() for c in r.cells] for r in t.rows]
            yield "tbl", "", rows


def parse(docx_path: Path) -> dict:
    import docx
    doc = docx.Document(docx_path)

    out: dict = {
        "version": 1,
        "source_document": docx_path.name,
        "jurisdiction": "India",
        "domain": "land and property disputes",
        "claims": [],
        "defenses": [],
        "remedy_tiers": [],
        "evidence_groups": [],
        "schemas": {},
        "patterns_instantiated": [],
        "validation_gates": [],
        "design_rules": [],
    }

    section = None          # current H1
    claim = None            # current claim dict
    sub = None              # 'Elements' | 'Facts' | 'Evidence' | None
    pending_tables: list[list[list[str]]] = []

    def close_claim():
        nonlocal claim
        if claim:
            out["claims"].append(claim)
            claim = None

    for kind, style, payload in _iter_blocks(doc):
        if kind == "tbl":
            pending_tables.append((section, payload))
            continue

        text = payload

        if style == "Heading 1":
            close_claim()
            section = text
            sub = None
            continue
        if style in ("Heading 2",):
            sub = None
            continue

        if style == "Heading 3":
            m = CLAIM_RE.match(text)
            if m:
                close_claim()
                num, name = int(m.group(1)), m.group(2).strip()
                claim = {"claim_id": f"claim_{num:02d}", "number": num, "name": name,
                         "slug": slug(name), "issue": None, "notes": None,
                         "elements": [], "facts": [], "evidence": []}
                sub = None
            continue

        m = LABELLED_RE.match(text)
        if m and claim is not None:
            label, rest = m.group(1), m.group(2).strip()
            if label == "Issue":
                claim["issue"] = rest
            elif label == "Notes":
                claim["notes"] = rest
            elif label in ("Elements", "Facts", "Evidence"):
                sub = label
            continue

        if claim is not None and style.startswith("List Bullet"):
            # "List Bullet 2" under Elements: is an element; plain "List Bullet" under
            # Elements: is the boilerplate reminder that every element carries burden data.
            if sub == "Elements":
                if style == "List Bullet 2":
                    name, _, definition = text.partition(":")
                    name = name.strip().rstrip(".")
                    claim["elements"].append({
                        "element_id": f"{claim['claim_id']}_e{len(claim['elements'])+1:02d}",
                        "name": name,
                        "slug": slug(name),
                        "definition": definition.strip() or None,
                        # filled by ontology_burden.py; see §8.3 / CHANGELOG
                        "burden_on": None,
                        "burden_standard": None,
                        "burden_shifts_when": None,
                    })
            elif sub == "Facts":
                claim["facts"].append(text)
            elif sub == "Evidence":
                claim["evidence"].append(text)
            continue

        if style.startswith("List Bullet") and section:
            if section.startswith("4."):
                out["patterns_instantiated"].append(text)
            elif section.startswith("10."):
                out["validation_gates"].append(text)
            elif section.startswith("0."):
                out["design_rules"].append(text)

    close_claim()

    # --- tables: schema definitions, defenses, remedies, evidence groups
    for sec, rows in pending_tables:
        if not rows or len(rows[0]) < 2:
            continue
        head = rows[0][0].lower()
        body = rows[1:]
        if "defense category" in head:
            for r in body:
                out["defenses"].append({"category": r[0], "slug": slug(r[0]), "detail": r[1]})
        elif "remedy tier" in head:
            for r in body:
                out["remedy_tiers"].append({"tier": r[0], "remedies": r[1]})
        elif "evidence group" in head:
            for r in body:
                out["evidence_groups"].append({"group": r[0], "slug": slug(r[0]), "types": r[1]})
        else:
            name = slug(sec or head)[:60] or slug(head)
            fields = {r[0]: r[1] for r in body if r[0]}
            if fields:
                key = name
                i = 2
                while key in out["schemas"]:
                    key = f"{name}_{i}"; i += 1
                out["schemas"][key] = fields
    return out


def main() -> None:
    if not SRC_DOCX.exists():
        raise SystemExit(f"missing source: {SRC_DOCX}")
    ont = parse(SRC_DOCX)
    paths.ONTOLOGY.mkdir(parents=True, exist_ok=True)
    OUT_YAML.write_text(yaml.safe_dump(ont, sort_keys=False, allow_unicode=True, width=100))

    n_el = sum(len(c["elements"]) for c in ont["claims"])
    print(f"ontology v1 -> {OUT_YAML}")
    print(f"  claims: {len(ont['claims'])}   elements: {n_el}   "
          f"defenses: {len(ont['defenses'])}   remedy tiers: {len(ont['remedy_tiers'])}")
    print(f"  schemas: {list(ont['schemas'])}")
    print(f"  patterns: {len(ont['patterns_instantiated'])}  "
          f"validation gates: {len(ont['validation_gates'])}")
    missing = [c["name"] for c in ont["claims"] if not c["elements"]]
    if missing:
        print(f"  WARNING claims with no elements parsed: {missing}")
    for c in ont["claims"]:
        print(f"    {c['claim_id']} {c['name'][:44]:44s} "
              f"el={len(c['elements'])} facts={len(c['facts'])} ev={len(c['evidence'])}")


if __name__ == "__main__":
    main()
