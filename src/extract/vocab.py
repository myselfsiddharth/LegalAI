"""§6.2 step 1: seed the canonical fact vocabulary from the ontology.

The vocabulary is what makes facts *itemizable* -- FP-Growth (§8) mines sets of labels, so two
judgments saying "he bought the land under a registered sale deed" and "a sale deed was
executed and registered in his favour" must reduce to the same atom or no pattern can ever
reach support.

Every entry is generated from a named field of `ontology_v1.yaml` and records that field as its
`source`, so a reviewer can see why a label exists and the §6.3 change log has something
concrete to diff. Nothing here is hand-invented: an invented category would be worse than an
invented citation, because nothing downstream would catch it.

This is a SEED, not the frozen vocabulary. §6.2 requires an iterative loop -- label a sample
with a `NEW:<proposal>` escape, cluster the proposals, review, merge -- until the NEW rate is
under 5%. `vocab_v1.yaml` is the input to that loop.

Polarity is carried as a property rather than baked into labels, but the DISTINCTION it encodes
is required: §10.4's reading of the British Nationality Act work insists that "not proved" and
"proved false" are different facts. `notice.not_shown` (nobody established service) is not
`notice.not_served` (service was established as absent), and a burden-shift rule that cannot
tell them apart is wrong.
"""
from __future__ import annotations

import re
from collections import Counter

import yaml

from src import paths

OUT = paths.ONTOLOGY / "vocab_v1.yaml"

# Category -> which ontology field seeds its subcategories.
# `schema` entries name (schema_key, field); `claim_elements` pulls element names for a claim.
SEEDS = {
    "Instrument": [("schema", "3_parties_subject_property_and_relationships_4",
                    "Governing Instrument")],
    "Relationship": [("schema", "3_parties_subject_property_and_relationships_4",
                      "Property Interest Relationship")],
    "RightClaimed": [("schema", "3_parties_subject_property_and_relationships_4",
                      "Property Rights Claimed")],
    "Statute": [("schema", "3_parties_subject_property_and_relationships_4",
                 "Governing Statutory Framework")],
    "Constitutional": [("schema", "3_parties_subject_property_and_relationships_4",
                        "Constitutional / Equitable Relationship")],
    "Possession": [("schema", "3_parties_subject_property_and_relationships",
                    "Possession Status")],
    "PropertyType": [("schema", "3_parties_subject_property_and_relationships_3",
                      "Property Type")],
    "Tenure": [("schema", "3_parties_subject_property_and_relationships_3", "Tenure Type")],
    "Encumbrance": [("schema", "3_parties_subject_property_and_relationships_3",
                     "Encumbrance Status")],
    "LandUse": [("schema", "3_parties_subject_property_and_relationships_3",
                 "Land Use Classification")],
    "PartyRole": [("schema", "3_parties_subject_property_and_relationships",
                   "Property Role"),
                  ("schema", "3_parties_subject_property_and_relationships_2",
                   "Property Role")],
    "Event": [("schema", "1_procedural_history_and_events", "Event.subtype")],
    "Evidence": [("evidence_groups",)],
    "Defense": [("defenses",)],
    # Element-derived categories: the facts that decide a claim are the facts that speak to its
    # elements, so the element inventory is a fact vocabulary as much as a proof checklist.
    "Element": [("claim_elements",)],
}

# Properties every canonical fact may carry. `polarity` and `proof_status` are separate on
# purpose -- see the module docstring on negation.
PROPERTIES = {
    "polarity": ["pos", "neg"],
    "proof_status": ["asserted", "admitted", "contested", "not_shown", "proved_false",
                     "inferred", "unknown"],
    "actor": ["claimant", "respondent", "authority", "third_party", "court_below", "unknown"],
    "asserted_by": ["plaintiff", "defendant", "court_narrative", "admitted", "unknown"],
    # Numeric properties are binned so they are itemizable (§6.2 step 5).
    "duration_years_bin": ["lt3", "3to12", "gt12", "gt30", "unknown"],
    "amount_bin": ["lt10k", "10k_1l", "1l_10l", "10l_1cr", "gt1cr", "unknown"],
    "delay_years_bin": ["lt1", "1to3", "3to12", "gt12", "unknown"],
}


def slug(s: str) -> str:
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^\w\s/&.-]", " ", s).strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[\s/.-]+", "_", s)
    return re.sub(r"_+", "_", s).strip("_")


def _split_enum(v: str) -> list[str]:
    """Ontology enum cells are '; '-separated, sometimes with a trailing sentence."""
    parts = re.split(r"[;\n]", v)
    out = []
    for p in parts:
        p = p.strip().rstrip(".")
        if not p or len(p) > 70:
            continue
        # a cell may end in prose ("... Purpose of the field")
        if p.lower().startswith(("string", "reference", "container", "date", "list")):
            continue
        out.append(p)
    return out


def build() -> dict:
    ont = yaml.safe_load(OUT.parent.joinpath("ontology_v1.yaml").read_text())
    vocab: dict = {"version": 1, "derived_from": ont["source_document"],
                   "note": "SEED vocabulary -- §6.2 iterates this to a frozen vN.",
                   "properties": PROPERTIES, "categories": {}}

    for cat, sources in SEEDS.items():
        subs: dict[str, dict] = {}
        for src in sources:
            if src[0] == "schema":
                _, key, field = src
                cell = ont["schemas"].get(key, {}).get(field)
                if not cell:
                    print(f"  WARNING no ontology field {key}.{field!r} for {cat}")
                    continue
                for v in _split_enum(cell):
                    subs.setdefault(slug(v), {"name": v, "source": f"schema:{key}.{field}"})
            elif src[0] == "evidence_groups":
                for g in ont["evidence_groups"]:
                    for v in re.split(r",", g["types"]):
                        v = v.strip().rstrip(".")
                        if 2 < len(v) < 50:
                            subs.setdefault(slug(v), {"name": v,
                                                      "source": f"evidence_groups:{g['slug']}"})
            elif src[0] == "defenses":
                for d in ont["defenses"]:
                    head = d["detail"].split("Rebuttal:")[0]
                    for v in re.split(r";", head):
                        v = v.strip().rstrip(".")
                        if 2 < len(v) < 60:
                            subs.setdefault(slug(v), {"name": v,
                                                      "source": f"defenses:{d['slug']}"})
            elif src[0] == "claim_elements":
                for c in ont["claims"]:
                    for e in c["elements"]:
                        subs.setdefault(e["slug"], {
                            "name": e["name"], "definition": e.get("definition"),
                            "source": f"claim_elements:{c['claim_id']}",
                            "claim_id": c["claim_id"], "element_id": e["element_id"]})
        vocab["categories"][cat] = {"n": len(subs), "subcategories": subs}

    # Claim-level fact templates: the ontology's per-claim `facts` lines are the questions an
    # extractor should be able to answer, so they are kept as guidance for the prompt rather
    # than as labels.
    vocab["claim_fact_templates"] = {
        c["claim_id"]: {"name": c["name"], "issue": c["issue"], "facts": c["facts"]}
        for c in ont["claims"]}
    return vocab


def main() -> None:
    v = build()
    OUT.write_text(yaml.safe_dump(v, sort_keys=False, allow_unicode=True, width=100))
    total = sum(c["n"] for c in v["categories"].values())
    print(f"seed vocabulary -> {OUT}")
    print(f"  {len(v['categories'])} categories, {total} subcategories, "
          f"{len(v['properties'])} property axes")
    for cat, c in sorted(v["categories"].items(), key=lambda x: -x[1]["n"]):
        ex = ", ".join(list(c["subcategories"])[:4])
        print(f"    {cat:15s} {c['n']:4d}  e.g. {ex[:78]}")
    prov = Counter(s["source"].split(":")[0]
                   for c in v["categories"].values() for s in c["subcategories"].values())
    print(f"  provenance: {dict(prov)}")


if __name__ == "__main__":
    main()
