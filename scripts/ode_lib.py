"""Shared grounding/rule-matching code, factored out of ode_pipeline.py so Phase III
(verification + IRAC) can reuse GROUND/EXTRACT without duplicating it."""

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ONTOLOGY_PATH = ROOT / "scripts" / "ontology" / "land_dispute_ontology.json"
RULES_PATH = ROOT / "scripts" / "ontology" / "learned_rules.json"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def get_client():
    load_dotenv()
    api_key = os.environ.get("VOYAGER_API_KEY")
    if not api_key:
        sys.exit("VOYAGER_API_KEY not set (export it, or put it in .env at the project root).")
    from openai import OpenAI
    base_url = os.environ.get("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


def load_ontology() -> dict:
    return json.loads(ONTOLOGY_PATH.read_text())


def load_rules() -> dict:
    if not RULES_PATH.exists():
        sys.exit("No learned rules yet -- run `python3 scripts/ode_pipeline.py learn` first")
    rules = json.loads(RULES_PATH.read_text())
    return {(r["class"], r["role"]): r["concept"] for r in rules}


def build_grounding_prompt(ontology: dict) -> str:
    leaf_classes = sorted({c for info in ontology["classes"].values() for c in info["children"]})
    term_items = [(k, v) for k, v in ontology["terminology"].items() if not k.startswith("_")]
    terms = "; ".join(f'"{k}" -> {v["isa"]}' for k, v in term_items)
    return f"""You are grounding a sentence from an Indian court judgment into a fixed ontology. \
Do not summarize or explain -- only identify which ontology classes are instantiated by which \
exact spans of the sentence, and what role-fillers attach to each.

Valid classes (use ONLY these leaf classes -- never a parent category like "Entity" or "Action"):
{', '.join(leaf_classes)}

Roles: {', '.join(ontology["roles"])}

Known trigger words (a hint, not exhaustive -- ground other words you recognize too):
{terms}

Worked example:
Sentence: "Held, that the suit was maintainable."
Output: [{{"trigger": "Held", "class": "RulingAction", "roles": {{"Patient": "the suit was maintainable"}}}}]
(The verb/action word is the trigger; what it acts on or concludes is the role-filler --
not the other way around.)

Another worked example:
Sentence: "Toleman v. Portbury (L.R. 6 Q.B. 245) relied on."
Output: [{{"trigger": "relied on", "class": "TreatmentAction", "roles": {{"Authority": "Toleman v. Portbury"}}}}]

Output ONLY a JSON array in this same shape. Use exact substrings from the sentence for \
"trigger" and role values. If nothing in the sentence matches the ontology, output []. \
No commentary, no markdown fences."""


def ground_sentence(client, model: str, sentence: str, ontology: dict) -> list[dict]:
    system_prompt = build_grounding_prompt(ontology)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sentence},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        facts = json.loads(raw)
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []


def extract_concepts(client, model: str, sentence: str, ontology: dict, rule_lookup: dict) -> list[dict]:
    """Ground a sentence and apply learned rules; returns typed objects with provenance."""
    facts = ground_sentence(client, model, sentence, ontology)
    emitted = []
    for fact in facts:
        fact_class = fact.get("class")
        if not fact_class or not isinstance(fact.get("roles"), dict):
            continue
        for role_name, filler in fact["roles"].items():
            if not isinstance(filler, str):
                continue
            concept = rule_lookup.get((fact_class, role_name))
            if concept:
                emitted.append({
                    "concept": concept,
                    "filler": filler,
                    "via": f"{fact_class}.{role_name}",
                    "trigger": fact.get("trigger"),
                    "source_sentence": sentence,
                })
    return emitted
