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
    # The client default (600s read timeout x 3 attempts) let one dead connection stall the
    # 50-case batch for 30+ minutes with no output; a grounding call normally takes ~2s.
    return OpenAI(base_url=base_url, api_key=api_key, timeout=90, max_retries=3)


def load_ontology() -> dict:
    return json.loads(ONTOLOGY_PATH.read_text())


def load_rules() -> dict:
    """(class, role, guard) -> concept. Guard "*" is the unguarded default (see content_guard)."""
    if not RULES_PATH.exists():
        sys.exit("No learned rules yet -- run `python3 scripts/ode_pipeline.py learn` first")
    rules = json.loads(RULES_PATH.read_text())
    return {(r["class"], r["role"], r.get("guard", "*")): r["concept"] for r in rules}


# Found via manual audit on Kedar Nath Yadav: "approved" is in the ontology as a
# TreatmentAction trigger for judicial endorsement ("X v. Y ... approved"), but the same
# word also means ordinary administrative sign-off ("approved by the Chief Minister").
# Grounding doesn't disambiguate by context, so it fired on "the Cabinet", "TML", "Rs. 90
# lakhs per annum" -- none of which are case citations. Rather than fix the grounding
# prompt (which needs more contrastive gold examples to do properly), filter cheaply here:
# an AuthorityCited candidate that doesn't look like a case name isn't a case citation,
# full stop, regardless of what triggered it.
#
# A second, different failure showed up at batch scale: when a headnote sentence lists
# several citations before a shared verb ("X v. Y (cite1), A v. B (cite2) relied on."),
# grounding sometimes drops the first party's name and the "v." connector, emitting just
# "Gilbert Pinto (I.L.R. 42 Mad. 654)" instead of "Krishna Shetti v. Gilbert Pinto (...)".
# That's still clearly a case reference (it has a reporter citation), just missing "v.",
# so the filter accepts either signal -- a v./vs. connector, OR a law-reporter citation.
#
# Two further bugs found auditing the first 50-case batch (all 1950s):
#  - Indian reporter citations usually sit OUTSIDE parentheses -- "Ayyappa Reddy, (1913)
#    I.L.R. 38 Mad. 738", "[1954] S.C.R. 177" -- so a parenthetical-only check rejected
#    ~25 genuine case citations as REJECTED_NOT_A_CASE. Search the whole string instead.
#  - The old pattern ran under IGNORECASE, which turned "[A-Z]{2,}" into "any two
#    letters" and accepted any parenthetical: "clause (iv)", "(Act XVIII of 1937)". Those
#    statute fragments then landed in UNVERIFIED and inflated the unverified count.
#    Reporter abbreviations are matched case-sensitively now.
CASE_CONNECTOR = re.compile(r"(?:\bv\.?|\bV\.|\b[Vv][Ss]\.?|\bversus)(?=\s|$)")
# Dotted capitals ("I.L.R.", "S.C.R.", "L.R. ... I.A.", "K.B.") or a known undotted modern
# reporter ("(2005) 3 SCC 123", "AIR 1950 SC 27"). The undotted list is explicit rather
# than "any capitals + digit", which would also accept "Section 302 IPC".
REPORTER = re.compile(r"(?:\b[A-Z]\.\s?){2,}|\b(?:SCC|SCR|AIR|SCALE|JT|ITR|MLJ|SCJ|KB|QB|AC|WLR)\b")


def looks_like_case_name(text: str) -> bool:
    if len(text) < 8:
        return False
    return bool(CASE_CONNECTOR.search(text)) or (
        bool(REPORTER.search(text)) and bool(re.search(r"\d", text)))


# A (class, role) pair alone is too coarse a key for a rule: RulingAction.Patient fires on
# "held that X" whether X is a holding, a statute, a case or the question being decided, so
# winner-take-all over that pair kept Holding and silently discarded every Issue and
# RuleCited gold example. The guard is a cheap, content-based read of the ROLE-FILLER, used
# as part of the rule key so one grounded pair can route to different concepts.
#
# Deliberately conservative: a bare "the Act" or "the Code" is not enough, because those
# phrases turn up in most land-dispute sentences. A section-style word must carry a number,
# or an Act must carry its year.
STATUTE_REF = re.compile(
    r"\b(?:section|sections|sec|art|arts|article|articles|rule|rules|order|clause|"
    r"schedule|paragraph|para|proviso)\.?\s*\(?\d"
    r"|\bAct\s*,?\s*(?:of\s*)?\d{4}\b"
    r"|\b[IVXLC]+\s+of\s+\d{4}\b", re.IGNORECASE)
# "whether ..." is the usual Indian framing, but LEARN showed it is not enough: the gold
# Issue "What are the consequences of non-deposit in Court ..." carries neither "whether"
# nor a "?", so it fell through to the statute guard and cost that rule its confidence.
# Sentence-initial wh-words and auxiliaries are added, along with the stock phrase Indian
# judgments use to announce an issue. Deliberately NOT included: initial is/are/was/were,
# which open too many ordinary holding fragments ("was in possession since 1952").
QUESTION_FRAME = re.compile(
    r"\bwhether\b"
    r"|\?\s*$"
    r"|^\s*(?:what|why|how|when|which|who|whom|whose|does|do|did|can|could|should|must)\b"
    r"|\b(?:point|points|question|questions|issue|issues)\s+(?:for|of)\s+"
    r"(?:determination|consideration|decision)\b", re.IGNORECASE)


# Mentioning a statute is not the same as citing one as the governing rule. Both gold
# RuleCited fillers put the provision in SUBJECT position ("that Section 238A ... would not
# extend", "the deletion of Section 29 did not have the effect"), while the Holding and
# Issue fillers that collided with them only reach a section late, inside the predicate
# ("a liability ... would be a financial debt ... under section 5(8)"). Approximate the
# subject/predicate split by the first finite verb: a bare reference with no verb at all
# ("Section 100 of the Code of Civil Procedure") is the purest RuleCited of them all.
# Fitted to four examples, so re-check this against any new gold data before trusting it.
PREDICATE_VERB = re.compile(
    r"\b(?:would|shall|will|is|are|was|were|does|do|did|has|have|had|may|might|must|can|"
    r"could|should|ought)\b", re.IGNORECASE)


def _statute_in_subject(filler: str) -> bool:
    ref = STATUTE_REF.search(filler)
    if not ref:
        return False
    verb = PREDICATE_VERB.search(filler)
    return verb is None or ref.start() < verb.start()


def content_guard(filler: str) -> str:
    """Which content shape a role-filler has: question, case, statute, or other.

    Order matters. Question framing wins over everything, because "whether section 9
    applies" is an Issue, not a RuleCited. A case name beats a statute reference, since a
    reported citation often carries both ("Krishna v. Pinto, (1913) I.L.R. 38 Mad. 738").
    """
    if not filler:
        return "other"
    if QUESTION_FRAME.search(filler):
        return "question"
    if looks_like_case_name(filler):
        return "case"
    if _statute_in_subject(filler):
        return "statute"
    return "other"


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
            guard = content_guard(filler)
            concept = (rule_lookup.get((fact_class, role_name, guard))
                       or rule_lookup.get((fact_class, role_name, "*")))
            if concept:
                emitted.append({
                    "concept": concept,
                    "filler": filler,
                    "via": f"{fact_class}.{role_name}[{guard}]",
                    "guard": guard,
                    "trigger": fact.get("trigger"),
                    "source_sentence": sentence,
                })
    return emitted
