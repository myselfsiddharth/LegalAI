"""Classify authority-treatment (sign) for post-2000 judgments via LLM grounding.

Pre-2000 judgments carry an explicit HEADNOTE section that states, in semi-structured
text, how the citing court treated each authority ("... relied on.", "... distinguished.").
extract_headnote_signs.py pulls that out with regexes and needs no LLM. That convention
disappears entirely after the 1990s (see Data/processed/ -- 0 of 190 sampled 2000-2025
judgments have a HEADNOTE section), so for modern cases the same information has to be
read out of free-form reasoning text instead, e.g.:

  "We are of the view that the said judgment would not even remotely be applicable
   to the facts of the present case."  -> distinguishes the cited case, but uses none
   of the fixed verbs a regex could key on.

This is exactly the "LLM grounds text into a small ontology" step from the ODE method
(see scripts/ontology/land_dispute_ontology.json, TreatmentAction) -- the LLM's only
job is to say which of the ontology's known treatment classes (or "unclear") applies to
a given citation mention in context; it does not free-generate an explanation.

Requires an OpenAI-spec-compatible endpoint. Set via environment variables:
  VOYAGER_API_KEY   -- required
  VOYAGER_BASE_URL  -- defaults to ASU Voyager: https://openai.rc.asu.edu/v1
  VOYAGER_MODEL     -- defaults to "llama4-scout-17b" -- check your available models list
                       in the Voyager portal and override if that name isn't offered.

Usage:
  python3 scripts/classify_modern_signs.py --limit 20      # smoke test on 20 mentions
  python3 scripts/classify_modern_signs.py                 # full run over Data/raw_pdfs 2000+
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = ROOT / "Data" / "raw_pdfs"
CITATIONS_PATH = ROOT / "Data" / "processed" / "citations_classified.jsonl"
OUT_PATH = ROOT / "Data" / "processed" / "modern_signed_citations.jsonl"

SYSTEM_PROMPT = """You are grounding legal text into a fixed ontology. You will be given \
a short excerpt from an Indian court judgment and the name of a case cited somewhere in \
that excerpt. Classify how the citing court treated the cited case, using ONLY one of \
these labels:

- "relied_on": the court relied on, followed, or applied the cited case's reasoning
- "distinguished": the court found the cited case's facts/reasoning inapplicable here
- "overruled_or_disapproved": the court rejected or reversed the cited case's reasoning
- "referred_to": the cited case is mentioned/summarized without a clear stance
- "unclear": you cannot tell from this excerpt

Respond with ONLY a JSON object: {"label": "<one of the above>", "confidence": <0-1 float>}
No other text."""

POLARITY_MAP = {
    "relied_on": "positive",
    "distinguished": "negative",
    "overruled_or_disapproved": "negative",
    "referred_to": "neutral",
    "unclear": "neutral",
}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (no extra dependency): sets os.environ from KEY=VALUE lines,
    without overriding a variable that's already set in the real environment."""
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
        sys.exit(
            "VOYAGER_API_KEY is not set.\n"
            "  Either export it in this shell: export VOYAGER_API_KEY=<key>\n"
            "  Or put it in a .env file at the project root: VOYAGER_API_KEY=<key>\n"
            "  (get one at https://voyager.rc.asu.edu -> LLM Access -> Create Key)"
        )
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("Missing dependency: pip install openai")
    base_url = os.environ.get("VOYAGER_BASE_URL", "https://openai.rc.asu.edu/v1")
    return OpenAI(base_url=base_url, api_key=api_key)


# Extremely common leading words in Indian case captions ("State of Punjab v. ...",
# "Union of India v. ...", "Commissioner of ... v. ...") are useless as a search anchor:
# they match constantly throughout any judgment regardless of which specific case is
# meant. Found via manual audit -- "State of Punjab v. V.K. Khanna" anchored on "State"
# and pulled an excerpt about an unrelated advocate named Khanna. Skip these and use the
# next capitalized token instead.
GENERIC_CAPTION_WORDS = {
    "state", "union", "government", "commissioner", "board", "corporation",
    "the", "in", "re", "director", "collector", "secretary", "chief",
}


def find_citation_mentions(judgment_text: str, cited_case_name: str, window: int = 400):
    """Find short excerpts around plausible mentions of a cited case's surname pair.

    Full case names rarely appear verbatim inline (they get abbreviated to one party's
    surname, e.g. "the Dena Bank case" or "Shivakumar Reddy"), so this matches on a
    capitalized token of the cited name as a loose anchor -- a residual to refine further
    if it still proves too noisy (see the ontology's known_gaps).
    """
    candidate_tokens = re.findall(r"[A-Z][a-zA-Z.'-]+", cited_case_name)
    anchor = next(
        (t for t in candidate_tokens if len(t) >= 4 and t.lower() not in GENERIC_CAPTION_WORDS),
        None,
    )
    if not anchor:
        return []
    excerpts = []
    for m in re.finditer(re.escape(anchor), judgment_text):
        start = max(0, m.start() - window // 2)
        end = min(len(judgment_text), m.end() + window // 2)
        excerpts.append(judgment_text[start:end])
    return excerpts[:1]  # first mention only, for this prototype


def classify_excerpt(client, model: str, excerpt: str, cited_case_name: str) -> dict:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f'Cited case: "{cited_case_name}"\n\nExcerpt:\n"""{excerpt}"""',
            },
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"label": "unclear", "confidence": 0.0, "raw": raw}


def load_case_citations_by_doc_id() -> dict:
    """citing_doc_id -> list of {cited_text, cited_doc_id} for edge_type == case."""
    by_doc = {}
    with CITATIONS_PATH.open() as f:
        for line in f:
            e = json.loads(line)
            if e["edge_type"] == "case" and e["citing_doc_id"]:
                by_doc.setdefault(e["citing_doc_id"], []).append(e)
    return by_doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="max number of mentions to classify")
    parser.add_argument("--model", default=os.environ.get("VOYAGER_MODEL", "llama4-scout-17b"))
    args = parser.parse_args()

    if not CITATIONS_PATH.exists():
        raise SystemExit(f"Missing {CITATIONS_PATH} -- run scripts/classify_citations.py first")

    client = get_client()
    citations_by_doc = load_case_citations_by_doc_id()

    modern_pdfs = [
        p for p in PDF_DIR.rglob("*.PDF") if int(p.parent.name) >= 2000
    ]
    print(f"Modern-era PDFs available: {len(modern_pdfs)}")

    results = []
    n_classified = 0
    for pdf_path in modern_pdfs:
        if args.limit and n_classified >= args.limit:
            break
        reader = PdfReader(str(pdf_path))
        text = "\n".join((p.extract_text() or "") for p in reader.pages)
        title = text.split("\n")[0].strip()

        doc_id_match = re.search(r"doc/(\d+)/", text)
        doc_id = doc_id_match.group(1) if doc_id_match else None
        edges = citations_by_doc.get(doc_id, [])
        if not edges:
            continue

        for edge in edges:
            if args.limit and n_classified >= args.limit:
                break
            excerpts = find_citation_mentions(text, edge["cited_text"])
            if not excerpts:
                continue
            result = classify_excerpt(client, args.model, excerpts[0], edge["cited_text"])
            n_classified += 1
            results.append(
                {
                    "citing_case": title,
                    "citing_doc_id": doc_id,
                    "cited_case": edge["cited_text"],
                    "cited_doc_id": edge["cited_doc_id"],
                    "label": result.get("label", "unclear"),
                    "confidence": result.get("confidence", 0.0),
                    "polarity": POLARITY_MAP.get(result.get("label"), "neutral"),
                    "excerpt": excerpts[0],
                }
            )
            print(f"[{n_classified}] {edge['cited_text'][:40]:40s} -> {result.get('label')}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    label_counts = Counter(r["label"] for r in results)
    print(f"\nClassified {len(results)} citation mentions")
    for label, count in label_counts.items():
        print(f"  {label:25s}: {count}")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
