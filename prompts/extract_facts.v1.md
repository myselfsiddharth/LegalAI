# extract_facts.v1

You are reading an excerpt from a judgment of the Supreme Court of India in a land or property
dispute. The court's reasoning and its final order have already been removed: what remains is
the facts, the pleadings, and the parties' arguments.

Extract the **atomic factual propositions** in this excerpt.

## What counts as a fact here

A fact is a single, checkable proposition about what happened, what a document says, or what a
party asserts. Keep them atomic: "the sale deed was executed in 1972 and registered in 1974" is
**two** facts.

Extract:
- events (a deed executed, possession taken, a notice issued, a suit filed, an order passed)
- states of affairs (who held title, who was in possession, what a record showed)
- what a party pleaded, claimed or alleged
- what a document or record contains

Do **not** extract:
- the court's own reasoning, evaluation, or conclusion
- statements of law, or what a cited case held
- procedural boilerplate (counsel names, case numbers, dates of hearing)
- anything you cannot tie to a verbatim quote from the excerpt

## Output

Reply with ONLY a JSON array, no prose and no code fences. Each element:

```
{
  "text": "<the proposition, in your own words, one sentence>",
  "quote": "<verbatim span from the excerpt that supports it>",
  "asserted_by": "plaintiff" | "defendant" | "court_narrative" | "admitted" | "unknown",
  "disputed_status": "admitted" | "contested" | "unclear" | "inferred",
  "event_date": "<YYYY, YYYY-MM, YYYY-MM-DD, a range, or null>",
  "entities": ["<party, property, authority or instrument named>"],
  "evidence_refs": ["<exhibit or document cited as proof of it, if any>"]
}
```

### The quote is the evidence

`quote` must be copied **character for character** from the excerpt. Do not paraphrase inside
it, do not join separated fragments, do not use an ellipsis. If you cannot produce such a
quote, leave the fact out. A fact whose quote does not appear in the excerpt will be discarded.

### `asserted_by` matters more than it looks

Later stages separate what the plaintiff's case rests on from what the defendant's does, so be
careful:
- `plaintiff` — the party who brought the proceeding asserts it
- `defendant` — the opposing party asserts it
- `admitted` — both sides accept it, or it is recorded as admitted
- `court_narrative` — the judgment recites it as background without attributing it
- `unknown` — you cannot tell

### Distinguish "not proved" from "proved false"

These are different facts and must not be merged. If the excerpt says notice was never served,
that is a fact with `disputed_status` reflecting how it is established. If the excerpt merely
fails to establish service, do not invent a fact that it was not served.

Return `[]` if the excerpt contains no extractable facts.
