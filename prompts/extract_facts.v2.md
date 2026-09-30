# extract_facts.v2

You are reading an excerpt from a judgment of the Supreme Court of India in a land or property
dispute. The court's reasoning and its final order have already been removed: what remains is
the facts, the pleadings, and the parties' arguments.

You will be told who the parties are before the excerpt. Use that to attribute facts.

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
- **statements of law.** "Section 13 of the Act provides for a second appeal", "it is well
  settled that possession must be hostile", "in the former case the land would be prima facie
  resumable" — these describe the law, not this dispute. Leave them out even when the excerpt
  spends paragraphs on them.
- **what a cited case held.** "The Privy Council pointed out the distinction between...",
  "Ram Gopal v. Nand Lal held that..." — these are about an authority, not about this property.
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

### `asserted_by` decides whether the fact is usable at all

Later stages build one view of the case from the plaintiff's side and one from the defendant's.
A fact marked `court_narrative` enters **neither** view and is effectively discarded, so that
label is a last resort, not a default. Choose in this order:

1. `plaintiff` — the party who brought this proceeding asserts it, pleaded it, or relies on it.
   A court *reciting* that party's case is still that party asserting it: "the plaintiff's case
   is that he purchased the land in 1972" is `plaintiff`, not `court_narrative`.
2. `defendant` — the opposing party asserts it, pleaded it, or relies on it. Same rule about
   recitation.
3. `admitted` — undisputed. This covers the ordinary background a judgment states flatly and
   neither side contests: the survey number, the date of a registered deed, which court passed
   which earlier order, who died when. **Most background narration belongs here.**
4. `court_narrative` — the fact is contested or partisan, but the judgment does not say whose
   assertion it is. Use it only when you genuinely cannot attribute a *disputed* fact.
5. `unknown` — you cannot tell even that much.

If you find yourself choosing `court_narrative` for a quiet, uncontested background fact, the
answer is `admitted`.

### Distinguish "not proved" from "proved false"

These are different facts and must not be merged. If the excerpt says notice was never served,
that is a fact with `disputed_status` reflecting how it is established. If the excerpt merely
fails to establish service, do not invent a fact that it was not served.

Return `[]` if the excerpt contains no extractable facts.
