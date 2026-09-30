# label_outcome.v1

You are reading a judgment of the Supreme Court of India in a land or property dispute.
Your task is to record what the Court actually DECIDED. You are not predicting anything and
you are not evaluating whether the decision was correct.

You will be shown two excerpts of one judgment:
- OPENING: the first part, which identifies the parties and recites the history of the case.
- ORDER: the final part, which contains the Court's operative order.

Answer with ONLY a JSON object, no prose and no code fences:

```
{
  "initiator_outcome": "WIN" | "LOSE" | "PARTIAL" | "REMAND" | "OTHER",
  "operative_quote": "<verbatim sentence from ORDER stating the disposition>",
  "initiator_role": "appellant" | "petitioner" | "plaintiff" | "applicant" | "unclear",
  "initiator_was_original_plaintiff": true | false | null,
  "original_plaintiff_outcome": "WIN" | "LOSE" | "PARTIAL" | "REMAND" | "OTHER" | null,
  "prior_court_outcome": "below_allowed" | "below_dismissed" | "below_decreed" | "below_remanded" | "none_stated",
  "confidence": 0.0-1.0
}
```

Definitions, applied strictly:

- `initiator_outcome` is from the perspective of **the party who brought THIS proceeding
  before the Supreme Court** — the appellant, or the petitioner in a writ petition. If the
  appeal is allowed, that party WINS. If the appeal is dismissed, that party LOSES.
  Do not reason about who "deserved" to win.
  - `PARTIAL`: allowed only in part, or allowed as to some appeals/issues and not others.
  - `REMAND`: the substantive question is not decided; the matter goes back to a lower
    court or authority for fresh consideration. Choose REMAND only when the Supreme Court
    itself remands. If the Supreme Court merely *approves* a remand ordered by the court
    below, that is not REMAND — label what the Supreme Court did to the appeal before it.
  - `OTHER`: withdrawn, settled, compromised, infructuous, does not survive, or disposed of
    without deciding the merits.

- `operative_quote` must be copied **character for character** from ORDER. Do not
  paraphrase, do not join fragments, do not use an ellipsis. If you cannot find such a
  sentence, use the empty string "".

- `initiator_was_original_plaintiff`: was the party appealing to the Supreme Court the same
  party who originally sued in the trial court? Use `null` if OPENING does not say.
  `original_plaintiff_outcome` is the result **for the original plaintiff**, which is the
  opposite of `initiator_outcome` when the appellant was the original defendant. Use `null`
  when you cannot tell, and never guess.

- `prior_court_outcome` is what the court immediately below did, as recited in OPENING.
  Use `none_stated` if it is not stated.

Report `confidence` below 0.5 when the order disposes of several matters differently, or
when the operative sentence is unclear.
