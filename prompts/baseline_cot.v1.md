# baseline_cot.v1

You are predicting the outcome of a land or property dispute before the Supreme Court of India.

The case text has been filtered: the court's reasoning and its order are removed. What remains is
the facts, the pleadings, and the parties' arguments.

Work through the claim ELEMENT BY ELEMENT before answering. For the claim the case chiefly turns on,
consider each element listed below that applies, decide whether the facts establish it, and note who
bore the burden. Then give the outcome.

Reply with ONLY a JSON object, no prose and no code fences:

```
{
  "claim": "<the claim the case chiefly turns on>",
  "elements": [
    {"element": "<name>", "established": true | false | "unclear", "why": "<one short clause>"}
  ],
  "outcome": "WIN" | "LOSE",
  "confidence": 0.0-1.0
}
```

`outcome` is from the perspective of the party who brought the proceeding — the appellant, or the
petitioner in a writ petition. You must pick WIN or LOSE.

Judge the elements from the facts shown. Do not assume an element is established because it would
be convenient for one side.
