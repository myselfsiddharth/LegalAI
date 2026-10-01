# baseline_cf.v1

You are predicting the outcome of a land or property dispute before the Supreme Court of India.

Instead of the judgment text you are given a STRUCTURED SUMMARY of the case: canonical fact labels
extracted from the facts and pleadings, each with the party who asserted it and whether it was
contested. Labels follow a controlled vocabulary — `instrument.sale_deed`, `possession.dispossessed`,
`event.notice_served.not` — where a trailing `.not` marks a negated fact.

Predict whether the party who brought the proceeding — the appellant, or the petitioner in a writ
petition — won or lost.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"outcome": "WIN" | "LOSE", "confidence": 0.0-1.0}
```

Do not explain. You must pick WIN or LOSE. Set `confidence` near 0.5 when the labels do not settle
it — they often will not, and a hedged answer is more useful than a confident guess.
