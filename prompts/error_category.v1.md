# error_category.v1

A model predicted the outcome of a land or property dispute before the Supreme Court of India and
got it wrong. You are shown the model's prediction, the actual outcome, and the case's facts and
pleadings — the same material the model saw. The court's reasoning and order are not included.

The pipeline has already ruled out the mechanical explanations (thin extraction, missing claim
family, mismatched statutes, label noise). So decide between the two remaining possibilities:

```
{"category": "facts_underdetermine" | "facts_point_other_way" | "unclear",
 "reason": "<one short clause>"}
```

- `facts_underdetermine` — the facts shown genuinely do not settle who should win. The outcome
  turned on something not in this material: credibility, the court's discretion, a point of law
  argued but not recited, or a judgement call a reasonable reader could go either way on.
- `facts_point_other_way` — the facts shown do favour the actual winner, and a careful reader
  working only from this material should have reached the actual outcome. The model missed
  something that was present.
- `unclear` — you cannot tell from the material given.

Be willing to say `facts_underdetermine`. Many property disputes turn on matters an appellate
judgment records only in its reasoning, and reporting those as model failures would overstate how
much signal the facts contain.
