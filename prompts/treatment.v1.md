# treatment.v1

Each numbered item is a passage from an Indian Supreme Court judgment (the CITING case) around a
place where it mentions an earlier decision (the CITED case, named after "CITED:"). Decide how the
citing court itself treats the cited case.

- `FOLLOWED` — the court relies on it, applies it, approves it, or treats it as binding.
- `DISTINGUISHED` — the court says it does not apply here because the facts or question differ.
- `OVERRULED` — the court overrules it, disapproves it, or declares it no longer good law.
- `DOUBTED` — the court questions, doubts, narrows or refers it to a larger bench, without
  overruling.
- `REFERRED` — merely mentioned, summarised, or relied on by COUNSEL without the court expressing a
  view; or the passage does not show the court's treatment.

Give `quote`: a short span (8 to 40 words) copied character-for-character from the passage that
shows the treatment. For `REFERRED` the quote may be empty.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"labels": [{"i": 1, "treatment": "FOLLOWED", "quote": "..."}]}
```

Return exactly one entry per item.
