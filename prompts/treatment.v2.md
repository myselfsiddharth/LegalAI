# treatment.v2

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

## Whose treatment: the CITED case only

Judge what happens TO the cited case named after "CITED:", never to some other case in the passage.
Passages often describe one case overruling another. Ask: is the CITED case the one being overruled,
or the one doing the overruling?

- "In Indore Development Authority, the Constitution Bench overruled Pune Municipal Corporation" --
  if CITED is *Pune Municipal Corporation*: `OVERRULED`. If CITED is *Indore Development Authority*:
  `FOLLOWED` (it is the authority the court relies on).
- "Kesavananda Bharati overruled Golaknath" -- CITED *Kesavananda Bharati*: `FOLLOWED`.
- "X v. Y, [1964] 6 SCR 666, overruled." in a list -- applies to X v. Y.
- "the Court declared the Act unconstitutional in A v. B" -- A v. B is the deciding authority:
  `FOLLOWED`, not `OVERRULED`.
- A decision set aside or reversed ON APPEAL in the same litigation is not a precedent being
  overruled; use `REFERRED` unless the passage says its reasoning was disapproved.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"labels": [{"i": 1, "treatment": "FOLLOWED", "quote": "..."}]}
```

Return exactly one entry per item.
