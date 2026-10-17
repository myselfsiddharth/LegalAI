# case_card.v1

You are given an Indian Supreme Court judgment (its opening and its closing portion). Write a
precedent card a lawyer would use to decide whether this case is worth citing.

Return ONLY a JSON object, no prose and no code fences:

```
{
  "dispute": "one sentence: who fought over what",
  "issues_decided": ["...", "..."],
  "holdings": ["...", "..."],
  "provisions": ["Act, Section", "..."],
  "doctrines": ["...", "..."]
}
```

- `issues_decided` — the legal questions the court actually answered (1 to 4).
- `holdings` — what the court decided on each, as general propositions a later court could rely on
  (1 to 4), each under 40 words.
- `provisions` — Acts and sections the decision turned on, full Act names.
- `doctrines` — legal doctrines applied.

Base everything on the text given. If the closing portion does not show a holding clearly, say what
the court decided as far as the text shows; do not invent.
