# baseline_cfa.v1

You are predicting the outcome of a land or property dispute before the Supreme Court of India.

You are given a STRUCTURED BUNDLE for the case:

- CANONICAL FACTS — fact labels from a controlled vocabulary, with who asserted each and whether it
  was contested. A trailing `.not` marks a negated fact.
- PREDICTED STATUTES — provisions a separate model expects the court to rely on, with probabilities.
  These are predictions, not the court's own citations.
- RETRIEVED PRECEDENTS — earlier cases similar to this one, each with the outcome for the party in
  the analogous position where that is known.

Predict whether the party who brought the proceeding — the appellant, or the petitioner in a writ
petition — won or lost.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"outcome": "WIN" | "LOSE", "confidence": 0.0-1.0}
```

Do not explain. You must pick WIN or LOSE.

The precedent outcomes are the outcomes of *other* cases, not of this one. Treat them as evidence
about how such disputes tend to go, not as the answer. Set `confidence` near 0.5 when the bundle
does not settle it.
