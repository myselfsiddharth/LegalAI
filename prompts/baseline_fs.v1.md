# baseline_fs.v1

You are predicting the outcome of a land or property dispute before the Supreme Court of India.

You will be shown EXAMPLES of earlier decided cases with their outcomes, then the CASE to predict.
The examples are all decided BEFORE the case you must predict, so nothing about the future is
available to you.

The case text has been filtered: the court's reasoning and its order are removed. What remains is
the facts, the pleadings, and the parties' arguments.

Predict whether the party who brought the proceeding — the appellant, or the petitioner in a writ
petition — won or lost.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"outcome": "WIN" | "LOSE", "confidence": 0.0-1.0}
```

Do not explain. You must pick WIN or LOSE. Set `confidence` near 0.5 when you are guessing.
