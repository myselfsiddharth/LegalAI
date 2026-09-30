# probe_outcome.v1

You are shown an excerpt from a judgment of the Supreme Court of India in a land or property
dispute. The excerpt has been filtered: the court's reasoning and its final order have been
removed, so it should contain only the facts, the pleadings, and the parties' arguments.

Your task is to say whether the party who brought the proceeding — the appellant, or the
petitioner in a writ petition — **won or lost**.

Answer with ONLY a JSON object, no prose and no code fences:

```
{"outcome": "WIN" | "LOSE", "confidence": 0.0-1.0}
```

Do not explain. Do not reason step by step. Give your immediate judgement.

If the excerpt still states the result outright, use it. If it does not, make your best guess
from the facts. You must pick WIN or LOSE — "unknown" is not an option. Set `confidence` near
0.5 when you are guessing.
