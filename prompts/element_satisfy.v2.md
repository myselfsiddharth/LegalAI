# element_satisfy.v2

You are given the FACTS of a land or property dispute before the Supreme Court of India, and one
legal CLAIM with its constituent ELEMENTS. Decide, for each element separately, whether the facts
establish it.

The facts are the only evidence you may use. They were extracted verbatim from the judgment's
factual and pleading sections; the court's reasoning and its order are not available to you, and you
must not speculate about how the case was decided.

## For each element, return

- `element_id` — copied exactly from the list below.
- `verdict` — one of:
  - `SATISFIED` — the facts affirmatively establish this element.
  - `NOT_SATISFIED` — the facts affirmatively establish that this element FAILS. This requires
    positive evidence of failure, not absence of evidence.
  - `UNCLEAR` — the facts do not settle it, including when they simply say nothing about it.
- `quote` — for `SATISFIED` and `NOT_SATISFIED` only: a span **copied character-for-character from
  the numbered facts above**. Do not paraphrase, do not tidy the grammar, do not join two separate
  facts, and do not use an ellipsis. For `UNCLEAR`, use an empty string.
- `favours` — `claimant`, `respondent`, or `neither`.

## Use all three verdicts

Each verdict needs positive evidence, and each is a real possibility you should expect to use.

`SATISFIED` — a fact affirmatively establishes the element. Quote it.

`NOT_SATISFIED` — a fact affirmatively contradicts the element. Quote it. This is **common in
contested litigation**, where the facts frequently record the opposite of what a party must prove:
the owner DID consent, notice WAS served, the possession WAS permissive, the agreement WAS
unregistered, the suit WAS filed after the limitation period. If a fact states the contrary of the
element, that is `NOT_SATISFIED`, and you should say so rather than retreating to `UNCLEAR`.

`UNCLEAR` — the facts genuinely do not address the element, in either direction.

The one error to avoid is treating **silence** as failure. Many elements are framed negatively
("possession not by force", "no permission given"); when the facts say nothing at all about such an
element, that is `UNCLEAR`, not `NOT_SATISFIED`. The test is simple: if you can quote a fact that
contradicts the element, use `NOT_SATISFIED`; if you are inferring failure from the absence of
mention, use `UNCLEAR`.

## The quote is checked mechanically

Every `quote` is verified against the facts shown, with whitespace and case normalised. A verdict
whose quote cannot be found is discarded and recorded as a refusal. A real quote supporting a
cautious verdict is worth more than a confident verdict you cannot source. If no exact span supports
your verdict, the verdict is `UNCLEAR`.

## Reply format

Reply with ONLY a JSON object, no prose and no code fences:

```
{"verdicts": [{"element_id": "...", "verdict": "...", "quote": "...", "favours": "..."}]}
```

Return exactly one entry per element listed, in the order given.
