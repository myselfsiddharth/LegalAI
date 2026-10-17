# fact_type.v1

You are given numbered FACTS extracted from an Indian Supreme Court judgment in a land or property
dispute. Each fact comes with its SOURCE: the sentence of the judgment it was extracted from, with
the extracted span marked ⟪like this⟫. The extraction sometimes drops who said or decided
something, so **read the SOURCE, not just the FACT**, to decide what kind of statement it is.

Label every fact with exactly one type:

- `PRIMARY` — something that happened or exists in the world of the dispute: an act, transaction,
  document and what it says, payment, possession, relationship, date, place, measurement, or a
  piece of evidence. Example: "The second defendant sold the suit property to the plaintiff by sale
  deed dated 5.6.1967."
- `PROCEDURAL` — a step in the litigation itself: a suit, petition or appeal filed; an order or
  decree passed; an appeal allowed or dismissed; a remand; a notice issued by a court. It records
  THAT a forum acted, not WHY. Example: "The trial court dismissed the suit on 12.2.1985."
- `FINDING` — a conclusion of fact or law about THIS dispute reached by a court, tribunal, or
  authority, **whether or not the FACT names who reached it**. If the SOURCE shows the span is what
  a court "held", "found", "concluded", "opined", "took the view", or "was right in holding", it is
  a `FINDING`. Example: FACT "The plaintiff did not acquire valid title." with SOURCE "...it was
  concluded that ⟪the plaintiff did not acquire any good or valid title⟫" is a `FINDING`.
  A characterisation that only a court could make — "the notice was valid", "the sale was a
  nullity", "possession was illegal from its inception", "the deed was genuine" — is a `FINDING`
  even without an explicit attribution.
- `LAW` — a general legal proposition not specific to this dispute's facts: what a statute
  provides, a rule stated in a precedent, a definition, a principle of interpretation. Example:
  "The mortgage deed is a document required to be attested by two attestors under section 59."
- `ASSERTION` — what a party claims, pleads, contends, or alleges, not established. Example: "The
  defendant pleaded that the mortgage deed was a sham document."

When torn between `PRIMARY` and `FINDING`, ask: is this a raw happening anyone could witness, or a
legal or evaluative conclusion someone had to reach? The second is a `FINDING`.

Reply with ONLY a JSON object, no prose and no code fences:

```
{"labels": [{"i": 1, "type": "PRIMARY"}, {"i": 2, "type": "FINDING"}]}
```

Return exactly one entry per fact, using the fact numbers given.
