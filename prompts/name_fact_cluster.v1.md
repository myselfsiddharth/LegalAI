# name_fact_cluster.v1

You are given a group of factual statements drawn from Indian Supreme Court land and property
judgments. They were grouped automatically by semantic similarity, not by any legal theory. Your job
is to say what the group is *about*, and to classify what KIND of thing it is.

## Reply with ONLY a JSON object, no prose and no code fences

```
{
  "name": "<short snake_case name, e.g. mortgage_with_possession>",
  "definition": "<one sentence: what a fact must state to belong to this group>",
  "kind": "legal_substance" | "procedural" | "party_entity" | "temporal_quantum" | "incoherent",
  "legal_element": "<the legal element or doctrinal requirement this group bears on, or null>",
  "confidence": 0.0-1.0
}
```

## The `kind` field is the point of this task — classify honestly

- **`legal_substance`** — a fact bearing on a substantive legal requirement: title, consideration,
  possession and its character, notice, breach, fraud, consent, registration, succession.
- **`procedural`** — the machinery rather than the merits: which forum, what stage, how the matter
  reached this court, appeal or writ, who appealed, whether leave was granted, remand history.
- **`party_entity`** — chiefly about *who* a party is: the State, a municipal body, a company, a
  temple or trust, a tenant, a co-operative society.
- **`temporal_quantum`** — chiefly dates, durations, areas, rents, sums of money.
- **`incoherent`** — the statements share no single concept. Say so; do not invent a theme. Set
  `confidence` below 0.3.

Do not inflate a group into `legal_substance` because that sounds more useful. A group of filing
dates is `temporal_quantum`, and a group about appeals reaching this court by special leave is
`procedural`. **An honest `procedural` label is more valuable here than a flattering legal one**,
because the whole purpose of this exercise is to find out which kind of fact actually carries signal.

## `legal_element`

Name the doctrinal requirement the group bears on in plain words — "hostile possession",
"valid consideration", "statutory notice", "adverse possession limitation period". Use `null` when
the group bears on no legal requirement at all, which is the correct answer for most `procedural`,
`party_entity` and `temporal_quantum` groups.

## Name the general case

`possession.permissive`, not `possession_by_bhagwant_since_1923`. If every statement is specific to
its own case and they share no general concept, that is `incoherent`.
