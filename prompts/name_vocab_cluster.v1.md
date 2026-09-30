# name_vocab_cluster.v1

You are naming a cluster of proposed labels for a controlled vocabulary of facts in Indian land
and property disputes.

Each proposal was made by a labeller that could not find a fitting label in the existing
vocabulary. Proposals that mean the same thing have been grouped; your job is to give the group
one general name that future cases can also match.

Reply with ONLY a JSON object, no prose and no code fences:

```
{
  "category": "<an existing category from the list given, or NEW:<proposal>>",
  "subcategory": "<a short snake_case name for this group>",
  "definition": "<one sentence: what a fact must state to take this label>",
  "merge_into": "<an existing category.subcategory this group duplicates, or null>",
  "confidence": 0.0-1.0
}
```

## Rules

**Prefer merging.** If the group is a rewording of a label the vocabulary already has, set
`merge_into` and the group will be folded in rather than added. A vocabulary that grows a near
duplicate for every phrasing cannot support pattern mining, which is the whole reason it exists.

**Name the general case, not the instance.** `possession.permissive`, not
`possession.by_bhagwant_since_1923`. If the proposals in a group are all specific to individual
cases and share no general concept, say so by setting `confidence` below 0.3.

**One concept per group.** If the proposals plainly cover two different things, name the dominant
one and set `confidence` below 0.5 so a reviewer splits it.

**Reject non-facts.** If the group is really about a statute, a cited case, or a legal principle
rather than a fact about a dispute, set `category` to `NEW:NotAFact` — these should not be in a
fact vocabulary at all.
