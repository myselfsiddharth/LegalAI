# canonicalize_fact.v3

You map facts from Indian land/property judgments onto a **controlled vocabulary**, so that two
judgments describing the same thing in different words reduce to the same label.

You will be given a numbered list of FACTS. **Each fact carries its own CANDIDATES line** — the
labels shortlisted for that fact specifically. Choose from the candidates listed under the fact you
are labelling.

Reply with ONLY a JSON array, no prose and no code fences — **one object per fact, and you must
include every fact index exactly once**:

```
[
  {
    "i": <the fact's number, exactly as given>,
    "category": "<one category from the candidate list, or NEW>",
    "subcategory": "<one subcategory from that category, or NEW:<your proposal>>",
    "properties": {
      "polarity": "pos" | "neg",
      "proof_status": "asserted" | "admitted" | "contested" | "not_shown" | "proved_false" | "inferred" | "unknown",
      "actor": "claimant" | "respondent" | "authority" | "third_party" | "court_below" | "unknown",
      "duration_years_bin": "lt3" | "3to12" | "gt12" | "gt30" | "unknown",
      "amount_bin": "lt10k" | "10k_1l" | "1l_10l" | "10l_1cr" | "gt1cr" | "unknown",
      "delay_years_bin": "lt1" | "1to3" | "3to12" | "gt12" | "unknown"
    },
    "confidence": 0.0-1.0
  },
  ...
]
```

Include a property only when the fact actually says something about it. Omit the rest; do not pad
with `"unknown"`.

## Rules

**`i` must be the fact's own number.** Do not renumber, reorder or merge facts. If you cannot label
one, still emit an object for it with `"confidence": 0.0`.

**Pick from that fact's own CANDIDATES when anything fits.** The shortlist exists so labels are
comparable across thousands of cases, and it was retrieved for this fact by similarity — a fitting
label is usually in it. Read the whole line before deciding nothing fits. Only answer `NEW:` when no
candidate covers the fact — and then propose a short,
general label other cases could also match (`possession.permissive`, not
`possession.by_bhagwant_since_1923`).

**Polarity is part of the meaning.** "Notice was served" and "notice was not served" take the same
subcategory with `polarity` `pos` and `neg`. Never fold a negative into a positive.

**`proof_status` distinguishes "not proved" from "proved false".** These are different facts:
- `not_shown` — nobody established it either way.
- `proved_false` — it was established that the opposite is true.
Getting this wrong corrupts any rule about who carried a burden, so use `unknown` if unsure rather
than guessing `proved_false`.

**Bin numbers, do not invent precision.** "in possession since 1954, suit filed 1971" is
`duration_years_bin: "gt12"`. Only set a bin the fact supports.

**`actor` is who did or holds the thing**, not who is telling you about it.
