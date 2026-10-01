# Data/gold — human annotation, and why these specific items

Two blockers in `reports/BLOCKERS.md` need a person rather than more compute. Everything here is
prepared so the only remaining cost is reviewer time.

Regenerate with `make gold`. Annotate with `make annotate`.

## 1. `gold_tasks.jsonl` — 150 cases (§6.4)

Pre-filled with pipeline output for **correction, not authoring**. 2,126 individual fact judgements.

| stratum | n | what it buys |
|---|---|---|
| `conflict` | 40 | The rules and LLM labellers disagreed, so these have **no usable label at all**. Resolving them settles §5.1's outstanding human check *and* returns 40 cases to the eligible pool. Highest information per annotation. Outcome-only tasks — they were never fact-extracted, because extraction required a binary label. |
| `stratified` | 80 | Decade × merged claim family, 9–11 cases per decade across all eight. This is the backbone that makes §6.1's extraction F1 and §6.2's canonical-label accuracy measurable at all — currently neither can be computed (BLOCKER B4). |
| `error` | 30 | Cases the best model got wrong that trip **none** of §10.3's deterministic signals. §10.3 found no signal explaining its errors (every lift ≈ 1.0), and an LLM's opinion on them was discounted because that LLM scores 0.583 itself. A reviewer can say whether the facts determine the outcome at all. |
| `agreement` | 30 | A second-annotator overlap block, drawn from `stratified` so §6.4's inter-annotator agreement is measured on ordinary cases rather than hard ones. Flagged `second_annotator: true`. |

### How the tool behaves, and why

- **Nothing is pre-selected.** Every control starts at `(not reviewed)`. If the pipeline's answer
  were the default, clicking through would manufacture agreement and §6.1's F1 would measure the
  reviewer's patience instead of the extractor. The pipeline's answer is shown beside each control,
  so agreeing is fast — but it is an action.
- **Disagreement is kept, not overwritten.** Pipeline and reviewer values sit side by side, so
  agreement can be computed per field and per stratum.
- **Missed facts can be added.** §6.1 asks for recall as well as precision; without a box for facts
  the extractor failed to find, only precision would be measurable.
- **Every fact shows the quote it was extracted from**, so the judgement is against the span the
  extractor actually relied on.

Output goes to `gold_annotations.jsonl`, append-only — a crashed session loses nothing and the
correction history survives.

## 2. `element_burden_sheet.csv` — 66 elements (BLOCKER B2)

`burden_on`, `burden_standard`, `burden_shifts_when` for every ontology element. §8.3's element
mapping and §10.1's `E` feature group both read these, and the `E` group is **empty** without them.

**Deliberately not pre-filled with a model's draft.** Two models have failed the degeneracy gate in
`src/data/ontology_burden.py`:

| model | result |
|---|---|
| `llama4-scout-17b` | refused — 92% `claimant`, `clear_proof_required` used **0 times** including for adverse possession, 2 distinct confidence values |
| `glm-5-3` | refused — 87% `claimant`, over the 85% limit |

Both annotations looked complete and carried almost no information. Showing a reviewer a plausible
wrong answer invites acceptance, and plausible-but-constant is precisely this task's failure mode.
So the sheet carries only the ontology's **own** claim-level guidance (28 of 66 rows) and leaves the
rest blank.

Rows are ordered by how many corpus cases the claim covers, so elements that can actually affect a
result come first. Possession and Adverse Possession (1,218 cases) lead; a claim with 3 cases cannot
move any number.

**Leave a row blank rather than guessing.** A blank is a known gap; a wrong value becomes an `E`
feature and nothing downstream will question it.
