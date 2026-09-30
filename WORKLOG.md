# WORKLOG

Chronological record of what was built, what was measured, and **why decisions were made**.

Three documents, three jobs — don't duplicate between them:

| file | job |
|---|---|
| `PROJECT.md` | the spec. What we intend to build. Changes rarely. |
| `reports/M*_report.md` | stage acceptance. The numbers a stage must hit, in final form. |
| **`WORKLOG.md`** (this) | the running log. Decisions, dead ends, bugs, and the reasoning behind each. |

Rule for this file: **record the reasoning, not just the outcome.** A number without the
decision it drove is already in the stage report. What belongs here is what we tried, what
failed, and what we would otherwise re-derive from scratch next session.

---

## Current state at a glance

| stage | status | entry point |
|---|---|---|
| §4 Stage 0 — registry, text, ontology, LLM client | **done** | `make stage0` |
| §5 Stage 1 — screen, labels, masking, splits, leakage probes | **done, gate passed** | `make stage1 labels-llm merge probe llm-probe` |
| §6 Stage 2 — seed vocabulary, fact extraction, canonicalisation | facts running at scale; canonicalisation pilot only | `make stage2` |
| §7 Stage 3 — claims, defences, family clustering | code written, **not yet run at scale** | `make stage3` |
| §8 Stage 4 — FP-Growth fact patterns | not started | — |
| §9 Stage 5 — statutes, precedents, citation graph | not started (assets exist in `scripts/`) | — |
| §10 Stage 6 — outcome models, ablations | not started | — |
| §11 trace | not started | — |

**Dataset**: 6,954 land/property cases · 5,329 binary WIN/LOSE labels · 4,707 eligible for
outcome experiments (binary label ∧ ≥500 masked chars ∧ property domain).

**The four headline results so far** (forum-held-out split, AUROC):
`order_only` 0.983 → `full` 0.827 → **`masked` 0.655** → `prior_court` 0.581 → chance 0.500.

---

## 2026-09-29/30 — Session 1: rebuild on PROJECT.md, Stages 0–2

### Framing decision

Asked whether to target one paper or several. **User's answer: build the pipeline and get the
best results; the papers will emerge.** Accepted — the stages are strictly dependent
(corpus → labels → masking → facts → patterns → authorities → outcome), so the critical path is
the same either way. Consequence: build for correctness at each stage rather than optimising
any single paper's headline.

I had proposed recasting P3 as a leakage-first benchmark because a horse race against a frontier
model is a bet that can lose. That reframing is *not* adopted as the plan, but the leakage work
it implied was built anyway because §5.2 requires it, and it turned out to be where the
defensible results are.

### Reconnaissance before writing code

Measured rather than assumed, because CLAUDE.md's warnings suggested the data was worse than it is.

- **The land-dispute PDF join is exact.** `filtered_land_disputes.xlsx` has a `filename` column;
  all 6,954 resolve on disk. **CLAUDE.md gotcha #15's fragile title normaliser is unnecessary
  for this subset** — it was only needed for the general corpus. Asserted in code at a 99% floor.
- Only 2,921 of the land cases had cached text; the old 5,462-doc cache was built for the
  *general* corpus. Extracting the rest was the cheapest available win (+2.4x data, 4 minutes).
- **No section headers to segment on**: `ORDER` as a header 3.7%, `FACTS` 29.7%. But numbered
  paragraphs are in 92.7% of post-2000 judgments, median 32. So the unit is the paragraph.
- **19.2% of outcome cues sit in the first 30% of a judgment** — the procedural recital. This
  drove the decision to give prior-court disposition its own channel (see below).
- `data/` and `Data/` are **the same directory** on this machine (APFS is case-insensitive).
  Standardised on `Data/` so the code also works on a case-sensitive filesystem, where
  PROJECT.md §12's `data/` would be a different directory.

### Blocker opened and closed

`Land_Property_Dispute_India_Ontology_Refined.docx` was missing from `Docs/`
(`reports/BLOCKERS.md`). User pointed at `~/Downloads`. Copied both v0 and v1 into
`Docs/ontology_source/`, and un-ignored that one subdirectory so the YAML's provenance is
versioned while the confidential Templeton grant doc stays ignored.

Note: `Docs/` had to become `Docs/*` first — **git cannot re-include a file whose parent
directory is excluded**, so a `!Docs/ontology_source/**` negation silently did nothing.

`ontology_convert.py` **parses** the docx rather than retyping it, so a corrected docx can be
re-converted instead of hand-merged: 15 claims, 66 elements, 6 defence families, 4 remedy tiers.

### The LLM client came first, deliberately

`src/llm/client.py` is the single door to Voyager. The previous client had no cache, no retries,
no structured-output validation, and no call logging, and that cost this project three times over
— most sharply when a swallowed `APIConnectionError` per item turned an outage into a substantive
finding ("0 elements with support ≥2" when 150 of 253 sentences were never asked).

Design points worth keeping:
- Cache key is `sha256(model, messages, every param that changes output)`. Temperature defaults
  to 0, so a cache hit is a genuine repeat and not a resampling.
- `LLMResult.ok=False` means **we never got an answer**. An empty answer with `ok=True` means the
  model said nothing. Every batch summary must report these separately.
- 90s timeout, not the SDK's 600s × 3.
- Prompts live in `prompts/*.v*.md`, never inline, so every number can name its prompt.

Smoke-tested all five paths. Incidental finding: **Voyager's embeddings come back L2-normalised**
(norm 1.000), so cosine similarity is just a dot product.

### Stage 1 decisions

**Prior-court disposition is a separate channel, not part of the mask.** It is simultaneously a
legitimate historical fact and a potential shortcut. Deleting it removes real facts; keeping it
silently lets a model win by reading the court below. So `mask.py` emits `prior_court_text`
alongside `masked_text`, and §10.2 ablates it.
*Then the measurement contradicted the hypothesis* — see Findings below.

**§5.3's court-held-out split is impossible.** The corpus is Supreme Court only; there is no
second court to hold out. Substituted **originating High Court** (recovered for 78.9% of cases by
`screen.py`) and labelled it as the weaker control it is.

**Domain contamination is tagged, not deleted.** The spreadsheet's keyword filter admits cases
that merely *happen* on land — 5.5% criminal appeals (one is a 2005 murder appeal arising from an
affray over paddy-cutting), 3.9% tax-heavy. A criminal appeal over a boundary affray may carry
usable possession facts, so each experiment states its subset instead of us deciding globally.

**Conflicting labels are held as UNRESOLVED, not broken by coin-flip.** Breaking them would
manufacture label noise and quietly cap every downstream number. 562 cases are queued in
`label_conflicts.jsonl` for §5.1's human check.

**The leakage probe needs a sensitivity control or it is uninterpretable.** "Suspiciously high
relative to what?" is the whole question, so the probe runs `order_only` (the text masking
removes) beside `masked`, `full` and `majority`. Without the 0.983 arm, a low `masked` score is
indistinguishable from a blind probe.

### Findings that change the plan

1. **The procedural shortcut is weak — my hypothesis was wrong.** Predicted prior-court
   disposition would dominate. Measured AUROC **0.581** (forum) and **0.518** (temporal, i.e.
   chance). Likely cause: the Supreme Court grants leave precisely in contestable cases, and the
   recital is near-balanced (26.1% below_dismissed vs 23.1% below_allowed). Keep it as a control;
   do not build an argument on it.
2. **Base-rate drift is large and real.** WIN rate runs **39.2% → 63.8%** across the temporal
   split. Confirmed in raw cue counts (`allowed` 23.6% pre-2000 vs 41.0% post), so it is the data,
   not the labeller. **Always report against a per-period majority baseline, never a global one.**
3. **Zero-shot LLM-0 is *worse* than TF-IDF+LR on honest inputs** — AUROC 0.552 vs 0.655 — and
   badly miscalibrated: predicts WIN 13% of the time where truth is 43%. §0.1's "the LLM is always
   the baseline" does not mean the LLM is the *strong* baseline.
4. **Residual headroom is ~0.655 AUROC.** This is the number to quote against LJP work that does
   not control leakage.

### Bugs found, and how

Almost all of these came from **reading output**, not from a metric moving.

| # | bug | how found | effect of fix |
|---|---|---|---|
| 1 | Segmentation returned **one unit** for 57.7% of cases — extracted text has no blank lines, so the blank-line fallback returned the whole judgment. One unit takes one role, so a single reasoning phrase discarded the entire document. | masking reported median 0% retained | added a sentence-window tier: median 47 units, min 6; unusable cases 57% → **2.6%** |
| 2 | Quote grounding rejected **correct** quotes at 42%. Two causes, both mine: pre-1980 PDFs carry hyphenated line wraps (`plain- tiff's`) that a model silently repairs; and a 20-char floor rejected `Appeal dismissed.`, the real operative line in old law-report style (15 of 16 were verbatim present). | diagnosed the ungrounded set instead of assuming the model paraphrased | grounding 58.3% → **91.7%** |
| 3 | **Consequential dismissals read as the disposition** — the largest label error. Orders say "the appeal is allowed … and the writ petition filed by the respondent stands dismissed"; taking the last disposition gives LOSE where the appellant plainly won. | audited the 196 LOSE→WIN conflicts; the LLM was right in all 10 sampled | rank the proceeding **before this court** above the underlying one; conflicts 196 → 107 |
| 4 | **Cross-verb binding**: `appeal … is dismissed` matched *across* "the appeal is allowed and the suit is dismissed" because the noun→verb gap was unconstrained. | remaining conflicts after fix 3 | each proceeding noun binds to its nearest disposition verb; ambiguous 3.0% → 1.2% |
| 5 | Group-whole and time-respect **cannot both hold** for a dispute litigated across a split boundary: by latest year, 1970 cases landed in "test > 2013"; by earliest year, 2025 cases landed in "train ≤ 2004". | `tests/test_stage1.py` caught both | drop straddling groups — 13 cases, 0.3% |
| 6 | `asserted_by` was **84% `court_narrative`**, which under §8.1 puts a fact in *neither* party view. Prompt tightening + case caption barely moved it (→79.5%) and `asserted_by="admitted"` stayed at **0%**. But `disputed_status="admitted"` was 81% — the model uses one field for attribution, the other for admittedness. | inspected the field distribution instead of iterating on the prompt | orientation reads **both** fields: coverage 17% → **96%**, orphaned facts 79.5% → **4.0%** |
| 7 | Statements of law leaked into facts despite the prompt forbidding them ("Section 13 provided for a second appeal", "The Privy Council pointed out…"). These would make FP-Growth encode the statute book rather than the case. | read sampled facts | `fact_filters.py` rejects them deterministically (10/10 on a hand-built set), as the earlier content guard did |
| 8 | **The canonicaliser's NEW rate of 0.7% was untrustworthy** — 16.7% of its answers were labels outside the vocabulary that it had not flagged NEW. Trusting it would have frozen a vocabulary that could not name a sixth of the facts. | checked its answers against the vocabulary rather than counting its flags | out-of-vocabulary is decided by checking, not trusting; real rate ~17.4%, so §6.2's loop is not done |
| 9 | Fact extraction was **not applying the property screen** that `splits.py` applies, so tax and criminal cases were extracted. | saw `Statute.fee_for_appeal` and a fact about a whisky dealer in canonicalisation output | eligibility unified: 5,102 → 4,611 cases |

**Two things that looked like bugs were my own detectors, not the data.** The LOSE "negation"
flags were all `no order as to costs` trailing the verb; and in every REMAND flagged for naming a
lower court, that court was the remand's *destination*, not the actor. Both labels were correct.
Worth remembering before "fixing" on the strength of a crude detector.

### Open items

- [ ] §5.1 human check of 200 — draw from the 562 UNRESOLVED conflicts (highest information per
      annotation) plus a random stratum for an unbiased estimate. **Needs a human.**
- [ ] §6.2 vocabulary loop: cluster the out-of-vocabulary proposals, review, merge, repeat until
      under 5%, then freeze as `vocab_v2.yaml`. Currently ~17.4%.
- [ ] `unclear` is 27.6% of masking units and is discarded. An LLM section classifier over just
      those units is the cheapest way to recover facts text; §5.2 permits it.
- [ ] Reuse from `scripts/`: `retrieval_lib.py` (BM25 + dense behind one `search()`, and
      `before_year=` is already time-respecting per §5.3), `signed_graph_edges.csv` (8,467 edges),
      `dense_index.npy`, `verify_authority`. These serve §9 directly.
- [ ] Atom sparsity: the canonicalisation pilot gave 107 distinct atoms over 150 facts (1.4
      facts/atom). FP-Growth support is per **case**, not per fact, so this may be fine — but
      measure atoms-per-case before running §8, since it is free and predicts the outcome.

---

## 2026-09-30 — Session 1 continued: Stages 3–4

### §7 result: the ontology's 15-family taxonomy is NOT supported by the data

This is a negative result and it is the honest headline of §7. Measured on 465 claims from 300
cases:

| clusterer | k | NMI | ARI | purity | note |
|---|---|---|---|---|---|
| HDBSCAN | 2 | 0.105 | **−0.005** | 0.495 | 80% outliers; ARI ≈ 0 means its partition is **uncorrelated** with the assigned families |
| agglomerative (k by silhouette) | 5 | 0.271 | 0.092 | 0.318 | **silhouette 0.035** — almost no structure at this granularity |

Supporting facts:
- Only **2 of 15** ontology families cleared §7's 30-case floor (Title 87 cases, Possession 78).
- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.
- `Redevelopment` was never assigned to any claim.
- 34% of cases fall in more than one family, consistent with §7.4's many-to-many requirement.

**But the clusters are legally coherent one level up**, which is the real finding — the taxonomy is
too *fine* for this corpus, not wrong:

```
cluster 0  Title 49, Possession 41, Partition 13, Trust 5              -> private title & possession
cluster 1  SpecificPerformance 25, Title 15, Cancellation 10           -> contract & instrument
cluster 2  Possession 28, Title 9, Tenure 7, LeaseTenancy 4            -> tenure & occupancy
cluster 3  Title 35, UltraVires 20, ConstitutionalDeprivation 19,
           LandAcquisition 13                                          -> state action against property
```

**Action taken** (`src/cluster/family_merge.py`, logged to `ontology/CHANGELOG.md` per §6.3):
merged to 5 super-families, with each family assigned to the cluster its claims most often land
in — the grouping is derived, only the names are authored. Families clearing the 30-case floor:
**2 → 3** (PrivateTitlePossession 152, StateAction 59, ContractInstrument 56; TenureOccupancy 20
and a 3-case residual still below).

Two problems this surfaced, both recorded rather than papered over:
- **Defence families were routed to claims** by the extractor: `Procedural/Forum` (14 cases),
  `Statutory/Regulatory Subservience` (9), `Public Interest/Planning Policy` (4), `Equitable` (2).
  These are grounds for resisting a claim, not claims. **Excluded** from the merge, not folded in,
  so the routing error stays visible.
- **Non-property leakage survives `screen.py`**: `Criminal Breach of Trust`, `ElectionDispute`,
  `Adoption`, `Insolvency`, `Taxation`, `RentControl` each appear once or twice. The merge absorbs
  them into super-families, which is sloppy at 1 case each but should be fixed at the screen.

**The merge is not frozen.** It is derived from 300 cases. Re-run after extraction scales; a
family below the floor now may clear it later.

### Element burden metadata: LLM drafting failed, and the gate now catches it

§8.3 and §10.1's `E` feature group read `burden_on` / `burden_standard` /
`burden_shifts_when`. The source ontology requires all three and supplies them for **0 of 66**
elements; 6 of 15 claims carry a claim-level default in a notes line. So they had to be produced.

**First attempt (`llama4-scout-17b`) produced an annotation that looked complete and carried almost
no information:**

- `burden_on` **92% `claimant`** (only 2 of 3 valid values ever used)
- `clear_proof_required` used **zero times** — including for adverse possession, which the prompt
  names explicitly and which is *the* textbook heightened standard in Indian law
- only **2 distinct confidence values** (0.8, 0.7); nothing below 0.6 despite the prompt asking
  for contested allocations to be flagged
- `burden_shifts_when` null for all four adverse-possession elements; 14 distinct values over 66
- claim_12 (mortgage), whose ontology note says the burden *depends on the relief sought*, came
  out uniformly `claimant`

This is the dangerous shape of failure: **nothing downstream would question it.** A constant
masquerading as legal metadata would silently become an `E` feature.

**Action:** `degeneracy_report()` now **gates the write**. It refuses when any single value exceeds
85%, when fewer than 3 distinct standards appear, when fewer than 3 confidence values appear, when
`burden_shifts_when` has under 0.25 distinct values per element, or when `clear_proof_required` is
never reached for adverse possession / cancellation-for-fraud / benami. `--force` overrides but
still tags `llm_draft`. The gate correctly rejects the scout annotation on 4 of those counts.

Retrying with `glm-5-3`, which scored best on instruction discipline in the earlier model
benchmark (CLAUDE.md: 16/24 and 9 rule keys vs scout's 16/8) — at roughly 12× the latency.

**Provenance is mandatory on every value**: `ontology_notes` (the document said so),
`llm_draft` (**not reviewed** — any result resting on it must be reported as such), `human`
(a reviewer confirmed it; only ever set by hand).

### §8 built: FP-Growth with the controls that make patterns mean something

`src/patterns/transactions.py` and `src/patterns/mine.py`. Four controls, each guarding a specific
failure:

1. **Outcome rules are fitted on the TRAINING split only.** A pattern→outcome rule fitted on test
   cases is a leak dressed as a feature, and it would be invisible in the final accuracy because
   the rule carries the answer. Family membership and support use all cases; anything touching the
   label is train-restricted.
2. **Benjamini–Hochberg on the chi-square p-values.** Mining thousands of itemsets and keeping
   those at p<0.05 manufactures ~5% of that count as discoveries.
3. **Bootstrap stability**, 20 resamples, ≥80% — a pattern present in one resample and absent in
   the next describes the sample, not the law. The distribution is reported so the threshold is a
   choice, not a hidden default.
4. **Closed itemsets only** — {possession, hostile} and {possession} at identical support say one
   thing; reporting both inflates every count.

Base rates are **per family, never global**: with outcome drift running 39%→64% across the temporal
split, a global base rate would make a pattern look discriminative purely because its family skews
late.

`transactions.py` also encodes two §8.1 requirements that are easy to get wrong: **support is per
CASE** (so transactions are sets — an atom appearing 20 times in one verbose judgment has support
1, not 20), and atoms carry an `@p`/`@d` view suffix so a rule can say "the *plaintiff* asserted
continuous possession" rather than "continuous possession appears somewhere". §10.2 ablation 3
turns the suffix off to measure what orientation is worth.

Statistics verified against hand-built cases: BH keeps exactly the 3 true positives out of 10 and
0 of 20 nulls; chi-square gives p=4.9e-06 on a strong 2×2 and p=1.000 on independence; closed-set
filtering drops the redundant subset.

### Claims extraction numbers (n=300)

- 300/300 calls ok, **0 dropped**, 0 unparseable
- claims: 594 proposed → **465 kept** (78.3%); 129 discarded for an unlocatable quote
- defences: 654 proposed → **522 kept** (79.8%); 132 discarded
- median 2 claims/case; 32 of 300 cases yielded none
- 15.5% of claim families fell outside the 15-item list

### Open items added

- [ ] `screen.py` misses non-property leakage that reaches the family level (Insolvency,
      ElectionDispute, Criminal Breach of Trust, Adoption, Taxation, RentControl). Low volume but
      it should be caught at the screen, not absorbed by a merge.
- [ ] `claims.py` lets DEFENCE families be assigned to claims (15.5% out-of-list overall). Validate
      against the right list per `kind` and re-route rather than accept.
- [ ] Only 3 merged families clear the 30-case floor at n=300 cases. §8 per-family mining needs
      more cases, or a coarser merge, or both.
- [ ] Burden metadata is `llm_draft` at best. §5.1-style human review needed before any result
      leans on the `E` feature group.

---

## 2026-09-30 — Session 1 continued: Stages 6–10 end to end, and where the signal dies

The pipeline now runs end to end. The result is negative for the structured approach and the
negative is **well-localised**, which makes it useful rather than just disappointing.

### Scale reached

| artifact | count |
|---|---|
| facts extracted (§6.1) | **13,919** over 795 cases · 84.7% of proposals kept · **0 dropped calls** |
| canonical facts (§6.2) | 13,919 labelled · 77.3% mapped · **22.7% out of vocabulary** |
| claims / defences (§7.1) | 1,408 claims + 1,563 defences over 862 cases (21 dropped calls, recoverable) |
| transactions (§8.1) | 794 cases across 5 merged families |
| patterns mined (§8.2) | 351 closed itemsets over 5 families |

### The headline: canonicalisation destroys the signal that extraction preserves

`src/eval/representation_ladder.py`. Same classifier, same split, same labels, and the **same
557 train / 143 test cases** at every rung — so a drop can only come from the representation.

| rung | representation | features | AUROC | ECE |
|---|---|---|---|---|
| 0 | majority | — | — | — |
| 1 | `masked_text` | 47,720 | **0.578** | 0.064 |
| 2 | extracted fact text (§6.1) | 9,638 | **0.577** | 0.060 |
| 3 | canonical atoms (§6.2) | 348 | **0.502** | 0.228 |

- **Extraction preserves the signal** — 0.578 → 0.577 while cutting features 5×. §6.1 works.
- **Canonicalisation destroys it** — 0.577 → 0.502, which is chance. §6.2's controlled vocabulary
  cannot carry outcome-relevant information even though the free-text facts it replaces can.

This one diagnostic explains §8's null result downstream, and it says the problem is the
**vocabulary**, not the extractor and not the law.

### §8: no fact pattern is associated with outcome

FP-Growth per merged family on `forum_heldout`, outcome rules fitted on train only:

| family | cases | closed itemsets | stable (≥80% of 20 bootstraps) | **BH-significant** |
|---|---|---|---|---|
| StateAction | 381 | 193 | 135 | **0** |
| PrivateTitlePossession | 224 | 52 | 36 | **0** |
| ContractInstrument | 152 | 49 | 32 | **0** |
| _UNASSIGNED | 113 | 25 | 13 | **0** |
| Mixed_Limitation_Procedure | 39 | 39 | 18 | **0** |

**351 patterns tested · 10 pass uncorrected p<0.05 · 17.6 expected by chance alone · 0 survive
Benjamini–Hochberg.** Fewer nominal "discoveries" than noise would produce, so this is not a weak
effect that needs more data to confirm — there is nothing there.

Patterns are *stable* (135/193 for StateAction) and still carry no outcome information. Stability
and significance are independent properties, and reporting the first without the second is how a
pattern library gets mistaken for a finding.

### §10: the structured pipeline is at chance; one procedural feature beats it

`forum_heldout`, 143 test cases with features (the limiter is fact coverage, not the split).

| system | acc | macro-F1 | AUROC | ECE |
|---|---|---|---|---|
| majority (global) | 0.566 | 0.362 | — | — |
| majority (per decade) | 0.552 | 0.524 | — | — |
| lr `F` (canonical facts) | 0.455 | 0.452 | 0.476 | 0.389 |
| lr `F+P+E+C` | 0.489 | 0.489 | 0.476 | 0.435 |
| lr `F+P+E+C+Q` | 0.489 | 0.489 | 0.484 | 0.423 |
| lr `C only` | 0.524 | 0.524 | 0.542 | 0.122 |
| **lr `Q only`** (prior court) | **0.629** | **0.619** | **0.686** | **0.068** |
| gbm `F+P+E+C` | 0.538 | 0.528 | 0.487 | 0.315 |
| gbm `Q only` | 0.629 | 0.619 | 0.686 | 0.073 |

McNemar: adding `Q` to the structured features changes nothing (lr p=1.00, gbm p=0.27), and
`F+P+E+C` vs `F` is not significant either (p=0.30 / 0.77). The structured features are too noisy
to combine with anything.

**`Q only` is the best model and the best calibrated.** Note this partly walks back the earlier
"the shortcut is weak" finding, and the reconciliation matters:

- as **text** through TF-IDF, `prior_court_text` gave AUROC 0.581 (902 test cases)
- as a **clean categorical feature**, `Q` gives AUROC 0.686 (143 test cases)

Same information, different encoding and a different, smaller test set. Both numbers are real;
quote them with their n and their representation. The shortcut is stronger than the first
measurement suggested, and it is the strongest thing we have.

### Honest limits on these numbers

- **n=143 test cases.** Facts ran on 800 cases chosen by doc_id order, which is arbitrary with
  respect to the split, so only 143 of `forum_heldout`'s 902 test cases have features. Every CI
  here is wide (±0.09 on AUROC). Scaling extraction is the single highest-value next action.
- **Groups `P` and `E` are empty**, and the ablation table now says so rather than printing
  "F+P+E" as a separate row. `P` is empty because §8 found no BH-significant pattern — a result.
  `E` is empty because burden metadata is unfilled — a gap.
- Non-property contamination survives `screen.py` and reaches the family level
  (`Criminal Breach of Trust`, `ElectionDispute`, `CompassionateAppointment`, `Insurance`).

### §7 at larger scale: the negative result strengthens

Re-run on 1,408 claims / 805 cases (was 465 / 268):

| clusterer | k | NMI | ARI | purity | note |
|---|---|---|---|---|---|
| HDBSCAN | 4 | 0.099 | **−0.006** | 0.724 | 88.9% outliers |
| agglomerative | 6 | 0.270 | 0.092 | 0.318 | silhouette **0.022**, down from 0.035 |

And the extractor assigned **84** distinct families, up from 42 at half the scale — its
open-ended `NEW:` escape is **not converging**. The merge now yields 4 families over the 30-case
floor (StateAction 424, PrivateTitlePossession 256, ContractInstrument 170,
Mixed_Limitation_Procedure 47), but it absorbs some clearly wrong members, so it is doing more
work than is fully justified. Recorded, not hidden.

### The canonicaliser's accounting fix, vindicated at scale

Out of 3,155 out-of-vocabulary labels, **the model flagged only 75 (2.4%)**. The other 97.6% were
caught solely by checking its answers against the vocabulary. Had we trusted the `NEW:` flag, the
measured rate would have read 0.5% — comfortably under §6.2's 5% freeze threshold — and we would
have frozen a vocabulary that cannot name **a fifth of the facts**, then spent the rest of the
project wondering why no pattern reached significance.

### What this means for the project

The defensible results are now:

1. **A leakage-controlled benchmark** with a sensitivity-checked probe: `order_only` 0.983,
   `masked` 0.655, chance 0.500 (902 test cases).
2. **A precisely localised negative result**: extraction preserves signal, canonicalisation
   destroys it, and that is why no pattern reaches significance.
3. **Multiple-testing discipline changing a conclusion**: 10 nominal hits against 17.6 expected by
   chance. Without BH this would have been a pattern library.
4. **Zero-shot LLM-0 below TF-IDF** on honest inputs (0.552 vs 0.655) and badly miscalibrated.

The next experiment is determined by result 2: the vocabulary is the bottleneck, so §6.2's
refinement loop (`src/extract/vocab_refine.py`, built and not yet run) is the highest-value action,
followed by scaling extraction so n is not 143.

### Open items added

- [ ] **Run `vocab_refine.py`** — 22.7% out-of-vocabulary, and the ladder says the vocabulary is
      the bottleneck. This is the critical path now.
- [ ] Scale fact extraction to cover the test sets properly (n=143 is the binding constraint).
- [ ] `glm-5-3` burden annotation ran 2h+ for 66 elements (71 calls with retries) and is still
      going. It is ~12× scout's latency and this task is sequential — parallelise it or accept a
      faster model plus the degeneracy gate.
- [ ] The `Q`-as-text vs `Q`-as-categorical gap (0.581 vs 0.686) deserves its own measurement on
      one case set; right now it is confounded with n.
