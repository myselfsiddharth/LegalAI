# M1 — Labels, masking, splits, leakage probe

Stage 1 (§5) acceptance report. Measured on this repo's data, 2026-09-30.
Reproduce: `make stage0 stage1 labels-llm merge probe`.

## Headline

**The mask holds, and the probe is sharp enough to prove it.** Shown the operative order
alone, a TF-IDF + logistic regression probe reaches AUROC **0.983**. Shown `masked_text`, it
reaches **0.655**, against **0.827** on the unmasked judgment. The probe finds the outcome
trivially when it is present and mostly cannot after masking.

**Residual headroom is ~0.655 AUROC, not ~0.9.** That is what a P3 model is competing for on
honest inputs, and it is the number to state whenever this work is compared with
legal-judgment-prediction results that do not control leakage.

**The procedural shortcut is weak — this contradicts the hypothesis we began with.** We
expected prior-court disposition to dominate, since 19.2% of outcome cues sit in the first 30%
of a judgment and an apex court affirms more often than it reverses. Measured, the recital
alone reaches AUROC **0.581** (forum) and **0.518** (temporal). The likely reason is
selection: the Supreme Court grants leave precisely in contestable cases, and the recital is
near-balanced (26.1% below_dismissed vs 23.1% below_allowed). It stays in the design as an
ablation control, not as the story.

## 1. Data

| | cases |
|---|---|
| land/property cases in the spreadsheet | 6,954 |
| resolved to a PDF on disk (exact `filename` join) | **6,954 (100%)** |
| judgment text extracted | 6,954 (0 failures, 0 near-empty) |
| genuinely property disputes (`screen.py`) | 6,243 (89.8%) |
| binary WIN/LOSE label, rules only | 4,003 (57.6%) |
| binary WIN/LOSE label, rules + LLM merged | **5,329 (76.6%)**, 44.4% WIN |
| **eligible for outcome experiments** | **4,707** |

Eligibility = binary label ∧ ≥500 chars masked text ∧ property domain. Merging the LLM pass
grew this from 3,585 to 4,707 (+31%).

**Domain contamination is real, and is tagged rather than deleted.** The spreadsheet's filter
is keyword-based, so it admits cases that merely *happen* on land: 5.5% criminal appeals
(e.g. a 2005 murder appeal arising from an affray over paddy-cutting, which mentions
agricultural land throughout), 3.9% tax-heavy, 0.9% mixed. Every experiment states its subset.

## 2. Outcome labels (§5.1)

| label | n | share |
|---|---|---|
| WIN | 2,365 | 34.0% |
| LOSE | 2,964 | 42.6% |
| PARTIAL | 624 | 9.0% |
| REMAND | 116 | 1.7% |
| OTHER | 34 | 0.5% |
| UNRESOLVED (labellers conflict) | 562 | 8.1% |
| UNKNOWN | 289 | 4.2% |

Provenance: 3,715 both agree · 2,032 LLM-recovered from rules-UNKNOWN · 316 rules-only ·
562 conflict · 289 undecided. Every label carries its matched span, so it is auditable and
§5.2 can confirm the mask removed its own evidence.

**Rules-vs-LLM agreement: 87.0%** (n=4,317, both decided and quote verified). Direct
WIN↔LOSE contradiction is **86 cases ≈ 2.0%** — the usable label-noise estimate for the binary
task. The larger disagreement blocks are taxonomy boundaries, not contradictions:
WIN→PARTIAL 111, LOSE→PARTIAL 91, REMAND→WIN 77 (a remand after setting aside the order below
*is* a win for the appellant; §5.1 makes REMAND its own class, so the rules follow the
taxonomy). Conflicts are held as UNRESOLVED and queued in `label_conflicts.jsonl` rather than
broken by coin-flip, which would manufacture noise.

### 2.1 Three rules bugs found by auditing conflicts, each now unit-tested

1. **Consequential dismissals read as the disposition** — the largest single error. Indian
   orders routinely say *"the appeal is allowed … and the writ petition filed by the
   respondent stands dismissed"*; taking the last disposition gives LOSE when the appellant
   plainly won. Fixed by ranking the proceeding **before this court** (appeal/SLP) above the
   underlying one (suit/petition). On audit the LLM was right in all 10 sampled cases.
   Conflicts 196 → 107.
2. **Cross-verb binding** — `appeal … is dismissed` matched *across* "the appeal is allowed
   and the suit is dismissed", because the gap between noun and verb was unconstrained. Each
   proceeding noun now binds to its nearest disposition verb. Ambiguous cases 3.0% → 1.2%.
3. **Negation and partial coverage** — "we do not think it proper to again remand the matter"
   is a *refusal* to remand; "succeed in part" and "allowed to the extent indicated" are
   partial allowances scored as full wins.

Net effect: agreement 84.0% → **87.0%**.

Two things that looked like bugs were **my detectors**, not the labels: the LOSE "negation"
flags were all `no order as to costs` trailing the verb, and in every REMAND flagged for
naming a lower court, that court was the remand's *destination*, not the actor.

### 2.2 Quote grounding

The LLM must quote the operative sentence verbatim, and the quote is checked against the
judgment; an unverifiable quote never carries a label, because the quote is the only evidence
offered. Final rate **91.3%** over 6,952 cases, **0 dropped calls**, 2 unparseable. 316 LLM
labels were refused for an unverifiable quote.

Getting there required fixing the verifier, not the model — the initial 58.3% rate was
**entirely my own two bugs**:

* **PDF hyphenation.** Pre-1980 typesetting wraps words as `execut- ing`, `plain- tiff's`,
  `sanc- tioned`. A model quoting the sentence silently repairs them, which is the correct
  reading of the page but not a string match. 9 of 25 rejections. Repair is now applied to
  both sides, so it cannot manufacture a match the words do not support.
* **A length floor that rejected real evidence.** `Appeal dismissed.` is the actual operative
  line in pre-1970 law-report style. 16 of 25 rejections were under the 20-char floor and
  **15 of those 16 were verbatim present**. The floor is replaced by a requirement that the
  quote *state a disposition*.

## 3. Masking (§5.2)

No section headers exist to segment on: `ORDER` appears as a header in 3.7% of judgments,
`FACTS` in 29.7%. Segmentation is three-tier — numbered paragraphs where the numbering is
present and ascending, blank-line blocks otherwise, **sentence windows** as the floor.

The third tier is load-bearing. Without it, extraction's lack of blank lines meant 57.7% of
cases segmented to a **single unit**; a unit takes one role, so a single reasoning phrase
discarded the whole judgment and 57% of cases emerged with under 500 usable characters. With
it: median 47 units per case, minimum 6, **2.6%** unusable, median **32.6%** of characters
retained.

| unit role | share | kept |
|---|---|---|
| facts | 26.6% | ✓ |
| unclear | 27.6% | ✗ — unclear is excluded by design |
| caption | 18.1% | ✗ |
| order | 7.9% | ✗ |
| headnote | 7.8% | ✗ |
| arguments | 5.8% | ✓ |
| analysis | 3.5% | ✗ |
| pleadings | 2.6% | ✓ |

HEADNOTE is dropped whole in 49.2% of cases — it states the outcome *and* lists the
authorities. Prior-court disposition is separated into its own channel in 55.5%.

## 4. Splits (§5.3)

Frozen with a content hash in `Data/splits/`. Party groups straddling splits: **0** for both,
asserted in code and in tests.

| split | hash | train | dev | test | train WIN | test WIN |
|---|---|---|---|---|---|---|
| `temporal_2004_2013` | `43d9d645667ad3a5` | 3,446 | 480 | 768 | 39.2% | **63.8%** |
| `forum_heldout` | `6e346dd75b7e7d84` | 3,350 | 455 | 902 | 44.2% | 47.2% |

### 4.1 Three confounds, reported rather than masked away

**Base-rate drift is large.** The temporal split's WIN rate runs **39.2% → 63.8%** between
train and test. This is in the data, not the labeller: raw `allowed` cues are 23.6% pre-2000
vs 41.0% post-2000, `dismissed` 39.0% vs 28.9%. The Court's allowance rate in land disputes
rose sharply. Any temporal model inherits a prior that is wrong at test time, which is most of
why the masked probe scores 0.576 there against 0.655 on the forum split. **Report accuracy
against a per-period majority baseline, never a global one.**

**§5.3's court-held-out split does not exist here.** The corpus is Supreme Court only, so
there is no second court to hold out. The substitute holds out by **originating High Court**
(recovered for 78.9% of cases; held out: Calcutta, Delhi, Madras). It is weaker than the spec
intends and is labelled as such.

**The temporal split carries a format shift.** HEADNOTE presence goes 79.5% pre-2000 to 0.0%
post-2000, so train and test are structurally different document types as well as different
eras. Masking removes the HEADNOTE but not the confound. The forum split is era-balanced
(66.7% vs 65.1% pre-2000) and exists precisely to separate format shift from temporal drift.

**Group-whole and time-respect cannot both hold** for a dispute litigated across a boundary.
Assigning a group by its latest year put 1970 cases in a "test > 2013" split; by its earliest
year it put 2025 cases in "train ≤ 2004", which is learning from the future. Both were caught
by `tests/test_stage1.py`. Straddling groups are now dropped — 13 cases, 0.3%.

## 5. Leakage probe (§5.2) — the acceptance gate

TF-IDF(1,2) + LR, `class_weight=balanced`, 1,000-sample bootstrap CIs.

| arm | temporal macro-F1 | temporal AUROC | forum macro-F1 | forum AUROC |
|---|---|---|---|---|
| majority | 0.266 | 0.500 | 0.345 | 0.500 |
| `prior_court` (recital only) | 0.492 | **0.518** | 0.563 | **0.581** |
| **`masked`** | 0.529 | **0.576** | 0.598 | **0.655** |
| `full` (unmasked) | 0.638 | 0.744 | 0.745 | 0.827 |
| `order_only` (sensitivity control) | 0.917 | **0.973** | 0.932 | **0.983** |

**Read AUROC, not macro-F1.** With balanced class weights, a model that merely emits both
classes beats the majority baseline's macro-F1 while carrying no signal — exactly what
`prior_court` does on the temporal split (macro-F1 0.492 but AUROC 0.518 ≈ chance).

The `order_only` arm is the reason the rest is interpretable: at AUROC 0.973–0.983 the probe
detects the outcome easily when it is present, so a low `masked` score is evidence about the
mask and not about a blind probe.

## 6. Next

- §5.1's human check of 200, drawn from the 562 UNRESOLVED conflicts (highest information per
  annotation) plus a random stratum for an unbiased estimate.
- `LLM-0` outcome-identification probe on `masked_text` — the second half of §5.2, not yet run.
- `unclear` is 27.6% of units and is discarded. An LLM section classifier over just those units
  is the cheapest way to recover facts text, and §5.2 permits it.
- Stage 2 (§6): fact extraction and the canonical vocabulary.
