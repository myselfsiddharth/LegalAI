# M1 — Labels, masking, splits, leakage probe

Stage 1 (§5) acceptance report. All numbers measured on this repo's data, 2026-09-30.
Reproduce with: `make stage0 stage1` (see Makefile).

## Headline

**The mask holds, and the probe is sharp enough to prove it.** A TF-IDF + logistic
regression probe scores AUROC **0.956** when shown the operative order alone, and **0.674**
on `masked_text`, against **0.788** on the unmasked judgment. So the probe can detect the
outcome easily when it is present, and masking removes most of it.

**The "procedural shortcut" is real but small — this corrects the hypothesis we started
with.** We expected prior-court disposition to be a dominant shortcut, because 19.2% of
outcome cues sit in the first 30% of a judgment and an apex court affirms more often than it
reverses. Measured, the recital alone reaches AUROC **0.590** (forum split) and **0.498**
(temporal split, i.e. chance). Knowing what the High Court did barely helps. The plausible
reason is selection: the Supreme Court grants leave precisely in contestable cases, and the
recital distribution is nearly balanced (26.1% below_dismissed vs 23.1% below_allowed). The
shortcut ablation stays in the design as a control, but it is not the story.

## 1. Data

| | cases |
|---|---|
| land/property cases in the spreadsheet | 6,954 |
| resolved to a PDF on disk (exact `filename` join) | **6,954 (100%)** |
| judgment text extracted | 6,954 (0 failures, 0 near-empty) |
| genuinely property disputes (`screen.py`) | 6,243 (89.8%) |
| binary WIN/LOSE label from rules | 4,074 (58.6%) |
| **eligible for outcome experiments** | **3,585** |

Eligibility = binary label ∧ ≥500 chars of masked text ∧ property domain.

**Domain contamination is real and is tagged, not deleted.** The spreadsheet's filter is
keyword-based, so it admits cases that merely *happen* on land: 5.5% are criminal appeals
(e.g. a 2005 murder appeal arising from an affray over paddy-cutting, which mentions
agricultural land throughout), 3.9% are tax-heavy, 0.9% criminal/property mixed. Each
experiment states the subset it used.

## 2. Outcome labels (§5.1)

Rules over the operative order region, 6,954 cases:

| label | n | share |
|---|---|---|
| WIN | 1,656 | 23.8% |
| LOSE | 2,418 | 34.8% |
| PARTIAL | 286 | 4.1% |
| REMAND | 192 | 2.8% |
| OTHER | 106 | 1.5% |
| UNKNOWN (→ LLM pass) | 2,296 | 33.0% |

Binary WIN/LOSE = 4,074, balance **40.6% WIN**. Ambiguous (both cues fired) 3.0%.

Every label carries the matched span and sentence, so it is auditable and §5.2 can confirm
the mask removed its own evidence.

### 2.1 Audit findings

Reading the evidence spans caught two of **my own** detectors being wrong, not the labels:
the LOSE "negation" flags were all `no order as to costs` trailing the verb, and in every
REMAND flagged for naming a lower court, that court was the *destination* of the remand, not
the actor. Those labels were correct.

Genuine bugs found and fixed: a negator governing the disposition verb
("we do not think it proper to again remand the matter" — a *refusal* to remand), and
"allowed to that extent" being scored WIN rather than PARTIAL. Both now have unit tests.

### 2.2 LLM cross-check and the quote-grounding gate

The LLM labeller must quote the operative sentence **verbatim**, and the quote is checked
against the judgment. On a 60-case pilot the initial grounding rate was 58.3%, and
diagnosing the failures found **both rejection causes were the verifier's bugs, not the
model's**:

* **PDF hyphenation.** Judgments before ~1980 are typeset with hyphenated line wraps that
  survive extraction as `execut- ing`, `plain- tiff's`, `sanc- tioned`. A model quoting the
  sentence silently repairs them — the correct reading of the page, but not a string match.
  9 of 25 rejections.
* **A length floor that rejected real evidence.** `Appeal dismissed.` is the actual
  operative line in pre-1970 law-report style. 16 of 25 rejections were quotes under the
  20-char floor, and **15 of those 16 were verbatim present**.

After repairing hyphenation on both sides and replacing the length floor with a requirement
that the quote *state a disposition*: grounding **58.3% → 91.7%**, and rules-vs-LLM
agreement **91.7%** where both decided and the quote verifies (n=36 pilot). The full
6,954-case run is in progress; final agreement and the §5.1 human check of 200 follow in M2.

## 3. Masking (§5.2)

No section headers exist to segment on — `ORDER` appears as a header in 3.7% of judgments,
`FACTS` in 29.7%. Segmentation is therefore three-tier: numbered paragraphs where the
numbering is present and ascending, blank-line blocks otherwise, and **sentence windows** as
the floor.

That third tier is not a nicety. Without it, extraction's lack of blank lines meant 57.7% of
cases segmented to a **single unit**; a unit takes a single role, so one reasoning phrase
anywhere discarded the whole judgment, and 57% of cases emerged with under 500 usable
characters. With it: median 47 units per case, minimum 6, and **2.6%** unusable.

Result over 6,954 cases: median **32.6%** of characters retained.

| unit role | share | kept? |
|---|---|---|
| facts | 26.6% | keep |
| unclear | 27.6% | drop — unclear is excluded by design |
| caption | 18.1% | drop |
| order | 7.9% | drop |
| headnote | 7.8% | drop |
| arguments | 5.8% | keep |
| analysis | 3.5% | drop |
| pleadings | 2.6% | keep |

HEADNOTE dropped whole in 49.2% of cases (it states the outcome *and* lists the
authorities). Prior-court disposition separated into its own channel in 55.5%.

## 4. Splits (§5.3)

`Data/splits/*.json`, frozen with a content hash. Party groups straddling splits: **0** for
both, asserted in code.

| split | hash | train | dev | test | train WIN | test WIN |
|---|---|---|---|---|---|---|
| `temporal_2005_2013` | `935046044db351f3` | 2,650 | 320 | 615 | 34.0% | **60.2%** |
| `forum_heldout` | `6356f9655d9cde84` | 2,540 | 348 | 697 | 40.6% | 42.6% |

### 4.1 Two confounds, reported rather than masked away

**Base-rate drift is large.** The temporal split's WIN rate goes **34.0% → 60.2%** between
train and test — a 26-point inversion. This is in the data, not the labeller: the raw corpus
shows `allowed` cues at 23.6% pre-2000 vs 41.0% post-2000, and `dismissed` at 39.0% vs
28.9%. The Supreme Court's allowance rate in land disputes rose sharply over the period.
Any temporal-split model inherits a prior that is wrong at test time, which is most of why
the masked probe scores 0.544 there against 0.674 on the forum split. Report accuracy
against a *per-period* majority baseline, never a global one.

**§5.3's court-held-out split is not available.** This corpus is Supreme Court only — there
is no second court to hold out. The substitute holds out by **originating High Court**
(recovered for 78.9% of cases; held out: Calcutta, Delhi, Madras). It is a weaker control
than the spec intends and is labelled as such. It is also era-balanced (64.8% vs 66.0%
pre-2000), which is why it isolates format shift from temporal drift: HEADNOTE presence goes
79.5% pre-2000 to 0.0% post-2000, so the temporal split separates two structurally different
document types as well as two eras.

## 5. Leakage probe (§5.2) — the acceptance gate

TF-IDF(1,2) + LR, `class_weight=balanced`, 1,000-sample bootstrap CIs.

| arm | temporal macro-F1 | temporal AUROC | forum macro-F1 | forum AUROC |
|---|---|---|---|---|
| majority | 0.285 | 0.500 | 0.365 | 0.500 |
| `prior_court` (recital only) | 0.490 | **0.498** | 0.548 | **0.590** |
| **`masked`** | 0.501 | **0.544** | 0.610 | **0.674** |
| `full` (unmasked) | 0.592 | 0.695 | 0.690 | 0.788 |
| `order_only` (sensitivity control) | 0.878 | **0.928** | 0.897 | **0.956** |

Read AUROC, not macro-F1: with balanced class weights a model that merely emits both classes
beats the majority baseline's macro-F1 without carrying any signal, which is exactly what
`prior_court` does on the temporal split (macro-F1 0.490 but AUROC 0.498 = chance).

**Conclusions.**
1. The probe is sensitive — AUROC 0.956 given the order region alone. A low `masked` score is
   therefore evidence about the mask, not about the probe.
2. Masking removes most outcome signal: 0.788 → 0.674 (forum), 0.695 → 0.544 (temporal).
3. Residual signal in the facts is real but modest (0.674 forum). That is the honest headroom
   any P3 model is competing for — **not** the ~0.9 figures that leakier legal-judgment-
   prediction setups report.
4. The prior-court shortcut is weak here (0.590 / 0.498), contrary to our hypothesis.

## 6. Open items

- Full-corpus LLM label agreement + §5.1 human check of 200 (in progress).
- Merge rules + LLM labels; recover the 2,296 rules-UNKNOWN cases, which should lift the
  eligible set well above 3,585.
- `unclear` is 27.6% of units and is discarded. An LLM section classifier over just those
  units is the cheapest way to recover facts text (§5.2 allows it).
- Re-run the probe on merged labels.
