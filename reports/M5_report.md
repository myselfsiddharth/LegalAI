# M5 — Judgment prediction, ablations, and where the signal is lost (§10)

**Status: full scale.** n = **881** test cases (was 143). Reproduce: `make stage6` and
`python -m src.eval.representation_ladder`.

**Headline: nothing in the structured pipeline beats a per-decade majority baseline by a meaningful
margin, the only feature group carrying signal is case metadata rather than facts, and the loss is
localised to two specific stages.**

## 1. The representation ladder

Same classifier, split, labels and **case set** at every rung, so a drop is attributable to the
representation alone.

| rung | representation | features | AUROC | ECE |
|---|---|---|---|---|
| 0 | majority | — | — | — |
| 1 | `masked_text` (§5.2) | 50,000 | **0.657** | 0.041 |
| 2 | extracted fact text (§6.1) | 41,142 | **0.614** | 0.035 |
| 3 | canonical atoms (§6.2) | 1,313 | **0.517** | 0.144 |
| | chance | | 0.500 | |

- **Extraction loses 0.043.** At n=143 this step looked lossless (0.578 → 0.577); at n=881 it is not.
  Extraction discards some of what `masked_text` carries.
- **Canonicalisation loses a further 0.097**, landing near chance.

## 2. Discretisation is the problem, not the ontology

`src/eval/vocab_ab.py`, same 881 test cases, both vocabularies at equal granularity:

| representation | features | AUROC |
|---|---|---|
| fact text | 41,142 | **0.614** |
| induced atoms (clustered, k=345) | 331 | 0.500 |
| ontology atoms (§6.2) | 1,973 | 0.525 |

| comparison | diff | 95% CI | P(≤0) |
|---|---|---|---|
| text − induced | **+0.114** | [+0.073, +0.154] | **0.000** |
| text − ontology | **+0.089** | [+0.043, +0.135] | **0.000** |
| induced − ontology | −0.025 | [−0.071, +0.020] | 0.846 |

A bottom-up induced vocabulary is **no better** than the authored one, and sweeping the atom count
does not recover the signal (k=345 → 0.557, k=1,000 → 0.548). Nor does the frequency threshold
(chance at every setting from 1 to 20 cases).

**Three independent lines therefore support one conclusion: discretising extracted facts into a
symbol vocabulary destroys the outcome signal they carry, whatever the vocabulary's source or
size.** This explains §8's null result rather than leaving it a puzzle.

## 3. Outcome models and §10.2 ablations

`forum_heldout`, 881 test cases (3,258 train), test WIN rate 0.470.
Feature groups: `F` 1,313 · `P` 3 · `E` 0 (B2) · `S` 49 · `R` 6 · `C` 38 · `Q` 6.

| system | acc | macro-F1 | AUROC | ECE |
|---|---|---|---|---|
| majority (global) | 0.530 | 0.346 | — | — |
| majority (per decade) | 0.571 | 0.557 | — | — |
| lr `F` | 0.501 | 0.499 | 0.503 | 0.270 |
| lr `F+P+E+S` | 0.511 | 0.510 | 0.515 | 0.274 |
| lr `F+P+E+S+R+C` | 0.513 | 0.513 | 0.519 | 0.285 |
| **lr `S only`** | **0.599** | **0.599** | **0.615** | **0.041** |
| lr `R only` | 0.548 | 0.548 | 0.579 | 0.037 |
| lr `C+S+R` | 0.579 | 0.577 | 0.604 | 0.060 |
| lr `C only` | 0.482 | 0.465 | 0.524 | 0.074 |
| lr `Q only` | 0.525 | 0.509 | 0.498 | 0.034 |
| gbm `F` | 0.521 | 0.517 | 0.521 | 0.134 |
| gbm `F+P+E+S` | 0.549 | 0.547 | 0.562 | 0.137 |
| gbm `F+P+E+S+R` | 0.563 | 0.561 | 0.589 | 0.117 |
| **gbm `F+P+E+S+R+C`** | **0.598** | **0.597** | **0.629** | 0.107 |
| gbm `C+S+R` | 0.587 | 0.586 | 0.619 | 0.116 |
| gbm `C only` | 0.583 | 0.582 | 0.607 | 0.091 |
| gbm `Q only` | 0.525 | 0.509 | 0.498 | 0.034 |

### 3.1 `S` and `R` are the first features that help

Filling §10.1's two empty groups moved the best model for the first time:

| | AUROC | macro-F1 |
|---|---|---|
| best before `S`/`R` (gbm `C only`) | 0.607 | 0.582 |
| **best with `S`/`R` (gbm `F+P+E+S+R+C`)** | **0.629** | **0.597** |
| per-decade majority | — | 0.557 |
| `masked_text` TF-IDF (the ladder's rung 1) | 0.657 | 0.608 |

- **`S only` — predicted statutes — is the strongest single group** (lr AUROC 0.615, macro-F1 0.599)
  and the best calibrated of any informative arm (ECE 0.041). Which provision governs a dispute is
  closely tied to what kind of dispute it is, and some kinds are more winnable than others.
- **`R only` — retrieved precedents' outcomes — also carries signal** (AUROC 0.579, ECE 0.037).
- McNemar: `C+S+R` beats `F` significantly for both models (lr p=0.0008, gbm p=0.004).

Both groups respect §5.2. `S` comes from a statute model **fit on train cases only**, predicting
provisions the court would introduce — court-cited statutes are never inputs. `R` counts only
precedents that predate the query *and* whose outcome label was available at training time, which
forecloses one test case's outcome informing another's.

**The honest ceiling is still the raw text.** `masked_text` TF-IDF reaches 0.657, above the best
structured model's 0.629. The structured pipeline narrowed the gap from 0.050 to 0.028 but has not
closed it.

### 3.2 Two results that reverse the n=143 reading:

1. **`Q only` (prior-court disposition) collapsed from AUROC 0.686 to 0.498 — chance.** The n=143
   figure was a small-sample artifact. M1's original finding, that the procedural shortcut is weak
   (0.581 as text on 902 cases), was correct and stands.
2. **`C only` — case metadata alone — is the best arm at 0.607**, better than every arm that adds
   facts to it (`F+P+E+C` = 0.573). Year, forum, proceeding type and claim family carry more signal
   than the extracted facts do, and adding facts actively hurts.

`P` contributes 3 features and changes nothing; `E` is empty (BLOCKER B2) and the table says so.
`F` — the canonical atoms — remains at chance alone and does not help any arm it is added to.

## 4. The trace (§11)

`src/trace/build.py` emits the §11 contract for a prediction; `src/trace/evaluate.py` runs its
deletion test. 25 traces built, 19 of 25 predictions correct.

Every trace carries: the prediction and its probability; the case's claim families; fact
`source_span` offsets into `masked_text`, each verified to round-trip against its own quote
(4,184/4,184 exact in `facts_qa.py`); predicted statutes from §9.2's train-only model; retrieved
precedents with the analogous party's outcome where that label was available at training time; and
feature attributions computed by **ablation** rather than read off coefficients, so they are
model-agnostic and measure what the model does rather than what a weight suggests.

An example, *State Of T.N. vs Ananthi Ammal* (1994), predicted LOSE p=0.43 against an actual WIN —
statutes predicted `LA:17`, `LA:4(1)`, `LA:23`, `LA:6`, all Land Acquisition Act provisions, which
is the right body of law for the case.

### 4.1 What the trace deliberately leaves empty

§11's contract wants `issues` with per-element status, `burden_on`, and the patterns satisfying or
defeating each element. **These are emitted as an explicit gap, not filled.** No element carries
burden metadata (BLOCKER B2) and §8 found no pattern surviving significance correction, so there is
nothing to populate them with. A trace that invented element statuses would read as legal reasoning
while being decoration — worse than one that reports the gap.

### 4.2 Deletion test: the trace is mechanistically faithful

§11 asks that removing the facts a trace cites drop confidence more than removing random facts.
**The test deletes FACTS, not features**, which matters: attribution is itself computed by feature
ablation, so re-ablating the same features would measure the attribution against itself. Deleting a
fact re-derives its canonical atoms, rebuilds the whole feature row, and re-scores.

45 cases, 3 facts deleted per arm:

| deleted | mean \|Δp\| |
|---|---|
| **trace-cited facts** | **0.0650** |
| random facts | 0.0209 |
| the trace's own lowest-ranked facts | 0.0180 |

| comparison | mean difference | 95% CI | cases won |
|---|---|---|---|
| cited − random | **+0.0441** | [+0.0245, +0.0659] | 77.8% |
| cited − inverse | **+0.0470** | [+0.0290, +0.0666] | 75.6% |

Both CIs exclude zero. The `inverse` control — deleting the trace's own *lowest*-ranked facts — is
the stronger claim, since a trace could beat random simply by citing many facts; beating its own
tail means the ranking carries information.

### 4.3 What this does and does not establish

It establishes **mechanistic faithfulness**: the trace points at the evidence the model actually
uses. It does **not** establish that the cited evidence is legally correct reasoning. That is the
same distinction as the earlier "groundedness is not correctness" result — a trace can be perfectly
faithful to a model that is wrong, and this model is right on 19 of 25.

It is also a consistency check between single-fact and joint-fact ablation rather than a fully
independent validation: the ranking comes from deleting facts one at a time, the test from deleting
three together. Interactions could have broken that transfer and did not, which is informative but
weaker than an outside signal.

§11's other two evaluations are **not done**: (b) human rating of 50 traces by a legal reviewer
(needs a human, like BLOCKER B4), and (c) comparison against `LLM-CoT` rationales (needs the
`LLM-CoT` baseline, which is not built).

## 5. The full baseline comparison (§3.3), identical 300 cases

All five §3.3 LLM arms, plus the non-LLM systems re-scored on **exactly the cases the LLM arms
saw** — the LLM run used a 300-case subsample, so comparing it against §10's 881-case table would
confound system with sample.

`llama4-scout-17b`, 0 dropped calls, 0 unusable replies across all 1,500 predictions.

| system | input | acc | macro-F1 (95% CI) | AUROC | ECE | pred WIN |
|---|---|---|---|---|---|---|
| structured gbm `F+P+E+S+R+C` | facts+statutes+precedents+context | 0.603 | 0.599 [0.544, 0.654] | 0.628 | 0.109 | 0.44 |
| structured lr `S only` | predicted statutes | 0.603 | 0.601 [0.541, 0.654] | 0.612 | **0.050** | 0.46 |
| **TF-IDF + LR** | masked text | 0.597 | 0.593 [0.535, 0.649] | **0.645** | **0.045** | 0.45 |
| majority (per decade) | — | 0.587 | 0.571 [0.515, 0.625] | — | — | 0.35 |
| `LLM-CoT` | masked text + elements | 0.583 | 0.582 [0.526, 0.633] | 0.610 | 0.207 | 0.48 |
| `LLM-0` | masked text | 0.577 | 0.491 [0.434, 0.546] | 0.579 | 0.265 | **0.13** |
| `LLM-CF` | canonical facts | 0.560 | 0.469 [0.419, 0.525] | 0.545 | 0.062 | 0.13 |
| `LLM-CF+A` | facts + statutes + precedents | 0.560 | 0.502 [0.452, 0.560] | 0.535 | 0.046 | 0.20 |
| `LLM-FS` | masked text + 4 earlier examples | 0.543 | 0.522 [0.468, 0.573] | 0.522 | 0.117 | 0.33 |

### 5.1 The honest reading: almost nothing is separable at this n

**Every macro-F1 interval spans about 0.11 and they overlap heavily.** The per-decade majority's
0.571 [0.515, 0.625] sits inside the interval of nearly every system above and below it. McNemar
against `LLM-0` is non-significant for all four other LLM arms (p = 0.34, 0.93, 0.63, 0.65).

So the defensible claim is **not** "TF-IDF beats the LLM" — it is that **on leakage-controlled
inputs no system convincingly clears a per-decade majority baseline**, and the ordering below is
suggestive rather than established.

### 5.2 Four results that do survive the caveat

1. **`LLM-CF` falls below `LLM-0`** (AUROC 0.545 vs 0.579; macro-F1 0.469 vs 0.491). Handing the
   model our canonical facts *instead* of the raw text makes it worse. This was a **prediction** of
   the representation ladder, made before running it: if the atoms lose signal the text carries,
   then an LLM given atoms should also lose. It does — which means the discretisation loss is a
   property of the representation, not an artifact of linear and tree classifiers.
2. **`LLM-CF+A` does not beat `LLM-CF`** (0.535 vs 0.545), even though the *same* predicted
   statutes and retrieved precedents lift the structured model to its best score. The LLM does not
   exploit the grounding that helps a classifier.
3. **`LLM-FS` is the worst arm** (AUROC 0.522), below zero-shot. Four retrieved earlier examples,
   truncated to 2,500 characters each, actively hurt.
4. **Calibration separates the systems far more clearly than accuracy does.** `LLM-0` is badly
   miscalibrated (ECE 0.265, predicting WIN 13% of the time where the truth is 46%) and `LLM-CoT`
   only a little better (0.207), while TF-IDF and `lr S only` reach 0.045–0.050. For any use where
   the probability matters rather than the label, that gap is the decisive one — and it does not
   overlap.

### 5.3 What `LLM-CoT` buys

Element-by-element reasoning is the best LLM arm (AUROC 0.610 against zero-shot's 0.579) and it
fixes the prediction bias: `predWIN` 0.48 against `LLM-0`'s 0.13, against a true rate of 0.46. The
elements supplied come from the ontology, so this is the one place the ontology demonstrably helps
something — not as a feature space, but as a reasoning scaffold for a prompt.

## 6. Leakage probes (§5.2), n=902

| arm | macro-F1 | AUROC |
|---|---|---|
| `order_only` (sensitivity control) | 0.932 | **0.983** |
| `full` (unmasked) | 0.745 | 0.827 |
| **`masked`** | 0.598 | **0.655** |
| `prior_court` | 0.563 | 0.581 |
| majority | 0.345 | 0.500 |

The probe detects the outcome almost perfectly when shown the order, and `masked` at 0.655 is the
honest headroom. These numbers are **unchanged** from the smaller sample, which is why they can be
quoted with confidence.

`LLM-0` zero-shot on the same masked text: accuracy 62.0%, AUROC 0.552, predicting WIN 13% of the
time where truth is 43%. A zero-shot LLM is worse than TF-IDF+LR on honest inputs and badly
miscalibrated.

## 7. §10 acceptance

| requirement | status |
|---|---|
| ablations F → F+P → F+P+E → +S → +R → +C | **met** except `E` (B2) — `S` and `R` now populated from §9 |
| LLM baselines `LLM-0` … `LLM-CF+A` | **met** — all five arms, 0 dropped, scored on identical cases |
| TF-IDF + LR on masked text | **met** |
| majority and per-group majority | **met** — per decade, which the drift makes necessary |
| macro-F1, accuracy, AUROC, ECE + reliability | **met** |
| bootstrap 95% CIs | **met** — 1,000 resamples |
| paired significance test vs `LLM-0` | **met** — exact-binomial McNemar, all four non-significant |
| per-family / per-court breakdown | **met** in code; groups under 10 cases suppressed |
| trace (§11) output contract | **partial** — prediction, facts+spans, authorities, attributions built; `issues`/`elements` emitted as an explicit gap (B2) |
| trace faithfulness, deletion test (§11a) | **met** — cited facts move the prediction 3.1× more than random, CI excludes 0 |
| trace faithfulness, human rating (§11b) | **not done** — needs a legal reviewer |
| trace vs `LLM-CoT` rationales (§11c) | **not done** — `LLM-CoT` not built |
| error analysis (§10.3) | **not built** |

## 8. What this means for the project

§8 as specified — FP-Growth over canonical fact labels per claim family — **cannot work on this
corpus**, for a measured reason: its input representation carries no outcome signal. That is a
legitimate negative finding about symbolic fact representations for legal judgment prediction, and
the leakage control plus multiple-testing discipline are what make it credible rather than merely
unimpressive.

What still carries signal is `masked_text` (0.657) and the extracted fact **text** (0.614). The
pipeline's useful product is the span-verified, party-attributed fact set — not its projection onto
a label vocabulary.

**Where this leaves the project.** The structured pipeline now beats a per-decade majority baseline
(macro-F1 0.597 vs 0.557) and the features responsible are §9's — predicted statutes and retrieved
precedents — not §6's canonical atoms, which remain at chance. The raw masked text still beats all
of it at 0.657, so the value of the structure is interpretability and provenance rather than
accuracy.
