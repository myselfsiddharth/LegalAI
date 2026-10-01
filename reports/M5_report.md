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

## 4. Leakage probes (§5.2), n=902

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

## 5. §10 acceptance

| requirement | status |
|---|---|
| ablations F → F+P → F+P+E → +S → +R → +C | **met** except `E` (B2) — `S` and `R` now populated from §9 |
| LLM baselines `LLM-0` … `LLM-CF+A` | **partial** — `LLM-0` only |
| TF-IDF + LR on masked text | **met** |
| majority and per-group majority | **met** — per decade, which the drift makes necessary |
| macro-F1, accuracy, AUROC, ECE + reliability | **met** |
| bootstrap 95% CIs | **met** — 1,000 resamples |
| paired significance test | **met** — exact-binomial McNemar |
| per-family / per-court breakdown | **met** in code; groups under 10 cases suppressed |
| trace (§11) | **not built** |
| error analysis (§10.3) | **not built** |

## 6. What this means for the project

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
