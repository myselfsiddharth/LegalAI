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

| system | acc | macro-F1 | AUROC | ECE |
|---|---|---|---|---|
| majority (global) | 0.530 | 0.346 | — | — |
| **majority (per decade)** | **0.571** | **0.557** | — | — |
| lr `F` | 0.501 | 0.499 | 0.503 | 0.270 |
| lr `F+P+E+C` | 0.498 | 0.498 | 0.501 | 0.283 |
| lr `F+P+E+C+Q` | 0.512 | 0.512 | 0.505 | 0.268 |
| lr `Q only` | 0.525 | 0.509 | 0.498 | 0.034 |
| gbm `F` | 0.521 | 0.517 | 0.521 | 0.134 |
| gbm `F+P+E+C` | 0.566 | 0.565 | 0.573 | 0.140 |
| gbm `F+P+E+C+Q` | 0.568 | 0.566 | 0.590 | 0.116 |
| **gbm `C only`** | **0.583** | **0.582** | **0.607** | 0.091 |
| gbm `Q only` | 0.525 | 0.509 | 0.498 | 0.034 |

**Two results that reverse the n=143 reading:**

1. **`Q only` (prior-court disposition) collapsed from AUROC 0.686 to 0.498 — chance.** The n=143
   figure was a small-sample artifact. M1's original finding, that the procedural shortcut is weak
   (0.581 as text on 902 cases), was correct and stands.
2. **`C only` — case metadata alone — is the best arm at 0.607**, better than every arm that adds
   facts to it (`F+P+E+C` = 0.573). Year, forum, proceeding type and claim family carry more signal
   than the extracted facts do, and adding facts actively hurts.

`P` contributes 3 features and changes nothing; `E` is empty (BLOCKER B2) and the table says so.

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
| ablations F → F+P → F+P+E → +S → +R → +C | **partial** — E empty (B2); S, R await §9 |
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

Next: §9 supplies the missing `S` and `R` groups, and statute prediction is a task where the facts
may well carry signal even though the outcome does not.
