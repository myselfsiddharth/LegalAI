# M5 — Judgment prediction, ablations, and where the signal is lost (§10)

**Status: INTERIM.** Numbers are from 143 test cases; extraction is scaling test coverage to
~900 and these will be refreshed. Reproduce: `make stage6` and
`python -m src.eval.representation_ladder`.

**Headline: the structured pipeline predicts outcome at chance, and the loss is localised to one
stage.** That localisation is what makes this a finding rather than a dead end.

## 1. The representation ladder — the most important result so far

Same classifier, same split, same labels, and the **same 557 train / 143 test cases** at every rung,
so a drop can only be attributed to the representation change.

| rung | representation | features | AUROC | ECE |
|---|---|---|---|---|
| 0 | majority | — | — | — |
| 1 | `masked_text` (§5.2) | 47,720 | **0.578** | 0.064 |
| 2 | extracted fact text (§6.1) | 9,638 | **0.577** | 0.060 |
| 3 | canonical atoms (§6.2) | 348 | **0.502** | 0.228 |

- **Extraction preserves the signal** — 0.578 → 0.577 while cutting features 5×. §6.1 works.
- **Canonicalisation destroys it** — 0.577 → 0.502, which is chance.

§6.2's controlled vocabulary cannot carry outcome-relevant information that the free-text facts it
replaces can. This single diagnostic explains §8's null result (FP-Growth mines these atoms) and
tells us the bottleneck is the **vocabulary**, not the extractor and not the law.

## 2. Outcome models and §10.2 ablations

`forum_heldout`, 143 test cases with features, test WIN rate 0.434.

| system | acc | macro-F1 | AUROC | ECE | predicted WIN |
|---|---|---|---|---|---|
| majority (global) | 0.566 | 0.362 | — | — | 0.00 |
| majority (per decade) | 0.552 | 0.524 | — | — | 0.32 |
| lr `F` (canonical facts) | 0.455 | 0.452 | 0.476 | 0.389 | 0.50 |
| lr `F+P+E+C` | 0.489 | 0.489 | 0.476 | 0.435 | 0.55 |
| lr `F+P+E+C+Q` | 0.489 | 0.489 | 0.484 | 0.423 | 0.57 |
| lr `C only` | 0.524 | 0.524 | 0.542 | 0.122 | 0.55 |
| **lr `Q only`** | **0.629** | **0.619** | **0.686** | **0.068** | 0.40 |
| gbm `F` | 0.517 | 0.503 | 0.484 | 0.250 | 0.40 |
| gbm `F+P+E+C` | 0.538 | 0.528 | 0.487 | 0.315 | 0.42 |
| gbm `Q only` | 0.629 | 0.619 | 0.686 | 0.073 | 0.40 |

Feature groups: `F` facts · `P` patterns · `E` elements · `C` context · `Q` prior-court disposition.

**Every structured arm sits at chance** (AUROC 0.476–0.516). **`Q` alone — one categorical feature —
is the best model and the best calibrated.**

Paired tests (McNemar, exact binomial):

| comparison | discordant | p |
|---|---|---|
| lr `F+P+E+C+Q` vs `F+P+E+C` | 4 | 1.00 |
| lr `F+P+E+C` vs `F` | 15 | 0.30 |
| gbm `F+P+E+C+Q` vs `F+P+E+C` | 13 | 0.27 |
| gbm `F+P+E+C` vs `F` | 45 | 0.77 |

Adding `Q` to the structured features changes nothing — the structured features are too noisy to
combine with anything.

### 2.1 Groups `P` and `E` are empty, and the table says so

`P` is empty because §8 found **no** pattern surviving both stability and BH correction — a result.
`E` is empty because no element carries burden metadata (BLOCKER B2) — a gap.

The first version of this table printed `F`, `F+P` and `F+P+E` with byte-identical numbers, which
would have read as "patterns and elements add nothing": right by accident, wrong in reasoning.
Ablation arms whose *added* group is empty are now suppressed and the emptiness is reported.

### 2.2 The prior-court shortcut, reconciled

Two measurements of the same information, and both are real:

| encoding | test cases | AUROC |
|---|---|---|
| `prior_court_text` via TF-IDF | 902 | 0.581 |
| `Q` one-hot categorical | 143 | **0.686** |

The clean categorical encoding is a much better representation of the same fact, and the second
measurement is on a smaller, different set. **Quote both with their n and encoding.** This partly
walks back M1's "the shortcut is weak" reading: it is weak as text, stronger as a feature, and
currently the strongest single thing in the pipeline.

Keeping `Q` in its own group — never folded into `C` — is what made this measurable. Burying a
shortcut inside a legitimate feature group is how a shortcut becomes invisible.

## 3. Leakage probes (§5.2), carried forward

| arm | forum AUROC | temporal AUROC |
|---|---|---|
| `order_only` (sensitivity control) | **0.983** | 0.973 |
| `full` (unmasked) | 0.827 | 0.744 |
| **`masked`** | **0.655** | 0.576 |
| `prior_court` | 0.581 | 0.518 |
| chance | 0.500 | 0.500 |

`LLM-0` zero-shot on the same masked text: accuracy 62.0%, **AUROC 0.552**, and it predicts WIN 13%
of the time where the truth is 43%.

**A zero-shot LLM is worse than TF-IDF+LR on honest inputs (0.552 vs 0.655) and badly
miscalibrated.** §0.1's "the LLM is always the baseline" does not mean it is the *strong* baseline.

## 4. Honest limits

- **n = 143 test cases**, so every CI is roughly ±0.09 on AUROC. Cause: extraction took the first N
  cases by doc_id, arbitrary with respect to the splits. Fixed by `--priority eval-first`; coverage
  is currently 775/902 and rising.
- **`masked` reads 0.655 on 902 cases but 0.578 on this 143-case subset.** The subset is harder.
  Comparisons must use one case set — which is why the ladder re-measures rung 1 rather than quoting
  0.655.
- The `E` group's element features use a crude subcategory-name match as a declared stand-in for
  §8.3's reviewed pattern→element edges.
- Temporal-split results carry three confounds (outcome drift 39%→64%, HEADNOTE 79.5%→0.0%, and
  era-dependent fact density and attribution). Read them against the era-balanced forum split.

## 5. §10 acceptance

| requirement | status |
|---|---|
| ablations F → F+P → F+P+E → +S → +R → +C | **partial** — P and E empty; S and R not built (§9) |
| LLM baselines `LLM-0` … `LLM-CF+A` | **partial** — `LLM-0` run; FS/CoT/CF/CF+A not built |
| TF-IDF + LR on masked text | **met** |
| majority and per-family majority | **met** — per decade, which matters given the drift |
| macro-F1, accuracy, AUROC, ECE + reliability | **met** |
| bootstrap 95% CIs | **met** — 1,000 resamples |
| paired significance vs `LLM-0` | **partial** — McNemar implemented and used between arms |
| per-family and per-court breakdown | **met** in code; groups under 10 cases suppressed |
| trace (§11) | **not built** |
| error analysis (§10.3) | **not built** |

## 6. Next, in the order the evidence dictates

1. **Repair §6.2's vocabulary (B3).** The ladder says this is where the signal dies. Every other §10
   improvement is downstream of it.
2. **Finish scaling coverage** so n stops being 143.
3. Only then re-run §8 and §10. Re-mining before the vocabulary is fixed would re-confirm the null.
4. §9 (statutes, precedents) supplies the missing `S` and `R` groups; `scripts/retrieval_lib.py` and
   the 8,467-edge signed citation graph already exist and are time-respecting.
