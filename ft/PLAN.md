# ft — does fine-tuning break the ceiling?

**Self-contained and additive.** Reads v1's `Data/interim/masked_text.jsonl`, labels and hashed
splits; writes only inside `ft/`. Nothing in `src/` or `v2/` is modified.

Read `../writeup/LexGraph_Report.tex` §15.6 for why this exists.

---

## 1. The one fair objection to the whole project

Every model in this project is **TF-IDF with logistic regression**, plus some zero-shot and few-shot
LLM calls. The published work we compare against used **fine-tuned transformer models with
hierarchical encoding**. So a reviewer can say:

> *"Your ceiling of 0.666 AUROC on facts-only is a ceiling on TF-IDF, not on the task. You never
> tried a real model."*

**That objection is correct and unanswered.** This folder answers it. The point is **not** to beat
the published 78% — §15.6 explains why doing that on the contaminated window would be worthless.
The point is to find out whether the facts-only ceiling survives a strong method.

## 2. Pre-registered readings — written before any training run

The bar is v1's best non-leaky arm on `forum_heldout`: **AUROC 0.666, accuracy 0.621** (TF-IDF on
masked text). Judged on the **same test cases**.

| fine-tuned AUROC on masked text | reading |
|---|---|
| **≤ 0.69** | **The ceiling claim is strengthened considerably.** A domain-pretrained transformer, fine-tuned, using all the text, gets no further than word counts. The wall is the task. |
| **0.69 – 0.75** | A real but modest gain. The ceiling softens; the 16-point gap to 78% narrows but does not close. Report the new number as the ceiling and revise §15.6's estimate. |
| **≥ 0.75** | **The ceiling claim is WRONG.** The facts carry much more than this project could extract, and the headline conclusions about diffuse signal and the 0.043 headroom must be re-examined. This would be the most important result in the project and is worth discovering before publication rather than after. |

No outcome is a failure. The middle band is the honest prior.

## 3. The capability control — the arm that makes the result interpretable

A weak result on masked text has two possible causes, and they are indistinguishable without a
control:

1. the task has no more signal, or
2. **our fine-tuning is simply bad** — wrong hyper-parameters, too few epochs, a bug.

So the identical pipeline also trains on **`last_512_tokens`**, the contaminated window. v1 measured
TF-IDF at 0.845 there. If fine-tuning reaches ~0.88 on that arm while stalling at ~0.68 on masked
text, the model class is demonstrably capable and the masked ceiling is a property of the data. If
it stalls on *both*, the training is broken and no conclusion may be drawn.

**This arm is leaky by construction and exists only as a capability check. Its number is never
reported as a result.**

## 4. Design

**Model.** `law-ai/InLegalBERT` — pretrained on Indian legal documents (IIT Kharagpur), so it is
domain-matched rather than a generic English encoder. `distilbert-base-uncased` as a faster
secondary, to show the result is not one model's quirk.

**The length problem.** Masked text runs ~2,500 words, well past BERT's 512-token limit. Three
options: truncate (throws most away), take the last 512 (the contaminated region on unmasked text,
and arbitrary here), or **chunk and pool**. We chunk into non-overlapping 512-token windows, train on
each chunk carrying its case's label, and at inference **average the chunk probabilities per case**.
This uses all the text and mirrors what the published hierarchical approach does. A truncate-to-first-512
arm is reported alongside as a cheap reference.

Training on chunks is weak supervision — many chunks will not contain whatever signal exists. That is
expected and is also true of the published approach.

**Honesty constraints.**
- Same hashed `forum_heldout` split as every other number in the project; the test set is never
  touched for model selection.
- Early stopping on the **dev** split only.
- Chunk-level and case-level metrics both reported, since chunk accuracy flatters.
- A single seed is reported with its own variance across seeds, because one fine-tuning run is a
  noisy estimate and quoting one number as *the* result would misrepresent it.

**Compute.** Apple MPS, locally. Voyager is inference-only and cannot fine-tune.

## 5. Running

```
python -m ft.data                 # build and cache the chunked datasets
python -m ft.train --arm masked   # the headline arm
python -m ft.train --arm last512  # the capability control (leaky; never reported as a result)
```

Results land in `ft/experiments/`.
