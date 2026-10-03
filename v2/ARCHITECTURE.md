# v2 — the element-centric architecture

**This folder is self-contained and additive. Nothing here modifies v1.** The root `src/`,
`experiments/` and `writeup/` are untouched; v2 *imports* from `src/` for data access and model
calls rather than reimplementing them, because v1's measured components (grounding, masking,
labels, the LLM cache) are the parts that already work.

Read `../writeup/LexGraph_Report.tex` first for what v1 established. This file is the plan to
follow for v2, and it is updated as stages land.

---

## 1. Why v2 exists

v1 implemented a different pipeline than the one intended, measured it against the hardest possible
target, and reported the result as a verdict on the architecture. Two specific errors:

**Error 1 — step 2 was built in its lossiest form.** The intended design says *"fact family —
classification model run on all embeddings of the facts."* v1 instead hard-mapped each fact onto one
of 345 authored labels. **S2 is now built and measured** (`experiments/s2_soft_families_*.json`),
same 885 test cases, k=345 throughout:

| representation | AUROC | dims |
|---|---|---|
| masked text | 0.666 | 50,000 |
| **soft histogram, temp 0.25–0.5** | **0.633** | **345** |
| extracted fact text | 0.629 | 47,354 |
| hard argmax, same centroids | 0.593 | 345 |
| embedding mean | 0.558 | 4,096 |
| canonical atoms (v1's choice) | 0.517 | 345 |

**Step 2 done right loses essentially nothing: 0.633 from 345 dims, against 0.629 for the full
47,354-feature fact text it is derived from.** The representation is not the bottleneck.

### A correction, recorded because the first version of this file got it wrong

This file initially claimed "the argmax alone costs +0.084 AUROC". **That was an over-attribution.**
The +0.084 came from comparing a soft histogram (0.601) against v1's canonical atoms (0.517), which
differ in *three* ways at once — soft vs hard, corpus-induced centroids vs authored labels, and
histogram vs binary-presence encoding. Attributing the whole gap to the argmax was exactly the
single-cause mistake v1 kept making.

Controlled properly — identical centroids, identical encoding, only soft vs hard varying — the
argmax costs **+0.040, 95% CI [−0.0015, +0.0826]: marginally NOT significant.**

The real gain from 0.517 to 0.633 is **+0.116 from the combination**, and it is still the largest
single improvement available to the pipeline. It is just not one mechanism.

### The corrected decomposition

```
masked text                                   0.666
  → extraction: WHICH facts get selected      −0.033 → 0.633   ← the remaining bottleneck
  → representation, done softly at 345 dims    0.000           ← solved
  ----- v1's additional, avoidable losses -----
  → hard argmax instead of soft               −0.040 → 0.593   (not significant alone)
  → authored labels + binary encoding         −0.076 → 0.517
```

Two further findings: a uniform blur (temp=0) scores 0.512, confirming the representation carries
nothing structural by construction; and the optimum is a *broad* similarity profile — AUROC declines
monotonically as temperature sharpens toward the argmax.

**Error 2 — every layer was scored on win/lose.** v1 has strong evidence that target is near
unlearnable from pre-decision facts: masked text 0.656–0.723 across four splits; every structured
model at or below a per-decade majority baseline; model errors statistically *independent* of LLM
errors (46 both-wrong against 49.6 expected). Scoring the pattern and element layers against it was
a category error. **In v2 every layer is scored on its own task.**

The consequence of those two errors is visible in v1's model today:
`features_per_group: {F: 1445, S: 49, R: 6, C: 38, Q: 6}` — **no `P`, no `E`.** Both middle layers
of the intended design contribute nothing.

---

## 2. The architecture

```
          evidence / facts
                 │
                 ▼
     ┌───────────────────────┐
     │ S1  facts (grounded)  │  reuse v1: 96,623 facts, 0.92 verbatim
     └───────────┬───────────┘
                 ▼
     ┌───────────────────────┐
     │ S2  fact families     │  SOFT assignment over embeddings. No argmax.
     └───────────┬───────────┘
                 ▼
     ┌───────────────────────┐        ┌──────────────────────────┐
     │ S3  fact patterns     │        │ A1  authorities          │
     │     by claim family   │        │  statutes · precedents   │
     └───────────┬───────────┘        └────────────┬─────────────┘
                 │                                  ▼
                 │                     ┌──────────────────────────┐
                 │                     │ A2  rules / tests that   │
                 │                     │     DEFINE each element  │
                 │                     └────────────┬─────────────┘
                 ▼                                  │
     ┌─────────────────────────────────────────────▼──────────┐
     │ S4  ELEMENT layer — do the facts SATISFY or DEFEAT     │
     │     each element, per party, with a verbatim quote     │
     └───────────────────────┬────────────────────────────────┘
                             ▼
     ┌────────────────────────────────────────────────────────┐
     │ S5  do the elements SUPPORT or DEFEAT each             │
     │     claim / defence                                     │
     └───────────────────────┬────────────────────────────────┘
                             ▼
     ┌────────────────────────────────────────────────────────┐
     │ S6  issues → conclusion                                 │
     └────────────────────────────────────────────────────────┘
```

Catalogue: `../ontology/ontology_v1.yaml` — **15 claims, 66 elements, 6 defence families,
4 remedy tiers**, each element carrying `element_id`, `name`, `slug`, `definition`.

---

## 3. The evaluation principle, and why it is the main change

> **Every stage is scored on its own task. Win/lose is reported last and never used to judge an
> intermediate layer.**

Each stage also carries a **gate that needs no ground truth**, so progress is measurable before the
gold set exists:

| stage | task-local metric | no-gold gate |
|---|---|---|
| S2 fact families | AUROC vs the v1 rungs | uniform-blur control must score ~chance |
| S3 patterns | discriminativeness **within a claim family** | BH correction; bootstrap stability |
| A2 element criteria | does the cited authority mention the element's terms | citation verifiable against the corpus |
| **S4 elements** | agreement with gold once it exists | **a SATISFIED verdict is rejected unless its quote occurs verbatim in the facts shown** |
| S5 claims | — | element support must cite element ids that S4 actually emitted |
| S6 conclusion | accuracy, calibration, vs per-decade majority | — |

**The architecture's own falsifiable claim** is that elements are the right intermediate
representation. That is testable now, with no gold:

> Do S4 element features beat S2 soft-fact features at predicting the outcome?
> If yes, the element layer earns its place. If no, it is decoration.

That is the acceptance test for v2 as a whole.

---

## 4. Reused from v1 — do not reimplement

| need | use |
|---|---|
| judgment text, masking | `src.data.mask` → `Data/interim/masked_text.jsonl` |
| facts + verbatim spans | `Data/interim/facts.jsonl` (96,623) |
| **cached fact embeddings** | `src.extract.induced_vocab.load_fact_texts` — every fact already embedded, zero cost |
| outcome labels | `src.data.label_merge.load_final` |
| splits (hashed) | `Data/splits/*.json` |
| LLM calls, cache, retries | `src.llm.client` — 90s timeout, `ok=False` ≠ empty answer |
| quote location | `src.grounding` — hyphenation repair included |
| metrics, paired bootstrap | `src.eval.metrics` |
| statutes, precedents | `src.authorities.*` — statutes 1.9× baseline, precedents 7.4× control |

---

## 5. Constraints v1 already established — design around these

1. **Party attribution is 19–20% and the cause is unknown.** S4 needs to know *who* asserted a fact
   to say satisfy-vs-defeat. Either fix it or scope S4 to attributed facts and report coverage
   honestly. This is a blocker, not a polish item.
2. **`E` group burden metadata is 0 of 66** (blocker B2). Two models failed a degeneracy gate; a
   bigger model is not the fix. S4 must work *without* burden weighting at first.
3. **No gold annotation set** (B4). 150 cases prepared, unannotated. Hence the no-gold gates above.
4. **Flat itemsets found nothing** — 498 mined, 0 BH-significant. S3 should mine over **graph**
   structure (fact → element → claim), which is what `gSpan` is for, not flat co-occurrence.
5. **The signed citation graph exists in git history** (8,467 edges, 62.6% signed) and is the
   substrate for `anco-HITS`. It is the only component no other design has. Restore it for A1/A2.
6. **Mean-pooling embeddings is bad** — 4,096-dim mean (0.558) loses to a 345-dim soft histogram
   (0.601). Averaging washes out the decisive fact.

---

## 6. Build order

| # | stage | status |
|---|---|---|
| 1 | **S2 soft fact families** | **done** — 0.633 at 345 dims, +0.116 over v1 |
| 2 | **S4 element layer** + the verbatim-quote gate | next |
| 3 | S4 → outcome, against S2 → outcome (the acceptance test) | — |
| 4 | S5 elements → claims / defences | — |
| 5 | A2 authorities → element criteria | — |
| 6 | S3 graph patterns (gSpan) within claim family | — |
| 7 | A1 signed graph + anco-HITS authority ranking | — |
| 8 | S6 conclusion, reported last | — |

## 7. Running

```
python -m v2.canon_soft --split forum_heldout      # S2
```

Results land in `v2/experiments/`. v1's `experiments/` is never written to.
