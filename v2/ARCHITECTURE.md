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
| 2 | **S4 element layer** + the verbatim-quote gate | **done** — 40,078 verdicts, 0.62% fabricated |
| 3 | **S4 acceptance test** | **done — FAILED.** 0.521 vs S2's 0.633 |
| 3b | **S4 correctness: gold element labels** | **built, awaiting a reviewer — 214 items, ~94 min** |
| 3c | **Why S4 fails** | **done** — the doctrinal basis, not the encoding |
| 3d | **Name the induced clusters** | **done** — and it dissolves the premise |
| 4 | S5 elements → claims / defences | deferred — inherits a 3-value channel |
| 5 | A2 authorities → element criteria | — |
| 6 | S3 graph patterns (gSpan) within claim family | — |
| 7 | A1 signed graph + anco-HITS authority ranking | — |
| 8 | S6 conclusion, reported last | — |

## 6a. S4 findings so far

**The quote gate works.** 1.44% of proposed verdicts were refused for a quote that is not in the
facts (40-case pilot), comparable to the legacy pipeline's 3.2%. Fabricated *support* is caught
without any ground truth. Wrong *reasoning* from a real quote is not caught, and that limit stands.

**The model had to change, and finding out why took three tests.** `llama4-scout-17b` — v1's default
everywhere, chosen on an extraction benchmark where it matched models 6.8× slower — emits **zero**
`NOT_SATISFIED` on this task under every condition tried. Refuted in order: prompt over-correction
(rewriting it to solicit negatives: 0 → 0); defeats living in the reasoning that §5.2 removes
(showing the model the reasoning: still 0). Confirmed: it is the model.
`qwen3-235b-a22b-instruct-2507` uses all three verdicts and fabricates *less* (0.0%).

A layer that cannot say `NOT_SATISFIED` cannot express the "or defeat" half of the design, so this
was blocking, not cosmetic. **Lesson: the model choice validated for extraction is not
transferable to judgement tasks.**

**The verdicts are legally sensible.** The strongest `NOT_SATISFIED` found so far: *"Sham Lal was a
tenant of a room in property unit No. B-VI-33"* defeating an adverse-possession element — tenancy is
permissive possession, so it affirmatively negates hostility. That is the inference the architecture
exists to make.

**Open on S4:** `UNCLEAR` is still the majority verdict (56 of 87 even on the better model), and
`MIN_QUOTE = 25` is permissive enough to admit a 44-character fragment as evidence. Both are
measured and neither is resolved.

## 6b. The acceptance test result — the element layer FAILED it

Full S4: 4,674 cases × top-2 claims = 9,348 calls, 102 minutes, `qwen3-235b-a22b-instruct-2507`.
**40,078 gated verdicts** — 13,008 SATISFIED (32%), 1,481 NOT_SATISFIED (3.7%), 25,589 UNCLEAR (64%).
**Fabrication 0.62%** (206 quotes absent, 44 too short, of 40,328 proposed). Zero never-answered,
zero schema failures.

Then the acceptance test from §3, on the same 885 test cases:

| arm | AUROC | features |
|---|---|---|
| S2 soft facts | **0.633** | 345 |
| **S4 elements** | **0.521** | 70 |
| S2 + S4 | 0.616 | 415 |

| comparison | ΔAUROC | 95% CI | |
|---|---|---|---|
| S4 − S2 | **−0.1125** | [−0.1625, −0.0639] | significantly WORSE |
| S2+S4 − S2 | −0.0169 | [−0.0345, +0.0011] | not significant |

**The element layer scores essentially at chance, and adding it to S2 does not help.** This is not a
null result; it is the same failure mode as v1's hard canonicalisation, one level up the stack.

### Why — the channel is about three values wide

| | |
|---|---|
| elements asked per case (median) | **8** of 66 in the catalogue |
| non-zero slots per case (median) | **3** |
| cases with no signal at all | **920 = 19.7%** |

S4 compresses a case to ~3 ternary values; S2 gives it 345 continuous dimensions. **The bottleneck is
channel width, not the choice of symbol.** The architecture discretises twice — facts→atoms and
facts→elements — and both times the channel is too narrow relative to the text it came from.

A free within-data check confirms the direction but not a rescue. AUROC by number of non-zero slots:

| non-zero slots | n | AUROC |
|---|---|---|
| 1–2 | 197 | 0.480 |
| 3–4 | 293 | 0.540 |
| 5–20 | 240 | 0.559 |

Monotone, so width genuinely limits it — but even the widest band reaches only 0.559 against S2's
0.633. Widening the budget (top-5 claims instead of top-2, ~23,000 calls, ~4 h) is worth one run,
and on this trend it is unlikely to close a 0.11 gap.

### What this does and does not establish

The reading below was written into `s4_eval.py` **before** the result was seen, so it is not a
post-hoc rationalisation.

**Established:** the element layer, as a *representation*, carries less outcome signal than the facts
it is derived from. Routing facts→elements→outcome is worse than facts→outcome. For the professor's
third arrow — facts + authorities → win/lose — the element layer is a liability.

**Not established:** that the layer is legally wrong. Three independent reasons it is still the most
defensible product in either version of this project:

- **It is reliable.** The two concurrent runs (see below) give **96.9% agreement** across 40,000
  verdicts, better than v1's 87.0% rules-vs-LLM label agreement.
- **It almost never fabricates.** 0.62% of verdicts refused by the quote gate.
- **Its verdicts are legally sensible**, e.g. *"Sham Lal was a tenant of a room in property unit
  No. B-VI-33"* defeating an adverse-possession element, because tenancy is permissive possession.

*"Here are the elements your facts satisfy, with the verbatim quote for each"* is **checkable**.
Win/lose never is. The honest conclusion is that the element layer is a good **explanation** layer
and a bad **feature** layer, and the architecture is wrong to route prediction through it.

### An accidental reliability study, and the bug that produced it

Two S4 processes ran concurrently — a launch I wrongly believed had failed, plus its replacement —
and both wrote to the same file: 18,696 rows for 9,348 jobs, exactly 2× every key. **This is v1's
"driver ran two stages at once" bug recurring**, and the lesson is the same: verify a background job
started rather than inferring it from absent output.

Deduplicated on `(case_id, claim_id)`, keeping the first. The accident bought a free test–retest
study at temperature 0:

| | |
|---|---|
| element verdicts comparable in both runs | 40,000 |
| **agree** | **38,752 (96.9%)** |
| disagree | 1,248 (3.1%) |
| — SATISFIED ↔ UNCLEAR | 854 |
| — NOT_SATISFIED ↔ UNCLEAR | 237 |
| — **SATISFIED ↔ NOT_SATISFIED** | **157 (0.4%) — direct sign flips** |
| identical quote when both agree on a non-UNCLEAR verdict | 92.7% |

Temperature 0 and a fixed seed are **not** sufficient for determinism on this endpoint, which is
worth knowing before any result is reported as exactly reproducible.

## 6c. Measuring whether S4 is RIGHT (not just useful)

S4 failed the acceptance test as a feature layer, so its only remaining claim is that its verdicts
are **correct** — and that has never been measured. B4's 150 prepared gold cases were built for
extraction annotation; they are repurposed here.

`python -m v2.gold_elements` builds the set: **40 cases, 214 element judgements**, from the 110 gold
cases that overlap S4's output.

| model verdict | n | 95% CI half-width at precision 0.80 |
|---|---|---|
| SATISFIED | 95 | ±0.080 |
| UNCLEAR | 76 | ±0.090 |
| NOT_SATISFIED | **43 (all of them)** | ±0.120 |

Reviewer budget **~94 minutes**. Four design choices, each bought with reviewer time:

1. **Grouped by case.** Reading a case's facts costs ~60 s; deciding one element costs ~15 s. Nine
   items from one case means one reading, not nine. The unit of work is a case.
2. **UNCLEAR is subsampled; the decisive classes are not.** Confirming "the facts don't address
   this" teaches little. Every NOT_SATISFIED is included — it is 3.7% of the corpus and the whole
   "or defeat" half of the design rests on it.
3. **Blind.** The reviewer never sees S4's verdict before giving their own; it is revealed after each
   case is submitted. Anchoring would turn an agreement measurement into a confirmation exercise.
4. **Quotes are clicked, not typed.** The reviewer picks a fact *number*, so the quote is exact by
   construction and the same mechanical gate that scores S4 scores the human. A reviewer who cannot
   point at a fact has, by this project's own standard, no evidence.

### The number that will interpret the result

`score_gold_elements.py` reports precision and recall **per class** rather than one accuracy, because
64% UNCLEAR would let an overall figure be carried by the easy class. It also separates *right verdict
from the wrong fact* from *wrong verdict*, which are different failures.

Then it compares human–model disagreement against **3.1%** — S4's disagreement with *itself* across
two independent runs at temperature 0. **If disagreement approaches that floor, prompt work cannot
close the remainder; if it is far above, there is real headroom.** That floor only exists because of
the accidental double-run, which is the one way that bug paid for itself.

### How to run it

```
.venv/bin/streamlit run v2/annotate_elements.py     # ~94 min, writes after every case
.venv/bin/python -m v2.score_gold_elements          # scores whatever exists
```

The annotation file is append-only and written per case, so **a partial pass still scores** — stop
whenever and run the scorer. Both paths were smoke-tested end to end with a synthetic record.

## 6d. WHY S4 fails — it is the basis, not the encoding

Three hypotheses, all testable for free because every fact already carries a cached embedding. **Two
were mine and both were refuted**, which is what made the third clean.

| arm | AUROC | features |
|---|---|---|
| masked text (ceiling) | 0.659 | 50,000 |
| S2 soft facts | 0.617 | 345 |
| S4, old ±1/0 encoding | 0.531 | 66 |
| S4, H1: asked/unclear split out | 0.529 | 264 |
| S4, H2: **soft** element similarity, no LLM | 0.524 | 66 |
| S2 + soft elements | 0.608 | 411 |

**H1 — "UNCLEAR and never-asked are both encoded 0, which conflates two different facts."** True, and
it buys **−0.003**. The distinction carries nothing.

**H2 — "we repeated S2's own mistake one level up: S4 hard-discretises into three labels."** The
analogue of S2's winning move is a continuous 66-dim element-similarity vector straight from the
embeddings. It scores **0.524 — no better than the hard version (−0.007)**. Softening bought +0.116
at the fact level and **nothing** at the element level.

### The matched-granularity control that settles it

The remaining confound was dimensionality: 345 for S2 against 66 for elements. So S2 was re-run at
**k=66**:

| 66-dimensional basis, soft, same embeddings, same cases | AUROC |
|---|---|
| **induced centroids** (data-driven) | **0.632** |
| **authored legal elements** (doctrine) | **0.524** |

And induced-66 (0.632) equals induced-345 (0.633), so **granularity is irrelevant**. The dimensionality
confound is dead, and what remains is the basis itself.

> **The element layer does not fail because of channel width, hard quantisation, or an encoding bug.
> It fails because the 66-element doctrinal vocabulary is the wrong basis. The outcome-predictive
> content of these facts is largely orthogonal to what legal doctrine says should matter.**

Project the same facts onto 66 clusters the corpus suggests → full signal. Project onto 66 elements
the law prescribes → near chance. Same dimensionality, same encoding, same embeddings, same cases.

### A flaw in the acceptance test, stated

Masked text reaches 0.659 and S2 reaches 0.617, so **any fact-derived layer has at most 0.043 to gain
over S2.** "Beat S2" was close to unwinnable and §3 should have said so. It does not excuse S4's
0.524 — that is far *below* S2, not merely failing to exceed it — but the bar was badly set and the
headroom is now reported alongside every arm.

### What this means for the architecture

The professor's instruction was *"Fact Patterning by Claim Family (**find claims, cluster**)"*. S2
clusters, and it works. S4 then **replaced those clusters with an authored ontology, and that is the
step that broke**. The architecture's shape was right; substituting doctrine for data was wrong.

So the element layer should be the **explanation** layer — checkable, 96.9% reliable, 0.62%
fabrication — and the **prediction** should run on the induced basis. Those are complementary in
function, not redundant in signal: S2 + soft elements (0.608) does not beat S2 alone (0.617), so the
elements add no predictive information on top.

**The constructive path, not yet built:** name the induced clusters. If the 66 data-driven centroids
that carry the signal correspond to something legally nameable — procedural posture, party type,
remedy sought — that is both a finding and a candidate ontology better than the authored one. v1
already has a `name_vocab_cluster.v1` prompt for exactly this, written and unused.

## 6e. Naming the clusters — and the finding that dissolves the question

Stage 3d was built on the hope that the predictive clusters would be a better, data-grounded
ontology. All 66 were named by `qwen3-235b` from their twelve nearest facts, with a `kind` field and
a prompt written to resist the flattering answer (it states that an honest `procedural` label is
worth more here than a legal-sounding one). 66 of 66 named.

| | all 66 | top 20 by TRAIN univariate strength |
|---|---|---|
| legal_substance | 27 (41%) | 8 (40%) |
| procedural | 19 | 8 |
| temporal_quantum | 18 | 4 |
| party_entity | 2 | 0 |

**Kind does not predict predictiveness.** Legal substance is 41% of all clusters and 40% of the most
predictive ones — neither enriched nor depleted. Mean similarity to the nearest authored element is
0.496 for the top 20 against 0.533 overall, so the best clusters sit marginally *further* from the
ontology, but the difference is small.

### The decomposition that explains everything

Restricting S2's 66-dim histogram to each kind:

| basis | AUROC | dims |
|---|---|---|
| all 66 clusters | **0.632** | 66 |
| procedural only | 0.609 | 19 |
| legal_substance only | 0.606 | 27 |
| temporal_quantum only | 0.604 | 18 |
| all **minus** procedural | 0.622 | 47 |

**Every subset lands at 0.604–0.632.** Legal substance alone, court machinery alone, and dates and
sums alone are *indistinguishable*. Removing all 19 procedural clusters costs 0.010.

> **There is no concentrated outcome signal in these facts. It is weak and smeared across the whole
> fact space, so any reasonably complete projection captures most of it and no projection captures
> much.**

That single fact explains the entire sequence of results:

- why S2 scores ~0.63 at k=66 *and* k=345 — both span the space;
- why the authored elements score 0.524 — 66 doctrinal elements are a *selective probe* (only ~8 asked
  per case, each a specific legal proposition), not a spanning basis, so they miss a diffuse signal;
- why softening the element layer bought nothing — you cannot recover a signal the basis never
  covered;
- why the headroom above S2 is only 0.043.

**So the premise of stage 3d was wrong.** The induced clusters are not a better ontology waiting to be
read off. They win because they *cover* the fact space, not because they carve it at better joints.
There is no ontology that unlocks outcome prediction here, because there is no concentrated structure
to carve.

### What the clusters DO say — a real subject-matter regularity

The content is coherent even though it is not elemental. The LOSE-direction legal clusters are almost
all one thing:

| cluster | favours | member facts |
|---|---|---|
| `land_reforms_legislation_enactment` | LOSE | *"The West Bengal Land Reforms Act, 1955 was introduced"* |
| `vesting_of_proprietary_rights` | LOSE | *"Proprietary rights in sir and khudkashat land … vested in the State"* |
| `land_vesting_in_state` | LOSE | |
| `accession_of_princely_states` | LOSE | |
| `religious_endowment_and_institution` | LOSE | |
| `document_execution_and_admission` | **WIN** | |
| `contract_performance_failure` | **WIN** | |

**Post-independence land reform and state vesting cases lose; private document and contract disputes
win.** That is a finding about Indian land jurisprudence — a *dispute-regime* regularity, not element
satisfaction. It is also exactly why an element ontology could not capture it: elements ask "is
hostility established?", while the data says "is this a zamindari abolition matter against the State,
or a private sale-deed dispute?"

### A concrete v1 masking bug found on the way

The single strongest cluster, `dismissal_of_appeal_or_petition`, has member facts that read literally
*"The appeal was dismissed"*. `mask.py`'s `OUTCOME_CUE` matches `is|are|stands|shall stand` and
**not the past tense**, and `PRIOR_COURT_DISPOSITION` requires a named forum. So all of these escape
**both** detectors:

```
"The appeal was dismissed"                        OUTCOME_CUE ✗   PRIOR ✗
"The suit was dismissed on the ground of limitation"  ✗           ✗
"The Munsif dismissed the suit"                       ✗           ✗   (Munsif not in the forum list)
```

It favours **WIN**, which is the signature of a *prior-court* disposition rather than leakage of this
court's order — the party lost below, obtained leave, and succeeded here. So it is v1's group-`Q`
procedural shortcut, which §5.2 intended to isolate, reappearing inside the fact stream. The
decomposition above bounds the damage at 0.010 AUROC, so it does not drive any result, but
`OUTCOME_CUE` should gain past-tense forms and `PRIOR_COURT_DISPOSITION` a wider forum list.

## 7. Running

```
python -m v2.canon_soft --split forum_heldout      # S2
```

Results land in `v2/experiments/`. v1's `experiments/` is never written to.
