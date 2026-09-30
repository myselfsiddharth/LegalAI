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
