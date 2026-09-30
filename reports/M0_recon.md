# M0 — Corpus reconnaissance against PROJECT.md

Run 2026-09-29. All numbers measured, not estimated. Scripts in the session scratchpad;
each is a single pass over `Data/processed/corpus_text.jsonl` or `Data/raw_pdfs/`.

## 1. The land/property dataset is 6,954 cases and the PDF join is exact

`Data/Legal AI Dataset/filtered_land_disputes.xlsx` has a **`filename`** column.
All 6,954 filenames resolve to a file in `Data/raw_pdfs/` (NFC-normalized, case-insensitive
fallback): **6954/6954 = 100.0%**.

This matters: CLAUDE.md gotcha #15 warns that joining doc_ids to PDFs needs a fragile title
normalizer that once produced a 0.6% join rate. **That gotcha does not apply to the
land-dispute subset** — the spreadsheet names the file directly. No fuzzy matching.

| | cases |
|---|---|
| land-dispute rows in xlsx | 6,954 |
| ...resolving to a PDF on disk | 6,954 (100%) |
| ...with cached text today | **2,921** |
| ...still needing text extraction | **4,033** |

The existing `corpus_text.jsonl` (5,462 docs) was built for the *general* corpus, so only
53% of the cached text is even a land dispute. Extracting the remaining 4,033 is a **2.4x
increase in usable data** and is the cheapest high-value action available.

Decade spread of all 6,954 (no decade starves a per-family FP-Growth run):

| 1950s | 1960s | 1970s | 1980s | 1990s | 2000s | 2010s | 2020s |
|---|---|---|---|---|---|---|---|
| 381 | 1205 | 1128 | 892 | 922 | 898 | 960 | 568 |

## 2. Outcome labels (§5.1) are feasible: ~61% from regex alone

A deliberately narrow regex over the **last 3,000 chars** (`<appeal|petition|suit> ... is/are
... allowed|dismissed`), over the 5,462 cached docs:

| cue | hits | share |
|---|---|---|
| ...allowed | 1,605 | 29.4% |
| ...dismissed | 1,945 | 35.6% |
| "in the result" | 996 | 18.2% |
| partly allowed | 210 | 3.8% |
| remanded | 238 | 4.4% |
| allowed **or** dismissed | **3,365** | **61.6%** |
| both (needs resolution) | 185 | 3.4% |

On the 2,921 land-dispute cases with text: **1,778 cleanly labelable (60.9%)** —
766 allowed / 1,012 dismissed. **Class balance is 43/57**, so no majority-class trap and
macro-F1 is meaningful.

This is a floor, not a ceiling: the regex ignores "in the result" (18.2%), decree/quash
language, and multi-appeal dispositions. Broader patterns plus LLM residuals per §5.1 should
clear 85%. Projected labelled land-dispute set once extraction completes: **~5,900 cases.**

## 3. Masking (§5.2) cannot be header-based, but can be paragraph-based

Structural markers across the 5,462 cached docs:

| marker | present | pre-2000 | post-2000 |
|---|---|---|---|
| `HEADNOTE` | 53.0% | **79.5%** | **0.0%** |
| `JUDGMENT` header | 96.1% | 97.5% | 93.2% |
| `ORDER` header | 3.7% | 3.2% | 4.7% |
| `FACTS` header | 29.7% | — | — |
| numbered paragraphs | 73.2% | 63.4% | **92.7%** |

So: **no usable section headers** (`ORDER` 3.7%, `FACTS` 29.7%) — this confirms the earlier
finding. But **numbered paragraphs are the segmentation unit**: post-2000 judgments have a
median of **32** numbered paragraphs and **85.1%** have at least 5. §5.2's "rules + an LLM
section classifier" should therefore classify *paragraphs*, not hunt for headers.

### 3.1 The leakage hazard PROJECT.md underweights: prior-court disposition

Position of cue matches within post-2000 judgments (0 = start, 1 = end):

| cue type | p10 | p25 | median | p75 | p90 |
|---|---|---|---|---|---|
| fact narrative | 0.08 | 0.14 | 0.45 | 0.67 | 0.88 |
| argument ("learned counsel submitted") | 0.11 | 0.17 | 0.35 | 0.52 | 0.80 |
| **outcome** | 0.13 | 0.31 | **0.85** | 0.98 | 0.99 |

Outcome cues concentrate late (59.2% in the last 30%) — but **19.2% of them fall in the
first 30% of the text.** Those early hits are not the Supreme Court's own disposition; they
are the **procedural recital** ("the High Court dismissed the suit", "the trial court
decreed"). That makes prior-court disposition simultaneously:

- a **legitimate fact** about the case's history, and
- the single strongest predictive shortcut available, since an apex court affirms far more
  often than it reverses.

A purely positional mask therefore leaks, and a mask that strips these removes real facts.
**Treat prior-court disposition as its own ablation axis (with / without), not as part of
the mask.** The gap between those two arms is a reportable result, and it is the number
that tells you how much of any headline accuracy is the shortcut rather than the law.
§5.1's `initiator_outcome` / `original_plaintiff_outcome` fields are necessary but not
sufficient for this.

### 3.2 The temporal split (§5.3) carries a format confound

`HEADNOTE` goes from 79.5% (pre-2000) to **0.0%** (post-2000), and numbered paragraphs from
63.4% to 92.7%. A train-on-old / test-on-new split therefore conflates *later in time* with
*structurally different document*. HEADNOTE must be stripped regardless (it states the
outcome **and** lists the authorities), but the confound survives stripping. Name it, and
report the court-held-out split (§5.3 secondary) alongside as the control.

## 4. Reusable assets — more than "random experiments"

| Asset | Serves | State |
|---|---|---|
| `cache_corpus_text.py` | corpus build | works, resumable; just needs re-pointing at the 4,033 |
| `retrieval_lib.py` | §9.3 retrievers | BM25 + dense behind **one `search()`**, and `search(..., before_year=)` is **already time-respecting per §5.3** |
| `dense_index.npy` + meta | §9.3 dense arm | 4,096-dim vectors already embedded (~85 min of API time) |
| `signed_graph_edges.csv` | §9.3 citation graph | 8,467 edges, 62.6% sign-labelled |
| `claim_elements.json` | §8.3 element mapping | 29 elements mined with support >= 2 distinct judgments across 6 claim types |
| `phase3_verify.verify_authority` | §9.1 normalization, §11 trace | heuristics hand-audited; see CLAUDE.md gotcha #9 before touching |
| `classify_citations.py` | §9.1 | only ~32.5% of listed "precedents" are real case cites; needed |

Not reusable and correctly dropped: the GROUND/LEARN/EXTRACT rule-induction path
(`ode_pipeline.py`, `learned_rules.json`) — it never emitted `Fact` or `Issue`, which is
exactly what §6.1 needs, and it is model-coupled (gotcha #13).

## 5. BLOCKER

`Land_Property_Dispute_India_Ontology_Refined.docx` — required by §6.3 as ontology v1 and
by §8.3 for element burden metadata (`burden_on`, `burden_standard`, `burden_shifts_when`)
— **is not in `Docs/`**. `Docs/` holds the Templeton proposal and lecture material only.
The 15 claim modules are recoverable from §7 of PROJECT.md itself; the element and burden
metadata are not. See `reports/BLOCKERS.md`.
