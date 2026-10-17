# LexGraph — CSE 573 Legal AI Project

Last rewritten 2026-10-03, when the repository was consolidated; updated 2026-10-16. **Read
`writeup/LexGraph_Report.tex` before touching anything** — it is the single authoritative
document and replaces the former `PROJECT.md`, `WORKLOG.md` and `reports/*.md` (all removed;
recoverable from git history). `writeup/LexGraph_Explained.tex` is the plain-language companion.

## What this is

A class project for **CSE 573: Semantic Web Mining** (ASU, Fall 2026, Prof. Hasan Davulcu, TA
Anshul Trivedi), assigned topic P15 "Legal AI – Agentic Law". Domain: Indian Supreme Court
land/property judgments, 1950–2025, inside a corpus of all 26,688 SC judgments.

**What it actually establishes** — two measurements, not a working predictor:

1. **Leakage.** The input construction used by the published state of the art beats a facts-only
   input by +0.111 to +0.181 AUROC, and essentially all of it is disposition language surviving
   deletion of the operative order. The residue reaches ~**128 words** past the boundary (found
   independently on four splits); a word-count-matched window set back past it scores no better
   than the facts anywhere. Significant in all eight decades over 26,688 judgments. The court's
   reasoning adds only ~+0.025.
2. **Discretisation.** Projecting facts into a symbolic vocabulary costs 0.097 AUROC. It is
   discretisation itself, not the ontology: an induced vocabulary of equal granularity is
   indistinguishable (P=0.846), and the cost concentrates where the vocabulary *succeeds*.

Plus three positives: statutes at 1.9× baseline, precedents at 7.4× a popularity control, and a
faithfulness-tested trace mechanism.

**Since then (Oct 4–16):**
- **v2** (`v2/`, plan in `v2/ARCHITECTURE.md`) rebuilt the intended pipeline properly. Soft fact
  families (S2) recover the fact signal (0.633 vs 0.629 for the text); the element layer (S4)
  fails as a predictor (0.521) because the doctrinal basis is wrong, not the encoding — the outcome
  signal is diffuse across the fact space.
- **Fine-tuning** (`ft/`): InLegalBERT on facts = **0.695 AUROC** (3 seeds, sd 0.005) vs TF-IDF
  0.667 on the same cases. No single seed is significantly better; the two-seed average is
  (+0.031 [+0.003, +0.059]). Same model on the leaky window = 0.907. Ceiling ≈0.70, not 0.75.
- **S4 correctness**: no human reviewer exists, so the 40-case gold set was annotated by a *blind
  AI reviewer*, stored and reported as such. SATISFIED P/R 80/82%; NOT_SATISFIED precision 37%.
  Tagging prior-court findings (`attribution_audit.py`, `s4_findings_rerun.py`) lifts it to 47%.
- **Fact types** (`v2/fact_types.py`): every fact labelled PRIMARY / PROCEDURAL / FINDING / LAW /
  ASSERTION by reading its source sentence; pilot recall 30/36, precision 30/41 against hand labels.
  Corpus: 15.3% FINDING, 7.6% LAW. **48.8% of S4's NOT_SATISFIED quote a FINDING** (16% of
  SATISFIED). Tagging them does not work — the model still quotes tagged findings in 54% of defeat
  verdicts; removing them halves the defeat verdicts (1,481 → 590). Demo uses the `tag` arm.
- **ILDC replication run** (`experiments/paper1/ildc_ablation.json`): shape holds on the benchmark
  itself (only w0 beats facts-only, +0.055 AUROC), a third of our magnitude with TF-IDF.
- Demo built (`demo/`), shows tagged S4 verdicts with defeat verdicts marked low-confidence.

**Neither critique is novel.** See the report's prior-art section — Nigam & Deroy (2023), Watson
et al. (2026), Sadowski & Chudziak (2026) got there first. What is ours is methodological. Do not
let anyone write "we broke the benchmark"; `writeup/` says exactly what is defensible.

## The product (from 2026-10-16): a research-brief tool for litigators

Direction set by the user: a world-class legal AI for **lawyers**, not class deliverables. Code in
`product/`; run `.venv/bin/streamlit run product/app.py` (first load ~100 s, then ~60 s per story).
`product/pipeline.py` is the single entry point: `LexGraph().analyse(story)`.

Evaluated on 93 held-out cases retold as **client narratives** (no court history, statute numbers or
case names; leakage-gated) -- never on judgment text, which no real user has:

| step | metric | value |
|---|---|---|
| statutes (kNN vote) | R@10 of court's provisions | 0.380 (prior 0.197) |
| precedents (full-text BM25 + dense, LLM rerank over case cards) | R@10 / MRR | **0.215** / 0.368 (judgment-text ref 0.290; popularity ~0.03) |
| good law (`treatment.py` + direction check) | OVERRULED precision (hand) | ~85-92%; 7,751 citations located |
| brief, grounded | authorities certified | 613/613; 15.5% were cited by the court |
| brief, closed-book | cited cases matching no SC judgment | 45.6% (upper bound on fabrication) |

Rules learned building it: **cards rerank, they must not retrieve** (card retrievers + card reranker =
correlated errors, 0.215 -> 0.192); memo-named provisions are right only 21% -- never show them
unverified; LLM treatment labels invert "X was overruled in Y" ~half the time without a second
yes/no direction check; read every name-matcher's matches (two matcher bugs found that way).

## Course logistics — I dropped these in a rewrite; do not drop them again

Class project is **60% of the grade**:

| deliverable | weight | when | status |
|---|---|---|---|
| proposal | 15% | **due Oct 7, 2026** | **submitted** — `proposal/FA26-Group-Proposal-Group34-Project15.pdf` |
| presentation | 10% | **Oct 28 – Nov 18** | not started |
| group demo | 15% | **Nov 23 – Dec 2** | **built** — `demo/` (Streamlit, reads only `demo/pack/`) |
| final report | 20% | **due Dec 9** | `writeup/LexGraph_Report.tex` is most of the content |

**Formatting rule for every written submission**: 1-inch margins, A4, 12pt Times New Roman, single
column, PDF. Violating it costs 20%. Both documents in `writeup/` already comply.

**The submitted proposal and the findings now diverge.** It is element- and ontology-heavy (15
mentions of "element", 3 of "leakage") and promises the element-centric pipeline that v2 has since
shown does not work predictively (0.521 against 0.633). This is not a problem to hide — "we built
exactly what we proposed, measured it rigorously, and here is what we found" is a stronger research
narrative than a pipeline that happened to work. But the presentation and final report must handle
the divergence deliberately rather than pretend the proposal said something else.

## Layout

```
writeup/          LexGraph_Report.tex + .pdf   <- the one document (build: tectonic)
v2/               element-centric rebuild: soft families, S4, gold set, attribution audit, fact types
ft/               fine-tuning (InLegalBERT); ft_masked_summary.json is the 3-seed record
demo/             build_pack.py precomputes demo/pack/; app.py is the Streamlit demo
                  LexGraph_Explained.tex + .pdf <- plain-language version
src/data/         adapter, labels, mask, segment, splits, corpus_wide, ontology_convert
src/extract/      facts, claims, canonicalize, vocab
src/eval/         leakage_probe, reasoning_ablation, leakage_setback, masked_integrity,
                  corpus_scale_setback, representation_ladder, vocab_ab, error_analysis,
                  ildc_ablation
src/authorities/  normalize, statute_predict, precedent_retrieve
src/predict/      features, models, llm_baselines
src/trace/        build, evaluate
src/llm/client.py the single door to Voyager (cache, 90s timeout, retries)
prompts/          versioned templates, keyed by the IDs the cache records
ontology/         ontology_v1.yaml (derived from Docs/ontology_source/ by ontology_convert.py)
experiments/      every result file — nothing in writeup/ is sourced elsewhere
tests/            48 tests, incl. the split-constraint test that caught a time-travel bug
Data/interim/     derived; large files gitignored, rebuild deterministically
Data/gold/        150 prepared gold cases + 66-element burden sheet (blockers B4, B2)
Data/raw_pdfs/    all 26,688 PDFs (gitignored, 6.8GB)
```

## Setup

1. `.venv/bin/pip install -r requirements.txt`
2. `.env` at root: `VOYAGER_API_KEY=<key>` from voyager.rc.asu.edu → LLM Access → Create Key.
   **Never propose a paid API** — Voyager is free to the user and call volume is not a constraint.
3. No VPN needed. Generation defaults to `llama4-scout-17b`; embeddings `qwen3-embedding-8b`
   (only 3 of ~49 models serve `/v1/embeddings` — probe, don't assume).

## Running

`./run_downstream.sh` for the chain (re-execs under `caffeinate -dimsu`, aborts on any stage's
non-zero exit, QA gate after extraction, every stage resumable). `make help` for individual
targets. The report's reproduction section lists the headline experiments.

## Hard-won rules — don't rediscover these

1. **A control is worth more than a method.** A copy control killed a statute result (0.447 → an
   honest 0.173); a popularity control validated the precedent result; a base-rate control
   dissolved an entire error taxonomy (every lift ≈1.0); Benjamini–Hochberg took a 498-pattern
   library to zero; volume-matching stopped a non-existent leak being reported. Every one changed
   the conclusion. **Add the control before believing the number.**
2. **An indirect argument that an objection is unlikely is not a measurement.** The last-512
   contamination was dismissed in writing on two true-but-unresponsive grounds, then found on the
   first direct test. A detector can mark a boundary *correctly* and still leave the sentence
   before it.
3. **`ok=False` ≠ empty answer.** Conflating "never got a reply" with "the model said nothing"
   produced two separate false findings. Any per-item LLM loop needs retries and must count drops
   separately.
4. **Assert your joins.** `MIN_PDF_JOIN_RATE = 0.99` exists because a normalised-title join once
   silently achieved 0.6%. A pipeline that processes 12% of its input is worse than one that
   crashes.
5. **Never `unzip` the corpus archives.** Two contain invalid UTF-8 filenames; APFS rejects them,
   unzip misreports it as "disk full", then takes EOF on the `Continue?` prompt and silently
   aborts the archive with exit 0. Use Python `zipfile` with sanitised names and verify 26,688.
6. **Elapsed time is not progress and 0% CPU is not a hang.** Three runs were killed before anyone
   checked `pmset -g log`. Long jobs run detached under `caffeinate`.
7. **Recall against *reachable* authorities**, with the raw figure beside it: only 25.6% of
   citation edges point at a case in this corpus.
8. **Read the match list after changing a matcher.** A false VERIFIED is worse than a false
   UNVERIFIED.
9. **Query/eval cases must be post-2000 where headnotes matter** — pre-2000 HEADNOTEs list the
   outcome and authorities outright.
10. **Model capability was almost never the binding constraint.** Instruction discipline,
    evaluation design and data coverage were. Two model swaps and five grounding models all
    failed to fix things that looked like capability problems.
11. **A gold set must show the reviewer exactly the evidence the model saw.** The first element
    gold set drew facts from a different extraction; only 49/138 model quotes were visible and
    agreement measured the mismatch (76% vs 48%). Check quote visibility before annotating.
12. **An AI annotation is not gold.** Store it under an annotator name that says so, write it to a
    separate file, and score it with `--ann/--out` so it can never be reported as the human result.

## Open blockers (all human time, not compute)

- **B6** — no human reviewer for S4's verdicts; only the blind AI pass exists.
- **B4** — 150 gold cases prepared, no reviewer. Blocks the *only* precision/recall figure for
  extraction; the biggest gap for a reviewer.
- **B2** — 66 element burdens unannotated, so feature group `E` is empty. Two models failed a
  degeneracy gate; a bigger model is not the fix.
- **B3** — vocabulary 20.3% out of vocabulary vs a 5% freeze threshold.
- **B5 closed (2026-10-16)** — ILDC access granted (`HF_TOKEN` in `.env`), files in `Data/ildc/`
  (gitignored: the authors' licensed release, never commit it). Replication run: shape holds
  (only w0 beats facts-only, +0.055), a third of our magnitude with TF-IDF, contamination reaches
  ~256 words not 128. Open: size it with a fine-tuned model.
- Not started: team roster; presentation.
