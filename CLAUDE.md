# LexGraph — CSE 573 Legal AI Project

Last rewritten 2026-10-03, when the repository was consolidated. **Read
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

**Neither critique is novel.** See the report's prior-art section — Nigam & Deroy (2023), Watson
et al. (2026), Sadowski & Chudziak (2026) got there first. What is ours is methodological. Do not
let anyone write "we broke the benchmark"; `writeup/` says exactly what is defensible.

## Layout

```
writeup/          LexGraph_Report.tex + .pdf   <- the one document
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

## Open blockers (all human time, not compute)

- **B4** — 150 gold cases prepared, no reviewer. Blocks the *only* precision/recall figure for
  extraction; the biggest gap for a reviewer.
- **B2** — 66 element burdens unannotated, so feature group `E` is empty. Two models failed a
  degeneracy gate; a bigger model is not the fix.
- **B3** — vocabulary 20.3% out of vocabulary vs a 5% freeze threshold.
- **B5** — ILDC licence gated indefinitely. **Optional**: `corpus_scale_setback` covers the same
  population. `src/eval/ildc_ablation.py` is setback-armed and smoke-tested; do not route around
  the gate. The authors are *not* inactive — email Modi rather than waiting on HuggingFace.
- Not started: demo dashboard; team roster; official proposal template.
