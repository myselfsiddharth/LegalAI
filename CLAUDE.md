# LexGraph — CSE 573 Legal AI Project

Last updated: 2026-09-18. This file exists so a fresh Claude session (or a teammate)
can pick this project up without re-deriving everything from scratch. Read this before
touching the code.

## What this is

A class project for **CSE 573: Semantic Web Mining** (ASU, Fall 2026, Prof. Hasan
Davulcu, hdavulcu@asu.edu, TA Anshul Trivedi). Assigned topic: **P15 "Legal AI –
Agentic Law"**. It's a scaled-down version of the professor's own Templeton Foundation
grant proposal ("Contestable Legal AI") — the original grant doc lives in `Docs/`
(gitignored, not in this repo: it's an unpublished/confidential funding document).

**The idea**: instead of asking an LLM to answer legal questions directly (which
hallucinates authorities), build a pipeline that (1) extracts typed legal objects
(claims, rules, cited authorities, holdings) from real judgments using an
ontology-constrained LLM-grounding method, not free generation, (2) builds a *signed*
citation graph (does a later case support or reject an earlier one), and (3)
deterministically verifies every cited authority against real citation data before
producing an IRAC-style (Issue/Rule/Application/Conclusion) answer — refusing to
certify anything it can't verify.

**Domain**: Indian Supreme Court land/property-dispute judgments (1950–2025).

## Course logistics (don't miss these)

- Class project = 60% of grade: proposal 15% (**due Oct 7, 2026**), presentation 10%
  (Oct 28–Nov 18), group demo 15% (Nov 23–Dec 2), final report 20% (**due Dec 9**,
  15–20 pages + supplemental code/data).
- **Formatting rule for all written submissions**: 1-inch margins, A4 paper, 12pt Times
  New Roman, single column, PDF. Violating this costs 20%.
- The professor said an official proposal outline/template would be provided — check
  Canvas before finalizing `LexGraph_Proposal_Draft.docx` against it.
- Team member names are still placeholders in the proposal doc — needs the actual team
  roster filled in.

## Repo layout

```
scripts/
  classify_citations.py       # splits real case-citations from statute/article noise
  build_sample.py             # stratified 500-case sample across decades 1950s-2020s
  build_drive_index.py        # filename -> Google Drive file ID lookup
  fetch_sample_pdfs.py        # downloads PDFs, resilient to Drive's rate limiting
  extract_headnote_signs.py   # pre-2000: regex-extracts citation sign from HEADNOTE text
  classify_modern_signs.py    # post-2000: LLM-classifies citation sign (no HEADNOTE exists)
  build_signed_graph.py       # merges both sign sources -> signed graph + Proof-Support Score
  ode_lib.py                  # shared GROUND (LLM) + rule-application logic
  ode_pipeline.py             # CLI: ground / learn / extract (Phase I)
  phase3_verify.py            # single-case IRAC report + citation verification (Phase III)
  run_evaluation_batch.py     # Phase I+III across many cases, aggregates real metrics
  ontology/
    land_dispute_ontology.json  # classes, roles, ie_concepts, terminology
    gold_examples.jsonl          # 24 hand-picked (sentence, concept, filler) examples
    learned_rules.json           # output of `ode_pipeline.py learn`
  requirements.txt
Data/
  Legal AI Dataset/           # original provided dataset (citations JSON + land-dispute index)
  processed/                  # everything derived by the scripts above
  raw_pdfs/                   # downloaded judgment PDFs (gitignored, ~155MB, regenerate via fetch_sample_pdfs.py)
LexGraph_Proposal_Draft.docx  # course proposal, A4/1in/12pt Times New Roman formatted
.env                          # VOYAGER_API_KEY (gitignored, not in repo -- see below)
```

## Environment setup (do this first in a new session)

1. `pip install -r scripts/requirements.txt` (gdown, pypdf, openai, networkx)
2. Create `.env` at project root: `VOYAGER_API_KEY=<key>` (get one at
   voyager.rc.asu.edu → LLM Access → Create Key). All scripts that call the LLM load
   this automatically (`ode_lib.load_dotenv()`).
3. **You must be on ASU's VPN** to reach the Voyager endpoint
   (`openai.rc.asu.edu`) — it resolves to a private `10.139.64.x` address, unreachable
   from the open internet. Without VPN, requests hang/timeout rather than failing fast.
4. Check available models with `curl -H "Authorization: Bearer $KEY"
   https://openai.rc.asu.edu/v1/models` before assuming a model name works — it's
   ASU's own hosted open-weight fleet (Llama 4, Qwen3, GLM-5, gpt-oss-120b, etc.), not
   GPT-4/Claude, and the list changes. Scripts default to `llama4-scout-17b`.

## Pipeline run order

```
python3 scripts/classify_citations.py
python3 scripts/build_sample.py --target 500
python3 scripts/build_drive_index.py <raw_gdown_listing.txt>   # see script docstring
python3 scripts/fetch_sample_pdfs.py                            # resumable, rate-limit-aware
python3 scripts/extract_headnote_signs.py
python3 scripts/classify_modern_signs.py                        # needs VPN + API key
python3 scripts/build_signed_graph.py
python3 scripts/ode_pipeline.py learn                            # needs VPN + API key
python3 scripts/run_evaluation_batch.py --n 50                   # needs VPN + API key
```

All of these are idempotent / safe to re-run — they either skip existing outputs or
fully regenerate deterministically from upstream data.

## Key gotchas — don't rediscover these the hard way

1. **Google Drive rate-limits anonymous bulk downloads** after roughly 100–150 rapid
   requests (a session-wide block, not per-file). `fetch_sample_pdfs.py` already
   handles this with exponential backoff (60s → 20min cap, gives up after 6h). Don't
   rewrite this naively with a simple retry loop.
2. **The HEADNOTE citation-sign convention only exists in pre-2000 judgments.** 0 of
   190 sampled 2000–2025 cases have one at all — it's a hard boundary (an old
   law-reporting style Indian Kanoon dropped), not a declining trend you can "try
   harder" on. Post-2000 sign extraction has to go through the LLM classifier instead.
3. **The provided citation metadata's "precedents" list mixes real case citations with
   statute/article citations** — only ~32.5% are genuine case-to-case cites. Always
   run through `classify_citations.py` before treating an edge as a case citation.
4. **Ontology terminology entries are word-sense ambiguous in real text.** `"approved"`
   fired on "approved by the Chief Minister" (administrative) not just "X v. Y ...
   approved" (judicial). Any single-word terminology entry needs validation against
   diverse real text, not just curated gold examples, before trusting it at scale.
5. **Grounding sometimes truncates multi-citation list sentences**, dropping the first
   party's name (e.g. "Krishna Shetti v. Gilbert Pinto" → just "Gilbert Pinto (I.L.R.
   ...)"). `phase3_verify.py`'s verification uses substring matching (not just exact
   "v./vs." pattern + fuzzy ratio) specifically to still catch these as real citations
   rather than false negatives.
6. **`RulingAction+Patient` is a genuinely overloaded grounding target** — "held that
   X" means Holding, RuleCited, Issue, or AuthorityCited depending on what X actually
   contains. Confidence is stuck around 0.5–0.6 despite more gold examples. This needs
   *content-based* disambiguation (does X mention a statute? a case name?), not just
   more (class, role) pairs. Documented in the ontology's `known_gaps`.
7. **Distinguish `UNVERIFIED` from `REJECTED_NOT_A_CASE` from a real citation-metadata
   gap.** The citation metadata itself is incomplete (independently scraped, not
   ground-truth-perfect) — an "UNVERIFIED" authority could be genuinely fabricated,
   OR a real citation the extraction garbled, OR a real citation the metadata just
   doesn't have. Don't report "UNVERIFIED" as "hallucination rate" without this caveat.

## Current status (2026-09-18)

- [x] Data pipeline complete: citation classification, 500-case stratified sample, full
      corpus of 500 PDFs downloaded and verified (clean digitized text, no OCR needed)
- [x] Signed graph built (`Data/processed/signed_graph_edges.csv`,
      `proof_support_scores.csv`): 8,467 edges, 62.6% carry a real sign label after
      merging HEADNOTE + LLM-classifier sources; sanity-checked against known landmark
      cases (Angurbala Mullick, Deoki Nandan vs Murlidhar)
- [x] Phase I (GROUND/LEARN/EXTRACT) implemented and validated against real held-out
      sentences across multiple rounds — found and fixed several real bugs (terminology
      truncation, abstract-class leakage, RulingAction/DispositionAction conflation,
      missing DisputeAction triggers). 24 gold examples, 7 learned rules.
- [x] Phase III (verification + IRAC) implemented, validated on one full case
      (Kedar Nath Yadav vs State of West Bengal, the Singur land acquisition case) plus
      a 3-case smoke test. Verification mechanism confirmed working correctly — it
      flags spurious "authorities" rather than certifying them.
- [ ] **IN PROGRESS**: `run_evaluation_batch.py --n 50` was running in the background
      as of last session end (~12/50 cases done). Check
      `Data/processed/evaluation_summary.json` and `Data/processed/irac_batch/` for
      results. If it finished, commit + push those (they were deliberately excluded
      from the last commit since the run was still in progress). If the process died,
      just re-run it — it doesn't resume mid-batch but is idempotent to re-run.
- [ ] **NOT STARTED**: baseline comparisons (vanilla LLM prompting, standard RAG) for
      the proposal's Evaluation Plan section
- [ ] **NOT STARTED**: dashboard (Streamlit was the plan) for the live demo
- [ ] **NOT STARTED**: filling in real team member names/roles in the proposal
- [ ] **NOT DONE**: checking the professor's official proposal outline once posted

## Git

Remote: `https://github.com/myselfsiddharth/LegalAI.git`, branch `main`.
`Docs/`, `.env`, `.venv/`, `Data/raw_pdfs/`, and `__pycache__/` are gitignored.

## Suggested next steps, in likely priority order

1. Check on / finish the 50-case evaluation batch, commit + push results
2. Fold real evaluation numbers into the proposal doc
3. Consider tackling gotcha #6 (RulingAction ambiguity) if there's time before scaling
   Phase I further
4. Build the baseline comparisons (needed for the Evaluation Plan either way)
5. Build the dashboard for the demo
6. Get team member names and finalize the proposal formatting against the official
   template once posted
