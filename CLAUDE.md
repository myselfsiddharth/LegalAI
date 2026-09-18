# LexGraph — CSE 573 Legal AI Project

Last updated: 2026-09-18 (evening). This file exists so a fresh Claude session (or a teammate)
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
  run_baselines.py            # closed-book LLM + dense RAG baselines, same verifier/metrics
  eval_lib.py                 # shared case selection, authority scoring, metric definitions
  ontology/
    land_dispute_ontology.json  # classes, roles, ie_concepts, terminology
    gold_examples.jsonl          # 24 hand-picked (sentence, concept, filler) examples
    learned_rules.json           # output of `ode_pipeline.py learn`
  requirements.txt
Data/
  Legal AI Dataset/           # original provided dataset (citations JSON + land-dispute index)
  processed/                  # everything derived by the scripts above
    irac_batch/               #   per-case pipeline outputs (Phase I+III), keyed by doc_id
    baselines/<system>/       #   per-case baseline outputs, incl. raw LLM response
    evaluation_summary.json   #   stratified pipeline run (the headline numbers)
    evaluation_summary_1950s.json  # the earlier 50-case, 1950s-only run, re-scored
    baseline_comparison.json  #   pipeline vs baselines on the same cases
  raw_pdfs/                   # downloaded judgment PDFs (gitignored, ~155MB, regenerate via fetch_sample_pdfs.py)
LexGraph_Proposal_Draft.docx  # course proposal, A4/1in/12pt Times New Roman formatted
.env                          # VOYAGER_API_KEY (gitignored, not in repo -- see below)
```

## Environment setup (do this first in a new session)

1. `.venv/bin/pip install -r scripts/requirements.txt` (gdown, pypdf, openai, networkx) --
   the project `.venv` exists but may be missing packages; system python has none of them
2. Create `.env` at project root: `VOYAGER_API_KEY=<key>` (get one at
   voyager.rc.asu.edu → LLM Access → Create Key). All scripts that call the LLM load
   this automatically (`ode_lib.load_dotenv()`).
3. **ASU VPN may or may not be needed** for the Voyager endpoint (`openai.rc.asu.edu`).
   It used to resolve to a private `10.139.64.x` address (VPN-only); as of 2026-09-18 it
   resolves to public Cloudflare IPs and answered over a plain residential connection.
   If requests hang, check `dig +short openai.rc.asu.edu` and try the VPN.
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
python3 scripts/run_evaluation_batch.py --per-decade 6 --resume  # needs API key; ~1 min/case
python3 scripts/run_baselines.py --per-decade 6 --resume         # needs API key; run after the above
```

`--resume` reuses per-case outputs already on disk, so an interrupted run continues where
it stopped. Both evaluation scripts re-verify saved outputs with the current verifier on
every run; after changing `phase3_verify.py`, `run_evaluation_batch.py ... --rescore-only`
re-scores everything with no LLM calls.

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
   ...)"). `phase3_verify.py` matches on normalized *party names* (either party is
   enough evidence, with a coverage floor) specifically to still catch these as real
   citations rather than false negatives. See #9 before changing the matcher.
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
   The evaluation now splits UNVERIFIED by whether the name appears in the judgment text
   at all (`unverified_named_in_judgment` / `_not_in_judgment` / `_name_too_short_to_check`)
   — only "not in judgment" is a real fabrication signal. The metadata is Indian Kanoon's
   links, so English and Privy Council authorities are systematically missing from it.
8. **`sample_cases.csv` is sorted by decade**, so `--n 50` means "the first 50 1950s
   cases", not a sample of the corpus. Use `--per-decade K` for anything you'll report.
9. **The verifier's heuristics were tuned against a hand-audit, so re-audit if you change
   them.** Things that went wrong before: Indian reporter citations sit *outside*
   parentheses (`Ayyappa Reddy, (1913) I.L.R. 38 Mad. 738`); a case-insensitive reporter
   regex accepts any parenthetical (`clause (iv)`); whole-string fuzzy ratios let a shared
   generic prefix carry a mismatch (`Commissioner of Income-tax, Madras` vs `..., Bihar and
   Orissa`); single-word containment verified `Bhubneshwar Prasad Narain Singh` against
   `Pannalal v. Naraini`. After any matcher change, print every VERIFIED match with its
   matched edge and read the list. A false VERIFIED is worse than a false UNVERIFIED.
10. **A single dead HTTP connection can stall a batch silently.** The OpenAI client
    default (600s timeout x 3 attempts) froze a run for 20+ minutes with no output;
    `ode_lib.get_client` now uses a 90s timeout. Batch runners catch per-case errors and
    continue — check `failed_cases` in the summary, then re-run with `--resume`.

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
- [x] Evaluation (2026-09-18): the original `--n 50` batch finished, but it was all
      1950s (see gotcha #8) and its 17% verification rate was mostly verifier bugs
      (gotcha #9). After the fix it re-scores to 55% (`evaluation_summary_1950s.json`).
      The headline run is `--per-decade 6` (48 cases, `evaluation_summary.json`).
- [x] Baselines (2026-09-18): closed-book LLM + dense RAG on the same 48 cases, scored
      by the same verifier (`baseline_comparison.json`, folded into proposal §6.1):

      | system      | case cites | verified   | not named in judgment | edge recall |
      |-------------|-----------:|-----------:|----------------------:|------------:|
      | closed-book | 174        | 12 (6.9%)  | 101                   | 0.5%        |
      | RAG         | 417        | 216 (52%)  | 5                     | 9.9%        |
      | LexGraph    | 50         | 30 (60%)   | 0                     | 1.4%        |

      **The honest takeaway: plain RAG beats LexGraph on coverage.** Phase I only grounds
      the first 30 keyword-matched sentences. That works on 1950s–60s judgments, whose
      HEADNOTE lists citations up front (LexGraph recovers 22/169 edges vs RAG's 34), and
      fails on later ones (16 case citations across 36 judgments). Also, 180 of 235
      AuthorityCited objects are statute/section spans, not cases (Phase I precision).
- [x] Proposal docx: added §6.1 results, rebuilt the three tables (they had been
      flattened into loose paragraphs), fixed schema errors, and normalized all text to
      12pt Times New Roman (it had 11pt table/path text and Courier New code spans).
- [ ] **NOT STARTED**: dashboard (Streamlit was the plan) for the live demo
- [ ] **NOT STARTED**: filling in real team member names/roles in the proposal
- [ ] **NOT DONE**: checking the professor's official proposal outline once posted

## Git

Remote: `https://github.com/myselfsiddharth/LegalAI.git`, branch `main`.
`Docs/`, `.env`, `.venv/`, `Data/raw_pdfs/`, and `__pycache__/` are gitignored.

## Suggested next steps, in likely priority order

1. Fix Phase I candidate selection (the recall bottleneck above): feed retrieval-selected
   passages, e.g. `run_baselines.retrieve()`, into `extract_concepts` instead of the first
   30 keyword-matched sentences. Then re-run `run_evaluation_batch.py --per-decade 6`
   *without* `--resume` (so cases are re-extracted) and `run_baselines.py --per-decade 6
   --resume` to rebuild the comparison. Also run "RAG + verifier" as an explicit
   ablation, as §6.1 promises.
2. Phase I AuthorityCited precision: most extracted "authorities" are statute sections.
   Related to gotcha #6 (RulingAction ambiguity), and needs content-based disambiguation.
3. Open `LexGraph_Proposal_Draft.docx` in Word to eyeball it. It passes schema validation
   and was previewed via QuickLook, but no full Word/LibreOffice render was possible here.
   Then fill in team names, check the timeline rows past Oct 6 (Phase III verifier and
   baseline comparisons are listed as future work but already exist in prototype), and
   check it against the professor's official template once posted.
4. Build the dashboard for the demo
