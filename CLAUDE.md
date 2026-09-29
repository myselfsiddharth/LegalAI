# LexGraph — CSE 573 Legal AI Project

Last updated: 2026-09-26 (evening). This file exists so a fresh Claude session (or a teammate)
can pick this project up without re-deriving everything from scratch. Read this before
touching the code.

## What this is

A class project for **CSE 573: Semantic Web Mining** (ASU, Fall 2026, Prof. Hasan
Davulcu, hdavulcu@asu.edu, TA Anshul Trivedi). Assigned topic: **P15 "Legal AI –
Agentic Law"**. It's a scaled-down version of the professor's own Templeton Foundation
grant proposal ("Contestable Legal AI") — the original grant doc lives in `Docs/`
(gitignored, not in this repo: it's an unpublished/confidential funding document).

**The idea (REVISED 2026-09-26 — the pipeline was inverted; read this before touching
anything)**: given the **facts and evidence of a dispute as input**, predict the rules and
authorities that apply, then map them to elements and claims. Concretely: (1) retrieve
candidate precedents from the corpus using the facts alone, (2) rank them with a *signed*
citation graph (did later courts follow or reject this case — is it still good law?), and
(3) deterministically verify every authority against real citation data before producing an
IRAC-style answer, refusing to certify anything it can't verify.

**What changed and why.** The project originally ran the other way: it *extracted* typed
objects (claims, rules, citations, holdings) out of a finished judgment. That solves a
problem nobody has — in the real world you hold facts and evidence, never a decided case.
The professor described the same shape in class (Step 1: facts → rules/authorities;
Step 2: facts + authorities → elements and claims), and our own baselines agreed: dense
retrieval beat the ontology extraction pipeline at finding citations (51.8% vs 43.6%
verified, 9.9% vs 3.9% edge recall).

**Phase I extraction is therefore demoted, not deleted** — it is now a labelling and
citation-scrubbing tool, not the product. Do not invest further in its precision. What
survives as the actual contribution: the deterministic verifier (fabricated authorities
58% → ~1%) and the signed citation graph, neither of which the professor's sketch has.

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
  fetch_sample_pdfs.py        # LEGACY: Drive downloader, no longer needed (corpus is local)
  extract_headnote_signs.py   # pre-2000: regex-extracts citation sign from HEADNOTE text
  classify_modern_signs.py    # post-2000: LLM-classifies citation sign (no HEADNOTE exists)
  build_signed_graph.py       # merges both sign sources -> signed graph + Proof-Support Score
  ode_lib.py                  # shared GROUND (LLM) + rule-application logic
  ode_pipeline.py             # CLI: ground / learn / extract (Phase I)
  build_doc_id_map.py         # doc_id <-> PDF join by normalized title (see gotcha #15)
  cache_corpus_text.py        # one-off: extract text for every mapped case -> corpus_text.jsonl
  retrieval_lib.py            # BM25Index + DenseIndex (one search() contract) + scrub_citations
  build_dense_index.py        # embeds the corpus with qwen3-embedding-8b -> dense_index.npy
  run_precedent_prediction.py # THE NEW EVALUATION: facts -> authorities, held-out citations
  answer_from_facts.py        # end-to-end proof-carrying answer: facts -> IRAC, every cite certified
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
    doc_id_to_pdf.json        #   5,462 doc_id -> PDF paths (build_doc_id_map.py)
    corpus_text.jsonl         #   cached judgment text, 320MB, gitignored
    dense_index.npy / _meta.json  # 4096-dim document vectors
    precedent_prediction.json #   facts -> authorities results, all query-length arms
    irac_batch/               #   per-case pipeline outputs (Phase I+III), keyed by doc_id
    baselines/<system>/       #   per-case baseline outputs, incl. raw LLM response
    evaluation_summary.json   #   stratified pipeline run (the headline numbers)
    evaluation_summary_1950s.json  # the earlier 50-case, 1950s-only run, re-scored
    baseline_comparison.json  #   pipeline vs baselines on the same cases
  raw_pdfs/<year>/            # FULL corpus: all 26,688 judgment PDFs (gitignored, 6.8GB)
                              #   re-extract from the 4 supreme_court_judgments-*.zip (see gotcha #1)
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
# shared: corpus prep and the signed graph
python3 scripts/classify_citations.py
python3 scripts/build_doc_id_map.py                              # asserts its own calibration
python3 scripts/cache_corpus_text.py                             # ~20 min, resumable
python3 scripts/extract_headnote_signs.py
python3 scripts/classify_modern_signs.py                         # API key
python3 scripts/build_signed_graph.py

# CURRENT track: facts -> authorities
python3 scripts/run_precedent_prediction.py --n 100 --min-year 2000 --facts-chars 0 2000
python3 scripts/build_dense_index.py                             # ~85 min, resumable, API key
python3 scripts/run_precedent_prediction.py --retriever dense --n 100 --min-year 2000
python3 scripts/answer_from_facts.py --n 25 --k 20                # API key; grounded vs closed-book

# LEGACY track: extraction from finished judgments (kept for the baseline comparison)
python3 scripts/build_sample.py --target 500
python3 scripts/ode_pipeline.py learn                            # API key; 24 calls
python3 scripts/run_evaluation_batch.py --per-decade 6 --resume   # API key; ~1 min/case
python3 scripts/run_baselines.py --per-decade 6 --resume          # API key; after the above
```

**Long jobs must be detached** (`nohup ... & disown`), not run under a 10-minute tool
timeout — `build_dense_index.py` and `cache_corpus_text.py` both outlive it. Both are
resumable, so a killed run loses at most one batch.

`--resume` reuses per-case outputs already on disk, so an interrupted run continues where
it stopped. Both evaluation scripts re-verify saved outputs with the current verifier on
every run; after changing `phase3_verify.py`, `run_evaluation_batch.py ... --rescore-only`
re-scores everything with no LLM calls.

All of these are idempotent / safe to re-run — they either skip existing outputs or
fully regenerate deterministically from upstream data.

## Key gotchas — don't rediscover these the hard way

1. **Never extract the corpus zips with `unzip`.** Two of the four
   `supreme_court_judgments-*.zip` archives contain filenames with invalid UTF-8 bytes
   (e.g. `T_N_Godavarman_Thirumulpad_+<bad>_vs_...`). APFS rejects those names, `unzip`
   misreports it as `write error (disk full?)`, then blocks on a `Continue? (y/n)` prompt
   — on a non-interactive stdin it takes EOF and **silently aborts the whole archive**.
   That is how a first attempt produced 12,422 of 26,688 files and still exited 0.
   Extract with Python's `zipfile`, sanitizing each name (`encode("utf-8","replace")`,
   NFC-normalize, strip control chars), skip-if-same-size for resumability. Verify the
   count is 26,688 afterwards. The whole corpus is now local, so Google Drive and its
   rate limiting are out of the pipeline entirely.
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
11. **Candidate ORDER, not the keyword filter, sets the recall ceiling.** Judgments
    average ~515 keyword-matched sentences and we can only afford to ground a few dozen.
    Measured over the 48-case set: 88.9% of real citation edges have a party name
    somewhere in the text, 51.1% survive the keyword filter, but only **6.9%** fell inside
    the old `[:30]` document-order slice. That is why the pipeline worked on 1950s-60s
    judgments (HEADNOTE lists citations up front) and produced almost nothing after 1990.
    `phase3_verify.select_candidates` now spends the same budget on citation-shaped
    sentences first, raising the ceiling to 23.5% at identical cost. If you change
    selection, re-measure the ceiling before re-running anything expensive — it is a
    free, LLM-free calculation and it predicts the outcome.
12. **Learned rules are keyed on `(class, role)` and resolved winner-take-all, so
    overloaded pairs silently lose concepts.** `RulingAction.Patient` has confidence 0.57
    (4/7) because Holding won and the Issue / RuleCited gold examples were discarded.
    Consequence: `Issue`, `RuleCited` and `Fact` are **never emitted by any rule**, so the
    Issue and Rule sections of every IRAC report are empty — the I and R of IRAC. Only
    Holding, Claim, AuthorityCited, Party and Outcome are ever produced. Gold examples for
    all three missing concepts already exist and are being thrown away by `cmd_learn`.
13. **Learned rules are coupled to the grounding model that produced them.** Rules key on
    the `(class, role, guard)` triples a specific model emits, so rules learned with one
    model silently fire on nothing under another. Measured: with glm-5-3-learned rules but
    scout doing extraction, the Fact sentence came out as `Claim` and both Issue sentences as
    `Holding` — every new rule dead. If you change `--model` for `learn`, you must change it
    for `run_evaluation_batch.py` and `answer_from_facts.py` too, and re-run everything
    downstream. Cheapest guard: keep one model for both, and treat `learned_rules.json` as
    invalid whenever the grounding model changes.
14. **A per-item LLM loop without retries turns an outage into a finding.** Hit twice in
    one session. `mine_claim_elements.py` swallowed a transient `APIConnectionError` per
    sentence and reported "0 elements with support >= 2" for four of six claims -- which
    reads as *the corpus does not state these elements* when in fact 150 of 253 sentences
    were never asked. `map_facts_to_elements.py` then lost 23 of 80 calls the same way.
    Both now retry with quadratic backoff and count drops separately from genuine empty
    results. Any new loop that makes one call per item needs the same, and the summary line
    must distinguish "model said nothing" from "we never got an answer".
15. **Nothing in the dataset joins doc_ids to PDFs for the full corpus**, and the obvious
    normalizer is booby-trapped. The land-disputes spreadsheet covers only its own 6,954
    rows; `build_doc_id_map.py` joins the remainder on normalized titles. Filenames carry a
    copy marker after the year (`..._on_23_May_1957_1`) and titles do not, so stripping
    "trailing digits" from both removes the YEAR from titles and every pair then mismatches
    by exactly that year — it silently produced a 0.6% join rate before being caught. The
    script now asserts its rate against the 500 verified pairs in `sample_cases.csv` (94%)
    and refuses to write a join nothing downstream could trust.
16. **The corpus caps precedent recall at ~26%, and no method fixes that.** Of 46,904 case
    citation edges only 11,991 (25.6%) point at a case that is in this corpus at all —
    it is Supreme Court only, while judgments cite High Courts, the Privy Council and
    English decisions. Report recall against *reachable* authorities with the raw figure
    beside it, or the number is meaningless. 1,206 cases have their own PDF plus >=3
    reachable authorities; 771 of those are post-2000.
17. **Query cases must be post-2000.** Pre-2000 headnotes list the cited authorities up
    front, so a query built from such a judgment contains its own answer. This exactly
    inverts the old difficulty profile: the decades the extraction pipeline handled *best*
    are the ones precedent prediction cannot use.
18. **Voyager needs no VPN, and `qwen3-embedding-8b` is already the best embedder it has.**
    The endpoint resolves to public Cloudflare IPs and answers from a residential
    connection. `/v1/models` returns no capability metadata (just id/object/created/
    owned_by), so the only way to tell an embedder from a generator is to probe
    `/v1/embeddings` — opaque names like `muse-glimmer-30b` and `laguna-s-2-1` 404 there.
    Only `qwen3-embedding-4b`, `qwen3-embedding-8b` and a vision-language variant exist.
    The endpoint is *token*-throughput-limited, not request-limited, so batching more chunks
    per call buys almost nothing (0.95 s/doc vs 1.06). Generation is a different story: the
    pipeline defaults to `llama4-scout-17b` while `qwen3-235b-a22b-instruct-2507`,
    `glm-5-3`, `gpt-oss-120b` and `qwen35-122b-a10b` are all available and untested.

## Current status (2026-09-26)

- [x] Data pipeline complete: citation classification, 500-case stratified sample.
- [x] **Full corpus local (2026-09-24)**: all 26,688 judgment PDFs extracted to
      `Data/raw_pdfs/<year>/` (6.8GB), verified against the dataset readme's count.
      Google Drive is no longer in the pipeline. See gotcha #1 before re-extracting.
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
- [x] Evaluation: the original `--n 50` batch was all 1950s (gotcha #8) and its 17%
      verification rate was mostly verifier bugs (gotcha #9); it re-scores to 55%
      (`evaluation_summary_1950s.json`). The headline run is `--per-decade 6` (48 cases,
      `evaluation_summary.json`).
- [x] **Candidate-selection fix (2026-09-24)**, gotcha #11. Same 30 calls/case, re-run
      without `--resume`. The pre-fix arm is preserved as `evaluation_summary_docorder.json`
      and `Data/processed/irac_batch_docorder/` so the report has a controlled A/B:

      | decade | old cites/ver/recall | new cites/ver/recall |
      |--------|---------------------:|---------------------:|
      | 1950s  | 16 / 11 / 0.128      | 28 / 14 / 0.140      |
      | 1970s  |  5 /  3 / 0.010      | 47 / 20 / 0.050      |
      | 1990s  |  2 /  0 / 0.000      | 25 /  7 / 0.036      |
      | 2020s  |  0 /  0 / 0.000      | 24 / 17 / 0.022      |

      The real finding is not "recall 1.4% -> 3.9%" but that the old pipeline emitted
      **five citations total across 1990-2025** on 24 judgments. The much-quoted "60%
      verified" was precision over 50 citations, 34 of them 1950s-60s. It now runs across
      the whole 1950-2025 range, and the 2020s has the best precision of any decade
      (17/24) because modern judgments use clean SCC/AIR reporters.
- [x] Baselines (2026-09-18): closed-book LLM + dense RAG on the same 48 cases, scored
      by the same verifier (`baseline_comparison.json`, folded into proposal §6.1):

      | system      | case cites | verified     | not named in judgment | edge recall |
      |-------------|-----------:|-------------:|----------------------:|------------:|
      | closed-book | 174        | 12 (6.9%)    | 101                   | 0.5%        |
      | RAG         | 417        | 216 (51.8%)  | 5                     | 9.9%        |
      | LexGraph    | 220        | 96 (43.6%)   | 3                     | 3.9%        |

      **The honest takeaway, updated 2026-09-24: RAG still beats LexGraph on both recall
      and verified rate.** The gotcha #11 fix closed most of the recall gap (1.4% -> 3.9%
      vs RAG's 9.9%) but did not overtake it, and it cost the old "zero fabrications"
      line: normalized, RAG is 5/417 = 1.2% absent-from-judgment vs LexGraph 3/220 = 1.4%.
      What the numbers *do* support is that **the deterministic verifier is the
      contribution, not Phase I extraction** — the same `verify_authority` cuts fabricated
      authorities from 58% (closed-book, 101/174) to ~1% for either retrieval method. Say
      that, not "our pipeline beats RAG". Note the `rag` row already IS the "RAG +
      verifier" ablation §6.1 promises, since eval_lib scores every system identically.
      Remaining Phase I weakness: 196 of 424 outputs are REJECTED_NOT_A_CASE (statute
      spans misrouted into AuthorityCited), which is gotcha #12's root cause.
- [x] VERIFIED audit per gotcha #9 (2026-09-24): of 96, **73 are clean single-edge
      matches** (including the truncation/OCR cases the matcher was tuned for). 23 are
      ambiguous (>1 candidate edge) and at least 3 are wrong — `"the statement of the law
      laid down in the cases of"` verified against 3 edges, `"Union Government of India"`
      matched 12, and the 2022 Wakf Board case verified its own title, because generic
      institutional parties (`State of A.P.`, `Union of India`) match many edges. Modern
      judgments carry more of these, so citation-ranked selection surfaces more of them.
      `verify_authority` checks `matches` before SELF_REFERENCE, so a self-citation that
      also matches an edge is reported VERIFIED.
- [x] Proposal docx: added §6.1 results, rebuilt the three tables (they had been
      flattened into loose paragraphs), fixed schema errors, and normalized all text to
      12pt Times New Roman (it had 11pt table/path text and Courier New code spans).
- [x] **Content guards (2026-09-26)**, gotcha #12. Rules are now keyed
      `(class, role, guard)` where `ode_lib.content_guard` reads the filler's content as
      question / case / statute / other. Result on the same 48 cases:
      REJECTED_NOT_A_CASE **196 -> 0**, outputs 424 -> 228, and case-shaped / VERIFIED /
      rate / recall **byte-identical** (220 / 96 / 0.436 / 0.039). Pure precision, zero
      recall cost. `RuleCited: 38` objects — IRAC's Rule section is populated for the first
      time. Arms preserved: `irac_batch_docorder/`, `irac_batch_preguard/`.
      Two traps found by the gold-example gate and worth not repeating: the `statute` guard
      must require the provision in SUBJECT position (both real RuleCited fillers read
      "that Section 238A ... would not extend", while colliding Holding/Issue fillers only
      reach a section inside the predicate) — without that, RuleCited sat at confidence
      0.33; and emitting an unguarded `*` fallback for *every* pair defeats the whole fix,
      since statute spans keep reaching AuthorityCited through it. Fallbacks now exist only
      for pairs genuinely seen with more than one content shape.
      **Still missing: `Issue`.** Not a guard problem — grounding truncates "What are the
      consequences of" out of the filler, so what reaches the guard is not question-shaped.
- [x] **Precedent prediction works (2026-09-26)** — the new track. 100 post-2000 query
      cases, citations held out as labels, BM25 over 5,462 cached judgments, authorities
      constrained to predate the judgment:

      | k   | full judgment | 2,000-char facts | popularity control |
      |-----|--------------:|-----------------:|-------------------:|
      | 10  | 31.0%         | 15.6%            | 5.3%               |
      | 50  | 51.5%         | 30.4%            | 14.8%              |
      | 100 | 62.4%         | 42.1%            | 23.7%              |

      precision@10 16.5% / 8.3%. Recall is against *reachable* authorities (gotcha #16).
      Two things make this credible: it beats the popularity control 3-6x, so it is not
      just exploiting citation skew; and cutting the query from ~60k chars to 2,000 — 30x
      less text — costs only half the recall@10, meaning the FACTS carry most of the
      signal, not the court's reasoning. On the old comparable basis the strict facts-only
      arm gets 5.0% of all cited authorities at k=10 against the extraction pipeline's 2.2%
      on the same post-2000 slice.
      Party-name leakage was checked and is minor: correct hits share a party name at
      16.9% vs an 8.5% base rate, worth ~1.7 points of recall@10 (15.6% -> 13.9% if every
      same-party case is dropped). That floor over-corrects — it also removes legitimate
      same-matter appeals like `A.P. Pollution Control Board II` citing the earlier
      `A.P. Pollution Control Board` — and the detector over-flags common surnames
      ("singh") and descriptors ("transport"). The headline stands.
- [x] **Proof-carrying answers work (2026-09-26)** — `answer_from_facts.py`, 25 cases, k=20.
      Retrieval alone cannot fabricate (every candidate is a real corpus doc_id), so the
      verifier only earns its place one step later, when a model drafts the answer:

      | arm         | proposed | CERTIFIED | refused: real, unbriefed | refused: resolves to NOTHING | groundedness |
      |-------------|---------:|----------:|-------------------------:|-----------------------------:|-------------:|
      | grounded    | 61       | 61        | 0                        | 0                            | 100.0%       |
      | closed_book | 61       | 3         | 12                       | **46**                       | 4.9%         |

      **46 of 61 closed-book citations resolve to nothing across all 5,462 judgments** — a
      75% fabrication rate, driven to zero by drafting from a retrieved brief and checking
      every cite. Use this, not the older 58% -> 1% figure; it is better instrumented.
      **The honest limit: only 13.1% of the 61 certified authorities were really cited by
      the court.** Groundedness is not correctness — the verifier guarantees provenance
      (real, retrievable, pre-dating), not that the right case was picked.
      Certification is two-stage (verbatim, then party matching) on purpose: the party
      matcher's MIN_COVERAGE rightly refuses "State of Bombay v. Advani" for "Province Of
      Bombay vs Kusaldas S Advani" (gotcha #9), and without the split a refusal for sloppy
      paraphrase is indistinguishable from one for fabrication.
- [x] **Dense vs BM25 (2026-09-26): it depends on query length.** Both retrievers share one
      `search()` contract and one evaluation path. Recall, 100 post-2000 cases:

      | k   | BM25 facts | Dense facts | BM25 full | Dense full | popularity |
      |-----|-----------:|------------:|----------:|-----------:|-----------:|
      | 10  | 15.6%      | **16.2%**   | **31.0%** | 25.9%      | 5.3%       |
      | 50  | 30.4%      | **33.7%**   | **51.5%** | 48.5%      | 14.8%      |
      | 100 | **42.1%**  | 41.3%       | **62.4%** | 58.7%      | 23.7%      |

      Dense wins on short realistic fact queries, BM25 on long ones where exact overlap on
      statute and doctrine names beats a single mean-pooled vector. The @10 gap is within
      noise on 532 authorities; the @50 gap looks real. Hybrid fusion is the untried lever.
- [x] **Grounding model benchmarked (2026-09-26), and it is not about size.** Five Voyager
      models on the 24-example LEARN gate:

      | model                         | matched/24 | keys | recovers `Fact` |
      |-------------------------------|-----------:|-----:|-----------------|
      | **glm-5-3**                   | **16**     | **9**| **yes**         |
      | llama4-scout-17b              | 16         | 8    | no              |
      | qwen3-235b-a22b-instruct-2507 | 15         | 7    | yes             |
      | gpt-oss-120b                  | 11         | 6    | no (6 empty)    |
      | qwen35-122b-a10b              | 5          | 3    | no (16 API errors) |

      Two of the five larger models score *below* the 17B baseline and the 122B one is
      effectively broken here. Grounding rewards instruction discipline (exact substrings,
      a fixed class list, no commentary), not capability — a model that paraphrases is
      penalised by the similarity match. **But `glm-5-3` is ~12x slower than scout
      (~25s/call vs ~2s)**, which matters for any per-sentence batch job.
- [x] **Swapping the grounding model does NOT fix `Fact` / `Issue` (tested 2026-09-26).**
      `learn --model glm-5-3` does produce 11 rules including `ProceduralAction.Time[other]
      -> Fact` and `DispositionAction.Agent[other] -> Issue`, so all 8 concepts finally have
      a rule. But both routes are semantically accidental single-example artifacts and they
      do not generalise: run against real sentences, the `Fact` rule fires on **nothing**
      (even under glm-5-3, on the very sentence type it was learned from), and `Issue` fires
      on one of two. Reverted to the committed scout rules rather than destabilise a measured
      pipeline for a capability that does not work. The real fix is contrastive gold
      examples, per next step #1.
- [x] **Step 2 built (2026-09-28): facts -> elements, with checkable support.**
      `mine_claim_elements.py` mines the requirements of a claim from judgments that state
      them, rather than anyone authoring a list -- an invented element would be worse than an
      invented citation, since nothing downstream would catch it. Window matching failed
      first (enumeration markers like "(i)" mark numbered findings and statutory sub-clauses
      in Indian judgments, not requirement lists); requiring the claim term and a requirement
      trigger in the SAME sentence works. 29 elements survive support >= 2 distinct
      judgments: adverse possession 17, land acquisition 5, injunction 4, specific
      performance 3; easement 0 (only 7 candidate sentences corpus-wide) and partition 0 (44
      raw, none cluster). The injunction result is the method validating itself -- prima
      facie case / balance of convenience / irreparable injury, the textbook three-part test,
      with one 2001 judgment stating all three in a single sentence.
      `map_facts_to_elements.py` then decides each element, and **a SATISFIED verdict is
      only accepted if its supporting quote occurs in the facts shown** (normalized, min 25
      chars). 12 cases x 10 elements: 60 verified, **2 refused (3.2%)**, 27 not satisfied,
      30 unclear, 0 dropped. The two refusals are distinct real failures -- a two-word
      fragment too short to be evidence, and a quote containing "..." where the model elided
      text and presented a reconstruction as a quotation. This yields a fabrication metric
      needing **no ground truth**: it catches manufactured support, not wrong reasoning.
      Two known weaknesses: the catalogue clusters on normalized strings so semantic
      duplicates survive (`nec vi`/`nec clam`/`nec precario` are one classical triple;
      `hostile possession` duplicates `hostile to the real owner`), and negatively-framed
      elements score badly -- `nec vi` is 2 satisfied / 6 unclear because the model reads
      silence as failure. Ontology now carries `Element` plus `Claim -requires-> Element`
      and `Element -satisfiedBy-> Fact`.
- [ ] **NOT STARTED**: dashboard (Streamlit was the plan) for the live demo
- [ ] **NOT STARTED**: filling in real team member names/roles in the proposal
- [ ] **NOT DONE**: checking the professor's official proposal outline once posted

## Git

Remote: `https://github.com/myselfsiddharth/LegalAI.git`, branch `main`.
`Docs/`, `.env`, `.venv/`, `Data/raw_pdfs/`, and `__pycache__/` are gitignored.

## Suggested next steps, in likely priority order

1. **`Fact` extraction — the input side of the whole pipeline is empty.** The professor's
   Step 1 takes evidence and facts as input and we extract exactly zero `Fact` objects. The
   failure is at GROUNDING, not at the rule step (both gold examples score 0.30 and 0.35
   against a 0.45 floor), so content guards cannot reach it. **The model swap was already
   tried and does not work** (see status above and gotcha #13) — so this needs
   **contrastive gold examples** for `Fact` and `Issue`: several each, drawn from real
   judgment text rather than curated sentences, chosen to sit next to the Holding and Claim
   examples they currently lose to. The 24-example LEARN gate is the benchmark and costs 24
   calls per run, so iteration is cheap.
2. **Fact segmentation for real queries.** `--facts-chars N` takes the first N characters
   of scrubbed text as a proxy for the fact narrative, which is crude and drags the case
   caption into the query. There are no `FACTS` section headers to lean on: sampled across
   six decades, `FACTS` appeared as a header in **0** of 12 judgments, `ISSUE` in 2,
   `ARGUMENT` in 2, `REASONING` in 0 — the dataset readme's claim of structured sections is
   not true of the PDFs. This needs a real segmenter or a sentence classifier.
3. **Finish the dense arm and pick a retriever.** `build_dense_index.py` then
   `run_precedent_prediction.py --retriever dense`. Both retrievers share one `search()`
   contract and one evaluation path, so the table is directly comparable. If dense loses,
   check the representation before the method — documents are 10 mean-pooled chunks.
4. **Wire the verifier onto retrieval output.** Predicted authorities should go through
   `verify_authority` and be certified or refused, which turns precedent prediction into the
   proof-carrying pipeline the project is actually about, and gives the demo its story.
5. **Rank by the signed graph.** Re-rank retrieved candidates by proof-support and edge
   polarity — "is this still good law" is the differentiator nobody else's sketch has. Temper
   expectations: only **174 of 8,467** edges are negative (5 `overruled`, 17
   `overruled_or_disapproved`, 141 `distinguished`), so the effect will be small and
   proof-support is partly in-degree, which risks collapsing toward the popularity control.
6. **`Element` in the ontology** — the professor's Step 2, and absent entirely. Needs
   `Claim -> requires -> Element` and `Element -> satisfied_by -> Fact`, plus gold examples.
   The `(class, role, guard)` rule keying is the template: deciding which element a fact
   satisfies is the same shape of problem as the content guard.
7. **Rewrite the proposal for Oct 7.** §6.1's numbers are stale and its argument ("we beat
   RAG") is one the data contradicts. Lead with the two defensible results: the verifier cuts
   fabricated authorities 58% -> ~1% regardless of retrieval method, and facts-only
   precedent prediction beats the popularity control 3x. Fill in the team roster and check
   the official template.
8. Build the dashboard for the demo.
