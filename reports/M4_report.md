# M4 — Grounding facts in law (§9)

**Status: §9.1, §9.2 and §9.3 built and measured.**
Reproduce: `python -m src.authorities.statute_predict --split forum_heldout --rebuild-targets`.

**Headline: statute prediction is the first task on this project with clear positive signal from
the facts** — 1.9× the strongest baseline — in sharp contrast to outcome prediction, which sits at
chance. But getting there required discarding a first result that was an artifact.

## 1. Statute normalisation (§9.1)

`src/authorities/normalize.py` reduces citations to canonical `CODE:provision` form (`TPA:53A`,
`SRA:16(c)`, `CONST:Art300A`) via a registry of 42 Acts with alias patterns, seeded from the
ontology's §3.4 statutory framework list and extended with what the corpus actually cites.

Three properties of real Indian citations forced the design:

- **Act names arrive truncated.** The most frequent "Act" in 1,200 sampled judgments is
  `Property Act` (644 mentions) — *Transfer of Property Act* with its head cut off by a line break.
  `India Act` (516) is *Government of India Act*. Truncations resolve through a separate path tagged
  `resolved_by="truncated"`, so a guess is never reported as a full-name match.
- **Generic back-references dominate.** `the Act`, `the said Act`, `the Amending Act` (311+
  mentions). Resolvable only by looking back to the last Act actually named, within a bounded
  window.
- **Articles and Orders carry implicit Acts.** An `Article` with no Act named is the Constitution;
  an `Order`/`Rule` is the CPC.

Coverage over 1,500 judgments: **58,295 citations, 61.4% resolved** to a canonical code
(12.2% named, 19.9% kind-default, 29.0% context, 0.2% truncation), median 6 distinct provisions per
case, 149 cases with none.

### 1.1 A misattribution bug, and why coverage went DOWN after fixing it

The first version resolved 74.3%. Auditing the output showed `section 6 of the Constituent Assembly
Act` resolving to **`CONST:6`**, and `sections 3 and 4 of the Extra-Provincial Jurisdiction Act` to
`CONST:3`/`CONST:4`. The Act *was* named — it simply was not in the registry — so `resolve_act`
returned nothing and **the context resolver reassigned the citation to whatever Act was mentioned
nearby.**

A citation attributed to the wrong statute is worse than one left unresolved, because §9.2 trains
on it as a target. Two rules now apply:

- an explicitly **named** Act the registry does not know stays `UNRESOLVED`, never falling through
  to context;
- a **section** is never attributed to the Constitution, which is cited by Article.

Coverage fell 74.3% → **61.4%**, and the spurious `CONST:3/4/5/6/7` entries vanished from the top
provisions. The lower number is the correct one.

Top provisions now read as they should for a 1950–2025 land corpus: `CONST:Art31` (1,371),
`CONST:Art368`, `CONST:Art14`, `CONST:Art226`, `CONST:Art19(1)(f)` — the pre-1978 property-rights
articles — then `LA:6`, `LA:4`, `CPC:92`.

## 2. Statute prediction (§9.2)

### 2.1 The first result was an artifact, and a control caught it

Targets taken from the **full** judgment gave an encouraging table: masked text micro-F1 0.447,
P@1 0.582, against a global prior of 0.106. Before reporting it, one control:

> **`copy from masked text`** — predict exactly the provisions that appear in the input.
> **micro-F1 0.713, P@1 0.754.**

Copying beat every learned model. The statutes a court relies on are mostly already named in the
facts and pleadings, so predicting them was never inference — it was extraction, which §9.1's
normaliser already does better than a classifier infers it.

§5.2 anticipates precisely this: *"only use authorities cited in pleadings/arguments, or predicted
authorities. Court-cited authorities are targets."* Targets drawn from the full text overlap the
inputs by construction.

### 2.2 The well-posed task: provisions the court introduced

`labels_novel` = provisions resolved in the judgment but **absent from `masked_text`** — the ones
the court reached for that nobody pleaded. Mean 1.0 per case (against 2.7 for all provisions), 46
provisions clearing a 30-train-case floor.

| system | micro-F1 | macro-F1 | P@1 | P@5 | R@5 | R@10 |
|---|---|---|---|---|---|---|
| prior (top-5 global) | 0.069 | 0.008 | 0.134 | 0.085 | 0.281 | 0.418 |
| family prior (top-5) | 0.093 | 0.036 | 0.173 | 0.115 | 0.342 | 0.503 |
| `copy from masked text` | **0.000** | 0.000 | 0.081 | 0.070 | 0.256 | 0.385 |
| **masked text** | **0.173** | **0.099** | **0.236** | **0.167** | **0.491** | **0.660** |
| extracted fact text | 0.159 | 0.093 | 0.206 | 0.153 | 0.451 | 0.581 |
| canonical atoms | 0.109 | 0.076 | 0.161 | 0.106 | 0.296 | 0.464 |

n = 2,615 train / 702 test.

**The copy baseline at exactly 0.000 is a construction check, not a finding** — targets are defined
as provisions absent from the input, so copying must score zero. It confirms the split is clean.

**Results.**
1. **Masked text reaches 1.9× the strongest baseline** on micro-F1 (0.173 vs 0.093) and R@5 (0.491
   vs 0.342). The facts genuinely predict which statutes a court will invoke unprompted.
2. **Extraction preserves most of it** — fact text 0.159 against masked text's 0.173, a far smaller
   loss than on outcome prediction.
3. **Canonical atoms degrade but are not worthless** — 0.109, still above the family prior's 0.093.
   The same ordering as the representation ladder (text > facts > atoms), but here every rung beats
   the baselines.
4. The `family prior` beats the global prior throughout, so part of any model's score is "land
   acquisition cases cite the Land Acquisition Act". The margin over the *family* prior is the part
   that reads the facts.

### 2.3 Why this matters for the project

Outcome prediction sits at chance (`reports/M5_report.md`) and §8's patterns are all
non-significant. It would be easy to conclude the extracted facts carry nothing. **They do — just
not about who wins.** Statute prediction is the demonstration, and it is also the first task where
the canonical atoms clear a baseline at all.

## 3. Precedent retrieval (§9.3)

Rank earlier cases by how likely the court was to cite them. 150 queries with ≥3 reachable
precedents, 1,035 reachable targets, pool of 6,849 cases.

| system | input | R@10 | R@50 | MRR | nDCG@10 |
|---|---|---|---|---|---|
| `mention` (copy control) | — | 0.022 | 0.112 | 0.037 | 0.014 |
| `popularity` | — | 0.032 | 0.103 | 0.071 | 0.032 |
| `bm25` | masked text | 0.207 | 0.350 | **0.431** | 0.224 |
| `dense_meanpool` | facts | 0.053 | 0.130 | 0.114 | 0.054 |
| `dense_maxpool` | facts | 0.039 | 0.104 | 0.097 | 0.044 |
| `dense_top3` | facts | 0.056 | 0.127 | 0.144 | 0.065 |
| `atom_overlap` | atoms | 0.021 | 0.058 | 0.046 | 0.021 |
| `dense_masked_text` | masked text | **0.229** | 0.389 | 0.380 | 0.221 |
| **`rrf(bm25+dense_masked_text)`** | masked text | **0.238** | **0.453** | 0.428 | **0.242** |

**Time-respecting retrieval asserted, not assumed:** 0 of the retrieved cases post-date their
query, checked per result rather than trusted to the `before_year` filter.

**Reachability.** Corpus-wide, only 22.1% of case-to-case citations point at a case this corpus
contains. Within the selected queries it is 49.0% — a selection effect, since queries were chosen
for having ≥3 reachable precedents. **Recall above is against the reachable set**, and the raw
figures are stored in the results JSON.

### 3.1 Findings

1. **BM25 retrieves precedents from facts at 6.5× the popularity control** (R@10 0.207 vs 0.032)
   and 9.4× the mention control. MRR 0.431 means the first correct precedent typically lands around
   rank 2–3.
2. **The copy control passes cleanly here** (0.022), unlike §9.2 where it beat every model. Queries
   are `masked_text` with citation-bearing sentences scrubbed, so a precedent cannot be read off
   the input — the retrieval is real.
3. **Dense retrieval is not the problem — the extracted facts are.** Dense over fact embeddings
   scores 0.039–0.056 under *every* pooling tried, while dense over the **masked text** reaches
   **0.229**, slightly ahead of BM25.

   This corrects a diagnosis stated in an earlier version of this report. The hypothesis was that
   mean-pooling ~17 fact embeddings washed out the specifics identifying a precedent, and that
   max-pooling would fix it. **Max-pooling made it worse** (0.039 vs 0.053) — a single best
   fact-to-fact match rewards boilerplate ("the appellant filed an appeal"), which the mean at
   least averages away. A top-3 mean, the middle ground, reached only 0.056.

   Embedding the masked text directly is what settled it: the signal is in the text and does not
   survive into the facts. **The facts carry 4× less precedent-relevance signal than the text they
   came from** (0.056 vs 0.229).
4. **RRF fusion helps once it has a second strong arm.** With the weak fact-based dense arm it
   *hurt* (0.155 against BM25's 0.207); with the text-based one it gives the best system overall
   (R@10 0.238, R@50 0.453, nDCG 0.242) — 7.4× the popularity control. Fusion is not free; it
   amplifies or dilutes depending on what it is given.
5. **`atom_overlap` sits at baseline level** (0.021, against popularity's 0.032). The canonical
   atoms fail here as they do everywhere else.

### 3.2 The cross-task pattern is now consistent

Three independent tasks, same ordering of representations:

| task | masked text | extracted facts | canonical atoms | baseline |
|---|---|---|---|---|
| outcome (AUROC) | 0.657 | 0.614 | **0.517** | 0.500 |
| statutes (micro-F1) | 0.173 | 0.159 | **0.109** | 0.093 |
| precedents (R@10) | **0.238** | 0.056 | **0.021** | 0.032 |

**The canonical atoms lose to, or barely clear, the baseline on every task.** That is the
discretisation finding of `reports/M5_report.md` §2, replicated across three tasks of quite
different shape — which is much stronger evidence than the outcome task alone could give.

## 4. §9 acceptance

| requirement | status |
|---|---|
| §9.1 `AuthorityRecord` normalisation, `Act:Section` form | **partial** — statutes done (61.4% resolved); case-citation canonicalisation not built |
| §9.2 multi-label statute prediction with baselines | **met** — 3 representations, 3 baselines incl. the copy control |
| §9.2 micro/macro-F1, P@k, R@k | **met** |
| §9.2 association-rule classifier arm | **not built** — §8 produced no significant rules to classify with |
| §9.2 `LLM-0` / `LLM-FS` statute prediction | **not built** |
| §9.3 precedent retrieval, Recall@k / MRR / nDCG | **met** — 6 systems incl. 2 controls |
| §9.3 time-respecting retrieval asserted in code | **met** — asserted per result, 0 violations |
| §9.3 pattern-overlap similarity retriever | **met** — `atom_overlap`, IDF-weighted Jaccard |
| §9.3 hybrid fusion (RRF) | **met** — and it *hurts* here |
| §9.3 LLM re-ranker | **not built** |
| citation graph | exists from earlier work (8,467 signed edges) |

## 5. Next

1. ~~Fix the dense arm by max-pooling~~ — **done and the hypothesis was wrong** (§3.1 item 3).
   Dense works on masked text and fails on facts under every pooling; the deficit is in the facts.
2. **Re-rank by the signed citation graph** — 8,467 edges already exist with polarity, and "is this
   still good law" is a question none of these retrievers asks. Temper expectations: only 174 edges
   are negative.
3. **Feed §9's output into §10's empty `S` and `R` feature groups.** Predicted statutes and the
   outcome distribution of retrieved precedents are the two groups §10.1 specifies and has never
   had.
4. `LLM-0` / `LLM-FS` arms for both §9.2 and §9.3, to complete the baseline set.
