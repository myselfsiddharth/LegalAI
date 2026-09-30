# From Evidence to Conclusion: Traceable Judgment Prediction for Indian Land & Property Disputes

> **The program is in the data. The proof is in the trace.**

This document is the build specification for an autonomous coding/research agent. Read it top to bottom before writing any code. Every stage has explicit inputs, outputs, and acceptance criteria. Do not skip a stage's acceptance check before starting the next one.

---

## 0. Read This First (Agent Operating Rules)

1. **The LLM is always the baseline.** Every experiment that reports a number for our method must report the same number for the LLM baseline on the same split, same cases, same metric.
2. **The LLM is accessed only through the ASU Voyager API**, configured in the existing `.env` file. Never hard-code keys, never print key values, never commit `.env`. See §3.
3. **The database already exists.** Do not re-scrape or rebuild it. Your first job is to *discover* its schema and write an adapter (§4).
4. **No outcome leakage.** The target is the judgment. Any text or feature that reveals the outcome (operative order, "held", "allowed/dismissed", the court's own reasoning, authorities cited *by the court in its analysis*) must never be an input to a predictor. See §5. This is the single most important correctness rule in the project.
5. **Everything is traceable.** Every canonical fact carries a `source_span`. Every prediction produced by our method emits a trace (§11) from conclusion back to evidence.
6. **The ontology is editable.** `ontology/` is versioned. You may add, merge, split, or remove categories, but every change goes through the change log (§6.3) with data-backed justification.
7. **Cache every LLM call** (keyed on model + prompt hash + params). Experiments must be re-runnable without re-spending API budget.
8. **Stop and report** (write to `reports/BLOCKERS.md`) instead of guessing when: the DB lacks a field this spec requires, the Voyager API behaves differently than expected, or label quality is too low to proceed.

---

## 1. Project Goal

**Build a classifier that predicts the outcome of an Indian land/property dispute (from the plaintiff/petitioner's perspective: win / lose), using a transparent pipeline that goes from evidence → canonical facts → fact patterns → rules/statutes and precedents → conclusion, and compare it at every step against an LLM baseline.**

### 1.1 The reasoning chain (target architecture)

```
AUTHORITIES (Constitution, Statutes, Regulations, Judicial Decisions)
      │ state / interpret
      ▼
RULES / TESTS ──────────── define criteria for ───────────┐
                                                           ▼
1 EVIDENCE → 2 CANONICAL FACTS → 3 FACT PATTERNS → 4 SUBTESTS/FACTORS ─satisfy/defeat→ 5 ELEMENTS
                                                                                          │ support/defeat
                                                                                          ▼
                                             8 CONCLUSION ←resolved by─ 7 ISSUES ←raise─ 6 CLAIMS / DEFENSES
```

### 1.2 Core problems

**A. Information grouping, labeling, extraction, patterning**
- Fact detection → canonicalization → grouping/labeling with properties (category / sub-category).
- Claim detection → clustering claims into **Claim Families**.
- Fact patterning *per claim family*, group by group, using frequent-pattern mining (FP-Growth).

**B. Key inferences**
1. Fact patterns (by claim family) → **Rules & Statutes**
2. Fact patterns (by claim family) → **Precedents / Judicial decisions**
3. Fact patterns (by claim family) + Rules & Statutes + Precedents, **oriented by party (Plaintiff vs Defendant)** → **Win / Lose**

> **Definition.** A *Claim Family* is a cluster of semantically equivalent claims across cases (e.g., "declaration of title", "title suit", "suit for declaration of ownership" → *Title/Ownership*). The ontology's 15 claim modules are the seed families; clustering may confirm, split, or merge them.

---

## 2. Research Framing: The Papers

The work is designed to yield at least three papers that share one dataset and codebase. Keep experiment outputs organized by paper (`experiments/paper{1,2,3,4}/`).

| # | Working title | Core question | Primary deliverables | Key metrics |
|---|---|---|---|---|
| **P1** | *Canonical Facts and Claim-Family Fact Patterns in Indian Property Litigation* | Can we extract, canonicalize and mine recurring fact patterns per claim family, reliably and at scale? | Canonical fact schema + label vocabulary; claim-family clustering; FP-Growth pattern library; annotated gold subset; (dataset release if licensing permits) | Fact extraction P/R/F1 vs gold; canonicalization agreement; cluster purity/NMI vs ontology families; pattern support/stability across splits |
| **P2** | *Grounding Fact Patterns in Law: Linking Patterns to Statutes and Precedents* | Given a case's fact patterns, can we predict the governing statutory provisions and retrieve relevant earlier precedents? | Pattern→statute association model; pattern-based precedent retrieval; citation graph | Statute prediction micro/macro-F1, P@k; precedent Recall@k, MRR, nDCG vs LLM and BM25/embedding baselines |
| **P3** | *Traceable Judgment Prediction: Patterns + Authorities vs. LLMs* | Does a structured, pattern-and-authority pipeline match or beat an LLM at predicting outcomes, while producing a verifiable trace? | Outcome classifier(s); trace generator; party-oriented features; error analysis by claim family | Macro-F1, accuracy, AUROC, ECE (calibration), trace faithfulness; per-family breakdown |
| P4 *(optional)* | *Executable Property Law: Lessons from the British Nationality Act, Revisited with LLMs* | Can the mined patterns + ontology elements be compiled into an executable rule base (logic program) with burdens, defaults and exceptions? | Rule compiler to Prolog/ASP (or Python rule engine); handling of negation-as-failure, burden shifts, defaults | Coverage, agreement with court outcomes, agreement with P3 classifier, rule count/complexity |

The P4 thread is directly inspired by Sergot et al., *"The British Nationality Act as a Logic Program"* (CACM 1986). Lessons to carry over are in §10.4.

---

## 3. LLM Access: ASU Voyager API

### 3.1 Discovery (do this before anything else)

```bash
# List variable NAMES only. Never print values.
grep -oE '^[A-Za-z_][A-Za-z0-9_]*' .env
```

Identify the variables for: API key, base URL/endpoint, model name(s), and any embedding model. Record the *names* (not values) in `reports/ENV_NOTES.md`. If the endpoint is OpenAI-compatible, use the `openai` SDK with `base_url`; otherwise use `httpx` against the documented endpoint. **Do not guess the API format** — run a one-line smoke test and inspect the response shape. If it fails, write to `reports/BLOCKERS.md`.

### 3.2 Client wrapper (`src/llm/client.py`)

One module; everything else calls it. It must provide:

- `complete(messages, model=None, temperature=0, max_tokens=..., json_schema=None) -> LLMResult`
- `embed(texts, model=None) -> np.ndarray` (if Voyager exposes embeddings; otherwise fall back to a local sentence-transformer and record which one was used)
- **Disk cache** (SQLite) keyed on `sha256(model, messages, params)`.
- **Retries** with exponential backoff; rate limiting; timeout.
- **Structured output**: request JSON, strip code fences, validate against Pydantic model, retry once on validation failure with the error message appended.
- **Logging**: every call logs model, token counts, latency, cache hit, and a prompt ID (prompts live in `prompts/` as versioned files, never inline strings).
- **Determinism**: temperature 0 by default; record seed if supported.

### 3.3 LLM baselines (used throughout)

| ID | Baseline | Input | Used in |
|---|---|---|---|
| `LLM-0` | Zero-shot | Masked case text (facts + pleadings only) | P1, P2, P3 |
| `LLM-FS` | Few-shot (k retrieved training examples, earlier-dated only) | Same | P2, P3 |
| `LLM-CoT` | Zero-shot with element-by-element reasoning prompt (ontology elements listed) | Same | P3 |
| `LLM-CF` | Given our canonical facts instead of raw text | Canonical facts JSON | P3 (isolates value of our extraction) |
| `LLM-CF+A` | Given canonical facts + our predicted statutes + retrieved precedents | Structured bundle | P3 (isolates value of grounding) |

The LLM is also a *component* (extraction, canonical labeling). That is fine — the baseline comparison is about the **decision**, not about banning the LLM from the pipeline. Always state clearly which role the LLM plays in each number reported.

---

## 4. Stage 0 — Database Discovery & Adapter

The database is already downloaded and you know about it.

## 5. Stage 1 — Labels, Splits, and Leakage Control

### 5.1 Outcome label

Normalize `outcome_raw` into a **party-oriented** label from the perspective of the party who initiated the proceeding (plaintiff/petitioner/appellant):

| Label | Meaning |
|---|---|
| `WIN` | Allowed / decreed / granted in substance |
| `LOSE` | Dismissed / rejected |
| `PARTIAL` | Partly allowed |
| `REMAND` | Set aside and remanded |
| `OTHER` | Withdrawn, settled, disposed without merits, infructuous |

Primary task: binary `WIN` vs `LOSE` (drop others). Secondary task: 3-class with `PARTIAL`. Report class balance. Use rules first (regex over the order section), then LLM for residuals, then **human-check a sample of 200**; report agreement. Appeals: be explicit whether "appellant wins" means the original plaintiff won or lost, and store both `initiator_outcome` and `original_plaintiff_outcome`.

### 5.2 Leakage masking (mandatory)

Build `src/data/mask.py` producing `masked_text` that contains **only**: facts narrative, pleadings/prayers, and each party's arguments as summarized by the court. It must **exclude**: the analysis/discussion section, the operative order, and any sentence with outcome cues (`allowed`, `dismissed`, `we hold`, `decree`, `set aside`, `quashed`, `in the result`, costs orders, etc.).

- If section boundaries are not in the DB, segment with rules + an LLM section classifier; validate on 100 hand-checked judgments.
- **Leakage probe (required):** train a simple TF-IDF + logistic regression on `masked_text` → outcome. Also ask `LLM-0` to "identify the outcome from the text" with no reasoning. If either is suspiciously high relative to the other experiments, inspect and tighten masking. Report the probe numbers in every paper.
- **Authorities caveat:** statutes/cases cited *in the analysis section* are chosen by the court knowing the outcome. For P2/P3 prediction inputs, only use authorities cited in pleadings/arguments, or *predicted* authorities. Court-cited authorities are **targets** (P2) or **oracle upper bounds** (P3, clearly labeled), never ordinary inputs.

### 5.3 Splits

- **Primary: temporal split** (e.g., train ≤ year T1, dev T1–T2, test > T2). Choose T1, T2 so test ≈ 15–20%.
- **Secondary: court-held-out** split (train on some High Courts, test on another) for generalization.
- **Precedent retrieval is time-respecting:** for a case decided on date d, only cases decided before d may be retrieved or used as features.
- Same appeal chain must not straddle train/test.

**Acceptance:** `data/splits/*.json` frozen with a hash; label report; masking validation report; leakage probe numbers.

---

## 6. Stage 2 — Evidence → Canonical Facts (P1)

### 6.1 Fact extraction

For each case, from `masked_text`, extract atomic fact propositions with an LLM using ontology-guided prompts (`prompts/extract_facts.v*.md`). Chunk long texts; keep paragraph IDs.

```python
class Fact(BaseModel):
    fact_id: str
    case_id: str
    text: str                     # atomic proposition, as extracted
    source_span: Span             # section, paragraph_id, char offsets
    asserted_by: Literal["plaintiff","defendant","court_narrative","admitted","unknown"]
    disputed_status: Literal["admitted","contested","unclear","inferred"]
    event_date: str | None        # exact / approximate / range
    entities: list[EntityRef]     # parties, property, authority, instrument
    evidence_refs: list[str]      # exhibits/documents mentioned as proof
```

`asserted_by` is critical: it is what later lets us orient patterns by **Plaintiff vs Defendant**.

### 6.2 Canonicalization (category / sub-category / properties)

Map each free-text fact to a canonical label from a controlled vocabulary derived from the ontology:

```python
class CanonicalFact(BaseModel):
    fact_id: str
    category: str        # e.g. "Instrument", "Possession", "Notice", "StateAction", "Payment", "Succession"
    subcategory: str     # e.g. "SaleDeed.Registered", "Possession.Continuous", "Notice.NotServed"
    properties: dict     # e.g. {"polarity": "neg", "actor": "respondent", "duration_years": 14}
    label: str           # canonical atom used for mining, e.g. "possession.continuous.gt12y"
    confidence: float
```

Procedure (iterative, **must** converge to a frozen vocabulary before mining):
1. **Seed** vocabulary from ontology sections 1, 3, 6 (claim Facts lists), 8 (evidence types).
2. **Label** a stratified sample (~2k facts) with the LLM, allowing a `NEW:<proposal>` escape.
3. **Cluster** the `NEW` proposals (embeddings + HDBSCAN), have the LLM name clusters, human-review, and merge into vocabulary.
4. Repeat until `NEW` rate < 5% on a fresh sample. Freeze as `ontology/vocab_v{N}.yaml`.
5. Label the full corpus. Discretize numeric properties (durations, amounts, delays) into bins so labels are itemizable.

Include **negated** and **defendant-asserted** facts as distinct labels (e.g., `notice.served` vs `notice.not_served`; prefix or property for asserting party). Polarity matters (see §10.4 on negation).

### 6.3 Ontology change log

Every vocabulary or ontology change is appended to `ontology/CHANGELOG.md`: what changed, why, how many facts affected, before/after examples. The provided `Land_Property_Dispute_India_Ontology_Refined.docx` is **v1**; convert it to `ontology/ontology_v1.yaml` first (the earlier unrefined doc is v0, keep for reference only).

### 6.4 Gold set

Create `data/gold/` with **≥ 150 cases** annotated by a legal reviewer (facts, canonical labels, claims, claim family, statutes). Provide a lightweight annotation tool (Streamlit or Label Studio config) pre-filled with LLM output for correction. Report inter-annotator agreement on a 30-case overlap if two annotators are available.

**Acceptance (P1):** extraction F1 and canonical-label accuracy on gold reported for our pipeline vs `LLM-0` extraction without vocabulary; frozen vocabulary; `NEW` rate < 5%.

---

## 7. Stage 3 — Claims & Claim Families (P1)

1. **Claim extraction:** from pleadings/prayers, extract each claim and each defense with: text, raising party, relief sought, statutory hook (if pleaded), source span.
2. **Claim families:** embed claim texts; cluster (HDBSCAN; compare with agglomerative + k chosen by silhouette). Initialize/align with the ontology's 15 claim modules (Title, Possession/Injunction, Lease/Tenancy, Redevelopment, Constitutional Deprivation, Ultra Vires, Co-operative Society, Adverse Possession/Limitation, Partition, Specific Performance, Cancellation, Mortgage/Redemption, Succession, Benami, Land Acquisition).
3. **Report** cluster-to-module alignment (NMI, purity), clusters with no module (candidate new families → change log), modules with no cluster.
4. A case may belong to **multiple families**. Keep case↔family as many-to-many, and produce a "primary family" per case (the claim the relief chiefly turns on).
5. **Defenses** are clustered the same way, seeded by ontology §7 (Title/Tenure, Public Interest, Statutory Subservience, Consent/Compliance, Procedural/Forum, Equitable).

**Acceptance:** every case has ≥ 1 family; family sizes reported; low-support families (< 30 cases) flagged and either merged or excluded from P3 per-family metrics.

---

## 8. Stage 4 — Fact Patterning by Claim Family (P1)

### 8.1 Transactions

For each claim family F, each case c in F becomes a **transaction**: the set of canonical labels of facts relevant to that family's claims (use fact↔claim linkage; if unavailable, use all case facts and report both variants).

Build **two party-oriented views** of every transaction:
- `T_plaintiff(c)`: facts asserted by plaintiff or admitted
- `T_defendant(c)`: facts asserted by defendant or admitted

### 8.2 Mining

- `mlxtend.frequent_patterns.fpgrowth` per family, min_support tuned per family (start 0.05; ensure ≥ 50 and ≤ 5,000 itemsets), `max_len` 4–5.
- Derive **closed** or **maximal** itemsets to reduce redundancy.
- **Association rules** of the form `pattern → X` where X ∈ {statute label, outcome label, element label}. Report support, confidence, lift, and conviction. Rules to outcome are *computed only on the training split*.
- **Discriminative patterns:** patterns whose outcome distribution differs significantly from the family base rate (chi-square with Benjamini–Hochberg correction, or growth-rate / emerging-pattern mining).
- **Stability:** bootstrap the training set (e.g., 20 resamples); keep patterns appearing in ≥ 80% of resamples.

### 8.3 Pattern → Subtest/Factor → Element mapping

Each ontology claim has elements (e.g., Adverse Possession: possession, hostility, continuity/limitation period, knowledge/ouster). Map patterns to elements they **satisfy** or **defeat**:
- Seed by LLM proposal on pattern + element definitions, then human review of the top patterns per family.
- Store as `pattern_element_edges(pattern_id, element_id, relation ∈ {satisfies, defeats}, confidence, reviewer)`.
- Element carries burden metadata from the ontology (`burden_on`, `burden_standard`, `burden_shifts_when`).

**Acceptance:** per-family pattern library exported (`data/patterns/{family}.parquet`), stability report, top-20 discriminative patterns per family with human-readable descriptions, element mapping coverage.

---

## 9. Stage 5 — Grounding in Authorities (P2)

### 9.1 Authority records

Normalize statutes and cases into `AuthorityRecord` (from ontology §2): `authority_id`, `authority_type`, `citation_or_title`, `court_or_forum`, `holding_or_rule`, `cited_proposition`, `temporal_validity`, `jurisdiction_scope`. Normalize statute citations to `Act:Section(sub)` (e.g., `TPA:53A`, `SRA:16(c)`, `LimitationAct:Art65`, `Const:Art300A`). Resolve case citation variants to a canonical case ID; link to DB cases where possible.

### 9.2 Inference 1 — Fact patterns → Rules & Statutes

Task: multi-label prediction of the statutes the court relies on, given the case's patterns (targets = court-cited statutes; inputs = masked-text-derived patterns).

Models to compare:
- Association-rule classifier (patterns → statute rules from §8.2).
- Multi-label logistic regression / LightGBM on pattern + canonical-label features.
- `LLM-0` and `LLM-FS` predicting statutes from masked text.
- Hybrid: LLM re-ranks candidates from the pattern model.

Metrics: micro/macro-F1, P@k, R@k, per-family breakdown.

### 9.3 Inference 2 — Fact patterns → Precedents

Task: given case c, retrieve earlier cases (decided before c) that the court cited or that share the legal question.
- Targets: court-cited precedents present in the DB (plus, on a subsample, human-judged relevance).
- Retrievers: BM25 on masked text; dense embeddings; **pattern-overlap similarity** (weighted Jaccard over family-specific patterns, IDF-weighted); hybrid fusion (RRF); LLM re-ranker.
- Metrics: Recall@k (10, 50), MRR, nDCG@10.
- Also build the **citation graph** (case → cited case, case → statute) for P2 analysis and P3 features.

**Acceptance (P2):** all retrievers/predictors evaluated on the same test split, with LLM baselines; time-respecting retrieval verified by an assertion in code.

---

## 10. Stage 6 — Judgment Prediction (P3)

### 10.1 Inference 3 — Patterns + Statutes + Precedents, by party → Win/Lose

Feature groups (ablate each):

| Group | Features |
|---|---|
| **F** Facts | Canonical label indicators (binary / TF-IDF), counts per category |
| **P** Patterns | Indicators for stable discriminative patterns per family; separate for `T_plaintiff` and `T_defendant` |
| **E** Elements | Per element: #satisfying patterns, #defeating patterns, burden_on, net score; "all elements satisfied" flag |
| **S** Statutes | *Predicted* statutes (from §9.2) — oracle court-cited statutes only as a labeled upper bound |
| **R** Precedents | Outcome distribution of top-k retrieved earlier precedents, **oriented to the current plaintiff's position** (i.e., did the party in the analogous role win?), similarity-weighted |
| **C** Context | Court level, forum, claim family, defense families raised, party types (State vs private, society, developer), appeal stage |

Models:
- Logistic regression (interpretable reference)
- LightGBM / XGBoost
- Rule-list / scorecard learner (e.g., CORELS-style or `imodels`) for interpretability
- Executable rule base (P4) if built
- Hybrid: structured model + `LLM-CF+A` stacking

Baselines: `LLM-0`, `LLM-FS`, `LLM-CoT`, `LLM-CF`, `LLM-CF+A`, TF-IDF+LR on masked text, majority class, per-family majority.

Metrics: macro-F1 (primary), accuracy, AUROC, ECE + reliability diagrams, per-family and per-court breakdown, bootstrap 95% CIs, paired significance test vs `LLM-0` (McNemar or paired bootstrap).

### 10.2 Ablations (required for P3)

1. F only → F+P → F+P+E → +S → +R → +C
2. Oracle vs predicted statutes (quantifies the upper bound of better grounding)
3. With vs without party orientation (`T_plaintiff`/`T_defendant` split)
4. LLM-extracted facts vs gold facts on the gold subset (error propagation)
5. Pattern stability threshold sensitivity
6. Temporal vs court-held-out split

### 10.3 Error analysis

For 50 test errors per best model and per LLM baseline: categorize (extraction miss, wrong canonical label, family misassignment, missing pattern, statute mismatch, genuinely discretionary outcome, label noise). Report overlap between the model's and LLM's errors.

### 10.4 Lessons from the British Nationality Act as a Logic Program (apply them)

Sergot et al. (1986) formalized statute as Horn-clause rules executed by Prolog with an interactive "why/how" explanation. Their lessons map directly onto this project:

- **Top-down, goal-directed development.** Define high-level concepts (claim → elements) first and let lower-level facts be filled in as needed; do not try to fix all primitive concepts up front. Our ontology → element → pattern → fact order follows this.
- **Vague concepts** ("good character", "reasonable excuse") → in our domain: "readiness and willingness", "hostile possession", "public purpose". Handle them as *qualified conclusions* ("plaintiff wins **if** readiness and willingness is accepted") or learn rules of thumb from precedents — exactly the pattern → element mapping.
- **Negation as failure vs. classical negation.** Deeming provisions and presumptions ("unless the contrary is shown") are defaults. Distinguish *"not proved"* from *"proved false"* in canonical facts (`notice.not_shown` ≠ `notice.not_served`). Keep `disputed_status` and `asserted_by` so defaults and burden shifts can be modeled.
- **Non-monotonicity.** Conclusions drawn by default may be withdrawn when new facts arrive; the trace must show which default assumptions a conclusion depends on.
- **Counterfactual and discretionary provisions** ("would have… but for", "court may in its discretion" — e.g., specific performance is discretionary) → represent as alternative rules with an explicit discretion condition; flag discretionary families in P3 analysis.
- **Parameters grow over time.** Their "x is a parent of y" had to become "x is a parent of y on date z". Design fact labels with dates/periods from the start (limitation, adverse possession, tenure terms all depend on time).
- **Explanations are first-class.** Their APES shell answered *why* a question was asked and *how* a conclusion was reached. Our trace (§11) is the equivalent and is a core evaluation target, not decoration.

---

## 11. The Trace (Output Contract)

Every prediction from our method emits:

```json
{
  "case_id": "...",
  "prediction": {"label": "WIN", "probability": 0.78},
  "claim_families": ["AdversePossession"],
  "issues": [
    {
      "issue": "Has possession matured into title by adverse possession?",
      "claim": "claim_03",
      "elements": [
        {
          "element": "Hostility",
          "status": "satisfied",
          "burden_on": "plaintiff",
          "via_patterns": [
            {"pattern_id": "AP_p17", "labels": ["possession.open","possession.exclusive","title.owner_denied"],
             "relation": "satisfies", "facts": ["f_12","f_19"],
             "source_spans": ["facts:p4","facts:p7"]}
          ],
          "authorities": ["LimitationAct:Art65"]
        }
      ],
      "precedents": [{"case_id": "...", "similarity": 0.71, "analogous_party_outcome": "WIN"}]
    }
  ],
  "defaults_assumed": ["notice.not_shown treated as not served (burden on defendant)"],
  "top_feature_contributions": [{"feature": "AP_p17@plaintiff", "shap": 0.21}]
}
```

**Trace faithfulness evaluation:** (a) deletion test — remove the facts cited in the trace and confirm prediction confidence drops more than removing random facts; (b) human rating on 50 traces by a legal reviewer (correct / partially correct / wrong) on element status and authority relevance; (c) compare against LLM-CoT rationales on the same cases.

---

## 12. Repository Layout

```
.
├── PROJECT.md                  # this file
├── .env                        # existing; never commit, never print values
├── configs/                    # YAML configs per stage/experiment
├── ontology/
│   ├── ontology_v1.yaml        # converted from Refined docx
│   ├── vocab_vN.yaml           # frozen canonical labels
│   └── CHANGELOG.md
├── prompts/                    # versioned prompt files (*.v1.md, *.v2.md)
├── src/
│   ├── llm/client.py           # Voyager wrapper + cache
│   ├── data/{adapter,mask,labels,splits}.py
│   ├── extract/{facts,canonicalize,claims}.py
│   ├── cluster/claim_families.py
│   ├── patterns/{transactions,mine,elements}.py
│   ├── authorities/{normalize,statute_predict,precedent_retrieve,graph}.py
│   ├── predict/{features,models,llm_baselines,stack}.py
│   ├── trace/{build,evaluate}.py
│   ├── logic/                  # P4: rule compiler / engine (optional)
│   └── eval/{metrics,significance,reports}.py
├── data/{interim,processed,gold,patterns,splits,cache}/
├── experiments/paper{1,2,3,4}/ # configs, results JSON, figures
├── reports/                    # DB_SCHEMA, ENV_NOTES, BLOCKERS, stage reports
├── tests/                      # unit tests incl. leakage + time-respect asserts
└── Makefile                    # make stage0 ... make stage6, make paper1 ...
```

Stack: Python 3.11, `pydantic`, `pandas`/`polars`, `mlxtend` (FP-Growth), `scikit-learn`, `lightgbm`, `hdbscan`/`umap-learn`, `rank_bm25`, `shap`, `imodels`, `sqlalchemy` (DB), `httpx`/`openai` (Voyager), `diskcache` or SQLite for caching. Optional P4: `pyswip` (SWI-Prolog) or `clingo` (ASP).

---

## 13. Milestones

| Milestone | Contents | Exit criterion |
|---|---|---|
| **M0** | Voyager smoke test, DB schema report, adapter | §3.1 and §4 acceptance |
| **M1** | Labels, masking, splits, leakage probe | §5 acceptance |
| **M2** | Fact extraction + canonical vocabulary frozen; gold set started | §6 acceptance |
| **M3** | Claim families + fact patterns + element mapping | §7, §8 acceptance → **P1 draft** |
| **M4** | Statute prediction + precedent retrieval | §9 acceptance → **P2 draft** |
| **M5** | Outcome classifiers, ablations, traces, error analysis | §10, §11 → **P3 draft** |
| **M6** *(opt.)* | Rule compiler / logic program | → **P4 draft** |

At each milestone, write `reports/M{n}_report.md`: what was built, numbers (ours vs LLM), data statistics, open issues, ontology changes.

---

## 14. Reporting Standards

- Every results table includes: our method(s), all applicable LLM baselines, simple non-LLM baselines, n, split name, 95% CI.
- Every LLM number states: model name, prompt ID/version, temperature, date run, cache hit rate.
- Report API cost and token usage per experiment.
- Report data statistics per claim family: #cases, class balance, #facts, #patterns.
- Keep a `experiments/*/results.jsonl` append-only log; figures are regenerated from it.

---

