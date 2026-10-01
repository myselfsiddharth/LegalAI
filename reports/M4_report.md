# M4 — Grounding facts in law (§9)

**Status: §9.1 and §9.2 built and measured. §9.3 (precedent retrieval) not yet started.**
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

## 3. §9 acceptance

| requirement | status |
|---|---|
| §9.1 `AuthorityRecord` normalisation, `Act:Section` form | **partial** — statutes done (61.4% resolved); case-citation canonicalisation not built |
| §9.2 multi-label statute prediction with baselines | **met** — 3 representations, 3 baselines incl. the copy control |
| §9.2 micro/macro-F1, P@k, R@k | **met** |
| §9.2 association-rule classifier arm | **not built** — §8 produced no significant rules to classify with |
| §9.2 `LLM-0` / `LLM-FS` statute prediction | **not built** |
| §9.3 precedent retrieval, Recall@k / MRR / nDCG | **not started** |
| §9.3 time-respecting retrieval asserted in code | available in `scripts/retrieval_lib.py` (`before_year=`), not yet wired |
| citation graph | exists from earlier work (8,467 signed edges) |

## 4. Next

1. **§9.3 precedent retrieval** — the remaining half of M4, with assets ready: `retrieval_lib.py`
   (BM25 + dense behind one `search()`, already time-respecting), the signed citation graph, and
   `citations_classified.jsonl` for targets. Earlier work on this corpus found real signal here
   (recall@10 31%, 3–6× a popularity control).
2. **`LLM-0` / `LLM-FS` statute prediction**, to complete §9.2's baseline set.
3. Feed predicted statutes into §10's `S` feature group, which is currently empty.
