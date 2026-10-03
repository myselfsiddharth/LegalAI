# BLOCKERS

## RESOLVED

### ~~B1 — Ontology v1 source document missing~~ (closed 2026-09-30)

`Land_Property_Dispute_India_Ontology_Refined.docx` was not in `Docs/`. Located in the user's
Downloads and copied to `Docs/ontology_source/` (that one subdirectory is un-ignored so the YAML's
provenance is versioned, while the confidential Templeton grant document stays ignored).
`src/data/ontology_convert.py` parses it into `ontology/ontology_v1.yaml`: 15 claims, 66 elements,
6 defence families, 4 remedy tiers.

---

## OPEN

### B5 — ILDC replication blocked on licence acceptance (the highest-value open item)

`src/eval/ildc_ablation.py` is **written, setback-armed and verified end to end** via
`--smoke-test`, which substitutes this project's own corpus for the gated release so every code
path executes. It runs the moment `Data/ildc/` is populated.

**Why it matters.** Everything in the leakage result is measured on our 6,954 land/property cases
as a *comparable construction* of ILDC's task. ILDC itself is 34,816 cases and is what the field
cites. Until this runs, the strongest available claim is internal.

**What is needed, and it is not compute.** Accept the terms at
<https://huggingface.co/datasets/Exploration-Lab/IL-TUR>, then either:
- `export HF_TOKEN=<read token>` and run `python -m src.eval.ildc_ablation --download`, or
- drop `ILDC_multi.csv` (columns: text, label, split, name) into `Data/ildc/`.

Either layout is detected automatically. The module **does not route around the gate**: the authors
moved deliberately from an open 2021 link to a gated release, and going around that would ignore a
choice they made on purpose.

**Expected runtime** once the data lands: the masking step dominates. `--max-train` caps training
rows (default 8,000) because a TF-IDF model does not need all 32,305 and masking is the slow part.



### B2 — Element burden metadata cannot be produced by an LLM (blocks §8.3, §10.1 group `E`)

The source ontology requires every element to carry `burden_on`, `burden_standard` and
`burden_shifts_when`, and supplies them for **0 of 66** elements. 6 of 15 claims carry a
claim-level default in a notes line; the rest have nothing.

**Two models have now failed the degeneracy gate** (`src/data/ontology_burden.py`):

| model | result | why refused |
|---|---|---|
| `llama4-scout-17b` | refused | 92% `claimant`; `clear_proof_required` used **0 times**, including for adverse possession; only 2 distinct confidence values; `burden_shifts_when` null for all 4 adverse-possession elements |
| `glm-5-3` | refused | 87% `claimant` (limit 85%) |

Both annotations *looked* complete. That is the hazard: §8.3's element scores and §10.1's `E`
features read these values directly, and a constant masquerading as legal metadata would not be
questioned by anything downstream. The gate refuses rather than saving with a caveat.

**Impact.** Group `E` is empty. The §10 ablation table reports it as empty rather than printing
"F+P+E" as a distinct row — without that, `F`, `F+P` and `F+P+E` showed byte-identical numbers and
would have read as "patterns and elements add nothing", which is right by accident and wrong in
reasoning.

**What would actually fix it**, in order of preference:
1. **Hand annotation of 66 elements** by someone who can cite the doctrine. This is a few hours of
   expert time and the allocations are largely settled in Indian law.
2. **Decompose the task** — one call currently settles who bears the burden, to what standard, and
   what shifts it. Asking separately, with the claim's statutory framework supplied, may
   discriminate where a single call does not.
3. Neither is "try a bigger model". `glm-5-3` is larger and more instruction-disciplined than
   scout and did marginally better on one axis while still failing.

### B3 — §6.2's vocabulary cannot name ~23% of extracted facts (blocks freezing the vocabulary)

Measured over 13,919 canonical facts: **22.7% out of vocabulary**, against §6.2's 5% freeze
threshold. The seed vocabulary (345 subcategories, all generated from named ontology fields) covers
about three quarters of what real judgments state.

This is now known to be the pipeline's **binding quality constraint**, not merely untidy — see
`reports/M5_report.md`: the representation ladder shows canonicalisation is where predictive signal
is destroyed.

`src/extract/vocab_refine.py` implements one turn of §6.2's loop and runs in the downstream driver
as a **proposal only**; §6.2 step 3 requires human review before a vocabulary is frozen.

### B4 — No gold annotation set (§6.4 acceptance cannot be evaluated)

§6.4 asks for ≥150 cases annotated by a legal reviewer, with inter-annotator agreement on a 30-case
overlap. None exists, so extraction F1 against gold — §6.1's stated acceptance criterion — cannot
be computed.

**Substitutes in use, and what each does and does not establish:**
- quote grounding rate (0.92): bounds *fabrication*, says nothing about recall
- deterministic legal-statement filter: bounds one *precision* failure mode
- rules-vs-LLM label agreement (87.0%): estimates *outcome-label* noise, not fact quality
- the representation ladder: measures *downstream utility*, which is what we actually care about,
  but cannot attribute a loss to a specific extraction error

§5.1's human check of 200 outcome labels is also outstanding; the 562 UNRESOLVED label conflicts in
`Data/interim/label_conflicts.jsonl` are the highest-information place to start.
