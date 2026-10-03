# Is this novel? An honest positioning against the literature

Written 2026-10-02, after the leakage result landed and before anyone writes a paper around it.
**Short answer: both headline findings independently rediscover active, recently-published research
directions. What is ours is narrower than it felt, and real.**

Three papers were located and **verified by fetching the arXiv abstract** (not taken from a search
summary). Leads that were *not* verified are marked as such and must be checked before citing.

---

## 1. The leakage finding

### What we have
Deleting the operative order leaves ~128 words of outcome-telegraphing text behind. A 512-word window
set back past that residue scores no better than facts-only on either split (+0.013, +0.012, both CIs
spanning zero). Essentially all of the +0.181 apparent advantage of ILDC's chosen input is that
residue. Replicated on `forum_heldout` and `temporal_2004_2013`.

### Prior art — verified

**Nigam & Deroy, "Fact-based Court Judgment Prediction", arXiv:2311.13350 (Nov 2023).**
Extends ILDC for CJPE with a facts-only variation and a facts+lower-court-ruling variation.
**Reports that facts-only is worse than the original full-document setup** across transformer models,
even with weighting schemes in DELSumm.
→ **Our Headline 1's direction is published, on ILDC itself, three years ago.** We did not know this.

**Watson, Ribeiro de Faria, Tomalin, Magnusson, Xie, Yeung, Carter, Rutherford & Steffek,
"Shortcut Learning in Legal Judgment Prediction: Empirical Evidence from the UK Employment Tribunal",
arXiv:2607.04261 (5 July 2026, revised 10 July).**
33,158 UK Employment Tribunal claims. Finds "performance increases where outcome-revealing cues are
embedded in the narrative". Isolates the 4% of features flagged as leakage and retrains on those
alone; measures macro-F1 degradation when leakage features are masked.
→ **The leakage-gap framing is published, at 4x our scale, three months ago, by nine authors.**

*Unverified leads, to check before citing:* work on ECtHR reporting that masking verdict words was
"not completely successful in filtering out outcome information"; `Multi-Legal-Bench` on label
leakage in benchmarks built from court registries (arXiv:2605.29738); Medvedeva et al., "Rethinking
the field of automatic prediction of court decisions" (Springer AI & Law, 2021);
"A Challenging Benchmark for Legal Judgment Prediction" (NLLP 2022).

### What is actually ours

The prior work identifies leakage **lexically or by human judgment**: Watson et al. stratify by human
judgments of leakage and flag features; the ECtHR line masks verdict words. Nobody located uses a
**positional** method.

1. **The volume-controlled setback curve.** Four windows of *exactly* 512 words stepping back from
   the deletion boundary, so volume cannot explain any difference between them. It requires no cue
   detector, no human annotation and no feature flagging.
2. **Evidence that this matters, not just that it differs.** Our own cue-based audit **did not
   replicate**: `w0`'s cue enrichment was +0.150 on one split and −0.058 on the other, while the
   positional curve held on both. A lexical method would have given us a result that evaporated on
   the second split. This is a concrete argument for the positional design, and we have the failed
   lexical arm to show for it.
3. **A quantified margin: ~128 words**, found independently on two splits. No prior work located
   states a distance. This is the actionable output — a benchmark built this way needs that margin
   past the detected boundary, not zero.
4. **A cleared baseline.** `masked_integrity.py` shows the facts-only arm is sound (−0.0001 from a
   full disposition scrub; volume-matched setback +0.027, CI spanning zero). Prior work does not
   verify its own clean arm with the same instrument.

**Verdict: a methodological contribution inside an established critique.** Not a new critique.

---

## 2. The discretisation finding

### What we have
The representation ladder: masked text 0.657 → extracted fact text 0.614 → canonical atoms 0.517.
Extraction costs 0.043; **discretising into a label vocabulary costs a further 0.097**. An induced
vocabulary of equal granularity is statistically indistinguishable from the authored ontology
(−0.025, P=0.846), more labels do not help, and the cost is concentrated *where the vocabulary
succeeds in naming a fact*. Downstream: 0 of 498 fact patterns survive Benjamini–Hochberg.

### Prior art — verified

**Sadowski & Chudziak, "On Measuring Semantic Preservation in Legal Ontology Learning",
arXiv:2608.12326 (30 May 2026).**
Measures information loss by comparing LLM task performance on source documents against
ontology-transformed representations. Six LLMs, three ontology-learning methods, legal merger
agreements. Reports substantial semantic loss.
→ **"Ontologising legal text destroys downstream signal" is published, four months ago.**

### What is actually ours — and here the gap is in our favour

Sadowski & Chudziak explicitly conclude that the loss "stems from complex interactions between the
transformation process and reasoning capabilities, **not simply from vocabulary discretization
alone**." They do **not** compare an induced vocabulary against a hand-authored one, and they do not
isolate discretisation.

That isolation is exactly what our vocabulary A/B does:

| comparison | ΔAUROC | 95% CI | P |
|---|---|---|---|
| text − induced | +0.1138 | [+0.0732, +0.1545] | 0.000 |
| text − ontology | +0.0887 | [+0.0430, +0.1352] | 0.000 |
| **induced − ontology** | **−0.0251** | **[−0.0711, +0.0200]** | **0.846** |

Two vocabularies of equal granularity, one authored by a domain expert and one induced from the
corpus, are indistinguishable — and both lose heavily to raw text. **The loss is discretisation
itself, not the quality of anyone's ontology.** Plus the within-group diagnosis: the cost is 0.087
where the vocabulary can name the fact and 0.002 where it cannot, which is the opposite of the
intuitive "coverage is incomplete, fix the vocabulary" story.

**Verdict: the same direction as recent work, with the one control that work says it lacks.**
This is the stronger of our two findings for publication, and it is the less crowded area.

---

## 3. The constructive result, and the most defensible framing

Nothing located makes this specific triangulation on one corpus with one apparatus:

- the facts **do not** predict outcomes (every structured model at or below a per-decade majority
  baseline, every CI spanning ~0.11);
- the facts **do** predict the applicable law — statutes at micro-F1 0.173 against 0.093 for the
  strongest baseline (**1.9×**, with a copy control driven to zero), precedents at R@10 0.238 against
  a popularity control of 0.032 (**7.4×**);
- and the apparent outcome signal in the published setup is ~85% an artifact of order-removal
  residue.

**"Facts are informative about law but not about outcomes"** is a coherent, defensible thesis that
the three results support jointly, and it points somewhere constructive rather than only debunking.

---

## 4. Are we "breaking the industry"? No.

Stated plainly, because the temptation runs the other way:

- **The research subfield already knows.** Three verified papers in 2023–2026 cover both of our
  headline directions. Watson et al. is three months old. These are active conversations, not blind
  spots.
- **Commercial legal AI does not rest on this.** Document review, drafting, and research are what
  those products sell. Outcome prediction is not a headline feature of any major one — in part
  *because* it does not work well. A finding about CJPE benchmark construction does not touch them.
- **Our scale limits the claims.** 6,954 land/property cases against ILDC's 34,816 and the UK
  tribunal's 33,158. Every outcome-model confidence interval spans about 0.11.
- **The ILDC replication is not done.** Everything is measured on our corpus as a *comparable
  construction*. Until `src/eval/ildc_ablation.py` runs on the real release, the strongest claim
  available is internal.

### What this is instead
A well-controlled, honestly-reported workshop paper (NLLP at EMNLP, or JURIX) with two real
methodological contributions — the positional setback curve with its quantified margin, and the
induced-vs-authored control isolating discretisation — plus a constructive positive result. That is
a good outcome for a class project. It is not a disruption.

### What would raise the ceiling, in order
1. **Run the ILDC replication** *with a setback arm added*, or it inherits the contamination it is
   meant to measure. This is the one step that converts an internal result into a claim about the
   benchmark the field actually cites.
2. **Lead with the discretisation finding**, not the leakage finding. Less crowded, and we hold the
   control the closest paper states it does not have.
3. **Cite all three papers above and position against them explicitly.** A reviewer who knows
   Watson et al. and sees it uncited will reject on novelty alone; one who sees it cited and
   differentiated on method reads the paper as careful.
