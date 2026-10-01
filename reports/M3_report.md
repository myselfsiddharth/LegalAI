# M3 — Claim families and fact patterns (§7, §8)

**Status: full scale** (7,441 claims over 4,235 cases; §8 over 4,581 canonicalised cases). Reproduce: `make stage3 stage4`.

**Headline: §7's 15-family taxonomy is not supported by the data, and §8 finds no fact pattern
associated with outcome.** Both are results, not failures to reach a result, and §8's is airtight.

## 1. Claims and defences (§7.1)

Prompt `extract_claims.v1`, `llama4-scout-17b`, 862 cases:

| | proposed | kept | discarded, quote not locatable | family outside the list |
|---|---|---|---|---|
| claims | 1,154 | **943 (81.7%)** | 211 | 208 (18%) |
| defences | 1,283 | **1,041 (81.1%)** | 242 | 431 (34%) |

Median 2 claims per case; 63 of 600 cases yielded none. **21 dropped calls** (3.5%) — recoverable
with `--resume`, and reported rather than folded into "no claims found".

## 2. Claim families (§7.2–7.3): the taxonomy does not hold

Claim texts embedded and clustered, then compared with the assigned families. At 1,408 claims over
805 cases:

| clusterer | k | NMI | ARI | purity | note |
|---|---|---|---|---|---|
| HDBSCAN | 6 | 0.070 | **0.003** | 0.395 | 71.5% outliers |
| agglomerative (k by silhouette) | 6 | 0.237 | 0.083 | 0.314 | **silhouette 0.025** |

(At full scale, 7,441 claims / 4,235 cases. The earlier 1,408-claim run gave ARI −0.006 and
silhouette 0.022 — the same answer with looser precision.)

**ARI ≈ 0 means the unsupervised partition is uncorrelated with the assigned 15-way families**, and
silhouette 0.022 means there is almost no cluster structure at that granularity.

Supporting evidence:
- Only **2 of 15** ontology families cleared §7's 30-case floor (Title 235 cases, Possession 219).
- `Redevelopment` was never assigned to any claim.
- The extractor assigned **42** distinct families at 268 cases, **84** at 805, and **281** at 4,235.
  Its open-ended `NEW:` escape invents roughly in proportion to corpus size and **does not
  converge** — now measurable across three scales.
- 32% of cases fall in more than one family, consistent with §7.4's many-to-many requirement.

### 2.1 The clusters are coherent one level up

```
Title 49, Possession 41, Partition 13, Trust 5                    -> private title & possession
SpecificPerformance 25, Title 15, Cancellation 10                 -> contract & instrument
Possession 28, Title 9, Tenure 7, LeaseTenancy 4                  -> tenure & occupancy
Title 35, UltraVires 20, ConstitutionalDeprivation 19,
  LandAcquisition 13                                              -> state action against property
```

So the taxonomy is too **fine** for this corpus rather than wrong.

### 2.2 Merge, derived from the clusters (§7.3 → §6.3)

`src/cluster/family_merge.py` assigns each family to the cluster its claims most often land in; only
the merged names are authored. Logged to `ontology/CHANGELOG.md`.

| merged family | cases | clears 30-case floor |
|---|---|---|
| StateAction | 424 | yes |
| PrivateTitlePossession | 256 | yes |
| ContractInstrument | 170 | yes |
| Mixed_Limitation_Procedure | 47 | yes |
| TenureOccupancy | 23 | no |
| Mixed_Compensation | 15 | no |

Families over the floor: **2 → 4**.

### 2.3 Two problems the merge exposed, recorded not hidden

**Defence families were routed to claims** by the extractor: `Procedural/Forum` (43 cases),
`Statutory/Regulatory Subservience` (31), `Public Interest/Planning Policy` (6), `Equitable` (3),
`Title/Tenure` (2). These are grounds for resisting a claim, not claims. **Excluded** from the merge
rather than folded in, so the routing error stays visible.

**Non-property contamination survives `screen.py`** and reaches the family level: `Criminal Breach of
Trust`, `ElectionDispute`, `CompassionateAppointment`, `Insurance`, `Taxation`, `RentControl`, each
1–2 cases. The merge absorbs them into super-families, which is sloppy at this volume but should be
fixed at the screen.

**The merge is not frozen.** It is derived from 805 cases and should be re-run, not inherited, after
coverage grows.

## 3. Fact patterns (§8)

### 3.1 Transactions (§8.1)

794 cases across 5 merged families. Support is per **case**, so transactions are sets — an atom
appearing 20 times in one verbose judgment has support 1, not 20. Atoms carry an `@p`/`@d` view
suffix so a rule can say "the *plaintiff* asserted continuous possession" rather than "continuous
possession appears somewhere in this case".

### 3.2 Mining (§8.2): nothing survives correction

FP-Growth per family, `forum_heldout`, outcome rules fitted on **train only** (a rule fitted on test
is a leak that carries the answer):

| family | cases | closed itemsets | stable (≥80% of 20 bootstraps) | **BH-significant** |
|---|---|---|---|---|
| PrivateTitlePossession | 2,354 | 67 | 53 | **0** |
| StateAction | 1,148 | 196 | 115 | **0** |
| ContractInstrument | 859 | 91 | 58 | **0** |
| _UNASSIGNED | 550 | 38 | 23 | **0** |
| TenureOccupancy | 188 | 53 | 31 | **0** |
| Mixed_Compensation_LandAcquisition | 87 | 53 | 31 | **0** |
| **total** | | **498** | **311** | **0** |

Confirmed at 6× the earlier scale: 498 closed itemsets, 311 stable across 20 bootstraps, **zero**
significant after Benjamini–Hochberg in any family.

**351 patterns tested · 10 pass uncorrected p<0.05 · 17.6 expected by chance alone · 0 survive
Benjamini–Hochberg.**

Fewer nominal "discoveries" than noise would produce. This is not a weak effect awaiting more data;
there is nothing there. Without the correction, 10 patterns would have shipped as a pattern library.

**Patterns are stable and still carry no outcome information** (135/193 stable in the largest
family). Stability and significance are independent, and reporting the first without the second is
exactly how a pattern library gets mistaken for a finding.

### 3.3 Why: the atoms carry no signal

FP-Growth mines the canonical atoms of §6.2, and the ladder shows those atoms sit near chance
(AUROC 0.517 at n=881). **No pattern over uninformative atoms can be significant.**

The stronger version, established at full scale (`reports/M5_report.md` §2): an *induced* vocabulary
of equal granularity is no better, and more atoms do not help. So §8's null is a property of
**discretisation itself**, not of this ontology's particular choices — and the approach is
mis-specified for this corpus rather than merely under-tuned.

### 3.4 Element mapping (§8.3): not built

§8.3 requires `pattern_element_edges(pattern_id, element_id, relation, confidence, reviewer)`,
seeded by LLM proposal and human-reviewed. Blocked twice over: there are no significant patterns to
map, and the elements carry no burden metadata (BLOCKER B2). The §10 `E` group uses a crude
subcategory-name match as a declared stand-in.

## 4. §7 / §8 acceptance

| requirement | status |
|---|---|
| §7: every case has ≥1 family | partly — 113 of 794 transaction cases unassigned |
| §7: low-support families flagged and merged or excluded | **met** — merged with a data-backed §6.3 entry |
| §7: cluster-to-module NMI/purity reported | **met** — and the answer is that alignment fails |
| §8: per-family pattern library exported | **met** — `Data/patterns/patterns_*.json` |
| §8: stability report | **met** — 20 bootstraps, 234 of 351 stable |
| §8: top-20 discriminative patterns per family | **vacuous** — 0 are discriminative after BH |
| §8: element mapping coverage | **not built** (B2) |

## 5. Next

1. Re-run at full coverage; the family merge and the pattern mining are both sensitive to n.
2. **Fix the vocabulary first (B3).** §8 cannot produce a significant pattern while its atoms are at
   chance, so re-mining before §6.2 is repaired would just re-confirm the null.
3. Validate claim-family assignment against the right list per `kind` — 18% of claim families and 34%
   of defence families fell outside their lists.
