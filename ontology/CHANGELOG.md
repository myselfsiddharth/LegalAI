# Ontology CHANGELOG (§6.3)

Every ontology or vocabulary change is recorded here with its justification and the number of items affected.

## 2026-09-30 — element burden metadata filled
`src/data/ontology_burden.py`, prompt `element_burden.v1`, model `llama4-scout-17b`.
The source document requires every element to carry `burden_on`, `burden_standard` and `burden_shifts_when`, and supplies them for none of the 66 elements; 6 of 15 claims carry a claim-level default in a notes line.
- filled: **66** elements
- from the ontology's own notes: 28
- LLM-drafted and **not yet reviewed**: 38
- claim-level notes overrode the draft's `burden_on`: 0
- no answer, left blank: 0

Every value carries `burden_provenance`. Results that depend on an `llm_draft` value must be reported as resting on drafted metadata, not on the ontology. Reviewing an element means setting its provenance to `human`.

## 2026-09-30 — claim families merged 15 -> 5
`src/cluster/family_merge.py`. Derived from 465 claims over 268 cases.

**Why.** The 15-way taxonomy is not supported by the data at this scale:
- HDBSCAN on claim embeddings: ARI **-0.005** against the assigned families, 80% outliers.
- Agglomerative k=5 by silhouette: NMI 0.271, purity 0.318, silhouette **0.035** — almost no structure at 15-way granularity.
- Only **2 of 15** families cleared §7's 30-case floor.
- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.

The clusters are legally coherent one level up, so the taxonomy is too FINE for this corpus rather than wrong. Each family is assigned to the cluster its claims most often land in; only the merged names are authored.

**Effect.** Families clearing the 30-case floor: 2 -> 3, which is what makes §8 per-family mining runnable.

**Merged families.**

| merged | cases | absorbs |
|---|---|---|
| `PrivateTitlePossession` | 152 | Adoption, AdversePossession, Benami, Partition, Possession, Pre-emption, Preemption, Succession, Title, Trust |
| `StateAction` | 59 | ConstitutionalDeprivation, Criminal Breach of Trust, ElectionDispute, Evidence/Procedure, LandAcquisition, LandRevenue, NEW:MalaFides, Taxation, UltraVires |
| `ContractInstrument` | 56 | Cancellation, CooperativeSociety, Evidence, Insolvency, Limitation, MortgageRedemption, NEW: SettlementChallenge, Procedure, SpecificPerformance |
| `TenureOccupancy` | 20 | Contract, Injunction, LeaseTenancy, RentControl, RentTenancy, Tenancy, Tenure |
| `Mixed_Compensation_StampDuty` | 3 | Compensation, NEW:Compensation, StampDuty |

**Excluded.** {'Procedural/Forum': 14, 'Statutory/Regulatory Subservience': 9, 'Public Interest/Planning Policy': 4, 'Equitable': 2} — these are DEFENCE families the extractor routed to claims. They are grounds for resisting a claim, not claims, and folding them into a claim super-family would bury that routing error.

**Not frozen.** Derived from 300 cases. Re-run after extraction scales up; a family below the floor here may clear it later, and the merge should be revisited rather than inherited.

## 2026-09-30 — claim families merged 15 -> 6
`src/cluster/family_merge.py`. Derived from 1408 claims over 805 cases.

**Why.** The 15-way taxonomy is not supported by the data at this scale:
- HDBSCAN on claim embeddings: ARI **-0.005** against the assigned families, 80% outliers.
- Agglomerative k=6 by silhouette: NMI 0.271, purity 0.318, silhouette **0.022** — almost no structure at 15-way granularity.
- Only **9 of 15** families cleared §7's 30-case floor.
- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.

The clusters are legally coherent one level up, so the taxonomy is too FINE for this corpus rather than wrong. Each family is assigned to the cluster its claims most often land in; only the merged names are authored.

**Effect.** Families clearing the 30-case floor: 9 -> 4, which is what makes §8 per-family mining runnable.

**Merged families.**

| merged | cases | absorbs |
|---|---|---|
| `StateAction` | 424 | Ceiling, ConstitutionalDeprivation, Corruption/Public Office, Debt Relief, LandAcquisition, LandRevenue, Licence, License, NEW: MalaFides, Redevelopment, StatutoryInterpretation, Taxation, Taxation/Assessment, Tenure, Title, UltraVires, UnauthorizedConstruction |
| `PrivateTitlePossession` | 256 | Adoption, AdversePossession, Partition, Possession, Pre-emption, Preemption |
| `ContractInstrument` | 170 | Benami, Cancellation, Corruption, Evidence, Execution/Proceedings, Insolvency, Insolvency/Receiver, MortgageRedemption, SpecificPerformance, StampDuty, Succession, Suretyship, Trust, Trusts, Wakf |
| `Mixed_Limitation_Procedure` | 47 | CompassionateAppointment, Contract, Contractual Obligations, CooperativeSociety, Criminal Breach of Trust, ElectionDispute, Evidence/Procedure, FalseCharge, Limitation, Limitation/Delay, Maintenance, NEW: SettlementChallenge, NEW:AnticipatoryBail, NEW:Challenge to Conviction, NEW:Cheating, NEW:Condonation, NEW:CondonationOfDelay, NEW:Limitation, NEW:MalaFides, NEW:Procedural, NEW:ProceedingAgainstAccused, NEW:Reputation, Procedure, Procedure/Execution, ProfessionalMisconduct, Review, Service/Employment, Trusts and Charities |
| `TenureOccupancy` | 23 | Eviction, Injunction, LeaseTenancy, Procedure/Forum, Rent, RentControl, RentFixation, RentTenancy, Tenancy |
| `Mixed_Compensation_NEW:Compensation` | 15 | Compensation, Insurance, NEW: Monetary Relief, NEW:Compensation |

**Excluded.** {'Procedural/Forum': 43, 'Statutory/Regulatory Subservience': 31, 'Public Interest/Planning Policy': 6, 'Equitable': 3, 'Title/Tenure': 2} — these are DEFENCE families the extractor routed to claims. They are grounds for resisting a claim, not claims, and folding them into a claim super-family would bury that routing error.

**Not frozen.** Derived from 300 cases. Re-run after extraction scales up; a family below the floor here may clear it later, and the merge should be revisited rather than inherited.
