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

## 2026-09-30 — claim families merged 15 -> 5
`src/cluster/family_merge.py`. Derived from 7441 claims over 4235 cases.

**Why.** The 15-way taxonomy is not supported by the data at this scale:
- HDBSCAN on claim embeddings: ARI **-0.005** against the assigned families, 80% outliers.
- Agglomerative k=6 by silhouette: NMI 0.271, purity 0.318, silhouette **0.025** — almost no structure at 15-way granularity.
- Only **16 of 15** families cleared §7's 30-case floor.
- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.

The clusters are legally coherent one level up, so the taxonomy is too FINE for this corpus rather than wrong. Each family is assigned to the cluster its claims most often land in; only the merged names are authored.

**Effect.** Families clearing the 30-case floor: 16 -> 5, which is what makes §8 per-family mining runnable.

**Merged families.**

| merged | cases | absorbs |
|---|---|---|
| `PrivateTitlePossession` | 2365 | Accountability, Adoption, AdoptiveSon, AdversePossession, Agency, Contract/ Agreement, Corruption, Custody, Easement, Fraud, Injunction, Insolvency/Receiver, Laches/Delay, LandUsePlanning, License, Licensing, Maintenance, Management, Misappropriation, NEW: LegitimacyInheritance, NEW: SettlementChallenge, NEW:Accounting, NEW:Compensation, NEW:CompensationForCheating, NEW:CounterClaim, NEW:Declaratory, NEW:Estoppel/ Reliance, NEW:FinancialSupport, NEW:General Relief, NEW:InvestigationAndProsecution, NEW:Procedural, NEW:PropertyDispute, NEW:RelationshipStatus, NEW:RightToUseLand, Nationalisation, Partition, Partnership, Possession, Possession, Partition, Property, Protection from Eviction, Restitution, Right to Privacy, Self-Defence, ServiceTax, Specific endowment, SpecificEndowment, Stridhana, Succession, Tenancy, Title, Title/Possession, Trust, Trusts, Trusts and Charities, Trusts and Trustees, UnauthorizedConstruction, Wakf, Waqf, Will-Construction, Workmen's Rights |
| `StateAction` | 1162 | AdministrativeAction, Arbitrariness, Canal Charges, Capital Gains, Ceiling, Ceiling on landholding, Change in Government Policy, ConstitutionalDeprivation, ConsumerDispute, Contract, Contractual Obligations, ContractualInterpretation, CooperativeSociety, Criminal Liability, Development, Election, Election/Representation, ElectionDispute, Employment, Environmental Clearance, Environmental Concern, Environmental Protection, Equality, EstateDuty, Estoppel, Exemption, Exemption/Statutory, IncomeTax, Industrial Dispute, Insurance, Jurisdiction, LandAcquisition, LandRevenue, LegitimateExpectation, Licence, Membership, Mineral Concession, Mineral Royalties, Mining Lease, NEW: LenderLiability, NEW: MalaFides, NEW:Assignment, NEW:Challenge to High Court decision, NEW:Challenge to Land Allotment, NEW:Challenge to Proceedings, NEW:Constitutional Validity, NEW:Costs, NEW:Environmental Regulation, NEW:ExpungingOfRemarks, NEW:LoanRecoveryAbuse, NEW:MalaFides, NEW:Review, NEW:TenancyCeiling, Policy/Regulatory, Pre-emption, Preemption, Procedural Irregularity, ProceduralFairness, Procedure/Execution, PublicInterest/PlanningPolicy, PublicPurpose, RateableValue, Rectification, Redevelopment, Refund, Refund/Recovery, Registration, ResJudicata, Royalty/Payment, Service Matters, Service/Employment, Statutory Interpretation, StatutoryInterpretation, Suretyship, Tariff/Regulatory, TaxExemption, Taxation, Taxation/Assessment, Taxation/Challenge to Tax Demand, Taxation/Exemption, Taxation/Levy, Taxation/Liability, TownPlanning, UltraVires |
| `ContractInstrument` | 863 | Arbitrability, Arbitration, Arbitration/Challenge, Benami, Cancellation, Challenge to Court Order, CompassionateAppointment, Contempt, Contempt of Court, Corruption/Public Office, Criminal Breach of Trust, Criminal Defence, Criminal Procedure, CriminalProceedings, Debt Relief, DebtAdjustment, DebtRelief, Declaration, Evidence, Evidence/Procedure, Execution, Execution/Procedure, Execution/Proceedings, Exemption from Attachment, FalseCharge, Gift, Insolvency, Limitation, Limitation/Delay, MCOCA, MortgageRedemption, Murder, NEW: Auction Validity, NEW: Criminal, NEW: Criminal Appeal, NEW: Criminal Complaint, NEW:Abatement, NEW:AnticipatoryBail, NEW:Arbitration Award, NEW:Challenge to Conviction, NEW:Challenge to Criminal Proceedings, NEW:ChallengeToArbitralAward, NEW:Cheating, NEW:CheatingAndForgery, NEW:Claim under specific rules, NEW:Condonation, NEW:CondonationOfDelay, NEW:Conspiracy, NEW:Contempt, NEW:Corruption, NEW:CourtProceedings, NEW:Criminal Conspiracy, NEW:Criminal Defence, NEW:Criminal Liability, NEW:CriminalProceedings, NEW:Estoppel, NEW:Evidence, NEW:FalseAccusation, NEW:FalseClaim, NEW:FrivolousComplaint, NEW:ImpersonationToDefraud, NEW:InsufficientEvidence, NEW:Investigation, NEW:Limitation, NEW:Malafide, NEW:Misappropriation, NEW:Misrepresentation, NEW:Murder, NEW:MurderForHire, NEW:MurderProsecution, NEW:MurderTrial, NEW:ProceedingAgainstAccused, NEW:ProfessionalMisconduct, NEW:ProsecutionProceedings, NEW:PunishmentForOffence, NEW:Quashing of Proceedings, NEW:Representation, NEW:Reputation, NEW:ResignationLiability, NEW:SelfDefence, NEW:Specific Relief, NEW:Transfer, NEW:challenge to decree, None, Priority, Priority/Payment, Priority/Preference, Procedural, Procedure, Procedure/Limitation, ProfessionalConduct, ProfessionalMisconduct, PublicInterest, Review, Review Jurisdiction, Satisfaction, Service/Master and Servant, SpecificPerformance, Stamp Duty, StampDuty, Winding up and Insolvency |
| `TenureOccupancy` | 190 | Eviction, LeaseTenancy, NEW: Limitation, Procedure/Forum, Protection, ProtectionAgainstEviction, Recovery of Dues, Rent, Rent Control, RentControl, RentFixation, RentTenancy, Tenure |
| `Mixed_Compensation_Land Acquisition` | 87 | Compensation, Contract/Breach, Costs, Debt, Gratuity, Interest, Land Acquisition, NEW: Monetary Relief, NEW:ConspiracyToDefraud, NEW:Expenses, NEW:ProceduralFairness, NEW:Refund, NEW:fraud, Recovery, Recovery/Refund, Reversion |

**Excluded.** {'Procedural/Forum': 180, 'Statutory/Regulatory Subservience': 153, 'Public Interest/Planning Policy': 30, 'Equitable': 11, 'Title/Tenure': 8, 'Consent/Compliance': 1} — these are DEFENCE families the extractor routed to claims. They are grounds for resisting a claim, not claims, and folding them into a claim super-family would bury that routing error.

**Not frozen.** Derived from 300 cases. Re-run after extraction scales up; a family below the floor here may clear it later, and the merge should be revisited rather than inherited.

## 2026-10-01 — claim families merged 15 -> 4
`src/cluster/family_merge.py`. Derived from 7158 claims over 4216 cases.

**Why.** The 15-way taxonomy is not supported by the data at this scale:
- HDBSCAN on claim embeddings: ARI **-0.005** against the assigned families, 80% outliers.
- Agglomerative k=4 by silhouette: NMI 0.271, purity 0.318, silhouette **0.027** — almost no structure at 15-way granularity.
- Only **16 of 15** families cleared §7's 30-case floor.
- The extractor assigned **42** distinct families, inventing 27 beyond the ontology.

The clusters are legally coherent one level up, so the taxonomy is too FINE for this corpus rather than wrong. Each family is assigned to the cluster its claims most often land in; only the merged names are authored.

**Effect.** Families clearing the 30-case floor: 16 -> 4, which is what makes §8 per-family mining runnable.

**Merged families.**

| merged | cases | absorbs |
|---|---|---|
| `PrivateTitlePossession` | 1446 | Administrative Approval, Adoption, AdversePossession, Agency, Benami, Criminal, Easement, Easements, Educational Institution, Endowment, Inheritance, Injunction, Management, NEW: Ceremonial Rights, NEW: Legitimisation, NEW:Cheating and Breach of Trust, NEW:DeclaratoryRelief, NEW:Forgery, NEW:InvestigationAndProsecution, NEW:Procedural Challenge, NEW:adoptionEffect, NEW:adoptionValidity, Partition, Private Defence, Protection, ResJudicata, Shebaitship, SpecificEndowment, Succession, Title, Trust, Trust/Charity, Trust/Property, Trusts, Wakf, Wakf Administration |
| `TenureOccupancy` | 1325 | Allotment, BonaFideNeed, LeaseTenancy, NEW:Land Tenure, Possession, Protection from Eviction, Tenancy, TenantProtection, Tenure |
| `StateAction` | 1193 | AgrarianReform, Capital Gains, Ceiling, Ceiling Limit, CharitablePurpose, Compensation, Compensation/Interest, Constitutional Challenge, ConstitutionalDeprivation, ConsumerProtection, Continuity of Employment, Contract/ Agreement, Contract/Cancellation, Contract/Interpretation, Definition of Mineral, Environmental Law, Environmental Protection, Equality/Non-Arbitrariness, EstateDuty, Estoppel, Gratuity, Income from Other Sources, IncomeTax, Interest on Compensation, Land Acquisition, Land Allotment, LandAcquisition, LandRevenue, LandUsePlanning, Legislative Competence, Licence, Licensing, Malafides, Mineral Concession, Mineral Concessions, Mineral Rights, Mining Lease, NEW: AbuseOfPower, NEW: LandReforms, NEW: recovery, NEW: setoff, NEW:Assignment, NEW:Challenge to Proceedings, NEW:Claim for relief under Section 25, NEW:Corruption/AntiBribery, NEW:CounterClaim, NEW:EstoppelByAcquiescence, NEW:Malafide, NEW:PolicyEstoppel, NEW:Statutory Challenge, Pre-emption, Preemption, Priority, PublicInterest/PlanningPolicy, PublicPurpose, Rateable Value, RateableValue, Redevelopment, Refund/Cancellation, RentControl, RentFixation, RentTenancy, RentTenure, RevenueExpenditure, Review Jurisdiction, Right of Pre-emption, Royalty/Payment, Service Matters, Statutory Interpretation, TaxExemption, Taxation, Taxation/Exemption, Taxation/Levy, Taxation/Rate, TownPlanning, Trade Regulation, UltraVires, Water Charges, Water Dispute |
| `ContractInstrument` | 925 | Administrative Challenge, Arbitration, Cancellation, Cheating, Compensation/Entitlement, Contempt, Contempt of Court, Contract, Contract/Breach, Contract/Performance, Contract/SpecificPerformance, Contract/Termination, Contractual Obligations, Control, CooperativeSociety, CorporateInsolvency, Corruption, Corruption/Anti-Corruption, Criminal Defence, Criminal Liability, Criminal Procedure, CriminalProceedings, CriminalProsecution, Custody, Debt Recovery, Debt Relief, DebtRelief, Election, Election Disqualification, Election/Delimitation, Execution, Execution/Procedure, Execution/Stay, ExemptionFromAttachment, Financial Loss, Forgery, Fraud, Gift, Industrial Dispute, Insolvency, Insurance, Interest, Jurisdiction, Limitation, Limitation/Procedure, Maintenance, Misappropriation, Mortgage, MortgageRedemption, Murder, NEW: Damages, NEW: Defence to Specific Performance, NEW: FalseImplication, NEW: Licence vs Lease, NEW: Monetary Claim, NEW: locus standi, NEW:AbetmentToSuicide, NEW:AbuseOfProcess, NEW:Accounting, NEW:Challenge to Administrative Action, NEW:Challenge to Conviction, NEW:Challenge to Land Acquisition Proceedings, NEW:Challenge to Magistrate's Order, NEW:Challenge to Prosecution Case, NEW:ChequeBouncing, NEW:Compensation, NEW:Condonation, NEW:CondonationOfDelay, NEW:Conspiracy, NEW:ConspiracyToDefraud, NEW:Contempt, NEW:Costs, NEW:Criminal Defence, NEW:Criminal Investigation, NEW:Criminal Liability, NEW:CriminalBreachOfTrust, NEW:CriminalProceedings, NEW:DetentionLaw, NEW:FinancialRecovery, NEW:Fraud, NEW:ImpersonationToObtainProperty, NEW:Investigation, NEW:Miscellaneous Defence, NEW:Murder, NEW:MurderAccusation, NEW:Procedural, NEW:ProceduralChallenge, NEW:Procedure, NEW:QuashingOfProceedings, NEW:Recovery, NEW:Reputation/Defamation, NEW:SelfDefence, NEW:Tenancy, NEW:UnlawfulDetention, NEW:extension of time, NEW:fraud, NEW:fraudulent_obtainment, None, Partnership, Priority/Preference, Priority/Preferential Claim, Procedural, ProceduralFairness, Procedure, Procedure/Forum, Procedure/Limitation, ProfessionalMisconduct, ProfessionalRegulation, Quasi-Judicial Immunity, Recovery, Recovery of Money, Refund, Review, Service/Employment, SpecificPerformance, StampDuty, Suretyship, Taxation/Assessment, Taxation/Deductions, TenantEviction, Trust Administration, Trusts and Charities, Trusts and Endowments, UndueInfluence, Will and Testament, Will-Construction |

**Excluded.** {'Procedural/Forum': 175, 'Statutory/Regulatory Subservience': 131, 'Public Interest/Planning Policy': 33, 'Title/Tenure': 12, 'Equitable': 7, 'Consent/Compliance': 3} — these are DEFENCE families the extractor routed to claims. They are grounds for resisting a claim, not claims, and folding them into a claim super-family would bury that routing error.

**Not frozen.** Derived from 300 cases. Re-run after extraction scales up; a family below the floor here may clear it later, and the merge should be revisited rather than inherited.
