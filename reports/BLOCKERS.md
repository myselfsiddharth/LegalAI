# BLOCKERS

## B1 — Ontology v1 source document is missing (blocks §6.3, §8.3, §7)

`PROJECT.md` §6.3: "The provided `Land_Property_Dispute_India_Ontology_Refined.docx` is
**v1**; convert it to `ontology/ontology_v1.yaml` first."

That file is not on disk. `Docs/` contains the Templeton grant proposal, the course lecture
decks, and the dataset zip — no ontology document, refined or unrefined.

**What is recoverable without it:** the 15 claim modules, listed verbatim in §7 of
PROJECT.md (Title, Possession/Injunction, Lease/Tenancy, Redevelopment, Constitutional
Deprivation, Ultra Vires, Co-operative Society, Adverse Possession/Limitation, Partition,
Specific Performance, Cancellation, Mortgage/Redemption, Succession, Benami, Land
Acquisition); the 6 defense groups in §7.5; and `scripts/ontology/land_dispute_ontology.json`
from the earlier work (classes, roles, ie_concepts, terminology — but no elements).

**What is not:** per-claim **element** lists and their **burden metadata**
(`burden_on`, `burden_standard`, `burden_shifts_when`), the evidence-type taxonomy (§8,
seed for canonical vocabulary per §6.2 step 1), and the §2 `AuthorityRecord` field
definitions.

**Impact:** §8.3 pattern→element mapping and the `E` feature group in §10.1 have no
authored element inventory. `Data/processed/claim_elements.json` (29 elements, mined from
judgment text, 6 claim types) is a partial substitute but covers 6 of 15 families and
carries no burden metadata.

**Needed from the user:** the .docx, or confirmation to proceed with the mined element
catalogue plus an LLM-drafted, human-reviewed burden annotation.
