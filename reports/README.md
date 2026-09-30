# reports/

Stage acceptance against `PROJECT.md`. For the chronological record of decisions, dead ends and
bugs, see `../WORKLOG.md`.

| report | covers | status |
|---|---|---|
| `ENV_NOTES.md` | §3.1 Voyager configuration | current |
| `M0_recon.md` | §4 corpus reconnaissance | current |
| `M1_report.md` | §5 labels, masking, splits, leakage probes | **acceptance met** |
| `M2_report.md` | §6 fact extraction, canonical vocabulary | interim — acceptance NOT met (B3, B4) |
| `M3_report.md` | §7 claim families, §8 fact patterns | interim — taxonomy unsupported, patterns null |
| `M5_report.md` | §10 outcome models, ablations, the representation ladder | interim — n=143 |
| `BLOCKERS.md` | open blockers B2–B4 (B1 closed) | current |

M4 (§9 statutes and precedents) is not started; assets for it exist in `scripts/`.

## The one-paragraph state of the project

A leakage-controlled benchmark is in place and passes its own sensitivity check (`order_only` AUROC
0.983, `masked` 0.655, chance 0.500). Within it, the structured pipeline predicts outcome **at
chance**, and the representation ladder localises why: extraction preserves the signal
(0.578 → 0.577) and **canonicalisation destroys it** (→ 0.502). §8 consequently finds **zero** of 351
fact patterns significant after Benjamini–Hochberg, against 17.6 expected by chance. The strongest
single predictor is one procedural feature (prior-court disposition, AUROC 0.686). The binding
constraint is §6.2's vocabulary, which cannot name 22.7% of extracted facts.
