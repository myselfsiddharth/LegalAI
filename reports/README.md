# reports/

Stage acceptance against `PROJECT.md`. For the chronological record of decisions, dead ends and
bugs, see `../WORKLOG.md`.

| report | covers | status |
|---|---|---|
| `ENV_NOTES.md` | §3.1 Voyager configuration | current |
| `M0_recon.md` | §4 corpus reconnaissance | current |
| `M1_report.md` | §5 labels, masking, splits, leakage probes | **acceptance met** |
| `M2_report.md` | §6 fact extraction, canonical vocabulary | full scale — acceptance NOT met (B3, B4) |
| `M3_report.md` | §7 claim families, §8 fact patterns | full scale — taxonomy unsupported, patterns null |
| `M5_report.md` | §10 outcome models, ablations, the representation ladder | **full scale, n=881** |
| `BLOCKERS.md` | open blockers B2–B4 (B1 closed) | current |

M4 (§9 statutes and precedents) is not started; assets for it exist in `scripts/`.

## The one-paragraph state of the project

A leakage-controlled benchmark is in place and passes its own sensitivity check (`order_only` AUROC
0.983, `masked` 0.655, chance 0.500). Within it, measured on **881 test cases**, the structured
pipeline predicts outcome at chance, and the representation ladder localises why: extraction loses
0.043 (0.657 → 0.614) and **discretising facts into a label vocabulary loses a further 0.097**
(→ 0.517). That last step is not the ontology's fault — an induced vocabulary of equal granularity
is no better (P=0.846), and more atoms do not help. §8 consequently finds **zero of 498** fact
patterns significant after Benjamini–Hochberg. The best arm overall is case **metadata** alone
(AUROC 0.607), which beats every arm that adds facts to it; a per-decade majority baseline reaches
macro-F1 0.557. So the pipeline's useful product is the span-verified, party-attributed fact set,
not its symbolic projection — and §8 as specified is mis-specified for this corpus.
