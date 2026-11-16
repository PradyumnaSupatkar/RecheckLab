# Paper-ready subsection: learned-gate disagreements

## Disagreement and incremental policy value

The frozen RecheckLab gate disagreed with completion-based rechecking on 30 recorded rollout rows across 16 canonical question groups. There were two disagreements in the MATH source test (two questions), none in the CommonsenseQA target test, 18 in MATH500 (nine questions), and ten in GSM8K (five questions). The source-test gate added two stop-completed rechecks; both initial answers were parsed and correct, and both revisions were NO_CHANGE. In MATH500 and GSM8K, every disagreement occurred on a length-limited initial record with a stored parsed answer and a used finalizer. The gate skipped those completion-selected revisions, including four HELP rows in MATH500 and three in GSM8K. No disagreement was labeled HURT.

| Cohort | Disagreement rows / questions | Final accuracy: ML / completion | ML − completion (95% question-cluster CI, pp) | Mean revision tokens/question: ML / completion | Mean total tokens/question: ML / completion | ML − completion total tokens/question (95% CI) |
|---|---:|---:|---:|---:|---:|---:|
| MATH source test | 2 / 2 | 90.00% / 90.00% | 0.00 [0.00, 0.00] | 1,230.8 / 1,220.4 | 4,412.8 / 4,402.4 | +10.4 [0.0, 26.9] |
| CommonsenseQA target test | 0 / 0 | 76.6712% / 76.6712% | 0.00 [0.00, 0.00] | 91.2 / 91.2 | 2,334.8 / 2,334.8 | 0.0 [0.0, 0.0] |
| MATH500 | 18 / 9 | 75.30% / 75.70% | −0.40 [−0.90, 0.00] | 2,621.1 / 2,750.3 | 6,933.6 / 7,062.9 | −129.2 [−222.9, −50.0] |
| GSM8K | 10 / 5 | 94.3518% / 94.4655% | −0.1137 [−0.3033, 0.0000] | 132.7 / 151.0 | 1,313.1 / 1,331.4 | −18.3 [−37.0, −4.4] |

The gate produced no positive accuracy difference on any discordant question group. Five groups lost accuracy because the gate skipped a recorded helpful revision (three MATH500 and two GSM8K groups); the remaining 11 groups had no accuracy difference. The lower recorded token cost on MATH500 and GSM8K therefore reflects a cost-versus-accuracy trade-off, not a demonstrated learned-gating improvement. The source-test deviations add cost with no accuracy gain, while the CSQA decisions are identical. Confidence intervals are clustered by question and conditional on the saved response library; they do not establish equivalence or fresh generalization.

The released records show that all 28 MATH500/GSM8K discordances were length-limited and had a finalizer-produced parsed base answer. Parsed answers changed on 25 of the 30 discordant rows, but only seven rows were HELP; the other 18 were NO_CHANGE. These labels and response fields do not establish whether an apparent correction reflects semantic reasoning, answer finalization, or answer formatting. The gold answer strings are omitted, and no manual semantic adjudication was performed.

The paired question-cluster effects are visualized in [`figures/gate_vs_completion_effects.png`](figures/gate_vs_completion_effects.png); every rollout, score, threshold, action, correctness transition, and signed action-token difference is listed in [`tables/DISAGREEMENTS.csv`](tables/DISAGREEMENTS.csv).
