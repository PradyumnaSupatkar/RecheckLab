# Frozen Instruct evaluation results

**Sample:** 40 fresh, pinned GSM8K main/train questions; all selected and frozen before development inference.  
**Checkpoint:** Qwen3-4B-Instruct-2507, `qwen3:4b-instruct`, digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`.  
**Gate:** frozen v0.2, unchanged; cross-model/cross-protocol OOD transfer diagnostic.

## Result

- Calls: 80; successful recorded calls: 80; total measured inference wall time: 419.6s.
- Parse failures: initial 0/40; revision 0/40.
- Question-level transitions: HELP 1, HURT 0, NO_CHANGE 39.
- All four policies use the same saved initial/revision response pair per question. ML decisions use only the six canonical initial-only inputs; neither correctness, keys, nor revised responses enter the feature frame.

## Policies

| Policy | Accuracy (correct/40; Wilson 95% CI) | Rechecks | HELP captured/missed | HURT accepted/avoided | Mean prompt+output tokens/q | Replayed latency/q |
|---|---:|---:|---:|---:|---:|---:|
| Never recheck | 37/40 (92.5%; 80.1%-97.4%) | 0/40 (0.0%) | 0/1 | 0/0 | 181.2 | 5.22s |
| Always recheck | 38/40 (95.0%; 83.5%-98.6%) | 40/40 (100.0%) | 1/0 | 0/0 | 500.2 | 10.49s |
| Completion heuristic | 37/40 (92.5%; 80.1%-97.4%) | 0/40 (0.0%) | 0/1 | 0/0 | 181.2 | 5.22s |
| Frozen v0.2 ML gate (cross-model/cross-protocol OOD) | 37/40 (92.5%; 80.1%-97.4%) | 0/40 (0.0%) | 0/1 | 0/0 | 181.2 | 5.22s |

Selective policy latency is replayed from measured initial-call time plus measured revision-call time only for questions selected by that policy. It is not end-to-end deployed gate latency. All questions received a revision for paired comparison.

The question-level bootstrap uses 20,000 resamples with fixed seed 20261117 for paired accuracy differences versus never recheck. Intervals are descriptive; do not infer significance or generalization from this one 40-question dataset. See `POLICY_COMPARISON.csv`, `QUESTION_LEVEL.csv`, `ML_PREDICTIONS.csv`, and `RAW_CALLS.jsonl` for the complete evidence.

## Error and uncertainty detail

All 80 evaluation calls ended with `stop`, and none used the response thinking channel. Final-answer extraction succeeded on all 40 initial and all 40 revised responses. Five parseable responses were incorrect (three initial, two revised); these are model errors, not parser failures. One revision corrected an initially wrong response (HELP); no revision changed a correct answer to a wrong one (HURT).

The paired question bootstrap estimated the always-recheck accuracy difference versus never at +2.5 percentage points (95% interval 0.0 to +7.5); completion heuristic and ML gate each differed by 0 points (0.0 to 0.0) because neither selected any revisions. These small-sample intervals do not establish significance or generalization.
