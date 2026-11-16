# Final lightweight accuracy audit

**NO VERIFIED IMPROVEMENT**

No new policy was evaluated on source-test or target-test. Existing tests were replayed only to verify historical results, after all new choices were frozen. No inference, downloads, application changes, model promotion or production artifact writes occurred.

Bounded work: reuse all 11 prior nested-CV comparisons; fit one predeclared 20-column experimental HGB variant (the original 19 plus log visible-character/token ratio, 20 CV fits); compare six fixed rules; tune one threshold on 244 adaptation questions and confirm on 244 others.

Training-selected classifier: hgb_full_baseline. Training-selected rule: length_or_missing. Primary policy selected before development: Rule: length_or_missing.

Source development: previously inspected descriptive confirmation

| Policy | Accuracy % | HELP | HURT | Recheck % | Avg total tokens | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep initial | 80.750000 | 0.000 | 0.000 | 0.0000 | 3005.635 | N/A | N/A |
| Always recheck | 91.000000 | 47.000 | 6.000 | 100.0000 | 5521.676 | N/A | N/A |
| Completion rule | 92.000000 | 47.000 | 2.000 | 23.5000 | 4121.012 | N/A | N/A |
| Frozen gate | 92.000000 | 47.000 | 2.000 | 23.2500 | 4081.343 | 0.558724 | 0.391613 |
| hgb_full_baseline | 91.750000 | 47.000 | 3.000 | 78.5000 | 5070.836 | 0.558724 | 0.391613 |
| hgb_full_plus_log_visible_token_ratio | 91.750000 | 47.000 | 3.000 | 63.8750 | 4840.241 | 0.601554 | 0.376242 |
| Rule: length_or_missing | 92.000000 | 47.000 | 2.000 | 23.7500 | 4125.155 | N/A | N/A |

Target adaptation: fixed held-out confirmation half, never target-test

| Policy | Accuracy % | HELP | HURT | Recheck % | Avg total tokens | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep initial | 78.688525 | 0.000 | 0.000 | 0.0000 | 2213.680 | N/A | N/A |
| Always recheck | 71.926230 | 4.000 | 20.500 | 100.0000 | 4876.613 | N/A | N/A |
| Completion rule | 79.098361 | 1.000 | 0.000 | 4.0984 | 2379.848 | N/A | N/A |
| Frozen gate | 79.098361 | 1.000 | 0.000 | 4.0984 | 2379.848 | 0.308764 | 0.519297 |
| Tune-half threshold | 78.688525 | 0.000 | 0.000 | 3.2787 | 2351.391 | 0.308764 | 0.519297 |

Existing source_test results: AUDIT ONLY

| Policy | Accuracy % | HELP | HURT | Recheck % | Avg total tokens | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep initial | 74.750000 | 0.000 | 0.000 | 0.0000 | 3182.017 | N/A | N/A |
| Always recheck | 90.250000 | 65.000 | 3.000 | 100.0000 | 5747.435 | N/A | N/A |
| Completion rule | 90.000000 | 62.000 | 1.000 | 26.7500 | 4402.420 | N/A | N/A |
| Frozen gate | 90.000000 | 62.000 | 1.000 | 27.2500 | 4412.825 | 0.541671 | 0.411111 |

Existing target_test results: AUDIT ONLY

| Policy | Accuracy % | HELP | HURT | Recheck % | Avg total tokens | Macro-F1 | Log loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| Keep initial | 76.125512 | 0.000 | 0.000 | 0.0000 | 2243.604 | N/A | N/A |
| Always recheck | 72.851296 | 10.500 | 34.500 | 100.0000 | 4719.982 | N/A | N/A |
| Completion rule | 76.671214 | 4.000 | 0.000 | 3.4106 | 2334.847 | N/A | N/A |
| Frozen gate | 76.671214 | 4.000 | 0.000 | 3.4106 | 2334.847 | 0.393647 | 0.322128 |

HELP/HURT are question-equivalent counts; repeated rollouts share each question's weight. Action-only policies have no multiclass probabilities, so macro-F1/log loss are N/A.

Existing target-test oracle is diagnostic only: 77.55798090040928%. It averages paired max(initial,revised) within each question, rather than selecting the best rollout. The old pairs cannot support 80% under the fixed one-recheck action/recorded-outcome scoring.

Current source threshold already exhaustively maximizes source-development HELP-HURT for the fixed score. The adaptation confirmation does not select another threshold. Configuration/rule selection retains winner optimism; pooled OOF final thresholds are tuning statistics. Development and the adaptation pool have been inspected in earlier work. No independent new-test accuracy claim is made.

No transition, weighting, token arithmetic, rollout, standard-choice grading, saved-prediction or artifact-fingerprint error was verified. Public replay uses inherited parsed answers and correctness. Changing extraction from malformed transcripts would be a separately versioned scoring sensitivity, not a silent increase in accuracy. Local-study protocols are excluded.

Code evidence (paths relative to the project root):

| Finding | Exact implementation |
|---|---|
| HELP / HURT / NO_CHANGE correctness transitions | `rechecklab/data.py:34 transition_label`; `:74 validate_pairs` |
| Canonical question-group separation | `rechecklab/data.py:233 group_partition`; `:248 assert_disjoint` |
| Equal question weight; average rollout correctness and tokens | `rechecklab/metrics.py:9 question_weights`; `:22 policy_metrics`; `:60 predictor_metrics` |
| Strict probability-score threshold; exhaustive source-development objective | `rechecklab/predictor.py:303 decide_recheck`; `:308 choose_threshold` |
| Public answers/labels preserved; separate public replay protocol | `rechecklab/data.py:259 public_frames` |
| Independent replay arithmetic and saved target predictions | `scripts/verify_recorded_policies.py:27 independent_metrics`; `:183 verify_recorded_policies` |
| Original initial-only feature allowlist | `rechecklab/features.py:88 extract_one`; `:117 extract_features` |
| Isolated added feature and grouped nested CV | `outputs/final_accuracy_audit/final_audit.py:71 feature_frame`; `:91 train_feature_cv` |
| Rule shortlist and training-only selection | `outputs/final_accuracy_audit/final_audit.py:78 rules`; `:122 source_selection` |
| Source-development confirmation | `outputs/final_accuracy_audit/final_audit.py:168 source_confirmation` |
| Fixed adaptation tune/confirmation split | `outputs/final_accuracy_audit/final_audit.py:198 adaptation_confirmation` |
| Diagnostic oracle, rollout invariants and official-key audit | `outputs/final_accuracy_audit/final_audit.py:222 existing_results_audit` |

Additional audit interpretation: every target-test question has exactly two
rollouts, with identical initial inputs and decisions. Sixty questions have
different outcome labels across those rollouts; they remain one question each.
The saved target gate selects 8 helpful rows, equivalent to 4 helpful questions,
and no harmful rows. Its decision vector matches completion exactly.

Official-key checks found zero mismatches for the 1,422 standard target-test
initial answers and 1,361 standard revised answers. The remaining 44 initial and
105 revised answer strings are nonstandard and publisher-graded incorrect.
Two revised transcripts contain earlier final-answer lines but continue into
truncated self-discussion/placeholders; selecting those earlier lines would
change scoring. These test-audit observations did not select any feature, rule,
or threshold, and no outcomes were relabeled.

The added feature's development macro-F1/log loss improve to 0.601554/0.376242,
but its selected policy falls to 91.75% and uses more tokens. It also loses on
training nested-CV utility (151 versus 152), so it was not promoted after seeing
development. Improved outcome classification is not a verified policy-accuracy
gain. Always rechecking scores 90.25% on the existing source test, but only 91%
on development versus 92% for the frozen gate; selecting it from test results
would not be a legitimate improvement.

Independent read-only verification reproduced all 12 CV comparisons, source
policy masks, adaptation results and 284 protected-file hashes. The new feature
development classifier metrics were computed in the run from its fixed fitted
model; its saved decisions independently reproduce the reported policy metrics.

Next steps: (1) retain v0.2 and present its baseline tie, rare-HURT limitation and oracle honestly; (2) if accuracy research continues, use a separate protocol-matched generation study, since these recorded target pairs have less than one percentage point of remaining oracle headroom.

Runtime: 7.25 seconds. See integrity.json for original-file preservation.
