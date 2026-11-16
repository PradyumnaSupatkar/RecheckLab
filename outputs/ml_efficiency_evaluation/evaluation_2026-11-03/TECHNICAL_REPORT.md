# Technical definitions and complete metrics

policy_metrics.csv contains every requested efficiency/decision metric for all three partitions and five policies. Tokens equal recorded prompt-plus-generation usage; NO_CHANGE tokens remain in costs. Average calls avoided = questions minus weighted selected outcomes, for a single prospective recheck per question. The offline data collection itself paid for all recorded outcomes.

Classification uses argmax in HELP/NO_CHANGE/HURT order. Metrics have question weights 1/n_q per rollout. Weighted-F1 is support-weighted per-class F1. Raw support and question-equivalent support are both saved.

NO_CHANGE is neutral for accuracy: either action preserves correctness, while skip saves tokens. There is no unique correct binary decision without a compute objective. Helpful-decision precision counts neutral selections as nonhelpful; 1-precision is not the harmful rate. Undefined ratios are blank/null, not zero. sklearn per-class zero-division=0 is saved alongside precision_defined flags.

Oracle mask = observed label HELP. It is minimum-token among per-outcome accuracy-maximizing policies, since neutral/harmful revisions are skipped. This mask can differ across identical-input stochastic rollouts, unlike a realizable initial-only gate. Its costs, calls and frontier position are hindsight diagnostics.

accuracy_change_vs_always_pp is the accuracy difference. accuracy_ratio_to_always_pct is an aggregate accuracy ratio and may exceed 100%; it is not the fraction of identical correct answers retained.

## MATH source test

| Class | Precision | Recall | F1 | Raw support | Question-equivalent support | Distribution |
|---|---:|---:|---:|---:|---:|---:|
| HELP | 0.588235 | 0.923077 | 0.718563 | 65 | 65 | 16.250000% |
| NO_CHANGE | 0.975694 | 0.846386 | 0.906452 | 332 | 332 | 83.000000% |
| HURT | 0.000000 | 0.000000 | 0.000000 | 3 | 3 | 0.750000% |

Confusion matrices: true rows / predicted columns HELP, NO_CHANGE, HURT.

Raw outcome rows:
```text
[[ 60   5   0]
 [ 41 281  10]
 [  1   2   0]]
```
Question-equivalents:
```text
[[ 60.   5.   0.]
 [ 41. 281.  10.]
 [  1.   2.   0.]]
```

## CommonsenseQA target test

| Class | Precision | Recall | F1 | Raw support | Question-equivalent support | Distribution |
|---|---:|---:|---:|---:|---:|---:|
| HELP | 0.160000 | 0.380952 | 0.225352 | 21 | 10.5 | 1.432469% |
| NO_CHANGE | 0.942090 | 0.969477 | 0.955587 | 1376 | 688 | 93.860846% |
| HURT | undefined (stored 0 convention) | 0.000000 | 0.000000 | 69 | 34.5 | 4.706685% |

Confusion matrices: true rows / predicted columns HELP, NO_CHANGE, HURT.

Raw outcome rows:
```text
[[   8   13    0]
 [  42 1334    0]
 [   0   69    0]]
```
Question-equivalents:
```text
[[  4.    6.5   0. ]
 [ 21.  667.    0. ]
 [  0.   34.5   0. ]]
```

## Separate source-development policy metrics

| Dataset / policy | Final accuracy | Recheck | Avg. tokens | Total tokens saved | HELP captured | HURT incurred |
|---|---:|---:|---:|---:|---:|---:|
| MATH source development / Never recheck | 80.7500% | 0.000% | 3005.64 | 45.57% | 0.00% | 0 |
| MATH source development / Always recheck | 91.0000% | 100.000% | 5521.68 | 0.00% | 100.00% | 6 |
| MATH source development / Completion rule | 92.0000% | 23.500% | 4121.01 | 25.37% | 100.00% | 2 |
| MATH source development / Frozen ML gate | 92.0000% | 23.250% | 4081.34 | 26.09% | 100.00% | 2 |
| MATH source development / Oracle (hindsight) | 92.5000% | 11.750% | 3396.99 | 38.48% | 100.00% | 0 |

## Timing evidence

{
  "possible_timing_columns": {
    "source_development": [],
    "source_test": [],
    "target_test": []
  },
  "historical_offline_experiment_runtime_seconds": 5.769275000006019,
  "trustworthy_per_call_LLM_timing_available": false,
  "measured_LLM_latency_savings_seconds": null,
  "interpretation": "Offline experiment runtime is not LLM latency. Token savings are recorded replay cost proxies."
}

Full benchmark samples and scope are in gate_overhead.json. Those are current offline CPU timings, not v0.2 solver timings. No conversion of tokens into seconds, money, FLOPs or energy is made.

The saved single-record median process CPU time is 0.0 ms on this Windows host,
consistent with timer quantization on short calls. It does not mean zero CPU
work. Use the measured perf_counter wall medians/p95 (4.127/4.551 ms source;
3.999/4.504 ms target) for the reported offline decision overhead.

Independent verification reproduced 255 numeric comparisons, per-class metrics,
all 15 policy evaluations, Pareto membership and all 303 protected-file hashes.
The visualization was visually checked for labels, split separation, overlap
identification and the unattainable-oracle disclaimer.

## Exact code locations

- rechecklab/data.py:34 transition_label; :74 validate_pairs; :233 group_partition; :248 assert_disjoint.
- rechecklab/metrics.py:9 question_weights; :22 policy_metrics; :60 predictor_metrics.
- rechecklab/predictor.py:121 OutcomePredictor.probabilities; :297 revision_score; :303 decide_recheck.
- rechecklab/features.py:117 extract_features.
- rechecklab/experiment.py:34 experiment timer; :131 offline runtime; :133 token-accounting limitations.
- This script: classification, evaluate_policy, mark_pareto and benchmark_gate implement new calculations.
