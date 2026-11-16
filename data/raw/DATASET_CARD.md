---
license: other
language:
- en
task_categories:
- text-classification
pretty_name: SEVRA Recovery Outcomes
size_categories:
- 10K<n<100K
tags:
- reasoning
- selective-verification
- test-time-compute
- qwen3
- recoverability
arxiv: 2606.19808
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*.parquet
  - split: math500
    path: data/math500-*.parquet
  - split: gsm8k
    path: data/gsm8k-*.parquet
  - split: commonsenseqa
    path: data/commonsenseqa-*.parquet
---

# SEVRA Recovery Outcomes

Intervention outcomes used to train and evaluate SEVRA, the recoverability-aware controller from
[Think Again or Think Longer? Selective Verification for Budget-Aware
Reasoning](https://arxiv.org/abs/2606.19808).

Each row pairs a frozen Qwen3-4B base attempt with one post-generation action and records whether
that action repaired an incorrect answer (helpful_fix), damaged a correct answer (harmful_flip),
and how many realized tokens it consumed. Gold answers were used to compute offline labels but are
intentionally omitted from this release.

## Loading

~~~python
from datasets import load_dataset

dataset = load_dataset("Sajib-006/sevra-recovery-outcomes")
train = dataset["train"]
~~~

## Splits

| Split | Unique examples | Action rows | Actions |
|---|---:|---:|---|
| train | 2,000 | 8,000 | accept: 2,000, active_verify: 2,000, continue: 2,000, critique_repair: 2,000 |
| math500 | 500 | 1,500 | accept: 500, active_verify: 1,000 |
| gsm8k | 1,319 | 3,957 | accept: 1,319, active_verify: 2,638 |
| commonsenseqa | 1,221 | 3,663 | accept: 1,221, active_verify: 2,442 |

The train split contains MATH training examples and candidate actions used for intervention
screening and gate training. math500, gsm8k, and commonsenseqa are evaluation outcome logs. Do not
train on the evaluation splits when reproducing the paper.

Multiple rows may share an example_id because actions and stochastic rollouts are paired with the
same base attempt. Any new train/development split must therefore be grouped by example_id.

## Important fields

- query, base_response: problem and initial frozen-solver attempt.
- action, intervention_response: selected recovery action and resulting response.
- base_correct, final_correct: correctness before and after the action.
- helpful_fix: base incorrect and post-action answer correct.
- harmful_flip: base correct and post-action answer incorrect.
- *_actual_tokens: realized prompt-plus-generation token accounting.
- base_done_reason, base_finalizer_used: serving-visible completion state.
- difficulty, verification_need, constraint_density, ambiguity_score, retrieval_need: task-level
  observable features used by baseline gates.

## Data generation

- Solver and intervention model: frozen Qwen3-4B through Ollama.
- Base generation limit: 4,096 tokens.
- Candidate interventions: accept, continue, critique-and-repair, and active verification in the
  training split; evaluation logs focus on accept and active verification.
- Finalizer: up to 512 non-reasoning tokens when a reasoning call did not expose a final answer.
- Language: English.

The generation and gate-training code is available at
[Sajib-006/SEVRA](https://github.com/Sajib-006/SEVRA).

## Licensing and provenance

This dataset contains derived records based on MATH, GSM8K, and CommonsenseQA. Their original
licenses and terms continue to apply to benchmark content. Qwen3 is released under Apache-2.0.
Generated model responses and SEVRA annotations are provided for research and evaluation; users
are responsible for complying with the underlying benchmark terms.

## Limitations

The records come from one frozen solver family and primarily mathematical reasoning. Correctness
labels depend on exact-answer and mathematical-equivalence evaluation. The dataset should not be
treated as a general correctness classifier corpus, and model-generated reasoning may contain
errors even when the final answer is correct.

## Citation

~~~bibtex
@article{dip2026think,
  title={Think Again or Think Longer? Selective Verification for Budget-Aware Reasoning},
  author={Dip, Sajib Acharjee and Zhou, Dawei and Zhang, Liqing},
  journal={arXiv preprint arXiv:2606.19808},
  year={2026}
}
~~~
