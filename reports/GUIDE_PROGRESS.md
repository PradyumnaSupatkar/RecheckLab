# RecheckLab: progress milestone 01

Date: 4 October 2026

Working prototype: a lightweight HELP / NO CHANGE / HURT predictor, frozen transfer, limited-label threshold adaptation, an offline report, and a Streamlit dashboard.

Data: 1,999 independent source mathematics questions and 1,221 target commonsense questions. Real public Qwen3-4B outcome logs from SEVRA; secondary analysis, not newly generated runs.

First result: The frozen source gate reached 76.67% target accuracy versus 76.13% when keeping the first answer and 72.85% when always rechecking. Its harmful-flip rate was 0.00% of target questions. With 200 target questions for threshold adaptation, mean test accuracy was 76.08% (SD 0.38 percentage points across 5 subsets). The completion rule reached 76.67%. Compare the learned gate with this simple rule before attributing improvements to machine learning. None of the tested adaptation budgets improved mean accuracy over the frozen source gate in this pilot.

Chosen source model: hist_gradient_boosting. The source threshold is selected on development examples. Adaptation uses a disjoint target pool, while test questions remain fixed. All repeated rollouts and duplicate questions are grouped.

The adaptation implemented today adjusts only the rechecking decision threshold. It does not retrain the LLM or calibrate all predicted probabilities.

Next: collect our own GSM8K and BoolQ pairs using the included scripts; test more models and splits; add probability calibration and feature ablations.

Sources:
- https://huggingface.co/datasets/Sajib-006/sevra-recovery-outcomes
- https://arxiv.org/abs/2606.19808
- https://arxiv.org/abs/2609.35832
