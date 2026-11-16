# RecheckLab v0.2 progress

Date: 4 October 2026

The offline demo predicts the outcome of one specified recheck and separately chooses whether to spend the extra call. It replays published responses; no new LLM inference was executed. Skip recheck keeps an answer that may still be wrong.

1,999 source maths questions and 1,221 target CommonsenseQA questions. Source-only model selection chose hist_gradient_boosting_balanced. Majority, unbalanced and balanced logistic/boosting models were compared. Metrics and uncertainty give each question equal weight across its rollouts.

Frozen gate target accuracy: 76.67%; completion rule: 76.67%; keep initial: 76.13%. Target outcome macro-F1: 0.394 (majority baseline 0.323); multiclass log loss: 0.322. No tested label budget improved mean accuracy over the frozen gate. The learned gate does not beat the completion rule in accuracy.

Adaptation changes only the decision threshold using the reserved target pool. It does not calibrate probabilities. Target test labels are excluded from fitting and setting thresholds.

The two-tab Streamlit app uses actual recorded initial tokens/completion/finalizer values, validates matching model and results, and reveals the revision/outcome after prediction. Raw data and v0.1 results remain unchanged. See the root GUIDE_PROGRESS.md for completed verification and audit details.

Limits: one solver, one transfer direction and split; rare harmful source outcomes; inherited labels; incomplete public generation metadata; uncalibrated probabilities; token savings are replay estimates. No established advantage over the completion rule or reliable adaptation gain is assumed.
