# Experiment limitations

This is an exploratory early-stopped evaluation, not the planned completed 60-question prospective study. The primary policy analysis includes only the 43 questions with both initial and revision calls naturally completed and checkpointed.

The fixed question selection was frozen before inference and no question was replaced. However, stopping at this point leaves later questions unobserved, and incomplete calls were concentrated among questions with unusually long generations or external stop events. Runtime-related missingness may correlate with problem difficulty, response verbosity, context use, or answer validity. The completed-pair subset is therefore not guaranteed to be representative, and no generalization claim is warranted.

The v0.2 classifier is an out-of-distribution diagnostic under a cross-model/cross-protocol evaluation. Its predictions were made from initial-only features before revision; this does not establish deployment compatibility. The small sample yields wide uncertainty. No statistical equivalence or significance claim is made.

The original 60-question selection and trusted keys remain unchanged. The runtime-only timeout amendment is recorded in the evaluation directory; it does not rewrite the original experiment manifest or retroactively change earlier call settings. No additional inference, classifier fitting, threshold adjustment, or outcome-based question selection was performed during report generation.
