# Frozen v0.2 gate: OOD feature compatibility

**Status:** canonical feature-level compatibility passed; protocol compatibility is **cross-model/cross-protocol OOD only**.

- Input rows: 40/40.
- Required initial-only fields available and validated for every row: `query, base_response, base_answer, base_actual_tokens, base_done_reason, base_finalizer_used`.
- Feature schema: `pre-recheck-v2`, 19 features, fingerprint `9e09eba8571dd2b31fb70fe83cb12191ba0051d620324037cc18cf3cae22e4ca`.
- Exact original feature order: `question_log_chars, question_log_words, question_numbers, question_math_symbols, question_marks, initial_log_chars, initial_log_words, initial_unique_word_ratio, initial_uncertainty_per_100_words, initial_backtracking_per_100_words, initial_checking_per_100_words, initial_number_count, initial_paragraph_count, initial_question_marks, initial_log_actual_tokens, initial_finalizer_used, initial_budget_exhausted, initial_answer_present, initial_completion_unknown`.
- Original class order: `HELP, NO_CHANGE, HURT`.
- Frozen artifact SHA-256: `9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439`; threshold `0.25968145276947807`; score `p_HELP - 1.0 * p_HURT`; recheck iff score is strictly greater than threshold.

The scoring path loaded the unchanged artifact and called the repository's canonical `validate_initial_record`, `extract_features`, and original predictor probability method. The predictor input frame was explicitly projected to the six initial-only allowlist fields. It did not include revised text, trusted keys, correctness, transitions, or any other outcome. The model is Qwen3-4B-Instruct-2507, whereas the artifact was trained for the historical Qwen3-4B SEVRA active-verify protocol; therefore scores/actions are an OOD cross-model and cross-protocol research diagnostic, not protocol-validated deployment.
