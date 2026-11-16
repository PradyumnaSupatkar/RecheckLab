# Frozen gate OOD failure analysis

- **Incomplete initial answers:** 5/8 stopped at the length limit. The gate selected 0/5 of these for rechecking.
- **Helpful revisions:** the stored pairs contain 4 HELP transitions. The gate selected 0 and missed 4. Missed HELP source IDs: ['876', '5093', '551', '1952'].
- **Harmful revisions:** 0 HURT transitions occurred; the gate accepted 0. Zero observed HURT events does not establish harm safety.
- **Gate vs completion heuristic:** 5/8 action disagreements. IDs: ['1580', '876', '5093', '1952', '4130']. Completion rechecked 5/8; gate rechecked 0/8.
- **Feature distribution:** compared the eight live feature vectors with the frozen v0.2 source-training feature distribution (univariate, descriptive). Mean absolute standardized shift across nonconstant features: 3.19. Largest absolute standardized mean shifts: [('initial_answer_present', -25.95), ('initial_question_marks', 13.2), ('initial_uncertainty_per_100_words', 5.98), ('initial_backtracking_per_100_words', 3.13), ('initial_log_words', 1.37)]. Features with at least one live value outside the source-training min/max: 5/19. See `FEATURE_SHIFT.csv`.

The distribution differences coincide with prompt, output-budget, thinking, and finalizer-protocol differences; this eight-question analysis cannot attribute the shifts to any one change. It does not support significance, generalization, or safety conclusions. Any apparent gate/completion disagreements or feature shifts should motivate a separately frozen prospective test, not threshold changes or post-hoc calibration. Concrete prospective checks include protocol-matched gate validation, representation of both length/stop completions, and monitoring whether completion/finalizer/token features remain within source support.
