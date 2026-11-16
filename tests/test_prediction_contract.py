"""Initial-input, probability, and saved-model safeguards; fixtures are synthetic."""
import copy
import unittest

import numpy as np
import pandas as pd

from rechecklab.data import LABELS, validate_pairs
from rechecklab.features import FEATURE_NAMES, INITIAL_REQUIRED, extract_features, extract_one
from rechecklab.predictor import (
    OutcomePredictor, choose_threshold, decide_recheck, fit_predictor,
    make_artifact, revision_score, validate_artifact, validate_probabilities,
)


def initial_record(**changes):
    row = dict(query="How much is 2 + 3?", base_response="Maybe 5. FINAL: 5",
               base_answer="5", base_actual_tokens=31, base_done_reason="stop",
               base_finalizer_used=False)
    row.update(changes)
    return row


def training_pairs(prefix, count=54):
    records = []
    for index in range(count):
        state = index % 3
        records.append(initial_record(
            query=f"{prefix} question {index}", example_id=str(index), dataset=prefix,
            base_response=["Maybe 5", "FINAL: 5", "Wait, check 5 again"][state],
            base_actual_tokens=20 + 100 * state, base_done_reason="length" if state == 0 else "stop",
            base_correct=state != 0, final_correct=state != 2, action_actual_tokens=10,
            model="synthetic-test-solver", recheck_protocol="synthetic-test-recheck", action="recheck",
        ))
    return validate_pairs(pd.DataFrame(records))


class FixedEstimator:
    """A controlled probability producer for testing the boundary, not a model."""
    def __init__(self, values=(.2, .5, .3), classes=("HURT", "NO_CHANGE", "HELP")):
        self.classes_ = np.asarray(classes)
        self.values = values
        self.n_features_in_ = len(FEATURE_NAMES)
        self.feature_names_in_ = np.asarray(FEATURE_NAMES)

    def predict_proba(self, features):
        return np.tile(np.asarray(self.values), (len(features), 1))


class SignalEstimator(FixedEstimator):
    def __init__(self):
        super().__init__(classes=LABELS)

    def predict_proba(self, features):
        help_probability = np.clip(features.initial_log_actual_tokens.to_numpy() / 20, .05, .6)
        hurt_probability = .05 + .1 * features.initial_finalizer_used.to_numpy()
        return np.column_stack((help_probability, 1 - help_probability - hurt_probability, hurt_probability))


class PredictionContract(unittest.TestCase):
    def test_every_initial_signal_is_required(self):
        for field in INITIAL_REQUIRED:
            row = initial_record()
            row.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                extract_features(pd.DataFrame([row]))

    def test_missing_or_nonfinite_numeric_metadata_is_not_zero_imputed(self):
        for value in (None, "", "invalid", np.nan, np.inf, -np.inf, -1, 1.5, True):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "base_actual_tokens"):
                extract_one(initial_record(base_actual_tokens=value))
        self.assertEqual(extract_one(initial_record(base_actual_tokens=0))["initial_log_actual_tokens"], 0)

    def test_completion_and_finalizer_metadata_are_explicit(self):
        for value in (None, "", " ", "stopped", "invalid"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "base_done_reason"):
                extract_one(initial_record(base_done_reason=value))
        for value in (None, "unknown", 2, np.nan):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "base_finalizer_used"):
                extract_one(initial_record(base_finalizer_used=value))
        unknown = extract_one(initial_record(base_done_reason="unknown", base_answer=""))
        self.assertEqual(unknown["initial_completion_unknown"], 1)
        self.assertEqual(unknown["initial_answer_present"], 0)
        self.assertEqual(list(unknown), FEATURE_NAMES)
        self.assertEqual(extract_one(initial_record(base_done_reason=" LENGTH "))["initial_budget_exhausted"], 1)

    def test_probability_columns_follow_declared_order(self):
        predictor = OutcomePredictor(FixedEstimator(), "boundary-test")
        np.testing.assert_allclose(predictor.probabilities(pd.DataFrame([initial_record()])), [[.3, .5, .2]])
        predictor = OutcomePredictor(FixedEstimator((.8, .2), ("NO_CHANGE", "HELP")), "boundary-test")
        np.testing.assert_allclose(predictor.probabilities(pd.DataFrame([initial_record()])), [[.2, .8, 0]])

    def test_invalid_probability_outputs_are_rejected(self):
        malformed = ([.2, .5, .3], [[.2, .8]], [[np.nan, .5, .5]], [[np.inf, 0, 0]],
                     [[-.1, .6, .5]], [[1.1, 0, -.1]], [[.2, .2, .2]])
        for values in malformed:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate_probabilities(values)
        for values in ((np.nan, .5, .5), (-.1, .6, .5), (.2, .2, .2)):
            predictor = OutcomePredictor(FixedEstimator(values), "boundary-test")
            with self.subTest(values=values), self.assertRaises(ValueError):
                predictor.probabilities(pd.DataFrame([initial_record()]))
        for classes in (("HELP", "HELP", "HURT"), ("HELP", "WRONG", "HURT"), ()):
            predictor = OutcomePredictor(FixedEstimator(classes=classes), "boundary-test")
            with self.subTest(classes=classes), self.assertRaises(ValueError):
                predictor.probabilities(pd.DataFrame([initial_record()]))
        with self.assertRaises(ValueError):
            validate_probabilities([[.2, .5, .3]], expected_rows=2)

    def test_future_labels_and_answers_leave_predictions_and_decisions_unchanged(self):
        frame = pd.DataFrame([initial_record()])
        changed = frame.assign(gold_answer="999", base_correct=False, final_correct=True,
                               label="HURT", intervention_response="New answer", final_answer="999",
                               action_actual_tokens=999999, harmful_flip=True)
        predictor = OutcomePredictor(SignalEstimator(), "boundary-test")
        original_probabilities = predictor.probabilities(frame)
        changed_probabilities = predictor.probabilities(changed)
        np.testing.assert_array_equal(original_probabilities, changed_probabilities)
        np.testing.assert_array_equal(decide_recheck(original_probabilities, .1),
                                      decide_recheck(changed_probabilities, .1))

    def test_recorded_subset_uses_the_same_features_as_evaluation(self):
        frame = training_pairs("replay", 3)
        recorded = frame.iloc[[0]]
        ui_initial_inputs = pd.DataFrame([{key: recorded.iloc[0][key] for key in INITIAL_REQUIRED}])
        pd.testing.assert_frame_equal(extract_features(recorded).reset_index(drop=True), extract_features(ui_initial_inputs))
        features = extract_features(ui_initial_inputs)
        self.assertEqual(features.initial_budget_exhausted.iloc[0], 1)
        self.assertEqual(features.initial_finalizer_used.iloc[0], 0)

    def test_one_shared_decision_uses_a_strict_threshold(self):
        probabilities = np.array([[.4, .3, .3], [.1, .7, .2]])
        scores = revision_score(probabilities)
        np.testing.assert_array_equal(decide_recheck(probabilities, scores[0]), [False, False])
        np.testing.assert_array_equal(decide_recheck(probabilities, -.05), [True, False])
        for threshold in (np.nan, np.inf, None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                decide_recheck(probabilities, threshold)
        for harm_weight in (0, -1, np.inf, np.nan):
            with self.subTest(harm_weight=harm_weight), self.assertRaises(ValueError):
                revision_score(probabilities, harm_weight)

    def test_artifact_binds_schema_class_order_and_solver_protocol(self):
        protocol = dict(model="synthetic-test-solver", recheck_protocol="synthetic-test-recheck", action="recheck")
        bundle = make_artifact(OutcomePredictor(FixedEstimator(), "boundary-test"), .1, protocol=protocol)
        self.assertIs(validate_artifact(bundle, expected_protocol=protocol), bundle)
        for key, value in (("artifact_schema_version", 1), ("feature_schema_version", "old"),
                           ("feature_schema_fingerprint", "invalid"), ("class_order", list(reversed(LABELS))),
                           ("feature_names", list(reversed(FEATURE_NAMES))), ("threshold", np.nan),
                           ("harm_weight", np.inf)):
            altered = copy.deepcopy(bundle)
            altered[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_artifact(altered)
        with self.assertRaisesRegex(ValueError, "protocol"):
            validate_artifact(bundle, expected_protocol={**protocol, "model": "different-solver"})
        with self.assertRaisesRegex(ValueError, "incompatible model"):
            bundle["predictor"].probabilities(pd.DataFrame([initial_record(model="different-solver")]))

    def test_candidate_selection_is_source_only_and_includes_balanced_baselines(self):
        predictor, rows, candidates = fit_predictor(training_pairs("train"), training_pairs("dev", 18), return_candidates=True)
        self.assertEqual(set(candidates), {"majority_class", "logistic_regression", "logistic_regression_balanced",
                                          "hist_gradient_boosting", "hist_gradient_boosting_balanced"})
        expected = max(range(len(rows)), key=lambda index: (round(rows[index]["source_dev_macro_f1"], 12),
                       -round(rows[index]["source_dev_log_loss"], 12), -index))
        self.assertEqual(predictor.model_name, rows[expected]["model"])
        self.assertEqual(sum(row["selected"] for row in rows), 1)
        self.assertTrue(all("source_dev_metrics" in row for row in rows))
        with self.assertRaisesRegex(ValueError, "disjoint"):
            fit_predictor(training_pairs("same"), training_pairs("same", 18))

    def test_threshold_selection_weights_questions_and_rejects_invalid_scores(self):
        frame = training_pairs("threshold", 3)
        scores = np.array([.2, .1, .3])
        repeated = pd.concat([frame, frame.iloc[[0, 0, 0]]], ignore_index=True)
        repeated_scores = np.r_[scores, [.2, .2, .2]]
        self.assertEqual(choose_threshold(frame, scores), choose_threshold(repeated, repeated_scores))
        for invalid in ([], [.1], [np.nan, .1, .2], [[.1], [.2], [.3]]):
            with self.subTest(scores=invalid), self.assertRaises(ValueError):
                choose_threshold(frame, invalid)


if __name__ == "__main__":
    unittest.main()
