"""Consistency checks for the included v0.2 recorded policy results."""
import unittest

from scripts.verify_recorded_policies import verify_recorded_policies

class RecordedPolicyResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.verification = verify_recorded_policies()

    def test_default_gate_masks_and_score_boundary_explain_equal_policies(self):
        masks = self.verification["decision_vectors"]
        self.assertEqual(masks["completion_vs_frozen_mismatching_rows"], 0)
        self.assertEqual(masks["frozen_vs_selected_adapted_mismatching_rows"], 0)
        self.assertTrue(self.verification["score_boundary"]["frozen_and_adapted_inside_same_empty_gap"])

    def test_all_saved_policy_and_adaptation_metrics_replay_from_correctness(self):
        checks = self.verification["checks"]
        self.assertTrue(checks["all_target_policy_metrics_match"], checks["target_policy_metric_checks"])
        self.assertEqual(checks["adaptation_metric_mismatches"], [])
        self.assertTrue(checks["adaptation_summary_matches_all_runs"])

    def test_model_recorded_outputs_and_selected_cutoff_agree(self):
        checks = self.verification["checks"]
        for key in ("canonical_groups_match_questions", "saved_probabilities_match_model", "saved_scores_match_model",
                    "saved_frozen_decisions_match", "selected_threshold_matches_bundle_and_run",
                    "evaluation_features_match_prepared_initial_record"):
            with self.subTest(key=key):
                self.assertTrue(checks[key])
        self.assertTrue(all(self.verification["answer_fidelity"]["exact_initial_fields_match_prepared_vs_csv"].values()))



if __name__ == "__main__":
    unittest.main()
