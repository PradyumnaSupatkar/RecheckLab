"""Boundary and question-equal metric checks; no research data are synthesized."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rechecklab.experiment import PROJECT_ROOT, run_experiment
from rechecklab.metrics import policy_metrics, predictor_metrics
from test_research_safeguards import pairs


class EvaluationContract(unittest.TestCase):
    def test_partial_custom_data_never_load_public_defaults_or_write_output(self):
        frame = pairs(count=10)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with patch("rechecklab.experiment.public_frames") as load:
                for source, target in ((frame, None), (None, frame)):
                    with self.assertRaisesRegex(ValueError, "both"):
                        run_experiment(output, source, target)
                load.assert_not_called()
            self.assertFalse(output.exists())

    def test_incompatible_custom_protocol_is_rejected_before_output(self):
        source, target = pairs("source"), pairs("target")
        target["recheck_protocol"] = "another-step"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "unused"
            with self.assertRaisesRegex(ValueError, "compatibility mismatch"):
                run_experiment(output, source, target, sizes=(10,))
            self.assertFalse(output.exists())

    def test_original_baseline_cannot_be_an_output_directory(self):
        with self.assertRaisesRegex(ValueError, "baseline"):
            run_experiment(PROJECT_ROOT)

    def test_predictor_metrics_and_intervals_weight_questions_consistently(self):
        # q1 has two outcomes; q2 has one. Replicating q1's entire distribution
        # must change neither confusion mass, log loss nor question bootstrap.
        frame = pd.DataFrame({"group_id": ["q1", "q1", "q2"],
                              "label": ["HELP", "NO_CHANGE", "HURT"]})
        probabilities = np.array([[.7, .2, .1], [.2, .7, .1], [.1, .6, .3]])
        metrics = predictor_metrics(frame, probabilities, bootstrap_repeats=200)
        duplicated = pd.concat([frame.iloc[:2]] * 7 + [frame.iloc[2:]], ignore_index=True)
        duplicated_probabilities = np.vstack([probabilities[:2]] * 7 + [probabilities[2:]])
        repeated = predictor_metrics(duplicated, duplicated_probabilities, bootstrap_repeats=200)
        for key in ("macro_f1", "log_loss", "confusion_matrix", "macro_f1_95pct_ci", "log_loss_95pct_ci"):
            np.testing.assert_allclose(metrics[key], repeated[key], atol=1e-12)
        self.assertAlmostEqual(np.asarray(metrics["confusion_matrix"]).sum(), 2)
        self.assertEqual(metrics["class_counts"], {"HELP": 1, "NO_CHANGE": 1, "HURT": 1})

    def test_policy_decisions_reject_wrong_shape_and_truthy_numeric_values(self):
        frame = pairs(count=6)
        for decisions in ([1] * 6, [True] * 5, ["False"] * 6, [[True]] * 6):
            with self.subTest(decisions=decisions), self.assertRaisesRegex(ValueError, "boolean"):
                policy_metrics(frame, decisions)


if __name__ == "__main__":
    unittest.main()
