"""Verify recorded UI inputs, artifact matching, and the main Streamlit interaction."""
import copy
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import app
from rechecklab.features import extract_features


class RecordedDemoSafeguards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stamps = tuple((app.RUN_ROOT / name).stat().st_mtime_ns for name in app.REPLAY_FILES)
        cls.metadata, _, _, _, cls.predictions, cls.bundle = app.load_replay(str(app.RUN_ROOT), stamps)

    def test_initial_answer_uses_saved_parsed_value(self):
        present = self.predictions.base_answer.astype(str).str.strip().ne("")
        record = self.predictions.loc[present].iloc[0].copy()
        expected = str(record["base_answer"]).strip()
        self.assertEqual(app.recorded_initial_answer(record), expected)
        record["base_response"] = "A different text ending with FINAL: Z"
        record["final_answer"] = "A different revised answer"
        self.assertEqual(app.recorded_initial_answer(record), expected)
        record["base_answer"] = ""
        self.assertEqual(
            app.recorded_initial_answer(record),
            "No initial answer was parsed in this record.",
        )

    def test_ui_features_match_recorded_evaluation_features(self):
        # Exercise both completion states/finalizer values where present, without
        # replacing the selected record's initial metadata with widget defaults.
        selected = self.predictions.drop_duplicates(
            subset=["base_done_reason", "base_finalizer_used"]
        )
        for _, record in selected.iterrows():
            with self.subTest(record=record.example_id):
                ui_frame = app.initial_frame(record)
                self.assertEqual(ui_frame.iloc[0].base_actual_tokens, record.base_actual_tokens)
                self.assertEqual(ui_frame.iloc[0].base_done_reason, record.base_done_reason)
                self.assertEqual(ui_frame.iloc[0].base_finalizer_used, record.base_finalizer_used)
                pd.testing.assert_frame_equal(
                    extract_features(ui_frame),
                    extract_features(pd.DataFrame([record.to_dict()])),
                )
                probabilities, recheck = app.predict_initial(self.bundle, record)
                np.testing.assert_allclose(
                    probabilities,
                    [record["p_" + label] for label in app.LABELS],
                    rtol=1e-10,
                )
                self.assertEqual(recheck, bool(record.source_gate_rechecks))

    def test_future_answer_and_label_changes_cannot_change_ui_prediction(self):
        record = self.predictions.iloc[0].copy()
        probabilities, decision = app.predict_initial(self.bundle, record)
        record["base_correct"] = not bool(record["base_correct"])
        record["final_correct"] = not bool(record["final_correct"])
        record["label"] = "HURT"
        record["final_answer"] = "future answer changed"
        record["intervention_response"] = "future response changed " * 100
        record["action_actual_tokens"] = 999999
        changed_probabilities, changed_decision = app.predict_initial(self.bundle, record)
        np.testing.assert_array_equal(probabilities, changed_probabilities)
        self.assertEqual(decision, changed_decision)
        self.assertEqual(
            app.input_key(app.initial_frame(self.predictions.iloc[0]), "same-record"),
            app.input_key(app.initial_frame(record), "same-record"),
        )
        self.assertNotEqual(
            app.input_key(app.initial_frame(record), "same-record", "previous-model-hash"),
            app.input_key(app.initial_frame(record), "same-record", "new-model-hash"),
        )

    def test_missing_initial_metadata_is_rejected(self):
        for key in ("base_actual_tokens", "base_done_reason", "base_finalizer_used"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, "Missing initial signals"):
                    app.initial_frame(self.predictions.iloc[0].drop(key))

    def test_mismatched_results_are_rejected(self):
        changed_hashes = copy.deepcopy(self.metadata)
        changed_hashes["artifact_sha256"]["reports/policy_results.csv"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "artifact fingerprint"):
            app.validate_file_hashes(app.RUN_ROOT, changed_hashes)
        changed = copy.deepcopy(self.metadata)
        changed["source_threshold"] += 0.01
        with self.assertRaisesRegex(ValueError, "source_threshold"):
            app.validate_replay(changed, self.bundle, self.predictions)
        changed = copy.deepcopy(self.metadata)
        changed["demo_adaptation"]["threshold"] += 0.01
        with self.assertRaisesRegex(ValueError, "adapted gate"):
            app.validate_replay(changed, self.bundle, self.predictions)
        changed = copy.deepcopy(self.metadata)
        changed["class_order"] = list(reversed(app.LABELS))
        with self.assertRaisesRegex(ValueError, "class_order"):
            app.validate_replay(changed, self.bundle, self.predictions)
        mismatched = self.predictions.copy()
        mismatched.loc[0, "p_HELP"] = np.nan
        with self.assertRaises(ValueError):
            app.validate_replay(self.metadata, self.bundle, mismatched)

    def test_recorded_solver_must_match_the_saved_protocol(self):
        for key, changed, expected in (
            ("model", "different solver", "incompatible model"),
            ("action", "different action", "incompatible action"),
            ("action_description", "different specified recheck", "incompatible action_description_sha256"),
        ):
            with self.subTest(key=key):
                record = self.predictions.iloc[0].copy()
                record[key] = changed
                with self.assertRaisesRegex(ValueError, expected):
                    app.predict_initial(self.bundle, record)


class StreamlitInteractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        stamps = tuple((app.RUN_ROOT / name).stat().st_mtime_ns for name in app.REPLAY_FILES)
        cls.metadata, _, _, _, cls.predictions, cls.bundle = app.load_replay(str(app.RUN_ROOT), stamps)

    def open_app(self):
        ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=30).run()
        self.assertEqual(len(ui.exception), 0, str(ui.exception))
        self.assertEqual(len(ui.error), 0, str(ui.error))
        return ui

    def assert_initial_answer_matches_record(self, ui, selected):
        record = self.predictions.iloc[selected]
        displayed = [entry.value for entry in ui.tabs[0].markdown]
        displayed.extend(entry.value for entry in ui.tabs[0].code)
        self.assertIn(app.recorded_initial_answer(record), displayed)
        self.assertIn(record["query"], displayed)
        initial_expander = next(
            expander for expander in ui.tabs[0].expander
            if expander.label == "Full initial response"
        )
        full_response = [entry.value for entry in initial_expander.markdown]
        full_response.extend(entry.value for entry in initial_expander.code)
        self.assertIn(record["base_response"], full_response)

    def assert_prediction_matches_evaluation(self, ui, selected):
        record = self.predictions.iloc[selected]
        self.assertEqual(len(ui.exception), 0, str(ui.exception))
        self.assertEqual(len(ui.error), 0, str(ui.error))
        self.assertEqual([metric.label for metric in ui.metric], list(app.DISPLAY_OUTCOMES))
        for metric in ui.metric:
            self.assertEqual(metric.value, f"{float(record['p_' + metric.label]):.1%}")
        decisions = [entry.value for entry in ui.markdown if entry.value.startswith("**Decision:")]
        decision = "Recheck" if bool(record.source_gate_rechecks) else "Skip recheck"
        self.assertEqual(decisions, [f"**Decision: {decision}**"])
        self.assertIn("What actually happened?", [expander.label for expander in ui.expander])

    def test_load_and_main_prediction_interaction(self):
        ui = self.open_app()
        self.assertEqual([tab.label for tab in ui.tabs], ["Demo", "Results"])
        self.assertEqual(len(ui.selectbox), 1)
        self.assertEqual(len(ui.radio), 1)
        self.assertEqual(ui.radio[0].value, "Recorded examples")
        self.assertEqual(list(ui.radio[0].options), ["Recorded examples", "Live question"])
        for widget in ("text_area", "text_input", "number_input", "checkbox"):
            with self.subTest(widget=widget):
                self.assertEqual(len(getattr(ui, widget)), 0)
        options = ui.selectbox[0].options
        self.assertTrue(any("Attempt 1" in option for option in options))
        self.assertTrue(any("Attempt 2" in option for option in options))
        self.assertFalse(any("rollout" in option.lower() for option in options))
        self.assertEqual([button.label for button in ui.button], ["Should I recheck?"])
        self.assert_initial_answer_matches_record(ui, 0)
        self.assertNotIn("What actually happened?", [expander.label for expander in ui.expander])
        self.assertEqual(len(ui.metric), 0)
        results = ui.tabs[1]
        self.assertEqual(len(results.expander), 1)
        self.assertFalse(results.expander[0].proto.expanded)
        self.assertEqual(len(results.dataframe[0].value), 5)
        result_captions = " ".join(caption.value.lower() for caption in results.caption)
        self.assertIn("200", result_captions)
        self.assertIn("subset 1", result_captions)
        self.assertIn("mean", result_captions)
        self.assertTrue("five" in result_captions or "5" in result_captions)
        self.assertTrue("sd" in result_captions or "standard deviation" in result_captions)
        ui.button[0].click().run()
        self.assert_prediction_matches_evaluation(ui, 0)
        # Selecting another recorded rollout cannot reveal its future result
        # until the user requests a prediction for that selected record.
        ui.selectbox[0].set_value(1).run()
        self.assertEqual(len(ui.exception), 0, str(ui.exception))
        self.assert_initial_answer_matches_record(ui, 1)
        self.assertNotIn("What actually happened?", [expander.label for expander in ui.expander])
        self.assertEqual(len(ui.metric), 0)

    def test_selected_recheck_example_matches_evaluation(self):
        recheck_index = int(self.predictions.index[self.predictions.source_gate_rechecks][0])
        skip_index = int(self.predictions.index[~self.predictions.source_gate_rechecks][0])
        ui = self.open_app()
        ui.selectbox[0].set_value(recheck_index).run()
        self.assert_initial_answer_matches_record(ui, recheck_index)
        self.assertEqual(len(ui.metric), 0)
        self.assertNotIn("What actually happened?", [expander.label for expander in ui.expander])
        ui.button[0].click().run()
        self.assert_prediction_matches_evaluation(ui, recheck_index)
        ui.selectbox[0].set_value(skip_index).run()
        self.assert_initial_answer_matches_record(ui, skip_index)
        self.assertEqual(len(ui.metric), 0)
        self.assertNotIn("What actually happened?", [expander.label for expander in ui.expander])
        ui.button[0].click().run()
        self.assert_prediction_matches_evaluation(ui, skip_index)


if __name__ == "__main__":
    unittest.main()
