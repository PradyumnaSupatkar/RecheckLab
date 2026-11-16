"""Mocked UI integration checks; these fixtures execute no real LLM calls."""
import copy
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import app
from rechecklab import live


QUESTION = "What is two plus two?"


def backend_status(ready=True, experimental=False):
    settings = dict(live.DEFAULT_SETTINGS, initial_seed=42, recheck_seed=43,
                    finalizer_used=False, recheck_protocol="mock-ui-protocol")
    return dict(
        ready=ready, status="experimental" if experimental else "compatible",
        message="Mock backend is ready with the saved solver.", model="qwen3:4b",
        model_digest="mock-ui-checkpoint", experimental=experimental,
        protocol_match=not experimental, generation_settings=settings,
        mismatches=["Mock same-solver protocol differs from replay."] if experimental else [],
        protocol_fingerprint="mock-ui-protocol-fingerprint", installed_models=["qwen3:4b"],
        live_protocol=dict(model="qwen3:4b", model_digest="mock-ui-checkpoint",
                           action="recheck", recheck_protocol="mock-ui-protocol"),
    )


def call_output(answer, response):
    return dict(answer=answer, response=response, tokens=31, done_reason="stop",
                finalizer_used=False, model="qwen3:4b", parse_valid=bool(answer),
                wall_seconds=0.01)


def live_result(recheck=False):
    initial = call_output("3", "Synthetic initial reasoning. FINAL: 3")
    revised = call_output("4", "Synthetic recorded live revision. FINAL: 4") if recheck else None
    selected = revised if recheck else initial
    return dict(
        status="complete", initial=initial, revised=revised,
        probabilities=[0.80, 0.15, 0.05] if recheck else [0.02, 0.90, 0.08],
        decision="Recheck" if recheck else "Skip recheck",
        recheck_attempted=recheck, rechecked=recheck,
        selected_answer=selected["answer"], selected_response=selected["response"],
        selected_source="revised" if recheck else "initial", error=None,
        backend=backend_status(), live_protocol=backend_status()["live_protocol"],
        initial_record=dict(query=QUESTION, base_response=initial["response"],
                            base_answer=initial["answer"], base_actual_tokens=initial["tokens"],
                            base_done_reason=initial["done_reason"], base_finalizer_used=False),
        actual_outcome="Unknown", correctness="Unknown", recheck_tokens=31 if recheck else 0,
        total_tokens=62 if recheck else 31,
    )


class LiveUIInteractions(unittest.TestCase):
    def setUp(self):
        probe_patch = patch("rechecklab.live.probe_backend", return_value=backend_status())
        runner_patch = patch("rechecklab.live.run_live_question")
        self.probe = probe_patch.start()
        self.runner = runner_patch.start()
        self.addCleanup(probe_patch.stop)
        self.addCleanup(runner_patch.stop)
        self.progress_events = []
        self.set_result(live_result())

    def set_result(self, result):
        def fake_run(question, bundle, **kwargs):
            self.assertTrue(question.strip())
            self.assertNotIn("gold_answer", kwargs)
            self.assertNotIn("base_correct", kwargs)
            callback = kwargs.get("on_progress")
            self.assertTrue(callable(callback), "Live execution must provide its progress callback.")
            stages = ["checking_backend", "generating_initial"]
            if result.get("initial"):
                stages.append("predicting")
            if result.get("recheck_attempted"):
                stages.append("rechecking")
            stages.append("complete" if result["status"] == "complete" else "failed")
            for stage in stages:
                self.progress_events.append(stage)
                callback(stage, "Mock stage: " + stage)
            returned = copy.deepcopy(result)
            returned["backend"] = copy.deepcopy(self.probe.return_value)
            return returned

        self.runner.side_effect = fake_run

    def open_app(self):
        ui = AppTest.from_file(str(app.ROOT / "app.py"), default_timeout=30).run()
        self.assert_no_exceptions(ui)
        return ui

    def open_live(self):
        ui = self.open_app()
        self.assertEqual(ui.radio[0].value, "Recorded examples")
        self.probe.assert_not_called()  # The included offline replay does not contact Ollama.
        self.runner.assert_not_called()
        ui.radio[0].set_value("Live question").run()
        self.assert_no_exceptions(ui)
        self.assertEqual([tab.label for tab in ui.tabs], ["Demo", "Results"])
        self.assertEqual(len(ui.selectbox), 0)
        self.assertEqual(len(ui.text_area), 1)
        self.assertEqual(ui.text_area[0].label, "Question")
        self.assertEqual(len(ui.text_input), 0)
        self.assertEqual(len(ui.number_input), 0)
        self.assertEqual(len(ui.checkbox), 0)
        self.assertEqual([button.label for button in ui.button], ["Should I recheck?"])
        return ui

    def assert_no_exceptions(self, ui):
        self.assertEqual(len(ui.exception), 0, str(ui.exception))

    def visible_text(self, ui):
        return "\n".join(
            entry.value
            for kind in ("markdown", "caption", "warning", "error", "info", "code")
            for entry in getattr(ui, kind)
        )

    def assert_result(self, ui, result):
        self.assert_no_exceptions(ui)
        self.assertEqual([metric.label for metric in ui.metric], list(app.DISPLAY_OUTCOMES))
        for metric in ui.metric:
            probability = result["probabilities"][app.LABELS.index(metric.label)]
            self.assertEqual(metric.value, f"{probability:.1%}")
        self.assertIn("**Decision: " + result["decision"] + "**", self.visible_text(ui))
        saved = ui.session_state["live_result"]["result"]
        self.assertEqual(saved["selected_answer"], result["selected_answer"])
        self.assertEqual(saved["selected_source"], result["selected_source"])
        rendered = [entry.value for entry in ui.tabs[0].markdown]
        selected_index = rendered.index("**Selected final answer**") + 1
        self.assertEqual(rendered[selected_index], result["selected_answer"])
        self.assertNotIn("What actually happened?", [expander.label for expander in ui.expander])
        self.assertIn("unknown", self.visible_text(ui).lower())

    def test_unavailable_backend_leaves_recorded_mode_offline_and_blocks_live(self):
        unavailable = backend_status(ready=False)
        unavailable.update(status="unavailable", message="Mock Ollama backend is unavailable.")
        self.probe.return_value = unavailable
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        self.assertTrue(ui.button[0].disabled)
        self.assertIn(unavailable["message"], self.visible_text(ui))
        ui.run()
        self.runner.assert_not_called()
        self.assertEqual(len(ui.metric), 0)

    def test_predictor_failure_is_reported_separately_from_backend_connection(self):
        unavailable = backend_status(ready=False)
        unavailable.update(status="predictor_load_error", error_stage="predictor",
                           message="Saved predictor validation failed: The saved gate must contain an OutcomePredictor.")
        self.probe.return_value = unavailable
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        self.assert_no_exceptions(ui)
        self.assertTrue(ui.button[0].disabled)
        text = self.visible_text(ui.tabs[0])
        self.assertIn("Saved predictor: unavailable", text)
        self.assertEqual([entry.value for entry in ui.tabs[0].error], [unavailable["message"]])
        self.assertNotIn("Live backend:", text)
        self.assertNotIn("Start Ollama", text)
        self.assertNotIn("ollama pull", text)
        self.assertEqual(len(ui.metric), 0)
        ui.run()
        self.runner.assert_not_called()
        ui.radio[0].set_value("Recorded examples").run()
        self.assert_no_exceptions(ui)
        self.assertEqual(len(ui.selectbox), 1)
        self.assertEqual(len(ui.tabs[0].error), 0)
        self.runner.assert_not_called()

    def test_different_solver_is_blocked_without_calling_engine(self):
        incompatible = backend_status(ready=False)
        incompatible.update(status="model_missing", model="qwen3:4b",
                            installed_models=["qwen3:1.7b"],
                            message="The saved gate requires qwen3:4b; a different solver is blocked.")
        self.probe.return_value = incompatible
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        self.assertTrue(ui.button[0].disabled)
        self.assertIn("different solver", self.visible_text(ui).lower())
        self.runner.assert_not_called()

    def test_skip_retains_initial_answer_and_same_input_never_repeats_calls(self):
        expected = live_result()
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        self.runner.assert_not_called()
        ui.button[0].click().run()
        self.assertEqual(self.runner.call_count, 1)
        self.assert_result(ui, expected)
        self.assertNotIn("rechecking", self.progress_events)
        ui.run()
        self.assert_result(ui, expected)
        ui.button[0].click().run()
        self.assert_result(ui, expected)
        self.assertEqual(self.runner.call_count, 1)

    def test_experimental_recheck_uses_generated_revision_and_progress_callback(self):
        experimental = backend_status(experimental=True)
        experimental["message"] = "Mock backend ready: experimental same-solver protocol, not evaluated replay."
        self.probe.return_value = experimental
        expected = live_result(recheck=True)
        self.set_result(expected)
        ui = self.open_live()
        self.assertIn("experimental", self.visible_text(ui).lower())
        ui.text_area[0].input(QUESTION).run()
        ui.button[0].click().run()
        self.assert_result(ui, expected)
        self.assertTrue(self.runner.call_args.kwargs["allow_experimental"])
        self.assertEqual(self.runner.call_args.kwargs["answer_type"], "open")
        self.assertIn("rechecking", self.progress_events)
        self.assertEqual(self.progress_events[-1], "complete")

    def test_question_and_mode_changes_clear_visible_live_results_without_calls(self):
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        ui.button[0].click().run()
        self.assert_result(ui, live_result())
        ui.text_area[0].input("What is three plus three?").run()
        self.assertEqual(len(ui.metric), 0)
        self.assertEqual(self.runner.call_count, 1)
        self.assertNotIn("live_result", ui.session_state)
        self.assertNotIn("live_error", ui.session_state)
        ui.button[0].click().run()
        self.assertEqual(self.runner.call_count, 2)
        ui.radio[0].set_value("Recorded examples").run()
        self.assertNotIn("live_result", ui.session_state)
        self.assertNotIn("live_error", ui.session_state)
        self.assertEqual(len(ui.metric), 0)
        ui.radio[0].set_value("Live question").run()
        self.assertEqual(len(ui.metric), 0)
        self.assertEqual(self.runner.call_count, 2)
        self.assertNotIn("live_result", ui.session_state)

    def test_busy_request_is_reported_without_starting_another_engine(self):
        with patch("rechecklab.live_session.execute_once", return_value={"status": "busy"}):
            ui = self.open_live()
            ui.text_area[0].input(QUESTION).run()
            ui.button[0].click().run()
            self.assert_no_exceptions(ui)
            self.assertIn("in progress", self.visible_text(ui).lower())
            self.runner.assert_not_called()
            self.assertEqual(len(ui.metric), 0)

    def test_failed_recheck_preserves_initial_answer_and_caches_failure(self):
        failed = live_result(recheck=True)
        failed.update(status="recheck_failed", revised=None, rechecked=False,
                      selected_answer=failed["initial"]["answer"],
                      selected_response=failed["initial"]["response"],
                      selected_source="initial_fallback", recheck_tokens=None, total_tokens=None,
                      error="Mock recheck failed; the initial answer is retained.")
        self.set_result(failed)
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        ui.button[0].click().run()
        self.assert_result(ui, failed)
        self.assertIn(failed["error"], self.visible_text(ui))
        self.assertNotIn("4", [entry.value for entry in ui.tabs[0].markdown])
        ui.run()
        ui.button[0].click().run()
        self.assertEqual(self.runner.call_count, 1)
        self.assertIn(failed["error"], self.visible_text(ui))

    def test_prediction_failure_shows_partial_initial_response_without_inventing_answer(self):
        failed = live_result()
        partial = "Mock incomplete reasoning with no final answer marker."
        failed["initial"].update(answer="", response=partial, parse_valid=False, done_reason="length")
        failed.update(status="prediction_failed", probabilities=None, decision=None,
                      selected_answer="", selected_response=partial, selected_source="initial",
                      error="Mock prediction failed; the partial initial response is retained.")
        self.set_result(failed)
        ui = self.open_live()
        ui.text_area[0].input(QUESTION).run()
        ui.button[0].click().run()
        self.assert_no_exceptions(ui)
        self.assertIn(failed["error"], self.visible_text(ui))
        self.assertIn(partial, self.visible_text(ui))
        self.assertEqual(len(ui.metric), 0)
        self.assertEqual(ui.session_state["live_result"]["result"]["selected_answer"], "")
        self.assertNotIn("4", [entry.value for entry in ui.tabs[0].markdown])
        ui.text_area[0].input("A new live question").run()
        self.assertNotIn(failed["error"], self.visible_text(ui))
        self.assertNotIn(partial, self.visible_text(ui))
        self.assertEqual(self.runner.call_count, 1)

    def test_unvalidated_initial_failure_preserves_evidence_without_scores_or_correctness(self):
        partial = "Mock returned reasoning with unsupported completion metadata."
        evidence = dict(response=partial, validated=False,
                        reported_metadata=dict(done_reason="'<missing>'", eval_count="31"))
        failed = live_result()
        failed.update(status="initial_failed", initial=None, probabilities=None, decision=None,
                      selected_answer=None, selected_response=None, selected_source=None,
                      initial_attempt=evidence, error="Initial generation failed: missing actual completion status.")
        self.set_result(failed)
        with patch("scripts.check_live_backend.score_after_execution") as scorer:
            ui = self.open_live()
            ui.text_area[0].input(QUESTION).run()
            ui.button[0].click().run()
            self.assert_no_exceptions(ui)
            self.assertIn(failed["error"], self.visible_text(ui.tabs[0]))
            initial_expander = next(
                item for item in ui.tabs[0].expander if item.label == "Unvalidated initial response"
            )
            self.assertFalse(initial_expander.proto.expanded)
            self.assertEqual(len(initial_expander.json), 1)
            self.assertIn(partial, str(initial_expander.json[0].value))
            self.assertEqual(ui.session_state["live_result"]["result"]["initial_attempt"], evidence)
            self.assertEqual(len(ui.metric), 0)
            self.assertNotIn("**Decision:", self.visible_text(ui.tabs[0]))
            self.assertNotIn("**Selected final answer**", self.visible_text(ui.tabs[0]))
            scorer.assert_not_called()
            ui.run()
            ui.button[0].click().run()
            self.assertEqual(self.runner.call_count, 1)
            scorer.assert_not_called()

    def test_supplied_key_is_used_after_execution_without_fabricating_skipped_outcome(self):
        from scripts.check_live_backend import load_questions, load_trusted_keys, score_after_execution

        questions = load_questions(app.ROOT / "data" / "live_demo_questions.jsonl")
        keys = load_trusted_keys(app.ROOT / "data" / "live_demo_keys.jsonl", questions)
        supplied = questions[0]
        correct = keys[supplied["example_id"]]["normalized_answer"]
        for answer, expected_correct in ((correct, True), ("999999", False)):
            with self.subTest(initial_correct=expected_correct):
                self.probe.reset_mock()
                self.runner.reset_mock()
                expected = live_result()
                expected["initial"].update(answer=answer, response="Mock initial. FINAL: " + answer)
                expected.update(selected_answer=answer, selected_response=expected["initial"]["response"])
                self.set_result(expected)
                with patch("scripts.check_live_backend.score_after_execution", wraps=score_after_execution) as scorer:
                    ui = self.open_live()
                    ui.text_area[0].input(supplied["query"]).run()
                    scorer.assert_not_called()
                    self.runner.assert_not_called()
                    ui.button[0].click().run()
                    self.assert_no_exceptions(ui)
                    self.assertEqual(self.runner.call_count, 1)
                    self.assertTrue(scorer.called)
                    self.assertEqual(self.runner.call_args.args[0], supplied["query"])
                    for forbidden in ("gold_answer", "trusted_key", "base_correct", "final_correct", "label"):
                        self.assertNotIn(forbidden, self.runner.call_args.kwargs)
                    text = self.visible_text(ui.tabs[0])
                    self.assertIn(f"Separate supplied key: initial correct = {expected_correct}.", text)
                    self.assertIn("unobserved", text)
                    self.assertNotIn("Actual outcome: NO_CHANGE", text)
                    self.assertIn("**Decision: Skip recheck**", text)

    def test_supplied_key_reports_help_only_after_completed_recheck(self):
        from scripts.check_live_backend import load_questions, load_trusted_keys

        questions = load_questions(app.ROOT / "data" / "live_demo_questions.jsonl")
        keys = load_trusted_keys(app.ROOT / "data" / "live_demo_keys.jsonl", questions)
        supplied = questions[0]
        expected = live_result(recheck=True)
        correct = keys[supplied["example_id"]]["normalized_answer"]
        expected["initial"].update(answer="999999", response="Mock initial. FINAL: 999999")
        expected["revised"].update(answer=correct, response="Mock revision. FINAL: " + correct)
        expected.update(selected_answer=correct, selected_response=expected["revised"]["response"])
        self.set_result(expected)
        ui = self.open_live()
        ui.text_area[0].input(supplied["query"]).run()
        self.assertNotIn("Actual outcome:", self.visible_text(ui.tabs[0]))
        ui.button[0].click().run()
        self.assert_no_exceptions(ui)
        self.assertIn("Actual outcome: HELP", self.visible_text(ui.tabs[0]))
        self.assertEqual(self.runner.call_count, 1)
        self.assertNotIn("gold_answer", self.runner.call_args.kwargs)

    def test_truncated_skipped_response_keeps_scores_and_reports_extraction_failure(self):
        from scripts.check_live_backend import load_questions

        questions = load_questions(app.ROOT / "data" / "live_demo_questions.jsonl")
        supplied = next(row for row in questions if row["example_id"] == "guide_bat_ball")
        partial = "Mock reasoning stopped at its token limit without a final answer marker."
        expected = live_result()
        expected["initial"].update(answer="", response=partial, tokens=829, done_reason="length",
                                   parse_valid=False, prompt_eval_count=61, eval_count=768)
        expected["initial_record"].update(query=supplied["query"], base_answer="", base_response=partial,
                                          base_actual_tokens=829, base_done_reason="length")
        expected.update(selected_answer="", selected_response=partial, total_tokens=829)
        self.set_result(expected)
        ui = self.open_live()
        ui.text_area[0].input(supplied["query"]).run()
        self.runner.assert_not_called()
        ui.button[0].click().run()
        self.assert_no_exceptions(ui)
        self.assertEqual([metric.label for metric in ui.metric], list(app.DISPLAY_OUTCOMES))
        for metric in ui.metric:
            probability = expected["probabilities"][app.LABELS.index(metric.label)]
            self.assertEqual(metric.value, f"{probability:.1%}")
        text = self.visible_text(ui.tabs[0])
        self.assertIn("**Decision: Skip recheck**", text)
        self.assertIn("Initial response reached its token limit; it may be incomplete.", text)
        self.assertIn("No initial answer was parsed. See the full response.", text)
        rendered = [entry.value for entry in ui.tabs[0].markdown]
        selected_index = rendered.index("**Selected final answer**") + 1
        self.assertEqual(rendered[selected_index], "No final answer was parsed. See the full response.")
        self.assertIn("Separate supplied key: initial correct = False. Extraction failures count incorrect.", text)
        self.assertIn("Actual recheck outcome: unobserved; no completed recheck.", text)
        self.assertNotIn("Actual outcome: NO_CHANGE", text)
        self.assertIn(partial, text)
        saved = ui.session_state["live_result"]["result"]
        self.assertEqual(saved["selected_answer"], "")
        self.assertFalse(saved["rechecked"])
        self.assertEqual(saved["initial"]["tokens"], 829)
        self.assertEqual(self.runner.call_count, 1)
        self.assertNotIn("gold_answer", self.runner.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
