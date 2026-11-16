"""Mock live I/O to verify the zero-or-one recheck protocol and failure handling."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import numpy as np

from rechecklab.collector import INITIAL_INSTRUCTION, RECHECK_INSTRUCTION, actual_tokens, collect_pairs, ollama_chat
from rechecklab.data import LABELS, collection_protocol, load_pairs
from rechecklab.features import FEATURE_NAMES
from rechecklab.live import DEFAULT_SETTINGS, probe_backend, protocol_fingerprint, run_live_question
from rechecklab.predictor import OutcomePredictor, make_artifact


MODEL = "qwen3:4b"
DIGEST = "mock-checkpoint-digest"


class FixedEstimator:
    def __init__(self, probabilities):
        self.classes_ = np.asarray(LABELS)
        self.n_features_in_ = len(FEATURE_NAMES)
        self.feature_names_in_ = np.asarray(FEATURE_NAMES)
        self.values = probabilities
        self.seen = []

    def predict_proba(self, frame):
        self.seen.append(frame.copy())
        return np.tile(self.values, (len(frame), 1))


def artifact(recheck=False, public=False):
    protocol_id, settings = collection_protocol(DEFAULT_SETTINGS)
    protocol = dict(model=MODEL, model_digest=DIGEST, recheck_protocol=protocol_id,
                    action="recheck", generation_settings=settings)
    if public:
        protocol = dict(model=MODEL, recheck_protocol="sevra:mock-revision:active_verify", action="active_verify",
                        published_generation_settings=dict(base_generation_limit=4096, finalizer_max_tokens=512))
    probabilities = [.6, .3, .1] if recheck else [.1, .7, .2]
    return make_artifact(OutcomePredictor(FixedEstimator(probabilities), "mock-test-only"), .1, protocol=protocol)


def tags(model=MODEL, digest=DIGEST):
    return io.BytesIO(json.dumps(dict(models=[dict(name=model, digest=digest)])).encode())


def generated(text="Work.\nFINAL: 5", tokens=(10, 5), **changes):
    result = dict(model=MODEL, done=True, done_reason="stop", message=dict(content=text),
                  prompt_eval_count=tokens[0], eval_count=tokens[1], wall_seconds=.01)
    result.update(changes)
    return result


class LiveBackend(unittest.TestCase):
    def test_probe_validates_the_exact_saved_model_and_checkpoint(self):
        bundle = artifact()
        with patch("rechecklab.live.urlopen", return_value=tags()):
            backend = probe_backend(bundle)
        self.assertTrue(backend["ready"])
        self.assertEqual(backend["status"], "compatible")
        self.assertTrue(backend["protocol_match"])
        self.assertEqual(backend["live_protocol"]["model_digest"], DIGEST)
        with patch("rechecklab.live.urlopen", return_value=tags("different:latest")):
            self.assertEqual(probe_backend(bundle)["status"], "model_missing")
        with patch("rechecklab.live.urlopen", return_value=tags(digest="different-checkpoint")):
            self.assertEqual(probe_backend(bundle)["status"], "checkpoint_mismatch")

    def test_unavailable_and_invalid_backend_responses_are_visible(self):
        with patch("rechecklab.live.urlopen", side_effect=URLError("offline")):
            status = probe_backend(artifact())
        self.assertFalse(status["ready"])
        self.assertEqual(status["status"], "unavailable")
        self.assertEqual(status["error_stage"], "backend")
        self.assertIn("Ollama is unavailable", status["message"])
        for payload in ("invalid JSON", json.dumps({}), json.dumps(dict(models=[dict(name=MODEL, digest="")]))):
            with patch("rechecklab.live.urlopen", return_value=io.BytesIO(payload.encode())):
                self.assertEqual(probe_backend(artifact())["status"], "invalid_backend_response")

    def test_wrong_predictor_is_rejected_before_backend_or_llm_calls(self):
        bundle = artifact()
        bundle["predictor"] = {"model_name": "wrong-type", "estimator": FixedEstimator([.1, .7, .2])}
        with patch("rechecklab.live.urlopen") as request, patch("rechecklab.live.ollama_chat") as chat:
            status = probe_backend(bundle)
            result = run_live_question("A question?", bundle)
        self.assertFalse(status["ready"])
        self.assertEqual(status["status"], "predictor_load_error")
        self.assertEqual(status["error_stage"], "predictor")
        self.assertIn("Saved predictor validation failed:", status["message"])
        self.assertIn("OutcomePredictor", status["message"])
        self.assertEqual(result["status"], "predictor_load_error")
        self.assertEqual(result["error_stage"], "predictor")
        self.assertEqual(result["backend"]["status"], "predictor_load_error")
        self.assertIsNone(result["initial"])
        self.assertIsNone(result["probabilities"])
        request.assert_not_called()
        chat.assert_not_called()

    def test_invalid_endpoint_is_configuration_error_without_backend_calls(self):
        with patch("rechecklab.live.urlopen") as request, patch("rechecklab.live.ollama_chat") as chat:
            status = probe_backend(artifact(), host="file:///invalid-endpoint")
        self.assertFalse(status["ready"])
        self.assertEqual(status["status"], "invalid_configuration")
        self.assertEqual(status["error_stage"], "configuration")
        self.assertNotIn("Saved predictor validation failed", status["message"])
        request.assert_not_called()
        chat.assert_not_called()

    def test_unknown_actual_checkpoint_blocks_even_public_experimental_inference(self):
        unknown = io.BytesIO(json.dumps(dict(models=[dict(name=MODEL)])).encode())
        with patch("rechecklab.live.urlopen", return_value=unknown), patch("rechecklab.live.ollama_chat") as chat:
            result = run_live_question("A question?", artifact(public=True), allow_experimental=True)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["backend"]["status"], "checkpoint_unknown")
        self.assertFalse(result["backend"]["ready"])
        self.assertIsNone(result["backend"]["model_digest"])
        chat.assert_not_called()

    def test_protocol_mismatch_blocks_custom_gates_even_with_experimental_flag(self):
        bundle = artifact()
        bundle["protocol"]["generation_settings"]["initial_instruction"] = "Different prompt"
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat") as chat:
            result = run_live_question("What is 2 + 3?", bundle, allow_experimental=True)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["backend"]["status"], "protocol_mismatch")
        chat.assert_not_called()

    def test_legacy_custom_context_is_unknown_and_blocks_new_live_execution(self):
        bundle = artifact()
        saved = bundle["protocol"]["generation_settings"]
        saved.pop("context_window")
        saved["protocol_version"] = "0.1"
        bundle["protocol"]["recheck_protocol"] = collection_protocol(saved)[0]
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat") as chat:
            result = run_live_question("A question?", bundle, allow_experimental=True)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["backend"]["status"], "protocol_mismatch")
        self.assertTrue(any("context_window" in reason for reason in result["backend"]["mismatches"]))
        chat.assert_not_called()

    def test_public_experiment_requires_explicit_allow_and_retains_actual_protocol(self):
        bundle = artifact(public=True)
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat") as chat:
            blocked = run_live_question("What is 2 + 3?", bundle)
        self.assertEqual(blocked["status"], "blocked")
        chat.assert_not_called()
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=generated()):
            result = run_live_question("What is 2 + 3?", bundle, allow_experimental=True)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["backend"]["experimental"])
        self.assertTrue(result["live_protocol"]["recheck_protocol"].startswith("ollama:"))
        self.assertEqual(result["live_protocol"]["action"], "recheck")
        self.assertEqual(result["initial_record"]["recheck_protocol"], result["live_protocol"]["recheck_protocol"])
        self.assertNotEqual(result["initial_record"]["recheck_protocol"], bundle["protocol"]["recheck_protocol"])
        reasons = " ".join(result["backend"]["mismatches"])
        self.assertIn("4096", reasons)
        self.assertIn("512", reasons)
        self.assertEqual(result["live_protocol"]["generation_settings"]["context_window"], 4096)
        self.assertEqual(result["live_protocol"]["generation_settings"]["protocol_version"], "0.2")

    def test_skip_generates_only_initial_and_keeps_truth_unknown(self):
        bundle = artifact()
        progress = []
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=generated()) as chat:
            result = run_live_question("What is 2 + 3?", bundle, on_progress=lambda *args: progress.append(args))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["decision"], "Skip recheck")
        self.assertFalse(result["rechecked"])
        self.assertEqual(result["selected_source"], "initial")
        self.assertEqual(result["selected_answer"], "5")
        self.assertEqual(result["actual_outcome"], "Unknown")
        self.assertEqual(result["correctness"], "Unknown")
        self.assertEqual(result["recheck_tokens"], 0)
        self.assertEqual(result["total_tokens"], 15)
        self.assertFalse(result["initial"]["finalizer_used"])
        self.assertEqual(result["initial"]["prompt_eval_count"], 10)
        self.assertEqual(result["initial"]["eval_count"], 5)
        self.assertTrue(result["initial_attempt"]["validated"])
        self.assertEqual([stage for stage, _ in progress], ["checking_backend", "generating_initial", "predicting", "complete"])

    def test_recheck_is_exactly_one_fixed_initial_recheck_without_keys(self):
        bundle = artifact(recheck=True)
        responses = [generated(), generated("Checked.\nFINAL: 6", tokens=(20, 7))]
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", side_effect=responses) as chat:
            result = run_live_question("What is 2 + 3?", bundle)
        self.assertEqual(chat.call_count, 2)
        first_messages = chat.call_args_list[0].args[1]
        second_messages = chat.call_args_list[1].args[1]
        self.assertEqual(first_messages, [dict(role="user", content=INITIAL_INSTRUCTION + "\n\nWhat is 2 + 3?")])
        self.assertEqual(second_messages, first_messages + [dict(role="assistant", content=responses[0]["message"]["content"]),
                                                          dict(role="user", content=RECHECK_INSTRUCTION)])
        self.assertNotIn("gold", json.dumps(second_messages))
        self.assertNotIn("correct", json.dumps(second_messages))
        self.assertEqual(chat.call_args_list[0].args[4], 42)
        self.assertEqual(chat.call_args_list[1].args[4], 43)
        self.assertEqual(chat.call_args_list[0].kwargs["timeout"], 120)
        self.assertEqual(chat.call_args_list[0].kwargs["context_window"], 4096)
        self.assertEqual(chat.call_args_list[1].kwargs["context_window"], 4096)
        self.assertEqual(result["decision"], "Recheck")
        self.assertTrue(result["recheck_attempted"])
        self.assertTrue(result["rechecked"])
        self.assertEqual(result["selected_source"], "revised")
        self.assertEqual(result["selected_answer"], "6")
        self.assertEqual(result["total_tokens"], 42)
        self.assertEqual(result["revised"]["prompt_eval_count"], 20)
        self.assertEqual(result["revised"]["eval_count"], 7)
        self.assertTrue(result["initial_attempt"]["validated"])
        self.assertTrue(result["recheck_attempt"]["validated"])
        self.assertEqual(result["actual_outcome"], "Unknown")

    def test_live_record_features_use_actual_metadata(self):
        bundle = artifact()
        response = generated(tokens=(100, 20), done_reason="length")
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=response):
            result = run_live_question("A question?", bundle)
        seen = bundle["predictor"].estimator.seen[-1]
        self.assertAlmostEqual(seen.initial_log_actual_tokens.iloc[0], np.log1p(120))
        self.assertEqual(seen.initial_budget_exhausted.iloc[0], 1)
        self.assertEqual(seen.initial_finalizer_used.iloc[0], 0)
        self.assertEqual(result["initial"]["tokens"], 120)

    def test_initial_invalid_metadata_stops_before_prediction_or_recheck(self):
        for changes in (dict(done_reason=None), dict(done_reason=""), dict(done_reason="unload"),
                        dict(prompt_eval_count=-1), dict(eval_count=1.5), dict(eval_count=None),
                        dict(model="different-model"), dict(done=False), dict(message=dict(content=None))):
            bundle = artifact(recheck=True)
            response = generated(**changes)
            with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=response) as chat:
                result = run_live_question("A question?", bundle)
            with self.subTest(changes=changes):
                self.assertEqual(result["status"], "initial_failed")
                self.assertEqual(chat.call_count, 1)
                self.assertIsNone(result["probabilities"])
                self.assertIsNone(result["decision"])
                self.assertIsNotNone(result["initial_attempt"])
                self.assertFalse(result["initial_attempt"]["validated"])
        response = generated()
        response.pop("eval_count")
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=response):
            self.assertEqual(run_live_question("A question?", artifact())["status"], "initial_failed")

    def test_initial_connection_error_stops_the_pipeline(self):
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", side_effect=URLError("disconnected")) as chat:
            result = run_live_question("A question?", artifact(recheck=True))
        self.assertEqual(result["status"], "initial_failed")
        self.assertEqual(chat.call_count, 1)
        self.assertIsNone(result["initial"])

    def test_incomplete_helper_response_is_retained_as_unvalidated_audit(self):
        incomplete = generated("Partial reasoning", done=False)
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.collector.urlopen", return_value=io.BytesIO(json.dumps(incomplete).encode())):
            result = run_live_question("A question?", artifact(recheck=True))
        self.assertEqual(result["status"], "initial_failed")
        self.assertIsNone(result["initial"])
        self.assertEqual(result["initial_attempt"]["response"], "Partial reasoning")
        self.assertEqual(result["initial_attempt"]["reported_metadata"]["done"], "False")
        self.assertFalse(result["initial_attempt"]["validated"])

    def test_recheck_failure_retains_initial_and_does_not_invent_an_outcome(self):
        for failure in (URLError("disconnected"), generated(eval_count=np.nan)):
            with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", side_effect=[generated(), failure]) as chat:
                result = run_live_question("A question?", artifact(recheck=True))
            with self.subTest(failure=str(failure)):
                self.assertEqual(chat.call_count, 2)
                self.assertEqual(result["status"], "recheck_failed")
                self.assertEqual(result["decision"], "Recheck")
                self.assertEqual(result["selected_source"], "initial_fallback")
                self.assertEqual(result["selected_answer"], "5")
                self.assertFalse(result["rechecked"])
                self.assertIsNone(result["revised"])
                self.assertIsNone(result["total_tokens"])
                self.assertEqual(result["actual_outcome"], "Unknown")

    def test_invalid_probability_output_retains_initial_without_recheck(self):
        bundle = artifact(recheck=True)
        bundle["predictor"].estimator.values = [np.nan, .5, .5]
        with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=generated()) as chat:
            result = run_live_question("A question?", bundle)
        self.assertEqual(result["status"], "prediction_failed")
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(result["selected_source"], "initial")
        self.assertIsNone(result["decision"])

    def test_open_answer_extraction_never_guesses_an_intermediate_answer(self):
        for content, answer in (("First calculate 84, then think again.", ""),
                                ("Answer: guessed", ""), ("FINAL: first\nFINAL: last", "last")):
            with patch("rechecklab.live.urlopen", return_value=tags()), patch("rechecklab.live.ollama_chat", return_value=generated(content)):
                result = run_live_question("A question?", artifact())
            with self.subTest(content=content):
                self.assertEqual(result["selected_answer"], answer)
                self.assertEqual(result["selected_response"], content)

    def test_empty_question_or_invalid_timeout_never_calls_backend(self):
        with patch("rechecklab.live.urlopen") as backend, patch("rechecklab.live.ollama_chat") as chat:
            self.assertEqual(run_live_question(" ", artifact())["status"], "input_error")
            self.assertEqual(run_live_question("A question", artifact(), timeout=600)["status"], "input_error")
        backend.assert_not_called()
        chat.assert_not_called()

    def test_fingerprint_changes_for_any_actual_solver_setting(self):
        original = protocol_fingerprint("http://localhost:11434", MODEL, DIGEST, DEFAULT_SETTINGS)
        self.assertEqual(original, protocol_fingerprint("http://localhost:11434/", MODEL, DIGEST, DEFAULT_SETTINGS))
        self.assertNotEqual(original, protocol_fingerprint("http://localhost:11434", MODEL, DIGEST, {**DEFAULT_SETTINGS, "budget": 10}))
        self.assertNotEqual(original, protocol_fingerprint("http://localhost:11434", MODEL, "different-digest", DEFAULT_SETTINGS))

    def test_shared_token_helper_rejects_fractional_negative_and_missing_counts(self):
        for value in (-1, 1.5, True, np.nan, np.inf, None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                actual_tokens(generated(eval_count=value))
        self.assertEqual(actual_tokens(generated(tokens=(0, 0))), 0)

    def test_collector_timeout_keyword_reaches_the_network_call(self):
        with patch("rechecklab.collector.urlopen", return_value=io.BytesIO(json.dumps(generated()).encode())) as request:
            ollama_chat(MODEL, [dict(role="user", content="A question")], "http://localhost:11434", 768, 42, 0.0, False, timeout=15)
        self.assertEqual(request.call_args.kwargs["timeout"], 15)

    def test_context_option_reaches_ollama_without_changing_legacy_default(self):
        for context in (None, 4096):
            with patch("rechecklab.collector.urlopen", return_value=io.BytesIO(json.dumps(generated()).encode())) as request:
                ollama_chat(MODEL, [dict(role="user", content="A question")], "http://localhost:11434", 768, 42, 0.0, False,
                            context_window=context)
            payload = json.loads(request.call_args.args[0].data)
            with self.subTest(context=context):
                self.assertEqual(payload["options"].get("num_ctx"), context)
                if context is None:
                    self.assertNotIn("num_ctx", payload["options"])
        for invalid in (0, -1, True, 1.5):
            with self.subTest(context=invalid), self.assertRaises(ValueError):
                ollama_chat(MODEL, [], "http://localhost:11434", 768, 42, 0.0, False, context_window=invalid)

    def test_http_error_exposes_bounded_ollama_server_diagnostics(self):
        body = io.BytesIO(json.dumps(dict(error="runner needs 38 GiB for its context cache")).encode())
        error = HTTPError("http://localhost:11434/api/chat", 500, "Server Error", None, body)
        with patch("rechecklab.collector.urlopen", side_effect=error), self.assertRaisesRegex(URLError, "Ollama HTTP 500: runner needs 38 GiB"):
            ollama_chat(MODEL, [], "http://localhost:11434", 768, 42, 0.0, False, context_window=4096)
        huge = io.BytesIO(json.dumps(dict(error="x" * 9000)).encode())
        error = HTTPError("http://localhost:11434/api/chat", 500, "Server Error", None, huge)
        with patch("rechecklab.collector.urlopen", side_effect=error), self.assertRaises(URLError) as caught:
            ollama_chat(MODEL, [], "http://localhost:11434", 768, 42, 0.0, False)
        self.assertEqual(huge.tell(), 4097)
        self.assertLess(len(str(caught.exception)), 4250)
        self.assertIn("truncated", str(caught.exception))

    def test_new_collection_records_context_for_future_matching_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            pairs = Path(directory) / "pairs.jsonl"
            questions.write_text(json.dumps(dict(example_id="unit-only", dataset="unit-only", query="What is 2 + 3?",
                                                 gold_answer="5", answer_type="number")) + "\n", encoding="utf-8")
            with patch("rechecklab.collector.urlopen", return_value=tags()), patch("rechecklab.collector.ollama_chat", side_effect=[generated(), generated()]) as chat:
                self.assertEqual(collect_pairs(questions, pairs, model=MODEL, context_window=4096), 1)
            self.assertTrue(all(call.kwargs["context_window"] == 4096 for call in chat.call_args_list))
            saved = json.loads(pairs.with_suffix(".metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["context_window"], 4096)
            self.assertEqual(saved["protocol_version"], "0.2")
            loaded = load_pairs(pairs)
            self.assertEqual(loaded.attrs["generation_settings"]["context_window"], 4096)
            expected_id = collection_protocol(saved)[0]
            self.assertEqual(loaded.recheck_protocol.iloc[0], expected_id)


if __name__ == "__main__":
    unittest.main()
