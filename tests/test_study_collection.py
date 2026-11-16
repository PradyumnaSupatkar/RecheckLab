"""Protocol/journal fixtures verify safety; they are never research evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import URLError

import numpy as np
import pandas as pd

from rechecklab.data import LABELS, question_hash
from rechecklab.features import FEATURE_NAMES, extract_features
from rechecklab.live import run_live_question
from rechecklab.predictor import OutcomePredictor, make_artifact
from rechecklab.study_collection import append_jsonl, collect_study, read_jsonl, validate_manifest
from rechecklab.study_protocol import (call_messages, create_protocol, freeze_protocol, grade_answer,
                                      normalize_call, parse_study_answer, protocol_hash, validate_protocol)


def fixture_protocol():
    backend = {"ollama_version": "fixture", "installed_model": {"name": "qwen3:4b", "digest": "fixture-digest"},
               "details": {}, "model_info": {}, "supported_thinking": {"values": [True]},
               "template": "fixture-template", "parameters": "fixture-parameters", "capabilities": []}
    with patch("rechecklab.study_protocol.inspect_backend", return_value=backend):
        return create_protocol()


def generated(answer="1", **changes):
    result = {"model": "qwen3:4b", "done": True, "done_reason": "stop", "message": {"content": "FINAL: " + answer, "thinking": "fixture reasoning"},
              "prompt_eval_count": 12, "eval_count": 7, "wall_seconds": .05}
    result.update(changes)
    return result


class FixtureEstimator:
    def __init__(self, recheck):
        self.classes_ = np.asarray(LABELS)
        self.n_features_in_ = len(FEATURE_NAMES)
        self.feature_names_in_ = np.asarray(FEATURE_NAMES)
        self.values = [.7, .2, .1] if recheck else [.1, .8, .1]
        self.seen = []

    def predict_proba(self, frame):
        self.seen.append(frame.copy())
        return np.tile(self.values, (len(frame), 1))


class StudyProtocol(unittest.TestCase):
    def test_explicit_numeric_and_choice_parsing_and_offline_grades(self):
        for text, expected in (("FINAL: ₹1,200.00", "1.2E+3"), ("FINAL: 1/2", "0.5"), ("FINAL: -2.5e2", "-2.5E+2")):
            self.assertEqual(parse_study_answer(text, "number"), expected)
        for text in ("intermediate 12 then 1", "FINAL: 1,5", "FINAL: NaN", "FINAL: 1/0", "FINAL: 2 or 3"):
            self.assertIsNone(parse_study_answer(text, "number"))
        self.assertEqual(parse_study_answer("FINAL: (b).", "choice"), "B")
        self.assertIsNone(parse_study_answer("Reasoning mentions option A", "choice"))
        self.assertTrue(grade_answer("1.2E+3", "1,200", "number"))
        self.assertFalse(grade_answer(None, "A", "choice"))

    def test_protocol_changes_fail_without_new_version(self):
        protocol = fixture_protocol()
        for section, key, value in (("generation", "initial_token_budget", 2048), ("prompts", "initial", "different"), ("parser", "version", "different")):
            bad = copy.deepcopy(protocol)
            bad[section][key] = value
            self.assertNotEqual(protocol_hash(bad), protocol_hash(protocol))
            with self.assertRaises(ValueError):
                validate_protocol(bad)

    def test_recheck_preserves_question_and_full_saved_response_without_key(self):
        messages = call_messages(fixture_protocol(), "Original question?", "recheck", "Saved thinking\n\nFINAL: 1")
        self.assertIn("Original question?", messages[0]["content"])
        self.assertEqual(messages[1]["content"], "Saved thinking\n\nFINAL: 1")
        self.assertNotIn("gold_answer", json.dumps(messages))

    def test_actual_metadata_is_required_and_parsing_failure_is_separate(self):
        protocol = fixture_protocol()
        bad = generated()
        del bad["prompt_eval_count"]
        with self.assertRaises(ValueError):
            normalize_call(protocol, "Q?", "number", bad)
        record, output = normalize_call(protocol, "Q?", "number", generated("not an answer"))
        self.assertFalse(output["parse_valid"])
        self.assertEqual(record["base_actual_tokens"], 19)
        self.assertFalse(record["base_finalizer_used"])


class StudyCollection(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.protocol = fixture_protocol()
        self.digest = freeze_protocol(self.root, self.protocol)
        query = "Fixture question with NEL\u0085 and LS\u2028: how many?"
        self.question = {"example_id": "fixture:1", "dataset": "gsm8k", "domain": "maths", "partition": "source_train",
                         "group_id": question_hash(query), "query": query, "answer_type": "number"}
        append_jsonl(self.root / "questions.jsonl", self.question)
        append_jsonl(self.root / "trusted_keys.jsonl", {"example_id": "fixture:1", "gold_answer": "1", "answer_type": "number"})
        (self.root / "study_plan.json").write_text(json.dumps({"manifest_sha256": {
            name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
            for name in ("questions.jsonl", "trusted_keys.jsonl")}}), encoding="utf-8")

    def chat(self, protocol, question, answer_type, phase, previous, **kwargs):
        self.assertEqual(question, self.question["query"])
        self.assertNotIn("gold_answer", kwargs)
        response = generated()
        record, output = normalize_call(protocol, question, answer_type, response)
        return response, record, output

    def test_unicode_jsonl_and_completed_resume_do_not_duplicate_calls(self):
        self.assertEqual(read_jsonl(self.root / "questions.jsonl"), [self.question])
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call", side_effect=self.chat) as call:
            status = collect_study(self.root, minutes=5)
            self.assertEqual(status["status"], "COMPLETE")
            self.assertEqual(call.call_count, 2)
            again = collect_study(self.root, minutes=5)
            self.assertEqual(call.call_count, 2)
            self.assertEqual(again["completed_pairs"], 1)
        pair = read_jsonl(self.root / "pairs.jsonl")[0]
        self.assertEqual(pair["label"], "NO_CHANGE")
        self.assertEqual(pair["base_actual_tokens"], pair["base_prompt_eval_count"] + pair["base_eval_count"])

    def test_failed_recheck_preserves_initial_and_requires_explicit_retry(self):
        calls = []
        def fail(protocol, question, kind, phase, previous, **kwargs):
            calls.append(phase)
            if phase == "recheck":
                raise URLError("fixture connection failure")
            return self.chat(protocol, question, kind, phase, previous, **kwargs)
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call", side_effect=fail):
            result = collect_study(self.root, minutes=5)
        self.assertEqual(result["status"], "BLOCKED")
        self.assertFalse((self.root / "pairs.jsonl").exists())
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call", side_effect=self.chat) as retry:
            self.assertEqual(collect_study(self.root, minutes=5)["status"], "BLOCKED")
            retry.assert_not_called()
            self.assertEqual(collect_study(self.root, minutes=5, retry_failures=True)["status"], "COMPLETE")
            self.assertEqual(retry.call_count, 1)
            self.assertEqual(retry.call_args.args[3], "recheck")
        self.assertEqual(calls, ["initial", "recheck"])

    def test_both_call_checkpoints_recover_pair_without_inference(self):
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call", side_effect=self.chat):
            collect_study(self.root, minutes=5)
        (self.root / "pairs.jsonl").unlink()
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call") as call:
            result = collect_study(self.root, minutes=5)
            call.assert_not_called()
        self.assertEqual(result["completed_pairs"], 1)

    def test_uncertain_started_call_blocks_and_budget_does_not_start_unfinishable_call(self):
        append_jsonl(self.root / "calls.jsonl", {"example_id": "fixture:1", "group_id": self.question["group_id"],
                     "phase": "initial", "call_id": "uncertain-fixture", "status": "started", "protocol_hash": self.digest})
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call") as call:
            status = collect_study(self.root, minutes=5)
            call.assert_not_called()
        self.assertEqual(status["status"], "BLOCKED")
        self.assertIn("Uncertain started", status["errors"][0]["error"])
        (self.root / "calls.jsonl").unlink()
        with patch("rechecklab.study_collection.verify_backend"), patch("rechecklab.study_collection.execute_call") as call:
            status = collect_study(self.root, minutes=.01)
            call.assert_not_called()
        self.assertEqual(status["status"], "PARTIAL")

    def test_duplicate_groups_and_keys_in_manifest_are_rejected(self):
        keys = read_jsonl(self.root / "trusted_keys.jsonl")
        bad = {**self.question, "gold_answer": "1"}
        with self.assertRaisesRegex(ValueError, "must not contain"):
            validate_manifest([bad], keys)
        with self.assertRaisesRegex(ValueError, "Duplicate/overlapping"):
            validate_manifest([self.question, {**self.question, "example_id": "fixture:2"}], keys)

    def test_collection_and_live_use_identical_initial_features_for_both_policy_paths(self):
        expected, _ = normalize_call(self.protocol, self.question["query"], "number", generated())
        for recheck in (False, True):
            estimator = FixtureEstimator(recheck)
            identity = {"model": self.protocol["model"], "model_digest": self.protocol["model_digest"], "action": "recheck",
                        "recheck_protocol": "live-study:" + self.digest, "study_protocol_hash": self.digest,
                        "live_study_protocol": self.protocol}
            bundle = make_artifact(OutcomePredictor(estimator, "fixture-only"), .1, protocol=identity)
            with patch("rechecklab.study_protocol.verify_backend"), patch("rechecklab.study_protocol.ollama_chat", return_value=generated()) as chat:
                result = run_live_question(self.question["query"], bundle, answer_type="number", allow_experimental=True)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(chat.call_count, 2 if recheck else 1)
            pd.testing.assert_frame_equal(estimator.seen[0], extract_features(pd.DataFrame([expected])))
            if recheck:
                self.assertEqual(chat.call_args.args[1][1]["content"], result["initial"]["response"])


if __name__ == "__main__":
    unittest.main()
