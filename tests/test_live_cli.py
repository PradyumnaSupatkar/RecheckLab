"""CLI safeguard fixtures only; no model calls or research results are fabricated."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.check_live_backend import (
    load_questions, main, normalize_trusted_answer, score_after_execution,
)


class LiveCliSafeguards(unittest.TestCase):
    def test_question_file_cannot_supply_answers_to_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(json.dumps(dict(example_id="test", query="What is 17 × 6?", gold_answer="102")) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "question|Question|disallowed"):
                load_questions(path)

    def test_strict_postscoring_handles_rupee_prefixes_without_reading_reasoning(self):
        for answer in ("35", "35.00", "₹35", "Rs. 35", "Rs35"):
            with self.subTest(answer=answer):
                self.assertEqual(normalize_trusted_answer(answer, "number"), "35")
        for answer in ("The cost is135 and change35", "₹35 or₹102", "3,5", "NaN", "Infinity", "1e9999999999"):
            with self.subTest(answer=answer):
                self.assertIsNone(normalize_trusted_answer(answer, "number"))

    def test_skipping_does_not_invent_a_known_recheck_outcome(self):
        key = dict(answer_type="number", normalized_answer="35")
        result = dict(status="complete", initial=dict(answer="₹35"), rechecked=False, revised=None)
        score = score_after_execution(result, key)
        self.assertTrue(score["initial_correct"])
        self.assertIsNone(score["observed_outcome"])
        result.update(rechecked=True, revised=dict(answer="₹5"))
        self.assertEqual(score_after_execution(result, key)["observed_outcome"], "HURT")

    def test_distinct_long_numeric_answers_never_round_to_same_grade(self):
        first = normalize_trusted_answer("123456789012345678901234567890", "number")
        second = normalize_trusted_answer("123456789012345678901234567891", "number")
        self.assertNotEqual(first, second)

    def test_no_key_or_incomplete_call_cannot_claim_an_observed_outcome(self):
        key = dict(answer_type="number", normalized_answer="35")
        completed = dict(status="complete", initial=dict(answer="35"), rechecked=True, revised=dict(answer="5"))
        unknown = score_after_execution(completed, None)
        self.assertFalse(unknown["correctness_known"])
        self.assertIsNone(unknown["observed_outcome"])
        partial_initial = dict(status="initial_failed", initial=None,
                               initial_attempt=dict(response="FINAL: 35"), rechecked=False, revised=None)
        self.assertFalse(score_after_execution(partial_initial, key)["correctness_known"])
        partial_revision = {**completed, "status": "recheck_failed"}
        partial = score_after_execution(partial_revision, key)
        self.assertTrue(partial["initial_correct"])
        self.assertIsNone(partial["revised_correct"])
        self.assertIsNone(partial["observed_outcome"])

    def test_complete_missing_parsed_answer_follows_explicit_format_scoring(self):
        key = dict(answer_type="number", normalized_answer="35")
        completed = dict(status="complete", initial=dict(answer="35"), rechecked=True, revised=dict(answer=""))
        score = score_after_execution(completed, key)
        self.assertFalse(score["revised_correct"])
        self.assertEqual(score["observed_outcome"], "HURT")
        self.assertIn("missing/invalid parsed answers count incorrect", score["reason"])

    def test_separate_keys_never_reach_engine_and_every_repeat_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            questions = Path(directory) / "questions.jsonl"
            keys = Path(directory) / "keys.jsonl"
            output = Path(directory) / "attempts.jsonl"
            questions.write_text(json.dumps(dict(example_id="test", query="What is 17 × 6?", answer_type="open")) + "\n", encoding="utf-8")
            keys.write_text(json.dumps(dict(example_id="test", gold_answer="102", answer_type="number")) + "\n", encoding="utf-8")
            fixture = dict(status="complete", initial=dict(answer="102"), decision="Skip recheck",
                           recheck_attempted=False, rechecked=False, revised=None, error=None)
            arguments = ["check_live_backend.py", "--question-file", str(questions), "--key-file", str(keys),
                         "--output", str(output), "--allow-experimental"]
            with patch("sys.argv", arguments), patch("rechecklab.live.load_live_artifact", return_value={}), \
                 patch("rechecklab.live.probe_backend", return_value=dict(ready=True)), \
                 patch("rechecklab.live.run_live_question", return_value=fixture) as engine, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
                self.assertEqual(main(), 0)
            self.assertEqual(engine.call_count, 2)
            for call in engine.call_args_list:
                self.assertEqual(call.args, ("What is 17 × 6?", {}))
                self.assertNotIn("gold_answer", call.kwargs)
                self.assertEqual(call.kwargs["answer_type"], "open")
            records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["group_id"], records[1]["group_id"])
            self.assertNotEqual(records[0]["attempt_id"], records[1]["attempt_id"])
            self.assertTrue(all(row["trusted_evaluation"]["initial_correct"] for row in records))
            self.assertTrue(all(row["trusted_evaluation"]["observed_outcome"] is None for row in records))


if __name__ == "__main__":
    unittest.main()
