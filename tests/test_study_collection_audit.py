"""Independent resume/leakage checks using temporary, nonresearch fixtures."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from scripts import prepare_live_study
from rechecklab.data import question_hash
from rechecklab.study_collection import append_jsonl, collect_study, journal_state, read_jsonl
from rechecklab.study_protocol import freeze_protocol, normalize_call
from test_study_collection import fixture_protocol, generated


class PreparationCommandAudit(unittest.TestCase):
    def test_output_and_study_root_route_to_the_same_preparation_api(self):
        # Mock preparation: parsing either spelling must never download or call a solver.
        plan = {name: {} for name in ("planned_total_by_domain", "planned_questions_by_partition",
                                     "manifest_sha256", "duplicate_overlap_audit")}
        plan.update(status="PREPARED", study_plan_hash="fixture")
        for option in ("--output", "--study-root"):
            with self.subTest(option=option), patch("sys.argv", ["prepare_live_study.py", option,
                    "fixture-output", "--maths-target", "10", "--target-target", "5", "--seed", "7"]), \
                    patch.object(prepare_live_study, "prepare_study", return_value=plan) as prepare, \
                    redirect_stdout(io.StringIO()):
                prepare_live_study.main()
            prepare.assert_called_once_with(Path("fixture-output"), 10, 5, 7)


class CollectionIntegrityAudit(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.protocol = fixture_protocol()
        self.digest = freeze_protocol(self.root, self.protocol)
        self.question = {"example_id": "audit:1", "dataset": "gsm8k", "domain": "maths",
                         "partition": "source_train", "group_id": question_hash("Opaque fixture question?"),
                         "query": "Opaque fixture question?", "answer_type": "number"}
        append_jsonl(self.root / "questions.jsonl", self.question)
        append_jsonl(self.root / "trusted_keys.jsonl", {"example_id": "audit:1",
                    "gold_answer": "1001", "answer_type": "number"})
        self.plan = {"manifest_sha256": {name: hashlib.sha256((self.root / name).read_bytes()).hexdigest()
                     for name in ("questions.jsonl", "trusted_keys.jsonl")}}
        (self.root / "study_plan.json").write_text(json.dumps(self.plan), encoding="utf-8")

    def chat(self, protocol, question, answer_type, phase, previous, **kwargs):
        self.assertEqual(question, self.question["query"])
        self.assertEqual(answer_type, "number")
        self.assertEqual(set(kwargs), {"host", "timeout"})
        self.assertNotIn("1001", question)
        if phase == "initial":
            self.assertIsNone(previous)
        else:
            self.assertEqual(previous, "fixture reasoning\n\nFINAL: 1001")
        response = generated("1001")
        record, output = normalize_call(protocol, question, answer_type, response)
        return response, record, output

    def collected_fixture(self):
        with patch("rechecklab.study_collection.verify_backend"), \
                patch("rechecklab.study_collection.execute_call", side_effect=self.chat) as call:
            self.assertEqual(collect_study(self.root, minutes=5)["status"], "COMPLETE")
            self.assertEqual(call.call_count, 2)

    def terminal_fixture(self, status="failed"):
        started = {"example_id": self.question["example_id"], "group_id": self.question["group_id"],
                   "partition": "source_train", "phase": "initial", "call_id": "fixture-call",
                   "seed": 42, "attempt": 1, "session_id": "fixture-session", "protocol_hash": self.digest,
                   "status": "started"}
        terminal = {**started, "status": status}
        if status == "succeeded":
            response = generated("1001")
            _, output = normalize_call(self.protocol, self.question["query"], "number", response)
            terminal.update(response=response, output=output)
        return started, terminal

    def test_changed_manifest_or_keys_are_rejected_before_any_backend_call(self):
        for filename in ("questions.jsonl", "trusted_keys.jsonl"):
            path = self.root / filename
            original = path.read_bytes()
            # Even valid JSONL formatting changes invalidate the declared bytes.
            path.write_bytes(original + b"\n")
            with self.subTest(filename=filename), patch("rechecklab.study_collection.verify_backend") as backend, \
                    patch("rechecklab.study_collection.execute_call") as call:
                with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                    collect_study(self.root, minutes=5)
                backend.assert_not_called()
                call.assert_not_called()
            path.write_bytes(original)

    def test_missing_frozen_manifest_hashes_are_rejected(self):
        (self.root / "study_plan.json").write_text("{}", encoding="utf-8")
        with patch("rechecklab.study_collection.execute_call") as call:
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                collect_study(self.root, minutes=5)
            call.assert_not_called()

    def test_terminal_cannot_move_to_another_started_identity(self):
        second = {**self.question, "example_id": "audit:2", "query": "Different fixture question?",
                  "group_id": question_hash("Different fixture question?")}
        mutations = [{"phase": "recheck"}, {"seed": 43}, {"attempt": 2}, {"session_id": "other"},
                     {"example_id": second["example_id"], "group_id": second["group_id"]}]
        for changed in mutations:
            started, terminal = self.terminal_fixture()
            terminal.update(changed)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "Terminal call changed"):
                journal_state([started, terminal], [self.question, second], self.digest, self.protocol)

    def test_saved_success_must_normalize_from_the_recorded_backend_response(self):
        for field, changed in (("answer", "999"), ("tokens", 999), ("response", "Altered saved text")):
            started, terminal = self.terminal_fixture("succeeded")
            terminal["output"][field] = changed
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "actual response"):
                journal_state([started, terminal], [self.question], self.digest, self.protocol)

    def test_saved_pair_cannot_modify_outcomes_tokens_or_question_without_inference(self):
        self.collected_fixture()
        path = self.root / "pairs.jsonl"
        original = path.read_bytes()
        pair = read_jsonl(path)[0]
        mutations = {"label": "HELP", "base_correct": False, "base_actual_tokens": 999,
                     "query": "Altered question", "example_id": "foreign-question"}
        for field, changed in mutations.items():
            modified = copy.deepcopy(pair)
            modified[field] = changed
            path.write_text(json.dumps(modified) + "\n", encoding="utf-8")
            with self.subTest(field=field), patch("rechecklab.study_collection.verify_backend"), \
                    patch("rechecklab.study_collection.execute_call") as call:
                status = collect_study(self.root, minutes=5)
                self.assertEqual(status["status"], "BLOCKED")
                call.assert_not_called()
            path.write_bytes(original)

    def test_duplicate_pair_is_not_counted_as_a_new_completed_question(self):
        self.collected_fixture()
        append_jsonl(self.root / "pairs.jsonl", read_jsonl(self.root / "pairs.jsonl")[0])
        with patch("rechecklab.study_collection.verify_backend"), \
                patch("rechecklab.study_collection.execute_call") as call:
            status = collect_study(self.root, minutes=5)
            self.assertEqual(status["status"], "BLOCKED")
            call.assert_not_called()

    def test_original_call_and_two_explicit_retries_exhaust_the_frozen_limit(self):
        for attempt in (1, 2, 3):
            started, terminal = self.terminal_fixture()
            for event in (started, terminal):
                event.update(call_id=f"fixture-failure-{attempt}", attempt=attempt)
                append_jsonl(self.root / "calls.jsonl", event)
        with patch("rechecklab.study_collection.verify_backend"), \
                patch("rechecklab.study_collection.execute_call") as call:
            status = collect_study(self.root, minutes=5, retry_failures=True)
            self.assertEqual(status["status"], "BLOCKED")
            self.assertIn("retry limit", status["errors"][0]["error"])
            call.assert_not_called()

    def test_trusted_gold_is_absent_from_initial_and_recheck_engine_arguments(self):
        self.collected_fixture()
        self.assertEqual(read_jsonl(self.root / "pairs.jsonl")[0]["label"], "NO_CHANGE")

    def test_slow_progress_cannot_start_http_after_deadline_or_leave_uncertain_call(self):
        clock = [0.0]

        def consume_budget(message):
            clock[0] = 301.0

        with patch("rechecklab.study_collection.time.monotonic", side_effect=lambda: clock[0]), \
                patch("rechecklab.study_collection.verify_backend"), \
                patch("rechecklab.study_collection.execute_call") as call:
            status = collect_study(self.root, minutes=5, on_progress=consume_budget)
            self.assertEqual(status["status"], "PARTIAL")
            self.assertEqual(status["successful_calls"], 0)
            self.assertEqual(status["failed_calls"], 0)
            call.assert_not_called()
        events = read_jsonl(self.root / "calls.jsonl")
        self.assertEqual([event["status"] for event in events], ["started", "cancelled_no_request"])
        successes, failures, pending = journal_state(events, [self.question], self.digest, self.protocol)
        self.assertEqual((successes, failures, pending), ({}, [], []))
        # A request that never started needs no uncertain-resolution or failure retry flag.
        self.collected_fixture()

    def test_rejected_nonfinite_response_remains_durable_failure_evidence(self):
        rejected = generated("1001", prompt_eval_count=float("nan"))
        error = ValueError("Fixture backend prompt token count is nonfinite")
        error.response = rejected
        with patch("rechecklab.study_collection.verify_backend"), \
                patch("rechecklab.study_collection.execute_call", side_effect=error) as call:
            status = collect_study(self.root, minutes=5)
            self.assertEqual(status["status"], "BLOCKED")
            self.assertEqual(status["failed_calls"], 1)
            self.assertEqual(call.call_count, 1)
        events = read_jsonl(self.root / "calls.jsonl")
        self.assertEqual([event["status"] for event in events], ["started", "failed"])
        failed = events[-1]
        self.assertEqual(failed["failure_kind"], "response_metadata")
        self.assertEqual(failed["reported_response"]["prompt_eval_count"], {"invalid_nonfinite_float": "nan"})
        self.assertEqual(failed["reported_response"]["message"]["content"], "FINAL: 1001")
        self.assertNotIn("output", failed)
        self.assertFalse((self.root / "pairs.jsonl").exists())
        successes, failures, pending = journal_state(events, [self.question], self.digest, self.protocol)
        self.assertEqual(successes, {})
        self.assertEqual(len(failures), 1)
        self.assertEqual(pending, [])


if __name__ == "__main__":
    unittest.main()
