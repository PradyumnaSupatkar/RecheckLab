"""Verify that diagnostic handoffs describe actual partial-study evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from rechecklab.study_report import DEFAULT_TARGETS, OUTCOMES, generate_status


class StudyReport(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.project = Path(self.temporary.name)
        self.study = self.project / "outputs" / "live_study" / "v0.3.0"
        self.study.mkdir(parents=True)
        # Aggregation fixtures have no model calls. Shared science validation
        # has separate real-journal tests; this file checks its report handoff.
        audit_patch = patch("rechecklab.study_report.validate_study_inputs", side_effect=self.audit_fixture)
        self.audit = audit_patch.start()
        self.addCleanup(audit_patch.stop)

    def audit_fixture(self, root):
        rows = [json.loads(line) for line in (Path(root) / "pairs.jsonl").read_text(encoding="utf-8").split("\n") if line.strip()]
        return (None, None, None, None, None, None, rows)

    def write_json(self, name, value):
        path = self.study / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def write_rows(self, name, rows):
        (self.study / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def prepared(self, targets):
        self.write_json("study_plan.json", dict(planned_questions_by_partition=targets, datasets={"gsm8k": {"revision": "pinned-test-revision"}}))
        self.write_json("protocol.json", dict(model="qwen3:4b", model_digest="test-digest", generation={"thinking": True}))
        (self.study / "protocol_hash.txt").write_text("test-protocol-hash\n", encoding="utf-8")
        questions = [dict(example_id=f"{partition}:{index}", partition=partition,
                          domain="maths" if partition.startswith("source_") else "commonsenseqa", query=f"Fixture question {partition} {index}")
                     for partition, count in targets.items() for index in range(count)]
        self.write_rows("questions.jsonl", questions)
        return questions

    def pair(self, question, label="NO_CHANGE", **changes):
        row = dict(example_id=question["example_id"], partition=question["partition"], domain=question["domain"],
                   protocol_hash="test-protocol-hash", query=question["query"], label=label,
                   base_answer="1", final_answer="2", base_response="Fixture initial. FINAL: 1", intervention_response="Fixture recheck. FINAL: 2",
                   base_actual_tokens=15, action_actual_tokens=16, base_done_reason="stop", base_finalizer_used=False,
                   base_correct=label == "HURT", final_correct=label == "HELP")
        row.update(changes)
        return row

    def test_missing_run_still_writes_partial_status_and_bounded_bundle(self):
        public = self.project / "outputs" / "v0.2" / "reports"
        public.mkdir(parents=True)
        (public / "predictor_diagnostics.json").write_text('{"accuracy": 1.0}', encoding="utf-8")
        status = generate_status(self.project, self.study)
        self.assertEqual(status["overall_status"], "PARTIAL")
        self.assertEqual(status["collection"]["planned_questions"], 1500)
        self.assertEqual(status["collection"]["completed_pairs"], 0)
        self.assertEqual(status["stages"]["evaluation"]["status"], "NOT_RUN")
        self.assertEqual(status["training_evidence"]["metrics"], {})
        self.assertIn("collect_live_study.py", status["resume_commands"][0])
        directory = self.project / "reports" / "live_study"
        self.assertTrue((directory / "CURRENT_STATUS.md").is_file())
        self.assertTrue((directory / "CURRENT_STATUS.json").is_file())
        self.assertLess((directory / "diagnostic_bundle.zip").stat().st_size, 10 * 1024 * 1024)
        with zipfile.ZipFile(directory / "diagnostic_bundle.zip") as bundle:
            self.assertIn("CURRENT_STATUS.md", bundle.namelist())
            self.assertFalse(any("v0.2" in name for name in bundle.namelist()))

    def test_calls_and_pairs_distinguish_transport_failure_from_parse_failure(self):
        questions = self.prepared({"source_train": 2, "target_adaptation_pool": 1})
        self.write_rows("pairs.jsonl", [self.pair(questions[0], "HELP", base_answer="")])
        calls = [dict(example_id=questions[0]["example_id"], phase=phase, call_id=phase, attempt=1,
                      status="succeeded", protocol_hash="test-protocol-hash") for phase in ("initial", "recheck")]
        calls += [dict(example_id=questions[1]["example_id"], phase="initial", call_id="failed", attempt=1,
                       status="failed", protocol_hash="test-protocol-hash", error="Fixture connection failure", error_type="TimeoutError", traceback="Fixture trace"),
                  dict(example_id=questions[2]["example_id"], phase="initial", call_id="unfinished", attempt=2,
                       status="started", protocol_hash="test-protocol-hash")]
        self.write_rows("calls.jsonl", calls)
        self.write_json("collection_status.json", dict(elapsed_seconds=60))
        status = generate_status(self.project, self.study)
        collection = status["collection"]
        self.assertEqual(collection["completed_pairs"], 1)
        self.assertEqual(collection["failed_calls"], 1)
        self.assertEqual(collection["unfinished_calls"], 1)
        self.assertEqual(collection["retry_calls"], 1)
        self.assertEqual(collection["initial_parse_failures"], 1)
        self.assertEqual(collection["recheck_parse_failures"], 0)
        self.assertEqual(collection["class_counts"], dict(HELP=1, HURT=0, NO_CHANGE=0))
        self.assertEqual(status["counts_by_partition"]["source_train"]["completed_unique_questions"], 1)
        self.assertEqual(status["stages"]["training_adaptation"]["status"], "INCOMPLETE")
        self.assertEqual(status["collection_eta"]["estimated_remaining_seconds"], 120)
        self.assertTrue(any(row.get("traceback") == "Fixture trace" for row in status["errors"]))

    def test_duplicate_unknown_and_incompatible_pairs_are_not_counted(self):
        question = self.prepared({"source_train": 2})[0]
        valid = self.pair(question, "NO_CHANGE")
        self.write_rows("pairs.jsonl", [valid, valid, self.pair(question, "HELP", protocol_hash="other-protocol"),
                                        dict(valid, example_id="not-in-manifest")])
        status = generate_status(self.project, self.study)
        self.assertEqual(status["collection"]["completed_pairs"], 1)
        self.assertEqual(status["collection"]["rejected_pair_rows"], 3)
        self.assertEqual(status["collection"]["class_counts"]["NO_CHANGE"], 1)
        self.assertEqual(status["overall_status"], "PARTIAL")

    def test_smoke_execution_is_separate_until_exact_protocol_promotion(self):
        questions = self.prepared({"source_train": 2})
        smoke = self.study / "smoke"
        smoke.mkdir()
        (smoke / "pairs.jsonl").write_text(json.dumps(self.pair(questions[0])) + "\n", encoding="utf-8")
        self.write_json("smoke/collection_status.json", dict(status="PARTIAL", completed_pairs=1, elapsed_seconds=60,
                                                            estimated_remaining_hours=1.5, protocol_hash="test-protocol-hash"))
        status = generate_status(self.project, self.study)
        self.assertEqual(status["collection"]["completed_pairs"], 0)
        self.assertEqual(status["smoke"]["completed_pairs"], 1)
        self.assertEqual(status["smoke"]["promoted_unique_pairs"], 0)
        self.assertEqual(status["stages"]["evaluation"]["status"], "NOT_RUN")
        self.assertEqual(status["collection_eta"]["estimated_remaining_seconds"], 5400)
        self.assertIn("functional smoke", status["collection_eta"]["source"])
        self.write_rows("pairs.jsonl", [self.pair(questions[0])])
        promoted = generate_status(self.project, self.study)
        self.assertEqual(promoted["collection"]["completed_pairs"], 1)
        self.assertEqual(promoted["smoke"]["promoted_unique_pairs"], 1)

    def test_resolved_and_cancelled_events_are_terminal_without_successful_call_claims(self):
        question = self.prepared({"source_train": 1})[0]
        events = []
        for identifier, final_status in (("resolved", "resolved_uncertain"), ("cancelled", "cancelled_no_request")):
            shared = dict(example_id=question["example_id"], phase="initial", call_id=identifier,
                          attempt=2, protocol_hash="test-protocol-hash")
            events.extend([dict(shared, status="started"), dict(shared, status=final_status)])
        self.write_rows("calls.jsonl", events)
        status = generate_status(self.project, self.study)
        self.assertEqual(status["collection"]["unfinished_calls"], 0)
        self.assertEqual(status["collection"]["cancelled_no_request_calls"], 1)
        self.assertEqual(status["collection"]["resolved_uncertain_calls"], 1)
        self.assertEqual(status["collection"]["retry_calls"], 1)
        self.assertEqual(status["collection"]["call_accounting"]["successful_calls"], 0)
        self.assertEqual(status["collection"]["missing_initial_calls"], 1)

    def test_whole_run_elapsed_is_separate_from_measured_latest_session(self):
        self.prepared({"source_train": 1})
        self.write_json("run_context.json", dict(started_at="2026-10-05T00:00:00+00:00", ended_at="2026-10-05T00:00:10+00:00"))
        self.write_json("collection_status.json", dict(elapsed_seconds=4))
        completed = generate_status(self.project, self.study)
        self.assertEqual(completed["execution"]["elapsed_seconds"], 10)
        self.assertEqual(completed["execution"]["measured_latest_collection_session_seconds"], 4)
        self.write_json("run_context.json", dict(started_at="2026-10-01T00:00:00+00:00", ended_at=None,
                                                active_processes=[{"pid": 123}]))
        active = generate_status(self.project, self.study)
        from datetime import datetime
        expected = (datetime.fromisoformat(active["generated_at"]) - datetime.fromisoformat("2026-10-01T00:00:00+00:00")).total_seconds()
        self.assertEqual(active["execution"]["elapsed_seconds"], expected)
        self.assertIsNone(active["execution"]["ended_at"])

    def test_blocked_collector_remains_visible_in_stage_status(self):
        self.prepared({"source_train": 1})
        self.write_json("collection_status.json", dict(status="BLOCKED", errors=["An uncertain initial call requires explicit resolution."]))
        status = generate_status(self.project, self.study)
        self.assertEqual(status["overall_status"], "PARTIAL")
        self.assertEqual(status["stages"]["collection"]["status"], "BLOCKED")
        self.assertIn("uncertain initial call", status["stages"]["collection"]["reason"])

    def test_failed_shared_pair_audit_blocks_genuine_counts_and_keeps_raw_evidence(self):
        question = self.prepared({"source_train": 1})[0]
        self.write_rows("pairs.jsonl", [self.pair(question, "HURT", base_actual_tokens=999999)])
        self.audit.side_effect = ValueError("Pair base_actual_tokens differs from its actual successful call.")
        status = generate_status(self.project, self.study)
        self.assertEqual(status["collection"]["raw_pair_rows"], 1)
        self.assertEqual(status["collection"]["unverified_pair_rows"], 1)
        self.assertEqual(status["collection"]["completed_pairs"], 0)
        self.assertEqual(status["collection"]["class_counts"]["HURT"], 0)
        self.assertEqual(status["collection"]["pair_audit"]["status"], "FAILED")
        self.assertEqual(status["stages"]["collection"]["status"], "BLOCKED")
        self.assertTrue(any("base_actual_tokens" in row.get("error", "") and row.get("traceback") for row in status["errors"]))

    def test_complete_requires_current_metrics_and_no_active_execution(self):
        targets = dict.fromkeys(DEFAULT_TARGETS, 1)
        targets["source_train"] = 3
        questions = self.prepared(targets)
        labels = iter(OUTCOMES)
        pairs = [self.pair(question, next(labels) if question["partition"] == "source_train" else "NO_CHANGE") for question in questions]
        self.write_rows("pairs.jsonl", pairs)
        self.write_json("training_status.json", dict(training={"status": "COMPLETE"}, adaptation={"status": "COMPLETE"}, evaluation={"status": "COMPLETE"}))
        self.write_json("run_context.json", dict(live_integration={"status": "COMPLETE"}))
        missing = generate_status(self.project, self.study)
        self.assertEqual(missing["overall_status"], "PARTIAL")
        self.assertEqual(missing["stages"]["evaluation"]["status"], "NOT_RUN")
        self.write_json("reports/predictor_diagnostics.json", dict(target={"questions": 1, "macro_f1": .25}))
        (self.study / "reports" / "policy_results.csv").write_text("policy,questions\nCompletion rule,1\n", encoding="utf-8")
        complete = generate_status(self.project, self.study)
        self.assertEqual(complete["overall_status"], "COMPLETE")
        self.assertEqual(complete["training_evidence"]["metrics"]["target"]["questions"], 1)
        self.write_json("run_context.json", dict(live_integration={"status": "COMPLETE"}, active_processes=[{"pid": 123, "status": "running"}]))
        self.assertEqual(generate_status(self.project, self.study)["overall_status"], "PARTIAL")

    def test_malformed_journal_is_reported_without_losing_handoff(self):
        self.prepared({"source_train": 1})
        (self.study / "calls.jsonl").write_text('{"status":"started","call_id":"one"}\n{"partial":\n', encoding="utf-8")
        status = generate_status(self.project, self.study)
        self.assertEqual(status["overall_status"], "PARTIAL")
        self.assertTrue(any(row.get("line") == 2 for row in status["errors"]))
        self.assertTrue(Path(status["diagnostic_bundle"]["path"]).is_file())

    def test_jsonl_preserves_unicode_separator_inside_question_text(self):
        questions = self.prepared({"source_train": 1})
        questions[0]["query"] = "First sentence\u2028Second sentence\u2029Third sentence"
        (self.study / "questions.jsonl").write_text(json.dumps(questions[0], ensure_ascii=False) + "\n", encoding="utf-8")
        status = generate_status(self.project, self.study)
        self.assertEqual(status["collection"]["manifest_unique_questions"], 1)
        self.assertFalse(any(row.get("file", "").endswith("questions.jsonl") for row in status["errors"]))

    def test_bundle_excludes_keys_bulk_weights_credentials_and_large_artifacts(self):
        question = self.prepared({"source_train": 4})[0]
        self.write_rows("pairs.jsonl", [self.pair(dict(question, example_id=f"source_train:{index}"), "HELP", gold_answer="DO_NOT_INCLUDE_KEY") for index in range(4)])
        self.write_rows("trusted_keys.jsonl", [dict(example_id=question["example_id"], gold_answer="DO_NOT_INCLUDE_KEY")])
        self.write_json("run_context.json", dict(api_key="DO_NOT_INCLUDE_SECRET", commands=[{"command": "diagnostic", "exit_code": 1}]))
        logs = self.study / "logs"
        logs.mkdir()
        (logs / "run.txt").write_text("API_KEY=DO_NOT_INCLUDE_SECRET\nAuthorization: Bearer DO_NOT_INCLUDE_SECRET\nReal failure evidence", encoding="utf-8")
        (self.study / "raw").mkdir()
        (self.study / "raw" / "bulk.parquet").write_bytes(b"bulk fixture")
        (self.study / "models").mkdir()
        (self.study / "models" / "source_gate.joblib").write_bytes(b"x" * (2 * 1024 * 1024 + 1))
        status = generate_status(self.project, self.study)
        self.assertTrue(status["artifact"]["sha256"])
        with zipfile.ZipFile(status["diagnostic_bundle"]["path"]) as bundle:
            names = bundle.namelist()
            self.assertFalse(any("trusted_keys.jsonl" in name or "/raw/" in name or name.endswith(".joblib") for name in names))
            combined = b"\n".join(bundle.read(name) for name in names)
            self.assertNotIn(b"DO_NOT_INCLUDE_KEY", combined)
            self.assertNotIn(b"DO_NOT_INCLUDE_SECRET", combined)
            self.assertIn(b"Real failure evidence", combined)
            examples = [json.loads(line) for line in bundle.read("examples/real_records.jsonl").decode().splitlines()]
            self.assertEqual(len(examples), 2)
        self.assertLess(Path(status["diagnostic_bundle"]["path"]).stat().st_size, 10 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
