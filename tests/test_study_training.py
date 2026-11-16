"""Synthetic fixtures exercise safeguards; they are never research records."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from rechecklab.data import LABELS, question_hash, transition_label
from rechecklab.features import FEATURE_NAMES, extract_features
from rechecklab.predictor import MODEL_SELECTION_RULE, OutcomePredictor, load_artifact
from rechecklab.study_protocol import create_protocol, freeze_protocol, grade_answer, normalize_call
from rechecklab.study_collection import build_pair
from rechecklab.study_training import PARTITIONS, PLAN_THRESHOLD_OBJECTIVE, train_study, validate_study_inputs
from rechecklab.study_validation import PRECISION_LIMITATION


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def freeze_test_plan(root, counts):
    plan = {"data_schema_version": "live-study-questions-v1", "seed": 42, "planned_questions_by_partition": counts,
            "synthetic_fixture_for_tests_only": True,
            "manifest_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                for name in ("questions.jsonl", "trusted_keys.jsonl")}}
    identity = {key: plan[key] for key in ("data_schema_version", "seed", "planned_questions_by_partition", "manifest_sha256")}
    plan["study_plan_hash"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    (root / "study_plan.json").write_text(json.dumps(plan), encoding="utf-8")


def fixture(root, counts=None):
    counts = counts or dict(zip(PARTITIONS, (15, 3, 3, 4, 3)))
    backend = {"installed_model": {"name": "qwen3:4b", "digest": "test-only-digest"}, "ollama_version": "test-only"}
    with patch("rechecklab.study_protocol.inspect_backend", return_value=backend):
        protocol = create_protocol()
    digest = freeze_protocol(root, protocol)
    questions, pairs, keys, calls = [], [], [], []
    for partition, count in counts.items():
        source = partition.startswith("source_")
        for index in range(count):
            label = LABELS[index % 3]
            query = f"SYNTHETIC UNIT TEST {partition} {index}\u2028internal separator?"
            question = {"example_id": f"{partition}-{index}", "query": query, "group_id": question_hash(query),
                        "partition": partition, "domain": "maths" if source else "commonsenseqa",
                        "dataset": "gsm8k" if source else "commonsenseqa", "answer_type": "number" if source else "choice",
                        "source_dataset": "openai/gsm8k" if source else "tau/commonsense_qa", "source_revision": "0" * 40,
                        "source_split": "test" if partition == "source_test" else "validation" if partition == "target_test" else "train",
                        "source_question_id": str(index)}
            questions.append(question)
            correct, wrong = ("1", "0") if source else ("A", "B")
            key = {"example_id": question["example_id"], "answer_type": question["answer_type"], "gold_answer": correct}
            keys.append(key)
            events = []
            for phase, is_correct in (("initial", label != "HELP"), ("recheck", label != "HURT")):
                response = {"model": protocol["model"], "done": True,
                            "message": {"content": f"Synthetic test reasoning {index}.\nFINAL: {correct if is_correct else wrong}"},
                            "done_reason": "length" if index % 2 else "stop", "wall_seconds": 2.0 if phase == "initial" else 3.0,
                            "prompt_eval_count": 3 if phase == "initial" else 5, "eval_count": 7 + index}
                _, output = normalize_call(protocol, query, question["answer_type"], response)
                event = {"example_id": question["example_id"], "phase": phase, "protocol_hash": digest,
                         "group_id": question["group_id"], "partition": partition,
                         "status": "succeeded", "seed": protocol["generation"][phase + "_seed"],
                         "call_id": f"{question['example_id']}-{phase}", "response": response, "output": output}
                events.append(event)
                calls.append({key: value for key, value in {**event, "status": "started"}.items() if key not in ("response", "output")})
                calls.append(event)
            pairs.append(build_pair(question, key, events[0], events[1], protocol, digest))
    write_jsonl(root / "questions.jsonl", questions)
    write_jsonl(root / "pairs.jsonl", pairs)
    write_jsonl(root / "trusted_keys.jsonl", keys)
    write_jsonl(root / "calls.jsonl", calls)
    freeze_test_plan(root, counts)
    return questions, pairs


def fitted_test_classifier(train, dev, seed=42, return_candidates=False):
    estimator = LogisticRegression(random_state=seed, max_iter=300).fit(extract_features(train), train.label)
    predictor = OutcomePredictor(estimator, "test_real_logistic")
    selection = [{"model": predictor.model_name, "selected": True, "source_dev_metrics": {"macro_f1": 0.0}}]
    return (predictor, selection, {predictor.model_name: predictor})


class StudyTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.questions, self.pairs = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def run_study(self, **kwargs):
        with patch("rechecklab.study_training.fit_predictor", side_effect=fitted_test_classifier):
            return train_study(self.root, adaptation_budgets=(2,), adaptation_seeds=(142, 143), bootstrap_repeats=30, **kwargs)

    def frozen_selection_plan(self):
        declaration = {"minimum_training_examples_per_class": 5, "seed": 42,
                       "feature_schema": "pre-recheck-v2", "feature_count": 19,
                       "model_selection_rule": MODEL_SELECTION_RULE, "threshold_objective": PLAN_THRESHOLD_OBJECTIVE,
                       "adaptation_budgets": [2], "adaptation_seeds": [142, 143], "probabilities": "uncalibrated estimates",
                       "heldout_rule": "no final evaluation until all planned held-out and adaptation pairs are collected"}
        (self.root / "model_selection_plan.json").write_text(json.dumps(declaration), encoding="utf-8")

    def test_frozen_source_settings_cannot_change_after_outcomes(self):
        self.frozen_selection_plan()
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root, seed=43, adaptation_budgets=(2,), adaptation_seeds=(142, 143))
        fit.assert_not_called()
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("model_selection_plan: seed", status["error"])

    def test_declared_adaptation_collection_must_finish_before_final_evaluation(self):
        self.frozen_selection_plan()
        write_jsonl(self.root / "pairs.jsonl", [row for row in self.pairs if row["example_id"] != "target_adaptation_pool-3"])
        status = self.run_study()
        self.assertEqual(status["training"]["status"], "COMPLETE")
        self.assertEqual(status["evaluation"]["status"], "NOT_RUN")
        self.assertIn("target_adaptation_pool has 3/4", " ".join(status["evaluation"]["not_run_reasons"]))

    def test_missing_rare_class_prevents_any_fit_and_reports_exact_support(self):
        incomplete = [row for row in self.pairs if not (row["example_id"].startswith("source_train") and row["label"] == "HURT")]
        write_jsonl(self.root / "pairs.jsonl", incomplete)
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertEqual(status["status"], "INCOMPLETE")
        self.assertEqual(status["source_train_class_counts"], {"HELP": 5, "NO_CHANGE": 5, "HURT": 0})
        self.assertEqual(status["training"]["inadequate_classes"], {"HURT": 0})
        self.assertEqual(status["evaluation"]["status"], "NOT_RUN")
        self.assertEqual(status["adaptation"]["status"], "NOT_RUN")
        self.assertFalse((self.root / "models" / "source_gate.joblib").exists())
        self.assertTrue((self.root / "training_status.json").exists())
        self.assertIn(PRECISION_LIMITATION, status["limitations"])
        self.assertIn("exclude failed/retried call work", " ".join(status["limitations"]))

    def test_insufficient_class_cannot_be_hidden_by_lowering_minimum(self):
        status = self.run_study(minimum_per_class=1)
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("at least five", status["error"])

    def test_nonempty_development_required_before_fit(self):
        write_jsonl(self.root / "pairs.jsonl", [row for row in self.pairs if not row["example_id"].startswith("source_development")])
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertIn("No completed source-development", " ".join(status["training"]["not_run_reasons"]))

    def test_heldout_completion_blocks_evaluation_even_after_valid_fit(self):
        write_jsonl(self.root / "pairs.jsonl", [row for row in self.pairs if row["example_id"] != "target_test-2"])
        status = self.run_study()
        self.assertEqual(status["training"]["status"], "COMPLETE")
        self.assertEqual(status["evaluation"]["status"], "NOT_RUN")
        self.assertIn("2/3", " ".join(status["evaluation"]["not_run_reasons"]))
        self.assertTrue(status["training"]["artifact_roundtrip_verified"])
        self.assertFalse((self.root / "reports" / "policy_results.csv").exists())

    def test_partial_source_training_explicit_and_never_activates_live(self):
        # Plan has 18 train questions but 15 completed preserve all classes>=5.
        counts = dict(zip(PARTITIONS, (18, 3, 3, 4, 3)))
        _, pairs = fixture(self.root, counts)
        write_jsonl(self.root / "pairs.jsonl", [row for row in pairs if row["example_id"] not in {"source_train-15", "source_train-16", "source_train-17"}])
        status = self.run_study()
        self.assertEqual(status["status"], "PARTIAL")
        self.assertEqual(status["training"]["status"], "PARTIAL")
        self.assertTrue(status["evaluation"]["conditional_on_partial_source_training"])
        self.assertFalse(status["artifact"]["live_activation_eligible"])

    def test_complete_roundtrip_probabilities_and_completion_paired_metrics(self):
        status = self.run_study()
        self.assertEqual(status["status"], "COMPLETE", status.get("error"))
        bundle = load_artifact(self.root / status["artifact"]["path"], expected_protocol=status["protocol_contract"], expected_sha256=status["artifact"]["sha256"])
        self.assertEqual(bundle["feature_names"], FEATURE_NAMES)
        self.assertEqual(bundle["class_order"], LABELS)
        self.assertEqual(set(bundle["predictor"].estimator.classes_), set(LABELS))
        self.assertEqual(bundle["threshold"], status["training"]["threshold"])
        policies = json.loads((self.root / "reports" / "policy_results.json").read_text())
        completion = [row for row in policies if row["policy"] == "completion_rule"]
        self.assertTrue(all(row["accuracy_difference_pp_vs_completion"] == 0 for row in completion))
        self.assertTrue(all(row["accuracy_difference_95pct_ci_vs_completion"] == [0.0, 0.0] for row in completion))
        self.assertTrue(all(row["average_reconstructed_latency_seconds"] == 3.0 for row in completion))
        self.assertEqual(status["adaptation"]["completed_runs"], 2)
        runs = json.loads((self.root / "reports" / "adaptation_runs.json").read_text())
        for run in runs:
            self.assertEqual(sum(run["adaptation_class_counts"].values()), 2)
            self.assertFalse(run["classifier_updated"])
            self.assertFalse(run["probabilities_recalibrated"])
            self.assertEqual(len(run["adaptation_example_ids"]), 2)

    def test_production_five_candidate_selection_and_configuration_roundtrip(self):
        # Real fitted sklearn candidates, using test-only pairs in a temp root.
        with threadpool_limits(limits=1):
            status = train_study(self.root, adaptation_budgets=(2,), adaptation_seeds=(142, 143), bootstrap_repeats=10)
        self.assertEqual(status["status"], "COMPLETE", status.get("error"))
        selected = json.loads((self.root / "reports" / "model_selection.json").read_text())
        self.assertEqual([row["model"] for row in selected], ["majority_class", "logistic_regression", "logistic_regression_balanced",
                                                              "hist_gradient_boosting", "hist_gradient_boosting_balanced"])
        self.assertEqual(sum(row["selected"] for row in selected), 1)
        self.assertTrue(all("source_dev_metrics" in row for row in selected))
        configurations = json.loads((self.root / "reports" / "model_configurations.json").read_text())
        self.assertEqual(set(configurations), {row["model"] for row in selected})
        self.assertEqual(configurations["logistic_regression"]["parameters"]["steps"][0][1]["class"],
                         "sklearn.preprocessing._data.StandardScaler")
        self.assertEqual(configurations["hist_gradient_boosting_balanced"]["parameters"]["random_state"], 42)
        self.assertTrue(status["training"]["artifact_roundtrip_verified"])

    def test_target_test_labels_cannot_change_source_artifact_or_threshold(self):
        first = self.run_study()
        original_digest = first["artifact"]["sha256"]
        changed = json.loads(json.dumps(self.pairs))
        keys = [json.loads(line) for line in (self.root / "trusted_keys.jsonl").read_text().split("\n") if line]
        for key in keys:
            if key["example_id"].startswith("target_test"):
                key["gold_answer"] = "B"
        write_jsonl(self.root / "trusted_keys.jsonl", keys)
        freeze_test_plan(self.root, dict(zip(PARTITIONS, (15, 3, 3, 4, 3))))
        for row in changed:
            if row["example_id"].startswith("target_test"):
                row["base_correct"] = grade_answer(row["base_answer"], "B", "choice")
                row["final_correct"] = grade_answer(row["final_answer"], "B", "choice")
                row["label"] = transition_label(row["base_correct"], row["final_correct"])
        write_jsonl(self.root / "pairs.jsonl", changed)
        second = self.run_study()
        self.assertEqual(second["artifact"]["sha256"], original_digest)
        self.assertEqual(first["training"]["threshold"], second["training"]["threshold"])
        self.assertEqual(first["training"]["source_fit_content_sha256"], second["training"]["source_fit_content_sha256"])

    def test_insufficient_adaptation_budgets_get_not_run_reasons(self):
        with patch("rechecklab.study_training.fit_predictor", side_effect=fitted_test_classifier):
            status = train_study(self.root, bootstrap_repeats=10)
        self.assertEqual(status["evaluation"]["status"], "COMPLETE")
        self.assertEqual(status["adaptation"]["status"], "NOT_RUN")
        self.assertEqual(status["adaptation"]["planned_runs"], 20)
        self.assertEqual(status["adaptation"]["completed_runs"], 0)
        self.assertIn("Only 4", status["adaptation"]["not_run_reasons"][0])

    def test_invalid_pair_metadata_or_label_fail_closed(self):
        for field, value, phrase in (("base_eval_count", 2.5, "base_eval_count"), ("label", "HURT", "transition"),
                                     ("model_digest", "other", "model_digest"), ("base_actual_tokens", 99, "base_actual_tokens")):
            with self.subTest(field=field):
                changed = json.loads(json.dumps(self.pairs))
                changed[0][field] = value
                write_jsonl(self.root / "pairs.jsonl", changed)
                with patch("rechecklab.study_training.fit_predictor") as fit:
                    status = train_study(self.root)
                fit.assert_not_called()
                self.assertEqual(status["status"], "FAILED")
                self.assertIn(phrase, status["error"])

    def test_labels_without_actual_successful_call_evidence_rejected(self):
        (self.root / "calls.jsonl").write_text("", encoding="utf-8")
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("successful initial call journal", status["error"])

    def test_terminal_journal_cannot_switch_question_after_started_call(self):
        events = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().split("\n") if line]
        events[1]["example_id"] = events[-1]["example_id"]
        write_jsonl(self.root / "calls.jsonl", events)
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertIn("journal", status["error"])

    def test_cancelled_unstarted_request_preserves_verified_pairs_read_only(self):
        events = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().split("\n") if line]
        extra = {**events[0], "call_id": "synthetic-no-request", "status": "started"}
        events.extend([extra, {**extra, "status": "cancelled_no_request", "reason": "synthetic timeout before HTTP call"}])
        write_jsonl(self.root / "calls.jsonl", events)
        before = {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        _, _, _, manifest, pairs, _, _ = validate_study_inputs(self.root)
        self.assertEqual(len(manifest), len(self.questions))
        self.assertEqual(len(pairs), len(self.pairs))
        self.assertFalse((self.root / "training_status.json").exists())
        self.assertEqual(before, {path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()})
        self.assertEqual(pairs.attrs["numeric_precision_audit"]["successful_call_status_counts"],
                         {"EXACT": 42, "UNPARSEABLE": 0, "NOT_APPLICABLE": 14})

    def test_inexact_numeric_evidence_rejected_before_any_fit_without_relabeling(self):
        events = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().split("\n") if line]
        events[1]["response"]["message"]["content"] = "FINAL: 1.00000000000000000000000000001"
        write_jsonl(self.root / "calls.jsonl", events)
        unchanged = {name: (self.root / name).read_bytes() for name in ("pairs.jsonl", "protocol.json", "protocol_hash.txt")}
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("precision loss", status["error"])
        self.assertEqual(unchanged, {name: (self.root / name).read_bytes() for name in unchanged})

    def test_precision_guard_also_checks_successful_calls_without_completed_pairs(self):
        (self.root / "pairs.jsonl").write_text("", encoding="utf-8")
        events = [json.loads(line) for line in (self.root / "calls.jsonl").read_text().split("\n") if line]
        events[1]["response"]["message"]["content"] = "FINAL: 1/3"
        write_jsonl(self.root / "calls.jsonl", events)
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertEqual(status["status"], "FAILED")
        self.assertIn("nonterminating", status["error"])

    def test_missing_frozen_manifest_hashes_cannot_bypass_validation(self):
        plan = json.loads((self.root / "study_plan.json").read_text())
        del plan["manifest_sha256"]
        (self.root / "study_plan.json").write_text(json.dumps(plan), encoding="utf-8")
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertIn("missing its selection/hash declaration", status["error"])

    def test_fabricated_rare_class_correctness_rejected_by_offline_key(self):
        changed = json.loads(json.dumps(self.pairs))
        changed[0]["base_correct"], changed[0]["final_correct"], changed[0]["label"] = True, False, "HURT"
        write_jsonl(self.root / "pairs.jsonl", changed)
        with patch("rechecklab.study_training.fit_predictor") as fit:
            status = train_study(self.root)
        fit.assert_not_called()
        self.assertIn("strict offline grading", status["error"])

    def test_manifest_group_leak_and_silent_reduction_rejected(self):
        questions = json.loads(json.dumps(self.questions))
        questions[-1]["query"] = questions[0]["query"]
        questions[-1]["group_id"] = questions[0]["group_id"]
        write_jsonl(self.root / "questions.jsonl", questions)
        self.assertIn("Duplicate question", self.run_study()["error"])
        write_jsonl(self.root / "questions.jsonl", self.questions[:-1])
        self.assertIn("declared plan", self.run_study()["error"])


if __name__ == "__main__":
    unittest.main()
