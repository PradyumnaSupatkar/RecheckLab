"""Frozen benchmark-boundary/key-separation safeguards; fixtures are not results."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from rechecklab.study_data import (
    BENCHMARKS, benchmark_candidates, deduplicate_candidates, planned_counts, prepare_study,
    resolve_snapshot, select_questions, validate_question_manifest,
)


def fixture_frames():
    def maths(split, count):
        return pd.DataFrame([dict(question=f"{split} math problem {i}\u2028 Unit separator inside text.",
                                  answer=f"Worked solution, not a predictor input.\n#### {i + 2}")
                             for i in range(count)])

    def choices(split, count):
        return pd.DataFrame([dict(id=f"{split}-{i}", question=f"{split} common sense question {i}?",
            choices=dict(label=list("ABCDE"), text=[f"option {j}" for j in range(5)]), answerKey="B")
            for i in range(count)])

    return {"gsm8k": {"train": maths("train", 12), "test": maths("test", 4)},
            "commonsenseqa": {"train": choices("train", 6), "validation": choices("validation", 6)}}


def fixture_pools():
    frames = fixture_frames()
    return {dataset + "_" + split: benchmark_candidates(dataset, split, frame,
            ("a" if dataset == "gsm8k" else "b") * 40)
            for dataset, splits in frames.items() for split, frame in splits.items()}


def fake_snapshot_fetch():
    resources = {}
    revisions = {"gsm8k": "a" * 40, "commonsenseqa": "b" * 40}
    for dataset, splits in fixture_frames().items():
        repository = BENCHMARKS[dataset]["repository"]
        files = {"README.md": b"Unit fixture dataset card. MIT.\n"}
        for split, frame in splits.items():
            stream = io.BytesIO()
            frame.to_parquet(stream, index=False)
            files[BENCHMARKS[dataset]["files"][split]] = stream.getvalue()
        siblings = [dict(rfilename=name, lfs=dict(sha256=hashlib.sha256(content).hexdigest())) for name, content in files.items()]
        resources[f"https://huggingface.co/api/datasets/{repository}?blobs=true"] = json.dumps(dict(sha=revisions[dataset], siblings=siblings)).encode()
        for name, content in files.items():
            resources[f"https://huggingface.co/datasets/{repository}/resolve/{revisions[dataset]}/{name}?download=true"] = content

    def fetch(url, retries=3):
        if "/resolve/main/" in url:
            raise AssertionError("Unpinned benchmark download attempted.")
        return resources[url], 1

    return fetch


class StudyDataSafeguards(unittest.TestCase):
    def test_default_targets_and_reserved_original_splits(self):
        self.assertEqual(planned_counts(), dict(source_train=600, source_development=200,
            source_test=200, target_adaptation_pool=200, target_test=300))
        questions, keys, _, _ = select_questions(fixture_pools(), planned_counts(10, 5), 42)
        self.assertEqual(len(questions), 15)
        self.assertEqual(len(keys), 15)
        original_splits = {partition: {row["source_split"] for row in questions if row["partition"] == partition}
                           for partition in planned_counts(10, 5)}
        self.assertEqual(original_splits, dict(source_train={"train"}, source_development={"train"},
            source_test={"test"}, target_adaptation_pool={"train"}, target_test={"validation"}))
        self.assertEqual(len({row["group_id"] for row in questions}), 15)

    def test_selection_is_reproducible_and_independent_of_gold_labels(self):
        pools = fixture_pools()
        first = select_questions(pools, planned_counts(10, 5), 42)
        repeated = select_questions(pools, planned_counts(10, 5), 42)
        self.assertEqual(first, repeated)
        changed = copy.deepcopy(pools)
        for rows in changed.values():
            for row in rows:
                row["_gold_answer"] = "999" if row["answer_type"] == "number" else "E"
        questions, keys, _, _ = select_questions(changed, planned_counts(10, 5), 42)
        self.assertEqual(first[0], questions)
        self.assertNotEqual(first[1], keys)

    def test_canonical_duplicates_and_cross_boundary_overlap_are_removed_before_selection(self):
        pools = fixture_pools()
        repeated = copy.deepcopy(pools["gsm8k_train"][0])
        repeated["example_id"] = "different-within-pool-id"
        pools["gsm8k_train"].append(repeated)
        overlapping = copy.deepcopy(pools["gsm8k_train"][1])
        overlapping["example_id"] = "different-cross-split-id"
        pools["gsm8k_test"].append(overlapping)
        clean, audit = deduplicate_candidates(pools)
        self.assertEqual(audit["cross_boundary_overlapping_question_groups"], 1)
        self.assertEqual(audit["pools"]["gsm8k_train"]["removed_within_pool_duplicate_rows"], 1)
        self.assertNotIn(overlapping["group_id"], {row["group_id"] for rows in clean.values() for row in rows})
        questions, keys, _, _ = select_questions(pools, planned_counts(10, 5), 42)
        validate_question_manifest(questions, keys, planned_counts(10, 5))

    def test_conflicting_duplicate_gold_keys_are_rejected(self):
        pools = fixture_pools()
        repeated = copy.deepcopy(pools["gsm8k_train"][0])
        repeated["_gold_answer"] = "999"
        pools["gsm8k_train"].append(repeated)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            deduplicate_candidates(pools)

    def test_manifest_rejects_future_fields_forged_hashes_and_split_leakage(self):
        questions, keys, _, _ = select_questions(fixture_pools(), planned_counts(10, 5), 42)
        for field, value in (("gold_answer", "999"), ("base_correct", True), ("group_id", "forged"), ("source_split", "test")):
            changed = copy.deepcopy(questions)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_question_manifest(changed, keys)
        changed = copy.deepcopy(questions)
        changed[-1] = copy.deepcopy(changed[0])
        with self.assertRaisesRegex(ValueError, "Duplicate|overlapping"):
            validate_question_manifest(changed, keys)

    def test_snapshot_resolution_demands_full_commit_and_downloads_use_pinned_urls(self):
        with patch("rechecklab.study_data._fetch", return_value=(json.dumps(dict(sha="main", siblings=[])).encode(), 1)):
            with self.assertRaisesRegex(ValueError, "full commit"):
                resolve_snapshot("openai/gsm8k")
        with tempfile.TemporaryDirectory() as directory, patch("rechecklab.study_data._fetch", side_effect=fake_snapshot_fetch()) as network:
            plan = prepare_study(directory, 10, 5, 42)
            self.assertEqual(plan["datasets"]["gsm8k"]["revision"], "a" * 40)
            self.assertEqual(plan["datasets"]["commonsenseqa"]["revision"], "b" * 40)
            for call in network.call_args_list:
                if "/resolve/" in call.args[0]:
                    self.assertRegex(call.args[0], r"/resolve/[ab]{40}/")

    def test_preparation_is_offline_idempotent_and_unicode_jsonl_is_not_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rechecklab.study_data._fetch", side_effect=fake_snapshot_fetch()):
                first = prepare_study(root, 10, 5, 42)
            paths = [root / name for name in ("questions.jsonl", "trusted_keys.jsonl", "study_plan.json")]
            original = [path.read_bytes() for path in paths]
            self.assertIn("\u2028", paths[0].read_text(encoding="utf-8"))
            with patch("rechecklab.study_data._fetch", side_effect=AssertionError("Prepared resume must not require network")):
                self.assertEqual(prepare_study(root, 10, 5, 42), first)
            self.assertEqual([path.read_bytes() for path in paths], original)
            questions = [json.loads(line) for line in paths[0].read_text(encoding="utf-8").split("\n") if line]
            self.assertEqual(len(questions), 15)
            self.assertTrue(all("_gold_answer" not in row and "gold_answer" not in row for row in questions))
            self.assertNotIn("Worked solution", paths[0].read_text(encoding="utf-8"))

    def test_resume_rejects_changed_targets_or_any_manifest_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rechecklab.study_data._fetch", side_effect=fake_snapshot_fetch()):
                prepare_study(root, 10, 5, 42)
            with self.assertRaisesRegex(ValueError, "frozen"):
                prepare_study(root, 11, 5, 42)
            path = root / "questions.jsonl"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "altered"):
                prepare_study(root, 10, 5, 42)

    def test_plan_revision_and_selection_hash_cannot_disagree_with_questions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("rechecklab.study_data._fetch", side_effect=fake_snapshot_fetch()):
                prepare_study(root, 10, 5, 42)
            path = root / "study_plan.json"
            original = json.loads(path.read_text(encoding="utf-8"))
            changed = copy.deepcopy(original)
            changed["study_plan_hash"] = "forged"
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "selection hash"):
                prepare_study(root, 10, 5, 42)
            changed = copy.deepcopy(original)
            changed["datasets"]["gsm8k"]["revision"] = "c" * 40
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "provenance revision"):
                prepare_study(root, 10, 5, 42)

    def test_insufficient_or_public_baseline_output_is_never_silently_substituted(self):
        with self.assertRaisesRegex(ValueError, "too few"):
            select_questions(fixture_pools(), planned_counts(1000, 500), 42)
        with self.assertRaisesRegex(ValueError, "preserve"):
            prepare_study(Path(__file__).resolve().parents[1])


if __name__ == "__main__":
    unittest.main()
