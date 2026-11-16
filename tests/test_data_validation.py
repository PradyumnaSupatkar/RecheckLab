"""Focused input/protocol safeguards; fixtures are never research results."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from rechecklab.data import (
    COLLECTION_SETTINGS, assert_disjoint, collection_protocol, group_partition, load_pairs, public_frames,
    question_hash, transition_label, validate_pairs, validate_transfer_compatibility,
)
from rechecklab.features import extract_features


def records(prefix="source", count=15):
    return pd.DataFrame([
        dict(example_id=str(i), dataset=prefix, query=f"{prefix} question {i}",
             base_response="Reasoning. FINAL: 3", base_answer="3", base_correct=i % 2 == 0,
             final_correct=i % 3 == 0, base_actual_tokens=20, action_actual_tokens=10,
             base_done_reason="stop", base_finalizer_used=False, model="test-only-model",
             action="recheck", recheck_protocol="test-only-protocol", rollout=0)
        for i in range(count)
    ])


def metadata():
    return dict(model="test-only-model", model_digest="test-only-digest", host="http://localhost:11434",
                protocol_version="0.1", budget=768, seed=42, temperature=0.0, thinking=False,
                initial_instruction="Solve. FINAL:", recheck_instruction="Recheck. FINAL:")


class DataValidation(unittest.TestCase):
    def test_all_four_transitions_and_strict_correctness(self):
        for before, after, expected in ((False, True, "HELP"), (True, False, "HURT"),
                                        (False, False, "NO_CHANGE"), (True, True, "NO_CHANGE")):
            self.assertEqual(transition_label(before, after), expected)
        for value in (None, np.nan, "unknown", 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                transition_label(value, True)

    def test_required_initial_and_evaluation_metadata_never_defaults(self):
        frame = records()
        for key in ("base_answer", "base_actual_tokens", "base_done_reason", "base_finalizer_used",
                    "base_correct", "final_correct", "action_actual_tokens", "action", "model", "recheck_protocol"):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_pairs(frame.drop(columns=key))
        for key, value in (("base_actual_tokens", np.nan), ("base_actual_tokens", np.inf),
                           ("action_actual_tokens", -1), ("action_actual_tokens", 1.5),
                           ("action_actual_tokens", True), ("base_done_reason", ""),
                           ("base_finalizer_used", "maybe"), ("query", 123), ("base_answer", None)):
            changed = frame.copy()
            changed[key] = changed[key].astype(object)
            changed.loc[0, key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_pairs(changed)

    def test_explicit_missing_answer_and_unknown_status_are_supported(self):
        frame = records()
        frame["base_answer"] = ""
        frame["base_done_reason"] = "unknown"
        validated = validate_pairs(frame)
        features = extract_features(validated)
        self.assertTrue(features.initial_answer_present.eq(0).all())
        self.assertTrue(features.initial_completion_unknown.eq(1).all())

    def test_consistency_covers_initial_answer_completion_finalizer_and_query(self):
        frame = records()
        for key, value in (("base_answer", "different"), ("base_done_reason", "length"),
                           ("base_finalizer_used", True), ("query", "another question")):
            repeated = frame.iloc[:1].copy()
            repeated["rollout"] = 1
            repeated.loc[repeated.index[0], key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "inconsistent"):
                validate_pairs(pd.concat([frame, repeated], ignore_index=True))

    def test_canonical_question_groups_ignore_case_unicode_and_whitespace(self):
        self.assertEqual(question_hash("  STRASSE  café? "), question_hash("Straße cafe\u0301?"))
        frame = records(count=20)
        repeated = frame.iloc[:3].copy()
        repeated["query"] = repeated["query"].map(lambda q: "  " + q.upper().replace(" ", "  ") + "\n")
        repeated["rollout"] = 1
        combined = validate_pairs(pd.concat([frame, repeated], ignore_index=True))
        parts = group_partition(combined, (.6, .2, .2), 42)
        assert_disjoint(*parts)
        self.assertEqual(sum(len(part) for part in parts), len(combined))
        for group in repeated["query"].map(question_hash):
            self.assertEqual(sum(group in set(part.group_id) for part in parts), 1)

    def test_overlap_checks_recompute_hashes_instead_of_trusting_cached_ids(self):
        source, target = records(), records("target")
        target.loc[0, "query"] = source.loc[0, "query"].upper()
        source["group_id"], target["group_id"] = "forged-source", "forged-target"
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint(source, target)

    def test_model_protocol_and_checkpoint_must_match(self):
        source, target = validate_pairs(records()), validate_pairs(records("target"))
        identity = validate_transfer_compatibility(source, target)
        self.assertEqual(identity["model"], "test-only-model")
        for key in ("model", "recheck_protocol", "action"):
            changed = target.copy()
            changed[key] = "different"
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "compatibility"):
                validate_transfer_compatibility(source, changed)
        source["model_digest"] = "digest-a"
        with self.assertRaisesRegex(ValueError, "model_digest"):
            validate_transfer_compatibility(source, target)
        target["model_digest"] = "digest-b"
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            validate_transfer_compatibility(source, target)

    def test_non_recheck_or_mixed_actions_are_rejected(self):
        frame = records()
        frame["action"] = "accept"
        with self.assertRaisesRegex(ValueError, "recheck"):
            validate_pairs(frame)
        frame.loc[0, "action"] = "recheck"
        with self.assertRaisesRegex(ValueError, "fixed action"):
            validate_pairs(frame)

    def test_generation_metadata_is_checked_in_both_transfer_directions(self):
        source, target = validate_pairs(records()), validate_pairs(records("target"))
        target.attrs["generation_settings"] = collection_protocol(metadata())[1]
        for first, second in ((source, target), (target, source)):
            with self.assertRaisesRegex(ValueError, "generation settings"):
                validate_transfer_compatibility(first, second)
        source.attrs["generation_settings"] = target.attrs["generation_settings"]
        identity = validate_transfer_compatibility(source, target)
        self.assertEqual(identity["generation_settings"]["budget"], 768)

    def test_collector_sidecar_binds_protocol_and_retains_source_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collected.jsonl"
            rows = records().drop(columns="recheck_protocol")
            rows["model_digest"] = "test-only-digest"
            rows.to_json(path, orient="records", lines=True)
            original = path.read_bytes()
            path.with_suffix(".metadata.json").write_text(json.dumps(metadata()), encoding="utf-8")
            loaded = load_pairs(path)
            self.assertTrue(loaded.recheck_protocol.str.startswith("ollama:").all())
            self.assertEqual(loaded.attrs["generation_settings"]["budget"], 768)
            self.assertEqual(path.read_bytes(), original)
            rows["model"] = "different-model"
            rows.to_json(path, orient="records", lines=True)
            with self.assertRaisesRegex(ValueError, "conflicts"):
                load_pairs(path)

    def test_protocol_hash_excludes_host_and_detects_generation_changes(self):
        base = metadata()
        changed = {**base, "host": "http://other-host:11434"}
        self.assertEqual(collection_protocol(base)[0], collection_protocol(changed)[0])
        changed["budget"] = 1024
        self.assertNotEqual(collection_protocol(base)[0], collection_protocol(changed)[0])
        for key, value in (("thinking", "false"), ("temperature", np.nan), ("budget", 0)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                collection_protocol({**base, key: value})

    def test_optional_context_preserves_legacy_hash_and_changes_explicit_protocol(self):
        base = metadata()
        legacy = {key: base[key] for key in COLLECTION_SETTINGS}
        expected = "ollama:" + hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":"),
                                                          allow_nan=False).encode()).hexdigest()
        self.assertEqual(collection_protocol(base)[0], expected)
        self.assertNotIn("context_window", collection_protocol(base)[1])
        explicit = {**base, "context_window": 4096}
        self.assertNotEqual(collection_protocol(explicit)[0], expected)
        self.assertEqual(collection_protocol(explicit)[1]["context_window"], 4096)
        self.assertNotEqual(collection_protocol(explicit)[0], collection_protocol({**explicit, "context_window": 8192})[0])
        for invalid in (None, 0, -1, True, 4096.5, "4096", np.inf):
            with self.subTest(context=invalid), self.assertRaisesRegex(ValueError, "context_window"):
                collection_protocol({**base, "context_window": invalid})

    def test_answer_and_identifier_text_survives_file_loading(self):
        frame = records(count=1)
        frame["example_id"], frame["base_answer"] = "001", "003"
        with tempfile.TemporaryDirectory() as directory:
            for extension in ("csv", "jsonl"):
                path = Path(directory) / ("records." + extension)
                if extension == "csv":
                    frame.to_csv(path, index=False)
                else:
                    frame.to_json(path, orient="records", lines=True)
                loaded = load_pairs(path)
                with self.subTest(extension=extension):
                    self.assertEqual(loaded.loc[0, "example_id"], "001")
                    self.assertEqual(loaded.loc[0, "base_answer"], "003")

    def test_public_empty_completion_status_is_explicit_and_raw_frame_unchanged(self):
        source, target = records(), records("target")
        source["action"], target["action"] = "active_verify", "active_verify"
        source["model"], target["model"] = "qwen3:4b", "qwen3:4b"
        source.loc[0, "base_done_reason"] = ""
        with patch("rechecklab.data.pd.read_parquet", side_effect=[source, target]):
            prepared_source, prepared_target = public_frames(Path("test-only-root"))
        self.assertEqual(source.loc[0, "base_done_reason"], "")
        self.assertEqual(prepared_source.loc[0, "base_done_reason"], "unknown")
        self.assertEqual(prepared_source.attrs["unknown_initial_completion_rows"], 1)
        identity = validate_transfer_compatibility(prepared_source, prepared_target)
        self.assertEqual(identity["published_generation_settings"]["base_generation_limit"], 4096)


if __name__ == "__main__":
    unittest.main()
