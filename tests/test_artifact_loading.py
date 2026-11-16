"""Production save/load checks, including Streamlit module-reload identities."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from rechecklab.data import LABELS, collection_protocol
from rechecklab.features import FEATURE_NAMES, extract_features
from rechecklab.live import DEFAULT_SETTINGS, load_live_artifact
from rechecklab.predictor import (
    ArtifactLoadError, OutcomePredictor, load_artifact, make_artifact, save_artifact,
)

ROOT = Path(__file__).resolve().parents[1]


def initial_inputs():
    return pd.DataFrame([
        dict(query=f"Synthetic unit-test question {index}?",
             base_response=("Maybe" if index % 3 == 0 else "Check") + f" FINAL: {index}",
             base_answer=str(index), base_actual_tokens=20 + index * 5,
             base_done_reason="length" if index % 3 == 0 else "stop",
             base_finalizer_used=index % 3 == 2)
        for index in range(18)
    ])


def fitted_bundle():
    inputs = initial_inputs()
    estimator = LogisticRegression(max_iter=300, random_state=42).fit(
        extract_features(inputs), np.asarray(LABELS)[np.arange(len(inputs)) % len(LABELS)]
    )
    protocol_id, settings = collection_protocol(DEFAULT_SETTINGS)
    protocol = dict(model="synthetic-test-model", model_digest="synthetic-test-digest",
                    action="recheck", recheck_protocol=protocol_id, generation_settings=settings)
    return make_artifact(OutcomePredictor(estimator, "synthetic-fitted-logistic"), .123, protocol=protocol)


class ArtifactLoading(unittest.TestCase):
    def test_production_writer_and_live_loader_preserve_fitted_contract(self):
        original = fitted_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = save_artifact(original, Path(directory) / "models" / "gate.joblib")
            fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_live_artifact(path, expected_protocol=original["protocol"], expected_sha256=fingerprint)
            self.assertIsInstance(loaded, dict)
            self.assertIsInstance(loaded["predictor"], OutcomePredictor)
            self.assertEqual(loaded["predictor"].estimator.n_features_in_, 19)
            self.assertEqual(list(loaded["predictor"].estimator.feature_names_in_), FEATURE_NAMES)
            self.assertEqual(loaded["threshold"], original["threshold"])
            self.assertEqual(loaded["harm_weight"], original["harm_weight"])
            for key in ("artifact_schema_version", "feature_schema_version", "feature_schema_fingerprint",
                        "feature_names", "class_order", "protocol"):
                self.assertEqual(loaded[key], original[key])
            np.testing.assert_array_equal(loaded["predictor"].probabilities(initial_inputs()),
                                          original["predictor"].probabilities(initial_inputs()))

    def test_wrong_bundle_and_predictor_types_include_actionable_context(self):
        original = fitted_bundle()
        invalid = [("top-level-list", []), ("wrong-dictionary", {"estimator": original["predictor"].estimator}),
                   ("raw-estimator", {**original, "predictor": original["predictor"].estimator})]
        with tempfile.TemporaryDirectory() as directory:
            for name, value in invalid:
                path = Path(directory) / (name + ".joblib")
                joblib.dump(value, path)
                with self.subTest(name=name), self.assertRaises(ArtifactLoadError) as caught:
                    load_live_artifact(path)
                message = str(caught.exception)
                self.assertIn(str(path.resolve()), message)
                self.assertIn("loaded_type=", message)
                self.assertIn("expected_import=rechecklab.predictor.OutcomePredictor", message)
                if isinstance(value, dict):
                    self.assertIn("keys=", message)
                    self.assertIn("predictor_type=", message)

    def test_old_schema_unfitted_and_changed_order_are_rejected(self):
        original = fitted_bundle()
        changes = []
        legacy = copy.deepcopy(original)
        legacy["feature_names"] = FEATURE_NAMES[:-1]
        legacy["predictor"].feature_names = tuple(FEATURE_NAMES[:-1])
        changes.append(("legacy-18-features", legacy, "incompatible"))
        unfitted = copy.deepcopy(original)
        unfitted["predictor"].estimator = LogisticRegression()
        changes.append(("unfitted", unfitted, "unfitted"))
        feature_order = copy.deepcopy(original)
        feature_order["predictor"].estimator.feature_names_in_ = np.asarray(list(reversed(FEATURE_NAMES)))
        changes.append(("feature-order", feature_order, "feature ordering"))
        class_order = copy.deepcopy(original)
        class_order["class_order"] = list(reversed(LABELS))
        changes.append(("class-order", class_order, "class order"))
        with tempfile.TemporaryDirectory() as directory:
            for name, value, reason in changes:
                path = Path(directory) / (name + ".joblib")
                joblib.dump(value, path)
                with self.subTest(name=name), self.assertRaisesRegex(ArtifactLoadError, reason):
                    load_live_artifact(path)
                with self.subTest(writer=name), self.assertRaises(ValueError):
                    save_artifact(value, Path(directory) / "must-not-be-written.joblib")

    def test_path_hash_protocol_and_corrupt_bytes_fail_explicitly(self):
        original = fitted_bundle()
        with tempfile.TemporaryDirectory() as directory:
            path = save_artifact(original, Path(directory) / "gate.joblib")
            with self.assertRaisesRegex(ArtifactLoadError, "SHA-256"):
                load_artifact(path, expected_sha256="0" * 64)
            with self.assertRaisesRegex(ArtifactLoadError, "protocol"):
                load_artifact(path, expected_protocol={**original["protocol"], "model": "different-model"})
            with self.assertRaisesRegex(ArtifactLoadError, "not loaded"):
                load_artifact(Path(directory) / "missing.joblib")
            corrupt = Path(directory) / "corrupt.joblib"
            corrupt.write_bytes(b"invalid pickle bytes")
            with self.assertRaises(ArtifactLoadError):
                load_artifact(corrupt)

    def run_isolated(self, source, *arguments):
        completed = subprocess.run([sys.executable, "-c", textwrap.dedent(source), *map(str, arguments)],
                                   cwd=ROOT, text=True, capture_output=True, timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        return json.loads(completed.stdout.strip().splitlines()[-1])

    def test_production_live_loader_survives_class_eviction_with_strict_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = save_artifact(fitted_bundle(), Path(directory) / "gate.joblib")
            result = self.run_isolated("""
                import importlib, json, sys
                import numpy as np
                from rechecklab import live
                import rechecklab.predictor as old_module
                cached = live.load_live_artifact(sys.argv[1])
                old_loader = old_module.load_artifact
                old_class = type(cached['predictor'])
                sys.modules.pop('rechecklab.predictor')
                current = importlib.import_module('rechecklab.predictor')
                assert old_class is not current.OutcomePredictor
                try:
                    current.validate_artifact(cached)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Stale runtime types must still be rejected')
                fresh = live.load_live_artifact(sys.argv[1])
                from_stale_helper = old_loader(sys.argv[1])
                assert type(fresh['predictor']) is current.OutcomePredictor
                assert type(from_stale_helper['predictor']) is current.OutcomePredictor
                assert fresh['threshold'] == cached['threshold']
                np.testing.assert_array_equal(fresh['predictor'].estimator.coef_, cached['predictor'].estimator.coef_)
                import io
                from unittest.mock import patch
                payload = dict(models=[dict(name=fresh['protocol']['model'], digest=fresh['protocol']['model_digest'])])
                with patch.object(live, 'urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
                    status = live.probe_backend(fresh)
                assert status['ready'], status
                print(json.dumps(dict(fresh_class_match=True, stale_type_rejected=True, fitted_features=fresh['predictor'].estimator.n_features_in_)))
            """, path)
            self.assertTrue(result["fresh_class_match"])
            self.assertTrue(result["stale_type_rejected"])
            self.assertEqual(result["fitted_features"], 19)

    def test_actual_app_cache_retains_data_and_reloads_canonical_model_after_eviction(self):
        result = self.run_isolated("""
            import importlib, json, sys
            from pathlib import Path
            import numpy as np
            import app
            root = Path('outputs/v0.2')
            stamps = tuple((root / name).stat().st_mtime_ns for name in app.REPLAY_FILES)
            first = app.load_replay(str(root), stamps)
            old_predictor = first[-1]['predictor']
            probabilities = old_predictor.probabilities(app.initial_frame(first[-2].iloc[:3]))
            for name in list(sys.modules):
                if name == 'app' or name == 'rechecklab' or name.startswith('rechecklab.'):
                    sys.modules.pop(name, None)
            fresh_app = importlib.import_module('app')
            current = importlib.import_module('rechecklab.predictor')
            second = fresh_app.load_replay(str(root), stamps)
            assert all(left is right for left, right in zip(first[:5], second[:5])), 'Replay data cache unexpectedly changed'
            assert second[-1] is not first[-1], 'Imported predictor objects must not survive in the resource cache'
            assert type(second[-1]['predictor']) is current.OutcomePredictor
            assert type(old_predictor) is not current.OutcomePredictor
            assert first[-1]['threshold'] == second[-1]['threshold']
            assert first[-1]['feature_names'] == second[-1]['feature_names']
            assert first[-1]['class_order'] == second[-1]['class_order']
            np.testing.assert_array_equal(probabilities, second[-1]['predictor'].probabilities(fresh_app.initial_frame(second[-2].iloc[:3])))
            current.validate_artifact(second[-1], expected_protocol=second[0]['recheck_protocol'])
            print(json.dumps(dict(cached_data_retained=True, predictor_class_current=True, probabilities_identical=True)))
        """)
        self.assertEqual(result, dict(cached_data_retained=True, predictor_class_current=True, probabilities_identical=True))


if __name__ == "__main__":
    unittest.main()
