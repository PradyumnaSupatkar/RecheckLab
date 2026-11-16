"""Source-only model selection and a shared, validated recheck decision."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import io
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from .data import LABELS
from .features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION, extract_features

ARTIFACT_SCHEMA_VERSION = 2
MODEL_SELECTION_RULE = (
    "Fit on source training only. Select the highest question-equal source-development "
    "macro-F1; ties (12 decimal places) use lower question-equal multiclass log loss, "
    "then the fixed candidate order. No target labels are used for model selection."
)


class ArtifactLoadError(ValueError):
    """A saved artifact failed reading or its strict compatibility contract."""


def validate_probabilities(probabilities, expected_rows=None):
    """Check explicit HELP, NO_CHANGE, HURT columns; never repair invalid output."""
    try:
        result = np.asarray(probabilities, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probabilities must be a numeric matrix in the declared class order.") from exc
    if result.ndim != 2 or result.shape[1] != len(LABELS):
        raise ValueError(f"Probabilities must have shape (rows, {len(LABELS)}) in class order {LABELS}.")
    if expected_rows is not None and result.shape[0] != expected_rows:
        raise ValueError("Probability row count does not match the initial inputs.")
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("Probabilities must be finite numbers between zero and one.")
    if not np.allclose(result.sum(axis=1), 1.0, rtol=1e-7, atol=1e-8):
        raise ValueError("Outcome probabilities must sum to one in every row.")
    return result


def _positive_finite(value, name, allow_zero=False):
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.") from exc
    if not math.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}.")
    return number


def _finite_threshold(value):
    try:
        threshold = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("The recheck threshold must be finite.") from exc
    if not math.isfinite(threshold):
        raise ValueError("The recheck threshold must be finite.")
    return threshold


def _protocol_contract(protocol):
    if not isinstance(protocol, dict):
        raise ValueError("A model and recheck-protocol compatibility dictionary is required.")
    for key in ("model", "recheck_protocol", "action"):
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            raise ValueError(f"The compatibility contract requires a nonempty {key}.")
    if "model_digest" in protocol and (
        not isinstance(protocol["model_digest"], str) or not protocol["model_digest"].strip()
    ):
        raise ValueError("model_digest must be nonempty when supplied.")
    return dict(protocol)


@dataclass
class OutcomePredictor:
    estimator: object
    model_name: str
    feature_schema_version: str = FEATURE_SCHEMA_VERSION
    feature_schema_fingerprint: str = FEATURE_SCHEMA_FINGERPRINT
    feature_names: tuple = field(default_factory=lambda: tuple(FEATURE_NAMES))
    class_order: tuple = field(default_factory=lambda: tuple(LABELS))
    protocol: dict | None = None

    def validate_contract(self):
        if (getattr(self, "feature_schema_version", None) != FEATURE_SCHEMA_VERSION or
            getattr(self, "feature_schema_fingerprint", None) != FEATURE_SCHEMA_FINGERPRINT or
            list(getattr(self, "feature_names", ())) != FEATURE_NAMES or
            list(getattr(self, "class_order", ())) != LABELS):
            raise ValueError("The predictor uses an incompatible feature schema or class order; rerun the pilot.")
        if isinstance(self.estimator, BaseEstimator):
            try:
                check_is_fitted(self.estimator)
            except (NotFittedError, TypeError) as exc:
                raise ValueError("The predictor's scikit-learn estimator is unfitted.") from exc
        classes = np.asarray(getattr(self.estimator, "classes_", []))
        if (classes.ndim != 1 or not len(classes) or len(set(classes.tolist())) != len(classes) or
            any(name not in LABELS for name in classes)):
            raise ValueError("The estimator must declare unique, recognized outcome classes.")
        count = getattr(self.estimator, "n_features_in_", None)
        if count is not None and count != len(FEATURE_NAMES):
            raise ValueError("The estimator was fitted with an incompatible number of features.")
        fitted_names = getattr(self.estimator, "feature_names_in_", None)
        if fitted_names is not None and list(fitted_names) != FEATURE_NAMES:
            raise ValueError("The estimator's fitted feature ordering is incompatible.")
        if self.protocol is not None:
            _protocol_contract(self.protocol)

    def probabilities(self, frame):
        self.validate_contract()
        # Metadata present in a recorded dataset must agree with the saved solver/action.
        if self.protocol is not None:
            for key, expected in self.protocol.items():
                if key in frame.columns and not frame[key].eq(expected).all():
                    raise ValueError(f"Initial records have incompatible {key} metadata.")
        features = extract_features(frame)
        try:
            raw = np.asarray(self.estimator.predict_proba(features), dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"The estimator could not produce valid outcome probabilities: {exc}") from exc
        classes = list(self.estimator.classes_)
        if raw.ndim != 2 or raw.shape != (len(frame), len(classes)):
            raise ValueError("Estimator probability columns do not match its declared classes.")
        result = np.zeros((len(frame), len(LABELS)), dtype=float)
        for index, name in enumerate(classes):
            result[:, LABELS.index(name)] = raw[:, index]
        return validate_probabilities(result, len(frame))


def make_artifact(predictor, threshold, harm_weight=1.0, protocol=None):
    protocol = _protocol_contract(protocol)
    predictor.protocol = protocol
    bundle = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
        "feature_names": list(FEATURE_NAMES), "class_order": list(LABELS),
        "protocol": protocol, "predictor": predictor,
        "threshold": _finite_threshold(threshold),
        "harm_weight": _positive_finite(harm_weight, "harm_weight"),
    }
    return validate_artifact(bundle)


def validate_artifact(bundle, expected_protocol=None):
    if not isinstance(bundle, dict):
        raise ValueError("The saved gate must contain a versioned compatibility contract.")
    if (bundle.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION or
        bundle.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or
        bundle.get("feature_schema_fingerprint") != FEATURE_SCHEMA_FINGERPRINT or
        bundle.get("feature_names") != FEATURE_NAMES or bundle.get("class_order") != LABELS):
        raise ValueError("The saved gate has an incompatible artifact, feature schema, or class order; rerun the pilot.")
    predictor = bundle.get("predictor")
    if not isinstance(predictor, OutcomePredictor):
        raise ValueError("The saved gate must contain an OutcomePredictor.")
    predictor.validate_contract()
    protocol = _protocol_contract(bundle.get("protocol"))
    if predictor.protocol != protocol:
        raise ValueError("The predictor and artifact disagree about the model/recheck protocol.")
    if expected_protocol is not None and protocol != _protocol_contract(expected_protocol):
        raise ValueError("The saved gate is incompatible with the displayed experiment's model/recheck protocol.")
    _finite_threshold(bundle.get("threshold"))
    _positive_finite(bundle.get("harm_weight"), "harm_weight")
    return bundle


def save_artifact(bundle, path):
    """Write the production plain-dictionary format after strict validation."""
    # Resolve the active module after a file-watcher reload; this retains the
    # exact class check without accepting a similarly named foreign object.
    importlib.import_module(__name__).validate_artifact(bundle)
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def _artifact_description(value):
    value_type = type(value)
    description = f"loaded_type={value_type.__module__}.{value_type.__qualname__}"
    if isinstance(value, dict):
        description += "; keys=" + repr(sorted(str(key) for key in value))
        predictor_type = type(value.get("predictor"))
        description += f"; predictor_type={predictor_type.__module__}.{predictor_type.__qualname__}"
    return description


def load_artifact(path, *, expected_protocol=None, expected_sha256=None):
    """Read exact artifact bytes and validate against the active canonical class.

    Streamlit can retain an imported helper after reloading local modules.
    Joblib unpickles through the current module, so use that module's validator
    as well. Model/schema/protocol checks remain strict; no retraining or type
    coercion is performed.
    """
    path = Path(path).resolve()
    loaded = None
    description = "loaded_type=<not loaded>"
    expected_import = "rechecklab.predictor.OutcomePredictor"
    try:
        raw = path.read_bytes()
        if expected_sha256 is not None:
            if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or
                any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)):
                raise ValueError("Expected artifact SHA-256 must be a 64-character hexadecimal string.")
            if hashlib.sha256(raw).hexdigest() != expected_sha256.lower():
                raise ValueError("Artifact SHA-256 does not match the displayed experiment.")
        loaded = joblib.load(io.BytesIO(raw))
        description = _artifact_description(loaded)
        active_module = importlib.import_module(__name__)
        active_module.validate_artifact(loaded, expected_protocol=expected_protocol)
    except Exception as exc:
        raise ArtifactLoadError(f"Cannot load saved gate at {path}: {exc} [{description}; expected_import={expected_import}]") from exc
    return loaded


def _question_weights(frame):
    if frame.empty or "group_id" not in frame or frame.group_id.isna().any():
        raise ValueError("Model fitting and threshold selection require nonempty question-grouped records.")
    # Each question has total training weight one, regardless of its rollouts.
    return 1.0 / frame.groupby("group_id")["group_id"].transform("size").to_numpy()


def fit_predictor(train, dev, seed=42, return_candidates=False):
    """Compare a majority baseline and balanced/unbalanced models on source dev."""
    from .metrics import predictor_metrics

    if "label" not in train or "label" not in dev or train.label.nunique() < 2:
        raise ValueError("Training needs at least two outcome classes. Collect more paired responses.")
    if train.empty or dev.empty or not train.label.isin(LABELS).all() or not dev.label.isin(LABELS).all():
        raise ValueError("Source training and development require nonempty valid outcome labels.")
    _question_weights(train)
    _question_weights(dev)
    if set(train.group_id) & set(dev.group_id):
        raise ValueError("Source training and development questions must be disjoint.")
    x_train = extract_features(train)
    extract_features(dev)
    question_weights = _question_weights(train)
    present = list(train.label.unique())
    class_mass = {label: question_weights[train.label.to_numpy() == label].sum() for label in present}
    balanced_weights = question_weights * np.array([
        question_weights.sum() / (len(present) * class_mass[label]) for label in train.label
    ])
    candidates = [
        ("majority_class", DummyClassifier(strategy="prior"), False),
        ("logistic_regression", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1500, random_state=seed)), False),
        ("logistic_regression_balanced", make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1500, random_state=seed)), True),
        ("hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            l2_regularization=5, learning_rate=0.05, random_state=seed), False),
        ("hist_gradient_boosting_balanced", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
            l2_regularization=5, learning_rate=0.05, random_state=seed), True),
    ]
    scores, fitted = [], {}
    for name, estimator, balanced in candidates:
        weights = balanced_weights if balanced else question_weights
        if hasattr(estimator, "named_steps"):
            estimator.fit(x_train, train.label, standardscaler__sample_weight=question_weights,
                          logisticregression__sample_weight=weights)
        else:
            estimator.fit(x_train, train.label, sample_weight=weights)
        predictor = OutcomePredictor(estimator, name)
        # Build via the same public path used for evaluation and the demo.
        probabilities = predictor.probabilities(dev)
        metrics = predictor_metrics(dev, probabilities)
        loss = metrics.get("log_loss", metrics.get("multiclass_log_loss"))
        if loss is None:
            indices = np.array([LABELS.index(label) for label in dev.label])
            loss = np.average(-np.log(np.clip(probabilities[np.arange(len(dev)), indices], 1e-15, 1)),
                              weights=_question_weights(dev))
        scores.append({"model": name, "class_balanced_training": balanced,
                       "source_dev_macro_f1": float(metrics["macro_f1"]),
                       "source_dev_log_loss": float(loss), "source_dev_metrics": metrics,
                       "selection_rule": MODEL_SELECTION_RULE})
        fitted[name] = predictor
    winner = max(range(len(scores)), key=lambda index: (
        round(scores[index]["source_dev_macro_f1"], 12),
        -round(scores[index]["source_dev_log_loss"], 12), -index,
    ))
    for index, row in enumerate(scores):
        row["selected"] = index == winner
    result = (fitted[scores[winner]["model"]], scores)
    return (*result, fitted) if return_candidates else result


def revision_score(probabilities, harm_weight=1.0):
    probabilities = validate_probabilities(probabilities)
    harm_weight = _positive_finite(harm_weight, "harm_weight")
    return probabilities[:, LABELS.index("HELP")] - harm_weight * probabilities[:, LABELS.index("HURT")]


def decide_recheck(probabilities, threshold, harm_weight=1.0):
    """A decision to skip keeps the initial answer; it does not certify correctness."""
    return revision_score(probabilities, harm_weight) > _finite_threshold(threshold)


def choose_threshold(frame, scores, harm_weight=1.0, token_penalty=0.0):
    """Choose on development/adaptation labels; gold is never an inference input."""
    harm_weight = _positive_finite(harm_weight, "harm_weight")
    token_penalty = _positive_finite(token_penalty, "token_penalty", allow_zero=True)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) != len(frame) or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Threshold selection requires one finite score per development question rollout.")
    if "label" not in frame or not frame.label.isin(LABELS).all():
        raise ValueError("Threshold selection requires valid development outcome labels.")
    weights = _question_weights(frame)
    if "action_actual_tokens" not in frame:
        raise ValueError("Threshold selection requires recorded recheck token counts.")
    try:
        cost = frame.action_actual_tokens.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid recorded recheck token counts.") from exc
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Invalid recorded recheck token counts.")
    # Every observed score is a candidate, so repeated rollouts cannot bias a
    # quantile grid or hide a distinct decision boundary. Strict > is shared.
    thresholds = np.unique(np.r_[np.nextafter(scores.min(), -np.inf), scores, np.nextafter(scores.max(), np.inf)])
    thresholds = thresholds[np.isfinite(thresholds)]
    gain = (frame.label == "HELP").to_numpy().astype(float) - harm_weight * (frame.label == "HURT").to_numpy()
    normalized_cost = cost / max(float(np.average(cost, weights=weights)), 1)
    best = None
    for threshold in thresholds:
        decisions = scores > threshold
        utility = float(np.average(decisions * (gain - token_penalty * normalized_cost), weights=weights))
        used_tokens = float(np.average(decisions * cost, weights=weights))
        key = (round(utility, 12), -round(used_tokens, 12), float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold))
    return best[1]
