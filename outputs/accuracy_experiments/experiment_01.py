"""Training-only nested CV for the first RecheckLab accuracy experiment.

Run from any directory with the project's existing Python environment:
  python -B outputs/accuracy_experiments/experiment_01.py --run-name experiment_01

Only source_train.parquet is loaded during selection. Source development is
loaded once, after selection.json and the production-format artifact are saved.
No solver, network, application entry point, or test-data loader is invoked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import sys
import time

# All experiment writes must stay inside this directory, including bytecode.
sys.dont_write_bytecode = True
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = ROOT / "outputs" / "accuracy_experiments"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings

from rechecklab.data import LABELS, assert_disjoint, validate_pairs
from rechecklab.features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, extract_features
from rechecklab.metrics import policy_metrics, predictor_metrics, question_weights
from rechecklab.predictor import (
    OutcomePredictor, choose_threshold, decide_recheck, load_artifact,
    make_artifact, revision_score, save_artifact, validate_probabilities,
)

SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
SELECTION_RULE = (
    "Largest pooled outer-held-out question-equal HELP selected minus HURT selected; "
    "ties at 12 decimal places use fewer average extra recheck tokens, then fixed candidate order. "
    "Macro-F1 and log loss are secondary reports only."
)
LIMITATIONS = [
    "Each outer threshold is selected without its outer validation labels. Selecting the best "
    "configuration among these outer CV scores still introduces winner-selection optimism.",
    "The final threshold is optimized on the selected configuration's source-training OOF "
    "predictions. Its optimized training utility is a tuning statistic, not an unbiased estimate.",
    "Source development was previously inspected, and the existing frozen gate was selected on "
    "it historically. This one-pass development comparison is descriptive, not fresh confirmation.",
    "No source-test or target-test outcomes are loaded for selection or evaluation. Any future "
    "evaluation on those previously inspected tests must be labeled exploratory.",
    "One partition and one CV seed; HURT events are rare; no improvement in target accuracy is established.",
]


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_hashes():
    """Integrity-only byte hashing; no test/report outcomes are parsed here."""
    result = {}
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in excluded
                                and Path(parent, d).resolve() != EXPERIMENT_ROOT)
        for name in sorted(files):
            path = Path(parent, name)
            result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


def configurations():
    result = [{"id": "hgb_full_baseline", "family": "hgb", "balance_power": 1.0, "C": None}]
    for name, power in (("none", 0.0), ("sqrt", 0.5), ("full", 1.0)):
        for c_value in (0.1, 0.5, 2.0):
            result.append({"id": f"logistic_{name}_C{c_value:g}", "family": "logistic",
                           "balance_power": power, "C": c_value})
    result.append({"id": "hgb_unbalanced", "family": "hgb", "balance_power": 0.0, "C": None})
    return result


def fit_configuration(config, frame, features):
    weights = question_weights(frame)
    class_mass = {label: float(weights[frame.label.eq(label)].sum()) for label in LABELS}
    if any(mass <= 0 for mass in class_mass.values()):
        raise ValueError("Every fit subset must contain all three genuine outcome classes.")
    factors = np.array([(weights.sum() / (len(LABELS) * class_mass[label]))
                        ** config["balance_power"] for label in frame.label])
    fit_weights = weights * factors
    fit_weights *= weights.sum() / fit_weights.sum()
    if not np.isclose(fit_weights.sum(), weights.sum()):
        raise AssertionError("Balancing changed the total question weight.")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        if config["family"] == "logistic":
            estimator = make_pipeline(StandardScaler(), LogisticRegression(
                C=config["C"], max_iter=1500, random_state=SEED))
            estimator.fit(features, frame.label, standardscaler__sample_weight=weights,
                          logisticregression__sample_weight=fit_weights)
        else:
            estimator = HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=7, min_samples_leaf=20,
                l2_regularization=5, learning_rate=0.05, random_state=SEED)
            estimator.fit(features, frame.label, sample_weight=fit_weights)
    return estimator


def probabilities(estimator, features):
    raw = estimator.predict_proba(features)
    result = np.zeros((len(features), len(LABELS)))
    for column, label in enumerate(estimator.classes_):
        result[:, LABELS.index(label)] = raw[:, column]
    return validate_probabilities(result, len(features))


def make_splits(frame, requested, seed):
    # Reduce fold count if rare-class group support makes the request infeasible.
    support = frame.groupby("label").group_id.nunique().reindex(LABELS, fill_value=0)
    for count in range(min(requested, int(support.min())), 1, -1):
        splitter = StratifiedGroupKFold(n_splits=count, shuffle=True, random_state=seed)
        splits = list(splitter.split(np.zeros(len(frame)), frame.label, groups=frame.group_id))
        valid = all(set(frame.iloc[a].label) == set(LABELS)
                    and set(frame.iloc[b].label) == set(LABELS)
                    and not set(frame.iloc[a].group_id) & set(frame.iloc[b].group_id)
                    for a, b in splits)
        if valid:
            counts = np.zeros(len(frame), dtype=int)
            for _, validation in splits:
                counts[validation] += 1
            if not np.all(counts == 1):
                raise AssertionError("Every row must be held out exactly once.")
            return splits
    raise ValueError("Insufficient class-group support for stratified grouped CV.")


def summarize(frame, decisions, probs=None):
    result = policy_metrics(frame, decisions)
    weights = question_weights(frame)
    helpful = float(np.sum(weights * decisions * frame.label.eq("HELP").to_numpy()))
    harmful = float(np.sum(weights * decisions * frame.label.eq("HURT").to_numpy()))
    result.update(helpful_fixes_question_equivalents=helpful,
                  harmful_flips_question_equivalents=harmful,
                  utility_question_equivalents=helpful - harmful,
                  utility_per_question=(helpful - harmful) / weights.sum(),
                  macro_f1=None, log_loss=None)
    initial_accuracy = float(np.average(frame.base_correct.astype(float), weights=weights))
    if not np.isclose(result["accuracy_pct"], 100 * (initial_accuracy + result["utility_per_question"])):
        raise AssertionError("Policy accuracy disagrees with HELP minus HURT identity.")
    if probs is not None:
        measured = predictor_metrics(frame, probs)
        result.update(macro_f1=measured["macro_f1"], log_loss=measured["log_loss"],
                      HELP_recall=measured["per_class"]["HELP"]["recall"],
                      HURT_recall=measured["per_class"]["HURT"]["recall"])
    return result


def nested_comparison(train, features, configs, run_dir):
    outer_splits = make_splits(train, OUTER_FOLDS, SEED)
    outer_fold = np.full(len(train), -1)
    inner_splits, audit, assignments = {}, [], []
    for outer, (fit_indices, validation_indices) in enumerate(outer_splits):
        outer_fold[validation_indices] = outer
        subset = train.iloc[fit_indices].reset_index(drop=True)
        inner_splits[outer] = make_splits(subset, INNER_FOLDS, SEED + 100 + outer)
        audit.append({"outer_fold": outer, "fit_questions": int(subset.group_id.nunique()),
                      "validation_questions": int(train.iloc[validation_indices].group_id.nunique()),
                      "fit_class_counts": subset.label.value_counts().to_dict(),
                      "validation_class_counts": train.iloc[validation_indices].label.value_counts().to_dict(),
                      "inner_folds": len(inner_splits[outer])})
        for inner, (inner_fit, inner_validation) in enumerate(inner_splits[outer]):
            if set(train.iloc[validation_indices].group_id) & set(subset.group_id):
                raise AssertionError("Outer validation leaked into threshold tuning.")
            for index in fit_indices[inner_validation]:
                assignments.append({"outer_fold": outer, "inner_validation_fold": inner,
                                    "training_row_index": int(index), "group_id": train.iloc[index].group_id})
    pd.DataFrame({"training_row_index": np.arange(len(train)), "group_id": train.group_id,
                  "outer_validation_fold": outer_fold}).to_csv(run_dir / "outer_folds.csv", index=False)
    pd.DataFrame(assignments).to_csv(run_dir / "inner_folds.csv", index=False)
    write_json(run_dir / "fold_audit.json", audit)

    rows, fold_rows, prediction_frames, oof_predictions = [], [], [], {}
    for order, config in enumerate(configs):
        oof = np.full((len(train), len(LABELS)), np.nan)
        decisions = np.zeros(len(train), dtype=bool)
        thresholds = np.full(len(train), np.nan)
        for outer, (fit_indices, validation_indices) in enumerate(outer_splits):
            subset = train.iloc[fit_indices].reset_index(drop=True)
            subset_features = features.iloc[fit_indices].reset_index(drop=True)
            inner_oof = np.full((len(subset), len(LABELS)), np.nan)
            for inner_fit, inner_validation in inner_splits[outer]:
                estimator = fit_configuration(config, subset.iloc[inner_fit], subset_features.iloc[inner_fit])
                inner_oof[inner_validation] = probabilities(estimator, subset_features.iloc[inner_validation])
            inner_scores = revision_score(validate_probabilities(inner_oof, len(subset)))
            threshold = choose_threshold(subset, inner_scores)
            estimator = fit_configuration(config, subset, subset_features)
            heldout_probs = probabilities(estimator, features.iloc[validation_indices])
            heldout_actions = decide_recheck(heldout_probs, threshold)
            oof[validation_indices] = heldout_probs
            decisions[validation_indices] = heldout_actions
            thresholds[validation_indices] = threshold
            fold_rows.append({"configuration": config["id"], "outer_fold": outer, "threshold": threshold,
                              **summarize(train.iloc[validation_indices], heldout_actions, heldout_probs)})
        validate_probabilities(oof, len(train))
        result = {"configuration": config["id"], "candidate_order": order, "family": config["family"],
                  "balance_power": config["balance_power"], "C": config["C"],
                  "outer_folds": len(outer_splits), "inner_folds": min(len(s) for s in inner_splits.values()),
                  **summarize(train, decisions, oof)}
        rows.append(result)
        oof_predictions[config["id"]] = oof
        prediction_frames.append(pd.DataFrame({
            "configuration": config["id"], "training_row_index": np.arange(len(train)),
            "group_id": train.group_id, "label": train.label, "outer_fold": outer_fold,
            **{f"p_{label}": oof[:, column] for column, label in enumerate(LABELS)},
            "revision_score": revision_score(oof), "inner_selected_threshold": thresholds,
            "recheck": decisions,
        }))
        logging.info("%s: nested utility %.3f, accuracy %.4f%%, extra tokens %.3f",
                     config["id"], result["utility_question_equivalents"], result["accuracy_pct"],
                     result["average_extra_tokens"])
    ranking = sorted(range(len(rows)), key=lambda i: (
        -round(rows[i]["utility_per_question"], 12), round(rows[i]["average_extra_tokens"], 12), i))
    for rank, index in enumerate(ranking, 1):
        rows[index].update(selection_rank=rank, selected=rank == 1)
    winner_index = ranking[0]
    final_threshold = choose_threshold(train, revision_score(oof_predictions[configs[winner_index]["id"]]))
    for index, row in enumerate(rows):
        row["final_training_oof_threshold"] = final_threshold if index == winner_index else None
    pd.DataFrame(rows).sort_values("selection_rank").to_csv(run_dir / "candidate_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / "cv_fold_metrics.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(run_dir / "cv_predictions.csv", index=False)
    return configs[winner_index], final_threshold, rows[winner_index], oof_predictions[configs[winner_index]["id"]]


def development_evaluation(dev, selected, threshold, frozen, run_dir):
    # All selected configuration and threshold choices have already been written.
    selected_probs = selected.probabilities(dev)
    frozen_probs = frozen["predictor"].probabilities(dev)
    choices = [
        ("Selected CV gate", decide_recheck(selected_probs, threshold), selected_probs),
        ("Existing frozen gate", decide_recheck(frozen_probs, frozen["threshold"], frozen["harm_weight"]), frozen_probs),
        ("Completion rule", dev.base_done_reason.eq("length").to_numpy(), None),
        ("Always recheck", np.ones(len(dev), dtype=bool), None),
        ("Keep initial answer", np.zeros(len(dev), dtype=bool), None),
    ]
    rows = [{"policy": name, **summarize(dev, actions, probs)} for name, actions, probs in choices]
    pd.DataFrame(rows).to_csv(run_dir / "source_development_comparison.csv", index=False)
    predictions = pd.DataFrame({"development_row_index": np.arange(len(dev)), "group_id": dev.group_id,
                                "label": dev.label,
                                **{f"p_{label}": selected_probs[:, i] for i, label in enumerate(LABELS)}})
    for name, actions, _ in choices:
        predictions[name] = actions
    predictions.to_csv(run_dir / "source_development_predictions.csv", index=False)
    write_json(run_dir / "development_action_comparison.json", {
        name: {"differing_rows": int(np.sum(choices[0][1] != actions)),
               "differing_questions": int(dev.loc[choices[0][1] != actions, "group_id"].nunique())}
        for name, actions, _ in choices[1:]
    })
    return rows


def write_report(run_dir, selection, development, elapsed):
    best = selection["selected_configuration"]
    cv = selection["selected_nested_cv_metrics"]
    lines = [
        "# RecheckLab experiment 1: model and balancing comparison", "",
        f"Selected **{best['id']}**, threshold **{selection['threshold']:.15g}**; "
        f"fit on source training only using the unchanged 19 features.", "",
        f"It won by pooled nested-CV utility **{cv['utility_question_equivalents']:.3f}** "
        f"question-equivalents; nested-CV accuracy **{cv['accuracy_pct']:.4f}%**, "
        f"average extra tokens **{cv['average_extra_tokens']:.3f}**.", "",
        f"Selection used {cv['outer_folds']} stratified group outer folds and at least "
        f"{cv['inner_folds']} inner folds per outer fit. "
        "For each configuration, inner OOF predictions tuned its threshold before scoring the "
        "untouched outer fold. All configurations used identical splits. Utility then recheck "
        "tokens determined ranking; remaining ties used declared candidate order.", "",
        "Class balancing uses fit-subset question-weighted class mass raised to power 0, 0.5 or 1, "
        "normalized to preserve total sample weight. Logistic scaling uses question weights only. "
        "Logistic C values are 0.1, 0.5, 2; max_iter=1500. HGB uses 100 iterations, 7 maximum "
        "leaves, minimum leaf size 20, learning rate 0.05, L2=5. All estimator seeds are 42.", "",
        "The selected model's pooled outer OOF scores set its final threshold using training "
        "labels only. The selection JSON and compatible model artifact were saved before "
        "development was loaded. Development did not change any choice.", "",
        "| Source-development policy | Accuracy % | Helpful fixes | Harmful flips | Recheck % | Average total tokens | Macro-F1 | Log loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in development:
        f1 = "N/A" if row["macro_f1"] is None else f"{row['macro_f1']:.6f}"
        loss = "N/A" if row["log_loss"] is None else f"{row['log_loss']:.6f}"
        lines.append(f"| {row['policy']} | {row['accuracy_pct']:.4f} | "
                     f"{row['helpful_fixes_question_equivalents']:.3f} | "
                     f"{row['harmful_flips_question_equivalents']:.3f} | {row['recheck_pct']:.4f} | "
                     f"{row['average_total_tokens']:.3f} | {f1} | {loss} |")
    delta = development[0]["accuracy_pct"] - development[1]["accuracy_pct"]
    lines.extend(["", f"Selected minus existing frozen development accuracy: **{delta:+.4f} percentage points**.",
                  "Helpful/harmful counts are question-equivalents; repeated rollout rows share weight. "
                  "Action-only baselines have no three-class probabilities, so their classifier metrics are N/A.",
                  "", "## Interpretation and safeguards", ""])
    lines.extend("- " + limitation for limitation in LIMITATIONS)
    lines.extend(["", "No new dataset, LLM response, download, application change or deployment. "
                  "All writes are inside outputs/accuracy_experiments/. "
                  "integrity.json verifies original project files byte-for-byte (excluding existing "
                  "environment/cache directories). Integrity hashing reads bytes without parsing test outcomes.",
                  "", f"Runtime: {elapsed:.3f} seconds. See candidate_comparison.csv for all 11 configurations, "
                  "cv_fold_metrics.csv and cv_predictions.csv for nested decisions, and "
                  "source_development_comparison.csv for the single development evaluation.", ""])
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="experiment_01_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name must be a simple directory name")
    run_dir = EXPERIMENT_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)  # Never overwrite an earlier experiment.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log", encoding="utf-8")])
    started = time.perf_counter()
    before = original_hashes()
    write_json(run_dir / "original_files_before.json", before)
    configs = configurations()
    write_json(run_dir / "protocol.json", {
        "seed": SEED, "outer_folds_requested": OUTER_FOLDS, "inner_folds_requested": INNER_FOLDS,
        "candidates_in_fixed_order": configs, "selection_rule": SELECTION_RULE,
        "features": list(FEATURE_NAMES), "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
        "allowed_outcome_inputs": ["outputs/v0.2/data/prepared/source_train.parquet",
                                   "outputs/v0.2/data/prepared/source_development.parquet (after selection only)"],
        "harm_weight": 1.0, "token_penalty": 0.0, "limitations": LIMITATIONS,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")},
        "script_sha256": sha256(__file__),
    })
    try:
        train_path = ROOT / "outputs/v0.2/data/prepared/source_train.parquet"
        train = validate_pairs(pd.read_parquet(train_path))
        features = extract_features(train)
        frozen_path = ROOT / "outputs/v0.2/models/source_gate.joblib"
        frozen = load_artifact(frozen_path)
        if len(FEATURE_NAMES) != 19 or frozen["feature_names"] != list(FEATURE_NAMES):
            raise AssertionError("The experiment must use the original 19-feature schema.")
        logging.info("Training-only selection: %d questions, class counts %s", train.group_id.nunique(), train.label.value_counts().to_dict())
        winner, threshold, cv_metrics, winner_oof = nested_comparison(train, features, configs, run_dir)
        final_estimator = fit_configuration(winner, train, features)
        predictor = OutcomePredictor(final_estimator, winner["id"])
        bundle = make_artifact(predictor, threshold, harm_weight=1.0, protocol=frozen["protocol"])
        model_path = save_artifact(bundle, run_dir / "selected_gate.joblib")
        restored = load_artifact(model_path, expected_protocol=frozen["protocol"])
        if not np.array_equal(restored["predictor"].probabilities(train), predictor.probabilities(train)):
            raise AssertionError("Artifact roundtrip changed probabilities.")
        selection = {
            "selected_configuration": winner, "threshold": threshold, "selection_rule": SELECTION_RULE,
            "selected_nested_cv_metrics": cv_metrics,
            "final_oof_threshold_tuning_metrics_optimistic": summarize(train, decide_recheck(winner_oof, threshold), winner_oof),
            "source_training_sha256": sha256(train_path), "frozen_comparator_sha256": sha256(frozen_path),
            "selected_artifact_sha256": sha256(model_path), "source_development_loaded": False,
            "test_outcomes_loaded": False, "selected_before_development": True,
            "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT, "limitations": LIMITATIONS,
        }
        write_json(run_dir / "selection.json", selection)
        selection_hash = sha256(run_dir / "selection.json")
        logging.info("LOCKED %s, threshold %.15g; loading development for one evaluation", winner["id"], threshold)
        dev_path = ROOT / "outputs/v0.2/data/prepared/source_development.parquet"
        dev = validate_pairs(pd.read_parquet(dev_path))
        assert_disjoint(train, dev)
        development = development_evaluation(dev, restored["predictor"], threshold, frozen, run_dir)
        if sha256(run_dir / "selection.json") != selection_hash:
            raise AssertionError("Selection changed during development evaluation.")
        write_json(run_dir / "evaluation_metadata.json", {
            "source_development_sha256": sha256(dev_path), "selection_sha256": selection_hash,
            "selection_unchanged_after_evaluation": True, "source_development_evaluation_passes": 1,
            "source_test_evaluated": False, "target_test_evaluated": False,
            "selected_before_development": True, "runtime_seconds": time.perf_counter() - started,
        })
        write_report(run_dir, selection, development, time.perf_counter() - started)
        logging.info("Development selected %.4f%%, frozen %.4f%%; outputs %s",
                     development[0]["accuracy_pct"], development[1]["accuracy_pct"], run_dir)
    except Exception as exc:
        write_json(run_dir / "failure.json", {"error_type": type(exc).__name__, "error": str(exc)})
        raise
    finally:
        after = original_hashes()
        changed = [name for name in before if after.get(name) != before[name]]
        added = sorted(set(after) - set(before))
        write_json(run_dir / "integrity.json", {"original_files_checked": len(before),
                   "changed_or_removed_files": changed, "new_files_outside_experiment_directory": added,
                   "all_original_files_unchanged": not changed and not added,
                   "excluded_directories": ["outputs/accuracy_experiments", ".venv", ".git", "__pycache__",
                                            ".pytest_cache", ".mypy_cache", ".ruff_cache"]})
        if changed or added:
            raise RuntimeError("Original project integrity check failed: " + repr(changed + added))


if __name__ == "__main__":
    main()
