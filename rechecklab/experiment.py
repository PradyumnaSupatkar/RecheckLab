"""Frozen source model, reserved target threshold adaptation, fixed test labels."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (assert_disjoint, group_partition, public_frames, validate_pairs,
                   validate_transfer_compatibility, write_manifest, LABELS)
from .features import FEATURE_NAMES, extract_features
from .metrics import paired_accuracy_interval, policy_metrics, predictor_metrics
from .predictor import choose_threshold, fit_predictor, revision_score, decide_recheck, make_artifact, save_artifact

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_experiment(root, source=None, target=None, seed=42, repeats=5,
                   sizes=(25, 50, 100, 200, 400), harm_weight=1.0):
    if (source is None) != (target is None):
        raise ValueError("Provide both custom source and target datasets, or neither.")
    if repeats < 1 or int(repeats) != repeats or not np.isfinite(harm_weight) or harm_weight <= 0:
        raise ValueError("Repeats must be a positive integer; harm weight must be finite and positive.")
    sizes = tuple(sizes)
    if not sizes or any(not isinstance(n, (int, np.integer)) or n <= 0 for n in sizes) or len(set(sizes)) != len(sizes):
        raise ValueError("Adaptation sizes must be distinct positive integer question counts.")
    root = Path(root).resolve()
    if root == PROJECT_ROOT:
        raise ValueError("Choose a separate output folder; the shipped v0.1 baseline is preserved.")
    started = time.perf_counter()
    is_public = source is None
    if is_public:
        source, target = public_frames(PROJECT_ROOT)
        provenance = "Published Qwen3-4B SEVRA outcomes; secondary replay analysis, no new LLM inference."
    else:
        source, target = validate_pairs(source), validate_pairs(target)
        provenance = "User-supplied paired outcomes; generation provenance is supplied by their model/protocol metadata."
    protocol = validate_transfer_compatibility(source, target)
    assert_disjoint(source, target)
    source_train, source_dev, source_test = group_partition(source, (0.6, 0.2, 0.2), seed)
    adaptation_pool, target_test = group_partition(target, (0.4, 0.6), seed+1)
    partitions = dict(source_train=source_train, source_development=source_dev, source_test=source_test,
                      target_adaptation_pool=adaptation_pool, target_test=target_test)
    assert_disjoint(*partitions.values())
    valid_sizes = [n for n in sizes if n <= adaptation_pool.group_id.nunique()]
    if not valid_sizes:
        raise ValueError("Adaptation pool is too small for these sample sizes. Supply more questions or reduce sizes.")
    # This fixed presentation budget/repeat is declared BEFORE looking at target test outcomes.
    demo_budget = 200 if 200 in valid_sizes else min(valid_sizes)
    predictor, model_selection, candidates = fit_predictor(source_train, source_dev, seed, return_candidates=True)
    source_threshold = choose_threshold(source_dev, revision_score(predictor.probabilities(source_dev), harm_weight), harm_weight)
    source_probs = predictor.probabilities(source_test)
    target_probs = predictor.probabilities(target_test)
    target_scores = revision_score(target_probs, harm_weight)
    source_scores = revision_score(source_probs, harm_weight)
    policies = []
    for name, frame, probabilities in (("Source test", source_test, source_probs), ("Target test", target_test, target_probs)):
        choices = {"Keep initial answer": np.zeros(len(frame), dtype=bool),
                   "Always recheck": np.ones(len(frame), dtype=bool),
                   "Completion rule": (frame.base_done_reason == "length").to_numpy(),
                   "Frozen source gate": decide_recheck(probabilities, source_threshold, harm_weight)}
        for policy, decisions in choices.items():
            row = {"evaluation": name, "policy": policy, **policy_metrics(frame, decisions)}
            row["accuracy_gain_vs_initial_95pct_ci"] = paired_accuracy_interval(frame, decisions, seed=seed)
            policies.append(row)
    adaptation_results, selected_adaptation = [], None
    pool_groups = adaptation_pool.group_id.drop_duplicates().to_numpy()
    for repeat in range(repeats):
        shuffled = np.random.default_rng(seed+100+repeat).permutation(pool_groups)
        for count in valid_sizes:
            adaptation_sample = adaptation_pool[adaptation_pool.group_id.isin(shuffled[:count])].copy().reset_index(drop=True)
            assert_disjoint(source_train, source_dev, source_test, adaptation_sample, target_test)
            threshold = choose_threshold(adaptation_sample,
                revision_score(predictor.probabilities(adaptation_sample), harm_weight), harm_weight)
            decisions = decide_recheck(target_probs, threshold, harm_weight)
            metrics = policy_metrics(target_test, decisions)
            adaptation_results.append({"adaptation_questions": count, "repeat": repeat,
                "threshold": threshold, "adaptation_help_rows": int((adaptation_sample.label == "HELP").sum()),
                "adaptation_hurt_rows": int((adaptation_sample.label == "HURT").sum()), **metrics})
            if count == demo_budget and repeat == 0:
                selected_adaptation = dict(questions=count, repeat=repeat, threshold=threshold,
                    seed=seed+100, group_ids=sorted(adaptation_sample.group_id.unique().tolist()))
                policies.append({"evaluation": "Target test", "policy": f"Threshold-adapted gate ({count} questions)",
                    **metrics, "accuracy_gain_vs_initial_95pct_ci": paired_accuracy_interval(target_test, decisions, seed=seed)})
    adaptation = pd.DataFrame(adaptation_results)
    summary = adaptation.groupby("adaptation_questions").agg(
        accuracy_mean=("accuracy_pct", "mean"), accuracy_std=("accuracy_pct", "std"),
        harm_mean=("harmful_flip_pct_of_all_questions", "mean"), harm_std=("harmful_flip_pct_of_all_questions", "std"),
        recheck_mean=("recheck_pct", "mean"), token_mean=("average_total_tokens", "mean"),
        adaptation_hurt_rows_mean=("adaptation_hurt_rows", "mean")).reset_index().fillna(0)
    bundle = make_artifact(predictor, source_threshold, harm_weight, protocol=protocol)
    bundle["adapted_threshold"] = selected_adaptation["threshold"]
    bundle["demo_adaptation"] = selected_adaptation
    diagnostics = {name: {
        "source_test": predictor_metrics(source_test, model.probabilities(source_test), bootstrap_repeats=600, seed=seed),
        "target_test": predictor_metrics(target_test, model.probabilities(target_test), bootstrap_repeats=600, seed=seed+1),
        "selected_on_source_dev": name == predictor.model_name,
    } for name, model in candidates.items()}
    distribution = {}
    for partition, frame in partitions.items():
        weights = 1 / frame.groupby("group_id").group_id.transform("size").to_numpy()
        distribution[partition] = {label: float(np.average(frame.label.eq(label), weights=weights)) for label in LABELS}
    feature_shift = {}
    for partition, frame in (("source_train", source_train), ("target_adaptation_pool", adaptation_pool)):
        grouped_features = extract_features(frame).assign(group_id=frame.group_id.to_numpy()).groupby("group_id").mean()
        feature_shift[partition] = grouped_features.mean().to_dict()
    dataset_signature = hashlib.sha256(pd.util.hash_pandas_object(pd.concat([source, target], ignore_index=True).astype(str), index=False).values.tobytes()).hexdigest()
    metadata = {"version": "0.2", "seed": seed, "repeats": repeats, "harm_weight": harm_weight,
        "public_replay": is_public, "source_dataset": str(source.dataset.iloc[0]), "target_dataset": str(target.dataset.iloc[0]),
        "provenance": provenance, "model": predictor.model_name, "source_threshold": source_threshold,
        "recheck_protocol": protocol, "dataset_signature": dataset_signature,
        "artifact_schema_version": bundle["artifact_schema_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "feature_schema_fingerprint": bundle["feature_schema_fingerprint"],
        "class_order": bundle["class_order"], "features": FEATURE_NAMES,
        "adaptation_method": "threshold-only adaptation; classifier and uncalibrated probabilities stay frozen",
        "model_selection": model_selection, "candidate_test_diagnostics": diagnostics,
        "outcome_prevalence_by_partition": distribution,
        "initial_feature_means_source_vs_adaptation": feature_shift,
        "source_predictor_metrics": diagnostics[predictor.model_name]["source_test"],
        "target_predictor_metrics": diagnostics[predictor.model_name]["target_test"],
        "demo_adaptation": selected_adaptation,
        "adaptation_presentation_rule": "200 questions, repeat 0 fixed in advance; for small custom runs use the smallest feasible requested budget. Never choose a budget/repeat by target-test results.",
        "metric_unit": "Question-equal weighting in training, development selection, prediction and policy metrics; question bootstrap for uncertainty.",
        "missing_data_strategy": "No numeric imputation. Published empty completion status is explicitly unknown with a trained indicator; parsed empty answers are explicit absence. Tokens and finalizer flags must be provided.",
        "completion_unknown_rows": {"source": int((source.base_done_reason == "unknown").sum()), "target": int((target.base_done_reason == "unknown").sum())},
        "runtime_seconds": time.perf_counter()-started,
        "environment": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "matplotlib", "streamlit")},
        "token_accounting": "Realized initial plus selected revision prompt/generation tokens, averaged per question; replay estimates exclude gate overhead, latency, energy and cache pricing.",
        "decision_objective": "Maximize source-development HELP minus harm_weight*HURT; tie-break by fewer realized revision tokens. No priced compute objective or compute-optimality claim.",
        "limitations": ["One solver checkpoint, one transfer direction, one partition seed.",
            "Inherited public correctness labels; answer keys are absent and were not independently rescored.",
            "Threshold-only adaptation is not probability calibration or LLM fine-tuning.",
            "Adaptation subsets overlap; their SD measures subset variability, not independent-experiment uncertainty.",
            "Former public evaluation data are explicitly repartitioned for a new protocol; these are not the original SEVRA benchmark scores.",
            "Rare HURT source labels and strong source/target class and feature shift limit transfer.",
            "Published model/action metadata checks cannot establish unreleased prompt and sampling compatibility.",
            "Skip recheck keeps the initial answer and does not certify correctness."]}
    for folder in ("reports", "models", "data/prepared"):
        (root/folder).mkdir(parents=True, exist_ok=True)
    save_artifact(bundle, root/"models"/"source_gate.joblib")
    pd.DataFrame(policies).to_csv(root/"reports"/"policy_results.csv", index=False)
    adaptation.to_csv(root/"reports"/"adaptation_runs.csv", index=False)
    summary.to_csv(root/"reports"/"adaptation_summary.csv", index=False)
    predictions = target_test.copy()
    for i, label in enumerate(LABELS):
        predictions["p_"+label] = target_probs[:, i]
    predictions["revision_score"] = target_scores
    predictions["source_gate_rechecks"] = decide_recheck(target_probs, source_threshold, harm_weight)
    predictions.to_csv(root/"reports"/"target_predictions.csv", index=False)
    extract_features(target_test).to_csv(root/"reports"/"target_features.csv", index=False)
    comparison = []
    for name, result in diagnostics.items():
        for partition in ("source_test", "target_test"):
            metrics = result[partition]
            comparison.append({"model": name, "partition": partition, "selected": result["selected_on_source_dev"],
                "macro_f1": metrics["macro_f1"], "multiclass_log_loss": metrics["log_loss"],
                **{label+"_recall": metrics["per_class"][label]["recall"] for label in LABELS}})
    pd.DataFrame(comparison).to_csv(root/"reports"/"model_comparison.csv", index=False)
    write_manifest(root/"reports"/"split_manifest.json", partitions, metadata)
    metadata["artifact_sha256"] = {str(p.relative_to(root)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/"models"/"source_gate.joblib", root/"reports"/"policy_results.csv",
                  root/"reports"/"target_predictions.csv", root/"reports"/"target_features.csv",
                  root/"reports"/"split_manifest.json", root/"reports"/"adaptation_summary.csv",
                  root/"reports"/"adaptation_runs.csv", root/"reports"/"model_comparison.csv")}
    (root/"reports"/"experiment_metadata.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    for name, frame in partitions.items():
        frame.to_parquet(root/"data"/"prepared"/(name+".parquet"), index=False)
    return metadata, pd.DataFrame(policies), summary
