"""Verify shipped replay tables against fixed records, probabilities and cutoffs.

This reads the saved experiment; it does not fit a model or change any result.
Run this file with --write-report to save the concise independent audit JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.collector import parse_answer
from rechecklab.data import LABELS, question_hash
from rechecklab.features import INITIAL_REQUIRED, extract_features
from rechecklab.predictor import revision_score, validate_artifact

RUN = ROOT / "outputs" / "v0.2"


def independent_metrics(frame, decisions):
    """Compute question-equal replay metrics directly from correctness transitions."""
    before = frame.base_correct.to_numpy(dtype=bool)
    after = frame.final_correct.to_numpy(dtype=bool)
    counts = frame.groupby("group_id").group_id.transform("size").to_numpy()
    weights = 1.0 / counts
    questions = frame.group_id.nunique()
    base = frame.base_actual_tokens.to_numpy(dtype=float)
    revision = frame.action_actual_tokens.to_numpy(dtype=float)
    total = base + decisions * revision
    helpful = decisions & ~before & after
    harmful = decisions & before & ~after

    def mean(values):
        return float(np.dot(weights, values) / questions)

    correct_mass = float(np.dot(weights, before))
    wrong_mass = float(np.dot(weights, ~before))
    all_revision = mean(revision)
    all_total = mean(base + revision)
    return {
        "questions": int(questions), "rollout_rows": len(frame),
        "accuracy_pct": 100 * mean(np.where(decisions, after, before)),
        "harmful_flip_pct_of_all_questions": 100 * mean(harmful),
        "helpful_fix_pct_of_all_questions": 100 * mean(helpful),
        "harm_pct_of_initially_correct": 100 * np.dot(weights, harmful) / correct_mass if correct_mass else None,
        "recovery_pct_of_initially_wrong": 100 * np.dot(weights, helpful) / wrong_mass if wrong_mass else None,
        "recheck_pct": 100 * mean(decisions), "average_total_tokens": mean(total),
        "average_extra_tokens": mean(decisions * revision),
        "saved_extra_tokens_pct": 100 * (1 - mean(decisions * revision) / all_revision) if all_revision else 0,
        "saved_total_tokens_pct": 100 * (1 - mean(total) / all_total) if all_total else 0,
    }


def metrics_match(computed, saved):
    return all((pd.isna(saved[key]) if value is None else
                np.isclose(value, saved[key], rtol=0, atol=1e-10))
               for key, value in computed.items())


def build_verification(run=RUN):
    reports = run / "reports"
    frame = pd.read_parquet(run / "data" / "prepared" / "target_test.parquet")
    predictions = pd.read_csv(reports / "target_predictions.csv", keep_default_na=False,
                              dtype={"example_id": str, "base_answer": str, "final_answer": str})
    policies = pd.read_csv(reports / "policy_results.csv")
    adaptation = pd.read_csv(reports / "adaptation_runs.csv")
    summary = pd.read_csv(reports / "adaptation_summary.csv")
    metadata = json.loads((reports / "experiment_metadata.json").read_text(encoding="utf-8"))
    bundle = joblib.load(run / "models" / "source_gate.joblib")
    validate_artifact(bundle, expected_protocol=metadata["recheck_protocol"])
    probabilities = bundle["predictor"].probabilities(frame)
    scores = revision_score(probabilities, bundle["harm_weight"])
    selected = metadata["demo_adaptation"]
    selected_run = adaptation.loc[
        adaptation.adaptation_questions.eq(selected["questions"]) & adaptation["repeat"].eq(selected["repeat"])
    ].iloc[0]
    decisions = {
        "Keep initial answer": np.zeros(len(frame), dtype=bool),
        "Always recheck": np.ones(len(frame), dtype=bool),
        "Completion rule": frame.base_done_reason.eq("length").to_numpy(),
        "Frozen source gate": scores > bundle["threshold"],
        f"Threshold-adapted gate ({selected['questions']} questions)": scores > selected["threshold"],
    }
    independent = {name: independent_metrics(frame, mask) for name, mask in decisions.items()}
    target_policies = policies.loc[policies.evaluation.eq("Target test")].set_index("policy")
    policy_checks = {name: name in target_policies.index and metrics_match(metrics, target_policies.loc[name])
                     for name, metrics in independent.items()}
    bad_runs = []
    for row in adaptation.to_dict("records"):
        metrics = independent_metrics(frame, scores > row["threshold"])
        if not metrics_match(metrics, row):
            bad_runs.append(dict(questions=int(row["adaptation_questions"]), repeat=int(row["repeat"])))
    recomputed_summary = adaptation.groupby("adaptation_questions").agg(
        accuracy_mean=("accuracy_pct", "mean"), accuracy_std=("accuracy_pct", "std"),
        harm_mean=("harmful_flip_pct_of_all_questions", "mean"), harm_std=("harmful_flip_pct_of_all_questions", "std"),
        recheck_mean=("recheck_pct", "mean"), token_mean=("average_total_tokens", "mean"),
        adaptation_hurt_rows_mean=("adaptation_hurt_rows", "mean"),
    ).reset_index().fillna(0)
    summary_matches = summary.shape == recomputed_summary.shape and list(summary.columns) == list(recomputed_summary.columns)
    if summary_matches:
        summary_matches = bool(np.allclose(summary.to_numpy(), recomputed_summary.to_numpy(), rtol=0, atol=1e-10))
    frozen = decisions["Frozen source gate"]
    adapted = decisions[f"Threshold-adapted gate ({selected['questions']} questions)"]
    completion = decisions["Completion rule"]
    max_skipped = float(scores[~frozen].max())
    min_rechecked = float(scores[frozen].min())
    answers = frame.base_answer
    parser_choices = [parse_answer(response, "choice") for response in frame.base_response]
    standard_choices = answers.isin(list("ABCDE")).to_numpy()
    exact_initial_fields = {key: bool(frame[key].reset_index(drop=True).eq(predictions[key]).all())
                            for key in INITIAL_REQUIRED}
    values = frame.loc[:, list(INITIAL_REQUIRED)].copy()
    saved_features = pd.read_csv(reports / "target_features.csv")
    feature_match = bool(np.allclose(extract_features(values).to_numpy(), saved_features.to_numpy(), rtol=0, atol=1e-10))
    budget_summary = summary.loc[summary.adaptation_questions.eq(selected["questions"])].iloc[0]
    return {
        "verification": "Independent read-only replay audit; no new training or LLM inference.",
        "questions": int(frame.group_id.nunique()), "rollout_rows": len(frame),
        "model": bundle["predictor"].model_name,
        "thresholds": {"frozen": float(bundle["threshold"]), "adapted": float(selected["threshold"]),
                       "adaptation_questions": int(selected["questions"]), "repeat": int(selected["repeat"])},
        "decision_vectors": {
            "completion_vs_frozen_mismatching_rows": int(np.count_nonzero(completion != frozen)),
            "frozen_vs_selected_adapted_mismatching_rows": int(np.count_nonzero(frozen != adapted)),
            "rechecked_rows": int(frozen.sum()),
            "rechecked_questions": int(frame.loc[frozen, "group_id"].nunique()),
            "skipped_rows": int((~frozen).sum()),
        },
        "score_boundary": {
            "maximum_skipped_score": max_skipped, "minimum_rechecked_score": min_rechecked,
            "frozen_and_adapted_inside_same_empty_gap": bool(
                max_skipped <= min(bundle["threshold"], selected["threshold"]) and
                max(bundle["threshold"], selected["threshold"]) < min_rechecked),
            "explanation": "Strict score > threshold: different cutoffs produce the same target decisions when no target score lies between them.",
        },
        "independently_recomputed_target_policies": independent,
        "selected_adaptation_vs_budget_average": {
            "selected_single_run_accuracy_pct": float(selected_run.accuracy_pct),
            "budget_mean_accuracy_pct": float(budget_summary.accuracy_mean),
            "budget_accuracy_sd_percentage_points": float(budget_summary.accuracy_std),
            "budget_mean_harmful_flip_pct": float(budget_summary.harm_mean),
            "budget_mean_recheck_pct": float(budget_summary.recheck_mean),
            "budget_mean_tokens": float(budget_summary.token_mean),
            "repeats": int(adaptation.adaptation_questions.eq(selected["questions"]).sum()),
            "interpretation": "The policy table is the predeclared single subset; the curve averages all subsets. Subset SD is not a confidence interval.",
        },
        "answer_fidelity": {
            "base_answer_python_types": answers.map(lambda value: type(value).__name__).value_counts().to_dict(),
            "base_answer_values": answers.value_counts().to_dict(),
            "released_standard_choice_rows": int(standard_choices.sum()),
            "released_nonstandard_answer_text_rows": int((~standard_choices).sum()),
            "exact_initial_fields_match_prepared_vs_csv": exact_initial_fields,
            "collector_choice_parser_matches_released_base_answer_rows": int(sum(a == b for a, b in zip(parser_choices, answers))),
            "collector_choice_parser_cannot_parse_rows": int(sum(answer is None for answer in parser_choices)),
            "released_standard_choices_lost_by_collector_parser_rows": int(sum(
                standard and answer is None for standard, answer in zip(standard_choices, parser_choices))),
            "recommendation": "Display and reuse each record's existing parsed base_answer. The optional collector parser targets its own FINAL: protocol; reparsing public responses can discard or change valid recorded answers.",
        },
        "checks": {
            "canonical_groups_match_questions": bool(frame.group_id.eq(frame["query"].map(question_hash)).all()),
            "saved_probabilities_match_model": bool(np.allclose(probabilities, predictions[["p_" + label for label in LABELS]].to_numpy(), rtol=0, atol=1e-10)),
            "saved_scores_match_model": bool(np.allclose(scores, predictions.revision_score, rtol=0, atol=1e-10)),
            "saved_frozen_decisions_match": bool(np.array_equal(frozen, predictions.source_gate_rechecks.to_numpy())),
            "all_target_policy_metrics_match": all(policy_checks.values()),
            "target_policy_metric_checks": policy_checks,
            "adaptation_runs_checked": len(adaptation), "adaptation_metric_mismatches": bad_runs,
            "adaptation_summary_matches_all_runs": summary_matches,
            "selected_threshold_matches_bundle_and_run": bool(
                np.isclose(selected["threshold"], bundle["adapted_threshold"], rtol=0, atol=1e-12) and
                np.isclose(selected["threshold"], selected_run.threshold, rtol=0, atol=1e-12)),
            "evaluation_features_match_prepared_initial_record": feature_match,
        },
    }


def verify_recorded_policies(run=RUN):
    """Return findings only after saved records, probabilities and metrics agree."""
    result = build_verification(run)
    checks = result["checks"]
    required = (
        "canonical_groups_match_questions", "saved_probabilities_match_model",
        "saved_scores_match_model", "saved_frozen_decisions_match",
        "all_target_policy_metrics_match", "adaptation_summary_matches_all_runs",
        "selected_threshold_matches_bundle_and_run", "evaluation_features_match_prepared_initial_record",
    )
    failures = [name for name in required if not checks[name]]
    if checks["adaptation_metric_mismatches"]:
        failures.append("adaptation_metric_mismatches")
    if not all(result["answer_fidelity"]["exact_initial_fields_match_prepared_vs_csv"].values()):
        failures.append("exact_initial_fields_match_prepared_vs_csv")
    if failures:
        raise ValueError("Saved policy replay verification failed: " + ", ".join(failures))
    return result


if __name__ == "__main__":
    if "--write-report" in sys.argv:
        result = verify_recorded_policies()
        path = RUN / "reports" / "policy_verification.json"
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, allow_nan=False))
    else:
        print(json.dumps(verify_recorded_policies(), indent=2, allow_nan=False))
