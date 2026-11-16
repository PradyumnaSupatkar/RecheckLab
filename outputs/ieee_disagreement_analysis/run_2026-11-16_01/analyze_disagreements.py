"""Audit frozen v0.2 gate/completion disagreements without inference or tuning.

Outputs contain identifiers, predictions, inherited correctness transitions,
and token counts only. Gold answer strings and question text are never written.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
PREP = ROOT / "outputs" / "v0.2" / "data" / "prepared"
RAW = ROOT / "data" / "raw"
MODEL = ROOT / "outputs" / "v0.2" / "models" / "source_gate.joblib"
SAVED_PRED = ROOT / "outputs" / "ml_efficiency_evaluation" / "evaluation_2026-11-03" / "analysis_predictions.csv"
INPUT_MANIFEST = ROOT / "outputs" / "ieee_manuscript_preparation" / "run_2026-11-16_01" / "INPUT_SHA256.json"
BOOTSTRAPS = 10000
BOOTSTRAP_SEED = 20261116
COHORT_INDEX = {
    "MATH source test": 1,
    "CommonsenseQA target test": 2,
    "MATH500": 3,
    "GSM8K": 4,
}

sys.path.insert(0, str(ROOT))
from rechecklab.data import LABELS, question_hash, validate_pairs
from rechecklab.predictor import load_artifact, revision_score, decide_recheck


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_frozen_inputs() -> dict[str, str]:
    expected = json.loads(INPUT_MANIFEST.read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = ROOT / Path(relative)
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(f"Frozen input hash mismatch: {relative}")
    return expected


def read_cohorts(artifact: dict) -> dict[str, pd.DataFrame]:
    cohorts: dict[str, pd.DataFrame] = {
        "MATH source test": validate_pairs(pd.read_parquet(PREP / "source_test.parquet")),
        "CommonsenseQA target test": validate_pairs(pd.read_parquet(PREP / "target_test.parquet")),
    }
    protocol = artifact["protocol"]["recheck_protocol"]
    for raw_name, display in (("math500", "MATH500"), ("gsm8k", "GSM8K")):
        frame = pd.read_parquet(RAW / f"{raw_name}.parquet")
        frame = frame.loc[frame["action"].eq("active_verify")].copy().reset_index(drop=True)
        frame["dataset"] = f"SEVRA_{raw_name}"
        frame["recheck_protocol"] = protocol
        frame.loc[frame["base_done_reason"].eq(""), "base_done_reason"] = "unknown"
        cohorts[display] = validate_pairs(frame)
    return cohorts


def check_saved_decisions(name: str, frame: pd.DataFrame, gate: np.ndarray) -> None:
    if name not in ("MATH source test", "CommonsenseQA target test"):
        return
    partition = "source_test" if name == "MATH source test" else "target_test"
    saved = pd.read_csv(SAVED_PRED)
    saved = saved.loc[saved["partition"].eq(partition)].sort_values("row_index")
    if len(saved) != len(frame):
        raise ValueError(f"Saved-prediction row count mismatch for {name}.")
    if not np.array_equal(saved["group_id"].to_numpy(), frame["group_id"].to_numpy()):
        raise ValueError(f"Saved-prediction question alignment mismatch for {name}.")
    saved_gate = saved["Frozen ML gate"].astype(str).str.lower().map({"true": True, "false": False}).to_numpy(bool)
    if not np.array_equal(saved_gate, gate):
        raise ValueError(f"Recomputed frozen gate decisions mismatch saved results for {name}.")


def bool_array(values) -> np.ndarray:
    return np.asarray(values, dtype=bool)


def bootstrap_mean_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAPS, dtype=float)
    pos = 0
    while pos < BOOTSTRAPS:
        batch = min(250, BOOTSTRAPS - pos)
        indices = rng.integers(0, n, size=(batch, n))
        draws[pos:pos + batch] = values[indices].mean(axis=1)
        pos += batch
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def qmean(frame: pd.DataFrame, values: np.ndarray) -> pd.Series:
    return pd.Series(values, index=frame["group_id"]).groupby(level=0, sort=True).mean()


def analyze_cohort(name: str, frame: pd.DataFrame, artifact: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    proba = artifact["predictor"].probabilities(frame)
    score = revision_score(proba, artifact["harm_weight"])
    gate = decide_recheck(proba, artifact["threshold"], artifact["harm_weight"])
    completion = frame["base_done_reason"].eq("length").to_numpy(bool)
    check_saved_decisions(name, frame, gate)

    base_ok = bool_array(frame["base_correct"])
    revised_ok = bool_array(frame["final_correct"])
    ml_ok = np.where(gate, revised_ok, base_ok)
    completion_ok = np.where(completion, revised_ok, base_ok)
    acc_delta = ml_ok.astype(float) - completion_ok.astype(float)
    cost_delta = (gate.astype(float) - completion.astype(float)) * frame["action_actual_tokens"].to_numpy(float)
    ml_tokens = frame["base_actual_tokens"].to_numpy(float) + gate * frame["action_actual_tokens"].to_numpy(float)
    completion_tokens = frame["base_actual_tokens"].to_numpy(float) + completion * frame["action_actual_tokens"].to_numpy(float)

    direction = np.where(gate & ~completion, "ML_RECHECK_COMPLETION_SKIP",
                         np.where(~gate & completion, "ML_SKIP_COMPLETION_RECHECK", "AGREE"))
    effect = np.where(acc_delta > 0, "ML_HELPED_ACCURACY",
                      np.where(acc_delta < 0, "ML_HARMED_ACCURACY", "NO_ACCURACY_DIFFERENCE"))
    label = frame["label"].astype(str).to_numpy()
    done = frame["base_done_reason"].astype(str).str.casefold().to_numpy()
    base_answer_present = frame["base_answer"].fillna("").astype(str).str.strip().ne("").to_numpy()
    if "changed_answer" in frame:
        answer_changed = frame["changed_answer"].astype(bool).to_numpy()
    elif "final_answer" in frame:
        answer_changed = frame["base_answer"].fillna("").astype(str).to_numpy() != frame["final_answer"].fillna("").astype(str).to_numpy()
    else:
        answer_changed = np.full(len(frame), False, dtype=bool)

    disagreement = gate != completion
    row_ids = np.flatnonzero(disagreement)
    disagreements = pd.DataFrame({
        "cohort": name,
        "canonical_question_sha256": frame["group_id"].to_numpy()[row_ids],
        "rollout": frame["rollout"].to_numpy()[row_ids],
        "rollouts_in_question": frame.groupby("group_id")["group_id"].transform("size").to_numpy()[row_ids],
        "initial_completion_status": done[row_ids],
        "initial_length_limited": done[row_ids] == "length",
        "initial_extracted_answer_present": base_answer_present[row_ids],
        "initial_finalizer_used": frame["base_finalizer_used"].astype(bool).to_numpy()[row_ids],
        "initial_correct": base_ok[row_ids],
        "revised_correct": revised_ok[row_ids],
        "transition_label": label[row_ids],
        "parsed_answer_changed_flag": answer_changed[row_ids],
        "p_help": proba[row_ids, LABELS.index("HELP")],
        "p_hurt": proba[row_ids, LABELS.index("HURT")],
        "gate_score_p_help_minus_harm_weight_p_hurt": score[row_ids],
        "frozen_threshold": float(artifact["threshold"]),
        "frozen_harm_weight": float(artifact["harm_weight"]),
        "frozen_gate_action": np.where(gate[row_ids], "RECHECK", "SKIP"),
        "completion_action": np.where(completion[row_ids], "RECHECK", "SKIP"),
        "decision_direction": direction[row_ids],
        "effect_on_correctness_vs_completion": effect[row_ids],
        "accuracy_delta_ml_minus_completion": acc_delta[row_ids],
        "action_actual_tokens": frame["action_actual_tokens"].to_numpy()[row_ids],
        "incremental_replay_tokens_ml_minus_completion": cost_delta[row_ids],
    })

    # Question-level values: each rollout contributes within its canonical question;
    # bootstrap resamples questions, never individual rollouts.
    qacc_delta = qmean(frame, acc_delta)
    qcost_delta = qmean(frame, cost_delta)
    qml_acc = qmean(frame, ml_ok.astype(float))
    qcompletion_acc = qmean(frame, completion_ok.astype(float))
    qml_tokens = qmean(frame, ml_tokens)
    qcompletion_tokens = qmean(frame, completion_tokens)
    qml_revision = qmean(frame, (gate * frame["action_actual_tokens"].to_numpy(float)))
    qcompletion_revision = qmean(frame, (completion * frame["action_actual_tokens"].to_numpy(float)))
    qml_calls = qmean(frame, gate.astype(float))
    qcompletion_calls = qmean(frame, completion.astype(float))

    disagreement_groups = frame.loc[disagreement, "group_id"].nunique()
    labels_in_dis = frame.loc[disagreement].groupby("label")["group_id"].agg(["size", "nunique"]).to_dict("index")
    by_dir_label = disagreements.groupby(["decision_direction", "transition_label"], dropna=False).size().to_dict()
    n_questions = len(qacc_delta)
    # Match the frozen analysis run's cohort-indexed question-cluster bootstrap stream.
    acc_ci = bootstrap_mean_ci(qacc_delta.to_numpy(), BOOTSTRAP_SEED + COHORT_INDEX[name])
    token_ci = bootstrap_mean_ci(qcost_delta.to_numpy(), BOOTSTRAP_SEED + COHORT_INDEX[name] + 100)
    summary = {
        "cohort": name,
        "questions": n_questions,
        "recorded_action_rows": len(frame),
        "disagreement_rollouts": int(disagreement.sum()),
        "disagreement_question_groups": int(disagreement_groups),
        "disagreement_rollout_pct": 100.0 * float(disagreement.mean()),
        "gate_only_rechecks": int((gate & ~completion).sum()),
        "completion_only_rechecks": int((~gate & completion).sum()),
        "disagreement_help_rows": int((label[disagreement] == "HELP").sum()),
        "disagreement_hurt_rows": int((label[disagreement] == "HURT").sum()),
        "disagreement_no_change_rows": int((label[disagreement] == "NO_CHANGE").sum()),
        "missing_initial_extracted_answers_in_disagreements": int((~base_answer_present[disagreement]).sum()),
        "length_limited_initials_in_disagreements": int((done[disagreement] == "length").sum()),
        "stop_completed_but_incorrect_initials_in_disagreements": int(((done[disagreement] == "stop") & ~base_ok[disagreement]).sum()),
        "answer_changed_flag_true_in_disagreements": int(answer_changed[disagreement].sum()),
        "ml_accuracy_pct": 100.0 * float(qml_acc.mean()),
        "completion_accuracy_pct": 100.0 * float(qcompletion_acc.mean()),
        "accuracy_delta_ml_minus_completion_pp": 100.0 * float(qacc_delta.mean()),
        "accuracy_delta_cluster_bootstrap_95ci_low_pp": 100.0 * acc_ci[0],
        "accuracy_delta_cluster_bootstrap_95ci_high_pp": 100.0 * acc_ci[1],
        "ml_recheck_pct": 100.0 * float(qml_calls.mean()),
        "completion_recheck_pct": 100.0 * float(qcompletion_calls.mean()),
        "ml_mean_revision_tokens_per_question": float(qml_revision.mean()),
        "completion_mean_revision_tokens_per_question": float(qcompletion_revision.mean()),
        "delta_revision_tokens_ml_minus_completion_per_question": float(qcost_delta.mean()),
        "delta_tokens_cluster_bootstrap_95ci_low_per_question": token_ci[0],
        "delta_tokens_cluster_bootstrap_95ci_high_per_question": token_ci[1],
        "ml_mean_total_tokens_per_question": float(qml_tokens.mean()),
        "completion_mean_total_tokens_per_question": float(qcompletion_tokens.mean()),
        "delta_total_tokens_ml_minus_completion_per_question": float(qml_tokens.mean() - qcompletion_tokens.mean()),
        "direction_label_row_counts_json": json.dumps({f"{k[0]}|{k[1]}": int(v) for k, v in by_dir_label.items()}, sort_keys=True),
    }

    qeffects = pd.DataFrame({
        "cohort": name,
        "canonical_question_sha256": qacc_delta.index,
        "rollouts": frame.groupby("group_id").size().reindex(qacc_delta.index).to_numpy(),
        "disagreement_rollouts": frame.loc[disagreement].groupby("group_id").size().reindex(qacc_delta.index, fill_value=0).to_numpy(),
        "mean_accuracy_delta_ml_minus_completion": qacc_delta.to_numpy(),
        "mean_replay_token_delta_ml_minus_completion": qcost_delta.to_numpy(),
        "ml_final_accuracy_within_question": qml_acc.to_numpy(),
        "completion_final_accuracy_within_question": qcompletion_acc.to_numpy(),
    })
    qeffects = qeffects.loc[qeffects["disagreement_rollouts"].gt(0)].copy()
    return disagreements, summary, qeffects


def make_effect_figure(summary_frame: pd.DataFrame) -> None:
    order = ["MATH source test", "CommonsenseQA target test", "MATH500", "GSM8K"]
    data = summary_frame.set_index("cohort").loc[order]
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    acc = data["accuracy_delta_ml_minus_completion_pp"].to_numpy()
    acc_lo = data["accuracy_delta_cluster_bootstrap_95ci_low_pp"].to_numpy()
    acc_hi = data["accuracy_delta_cluster_bootstrap_95ci_high_pp"].to_numpy()
    axes[0].errorbar(acc, y, xerr=np.vstack([acc - acc_lo, acc_hi - acc]), fmt="o",
                     color="#1f77b4", ecolor="#4c78a8", capsize=3)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Accuracy difference (percentage points)")
    axes[0].set_title("Frozen ML minus completion")
    axes[0].grid(axis="x", alpha=0.25)

    tok = data["delta_total_tokens_ml_minus_completion_per_question"].to_numpy()
    lo = data["delta_tokens_cluster_bootstrap_95ci_low_per_question"].to_numpy()
    hi = data["delta_tokens_cluster_bootstrap_95ci_high_per_question"].to_numpy()
    axes[1].errorbar(tok, y, xerr=np.vstack([tok - lo, hi - tok]), fmt="o",
                     color="#d95f02", ecolor="#e08a45", capsize=3)
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_yticks(y, order)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Total replay-token difference / question")
    axes[1].set_title("Frozen ML minus completion")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Incremental effect of the frozen ML gate", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "gate_vs_completion_effects.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    hashes = verify_frozen_inputs()
    artifact_digest = hashes["outputs\\v0.2\\models\\source_gate.joblib"]
    artifact = load_artifact(MODEL, expected_sha256=artifact_digest)
    cohorts = read_cohorts(artifact)
    all_disagreements = []
    all_summaries = []
    all_qeffects = []
    expected_disagreements = {
        "MATH source test": 2,
        "CommonsenseQA target test": 0,
        "MATH500": 18,
        "GSM8K": 10,
    }
    for name in expected_disagreements:
        dis, summary, qeffects = analyze_cohort(name, cohorts[name], artifact)
        if len(dis) != expected_disagreements[name]:
            raise RuntimeError(f"Disagreement count changed unexpectedly for {name}: {len(dis)}")
        all_disagreements.append(dis)
        all_summaries.append(summary)
        all_qeffects.append(qeffects)

    dis = pd.concat(all_disagreements, ignore_index=True)
    summaries = pd.DataFrame(all_summaries)
    qeffects = pd.concat(all_qeffects, ignore_index=True)
    dis.to_csv(OUT / "tables" / "DISAGREEMENTS.csv", index=False, float_format="%.10g")
    summaries.to_csv(OUT / "tables" / "COHORT_POLICY_EFFECTS.csv", index=False, float_format="%.10g")
    qeffects.to_csv(OUT / "tables" / "DISAGREEMENT_QUESTION_EFFECTS.csv", index=False, float_format="%.10g")
    (OUT / "FROZEN_INPUT_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")

    # Compact mechanical failure-mode cross-tab, with counts at rollout and question levels.
    if len(dis):
        fm = dis.groupby(["cohort", "decision_direction", "initial_completion_status",
                          "initial_extracted_answer_present", "initial_correct", "transition_label",
                          "effect_on_correctness_vs_completion"], dropna=False).agg(
            disagreement_rollouts=("canonical_question_sha256", "size"),
            disagreement_question_groups=("canonical_question_sha256", "nunique"),
            mean_action_tokens_per_rollout=("action_actual_tokens", "mean"),
            signed_incremental_tokens_raw_rollout_sum=("incremental_replay_tokens_ml_minus_completion", "sum"),
        ).reset_index()
    else:
        fm = pd.DataFrame()
    fm.to_csv(OUT / "tables" / "FAILURE_MODE_CROSSTAB.csv", index=False, float_format="%.10g")
    make_effect_figure(summaries)
    manifest = {
        "run": "2026-11-16_01",
        "artifact_sha256": artifact_digest,
        "threshold": float(artifact["threshold"]),
        "harm_weight": float(artifact["harm_weight"]),
        "cohort_disagreement_rollouts": {row["cohort"]: int(row["disagreement_rollouts"]) for row in all_summaries},
        "bootstrap_replicates": BOOTSTRAPS,
        "bootstrap_seed_base": BOOTSTRAP_SEED,
        "analysis_script_sha256": sha256(Path(__file__)),
        "input_manifest": str(INPUT_MANIFEST.relative_to(ROOT)),
        "frozen_input_sha256": hashes,
        "gold_answer_fields_written": False,
        "llm_inference_or_training": False,
        "source_input_hashes_match_snapshot": True,
        "cohort_effects": all_summaries,
    }
    (OUT / "RUN_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summaries[[
        "cohort", "disagreement_rollouts", "disagreement_question_groups",
        "accuracy_delta_ml_minus_completion_pp", "accuracy_delta_cluster_bootstrap_95ci_low_pp",
        "accuracy_delta_cluster_bootstrap_95ci_high_pp", "delta_total_tokens_ml_minus_completion_per_question",
    ]].to_string(index=False))
    print("saved", len(dis), "rollout disagreements across", len(qeffects), "question clusters")


if __name__ == "__main__":
    main()
