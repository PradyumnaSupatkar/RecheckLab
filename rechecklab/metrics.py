"""Question-equal prediction/policy metrics and question bootstrap intervals."""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from .data import LABELS


def question_weights(frame):
    if frame.empty or "group_id" not in frame or frame.group_id.isna().any():
        raise ValueError("Metrics require nonempty, validated question groups.")
    return 1.0 / frame.groupby("group_id").group_id.transform("size").to_numpy()


def validate_decisions(frame, decisions):
    decisions = np.asarray(decisions)
    if decisions.shape != (len(frame),) or decisions.dtype != np.dtype(bool):
        raise ValueError("Decisions must be one explicit boolean per outcome row.")
    return decisions


def policy_metrics(frame, decisions):
    decisions = validate_decisions(frame, decisions)
    question_weights(frame)
    base = frame.base_correct.to_numpy(dtype=float)
    final = np.where(decisions, frame.final_correct.to_numpy(dtype=float), base)
    per_row = pd.DataFrame({
        "group_id": frame.group_id.to_numpy(),
        "accuracy": final, "base_correct": base,
        "help": decisions * (frame.label == "HELP").to_numpy(),
        "hurt": decisions * (frame.label == "HURT").to_numpy(),
        "recheck": decisions.astype(float),
        "base_tokens": frame.base_actual_tokens.to_numpy(dtype=float),
        "extra_tokens": decisions * frame.action_actual_tokens.to_numpy(dtype=float),
        "all_extra_tokens": frame.action_actual_tokens.to_numpy(dtype=float),
    })
    grouped = per_row.groupby("group_id").mean(numeric_only=True)
    result = {"questions": len(grouped), "rollout_rows": len(frame),
        "accuracy_pct": 100 * grouped.accuracy.mean(),
        "harmful_flip_pct_of_all_questions": 100 * grouped.hurt.mean(),
        "helpful_fix_pct_of_all_questions": 100 * grouped.help.mean(),
        "harm_pct_of_initially_correct": 100 * grouped.hurt.sum() / grouped.base_correct.sum() if grouped.base_correct.sum() else None,
        "recovery_pct_of_initially_wrong": 100 * grouped.help.sum() / (1-grouped.base_correct).sum() if (1-grouped.base_correct).sum() else None,
        "recheck_pct": 100 * grouped.recheck.mean(),
        "average_total_tokens": (grouped.base_tokens + grouped.extra_tokens).mean(),
        "average_extra_tokens": grouped.extra_tokens.mean(),
        "saved_extra_tokens_pct": 100 * (1 - grouped.extra_tokens.sum()/grouped.all_extra_tokens.sum()) if grouped.all_extra_tokens.sum() else 0,
        "saved_total_tokens_pct": 100 * (1 - (grouped.base_tokens+grouped.extra_tokens).sum()/(grouped.base_tokens+grouped.all_extra_tokens).sum())
            if (grouped.base_tokens+grouped.all_extra_tokens).sum() else 0,
    }
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in result.items()}


def _macro_from_matrix(matrix):
    diagonal = np.diag(matrix)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.divide(2*diagonal, denominator, out=np.zeros(3), where=denominator != 0).mean())


def predictor_metrics(frame, probabilities, bootstrap_repeats=0, seed=42):
    from .predictor import validate_probabilities
    probabilities = validate_probabilities(probabilities, len(frame))
    weights = question_weights(frame)
    truth = frame.label.to_numpy()
    if not set(truth).issubset(LABELS):
        raise ValueError("Unknown outcome labels in evaluation data.")
    predicted = np.array(LABELS)[probabilities.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=LABELS, sample_weight=weights, zero_division=0)
    indices = np.array([LABELS.index(y) for y in truth])
    losses = -np.log(np.clip(probabilities[np.arange(len(frame)), indices], 1e-15, 1))
    loss = float(np.average(losses, weights=weights))
    result = {
        "macro_f1": float(f1.mean()), "log_loss": loss, "multiclass_log_loss": loss,
        "accuracy": float(np.average(truth == predicted, weights=weights)),
        "per_class": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "question_weighted_support": float(support[i])} for i, label in enumerate(LABELS)},
        "confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS, sample_weight=weights).tolist(),
        "row_confusion_matrix": confusion_matrix(truth, predicted, labels=LABELS).tolist(),
        "label_order": list(LABELS), "class_counts": {k: int(v) for k,v in frame.label.value_counts().items()},
        "predicted_class_counts": {k: int((predicted == k).sum()) for k in LABELS},
        "questions": int(frame.group_id.nunique()), "rollout_rows": len(frame),
        "aggregation": "Each question has weight 1; its outcome rows share that weight.",
    }
    if bootstrap_repeats:
        codes, groups = pd.factorize(frame.group_id, sort=True)
        matrices = np.zeros((len(groups), 3, 3))
        pred_indices = probabilities.argmax(axis=1)
        for row, group in enumerate(codes):
            matrices[group, indices[row], pred_indices[row]] += weights[row]
        group_losses = pd.DataFrame({"group": codes, "loss": losses}).groupby("group").loss.mean().to_numpy()
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(groups), size=(bootstrap_repeats, len(groups)))
        fs = [_macro_from_matrix(matrices[draw].sum(axis=0)) for draw in draws]
        result["macro_f1_95pct_ci"] = np.quantile(fs, [.025, .975]).tolist()
        result["log_loss_95pct_ci"] = np.quantile(group_losses[draws].mean(axis=1), [.025, .975]).tolist()
        result["uncertainty"] = "Question bootstrap conditional on this fixed fitted model and test partition."
    return result


def paired_accuracy_interval(frame, decisions, repeats=600, seed=42):
    decisions = validate_decisions(frame, decisions)
    diff = np.where(decisions, frame.final_correct, frame.base_correct).astype(float) - frame.base_correct.to_numpy(dtype=float)
    means = pd.DataFrame({"group": frame.group_id.to_numpy(), "difference": diff}).groupby("group").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(means, size=(repeats, len(means)), replace=True).mean(axis=1)
    return [float(x) for x in np.quantile(100*sampled, [0.025, 0.975])]
