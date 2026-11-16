"""Frozen v0.2 classification, policy and token-efficiency evaluation.

No model fitting, threshold optimization, Ollama calls or application changes.
The oracle alone reads outcome labels to make hindsight decisions; it is never
treated as a deployable policy. All outputs go into an exclusive new run folder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

sys.dont_write_bytecode = True
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from rechecklab.data import LABELS, assert_disjoint, validate_pairs
from rechecklab.features import INITIAL_REQUIRED, extract_features
from rechecklab.metrics import policy_metrics, predictor_metrics, question_weights
from rechecklab.predictor import decide_recheck, load_artifact

V2 = ROOT / "outputs/v0.2"
PARTITIONS = ("source_development", "source_test", "target_test")
DISPLAY = {"source_development": "MATH source development", "source_test": "MATH source test",
           "target_test": "CommonsenseQA target test"}
POLICIES = ("Never recheck", "Always recheck", "Completion rule", "Frozen ML gate", "Oracle (hindsight)")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(run):
    result = {}
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in excluded and Path(parent, d).resolve() != run)
        for filename in sorted(files):
            path = Path(parent, filename)
            result[path.relative_to(ROOT).as_posix()] = sha(path)
    return result


def classification(frame, probs, partition):
    weights = question_weights(frame)
    predicted = np.array(LABELS)[probs.argmax(axis=1)]
    precision, recall, f1, support = precision_recall_fscore_support(
        frame.label, predicted, labels=LABELS, sample_weight=weights, zero_division=0)
    measured = predictor_metrics(frame, probs)
    weighted_f1 = float(np.average(f1, weights=support))
    rows = []
    for index, label in enumerate(LABELS):
        predicted_weight = float(weights[predicted == label].sum())
        rows.append({"partition": partition, "class": label, "precision": float(precision[index]),
                     "recall": float(recall[index]), "f1": float(f1[index]),
                     "precision_defined": predicted_weight > 0,
                     "support_rows": int(frame.label.eq(label).sum()),
                     "support_question_equivalents": float(support[index]),
                     "distribution_pct": float(100 * support[index] / support.sum()),
                     "predicted_rows": int((predicted == label).sum()),
                     "predicted_question_equivalents": predicted_weight})
    summary = {"partition": partition, "questions": frame.group_id.nunique(), "outcome_rows": len(frame),
               "three_class_accuracy_pct": 100 * measured["accuracy"], "macro_f1": measured["macro_f1"],
               "weighted_f1": weighted_f1, "log_loss": measured["log_loss"],
               "no_change_only_accuracy_pct": float(100 * np.average(frame.label.eq("NO_CHANGE"), weights=weights)),
               "HELP_precision_pct": 100 * precision[0], "HELP_recall_pct": 100 * recall[0],
               "HURT_precision_pct_convention_zero": 100 * precision[2], "HURT_recall_pct": 100 * recall[2],
               "HURT_precision_defined": rows[2]["precision_defined"]}
    matrices = {"label_order_true_rows_predicted_columns": LABELS,
                "raw_outcome_row_counts": confusion_matrix(frame.label, predicted, labels=LABELS).tolist(),
                "question_equivalent_counts": measured["confusion_matrix"]}
    return summary, rows, matrices


def evaluate_policy(frame, mask, partition, name, always_accuracy):
    metric = policy_metrics(frame, mask)
    weights = question_weights(frame)
    help_mask, hurt_mask = frame.label.eq("HELP").to_numpy(), frame.label.eq("HURT").to_numpy()
    help_total, hurt_total = float(weights[help_mask].sum()), float(weights[hurt_mask].sum())
    helpful, harmful = float(weights[mask & help_mask].sum()), float(weights[mask & hurt_mask].sum())
    selected = float(weights[mask].sum())
    neutral = mask & frame.label.eq("NO_CHANGE").to_numpy()
    base_accuracy = float(100 * np.average(frame.base_correct.astype(float), weights=weights))
    return {"partition": partition, "policy": name, **metric,
            "HELP_selected_question_equivalents": helpful, "HURT_selected_question_equivalents": harmful,
            "NO_CHANGE_selected_question_equivalents": float(weights[neutral].sum()),
            "HELP_missed_question_equivalents": help_total - helpful,
            "HURT_avoided_question_equivalents": hurt_total - harmful,
            "HELP_capture_pct": 100 * helpful / help_total if help_total else None,
            "helpful_recheck_precision_pct": 100 * helpful / selected if selected else None,
            "harmful_fraction_of_selected_rechecks_pct": 100 * harmful / selected if selected else None,
            "HURT_avoidance_pct": 100 * (hurt_total - harmful) / hurt_total if hurt_total else None,
            "initial_tokens_per_question": float(np.average(frame.base_actual_tokens, weights=weights)),
            "NO_CHANGE_extra_tokens_per_question": float(np.sum(weights[neutral] * frame.loc[neutral, "action_actual_tokens"]) / weights.sum()),
            "recheck_calls_selected_question_equivalents": selected,
            "recheck_calls_avoided_question_equivalents": float(weights.sum() - selected),
            "recheck_calls_avoided_pct": 100 - metric["recheck_pct"],
            "selected_outcome_rows": int(mask.sum()), "question_groups_with_any_selected_row": frame.loc[mask, "group_id"].nunique(),
            "decisions_constant_within_question": bool(pd.DataFrame({"group": frame.group_id, "mask": mask}).groupby("group")["mask"].nunique().le(1).all()),
            "accuracy_change_vs_initial_pp": metric["accuracy_pct"] - base_accuracy,
            "accuracy_change_vs_always_pp": metric["accuracy_pct"] - always_accuracy,
            "accuracy_ratio_to_always_pct": 100 * metric["accuracy_pct"] / always_accuracy,
            "deployable": name != "Oracle (hindsight)"}


def mark_pareto(rows):
    for row in rows:
        competitors = [other for other in rows if other["partition"] == row["partition"] and other["deployable"]]
        row["on_deployable_pareto_frontier"] = bool(row["deployable"] and not any(
            other["average_total_tokens"] <= row["average_total_tokens"] + 1e-10
            and other["accuracy_pct"] >= row["accuracy_pct"] - 1e-10
            and (other["average_total_tokens"] < row["average_total_tokens"] - 1e-10
                 or other["accuracy_pct"] > row["accuracy_pct"] + 1e-10) for other in competitors))


def benchmark_gate(predictor, threshold, frame, partition, harm_weight=1.0):
    # Label-blind fixed records; one initial call per real decision, not two rollouts.
    columns = list(INITIAL_REQUIRED) + [key for key in predictor.protocol if key in frame and key not in INITIAL_REQUIRED]
    initial = frame.drop_duplicates(list(INITIAL_REQUIRED))[columns].reset_index(drop=True)
    chosen = np.linspace(0, len(initial) - 1, min(24, len(initial)), dtype=int)
    def gate(inputs):
        return decide_recheck(predictor.probabilities(inputs), threshold, harm_weight)
    for index in chosen[:3]:
        gate(initial.iloc[[index]])
    wall_samples, cpu_samples = [], []
    for _ in range(3):
        for index in chosen:
            record = initial.iloc[[index]]
            cpu_start, wall_start = time.process_time_ns(), time.perf_counter_ns()
            gate(record)
            wall_samples.append((time.perf_counter_ns() - wall_start) / 1e6)
            cpu_samples.append((time.process_time_ns() - cpu_start) / 1e6)
    gate(initial)
    batches = []
    for _ in range(5):
        start = time.perf_counter_ns()
        gate(initial)
        batches.append((time.perf_counter_ns() - start) / 1e6)
    return {"partition": partition, "unique_initial_records": len(initial), "single_record_samples": len(wall_samples),
            "single_record_median_wall_ms": float(np.median(wall_samples)),
            "single_record_p95_wall_ms": float(np.quantile(wall_samples, .95)),
            "single_record_median_process_cpu_ms": float(np.median(cpu_samples)),
            "batch_median_wall_ms": float(np.median(batches)),
            "batch_amortized_ms_per_initial_record": float(np.median(batches) / len(initial)),
            "wall_samples_ms": wall_samples, "batch_samples_ms": batches,
            "scope": "Current resident CPU gate: compatibility validation, original features, predict_proba, score and decision. "
                     "Excludes imports/artifact loading, Streamlit and LLM time. Warmups excluded; one thread. "
                     "Not a historical end-to-end latency measurement."}


def draw_plot(rows, frames, run):
    # Keep even library/font caches inside this new analysis run.
    os.environ["MPLCONFIGDIR"] = str(run / "matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors = ["#64748b", "#c2410c", "#15803d", "#2563eb", "#9333ea"]
    markers = ["o", "s", "^", "D", "*"]
    offsets = {
        "source_development": [(8, -18), (-80, -24), (12, 10), (-70, -24), (10, 10)],
        "source_test": [(8, 8), (-80, 10), (-95, 15), (12, -25), (10, 10)],
        "target_test": [(12, -23), (-85, 10), (15, 0), (15, 0), (12, 10)],
    }
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(16.8, 5.8))
    for axis, partition in zip(axes, PARTITIONS):
        subset = [row for row in rows if row["partition"] == partition]
        frontier = sorted([r for r in subset if r["on_deployable_pareto_frontier"]], key=lambda r: r["average_total_tokens"])
        unique_frontier = {(r["average_total_tokens"], r["accuracy_pct"]) for r in frontier}
        xs, ys = zip(*sorted(unique_frontier))
        axis.plot(xs, ys, color="#94a3b8", linestyle="--", linewidth=1.4, zorder=1)
        for index, row in enumerate(subset):
            oracle = index == 4
            axis.scatter(row["average_total_tokens"], row["accuracy_pct"], marker=markers[index],
                         color=colors[index], s=145 if oracle else 65, edgecolors="white", linewidths=.65, zorder=4 if oracle else 3)
            if partition == "target_test" and index == 2:
                continue
            label = "Completion = ML gate" if partition == "target_test" and index == 3 else row["policy"].replace("Frozen ", "").replace(" (hindsight)", "*")
            axis.annotate(label, (row["average_total_tokens"], row["accuracy_pct"]), xytext=offsets[partition][index],
                          textcoords="offset points", fontsize=8.8, color=colors[index],
                          arrowprops={"arrowstyle": "-", "color": colors[index], "lw": .7})
        axis.set_title(DISPLAY[partition] + f"\n{frames[partition].group_id.nunique()} questions / {len(frames[partition])} outcome rows", fontsize=11, pad=12)
        axis.set_xlabel("Average recorded tokens per question")
        axis.set_ylabel("Final-answer accuracy (%)")
        axis.grid(alpha=.17)
        ymin, ymax = min(r["accuracy_pct"] for r in subset), max(r["accuracy_pct"] for r in subset)
        axis.set_ylim(ymin - max(1.2, (ymax - ymin) * .12), ymax + max(1.2, (ymax - ymin) * .16))
        xmin, xmax = min(r["average_total_tokens"] for r in subset), max(r["average_total_tokens"] for r in subset)
        axis.set_xlim(xmin - .06 * (xmax - xmin), xmax + .10 * (xmax - xmin))
    handles = [Line2D([], [], marker=markers[i], color="none", markerfacecolor=colors[i], markeredgecolor=colors[i],
                      markersize=8, label=POLICIES[i]) for i in range(5)]
    handles.append(Line2D([], [], color="#94a3b8", linestyle="--", label="Observed deployable frontier"))
    figure.suptitle("RecheckLab v0.2: accuracy versus recorded token cost", fontsize=16, y=.98)
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .91), ncol=6, frameon=False, fontsize=9)
    figure.text(.5, .025, "* Oracle uses future outcomes and is unattainable; excluded from the deployable frontier. "
                "Token savings are replay proxies, not measured LLM time savings.\nExisting, previously inspected partitions; no new model training or policy tuning.",
                ha="center", fontsize=9, color="#475569")
    figure.subplots_adjust(top=.74, bottom=.19, wspace=.29, left=.055, right=.98)
    figure.savefig(run / "accuracy_vs_tokens.png", dpi=190, bbox_inches="tight")
    figure.savefig(run / "accuracy_vs_tokens.svg", bbox_inches="tight")
    plt.close(figure)


def comparison_table(rows, partitions):
    lines = ["| Dataset / policy | Final accuracy | Recheck | Avg. tokens | Total tokens saved | HELP captured | HURT incurred |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        if row["partition"] not in partitions:
            continue
        capture = "N/A" if row["HELP_capture_pct"] is None else f"{row['HELP_capture_pct']:.2f}%"
        label = DISPLAY[row["partition"]] + " / " + row["policy"]
        lines.append(f"| {label} | {row['accuracy_pct']:.4f}% | {row['recheck_pct']:.3f}% | "
                     f"{row['average_total_tokens']:.2f} | {row['saved_total_tokens_pct']:.2f}% | "
                     f"{capture} | {row['HURT_selected_question_equivalents']:g} |")
    return "\n".join(lines)


def reports(run, policy_rows, class_rows, per_class, matrices, timing_audit, benchmarks):
    text = ["# RecheckLab: professor-ready ML contribution summary", "",
            "The frozen classifier and the answer-retention policy are different evaluations. "
            "All numbers below use existing v0.2 records; test sets were previously inspected.", "",
            "## ML classifier itself", "",
            "| Dataset | 3-class accuracy | Macro-F1 | Weighted-F1 | Log loss | HELP precision / recall | HURT recall |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for row in class_rows:
        if row["partition"] == "source_development":
            continue
        text.append(f"| {DISPLAY[row['partition']]} | {row['three_class_accuracy_pct']:.4f}% | "
                    f"{row['macro_f1']:.6f} | {row['weighted_f1']:.6f} | {row['log_loss']:.6f} | "
                    f"{row['HELP_precision_pct']:.2f}% / {row['HELP_recall_pct']:.2f}% | {row['HURT_recall_pct']:.2f}% |")
    text += ["", "NO_CHANGE is 83.00% of source-test outcomes and 93.8608% of target-test outcomes. "
             "An always-NO_CHANGE predictor reaches 93.8608% target classification accuracy, above the model's "
             "91.5416%; its poor rare-class detection is exposed by macro-F1 and class-specific recall.", "",
             "## Answer accuracy and recorded token trade-off", "", comparison_table(policy_rows, ("source_test", "target_test")), "",
             "HELP capture is selected HELP / all HELP, not helpful-recheck precision. Counts are question-equivalents. "
             "Oracle is a per-outcome hindsight bound selecting HELP only and skipping neutral/harmful outcomes; "
             "it is not an attainable learned model. Its target fractional counts/calls are not literal unique questions.", "",
             "The gate captures 62/65 HELP outcomes on source test (95.38%), with 62/109 helpful rechecks "
             "(56.88% precision). It selects 1 harmful recheck (0.917% of selected; 0.25% of all questions). "
             "On target test it captures 4/10.5 HELP question-equivalents (38.10%); 4/25 selected outcomes "
             "are helpful (16%), 21 are neutral, and none is harmful. Zero observed harm is not a guarantee.", "",
             "The three-class source HELP recall is 92.31%, different from policy HELP capture 95.38%: "
             "the gate uses P(HELP)-P(HURT)>0.25968145276947807, not argmax HELP. HURT argmax recall "
             "is zero on both tests; the policy's avoided harms do not establish successful HURT classification.", "",
             "## Efficiency and timing", "",
             "Source-test gate: 23.2210% total-token savings and 52.0231% recheck-token savings versus always; "
             "291/400 recheck calls avoided, with -0.25 percentage points answer accuracy versus always. "
             "Target-test gate: 50.5327% total-token savings and 96.3155% recheck-token savings; "
             "708/733 calls avoided, with +3.8199 points answer accuracy versus always.", "",
             "Source development is separate: gate 92.00%, 4,081.3425 tokens versus always 91.00%, "
             "5,521.67625 tokens. That is 26.0851% total-token savings, 57.2460% recheck-token savings "
             "and 307/400 calls avoided. Completion ties 92% using 4,121.0125 tokens.", "",
             "No trustworthy original LLM wall-time/latency records exist. Token/call savings are replay "
             "counterfactuals using actual recorded token counts, not observed collection savings or "
             "measured seconds saved. Historical offline experiment runtime is not LLM runtime.", ""]
    for item in benchmarks:
        text.append(f"Current offline {DISPLAY[item['partition']]} gate overhead: single-record median "
                    f"{item['single_record_median_wall_ms']:.3f} ms, p95 {item['single_record_p95_wall_ms']:.3f} ms; "
                    f"batch amortized {item['batch_amortized_ms_per_initial_record']:.4f} ms/initial record. "
                    "Includes feature extraction and compatibility validation; excludes loading, UI and LLM work.")
    text += ["", "## Does ML beat the heuristic?", "",
             "Target policies are identical. On source test completion achieves the same 90% with "
             "4,402.420 tokens versus gate 4,412.825, so completion dominates the gate there. "
             "The gate's small development token advantage does not establish a held-out ML advantage.", "",
             "Observed deployable Pareto frontier: source development = never/gate; source test = "
             "never/completion/always; target test = never and the coincident completion/gate. Oracle is excluded.", "",
             "## Strongest defensible contribution and readiness", "",
             "Ready to present as a reproducible selective-compute evaluation with leakage safeguards, "
             "grouped rollout accounting, explicit token costs and strong baseline comparisons. The evidence "
             "supports useful selective rechecking relative to always rechecking, but does not demonstrate "
             "that the learned classifier outperforms the simple completion heuristic. It is not an "
             "80%-accuracy system or a demonstrated end-to-end latency improvement.", "",
             "![Accuracy versus token cost](accuracy_vs_tokens.png)", ""]
    (run / "PROFESSOR_SUMMARY.md").write_text("\n".join(text), encoding="utf-8")
    technical = ["# Technical definitions and complete metrics", "",
        "policy_metrics.csv contains every requested efficiency/decision metric for all three partitions and five policies. "
        "Tokens equal recorded prompt-plus-generation usage; NO_CHANGE tokens remain in costs. "
        "Average calls avoided = questions minus weighted selected outcomes, for a single prospective recheck per question. "
        "The offline data collection itself paid for all recorded outcomes.", "",
        "Classification uses argmax in HELP/NO_CHANGE/HURT order. Metrics have question weights 1/n_q per rollout. "
        "Weighted-F1 is support-weighted per-class F1. Raw support and question-equivalent support are both saved.", "",
        "NO_CHANGE is neutral for accuracy: either action preserves correctness, while skip saves tokens. "
        "There is no unique correct binary decision without a compute objective. Helpful-decision precision "
        "counts neutral selections as nonhelpful; 1-precision is not the harmful rate. Undefined ratios are "
        "blank/null, not zero. sklearn per-class zero-division=0 is saved alongside precision_defined flags.", "",
        "Oracle mask = observed label HELP. It is minimum-token among per-outcome accuracy-maximizing policies, "
        "since neutral/harmful revisions are skipped. This mask can differ across identical-input stochastic "
        "rollouts, unlike a realizable initial-only gate. Its costs, calls and frontier position are hindsight diagnostics.", "",
        "accuracy_change_vs_always_pp is the accuracy difference. accuracy_ratio_to_always_pct is an aggregate "
        "accuracy ratio and may exceed 100%; it is not the fraction of identical correct answers retained.", ""]
    for partition in ("source_test", "target_test"):
        technical += ["## " + DISPLAY[partition], "",
            "| Class | Precision | Recall | F1 | Raw support | Question-equivalent support | Distribution |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for row in per_class:
            if row["partition"] != partition:
                continue
            precision = f"{row['precision']:.6f}" if row["precision_defined"] else "undefined (stored 0 convention)"
            technical.append(f"| {row['class']} | {precision} | {row['recall']:.6f} | {row['f1']:.6f} | "
                             f"{row['support_rows']} | {row['support_question_equivalents']:g} | {row['distribution_pct']:.6f}% |")
        technical += ["", "Confusion matrices: true rows / predicted columns HELP, NO_CHANGE, HURT.", "",
                      "Raw outcome rows:", "```text", np.array2string(np.asarray(matrices[partition]["raw_outcome_row_counts"])), "```",
                      "Question-equivalents:", "```text", np.array2string(np.asarray(matrices[partition]["question_equivalent_counts"])), "```", ""]
    technical += ["## Separate source-development policy metrics", "", comparison_table(policy_rows, ("source_development",)), "",
        "## Timing evidence", "", json.dumps(timing_audit, indent=2), "",
        "Full benchmark samples and scope are in gate_overhead.json. Those are current offline CPU timings, "
        "not v0.2 solver timings. No conversion of tokens into seconds, money, FLOPs or energy is made.", "",
        "## Exact code locations", "",
        "- rechecklab/data.py:34 transition_label; :74 validate_pairs; :233 group_partition; :248 assert_disjoint.",
        "- rechecklab/metrics.py:9 question_weights; :22 policy_metrics; :60 predictor_metrics.",
        "- rechecklab/predictor.py:121 OutcomePredictor.probabilities; :297 revision_score; :303 decide_recheck.",
        "- rechecklab/features.py:117 extract_features.",
        "- rechecklab/experiment.py:34 experiment timer; :131 offline runtime; :133 token-accounting limitations.",
        "- This script: classification, evaluate_policy, mark_pareto and benchmark_gate implement new calculations.", ""]
    (run / "TECHNICAL_REPORT.md").write_text("\n".join(technical), encoding="utf-8")
    (run / "professor_comparison.md").write_text(comparison_table(policy_rows, ("source_test", "target_test")) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="evaluation_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name must be a simple directory name")
    run = Path(__file__).resolve().parent / args.run_name
    run.mkdir(exist_ok=False)
    before = snapshot(run)
    save(run / "original_files_before.json", before)
    try:
        frozen = load_artifact(V2 / "models/source_gate.joblib")
        metadata_record = json.loads((V2 / "reports/experiment_metadata.json").read_text())
        fingerprints = {name: sha(V2 / name) == expected for name, expected in metadata_record["artifact_sha256"].items()}
        if not all(fingerprints.values()):
            raise AssertionError("Active v0.2 artifacts failed fingerprint checks.")
        frames = {name: validate_pairs(pd.read_parquet(V2 / f"data/prepared/{name}.parquet")) for name in PARTITIONS}
        assert_disjoint(*frames.values())
        class_rows, per_class, matrices, policy_rows, predictions, benchmarks = [], [], {}, [], [], []
        for partition, frame in frames.items():
            probs = frozen["predictor"].probabilities(frame)
            classified, classes, matrix = classification(frame, probs, partition)
            class_rows.append(classified)
            per_class.extend(classes)
            matrices[partition] = matrix
            actions = [np.zeros(len(frame), dtype=bool), np.ones(len(frame), dtype=bool),
                       frame.base_done_reason.eq("length").to_numpy(),
                       decide_recheck(probs, frozen["threshold"], frozen["harm_weight"]), frame.label.eq("HELP").to_numpy()]
            always_accuracy = policy_metrics(frame, actions[1])["accuracy_pct"]
            for name, mask in zip(POLICIES, actions):
                policy_rows.append(evaluate_policy(frame, mask, partition, name, always_accuracy))
            predictions.append(pd.DataFrame({"partition": partition, "row_index": np.arange(len(frame)), "group_id": frame.group_id,
                "label": frame.label, **{"p_" + label: probs[:, i] for i, label in enumerate(LABELS)},
                "argmax_class": np.array(LABELS)[probs.argmax(axis=1)],
                **{name: mask for name, mask in zip(POLICIES, actions)}}))
            if partition == "target_test":
                saved = pd.read_csv(V2 / "reports/target_predictions.csv", float_precision="round_trip")
                if not np.allclose(probs, saved[["p_" + label for label in LABELS]], rtol=0, atol=1e-12):
                    raise AssertionError("Frozen target predictions differ from recorded predictions.")
            if partition != "source_development":
                benchmarks.append(benchmark_gate(frozen["predictor"], frozen["threshold"], frame, partition, frozen["harm_weight"]))
        mark_pareto(policy_rows)
        pd.DataFrame(class_rows).to_csv(run / "classification_summary.csv", index=False)
        pd.DataFrame(per_class).to_csv(run / "per_class_metrics.csv", index=False)
        pd.DataFrame(policy_rows).to_csv(run / "policy_metrics.csv", index=False)
        pd.concat(predictions, ignore_index=True).to_csv(run / "analysis_predictions.csv", index=False)
        save(run / "confusion_matrices.json", matrices)
        save(run / "gate_overhead.json", {"scope": "New offline timing only; no historical solver-latency claim", "benchmarks": benchmarks})
        time_words = re.compile(r"time|latency|duration|elapsed|seconds", re.I)
        timing_audit = {"possible_timing_columns": {name: [column for column in frame if time_words.search(column)]
                       for name, frame in frames.items()}, "historical_offline_experiment_runtime_seconds": metadata_record["runtime_seconds"],
                       "trustworthy_per_call_LLM_timing_available": False,
                       "measured_LLM_latency_savings_seconds": None,
                       "interpretation": "Offline experiment runtime is not LLM latency. Token savings are recorded replay cost proxies."}
        save(run / "timing_evidence.json", timing_audit)
        save(run / "artifact_verification.json", {"artifact_fingerprints": fingerprints, "model": frozen["predictor"].model_name,
             "threshold_unchanged": frozen["threshold"], "new_training": False, "new_threshold_selection": False,
             "previously_inspected_partitions": True, "python": platform.python_version(), "script_sha256": sha(__file__)})
        draw_plot(policy_rows, frames, run)
        reports(run, policy_rows, class_rows, per_class, matrices, timing_audit, benchmarks)
        print(json.dumps({"outputs": str(run), "model": frozen["predictor"].model_name,
                          "classifier_metrics": class_rows, "gate_overhead": [{key: value for key, value in row.items()
                          if key not in ("wall_samples_ms", "batch_samples_ms")} for row in benchmarks]}, indent=2), flush=True)
    finally:
        after = snapshot(run)
        changed = [name for name in before if after.get(name) != before[name]]
        added = sorted(set(after) - set(before))
        save(run / "integrity.json", {"original_files_checked": len(before), "changed_or_removed": changed,
             "new_files_outside_evaluation_run": added, "all_original_files_unchanged": not changed and not added})
        if changed or added:
            raise RuntimeError("Original-file integrity check failed.")


if __name__ == "__main__":
    main()
