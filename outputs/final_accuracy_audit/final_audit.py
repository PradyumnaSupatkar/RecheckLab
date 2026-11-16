"""Bounded offline final audit. Never calls Ollama or changes original files.

New work: one feature variant (20 small CV fits), six fixed rules, and one
predeclared adaptation tune/confirmation split. Prior 11-model CV is reused.
Test outcomes are opened only after selections are frozen, to replay EXISTING
policies and verify labels/grading/oracles. No new policy is evaluated on tests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import time

sys.dont_write_bytecode = True
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from rechecklab.data import LABELS, assert_disjoint, question_hash, validate_pairs
from rechecklab.features import INITIAL_REQUIRED, extract_features
from rechecklab.metrics import policy_metrics, predictor_metrics, question_weights
from rechecklab.predictor import choose_threshold, decide_recheck, load_artifact, revision_score

V2 = ROOT / "outputs/v0.2"
PRIOR = ROOT / "outputs/accuracy_experiments/experiment_01_2026-11-03"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


helper = module("previous_accuracy_helpers", ROOT / "outputs/accuracy_experiments/experiment_01.py")
verifier = module("existing_policy_verifier", ROOT / "scripts/verify_recorded_policies.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def snapshot(run):
    hashes = {}
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in excluded and Path(parent, d).resolve() != run)
        for name in sorted(files):
            path = Path(parent, name)
            hashes[path.relative_to(ROOT).as_posix()] = sha(path)
    return hashes


def feature_frame(frame):
    result = extract_features(frame)
    # Initial-only interaction. No new underlying information or production change.
    result["audit_log_visible_token_ratio"] = result.initial_log_chars - result.initial_log_actual_tokens
    return result


def rules(frame):
    length = frame.base_done_reason.eq("length").to_numpy()
    finalizer = frame.base_finalizer_used.to_numpy(dtype=bool)
    missing = frame.base_answer.str.strip().eq("").to_numpy()
    return {"completion": length, "finalizer": finalizer, "length_or_missing": length | missing,
            "length_or_finalizer": length | finalizer, "keep": np.zeros(len(frame), dtype=bool),
            "always": np.ones(len(frame), dtype=bool)}


def rank_key(row, order):
    return (-round(row["utility_per_question"], 12), round(row["average_extra_tokens"], 12), order)


def train_feature_cv(train, run):
    config = {"id": "hgb_full_plus_log_visible_token_ratio", "family": "hgb", "balance_power": 1.0, "C": None}
    x = feature_frame(train)
    splits = helper.make_splits(train, 5, 42)
    probs = np.full((len(train), 3), np.nan)
    actions = np.zeros(len(train), dtype=bool)
    thresholds, folds = np.full(len(train), np.nan), np.full(len(train), -1)
    fold_rows = []
    for outer, (fit, validation) in enumerate(splits):
        subset = train.iloc[fit].reset_index(drop=True)
        inner_x = x.iloc[fit].reset_index(drop=True)
        inner_probs = np.full((len(subset), 3), np.nan)
        for a, b in helper.make_splits(subset, 3, 142 + outer):
            estimator = helper.fit_configuration(config, subset.iloc[a], inner_x.iloc[a])
            inner_probs[b] = helper.probabilities(estimator, inner_x.iloc[b])
        threshold = choose_threshold(subset, revision_score(inner_probs))
        estimator = helper.fit_configuration(config, subset, inner_x)
        p = helper.probabilities(estimator, x.iloc[validation])
        probs[validation] = p
        actions[validation] = decide_recheck(p, threshold)
        thresholds[validation], folds[validation] = threshold, outer
        fold_rows.append({"outer_fold": outer, "threshold": threshold,
                          **helper.summarize(train.iloc[validation], actions[validation], p)})
    pd.DataFrame(fold_rows).to_csv(run / "feature_cv_folds.csv", index=False)
    pd.DataFrame({"training_row_index": np.arange(len(train)), "group_id": train.group_id,
                  "outer_fold": folds, "inner_threshold": thresholds, "recheck": actions,
                  **{"p_" + name: probs[:, index] for index, name in enumerate(LABELS)}}).to_csv(run / "feature_cv_predictions.csv", index=False)
    return config, probs, {"configuration": config["id"], "origin": "new one-feature nested CV",
                           **helper.summarize(train, actions, probs)}


def source_selection(train, run):
    cached = pd.read_csv(PRIOR / "cv_predictions.csv", float_precision="round_trip")
    previous = pd.read_csv(PRIOR / "candidate_comparison.csv", float_precision="round_trip")
    rows, probabilities = [], {}
    for config in helper.configurations():
        part = cached[cached.configuration.eq(config["id"])].sort_values("training_row_index")
        if not np.array_equal(part.group_id.to_numpy(), train.group_id.to_numpy()) or not np.array_equal(part.label.to_numpy(), train.label.to_numpy()):
            raise AssertionError("Cached CV records do not match this source-training partition.")
        p = part[["p_" + name for name in LABELS]].to_numpy()
        actions = part.recheck.to_numpy(dtype=bool)
        if not np.array_equal(actions, revision_score(p) > part.inner_selected_threshold.to_numpy()):
            raise AssertionError("Cached nested decisions disagree with their saved inner thresholds.")
        metric = helper.summarize(train, actions, p)
        saved = previous.loc[previous.configuration.eq(config["id"])].iloc[0]
        for name in ("accuracy_pct", "utility_per_question", "average_extra_tokens", "macro_f1", "log_loss"):
            if not np.isclose(metric[name], saved[name], atol=1e-10, rtol=0):
                raise AssertionError("Prior model CV metric mismatch.")
        rows.append({"configuration": config["id"], "origin": "reused prior nested CV", **metric})
        probabilities[config["id"]] = p
    feature_config, feature_probs, feature_row = train_feature_cv(train, run)
    rows.append(feature_row)
    probabilities[feature_config["id"]] = feature_probs
    best_index = min(range(len(rows)), key=lambda i: rank_key(rows[i], i))
    best = rows[best_index]
    for index, row in enumerate(rows):
        row["selected_classifier"] = index == best_index
    pd.DataFrame(rows).to_csv(run / "classifier_cv_comparison.csv", index=False)
    rule_rows = [{"rule": name, **helper.summarize(train, action)} for name, action in rules(train).items()]
    best_rule_index = min(range(len(rule_rows)), key=lambda i: rank_key(rule_rows[i], i))
    pd.DataFrame(rule_rows).to_csv(run / "rule_training_comparison.csv", index=False)
    best_rule = rule_rows[best_rule_index]
    thresholds = {name: choose_threshold(train, revision_score(probabilities[name]))
                  for name in {best["configuration"], feature_config["id"]}}
    primary = ("classifier", best["configuration"]) if rank_key(best, 0) <= rank_key(best_rule, 1) else ("rule", best_rule["rule"])
    selected = {"best_classifier": best["configuration"], "best_rule": best_rule["rule"],
                "primary_policy_kind": primary[0], "primary_policy_name": primary[1],
                "thresholds_from_source_training_OOF_only": thresholds,
                "feature_configuration": feature_config, "development_loaded": False,
                "source_test_loaded": False, "target_test_loaded": False,
                "selection_rule": "Pooled question-equal HELP-HURT utility, then fewer extra tokens, then declared order",
                "optimism": "Configuration/rule winner selection is optimistic; final pooled OOF thresholds are tuning statistics. "
                            "Development is previously inspected descriptive confirmation, not fresh independent evidence."}
    save(run / "source_selection.json", selected)
    return selected


def source_confirmation(train, dev, frozen, selected, run):
    assert_disjoint(train, dev)
    configurations = {c["id"]: c for c in helper.configurations()}
    configurations[selected["feature_configuration"]["id"]] = selected["feature_configuration"]
    results, predictions = [], {}
    frozen_probs = frozen["predictor"].probabilities(dev)
    choices = [("Keep initial", np.zeros(len(dev), dtype=bool), None),
               ("Always recheck", np.ones(len(dev), dtype=bool), None),
               ("Completion rule", rules(dev)["completion"], None),
               ("Frozen gate", decide_recheck(frozen_probs, frozen["threshold"]), frozen_probs)]
    for name in dict.fromkeys([selected["best_classifier"], selected["feature_configuration"]["id"]]):
        is_feature = name == selected["feature_configuration"]["id"]
        x_train, x_dev = (feature_frame(train), feature_frame(dev)) if is_feature else (extract_features(train), extract_features(dev))
        estimator = helper.fit_configuration(configurations[name], train, x_train)
        probs = helper.probabilities(estimator, x_dev)
        choices.append((name, decide_recheck(probs, selected["thresholds_from_source_training_OOF_only"][name]), probs))
    choices.append(("Rule: " + selected["best_rule"], rules(dev)[selected["best_rule"]], None))
    for name, decisions, probs in choices:
        results.append({"policy": name, **helper.summarize(dev, decisions, probs)})
        predictions[name] = decisions
    pd.DataFrame(results).to_csv(run / "source_development_comparison.csv", index=False)
    pd.DataFrame({"group_id": dev.group_id, **predictions}).to_csv(run / "source_development_decisions.csv", index=False)
    current_optimum = choose_threshold(dev, revision_score(frozen_probs))
    save(run / "existing_source_threshold_check.json", {
        "saved_threshold": frozen["threshold"], "recomputed_dev_optimum": current_optimum,
        "matches": bool(current_optimum == frozen["threshold"]),
        "scope": "Existing threshold verification only, not a new selected threshold or independent validation"})
    return results


def adaptation_confirmation(frozen, run):
    pool = validate_pairs(pd.read_parquet(V2 / "data/prepared/target_adaptation_pool.parquet"))
    groups = np.array(sorted(pool.group_id.unique()))
    np.random.default_rng(46).shuffle(groups)
    tune_ids = set(groups[:len(groups) // 2])
    tune, confirmation = pool[pool.group_id.isin(tune_ids)].copy(), pool[~pool.group_id.isin(tune_ids)].copy()
    assert_disjoint(tune, confirmation)
    threshold = choose_threshold(tune, revision_score(frozen["predictor"].probabilities(tune)))
    selection = {"seed": 46, "tune_questions": len(tune_ids), "confirmation_questions": confirmation.group_id.nunique(),
                 "threshold": threshold, "tune_group_ids": sorted(tune_ids),
                 "confirmation_group_ids": sorted(set(confirmation.group_id)),
                 "confirmation_metrics_not_used_for_selection": True, "test_outcomes_not_used": True,
                 "limitation": "This pool was used by prior adaptation analyses; this is descriptive internal confirmation."}
    save(run / "adaptation_selection.json", selection)
    probs = frozen["predictor"].probabilities(confirmation)
    choices = {"Keep initial": np.zeros(len(confirmation), dtype=bool), "Always recheck": np.ones(len(confirmation), dtype=bool),
               "Completion rule": rules(confirmation)["completion"], "Frozen gate": decide_recheck(probs, frozen["threshold"]),
               "Tune-half threshold": decide_recheck(probs, threshold)}
    rows = [{"policy": name, **helper.summarize(confirmation, decisions, probs if name in ("Frozen gate", "Tune-half threshold") else None)}
            for name, decisions in choices.items()]
    pd.DataFrame(rows).to_csv(run / "adaptation_confirmation.csv", index=False)
    return rows


def existing_results_audit(frozen, run):
    # Selection/confirmation have finished. Tests are audited only for OLD policies.
    independent = verifier.verify_recorded_policies(V2)
    save(run / "existing_target_verification.json", independent)
    metadata = json.loads((V2 / "reports/experiment_metadata.json").read_text())
    fingerprints = {name: sha(V2 / name) == expected for name, expected in metadata["artifact_sha256"].items()}
    parts = {name: pd.read_parquet(V2 / f"data/prepared/{name}.parquet") for name in (
        "source_train", "source_development", "source_test", "target_adaptation_pool", "target_test")}
    assert_disjoint(*parts.values())
    checks, policies = {}, []
    for name, frame in parts.items():
        validated = validate_pairs(frame)
        for key in ("label", "group_id"):
            if not validated[key].equals(frame[key]):
                raise AssertionError(f"{name}: inconsistent {key}")
        if not frame.helpful_fix.eq(frame.label.eq("HELP")).all() or not frame.harmful_flip.eq(frame.label.eq("HURT")).all():
            raise AssertionError("Transition annotations disagree.")
        if not frame.total_actual_tokens.eq(frame.base_actual_tokens + frame.action_actual_tokens).all():
            raise AssertionError("Recorded token totals disagree.")
        weights = question_weights(frame)
        oracle = float(100 * np.average(np.maximum(frame.base_correct.astype(float), frame.final_correct.astype(float)), weights=weights))
        info = {"questions": frame.group_id.nunique(), "rows": len(frame), "row_class_counts": frame.label.value_counts().to_dict(),
                "pair_oracle_accuracy_pct_DIAGNOSTIC_ONLY": oracle,
                "duplicate_attempt_rollout_rows": int(frame.duplicated(["dataset", "example_id", "rollout"]).sum())}
        if name.startswith("target"):
            sizes = frame.groupby("group_id").size()
            if not sizes.eq(2).all() or not frame.groupby("group_id").rollout.apply(lambda x: set(x) == {0, 1}).all():
                raise AssertionError("Target rollout coverage changed.")
            initial_counts = frame.groupby("group_id")[[*INITIAL_REQUIRED, "base_correct"]].nunique()
            if not initial_counts.eq(1).all().all():
                raise AssertionError("Target initial inputs differ between rollouts.")
            info["questions_with_different_rollout_labels"] = int(frame.groupby("group_id").label.nunique().gt(1).sum())
        checks[name] = info
        if name in ("source_test", "target_test"):
            probs = frozen["predictor"].probabilities(frame)
            actions = {"Keep initial": np.zeros(len(frame), dtype=bool), "Always recheck": np.ones(len(frame), dtype=bool),
                       "Completion rule": rules(frame)["completion"], "Frozen gate": decide_recheck(probs, frozen["threshold"])}
            for policy, mask in actions.items():
                computed = helper.summarize(frame, mask, probs if policy == "Frozen gate" else None)
                direct = verifier.independent_metrics(frame, mask)
                if not all(np.isclose(computed[k], v, atol=1e-10, rtol=0) for k, v in direct.items() if v is not None):
                    raise AssertionError("Independent policy arithmetic mismatch.")
                policies.append({"partition": name, "policy": policy, **computed})
    # Official keys are used only for audit, never for rule/feature discovery.
    official = pd.read_parquet(ROOT / "outputs/live_study/v0.3.0/raw/commonsenseqa/94630fe30dad47192a8546eb75f094926d47e155/validation.parquet")
    key_map = {question_hash(row["question"]): row["answerKey"] for row in official.to_dict("records")}
    if len(key_map) != len(official):
        raise AssertionError("Official question stems are not unique.")
    grading = {}
    for name in ("target_adaptation_pool", "target_test"):
        frame = parts[name]
        stems = frame["query"].map(lambda text: re.split(r"\n\s*(?:Choices|Options):", text, maxsplit=1)[0].strip())
        keys = stems.map(question_hash).map(key_map)
        if keys.isna().any():
            raise AssertionError("Could not match official answer keys by unique question stem.")
        grading[name] = {}
        for prefix, column in (("base", "base_correct"), ("final", "final_correct")):
            standard = frame[prefix + "_answer"].isin(list("ABCDE"))
            mismatch = int((frame.loc[standard, prefix + "_answer"].eq(keys[standard]) != frame.loc[standard, column]).sum())
            grading[name][prefix] = {"standard_letter_rows": int(standard.sum()), "exact_key_mismatches": mismatch,
                                      "nonstandard_rows": int((~standard).sum()),
                                      "nonstandard_rows_graded_correct": int(frame.loc[~standard, column].sum())}
    pd.DataFrame(policies).to_csv(run / "existing_test_policies_AUDIT_ONLY.csv", index=False)
    result = {"artifact_fingerprints": fingerprints, "partitions": checks, "official_key_audit": grading,
              "new_policies_evaluated_on_tests": False, "parsing": "Public parsed answers/correctness preserved; local FINAL parser is not substituted.",
              "limitations": "MATH equivalence labels cannot be independently regraded without source keys. Nonstandard answer reinterpretation is a scoring change, not a metric correction."}
    save(run / "existing_metrics_audit.json", result)
    if not all(fingerprints.values()) or any(v[phase]["exact_key_mismatches"] for v in grading.values() for phase in ("base", "final")):
        raise AssertionError("Verified artifact/grading inconsistency requires review.")
    return result, policies


def write_report(run, selection, source, adaptation, audited, policies, elapsed):
    primary_name = selection["primary_policy_name"] if selection["primary_policy_kind"] == "classifier" else "Rule: " + selection["primary_policy_name"]
    primary = next(row for row in source if row["policy"] == primary_name)
    frozen = next(row for row in source if row["policy"] == "Frozen gate")
    verdict = "VERIFIED IMPROVEMENT ON SOURCE DEVELOPMENT ONLY" if primary["accuracy_pct"] > frozen["accuracy_pct"] + 1e-10 else "NO VERIFIED IMPROVEMENT"
    lines = ["# Final lightweight accuracy audit", "", "**" + verdict + "**", "",
             "No new policy was evaluated on source-test or target-test. Existing tests were replayed only "
             "to verify historical results, after all new choices were frozen. No inference, downloads, "
             "application changes, model promotion or production artifact writes occurred.", "",
             "Bounded work: reuse all 11 prior nested-CV comparisons; fit one predeclared 20-column "
             "experimental HGB variant (the original 19 plus log visible-character/token ratio, 20 CV fits); "
             "compare six fixed rules; tune one threshold on 244 adaptation questions and confirm on 244 others.", "",
             f"Training-selected classifier: {selection['best_classifier']}. Training-selected rule: {selection['best_rule']}. "
             f"Primary policy selected before development: {primary_name}.", ""]
    def table(title, rows):
        lines.extend([title, "", "| Policy | Accuracy % | HELP | HURT | Recheck % | Avg total tokens | Macro-F1 | Log loss |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|"])
        for row in rows:
            f1 = "N/A" if row["macro_f1"] is None else f"{row['macro_f1']:.6f}"
            loss = "N/A" if row["log_loss"] is None else f"{row['log_loss']:.6f}"
            lines.append(f"| {row['policy']} | {row['accuracy_pct']:.6f} | {row['helpful_fixes_question_equivalents']:.3f} | "
                         f"{row['harmful_flips_question_equivalents']:.3f} | {row['recheck_pct']:.4f} | "
                         f"{row['average_total_tokens']:.3f} | {f1} | {loss} |")
        lines.append("")
    table("Source development: previously inspected descriptive confirmation", source)
    table("Target adaptation: fixed held-out confirmation half, never target-test", adaptation)
    for partition in ("source_test", "target_test"):
        table("Existing " + partition + " results: AUDIT ONLY", [row for row in policies if row["partition"] == partition])
    lines.extend(["HELP/HURT are question-equivalent counts; repeated rollouts share each question's weight. "
                  "Action-only policies have no multiclass probabilities, so macro-F1/log loss are N/A.", "",
                  "Existing target-test oracle is diagnostic only: " + str(audited["partitions"]["target_test"]["pair_oracle_accuracy_pct_DIAGNOSTIC_ONLY"]) +
                  "%. It averages paired max(initial,revised) within each question, rather than selecting the best rollout. "
                  "The old pairs cannot support 80% under the fixed one-recheck action/recorded-outcome scoring.", "",
                  "Current source threshold already exhaustively maximizes source-development HELP-HURT for the fixed score. "
                  "The adaptation confirmation does not select another threshold. Configuration/rule selection retains "
                  "winner optimism; pooled OOF final thresholds are tuning statistics. Development and the adaptation "
                  "pool have been inspected in earlier work. No independent new-test accuracy claim is made.", "",
                  "No transition, weighting, token arithmetic, rollout, standard-choice grading, saved-prediction or "
                  "artifact-fingerprint error was verified. Public replay uses inherited parsed answers and correctness. "
                  "Changing extraction from malformed transcripts would be a separately versioned scoring sensitivity, "
                  "not a silent increase in accuracy. Local-study protocols are excluded.", "",
                  "Next steps: (1) retain v0.2 and present its baseline tie, rare-HURT limitation and oracle honestly; "
                  "(2) if accuracy research continues, use a separate protocol-matched generation study, since these "
                  "recorded target pairs have less than one percentage point of remaining oracle headroom.", "",
                  f"Runtime: {elapsed:.2f} seconds. See integrity.json for original-file preservation.", ""])
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    save(run / "verdict.json", {"verdict": verdict, "primary_source_dev_accuracy_pct": primary["accuracy_pct"],
         "existing_source_dev_accuracy_pct": frozen["accuracy_pct"], "new_target_test_result_claimed": False,
         "verified_scoring_bug": False, "runtime_seconds": elapsed})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_name):
        parser.error("run-name must be a simple directory name")
    run = Path(__file__).resolve().parent / args.run_name
    run.mkdir(exist_ok=False)
    before, started = snapshot(run), time.perf_counter()
    save(run / "original_files_before.json", before)
    save(run / "plan.json", {"new_feature": "initial_log_chars minus initial_log_actual_tokens; initial-only interaction",
         "feature_estimator": "balanced HGB, unchanged prior hyperparameters", "outer_inner_folds": [5, 3],
         "feature_cv_seeds": [42, 142, 143, 144, 145, 146],
         "rules": ["completion", "finalizer", "length_or_missing", "length_or_finalizer", "keep", "always"],
         "adaptation_split_seed": 46, "adaptation_split": "sorted groups then seeded shuffle, first half tune",
         "test_access": "Existing-policy verification only, after source/adaptation selection; no new-policy test scoring",
         "prior_cache_sha256": sha(PRIOR / "cv_predictions.csv"), "script_sha256": sha(__file__)})
    try:
        train = validate_pairs(pd.read_parquet(V2 / "data/prepared/source_train.parquet"))
        frozen = load_artifact(V2 / "models/source_gate.joblib")
        selection = source_selection(train, run)
        print("Training-only choices frozen:", selection["best_classifier"], selection["best_rule"], flush=True)
        source = source_confirmation(train, validate_pairs(pd.read_parquet(V2 / "data/prepared/source_development.parquet")), frozen, selection, run)
        adaptation = adaptation_confirmation(frozen, run)
        audited, policies = existing_results_audit(frozen, run)
        write_report(run, selection, source, adaptation, audited, policies, time.perf_counter() - started)
        print((run / "verdict.json").read_text(), flush=True)
    finally:
        after = snapshot(run)
        changed = [name for name in before if after.get(name) != before[name]]
        added = sorted(set(after) - set(before))
        save(run / "integrity.json", {"original_files_checked": len(before), "changed_or_removed": changed,
             "new_files_outside_audit_run": added, "all_original_files_unchanged": not changed and not added})
        if changed or added:
            raise RuntimeError("Original-file preservation failed.")


if __name__ == "__main__":
    main()
