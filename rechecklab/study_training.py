"""Source-only fitting and honest completion checks for the local live study.

This module never calls a solver. It consumes genuine, graded, completed pairs
and a frozen question manifest. Synthetic records belong only in unit tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .data import LABELS, question_hash, transition_label, validate_pairs
from .features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION, INITIAL_REQUIRED
from .metrics import policy_metrics, predictor_metrics
from .predictor import (MODEL_SELECTION_RULE, choose_threshold, decide_recheck, fit_predictor,
                        load_artifact, make_artifact, revision_score, save_artifact)
from .study_protocol import grade_answer, load_protocol, normalize_call
from .study_data import DATA_SCHEMA_VERSION, validate_question_manifest
from .study_validation import PRECISION_LIMITATION, audit_numeric_precision

PARTITIONS = ("source_train", "source_development", "source_test", "target_adaptation_pool", "target_test")
ADAPTATION_BUDGETS = (25, 50, 100, 200)
ADAPTATION_SEEDS = (142, 143, 144, 145, 146)
MINIMUM_CLASS_SUPPORT = 5
THRESHOLD_RULE = ("Question-equal HELP minus HURT utility on source development only; ties use fewer "
                  "recorded recheck tokens, then the larger threshold. Strict score > threshold.")
PLAN_THRESHOLD_OBJECTIVE = "Source-development HELP minus HURT; tie by fewer actual recheck tokens; strict score > threshold"
TOKEN_COMPONENTS = ("base_prompt_eval_count", "base_eval_count", "action_prompt_eval_count", "action_eval_count")
COST_SCOPE = ("Policy tokens and reconstructed latency use the successful initial/recheck calls of completed pairs. "
              "They exclude failed/retried call work and its unknown token usage; collection elapsed time and failure counts "
              "must be reported separately. They are not complete collection cost or measured live end-to-end latency.")
SUPPORT_LIMITATION = ("Five genuine source-training examples per outcome class is only an eligibility floor, "
                      "not evidence of research reliability. Class weights create no new rare events.")


def _json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _records(path, *, allow_missing=False):
    path = Path(path)
    if allow_missing and not path.exists():
        return []
    records = []
    # JSONL is delimited by LF. Unicode line separators can occur inside a
    # valid JSON string and must not split benchmark question records.
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").split("\n"), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name}:{line_number} must contain an object.")
        records.append(value)
    return records


def _digest_records(records):
    # Stable independently of collection/file ordering; excludes test records
    # when used to identify the fitted source artifact.
    ordered = sorted(records, key=lambda row: row["example_id"])
    raw = json.dumps(ordered, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _counts(frame):
    return {label: int((frame.label == label).sum()) if "label" in frame else 0 for label in LABELS}


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or value % 1:
        raise ValueError(f"{name} must be a finite nonnegative integer.")
    return int(value)


def _model_configuration(value):
    """JSON representation of exact nested sklearn hyperparameters."""
    if isinstance(value, BaseEstimator):
        return {"class": type(value).__module__ + "." + type(value).__qualname__,
                "parameters": _model_configuration(value.get_params(deep=False))}
    if isinstance(value, dict):
        return {str(key): _model_configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_model_configuration(item) for item in value]
    if isinstance(value, np.ndarray):
        return _model_configuration(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"Cannot record estimator parameter type {type(value).__name__} faithfully.")


def _selection_plan(root, minimum, seed, budgets, seeds):
    path = root / "model_selection_plan.json"
    if not path.exists():
        return None
    declaration = json.loads(path.read_text(encoding="utf-8"))
    expected = {"minimum_training_examples_per_class": minimum, "seed": seed,
                "feature_schema": FEATURE_SCHEMA_VERSION, "feature_count": len(FEATURE_NAMES),
                "model_selection_rule": MODEL_SELECTION_RULE, "threshold_objective": PLAN_THRESHOLD_OBJECTIVE,
                "adaptation_budgets": list(budgets), "adaptation_seeds": list(seeds),
                "probabilities": "uncalibrated estimates",
                "heldout_rule": "no final evaluation until all planned held-out and adaptation pairs are collected"}
    for key, value in expected.items():
        if declaration.get(key) != value:
            raise ValueError(f"Requested training settings differ from frozen model_selection_plan: {key}.")
    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "declaration": declaration}


def _load_inputs(root):
    protocol, digest = load_protocol(root)
    questions = _records(root / "questions.jsonl")
    if not questions:
        raise ValueError("The frozen question manifest is empty.")
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    if plan.get("data_schema_version") != DATA_SCHEMA_VERSION:
        raise ValueError("Unsupported or missing frozen study data schema.")
    hash_fields = ("data_schema_version", "seed", "planned_questions_by_partition", "manifest_sha256")
    if any(key not in plan for key in hash_fields):
        raise ValueError("Frozen study plan is missing its selection/hash declaration.")
    selection_identity = {key: plan[key] for key in hash_fields}
    serialized_identity = json.dumps(selection_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    if plan.get("study_plan_hash") != hashlib.sha256(serialized_identity.encode("utf-8")).hexdigest():
        raise ValueError("Frozen study plan selection hash differs from its declared contents.")
    expected = plan.get("planned_questions_by_partition")
    if not isinstance(expected, dict) or set(expected) != set(PARTITIONS):
        raise ValueError("study_plan must declare all five planned partition counts.")
    expected = {key: _integer(value, f"planned {key}") for key, value in expected.items()}
    if any(value == 0 for value in expected.values()):
        raise ValueError("All declared study partitions must contain questions.")
    seen_ids, seen_groups = set(), set()
    forbidden = {"gold_answer", "answer", "base_answer", "base_correct", "final_correct", "label"}
    for row in questions:
        for key in ("example_id", "dataset", "domain", "partition", "group_id", "query", "answer_type"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"Manifest requires explicit {key}.")
        if forbidden.intersection(row):
            raise ValueError("Question manifest must not contain answer keys or outcome labels.")
        partition = row["partition"]
        if partition not in PARTITIONS:
            raise ValueError(f"Unknown study partition: {partition}.")
        source = partition.startswith("source_")
        if (row["domain"], row["dataset"], row["answer_type"]) != (
                ("maths", "gsm8k", "number") if source else ("commonsenseqa", "commonsenseqa", "choice")):
            raise ValueError("Manifest domain/dataset/answer type disagrees with its source/target partition.")
        if row["group_id"] != question_hash(row["query"]):
            raise ValueError("Manifest group_id is not the canonical question hash.")
        if row["example_id"] in seen_ids or row["group_id"] in seen_groups:
            raise ValueError("Duplicate question or example identity across study partitions.")
        seen_ids.add(row["example_id"])
        seen_groups.add(row["group_id"])
    manifest = pd.DataFrame(questions)
    actual = {key: int((manifest.partition == key).sum()) for key in PARTITIONS}
    if actual != expected:
        raise ValueError("Frozen manifest counts differ from the declared plan; do not reduce the planned study silently.")
    keys = _records(root / "trusted_keys.jsonl")
    key_lookup = {}
    for key in keys:
        if key.get("example_id") in key_lookup or key.get("example_id") not in seen_ids:
            raise ValueError("Trusted offline keys contain duplicate or unknown question identities.")
        key_lookup[key["example_id"]] = key
    if set(key_lookup) != seen_ids:
        raise ValueError("Trusted offline keys do not cover the entire frozen question manifest.")
    if set(plan["manifest_sha256"]) != {"questions.jsonl", "trusted_keys.jsonl"}:
        raise ValueError("Frozen plan must hash BOTH the question manifest and separate trusted-key manifest.")
    for name, expected_hash in plan["manifest_sha256"].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected_hash:
            raise ValueError(f"Frozen {name} SHA-256 differs from the declared plan.")
    validate_question_manifest(questions, keys, expected)
    identity = {"model": protocol["model"], "model_digest": protocol["model_digest"],
                "action": "recheck", "recheck_protocol": "live-study:" + digest,
                "study_protocol_hash": digest, "live_study_protocol": protocol}
    raw_pairs = _records(root / "pairs.jsonl", allow_missing=True)
    calls = _records(root / "calls.jsonl", allow_missing=True)
    lookup = {row["example_id"]: row for row in questions}
    succeeded, started, terminal, successful_phases = {}, {}, set(), set()
    precision_counts = {"EXACT": 0, "UNPARSEABLE": 0, "NOT_APPLICABLE": 0}
    for event in calls:
        call_id = event.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise ValueError("Call journal contains a missing call identity.")
        question = lookup.get(event.get("example_id"))
        if question is None or event.get("group_id") != question["group_id"] or event.get("partition") != question["partition"]:
            raise ValueError("Call journal question/group/partition differs from the frozen manifest.")
        phase = event.get("phase")
        if phase not in ("initial", "recheck") or event.get("protocol_hash") != digest or event.get("seed") != protocol["generation"][phase + "_seed"]:
            raise ValueError("Call journal phase/protocol/seed differs from the frozen generation protocol.")
        if event.get("status") == "started":
            if call_id in started:
                raise ValueError("Duplicate started call journal identity.")
            started[call_id] = event
        elif event.get("status") in ("succeeded", "failed", "resolved_uncertain", "cancelled_no_request"):
            initial_event = started.get(call_id)
            if initial_event is None or call_id in terminal:
                raise ValueError("Unmatched or duplicate terminal call journal identity.")
            for key in ("example_id", "group_id", "partition", "phase", "protocol_hash", "seed", "attempt", "session_id", "smoke"):
                if event.get(key) != initial_event.get(key):
                    raise ValueError(f"Terminal call journal event differs from its started call: {key}.")
            terminal.add(call_id)
        else:
            raise ValueError("Unknown call journal event status.")
        if event.get("status") == "succeeded":
            question_phase = (event["example_id"], event["phase"])
            if question_phase in successful_phases:
                raise ValueError("Multiple successful calls for one fixed study question/phase.")
            successful_phases.add(question_phase)
            try:
                precision = audit_numeric_precision(protocol, event.get("response"), question["answer_type"])
            except ValueError as exc:
                raise ValueError(f"Successful call {call_id} numeric precision audit failed: {exc}") from exc
            precision_counts[precision["status"]] += 1
            succeeded[call_id] = event
    for example_id, key in key_lookup.items():
        answer_type = lookup[example_id]["answer_type"]
        if key.get("answer_type") != answer_type or "gold_answer" not in key:
            raise ValueError("Trusted key type/answer is missing or differs from its frozen question.")
        if answer_type == "choice" and str(key["gold_answer"]).strip().upper() not in ("A", "B", "C", "D", "E"):
            raise ValueError("Trusted choice key must be one explicit A-E option.")
        grade_answer(None, key["gold_answer"], answer_type)
    used = set()
    for row in raw_pairs:
        example_id = row.get("example_id")
        if example_id not in lookup or example_id in used:
            raise ValueError("Pairs must contain exactly one completed pair per known question; duplicate/foreign pair found.")
        used.add(example_id)
        question = lookup[example_id]
        for key in ("group_id", "query", "dataset"):
            if row.get(key) != question[key]:
                raise ValueError(f"Pair {example_id} {key} differs from the frozen manifest.")
        for key in ("model", "model_digest", "action", "recheck_protocol"):
            if row.get(key) != identity[key]:
                raise ValueError(f"Pair {example_id} has incompatible {key}.")
        if row.get("protocol_hash") != digest:
            raise ValueError("Pair protocol hash differs from the frozen study protocol.")
        trusted = key_lookup[example_id]
        if trusted.get("answer_type") != question["answer_type"]:
            raise ValueError("Trusted key answer type differs from its frozen question.")
        # Re-validate each recorded successful backend response, then re-grade
        # its parsed answer offline. A hand-written label is not sufficient
        # evidence for training a rare class.
        for phase, prefix, answer_key, response_key, parse_key, correct_key in (
                ("initial", "base", "base_answer", "base_response", "base_parse_valid", "base_correct"),
                ("recheck", "action", "final_answer", "intervention_response", "final_parse_valid", "final_correct")):
            event = succeeded.get(row.get(prefix + "_call_id"))
            if event is None or event.get("example_id") != example_id or event.get("phase") != phase or event.get("protocol_hash") != digest:
                raise ValueError(f"Pair {example_id} lacks its matching successful {phase} call journal evidence.")
            if event.get("seed") != protocol["generation"][phase + "_seed"]:
                raise ValueError("Successful call seed differs from the frozen generation protocol.")
            _, output = normalize_call(protocol, question["query"], question["answer_type"], event.get("response", {}))
            if output != event.get("output"):
                raise ValueError("Saved normalized call output differs from its actual backend response.")
            if row.get(answer_key) != output["answer"] or row.get(response_key) != output["response"] or row.get(parse_key) is not output["parse_valid"]:
                raise ValueError("Pair answer/response/parsing metadata differs from its successful call.")
            expected_correct = grade_answer(output["answer"] or None, trusted.get("gold_answer"), question["answer_type"])
            if row.get(correct_key) is not expected_correct:
                raise ValueError("Pair correctness differs from strict offline grading against its trusted benchmark key.")
            for pair_key, output_key in ((prefix + "_actual_tokens", "tokens"), (prefix + "_prompt_eval_count", "prompt_eval_count"),
                                         (prefix + "_eval_count", "eval_count"), (prefix + "_wall_seconds", "wall_seconds")):
                if row.get(pair_key) != output[output_key]:
                    raise ValueError(f"Pair {pair_key} differs from its actual successful call.")
            done_key = "base_done_reason" if phase == "initial" else "intervention_done_reason"
            if row.get(done_key) != output["done_reason"]:
                raise ValueError("Pair completion reason differs from its actual successful call.")
        if row.get("label") != transition_label(row.get("base_correct"), row.get("final_correct")):
            raise ValueError("Recorded outcome label disagrees with the correctness transition.")
        for key in ("base_wall_seconds", "action_wall_seconds"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Pair {example_id} requires measured finite nonnegative {key}.")
        for key in TOKEN_COMPONENTS:
            _integer(row.get(key), key)
        for stage, first, second in (("base", "base_prompt_eval_count", "base_eval_count"),
                                     ("action", "action_prompt_eval_count", "action_eval_count")):
            if _integer(row.get(stage + "_actual_tokens"), stage + "_actual_tokens") != row[first] + row[second]:
                raise ValueError(f"Pair {example_id} {stage} token total disagrees with actual call counts.")
        if row.get("base_finalizer_used") is not False:
            raise ValueError("The frozen study executes no initial finalizer.")
        if row.get("action_finalizer_used") is not False:
            raise ValueError("The frozen study executes no recheck finalizer.")
        row["partition"], row["domain"] = question["partition"], question["domain"]
    pairs = validate_pairs(pd.DataFrame(raw_pairs)) if raw_pairs else pd.DataFrame(columns=[*INITIAL_REQUIRED, "label", "group_id", "partition"])
    pairs.attrs["numeric_precision_audit"] = {"successful_call_status_counts": precision_counts,
                                             "affected_precision_loss_calls": 0,
                                             "limitation": PRECISION_LIMITATION,
                                             "scope": "All successful journal responses, including calls not yet forming a completed pair."}
    return protocol, digest, plan, manifest, pairs, identity, raw_pairs


def validate_study_inputs(root):
    """Read-only validation shared by training and the diagnostic handoff.

    Return (protocol, protocol_hash, plan, question_manifest_dataframe,
    verified_pairs_dataframe, artifact_protocol_contract, raw_pair_records).
    Empty completed pairs are valid. Invalid evidence raises ValueError; this
    function never writes files, fits a model, scores a policy or calls Ollama.
    """
    return _load_inputs(Path(root).resolve())


def _initial(frame):
    # The model never receives future responses, correctness labels or keys.
    return frame.loc[:, [*INITIAL_REQUIRED, "model", "model_digest", "action", "recheck_protocol"]].copy()


def _measured_probabilities(predictor, frame):
    probabilities, elapsed = [], []
    for index in range(len(frame)):
        start = perf_counter()
        probabilities.append(predictor.probabilities(_initial(frame.iloc[[index]]))[0])
        elapsed.append(perf_counter() - start)
    return np.asarray(probabilities), np.asarray(elapsed)


def _paired_interval(frame, decisions, completion, repeats, seed):
    final = frame.final_correct.to_numpy(dtype=float)
    base = frame.base_correct.to_numpy(dtype=float)
    difference = np.where(decisions, final, base) - np.where(completion, final, base)
    grouped = pd.DataFrame({"group": frame.group_id, "difference": difference}).groupby("group").difference.mean().to_numpy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(grouped, size=(repeats, len(grouped)), replace=True).mean(axis=1)
    return float(100 * grouped.mean()), np.quantile(100 * sampled, [.025, .975]).tolist()


def _policy_row(frame, decisions, completion, overhead, name, split, repeats, seed):
    metrics = policy_metrics(frame, decisions)
    difference, interval = _paired_interval(frame, decisions, completion, repeats, seed)
    latency = frame.base_wall_seconds.to_numpy(dtype=float) + decisions * frame.action_wall_seconds.to_numpy(dtype=float) + overhead
    return {"split": split, "policy": name, **metrics,
            "accuracy_difference_pp_vs_completion": difference,
            "accuracy_difference_95pct_ci_vs_completion": interval,
            "average_reconstructed_latency_seconds": float(latency.mean()),
            "average_predictor_overhead_seconds": float(np.mean(overhead)),
            "average_measured_initial_call_seconds": float(frame.base_wall_seconds.mean()),
            "average_measured_recheck_call_seconds": float(frame.action_wall_seconds.mean()),
            "cost_scope": COST_SCOPE,
            "latency_method": "Recorded initial wall time + selected recorded recheck wall time + measured per-question gate overhead; offline reconstruction, not deployed end-to-end timing.",
            "paired_ci_method": "Question bootstrap of paired selected-correctness difference against the completion rule."}


def _predictions(frame, probabilities, frozen, completion, overhead):
    result = frame[["example_id", "group_id", "dataset", "domain", "partition", "label", "base_correct",
                    "final_correct", "base_done_reason", "base_actual_tokens", "action_actual_tokens",
                    "base_wall_seconds", "action_wall_seconds"]].copy()
    for index, label in enumerate(LABELS):
        result["p_" + label] = probabilities[:, index]
    result["revision_score"] = revision_score(probabilities)
    result["frozen_gate_recheck"] = frozen
    result["completion_recheck"] = completion
    result["predictor_overhead_seconds"] = overhead
    return result


def _evaluate(root, parts, predictor, threshold, candidates, budgets, seeds, repeats, seed):
    reports = root / "reports"
    policies, adaptation_rows, diagnostics, comparisons, candidate_rows = [], [], {}, {}, []
    evaluated = {}
    for split in ("source_test", "target_test"):
        frame = parts[split]
        probabilities, overhead = _measured_probabilities(predictor, frame)
        frozen = decide_recheck(probabilities, threshold)
        completion = frame.base_done_reason.eq("length").to_numpy(dtype=bool)
        decisions = {"keep_initial": np.zeros(len(frame), dtype=bool), "always_recheck": np.ones(len(frame), dtype=bool),
                     "completion_rule": completion, "frozen_source_gate": frozen}
        for name, action in decisions.items():
            policies.append(_policy_row(frame, action, completion, overhead if name == "frozen_source_gate" else np.zeros(len(frame)),
                                        name, split, repeats, seed))
        diagnostics[split] = {"overall": predictor_metrics(frame, probabilities, bootstrap_repeats=repeats, seed=seed),
                              "completion_strata": {}}
        for done_reason in ("stop", "length", "unknown"):
            mask = frame.base_done_reason.eq(done_reason).to_numpy()
            if mask.any():
                stratum = frame.loc[mask].reset_index(drop=True)
                diagnostics[split]["completion_strata"][done_reason] = {
                    "predictor": predictor_metrics(stratum, probabilities[mask]),
                    "policies": [_policy_row(stratum, action[mask], completion[mask],
                                              overhead[mask] if name == "frozen_source_gate" else np.zeros(mask.sum()),
                                              name, split, repeats, seed) for name, action in decisions.items()]}
            else:
                diagnostics[split]["completion_strata"][done_reason] = {"status": "NOT_RUN", "reason": "No collected held-out questions in this completion stratum.", "questions": 0}
        comparisons[split] = {f"{first}__{second}": {"identical_actions": bool(np.array_equal(decisions[first], decisions[second])),
                                                   "different_questions": int(np.count_nonzero(decisions[first] != decisions[second]))}
                              for index, first in enumerate(decisions) for second in list(decisions)[index + 1:]}
        prediction_frame = _predictions(frame, probabilities, frozen, completion, overhead)
        evaluated[split] = (frame, probabilities, overhead, completion, frozen, prediction_frame)
        for name, candidate in candidates.items():
            candidate_rows.append({"split": split, "model": name, "selected": name == predictor.model_name,
                                   **predictor_metrics(frame, candidate.probabilities(_initial(frame)))})
    pool = parts["target_adaptation_pool"].sort_values("group_id").reset_index(drop=True)
    pool_probabilities = predictor.probabilities(_initial(pool)) if len(pool) else np.empty((0, 3))
    target, target_probabilities, overhead, completion, frozen, target_predictions = evaluated["target_test"]
    for budget in budgets:
        for subset_seed in seeds:
            if len(pool) < budget:
                adaptation_rows.append({"budget": budget, "seed": subset_seed, "status": "NOT_RUN",
                                        "reason": f"Only {len(pool)} completed target-adaptation pairs; {budget} required."})
                continue
            indices = np.random.default_rng(subset_seed).choice(len(pool), size=budget, replace=False)
            subset = pool.iloc[indices].reset_index(drop=True)
            adapted_threshold = choose_threshold(subset, revision_score(pool_probabilities[indices]))
            actions = decide_recheck(target_probabilities, adapted_threshold)
            row = _policy_row(target, actions, completion, overhead, f"adapted_budget_{budget}_seed_{subset_seed}",
                              "target_test", repeats, seed)
            row.update({"budget": budget, "seed": subset_seed, "status": "COMPLETE", "threshold": adapted_threshold,
                        "adaptation_class_counts": _counts(subset), "adaptation_example_ids": subset.example_id.tolist(),
                        "identical_to_frozen_actions": bool(np.array_equal(actions, frozen)),
                        "identical_to_completion_actions": bool(np.array_equal(actions, completion)),
                        "changed_actions_vs_frozen": int(np.count_nonzero(actions != frozen)),
                        "classifier_updated": False, "probabilities_recalibrated": False})
            row["completion_strata"] = {}
            for done_reason in ("stop", "length", "unknown"):
                mask = target.base_done_reason.eq(done_reason).to_numpy()
                row["completion_strata"][done_reason] = (
                    _policy_row(target.loc[mask].reset_index(drop=True), actions[mask], completion[mask], overhead[mask],
                                row["policy"], "target_test", repeats, seed)
                    if mask.any() else {"status": "NOT_RUN", "reason": "No held-out questions in this completion stratum.", "questions": 0})
            adaptation_rows.append(row)
            target_predictions[f"adapted_{budget}_{subset_seed}_recheck"] = actions
    adaptation_summary = []
    for budget in budgets:
        rows = [row for row in adaptation_rows if row["budget"] == budget and row["status"] == "COMPLETE"]
        summary = {"budget": budget, "completed_subsets": len(rows), "planned_subsets": len(seeds),
                   "status": "COMPLETE" if len(rows) == len(seeds) else "NOT_RUN"}
        if rows:
            for key in ("accuracy_pct", "accuracy_difference_pp_vs_completion", "harmful_flip_pct_of_all_questions",
                        "helpful_fix_pct_of_all_questions", "recheck_pct", "average_total_tokens",
                        "average_reconstructed_latency_seconds", "threshold"):
                values = [row[key] for row in rows]
                summary[key + "_mean"] = float(np.mean(values))
                summary[key + "_std_across_subsets"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary["uncertainty"] = "Mean and sample standard deviation across adaptation subsets; each subset also has a paired question-bootstrap interval on the same target test."
        else:
            summary["reason"] = next(row["reason"] for row in adaptation_rows if row["budget"] == budget)
        adaptation_summary.append(summary)
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(policies).to_csv(reports / "policy_results.csv", index=False)
    pd.DataFrame(adaptation_rows).to_csv(reports / "adaptation_runs.csv", index=False)
    pd.DataFrame(adaptation_summary).to_csv(reports / "adaptation_summary.csv", index=False)
    evaluated["source_test"][-1].to_csv(reports / "source_predictions.csv", index=False)
    target_predictions.to_csv(reports / "target_predictions.csv", index=False)
    _json(reports / "policy_results.json", policies)
    _json(reports / "adaptation_runs.json", adaptation_rows)
    _json(reports / "adaptation_summary.json", adaptation_summary)
    _json(reports / "predictor_diagnostics.json", diagnostics)
    _json(reports / "policy_action_comparison.json", comparisons)
    _json(reports / "model_comparison.json", candidate_rows)
    return adaptation_rows, adaptation_summary


def train_study(root, seed=42, minimum_per_class=MINIMUM_CLASS_SUPPORT, *,
                adaptation_budgets=ADAPTATION_BUDGETS, adaptation_seeds=ADAPTATION_SEEDS, bootstrap_repeats=600):
    """Fit only when genuine source support exists; always persist honest status.

    Missing class support is an ordinary INCOMPLETE stage, not an exception.
    Any invalid input is FAILED and no model is fitted. Neither test labels nor
    target-adaptation labels can select the classifier or frozen threshold.
    """
    root = Path(root).resolve()
    project = Path(__file__).resolve().parents[1]
    preserved = [project / name for name in ("data", "models", "reports", "outputs/v0.2")]
    if root == project or any(root == path.resolve() or path.resolve() in root.parents for path in preserved):
        raise ValueError("Use a separate versioned study directory; preserve the public pilot data/models/reports.")
    root.mkdir(parents=True, exist_ok=True)
    status = {"schema_version": 1, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "status": "INCOMPLETE",
              "training": {"status": "INCOMPLETE", "not_run_reasons": []},
              "evaluation": {"status": "NOT_RUN", "not_run_reasons": []},
              "adaptation": {"status": "NOT_RUN", "not_run_reasons": []},
              "artifact": None, "probabilities": "Uncalibrated three-class classifier outputs, not correctness probabilities.",
              "limitations": [SUPPORT_LIMITATION, PRECISION_LIMITATION, COST_SCOPE],
              "predeclared": {"class_order": list(LABELS), "minimum_source_train_per_class": minimum_per_class,
                              "feature_names": list(FEATURE_NAMES), "feature_schema_version": FEATURE_SCHEMA_VERSION,
                              "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
                              "model_selection_rule": MODEL_SELECTION_RULE, "threshold_selection_rule": THRESHOLD_RULE,
                              "harm_weight": 1.0, "token_penalty": 0.0,
                              "adaptation_budgets": list(adaptation_budgets), "adaptation_seeds": list(adaptation_seeds),
                              "source_seed": seed, "bootstrap_repeats": bootstrap_repeats}}
    try:
        if isinstance(minimum_per_class, bool) or not isinstance(minimum_per_class, int) or minimum_per_class < MINIMUM_CLASS_SUPPORT:
            raise ValueError("The predeclared study requires at least five source-training examples of EACH outcome class.")
        if isinstance(bootstrap_repeats, bool) or not isinstance(bootstrap_repeats, int) or bootstrap_repeats < 1:
            raise ValueError("Paired bootstrap requires a positive repetition count.")
        budgets = tuple(_integer(value, "adaptation budget") for value in adaptation_budgets)
        seeds = tuple(_integer(value, "adaptation seed") for value in adaptation_seeds)
        if not budgets or not seeds or any(budget < 1 for budget in budgets) or len(set(budgets)) != len(budgets) or len(set(seeds)) != len(seeds):
            raise ValueError("Adaptation budgets and seeds must be nonempty and unique; budgets must be positive.")
        selection_plan = _selection_plan(root, minimum_per_class, seed, budgets, seeds)
        status["model_selection_plan"] = selection_plan
        protocol, digest, plan, manifest, pairs, identity, raw_pairs = validate_study_inputs(root)
        planned = plan["planned_questions_by_partition"]
        completed = {name: int((pairs.partition == name).sum()) for name in PARTITIONS}
        parts = {name: pairs.loc[pairs.partition == name].reset_index(drop=True) for name in PARTITIONS}
        source_counts = _counts(parts["source_train"])
        status.update({"protocol_hash": digest, "protocol_contract": identity,
                       "study_plan_hash": plan["study_plan_hash"],
                       "planned_questions_by_partition": planned, "completed_questions_by_partition": completed,
                       "completed_pairs": len(pairs), "planned_pairs": len(manifest),
                       "source_train_class_counts": source_counts,
                       "class_counts_by_partition": {name: _counts(frame) for name, frame in parts.items()},
                       "questions_manifest_sha256": hashlib.sha256((root / "questions.jsonl").read_bytes()).hexdigest(),
                       "completed_pairs_content_sha256": _digest_records(raw_pairs)})
        status["numeric_precision_audit"] = pairs.attrs["numeric_precision_audit"]
        inadequate = {label: count for label, count in source_counts.items() if count < minimum_per_class}
        reasons = [f"Source training {label} support is {count}; at least {minimum_per_class} genuine completed pairs required."
                   for label, count in inadequate.items()]
        if parts["source_development"].empty:
            reasons.append("No completed source-development pairs are available for source-only model/threshold selection.")
        status["training"].update({"inadequate_classes": inadequate, "source_development_class_counts": _counts(parts["source_development"]),
                                    "not_run_reasons": reasons})
        if reasons:
            status["evaluation"]["not_run_reasons"] = ["Source predictor fitting has not run."]
            status["adaptation"]["not_run_reasons"] = ["No fixed source classifier/threshold exists."]
            return status
        source_records = [row for row in raw_pairs if row["partition"] in ("source_train", "source_development")]
        source_hash = _digest_records(source_records)
        predictor, selection, candidates = fit_predictor(parts["source_train"], parts["source_development"], seed=seed, return_candidates=True)
        if set(predictor.estimator.classes_) != set(LABELS):
            raise ValueError("Study estimator does not contain all three fitted outcome classes; refuse artifact.")
        dev_probabilities = predictor.probabilities(_initial(parts["source_development"]))
        threshold = choose_threshold(parts["source_development"], revision_score(dev_probabilities))
        bundle = make_artifact(predictor, threshold, protocol=identity)
        source_complete = all(completed[name] == planned[name] for name in ("source_train", "source_development"))
        configurations = {name: _model_configuration(candidate.estimator) for name, candidate in candidates.items()}
        feature_source = Path(importlib.import_module("rechecklab.features").__file__)
        bundle["study_training"] = {"status": "COMPLETE" if source_complete else "PARTIAL", "source_fit_content_sha256": source_hash,
                                    "source_train_class_counts": source_counts, "source_development_class_counts": _counts(parts["source_development"]),
                                    "source_seed": seed, "model_selection_rule": MODEL_SELECTION_RULE,
                                    "threshold_selection_rule": THRESHOLD_RULE, "target_labels_used_for_fitting": False,
                                    "test_labels_used_for_selection": False,
                                    "model_selection_plan_sha256": selection_plan["sha256"] if selection_plan else None,
                                    "model_configuration": configurations[predictor.model_name],
                                    "feature_transformations": {"implementation": "rechecklab.features.extract_features",
                                                                "source_sha256": hashlib.sha256(feature_source.read_bytes()).hexdigest(),
                                                                "inputs": list(INITIAL_REQUIRED),
                                                                "estimator_preprocessing": configurations[predictor.model_name]},
                                    "limitations": [SUPPORT_LIMITATION, PRECISION_LIMITATION, COST_SCOPE]}
        artifact_path = save_artifact(bundle, root / "models" / "source_gate.joblib")
        artifact_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        restored = load_artifact(artifact_path, expected_protocol=identity, expected_sha256=artifact_digest)
        if restored["threshold"] != threshold or not np.array_equal(restored["predictor"].probabilities(_initial(parts["source_development"])), dev_probabilities):
            raise ValueError("Production artifact roundtrip changed probabilities or the frozen source threshold.")
        status["training"].update({"status": "COMPLETE" if source_complete else "PARTIAL", "model": predictor.model_name,
                                    "threshold": threshold, "source_fit_content_sha256": source_hash,
                                    "artifact_roundtrip_verified": True, "not_run_reasons": [],
                                    "limitations": [SUPPORT_LIMITATION, PRECISION_LIMITATION, COST_SCOPE,
                                                    "Held-out bootstrap intervals condition on this fitted model; they exclude training/model-selection uncertainty."] +
                                                   ["Source-development lacks " + label + "; its class-selection evidence is unavailable."
                                                    for label, count in _counts(parts["source_development"]).items() if count == 0]})
        status["artifact"] = {"path": str(artifact_path.relative_to(root)), "sha256": artifact_digest,
                              "live_activation_eligible": False,
                              "activation_reason": "Study evaluation and explicit protocol-matched activation are separate; writing an artifact does not activate live inference."}
        _json(root / "reports" / "model_selection.json", selection)
        _json(root / "reports" / "model_configurations.json", configurations)
        _json(root / "reports" / "training_metadata.json", {**bundle["study_training"], "model": predictor.model_name,
                                                            "threshold": threshold, "protocol_contract": identity,
                                                            "feature_names": list(FEATURE_NAMES), "class_order": list(LABELS),
                                                            "feature_schema_version": FEATURE_SCHEMA_VERSION,
                                                            "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
                                                            "artifact_sha256": artifact_digest})
        required_evaluation_partitions = ("source_test", "target_test", "target_adaptation_pool") if selection_plan else ("source_test", "target_test")
        heldout_reasons = [f"{name} has {completed[name]}/{planned[name]} completed pairs; planned held-out/adaptation collection must finish before study evaluation."
                           for name in required_evaluation_partitions if completed[name] != planned[name]]
        if heldout_reasons:
            status["status"] = "PARTIAL"
            status["evaluation"]["not_run_reasons"] = heldout_reasons
            status["adaptation"]["not_run_reasons"] = ["Held-out study evaluation has not run."]
            return status
        rows, summaries = _evaluate(root, parts, predictor, threshold, candidates, budgets, seeds, bootstrap_repeats, seed)
        status["evaluation"] = {"status": "COMPLETE", "not_run_reasons": [],
                                "source_test_questions": completed["source_test"], "target_test_questions": completed["target_test"],
                                "conditional_on_partial_source_training": not source_complete,
                                "reports_directory": "reports", "selected_before_test_scoring": True}
        unavailable = [f"Budget {row['budget']}, seed {row['seed']}: {row['reason']}" for row in rows if row["status"] != "COMPLETE"]
        status["adaptation"] = {"status": "COMPLETE" if not unavailable else "PARTIAL" if any(row["status"] == "COMPLETE" for row in rows) else "NOT_RUN",
                                "not_run_reasons": unavailable, "classifier_updated": False, "probabilities_recalibrated": False,
                                "completed_runs": sum(row["status"] == "COMPLETE" for row in rows), "planned_runs": len(budgets) * len(seeds),
                                "budget_summary": summaries}
        status["status"] = "COMPLETE" if all(completed[name] == planned[name] for name in PARTITIONS) and not unavailable else "PARTIAL"
        return status
    except Exception as exc:
        status["status"] = "FAILED"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["training"]["not_run_reasons"].append(status["error"])
        if status["training"]["status"] == "INCOMPLETE":
            status["training"]["status"] = "FAILED"
        status["evaluation"] = {"status": "NOT_RUN", "not_run_reasons": [status["error"]]}
        status["adaptation"] = {"status": "NOT_RUN", "not_run_reasons": [status["error"]]}
        return status
    finally:
        _json(root / "training_status.json", status)
