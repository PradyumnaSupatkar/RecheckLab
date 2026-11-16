"""Evidence-based status and a bounded handoff bundle for one live study.

Only files inside the specified study are considered experiment evidence. The
preserved public replay is never used to fill missing live-study metrics.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import traceback
import zipfile

from .features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION

PARTITIONS = {
    "source_train": "maths", "source_development": "maths", "source_test": "maths",
    "target_adaptation_pool": "commonsenseqa", "target_test": "commonsenseqa",
}
DEFAULT_TARGETS = dict(zip(PARTITIONS, (600, 200, 200, 200, 300)))
OUTCOMES = ("HELP", "HURT", "NO_CHANGE")
ZIP_LIMIT = 10 * 1024 * 1024
ARTIFACT_LIMIT = 2 * 1024 * 1024
SECRET_FIELDS = {"api_key", "access_token", "refresh_token", "password", "authorization",
                 "credentials", "secret", "trusted_key", "trusted_keys", "gold_answer", "normalized_answer"}


def _safe(value):
    """Remove keys and credentials, retaining actual token accounting fields."""
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items() if str(k).lower() not in SECRET_FIELDS}
    if isinstance(value, list):
        return [_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return f"{value} (invalid nonfinite value)"
    if isinstance(value, str):
        value = re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1[redacted]@", value)
        value = re.sub(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s\"']+", r"\1[redacted]", value)
        value = re.sub(r"(?i)((?:api[_-]?key|access[_-]?token|password)[\"']?\s*[:=]\s*[\"']?)[^\s,;\"']+", r"\1[redacted]", value)
    return value


def _json_bytes(value):
    return (json.dumps(_safe(value), indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n").encode("utf-8")


def _read_json(path, errors):
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value
    except (OSError, ValueError, TypeError) as exc:
        errors.append(dict(file=str(path), error=str(exc), error_type=type(exc).__name__))
        return {}


def _read_jsonl(path, errors):
    if not path.exists():
        return []
    rows = []
    try:
        lines = path.read_text(encoding="utf-8-sig").split("\n")
    except OSError as exc:
        errors.append(dict(file=str(path), error=str(exc), error_type=type(exc).__name__))
        return rows
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Expected a JSON object.")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            errors.append(dict(file=str(path), line=line_number, error=str(exc), error_type=type(exc).__name__))
    return rows


def _read_policy_rows(path, errors):
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Expected a list of policy metric rows.")
        return rows
    except (OSError, ValueError, TypeError) as exc:
        errors.append(dict(file=str(path), error=str(exc), error_type=type(exc).__name__))
        return []


def _conclusion(rows, comparisons):
    frozen = next((row for row in rows if row.get("split") == "target_test" and row.get("policy") == "frozen_source_gate"), None)
    if not frozen:
        return "INCONCLUSIVE: no current target-test frozen-gate comparison is available."
    identical = comparisons.get("target_test", {}).get("completion_rule__frozen_source_gate", {}).get("identical_actions")
    if identical is True:
        return "MATCHES: frozen gate and completion rule select identical target-test actions. Accuracy and recheck costs match; gate computation adds measured overhead."
    difference, interval = frozen.get("accuracy_difference_pp_vs_completion"), frozen.get("accuracy_difference_95pct_ci_vs_completion")
    if not isinstance(difference, (int, float)) or not isinstance(interval, list) or len(interval) != 2:
        return "INCONCLUSIVE: a complete paired accuracy comparison and uncertainty are unavailable."
    if interval[0] <= 0 <= interval[1]:
        return f"INCONCLUSIVE: target-test accuracy difference is {difference:.2f} percentage points, with paired 95% interval {interval}; assess logged token, harm and latency trade-offs separately."
    finding = "higher" if interval[0] > 0 else "lower"
    return f"Frozen-gate target accuracy is {finding} than the completion rule ({difference:+.2f} percentage points; paired 95% interval {interval}). This is an accuracy finding; token, harmful-flip and reconstructed-latency costs must also be compared."


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write(path, data):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def validate_study_inputs(root):
    """Use training's read-only journal, key and protocol audit before counting pairs."""
    from .study_training import validate_study_inputs as validate
    return validate(root)


def _elapsed_between(started, ended):
    try:
        start = datetime.fromisoformat(started.replace("Z", "+00:00"))
        end = datetime.fromisoformat(ended.replace("Z", "+00:00"))
        elapsed = (end - start).total_seconds()
        return elapsed if elapsed >= 0 else None
    except (AttributeError, TypeError, ValueError):
        return None


def _call_phase(row):
    phase = row.get("phase")
    return "recheck" if phase in ("revision", "recheck") else phase


def _call_identity(row):
    return row.get("call_id") or (row.get("example_id"), _call_phase(row), row.get("attempt", 1))


def _is_retry(row):
    try:
        return int(row.get("attempt", 1)) > 1
    except (TypeError, ValueError):
        return False


def _call_accounting(calls):
    succeeded = {_call_identity(row): row for row in calls if row.get("status") == "succeeded"}
    outputs = [row.get("output", row.get("response", {})) for row in succeeded.values()]
    def total(key):
        values = [row[key] for row in outputs if isinstance(row, dict) and isinstance(row.get(key), (int, float)) and math.isfinite(row[key])]
        return sum(values) if values else None
    return dict(successful_calls=len(succeeded), actual_prompt_tokens=total("prompt_eval_count"),
                actual_generated_tokens=total("eval_count"), measured_call_wall_seconds=total("wall_seconds"),
                timing_treatment="Measured individual calls include cold load where logged; backend caching is uncontrolled. Policy latency is not measured end to end here.")


def _stage(source, default_reason):
    if isinstance(source, dict) and source.get("status"):
        result = dict(source)
        reasons = result.get("not_run_reasons")
        if not reasons and result["status"] in ("NOT_RUN", "INCOMPLETE", "BLOCKED", "FAILED"):
            result["not_run_reasons"] = [result.get("reason", default_reason)]
        return result
    return {"status": "NOT_RUN", "not_run_reasons": [default_reason]}


def _markdown(status):
    lines = [f"# Live study: {status['overall_status']}", "", status["guide_paragraph"], "",
             f"Generated: {status['generated_at']}", f"Study: `{status['study_root']}`", "",
             "## Stages", "", "| Stage | Status | Evidence / reason |", "|---|---|---|"]
    for name, stage in status["stages"].items():
        reason = stage.get("reason") or "; ".join(stage.get("not_run_reasons", [])) or stage.get("evidence", "See current-run files.")
        lines.append(f"| {name} | {stage['status']} | {str(reason).replace('|', '/').replace(chr(10), ' ')} |")
    lines += ["", "## Planned and completed questions", "",
              "| Partition | Planned | Manifest | Completed unique pairs | HELP | HURT | NO_CHANGE |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, counts in status["counts_by_partition"].items():
        labels = counts["class_counts"]
        lines.append(f"| {name} | {counts['planned_questions']} | {counts['manifest_questions']} | {counts['completed_unique_questions']} | {labels['HELP']} | {labels['HURT']} | {labels['NO_CHANGE']} |")
    collection = status["collection"]
    lines += ["", f"Actual completed pairs: {collection['completed_pairs']}; failed calls: {collection['failed_calls']}; retries: {collection['retry_calls']}; calls without terminal evidence: {collection['unfinished_calls']}.",
              f"Missing successful initial calls: {collection['missing_initial_calls']}; missing successful rechecks: {collection['missing_recheck_calls']}.",
              f"Parsed-answer failures in completed pairs: initial {collection['initial_parse_failures']}, recheck {collection['recheck_parse_failures']}. Unparseable answers count incorrect; transport/runtime failures do not become labeled pairs.",
              "", "## Functional smoke execution", "", "```json", json.dumps(_safe(status["smoke"]), indent=2, ensure_ascii=False), "```",
              "Smoke records are not held-out evaluation. They count toward planned collection only after exact-protocol promotion into the main pair journal.",
              "", "## Protocol, model and reproducibility", "",
              f"Protocol: `{status['protocol']['path']}`", f"Protocol hash: `{status['protocol']['hash'] or 'NOT AVAILABLE'}`",
              "", "```json", json.dumps(_safe(status["protocol"]["details"]), indent=2, ensure_ascii=False), "```", "",
              status["public_pilot_difference"], "", status["protocol"]["numeric_precision_limitation"],
              "", "GSM8K supplies numeric maths with transparent public keys; CommonsenseQA supplies labeled choice questions for independent target transfer. Original benchmark splits and pinned revisions are preserved as recorded below.",
              "", "```json", json.dumps(_safe(status["dataset_selection"]), indent=2, ensure_ascii=False), "```",
              "", "The public SEVRA replay remains a separate study and supplies no missing metrics here.",
              "", "## Training and evaluation", "",
              f"Current conclusion versus completion rule: **{status['completion_rule_conclusion']}**.",
              "", "```json", json.dumps(_safe(status["training_evidence"]), indent=2, ensure_ascii=False), "```", "",
              "Metrics are included only when this run's planned held-out partitions are complete and its evaluation status is COMPLETE. Subset standard deviations are not confidence intervals.",
              "", "## Execution, tests and errors", "", "```json",
              json.dumps(_safe(status["execution"]), indent=2, ensure_ascii=False), "```", "",
              "```json", json.dumps(_safe(status["errors"]), indent=2, ensure_ascii=False), "```", "",
              "## Resume and remaining work", ""]
    lines += [f"- {item}" for item in status["remaining_work"]]
    lines += ["", f"Collection ETA: {status['collection_eta']['description']}", f"ETA basis: {status['collection_eta']['source']}", "", "```powershell"]
    lines += status["resume_commands"] or ["Exact resume command has not yet been recorded by the run controller."]
    lines += ["```", "", "## Diagnostic handoff", "", f"Bundle: `{status['diagnostic_bundle']['path']}`",
              "The bundle excludes trusted keys, credentials, environments, model weights and bulk datasets. It contains bounded examples from this run, available logs, relevant code/tests, and an artifact hash/location; artifacts are included only below 2 MiB.", ""]
    return "\n".join(lines)


def generate_status(project_root, study_root):
    """Write CURRENT_STATUS.md/json and diagnostic_bundle.zip, even on partial runs."""
    project_root, study_root = Path(project_root).resolve(), Path(study_root).resolve()
    destination = project_root / "reports" / "live_study"
    destination.mkdir(parents=True, exist_ok=True)
    errors = []
    plan = _read_json(study_root / "study_plan.json", errors)
    protocol = _read_json(study_root / "protocol.json", errors)
    context = _read_json(study_root / "run_context.json", errors)
    collected = _read_json(study_root / "collection_status.json", errors)
    training = _read_json(study_root / "training_status.json", errors)
    questions = _read_jsonl(study_root / "questions.jsonl", errors)
    calls = _read_jsonl(study_root / "calls.jsonl", errors)
    raw_pairs = _read_jsonl(study_root / "pairs.jsonl", errors)
    audited_pairs = raw_pairs
    pair_audit = {"status": "NOT_RUN", "reason": "No main-journal pair rows are available to validate."}
    if raw_pairs:
        try:
            verified = validate_study_inputs(study_root)[-1]
            verified_by_id = {}
            for row in verified:
                verified_by_id.setdefault(row["example_id"], []).append(row)
            # A call may finish while the audit reads the active journal. Count
            # only this report's original pair snapshot, verified unchanged.
            audited_pairs = [row for row in raw_pairs if row in verified_by_id.get(row.get("example_id"), [])]
            pair_audit = {"status": "COMPLETE", "reason": "Training's read-only protocol/manifest/journal/key audit passed; no fitting was performed."}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            audited_pairs = []
            pair_audit = {"status": "FAILED", "error": str(exc), "error_type": type(exc).__name__}
            errors.append(dict(file=str(study_root / "pairs.jsonl"), affected_function="study_training.validate_study_inputs",
                               error=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc()))
    smoke_status = _read_json(study_root / "smoke" / "collection_status.json", errors)
    smoke_pairs = _read_jsonl(study_root / "smoke" / "pairs.jsonl", errors)
    smoke_calls = _read_jsonl(study_root / "smoke" / "calls.jsonl", errors)
    candidate = _read_json(study_root / "candidate_protocol.json", errors)
    hash_path = study_root / "protocol_hash.txt"
    protocol_hash = hash_path.read_text(encoding="utf-8-sig").strip() if hash_path.exists() else protocol.get("protocol_hash")
    targets = plan.get("planned_questions_by_partition", DEFAULT_TARGETS)
    if not isinstance(targets, dict) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in targets.values()):
        errors.append(dict(file=str(study_root / "study_plan.json"), error="Invalid planned partition counts; default targets remain unmet."))
        targets = DEFAULT_TARGETS
    targets = {name: targets.get(name, 0) for name in PARTITIONS}
    manifest = {}
    for row in questions:
        identifier = row.get("example_id")
        if not identifier or row.get("partition") not in PARTITIONS or identifier in manifest:
            errors.append(dict(file=str(study_root / "questions.jsonl"), error="Missing/duplicate question ID or unrecognized partition.", example_id=identifier))
            continue
        manifest[identifier] = row
    pairs = {}
    rejected_pairs = 0
    for row in audited_pairs:
        identifier = row.get("example_id")
        if identifier not in manifest or identifier in pairs or row.get("label") not in OUTCOMES or (protocol_hash and row.get("protocol_hash") != protocol_hash):
            rejected_pairs += 1
            errors.append(dict(file=str(study_root / "pairs.jsonl"), example_id=identifier,
                               error="Pair has duplicate/unknown question, unrecognized label, or incompatible protocol hash."))
            continue
        pairs[identifier] = row
    matching_calls = [row for row in calls if not protocol_hash or row.get("protocol_hash") == protocol_hash]
    successful = {(row.get("example_id"), _call_phase(row)) for row in matching_calls if row.get("status") == "succeeded"}
    terminal = {_call_identity(row) for row in matching_calls if row.get("status") in ("succeeded", "failed", "resolved_uncertain", "cancelled_no_request")}
    started = {_call_identity(row) for row in matching_calls if row.get("status") == "started"}
    cancelled = {_call_identity(row) for row in matching_calls if row.get("status") == "cancelled_no_request"}
    resolved = {_call_identity(row) for row in matching_calls if row.get("status") == "resolved_uncertain"}
    failed = {_call_identity(row): row for row in matching_calls if row.get("status") == "failed"}
    counts = {}
    for partition, domain in PARTITIONS.items():
        identifiers = {key for key, row in manifest.items() if row["partition"] == partition}
        completed = [pairs[key] for key in identifiers if key in pairs]
        labels = Counter(row["label"] for row in completed)
        counts[partition] = dict(domain=domain, planned_questions=targets[partition], manifest_questions=len(identifiers),
                                 completed_unique_questions=len(completed), completed_pairs=len(completed),
                                 class_counts={label: labels[label] for label in OUTCOMES})
    total_planned = sum(targets.values())
    collection_complete = bool(total_planned) and bool(plan) and all(counts[key]["completed_unique_questions"] >= targets[key] for key in PARTITIONS)
    held_out_complete = all(counts[key]["completed_unique_questions"] >= targets[key] for key in ("source_test", "target_test", "target_adaptation_pool")) and bool(total_planned)
    training_stage = _stage(training.get("training"), "No successful training execution has been recorded for this study.")
    adaptation_stage = _stage(training.get("adaptation"), "Threshold adaptation has not run on the reserved target pool.")
    evaluation_stage = _stage(training.get("evaluation"), "The planned held-out pairs are incomplete; evaluation has not run.")
    if not held_out_complete:
        evaluation_stage = {"status": "NOT_RUN", "not_run_reasons": ["Planned source-test, target-test or target-adaptation pairs are incomplete."]}
    metrics = {}
    policy_rows, comparisons = [], {}
    metrics_path = study_root / "reports" / "predictor_diagnostics.json"
    if held_out_complete and evaluation_stage["status"] == "COMPLETE":
        metrics = _read_json(metrics_path, errors)
        if not metrics or not (study_root / "reports" / "policy_results.csv").exists():
            evaluation_stage = {"status": "NOT_RUN", "not_run_reasons": ["Current-study evaluation metrics or policy results are missing."]}
        else:
            policy_rows = _read_policy_rows(study_root / "reports" / "policy_results.json", errors)
            comparisons = _read_json(study_root / "reports" / "policy_action_comparison.json", errors)
    else:
        metrics_path = None
    source_support = counts["source_train"]["class_counts"]
    missing_classes = [label for label in OUTCOMES if not source_support[label]]
    if missing_classes and training_stage["status"] != "FAILED":
        training_stage = {"status": "INCOMPLETE", "not_run_reasons": ["Source training lacks genuine outcome classes: " + ", ".join(missing_classes)]}
    stages = {
        "implementation": context.get("implementation", {"status": "IMPLEMENTED", "evidence": "Source files are packaged separately from execution evidence."}),
        "collection": {"status": "COMPLETE" if collection_complete else "PARTIAL", "reason": f"{len(pairs)} of {total_planned} planned unique pairs completed."},
        "training_adaptation": {"status": "COMPLETE" if training_stage["status"] == adaptation_stage["status"] == "COMPLETE" else "INCOMPLETE",
                                "reason": "; ".join(training_stage.get("not_run_reasons", []) + adaptation_stage.get("not_run_reasons", [])) or "See recorded training/adaptation execution status."},
        "evaluation": evaluation_stage,
        "live_integration": _stage(context.get("live_integration"), "No independently evaluated live-study gate has been activated; the app retains the public-pilot gate."),
    }
    if pair_audit["status"] == "FAILED":
        stages["collection"] = {"status": "BLOCKED", "reason": "Pair evidence audit failed: " + pair_audit["error"]}
    elif collected.get("status") in ("BLOCKED", "FAILED"):
        stages["collection"] = {"status": collected["status"], "reason": collected.get("error") or "; ".join(str(item) for item in collected.get("errors", [])) or "See collector's exact recorded blocking/failure evidence."}
    complete = (collection_complete and stages["collection"]["status"] == "COMPLETE" and stages["training_adaptation"]["status"] == "COMPLETE" and evaluation_stage["status"] == "COMPLETE"
                and stages["live_integration"]["status"] == "COMPLETE" and not context.get("active_processes") and not errors and not context.get("unresolved_blockers"))
    elapsed = collected.get("elapsed_seconds", context.get("elapsed_seconds"))
    eta_seconds = collected.get("estimated_remaining_seconds", collected.get("eta_seconds"))
    eta_source = "collection_status / observed completed pairs"
    if eta_seconds is None and isinstance(collected.get("estimated_remaining_hours"), (int, float)):
        eta_seconds = collected["estimated_remaining_hours"] * 3600
    if eta_seconds is None and isinstance(elapsed, (int, float)) and elapsed > 0 and pairs:
        eta_seconds = max(0, total_planned - len(pairs)) * elapsed / len(pairs)
    if eta_seconds is None and isinstance(smoke_status.get("estimated_remaining_hours"), (int, float)) and smoke_pairs:
        eta_seconds = smoke_status["estimated_remaining_hours"] * 3600
        eta_source = "functional smoke throughput; preliminary estimate before full collection"
    eta = dict(estimated_remaining_seconds=eta_seconds,
               description=f"Approximately {eta_seconds / 3600:.2f} hours at this run's measured rate; hardware, failures and warm-up can change it." if isinstance(eta_seconds, (int, float)) else "NOT AVAILABLE: no measured completed-pair throughput yet.",
               measured_elapsed_seconds=elapsed if elapsed is not None else smoke_status.get("elapsed_seconds"), source=eta_source)
    commands = context.get("commands", [])
    tests = context.get("tests", [row for row in commands if "unittest" in str(row.get("command", "")) or "pytest" in str(row.get("command", ""))])
    for row in failed.values():
        errors.append({key: row[key] for key in ("example_id", "phase", "call_id", "attempt", "error", "error_type", "failure_kind", "traceback", "timestamp", "timestamp_utc") if key in row})
    errors.extend(context.get("errors", []))
    for key in ("error", "traceback"):
        if training.get(key):
            errors.append(dict(file=str(study_root / "training_status.json"), **{key: training[key]}))
    hashes = {name: _hash(study_root / name) for name in ("study_plan.json", "questions.jsonl", "trusted_keys.jsonl", "protocol.json", "calls.jsonl", "pairs.jsonl", "training_status.json", "collection_status.json") if (study_root / name).is_file()}
    artifact_path = study_root / "models" / "source_gate.joblib"
    artifact = dict(path=str(artifact_path), exists=artifact_path.exists(), sha256=_hash(artifact_path) if artifact_path.exists() else None,
                    size_bytes=artifact_path.stat().st_size if artifact_path.exists() else None, active_in_app=False)
    remaining = []
    if not collection_complete:
        remaining.append(f"Collect {max(0, total_planned - len(pairs))} remaining planned unique pairs using the frozen protocol; recheck every offline question.")
    if missing_classes:
        remaining.append("Obtain genuine source-training support for " + ", ".join(missing_classes) + "; do not fabricate labels or mine held-out outcomes.")
    if stages["training_adaptation"]["status"] != "COMPLETE":
        remaining.append("Run source-only model/threshold selection and reserved target threshold adaptation when class support permits.")
    if evaluation_stage["status"] != "COMPLETE":
        remaining.append("Finish planned held-out collection, then evaluate all policies with paired uncertainty and timing/token accounting.")
    if context.get("active_processes"):
        remaining.append("Resolve the tracked active processes before declaring execution complete.")
    remaining.append("Activate a new live predictor only after artifact, protocol, feature schema and completed evaluation agree.")
    conclusion = _conclusion(policy_rows, comparisons) if evaluation_stage["status"] == "COMPLETE" else "INCONCLUSIVE: independent planned evaluation has not completed."
    guide = (f"The new Qwen live study's main journal contains {len(pairs)} genuine initial/recheck pairs out of {total_planned} planned questions. "
             f"Separate functional smoke records contain {len({row.get('example_id') for row in smoke_pairs})} completed pairs; exact-protocol reuse is tracked separately. "
             f"Its collection is {'complete' if collection_complete else 'partial'}; training/adaptation is {stages['training_adaptation']['status'].lower()} and evaluation is {evaluation_stage['status'].lower().replace('_', ' ')}. "
             "Working generation demonstrates software execution, not that the gate improves on the completion rule. The preserved public replay is reported separately.")
    packages = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "streamlit") if _installed(name)}
    resume = context.get("resume_commands", collected.get("resume_commands"))
    if not resume:
        try:
            run_argument = str(study_root.relative_to(project_root))
        except ValueError:
            run_argument = str(study_root)
        resume = [rf'.\.venv\Scripts\python.exe scripts\collect_live_study.py --study-root "{run_argument}" --minutes 60',
                  rf'.\.venv\Scripts\python.exe scripts\train_live_study.py --study-root "{run_argument}"',
                  rf'.\.venv\Scripts\python.exe scripts\report_live_study.py --study-root "{run_argument}"']
    if isinstance(resume, str):
        resume = [resume]
    smoke = dict(recorded_status=smoke_status, completed_pairs=len({row.get("example_id") for row in smoke_pairs}),
                 protocol_hash=smoke_status.get("protocol_hash"), promoted_unique_pairs=sum(row.get("example_id") in pairs for row in smoke_pairs),
                 partitions=dict(Counter(row.get("partition", "unknown") for row in smoke_pairs)),
                 candidate_protocol_path=str(study_root / "candidate_protocol.json"), candidate_protocol=candidate,
                 call_accounting=_call_accounting(smoke_calls))
    generated_at = datetime.now(timezone.utc).isoformat()
    started_at = context.get("started_at", collected.get("started_at", smoke_status.get("started_at")))
    ended_at = context.get("ended_at", collected.get("ended_at"))
    whole_elapsed = _elapsed_between(started_at, ended_at or generated_at)
    status = dict(schema_version=1, study_version=study_root.name, overall_status="COMPLETE" if complete else "PARTIAL",
                  generated_at=generated_at, study_root=str(study_root), stages=stages,
                  counts_by_partition=counts, planned_by_domain={domain: sum(targets[key] for key, value in PARTITIONS.items() if value == domain) for domain in set(PARTITIONS.values())},
                  collection=dict(planned_questions=total_planned, manifest_unique_questions=len(manifest), completed_unique_questions=len(pairs), completed_pairs=len(pairs),
                                  raw_pair_rows=len(raw_pairs), unverified_pair_rows=len(raw_pairs) - len(audited_pairs), pair_audit=pair_audit,
                                  rejected_pair_rows=rejected_pairs, failed_calls=len(failed), cancelled_no_request_calls=len(cancelled), resolved_uncertain_calls=len(resolved),
                                  retry_calls=len({_call_identity(row) for row in matching_calls if _is_retry(row)} - cancelled),
                                  unfinished_calls=len(started - terminal), missing_initial_calls=sum((key, "initial") not in successful for key in manifest),
                                  missing_recheck_calls=sum((key, "recheck") not in successful for key in manifest),
                                  initial_parse_failures=sum(not str(row.get("base_answer", "")).strip() for row in pairs.values()),
                                  recheck_parse_failures=sum(not str(row.get("final_answer", "")).strip() for row in pairs.values()),
                                  class_counts={label: sum(row["label"] == label for row in pairs.values()) for label in OUTCOMES}, call_accounting=_call_accounting(matching_calls), recorded_status=collected),
                  protocol=dict(path=str(study_root / "protocol.json"), hash=protocol_hash, frozen=bool(protocol and protocol_hash), details=protocol,
                                numeric_precision_limitation="The frozen numeric normalizer uses Python Decimal's default precision 28. Precision-sensitive normalization must be audited; precision-preserving parser/grader changes require a new protocol version."), smoke=smoke,
                  dataset_selection=plan, data_hashes=hashes, artifact=artifact,
                  training_evidence=dict(training=training_stage, adaptation=adaptation_stage, recorded_status=training,
                                         source_train_class_counts=source_support, missing_training_classes=missing_classes,
                                         implemented_feature_contract=dict(feature_names=FEATURE_NAMES, feature_schema_version=FEATURE_SCHEMA_VERSION, feature_schema_fingerprint=FEATURE_SCHEMA_FINGERPRINT),
                                         metrics=metrics, policy_metrics=policy_rows, policy_action_comparison=comparisons),
                  public_pilot_difference="This independently frozen live protocol has its own exact prompts, parser, model digest and call settings. The installed Qwen3:4b alias is a Thinking 2507 Q4_K_M checkpoint; historical requested thinking=false was not verified as effective. Read this run's protocol for actual thinking, budgets and runtime evidence; no historical setting is inferred as known.",
                  completion_rule_conclusion=conclusion,
                  execution=dict(started_at=started_at, ended_at=ended_at, elapsed_seconds=whole_elapsed,
                                 measured_latest_collection_session_seconds=collected.get("elapsed_seconds"),
                                 measured_successful_collection_call_seconds=_call_accounting(matching_calls)["measured_call_wall_seconds"],
                                 timing_basis="Whole-run elapsed is timestamp difference, ending at report generation while active; session time and summed measured successful-call processing are separate.",
                                 commands=commands, tests=tests, hardware=context.get("hardware", {"platform": platform.platform(), "processor": platform.processor() or "unknown"}),
                                 python=context.get("python", sys.version), packages=context.get("packages", packages), code_version=context.get("code_version", "unknown"),
                                 code_hashes=context.get("code_hashes", {}), baseline_preservation=context.get("baseline_preservation", "not yet recorded"),
                                 active_processes=context.get("active_processes", []), attempted_fixes=context.get("attempted_fixes", [])),
                  errors=errors, remaining_work=remaining, resume_commands=resume,
                  collection_eta=eta, guide_paragraph=guide,
                  diagnostic_bundle=dict(path=str(destination / "diagnostic_bundle.zip"), max_size_bytes=ZIP_LIMIT))
    status = _safe(status)
    example_pairs = list(pairs.values()) + [dict(row, collection_phase="smoke") for row in smoke_pairs if row.get("example_id") not in pairs and row.get("label") in OUTCOMES]
    example_failures = list(failed.values()) + [dict(row, collection_phase="smoke") for row in smoke_calls if row.get("status") == "failed"]
    entries = _bundle_entries(project_root, study_root, status, example_pairs, example_failures)
    status["diagnostic_bundle"]["included_files"] = list(entries) + ["CURRENT_STATUS.json", "CURRENT_STATUS.md"]
    json_data, markdown_data = _json_bytes(status), _markdown(status).encode("utf-8")
    capacity = ZIP_LIMIT - len(json_data) - len(markdown_data) - 32768
    omitted = []
    for name in reversed(list(entries)):
        if sum(len(data) for data in entries.values()) <= capacity:
            break
        omitted.append(name)
        del entries[name]
    if omitted:
        status["diagnostic_bundle"]["omitted_to_preserve_size_limit"] = omitted
        status["diagnostic_bundle"]["included_files"] = list(entries) + ["CURRENT_STATUS.json", "CURRENT_STATUS.md"]
        json_data, markdown_data = _json_bytes(status), _markdown(status).encode("utf-8")
    _atomic_write(destination / "CURRENT_STATUS.json", json_data)
    _atomic_write(destination / "CURRENT_STATUS.md", markdown_data)
    bundle_path = destination / "diagnostic_bundle.zip"
    with zipfile.ZipFile(bundle_path.with_name(bundle_path.name + ".tmp"), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("CURRENT_STATUS.json", json_data)
        bundle.writestr("CURRENT_STATUS.md", markdown_data)
        for name, data in entries.items():
            bundle.writestr(name, data)
    bundle_path.with_name(bundle_path.name + ".tmp").replace(bundle_path)
    return status


def _installed(name):
    try:
        importlib.metadata.version(name)
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _bundle_entries(project_root, study_root, status, pairs, failures):
    entries = {"summaries/data_and_splits.json": _json_bytes(dict(counts=status["counts_by_partition"], hashes=status["data_hashes"], datasets=status["dataset_selection"])),
               "summaries/artifact.json": _json_bytes(status["artifact"])}
    examples, support = [], Counter()
    for row in pairs:
        label = row["label"]
        if support[label] < 2:
            examples.append(dict(kind="completed_pair", **_safe(row)))
            support[label] += 1
    for row in failures:
        category = "failure:" + str(row.get("error_type", "unknown"))
        if support[category] < 2:
            examples.append(dict(kind="failed_call", **_safe(row)))
            support[category] += 1
    entries["examples/real_records.jsonl"] = b"".join((json.dumps(row, ensure_ascii=False, default=str) + "\n").encode("utf-8") for row in examples)
    candidates = [(study_root / name, "study/" + name) for name in ("protocol.json", "protocol_hash.txt", "candidate_protocol.json", "backend_runtime_snapshot.json", "model_selection_plan.json", "smoke/protocol.json", "smoke/protocol_hash.txt", "smoke/collection_status.json", "study_plan.json", "collection_status.json", "training_status.json", "run_context.json")]
    candidates += [(path, "study/logs/" + path.name) for path in sorted((study_root / "logs").glob("*")) if path.is_file() and path.suffix in (".txt", ".log")]
    if status["stages"]["evaluation"]["status"] == "COMPLETE":
        candidates += [(path, "study/reports/" + path.name) for path in sorted((study_root / "reports").glob("*")) if path.is_file() and path.suffix in (".json", ".csv")]
    else:
        candidates += [(study_root / "reports" / name, "study/reports/" + name) for name in ("data_preparation_verified.json", "training_metadata.json", "model_selection.json", "model_configurations.json", "model_comparison.csv")]
    candidates += [(path, "source/" + str(path.relative_to(project_root)).replace("\\", "/")) for pattern in
                   ("rechecklab/study*.py", "scripts/*study*.py", "tests/test_study*.py") for path in sorted(project_root.glob(pattern))]
    candidates += [(project_root / name, "source/" + name) for name in ("app.py", "README.md", "DATA_SOURCES.md", "scripts/freeze_live_protocol.py", "rechecklab/live.py", "rechecklab/features.py", "rechecklab/predictor.py", "rechecklab/collector.py", "rechecklab/metrics.py", "tests/test_live_backend.py", "tests/test_artifact_loading.py", "requirements.txt", "requirements-core.txt")]
    budget = 8 * 1024 * 1024 - sum(len(data) for data in entries.values())
    for path, name in candidates:
        if not path.is_file() or path.stat().st_size > 1024 * 1024 or name in entries:
            continue
        data = path.read_bytes()
        if path.suffix == ".json":
            try:
                data = _json_bytes(json.loads(data.decode("utf-8-sig")))
            except (ValueError, UnicodeError):
                data = _safe(data.decode("utf-8", errors="replace")).encode("utf-8")
        else:
            data = _safe(data.decode("utf-8", errors="replace")).encode("utf-8")
        if len(data) <= budget:
            entries[name], budget = data, budget - len(data)
    artifact = Path(status["artifact"]["path"])
    if artifact.is_file() and artifact.stat().st_size < ARTIFACT_LIMIT and artifact.stat().st_size <= budget:
        entries["study/models/source_gate.joblib"] = artifact.read_bytes()
    return entries
