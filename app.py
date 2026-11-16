"""Minimal recorded and experimental live demo. Launch from the README."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import streamlit as st

from rechecklab.data import LABELS
from rechecklab import live
from rechecklab.live_session import execute_once, live_input_key
from rechecklab.features import INITIAL_REQUIRED, extract_features
from rechecklab.predictor import decide_recheck, validate_artifact
from scripts.verify_recorded_policies import verify_recorded_policies

ROOT = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "outputs" / "v0.2"
INITIAL_SIGNALS = INITIAL_REQUIRED
DISPLAY_OUTCOMES = ("HELP", "HURT", "NO_CHANGE")
REPLAY_FILES = (
    "reports/experiment_metadata.json", "reports/split_manifest.json",
    "reports/policy_results.csv", "reports/adaptation_summary.csv",
    "reports/adaptation_runs.csv", "reports/model_comparison.csv",
    "reports/target_predictions.csv", "reports/target_features.csv", "models/source_gate.joblib",
    "reports/adaptation_curve.png",
)
MATCHED_ARTIFACTS = (
    "models/source_gate.joblib", "reports/policy_results.csv",
    "reports/target_predictions.csv", "reports/target_features.csv",
    "reports/split_manifest.json", "reports/adaptation_summary.csv",
    "reports/adaptation_runs.csv", "reports/model_comparison.csv",
    "reports/adaptation_curve.png",
)


def initial_frame(record):
    """Project a recorded row onto the same strict pre-recheck input schema."""
    if isinstance(record, pd.DataFrame):
        missing = set(INITIAL_SIGNALS) - set(record.columns)
        if missing:
            raise ValueError("Missing initial signals: " + ", ".join(sorted(missing)))
        frame = record.loc[:, list(INITIAL_SIGNALS)].copy()
    else:
        missing = set(INITIAL_SIGNALS) - set(record.keys())
        if missing:
            raise ValueError("Missing initial signals: " + ", ".join(sorted(missing)))
        frame = pd.DataFrame([{key: record[key] for key in INITIAL_SIGNALS}])
    extract_features(frame)  # Shared validation; missing metadata is never imputed.
    return frame


def predict_initial(bundle, record):
    frame = initial_frame(record)
    if len(frame) != 1:
        raise ValueError("The demo predicts one recorded initial answer at a time.")
    # Solver and specified-recheck identity are known before the intervention.
    # Retain them for the predictor's compatibility check, never as features.
    for key in ("model", "recheck_protocol", "action", "model_digest"):
        if key in record:
            frame[key] = record[key] if isinstance(record, pd.DataFrame) else [record[key]]
    if "action_description" in record and "action_description_sha256" in bundle["protocol"]:
        descriptions = record["action_description"] if isinstance(record, pd.DataFrame) else pd.Series([record["action_description"]])
        if not descriptions.map(lambda value: isinstance(value, str)).all():
            raise ValueError("The recorded action description must be text.")
        frame["action_description"] = descriptions.to_numpy()
        frame["action_description_sha256"] = descriptions.map(
            lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
        ).to_numpy()
    probabilities = bundle["predictor"].probabilities(frame)
    decisions = decide_recheck(probabilities, bundle["threshold"], bundle["harm_weight"])
    return probabilities[0], bool(decisions[0])


def validate_replay(metadata, bundle, predictions):
    """Reject old artifacts and any mixture of this run's model and result files."""
    if "recheck_protocol" not in metadata:
        raise ValueError("Results are missing the recheck protocol compatibility metadata.")
    validate_artifact(bundle, expected_protocol=metadata["recheck_protocol"])
    checks = {
        "artifact_schema_version": bundle["artifact_schema_version"],
        "feature_schema_version": bundle["feature_schema_version"],
        "feature_schema_fingerprint": bundle["feature_schema_fingerprint"],
        "class_order": bundle["class_order"],
        "features": bundle["feature_names"],
        "model": bundle["predictor"].model_name,
    }
    for key, expected in checks.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Model/results compatibility mismatch: {key}.")
    for key, bundle_key in (("source_threshold", "threshold"), ("harm_weight", "harm_weight")):
        value = metadata.get(key)
        if value is None or not np.isfinite(float(value)) or not np.isclose(float(value), bundle[bundle_key], rtol=0, atol=1e-12):
            raise ValueError(f"Model/results compatibility mismatch: {key}.")
    if metadata.get("demo_adaptation") != bundle.get("demo_adaptation"):
        raise ValueError("Model/results compatibility mismatch: the declared adapted gate.")
    adapted = metadata.get("demo_adaptation", {})
    threshold = bundle.get("adapted_threshold")
    if threshold is None or not np.isfinite(float(threshold)) or not np.isclose(
        float(threshold), adapted.get("threshold", np.nan), rtol=0, atol=1e-12
    ):
        raise ValueError("Model/results compatibility mismatch: the adapted threshold.")
    if predictions.empty:
        raise ValueError("The target-test replay contains no recorded examples.")
    protocol = bundle["protocol"]
    for key in ("model", "recheck_protocol", "action", "model_digest"):
        if key not in protocol:
            continue
        if key not in predictions or not predictions[key].eq(protocol[key]).all():
            raise ValueError(f"Recorded examples do not match the artifact's {key}.")
    if "action_description_sha256" in protocol:
        if "action_description" not in predictions or not predictions.action_description.map(
            lambda value: isinstance(value, str) and hashlib.sha256(value.encode("utf-8")).hexdigest() == protocol["action_description_sha256"]
        ).all():
            raise ValueError("Recorded examples do not match the artifact's action description.")
    computed = bundle["predictor"].probabilities(initial_frame(predictions))
    columns = ["p_" + label for label in LABELS]
    if not set(columns).issubset(predictions.columns):
        raise ValueError("Recorded results are missing explicitly ordered outcome probabilities.")
    recorded = predictions[columns].to_numpy(dtype=float)
    decisions = decide_recheck(recorded, bundle["threshold"], bundle["harm_weight"])
    if not np.allclose(computed, recorded, rtol=1e-10, atol=1e-12):
        raise ValueError("Saved outcome probabilities do not match this model and feature schema.")
    if "source_gate_rechecks" not in predictions or not np.array_equal(
        decisions, predictions.source_gate_rechecks.to_numpy()
    ):
        raise ValueError("Saved recheck decisions do not match the source threshold.")


def validate_file_hashes(run_root, metadata):
    hashes = metadata.get("artifact_sha256", {})
    if metadata.get("version") != "0.2" or not all(name in hashes for name in MATCHED_ARTIFACTS):
        raise ValueError("The v0.2 results are missing matching artifact fingerprints.")
    for name in MATCHED_ARTIFACTS:
        actual = hashlib.sha256((Path(run_root) / name).read_bytes()).hexdigest()
        if actual != hashes[name]:
            raise ValueError(f"The v0.2 artifact fingerprint does not match: {name}.")


@st.cache_resource
def _load_replay_data(run_root, stamps):
    """Only v0.2 artifacts are accepted; there is no fallback to baseline files."""
    del stamps  # Invalidates the cache whenever a required artifact changes.
    run_root = Path(run_root)
    reports = run_root / "reports"
    metadata = json.loads((reports / "experiment_metadata.json").read_text(encoding="utf-8"))
    validate_file_hashes(run_root, metadata)
    manifest = json.loads((reports / "split_manifest.json").read_text(encoding="utf-8"))
    policies = pd.read_csv(reports / "policy_results.csv")
    adaptation = pd.read_csv(reports / "adaptation_summary.csv")
    predictions = pd.read_csv(reports / "target_predictions.csv", keep_default_na=False)
    bundle = live.load_live_artifact(run_root / "models" / "source_gate.joblib",
                                   expected_protocol=metadata["recheck_protocol"],
                                   expected_sha256=metadata["artifact_sha256"]["models/source_gate.joblib"])
    validate_replay(metadata, bundle, predictions)
    saved_features = pd.read_csv(reports / "target_features.csv")
    computed_features = extract_features(initial_frame(predictions))
    if list(saved_features.columns) != bundle["feature_names"] or saved_features.shape != computed_features.shape or not np.allclose(
        saved_features.to_numpy(dtype=float), computed_features.to_numpy(dtype=float), rtol=1e-10, atol=1e-12
    ):
        raise ValueError("Saved evaluation features do not match the recorded examples used in the demo.")
    metadata["policy_verification"] = verify_recorded_policies(run_root)
    # Cache only replay data. Imported OutcomePredictor class identities can
    # change after a Streamlit file-watcher reload, while this cache survives.
    return metadata, manifest, policies, adaptation, predictions


def load_replay(run_root, stamps):
    data = _load_replay_data(run_root, stamps)
    metadata = data[0]
    bundle = live.load_live_artifact(Path(run_root) / "models" / "source_gate.joblib",
                                   expected_protocol=metadata["recheck_protocol"],
                                   expected_sha256=metadata["artifact_sha256"]["models/source_gate.joblib"])
    return (*data, bundle)


def input_key(frame, selection, model_signature=""):
    initial_json = frame.to_json(orient="records", double_precision=15)
    return hashlib.sha256((model_signature + selection + initial_json).encode("utf-8")).hexdigest()


def recorded_initial_answer(record):
    """Read the initial call's saved extraction without reparsing or user edits.

    The recorded parser output is the same input used during evaluation.
    A missing extraction stays missing; no answer is inferred from future text.
    """
    answer = str(initial_frame(record).iloc[0]["base_answer"]).strip()
    return answer or "No initial answer was parsed in this record."


def clear_demo_result():
    for key in ("prediction", "live_result", "live_error", "live_request_key"):
        st.session_state.pop(key, None)


def render_scores(probabilities, decision):
    st.markdown(f"**Decision: {decision}**")
    for column, outcome in zip(st.columns(3), DISPLAY_OUTCOMES):
        column.metric(outcome, f"{probabilities[LABELS.index(outcome)]:.1%}")
    st.caption("Uncalibrated estimated scores about rechecking, not initial-answer correctness. HELP: wrong → correct. HURT: correct → wrong. NO_CHANGE includes wrong → wrong.")
    st.write("Skipping keeps the initial answer; it does not confirm correctness.")


def render_recorded(predictions, bundle, model_signature):
    st.caption("Recorded replay: one saved initial answer and one saved recheck. No new LLM call.")
    selected = st.selectbox(
        "Recorded example", list(range(len(predictions))),
        format_func=lambda i: f"{predictions.iloc[i]['example_id']} · Attempt {int(predictions.iloc[i]['rollout']) + 1}",
        key="recorded_example", on_change=clear_demo_result,
    )
    recorded = predictions.iloc[selected]
    # Actual tokens, completion and finalizer state stay internal to the record.
    frame = initial_frame(recorded)
    current_key = input_key(frame, f"recorded:{selected}:", model_signature)
    st.markdown("**Question**")
    st.write(recorded["query"])
    st.markdown("**Initial answer**")
    st.write(recorded_initial_answer(recorded))
    with st.expander("Full initial response", expanded=False):
        st.write(recorded["base_response"])
    if st.button("Should I recheck?", type="primary"):
        try:
            probabilities, decision = predict_initial(bundle, recorded)
            st.session_state["prediction"] = dict(
                input_key=current_key, probabilities=probabilities.tolist(), recheck=decision,
            )
        except ValueError as exc:
            st.error(f"Cannot predict this record: {exc}")
    result = st.session_state.get("prediction")
    if result is None or result["input_key"] != current_key:
        return
    decision_text = "Recheck" if result["recheck"] else "Skip recheck"
    render_scores(result["probabilities"], decision_text)
    with st.expander("What actually happened?", expanded=False):
        st.caption("Recorded outcome. Excluded from prediction.")
        st.markdown("**Recorded revised answer**")
        final_answer = str(recorded.get("final_answer", "")).strip()
        st.write(final_answer or "No revised answer was recorded.")
        st.write("Actual outcome: " + str(recorded["label"]))
        st.write(f"Initial correct: {bool(recorded['base_correct'])}. Revised correct: {bool(recorded['final_correct'])}.")
        revised_response = str(recorded.get("intervention_response", "")).strip()
        if revised_response:
            st.markdown("**Full recorded revision**")
            st.write(revised_response)


def trusted_live_evaluation(question, result):
    """Only score after execution, using the separate user-supplied key file."""
    from scripts.check_live_backend import load_questions, load_trusted_keys, score_after_execution
    from rechecklab.data import question_hash

    question_file = ROOT / "data" / "live_demo_questions.jsonl"
    key_file = ROOT / "data" / "live_demo_keys.jsonl"
    trusted_key = None
    if question_file.exists() and key_file.exists():
        questions = load_questions(question_file)
        keys = load_trusted_keys(key_file, questions)
        matching = [row for row in questions if question_hash(row["query"]) == question_hash(question)]
        if len(matching) == 1:
            trusted_key = keys[matching[0]["example_id"]]
    return score_after_execution(result, trusted_key)


def render_live_result(question, result):
    if result.get("error"):
        st.error(result["error"])
    initial = result.get("initial")
    if not initial:
        if result.get("initial_attempt"):
            with st.expander("Unvalidated initial response", expanded=False):
                st.caption("Required call metadata could not be validated. No prediction or correctness claim was made.")
                st.json(result["initial_attempt"])
        return
    st.caption("Newly generated response. The frozen gate uses initial-call signals only.")
    st.markdown("**Initial answer**")
    st.write(initial.get("answer") or "No initial answer was parsed. See the full response.")
    if initial.get("done_reason") == "length":
        st.warning("Initial response reached its token limit; it may be incomplete.")
    if result.get("probabilities") is not None:
        render_scores(result["probabilities"], result["decision"])
    st.markdown("**Selected final answer**")
    st.write(result.get("selected_answer") or "No final answer was parsed. See the full response.")
    if result.get("selected_source") == "initial_fallback":
        st.caption("Initial answer retained because the selected recheck failed. This run is incomplete.")
    else:
        st.caption("Revised answer selected." if result.get("rechecked") else "Initial answer retained.")
    if (result.get("revised") or {}).get("done_reason") == "length":
        st.warning("The revised response reached its token limit; it may be incomplete.")
    try:
        scored = trusted_live_evaluation(question, result)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f"Separate trusted-key scoring is unavailable: {exc}")
        scored = {"correctness_known": False}
    if scored["correctness_known"]:
        st.caption(f"Separate supplied key: initial correct = {scored['initial_correct']}. Extraction failures count incorrect.")
        if scored["observed_outcome"] is not None:
            st.caption(f"Revised correct = {scored['revised_correct']}. Actual outcome: {scored['observed_outcome']}.")
        else:
            st.caption("Actual recheck outcome: unobserved; no completed recheck.")
    else:
        st.caption("Actual correctness and outcome: unknown. No separate trusted answer key for this question.")
    with st.expander("Responses and actual call metadata", expanded=False):
        st.markdown("**Full initial response**")
        st.write(initial["response"])
        if result.get("revised"):
            st.markdown("**Full revision**")
            st.write(result["revised"]["response"])
        st.json({"initial": {k: v for k, v in initial.items() if k != "response"},
                 "revision": {k: v for k, v in (result.get("revised") or {}).items() if k != "response"},
                 "actual_protocol": result.get("live_protocol"),
                 "total_tokens": result.get("total_tokens"),
                 "recheck_attempted": result.get("recheck_attempted")})
        if result.get("recheck_attempt") and not result["recheck_attempt"].get("validated"):
            st.caption("Unvalidated recheck evidence; excluded from prediction and correctness scoring.")
            st.json(result["recheck_attempt"])


def render_live(bundle, model_signature):
    host = os.environ.get("RECHECK_OLLAMA_HOST", live.DEFAULT_HOST)
    backend = live.probe_backend(bundle, host=host, timeout=2)
    if backend["ready"]:
        st.caption(f"Connected: Ollama · {backend['model']}")
        if backend["experimental"]:
            st.warning("Experimental live protocol. The public checkpoint digest and full prompts were not released; these calls differ from the evaluated replay. Scores are unvalidated for live use.")
        else:
            st.caption("Saved checkpoint and collection protocol match.")
    elif backend["status"] == "predictor_load_error":
        st.caption("Saved predictor: unavailable")
        st.error(backend["message"])
    else:
        st.caption(f"Live backend: {backend['status'].replace('_', ' ')}")
        st.info(f"{backend['message']} Recorded examples remain usable.")
        if backend["status"] == "unavailable":
            st.caption("Start Ollama and install qwen3:4b. See README.md for Windows setup.")
    question = st.text_area("Question", key="live_question", on_change=clear_demo_result,
                            placeholder="Enter a question", height=100)
    current_key = live_input_key(question, model_signature, host, backend)
    if st.button("Should I recheck?", type="primary", disabled=not backend["ready"] or not question.strip()):
        st.session_state["live_request_key"] = current_key
        progress = st.empty()

        def report_progress(stage, message):
            progress.info(message)

        with st.spinner("Generating and checking…"):
            job = execute_once(st.session_state, current_key, lambda: live.run_live_question(
                question.strip(), bundle, host=host, allow_experimental=backend["experimental"],
                on_progress=report_progress, answer_type="open"))
        progress.empty()
        # A question-change callback may have run while the request was active.
        if st.session_state.get("live_request_key") == current_key:
            if job["status"] == "complete":
                st.session_state["live_result"] = dict(input_key=current_key, result=job["result"])
                st.session_state.pop("live_error", None)
            elif job["status"] == "failed":
                st.session_state["live_error"] = dict(input_key=current_key, error=job["error"])
            else:
                st.session_state["live_error"] = dict(input_key=current_key,
                    error="A live request is in progress or was interrupted. Its calls will not be duplicated in this session.")
    error = st.session_state.get("live_error")
    if error and error["input_key"] == current_key:
        st.warning(error["error"])
    saved = st.session_state.get("live_result")
    if saved and saved["input_key"] == current_key:
        render_live_result(question.strip(), saved["result"])


def render_demo(predictions, bundle, model_signature):
    mode = st.radio("Demo mode", ("Recorded examples", "Live question"),
                    horizontal=True, key="demo_mode", on_change=clear_demo_result)
    if mode == "Recorded examples":
        render_recorded(predictions, bundle, model_signature)
    else:
        render_live(bundle, model_signature)


def render_results(metadata, manifest, policies, adaptation):
    study_status_path = ROOT / "reports" / "live_study" / "CURRENT_STATUS.json"
    if study_status_path.exists():
        try:
            study_status = json.loads(study_status_path.read_text(encoding="utf-8"))
            collected = study_status["collection"]
            st.warning(f"Independent live study {study_status['study_version']}: {study_status['overall_status']} · "
                       f"{collected['completed_unique_questions']} / {collected['planned_questions']} planned pairs completed.")
            evaluation = study_status["stages"]["evaluation"]
            reason = "; ".join(evaluation.get("not_run_reasons", []))
            st.caption(f"Live-study evaluation: {evaluation['status']}. {reason}")
        except (OSError, ValueError, KeyError, TypeError):
            st.caption("Independent live-study status is unavailable. Regenerate it with scripts/report_live_study.py.")
    st.caption("Public SEVRA pilot (v0.2) · recorded replay")
    st.caption(f"Maths → CommonsenseQA · {metadata['target_predictor_metrics']['questions']} held-out questions · recorded replay")
    target = policies.loc[policies.evaluation == "Target test"].copy()
    expected = ("Keep initial answer", "Always recheck", "Completion rule", "Frozen source gate", "Threshold-adapted gate (200 questions)")
    if set(target.policy) != set(expected) or len(target) != len(expected):
        raise ValueError("The v0.2 target policy table must contain the five declared comparison policies.")
    target = target.set_index("policy").loc[list(expected)].reset_index()
    show = target[["policy", "accuracy_pct", "harmful_flip_pct_of_all_questions", "recheck_pct", "average_total_tokens"]].rename(columns={
        "policy": "Policy", "accuracy_pct": "Accuracy (%)",
        "harmful_flip_pct_of_all_questions": "Harmful flips (%)",
        "recheck_pct": "Recheck rate (%)", "average_total_tokens": "Average tokens",
    })
    show["Policy"] = ["Keep initial", "Always recheck", "Completion rule", "Frozen gate", "Adapted gate"]
    st.dataframe(show.round(2), hide_index=True, width="stretch")
    st.caption("Harmful flips: % of all questions. Tokens: replay estimates including initial and selected recheck calls, excluding gate overhead.")
    adaptation_choice = metadata["demo_adaptation"]
    st.caption(f"Table: adapted gate is one predeclared run — {adaptation_choice['questions']} adaptation questions, subset {adaptation_choice['repeat'] + 1}.")
    audit = metadata["policy_verification"]
    masks = audit["decision_vectors"]
    identical_gates = masks["completion_vs_frozen_mismatching_rows"] == 0 and masks["frozen_vs_selected_adapted_mismatching_rows"] == 0
    if identical_gates:
        st.write(f"The completion rule, frozen gate and displayed adapted gate recheck the same {masks['rechecked_questions']} questions, so their results are identical.")
    st.image(str(RUN_ROOT / "reports" / "adaptation_curve.png"))
    st.caption(f"Plot: mean accuracy across {metadata['repeats']} repeated subsets at each budget; bars show SD, not confidence intervals. Only the threshold changes.")
    budget = audit["selected_adaptation_vs_budget_average"]
    st.write(f"At {adaptation_choice['questions']} labels, mean accuracy is {budget['budget_mean_accuracy_pct']:.2f}% (SD {budget['budget_accuracy_sd_percentage_points']:.2f} percentage points).")
    frozen_accuracy = float(target.loc[target.policy.eq("Frozen source gate"), "accuracy_pct"].iloc[0])
    if adaptation.accuracy_mean.max() <= frozen_accuracy + 1e-10:
        st.caption("No tested budget improves mean accuracy over the frozen gate.")
    if identical_gates:
        st.caption("The learned gate shows no advantage over the completion rule on this test set.")
    with st.expander("Model diagnostics and research details", expanded=False):
        study_bundle = ROOT / "reports" / "live_study" / "diagnostic_bundle.zip"
        if study_bundle.exists():
            st.download_button("Download live study diagnostics (ZIP)", study_bundle.read_bytes(),
                               file_name="RecheckLab_live_study_diagnostics.zip", mime="application/zip")
        st.write(metadata["provenance"])
        diagnostics = []
        for key, name in (("source_predictor_metrics", "Source test"), ("target_predictor_metrics", "Target test")):
            metrics = metadata[key]
            diagnostics.append({"Partition": name, "Macro-F1": metrics["macro_f1"], "Log loss": metrics["log_loss"]})
        st.dataframe(pd.DataFrame(diagnostics).round(3), hide_index=True, width="stretch")
        metrics = metadata["target_predictor_metrics"]
        st.markdown("**Target outcomes: actual rows, predicted columns**")
        st.dataframe(pd.DataFrame(metrics["confusion_matrix"], index=LABELS, columns=LABELS), width="stretch")
        st.caption(f"Each question has equal weight across its rollouts. Target HURT recall: {metrics['per_class']['HURT']['recall']:.1%}. Probabilities are uncalibrated.")
        if audit["score_boundary"]["frozen_and_adapted_inside_same_empty_gap"]:
            st.write("The frozen and adapted cutoffs differ, but no target score falls between them. Both select exactly the completion-rule questions.")
        st.write("Adaptation uses only reserved target labels; test labels do not choose settings. This pilot uses one solver and split with inherited labels, and evaluates one bounded recheck.")
        st.caption("Initial answers use the recorded extraction, including any extraction errors.")
        report = dict(metadata=metadata, split_manifest=manifest,
                      policy_results=json.loads(policies.to_json(orient="records")),
                      adaptation_summary=json.loads(adaptation.to_json(orient="records")))
        html_report = RUN_ROOT / "reports" / "Progress_Report.html"
        if html_report.exists():
            st.download_button("Download research report (HTML)", html_report.read_bytes(),
                               file_name="RecheckLab_v0.2_report.html", mime="text/html")
        st.download_button("Download diagnostics (JSON)", json.dumps(report, indent=2),
                           file_name="RecheckLab_v0.2_report.json", mime="application/json")


def main():
    st.set_page_config(page_title="RecheckLab", layout="centered")
    st.title("Should I recheck?")
    st.caption("RecheckLab · research prototype v0.2")
    try:
        stamps = tuple((RUN_ROOT / name).stat().st_mtime_ns for name in REPLAY_FILES)
        metadata, manifest, policies, adaptation, predictions, bundle = load_replay(str(RUN_ROOT), stamps)
        if not (RUN_ROOT / "reports" / "adaptation_curve.png").exists():
            raise ValueError("The matching v0.2 adaptation plot is missing.")
    except live.ArtifactLoadError as exc:
        st.error(f"Saved predictor loading failed: {exc}")
        st.caption("The app requires the matching 19-feature gate in outputs/v0.2/models/source_gate.joblib.")
        st.stop()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The v0.2 replay could not be loaded: {exc}")
        st.caption("Use the v0.2 pilot command in README.md to regenerate the matching files.")
        st.stop()
    demo, results = st.tabs(("Demo", "Results"))
    with demo:
        render_demo(predictions, bundle, metadata["artifact_sha256"]["models/source_gate.joblib"])
    with results:
        try:
            render_results(metadata, manifest, policies, adaptation)
        except (ValueError, KeyError) as exc:
            st.error(f"The v0.2 results are incomplete: {exc}")


if __name__ == "__main__":
    main()
