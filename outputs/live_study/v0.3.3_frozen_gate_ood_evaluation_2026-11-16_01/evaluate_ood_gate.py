"""Research-only OOD scoring of the frozen v0.2 gate on existing live pairs.
No Ollama calls, model fitting, or artifact writes are performed.
"""
from pathlib import Path
import csv, hashlib, json, sys
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[2]
sys.path.insert(0, str(PROJECT))
from rechecklab.data import LABELS
from rechecklab.features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION, INITIAL_REQUIRED, extract_features, validate_initial_record
from rechecklab.predictor import decide_recheck, load_artifact, revision_score, validate_probabilities

PILOT = PROJECT / "outputs/live_study/v0.3.3_independent_pilot_2026-11-16_01"
ARTIFACT = PROJECT / "outputs/v0.2/models/source_gate.joblib"
PAIRS = PILOT / "pairs.jsonl"
CALLS = PILOT / "calls.jsonl"
TRAIN = PROJECT / "data/prepared/source_train.parquet"
THRESHOLD_EXPECTED = 0.25968145276947807


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line.strip()]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main():
    if not all(p.is_file() for p in (ARTIFACT, PAIRS, CALLS, TRAIN)):
        raise FileNotFoundError("A required frozen input file is missing.")
    bundle = load_artifact(ARTIFACT)  # validates the original artifact; no expected live protocol is asserted
    predictor = bundle["predictor"]
    predictor.validate_contract()
    if bundle["feature_names"] != FEATURE_NAMES or bundle["class_order"] != LABELS:
        raise ValueError("Frozen artifact feature or class ordering differs from the canonical implementation.")
    if bundle["threshold"] != THRESHOLD_EXPECTED:
        raise ValueError("Frozen artifact threshold differs from the documented v0.2 threshold.")
    if len(FEATURE_NAMES) != 19 or FEATURE_SCHEMA_VERSION != "pre-recheck-v2":
        raise ValueError("Unexpected feature schema; refusing a noncanonical OOD evaluation.")

    pairs = read_jsonl(PAIRS)
    if len(pairs) != 8 or len({p["example_id"] for p in pairs}) != 8:
        raise ValueError("Expected exactly the eight unique stored pilot question pairs.")

    # OOD scoring path: deliberately project only the initial-time allowlist.
    # Revised text, gold keys, correctness, labels, and future token counts are not inputs.
    initial_rows = [{k: p[k] for k in INITIAL_REQUIRED} for p in pairs]
    for row in initial_rows:
        validate_initial_record(row)
    initial_frame = pd.DataFrame(initial_rows, columns=INITIAL_REQUIRED)
    features = extract_features(initial_frame)
    raw = np.asarray(predictor.estimator.predict_proba(features), dtype=float)
    model_classes = list(predictor.estimator.classes_)
    probs = np.zeros((len(initial_frame), len(LABELS)), dtype=float)
    for j, class_name in enumerate(model_classes):
        probs[:, LABELS.index(class_name)] = raw[:, j]
    probs = validate_probabilities(probs, expected_rows=8)
    scores = revision_score(probs, bundle["harm_weight"])
    gate_actions = decide_recheck(probs, bundle["threshold"], bundle["harm_weight"])

    # Store complete feature values and OOD probabilities before consulting outcomes.
    feature_rows = []
    prediction_rows = []
    for i, (pair, frow) in enumerate(zip(pairs, features.to_dict("records"))):
        feature_rows.append({"example_id": pair["example_id"], **frow})
        prediction_rows.append({
            "example_id": pair["example_id"], "source_question_id": pair["source_question_id"],
            "initial_done_reason": pair["base_done_reason"],
            "p_HELP": float(probs[i, LABELS.index("HELP")]),
            "p_NO_CHANGE": float(probs[i, LABELS.index("NO_CHANGE")]),
            "p_HURT": float(probs[i, LABELS.index("HURT")]),
            "revision_score": float(scores[i]), "threshold": float(bundle["threshold"]),
            "gate_recheck": bool(gate_actions[i]),
            "completion_heuristic_recheck": pair["base_done_reason"] == "length",
            "initial_correct": bool(pair["base_correct"]),
            "revision_correct": bool(pair["final_correct"]), "transition": pair["label"],
        })
    write_csv(OUT / "OOD_FEATURES.csv", feature_rows)
    write_csv(OUT / "ML_PREDICTIONS.csv", prediction_rows)

    # Descriptive feature-distribution comparison to v0.2 source training records.
    train = pd.read_parquet(TRAIN)
    if train["base_done_reason"].isna().any():
        raise ValueError("v0.2 source training contains null completion reasons; documented normalization cannot be applied.")
    train_unknown_completion_rows = int(train["base_done_reason"].eq("").sum())
    # Match original public-data preprocessing: published missing stop reasons are explicit unknowns.
    train.loc[train["base_done_reason"].eq(""), "base_done_reason"] = "unknown"
    train_features = extract_features(train)
    shift_rows = []
    for name in FEATURE_NAMES:
        ref = train_features[name].astype(float)
        live = features[name].astype(float)
        sd = float(ref.std(ddof=0))
        delta = float(live.mean() - ref.mean())
        shift_rows.append({
            "feature": name, "train_mean": float(ref.mean()), "train_sd": sd,
            "live_mean": float(live.mean()),
            "standardized_mean_difference": delta / sd if sd > 0 else (0.0 if delta == 0 else None),
            "train_min": float(ref.min()), "train_max": float(ref.max()),
            "live_outside_train_range_count": int(((live < ref.min()) | (live > ref.max())).sum()),
            "live_questions": int(len(live)),
        })
    write_csv(OUT / "FEATURE_SHIFT.csv", shift_rows)

    # Outcomes enter only after frozen gate scores/actions have been computed.
    policy_actions = {
        "Never recheck": np.zeros(8, dtype=bool),
        "Always recheck": np.ones(8, dtype=bool),
        "Completion heuristic": np.array([p["base_done_reason"] == "length" for p in pairs], dtype=bool),
        "Frozen ML gate (OOD transfer)": np.asarray(gate_actions, dtype=bool),
    }
    policy_rows = []
    for name, actions in policy_actions.items():
        correct = np.array([bool(p["final_correct"] if actions[i] else p["base_correct"]) for i, p in enumerate(pairs)])
        helps = np.array([p["label"] == "HELP" for p in pairs])
        hurts = np.array([p["label"] == "HURT" for p in pairs])
        nochange = np.array([p["label"] == "NO_CHANGE" for p in pairs])
        initial_total = sum(int(p["base_actual_tokens"]) for p in pairs)
        revision_total = sum(int(p["action_actual_tokens"]) if actions[i] else 0 for i, p in enumerate(pairs))
        latency = sum(float(p["base_wall_seconds"]) + (float(p["action_wall_seconds"]) if actions[i] else 0.0) for i, p in enumerate(pairs))
        policy_rows.append({
            "policy": name, "questions": 8, "correct_count": int(correct.sum()),
            "accuracy": float(correct.mean()), "rechecks": int(actions.sum()),
            "recheck_rate": float(actions.mean()),
            "HELP_total": int(helps.sum()), "HELP_captured": int((helps & actions).sum()),
            "HELP_missed": int((helps & ~actions).sum()),
            "HURT_total": int(hurts.sum()), "HURT_revision_accepted": int((hurts & actions).sum()),
            "HURT_avoided": int((hurts & ~actions).sum()),
            "NO_CHANGE_rechecks": int((nochange & actions).sum()),
            "mean_prompt_plus_output_tokens_per_question": (initial_total + revision_total) / 8,
            "replayed_latency_seconds_total": latency,
            "replayed_latency_seconds_per_question": latency / 8,
        })
    write_csv(OUT / "POLICY_COMPARISON.csv", policy_rows)

    question_rows = []
    for row in prediction_rows:
        question_rows.append({
            **{k: row[k] for k in ("example_id", "source_question_id", "initial_done_reason", "p_HELP", "p_NO_CHANGE", "p_HURT", "revision_score", "threshold", "gate_recheck", "completion_heuristic_recheck", "initial_correct", "revision_correct", "transition")},
        })
    write_csv(OUT / "QUESTION_LEVEL.csv", question_rows)

    disagreements = [r for r in prediction_rows if r["gate_recheck"] != r["completion_heuristic_recheck"]]
    train_shift = sorted(shift_rows, key=lambda r: abs(r["standardized_mean_difference"] or 0), reverse=True)
    out_of_range = sorted(shift_rows, key=lambda r: r["live_outside_train_range_count"], reverse=True)
    fields = ", ".join(FEATURE_NAMES)
    compat = f"""# OOD feature and protocol compatibility\n\n**Feature status: compatible. Protocol status: OOD / cross-protocol diagnostic only.**\n\nAll eight records contain the six required initial-only inputs: `query`, `base_response`, `base_answer`, `base_actual_tokens`, `base_done_reason`, and `base_finalizer_used`. Values passed the canonical `validate_initial_record` checks. The actual token count is Ollama prompt plus generated tokens as recorded by the collector; completion is the actual `stop` or `length` reason; finalizer use is explicitly false. No missing fields were imputed. The other 13 features are deterministic derivatives of those initial inputs.\n\nThe v0.2 source-training distribution converts published empty completion status to the explicit `unknown` category, as documented in `rechecklab.data.public_frames`; this normalization is applied only to the reference training data, never to live initial records. Count normalized: {train_unknown_completion_rows}. The exact frozen feature order is: `{fields}`. The canonical schema is `{FEATURE_SCHEMA_VERSION}` with fingerprint `{FEATURE_SCHEMA_FINGERPRINT}`.\n\nThe gate artifact is v0.2, threshold `{bundle['threshold']}`, harm weight `{bundle['harm_weight']}`, class order `{LABELS}`. It was evaluated through the original fitted estimator, original feature extractor/order, original probability-column mapping, `revision_score`, and `decide_recheck`. This separate script intentionally skips only the production live-protocol compatibility comparison; it does not alter or falsify protocol metadata.\n\nThe artifact expects SEVRA `active_verify`, while these responses use the separately frozen `live-study` recheck protocol. SEVRA full prompts and checkpoint digest are unavailable, and generation/finalizer settings differ. Consequently every gate score/action here is an **OOD transfer diagnostic**, not a protocol-validated deployment prediction.\n"""
    (OUT / "OOD_COMPATIBILITY.md").write_text(compat, encoding="utf-8")
    gate_policy = next(r for r in policy_rows if r["policy"] == "Frozen ML gate (OOD transfer)")
    completion_policy = next(r for r in policy_rows if r["policy"] == "Completion heuristic")
    helpmiss = gate_policy["HELP_missed"]
    incompletes = sum(p["base_done_reason"] == "length" for p in pairs)
    gate_incomplete = sum(bool(gate_actions[i]) and pairs[i]["base_done_reason"] == "length" for i in range(8))
    missed_help_ids = [pairs[i]["source_question_id"] for i in range(8) if pairs[i]["label"] == "HELP" and not gate_actions[i]]
    fail = f"""# Frozen gate OOD failure analysis\n\n- **Incomplete initial answers:** {incompletes}/8 stopped at the length limit. The gate selected {gate_incomplete}/{incompletes} of these for rechecking.\n- **Helpful revisions:** the stored pairs contain {gate_policy['HELP_total']} HELP transitions. The gate selected {gate_policy['HELP_captured']} and missed {helpmiss}. Missed HELP source IDs: {missed_help_ids or 'none'}.\n- **Harmful revisions:** {gate_policy['HURT_total']} HURT transitions occurred; the gate accepted {gate_policy['HURT_revision_accepted']}. Zero observed HURT events does not establish harm safety.\n- **Gate vs completion heuristic:** {len(disagreements)}/8 action disagreements. IDs: {[r['source_question_id'] for r in disagreements] or 'none'}. Completion rechecked {completion_policy['rechecks']}/8; gate rechecked {gate_policy['rechecks']}/8.\n- **Feature distribution:** compared the eight live feature vectors with the frozen v0.2 source-training feature distribution (univariate, descriptive). Mean absolute standardized shift across nonconstant features: {float(np.mean([abs(x['standardized_mean_difference']) for x in shift_rows if x['standardized_mean_difference'] is not None])):.2f}. Largest absolute standardized mean shifts: {[(x['feature'], round(x['standardized_mean_difference'],2) if x['standardized_mean_difference'] is not None else None) for x in train_shift[:5]]}. Features with at least one live value outside the source-training min/max: {sum(x['live_outside_train_range_count']>0 for x in shift_rows)}/19. See `FEATURE_SHIFT.csv`.\n\nThe distribution differences coincide with prompt, output-budget, thinking, and finalizer-protocol differences; this eight-question analysis cannot attribute the shifts to any one change. It does not support significance, generalization, or safety conclusions. Any apparent gate/completion disagreements or feature shifts should motivate a separately frozen prospective test, not threshold changes or post-hoc calibration. Concrete prospective checks include protocol-matched gate validation, representation of both length/stop completions, and monitoring whether completion/finalizer/token features remain within source support.\n"""
    (OUT / "FAILURE_ANALYSIS.md").write_text(fail, encoding="utf-8")

    input_paths = [ARTIFACT, PAIRS, CALLS, PILOT / "questions.jsonl", PILOT / "protocol.json", PILOT / "protocol_hash.txt", PILOT / "EXPERIMENT_MANIFEST.json", PILOT / "QUESTION_SELECTION.json", TRAIN]
    provenance = {
        "evaluation_type": "research-only OOD/cross-protocol frozen classifier scoring",
        "new_llm_calls": 0, "questions": 8,
        "artifact_sha256": sha256(ARTIFACT), "artifact_threshold": bundle["threshold"],
        "artifact_harm_weight": bundle["harm_weight"], "feature_schema": FEATURE_SCHEMA_VERSION,
        "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT, "feature_names": FEATURE_NAMES,
        "class_order": LABELS, "score_formula": "p_HELP - harm_weight * p_HURT",
        "decision_rule": "score > threshold", "feature_input_fields": INITIAL_REQUIRED,
        "trusted_key_file_read": False, "revised_response_or_outcome_used_for_features_or_predictions": False,
        "protocol_hash": (PILOT / "protocol_hash.txt").read_text(encoding="utf-8").strip(),
        "input_sha256": {str(p.relative_to(PROJECT)): sha256(p) for p in input_paths},
        "output_sha256": {},
    }
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name != "PROVENANCE.json":
            provenance["output_sha256"][p.name] = sha256(p)
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"questions": 8, "features": 19, "threshold": bundle["threshold"], "gate_rechecks": gate_policy["rechecks"], "completion_rechecks": completion_policy["rechecks"], "disagreements": len(disagreements), "gate_correct": gate_policy["correct_count"], "gate_accuracy": gate_policy["accuracy"], "HELP_missed": helpmiss, "HURT_accepted": gate_policy["HURT_revision_accepted"], "directory": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()

