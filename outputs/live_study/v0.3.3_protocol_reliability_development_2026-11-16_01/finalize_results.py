"""Summarize the frozen paired development calls and score initial-only gate inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
sys.path.insert(0, str(ROOT))
from rechecklab.data import LABELS
from rechecklab.features import FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION, INITIAL_REQUIRED, extract_features, validate_initial_record
from rechecklab.predictor import decide_recheck, load_artifact, revision_score

ARTIFACT = ROOT / "outputs/v0.2/models/source_gate.joblib"
ARTIFACT_SHA_EXPECTED = "9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439"
TRAIN = ROOT / "outputs/v0.2/data/prepared/source_train.parquet"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    status = read_json(OUT / "COLLECTION_STATUS.json")
    if status.get("status") != "COMPLETE" or status.get("completed_calls") != 12:
        raise ValueError("Only a complete frozen 12-call comparison can be summarized.")
    if sha(ARTIFACT) != ARTIFACT_SHA_EXPECTED:
        raise ValueError("Frozen v0.2 model artifact checksum changed.")
    protocol = read_json(OUT / "FROZEN_PROTOCOL.json")
    pairs = status["completed_pairs"]
    keys = {r["example_id"]: str(r["gold_answer"]) for r in read_jsonl(OUT / "TRUSTED_KEYS.jsonl")}
    raw_calls = read_jsonl(OUT / "RAW_CALLS.jsonl")
    if len(raw_calls) != 12:
        raise ValueError("Raw call evidence does not contain all twelve calls.")

    initial_rows = []
    gate_input_audit = []
    call_rows = []
    for pair in pairs:
        qid, policy = pair["example_id"], pair["policy"]
        initial, revision = pair["calls"]
        for call in (initial, revision):
            call_rows.append({"example_id": qid, "policy": policy, "phase": call["phase"],
                              "completion_reason": call.get("completion_reason"),
                              "prompt_tokens": call["tokens"].get("prompt_tokens"),
                              "output_tokens": call["tokens"].get("output_tokens"),
                              "total_tokens": call["tokens"].get("total_tokens"),
                              "latency_seconds": call["latency_seconds"],
                              "system_free_ram_before_gib": call["resource_before"]["free_ram_gib"],
                              "system_free_ram_after_gib": call["resource_after"]["free_ram_gib"],
                              "system_used_percent_after": call["resource_after"]["used_ram_percent"],
                              "gpu_telemetry_after": call["resource_after"]["gpu"]})
        m = initial["response_message"]
        response = ((m.get("thinking") or "") + ("\n\n" if m.get("thinking") and m.get("content") else "") + (m.get("content") or ""))
        row = {"query": next(r["query"] for r in read_jsonl(OUT / "QUESTION_SELECTION.jsonl") if r["example_id"] == qid),
               "base_response": response, "base_answer": pair["initial_answer"] or "",
               "base_actual_tokens": initial["tokens"]["total_tokens"],
               "base_done_reason": initial["completion_reason"], "base_finalizer_used": False}
        validate_initial_record(row)
        initial_rows.append({**row, "example_id": qid, "policy": policy, "initial_valid": pair["initial_valid"],
                             "revision_valid": pair["revision_valid"], "initial_correct": pair["initial_correct"],
                             "revision_correct": pair["revision_correct"], "initial_latency": initial["latency_seconds"],
                             "revision_latency": revision["latency_seconds"],
                             "initial_tokens": initial["tokens"]["total_tokens"],
                             "revision_tokens": revision["tokens"]["total_tokens"],
                             "revision_reason": revision["completion_reason"]})
        gate_input_audit.append({"example_id": qid, "policy": policy,
                                 "required_fields_present": all(k in row for k in INITIAL_REQUIRED),
                                 "completion_reason_recorded": row["base_done_reason"],
                                 "actual_prompt_plus_output_tokens": row["base_actual_tokens"],
                                 "finalizer_flag": False,
                                 "answer_parsed_from_initial_content_only": row["base_answer"] != "",
                                 "revised_text_or_labels_in_gate_input": False})
    initial_frame = pd.DataFrame([{k: r[k] for k in INITIAL_REQUIRED} for r in initial_rows])
    features = extract_features(initial_frame)
    if list(features.columns) != FEATURE_NAMES or features.shape != (6, 19):
        raise ValueError("Canonical frozen gate feature extraction did not produce the expected 19 columns.")
    bundle = load_artifact(ARTIFACT, expected_sha256=ARTIFACT_SHA_EXPECTED)
    predictor = bundle["predictor"]
    # Only the initial allowlist enters the original frozen model. The production
    # protocol validator is not edited or overridden; this is explicitly OOD.
    probabilities = predictor.probabilities(initial_frame)
    scores = revision_score(probabilities, bundle["harm_weight"])
    actions = decide_recheck(probabilities, bundle["threshold"], bundle["harm_weight"])

    gate_rows = []
    feature_rows = []
    for i, meta in enumerate(initial_rows):
        gate_rows.append({"example_id": meta["example_id"], "policy": meta["policy"],
                          "p_HELP": probabilities[i, LABELS.index("HELP")],
                          "p_NO_CHANGE": probabilities[i, LABELS.index("NO_CHANGE")],
                          "p_HURT": probabilities[i, LABELS.index("HURT")],
                          "score_pHELP_minus_pHURT": scores[i], "threshold": bundle["threshold"],
                          "ml_gate_recheck_OOD": bool(actions[i]),
                          "completion_heuristic_recheck": meta["base_done_reason"] == "length",
                          "initial_reason": meta["base_done_reason"],
                          "initial_valid": meta["initial_valid"], "initial_correct": meta["initial_correct"]})
        feature_rows.append({"example_id": meta["example_id"], "policy": meta["policy"],
                             **{k: float(v) for k, v in features.iloc[i].to_dict().items()}})

    train = pd.read_parquet(TRAIN)
    train["base_done_reason"] = train["base_done_reason"].fillna("").replace("", "unknown")
    train_features = extract_features(train)
    shift_rows = []
    for name in FEATURE_NAMES:
        ref = train_features[name].astype(float)
        live = features[name].astype(float)
        sd = float(ref.std(ddof=0))
        shift_rows.append({"feature": name, "frozen_source_train_mean": float(ref.mean()), "source_train_sd": sd,
                           "candidate_A_mean": float(live.iloc[[i for i, r in enumerate(initial_rows) if r["policy"] == "A_thinking_off_1024"]].mean()),
                           "candidate_B_mean": float(live.iloc[[i for i, r in enumerate(initial_rows) if r["policy"] == "B_thinking_on_2048"]].mean()),
                           "mean_abs_standardized_difference_A": abs(float(live.iloc[[i for i, r in enumerate(initial_rows) if r["policy"] == "A_thinking_off_1024"]].mean()) - float(ref.mean())) / sd if sd else None,
                           "mean_abs_standardized_difference_B": abs(float(live.iloc[[i for i, r in enumerate(initial_rows) if r["policy"] == "B_thinking_on_2048"]].mean()) - float(ref.mean())) / sd if sd else None,
                           "outside_source_train_minmax_count": int(((live < ref.min()) | (live > ref.max())).sum())})

    write_csv(OUT / "COMPLETION_DIAGNOSTICS.csv", [{k: v for k, v in r.items() if k not in ("query", "base_response", "base_answer", "base_actual_tokens", "base_done_reason", "base_finalizer_used")} for r in initial_rows])
    write_csv(OUT / "RESOURCE_MEASUREMENTS.csv", call_rows)
    write_csv(OUT / "GATE_FEATURE_COMPATIBILITY.csv", gate_input_audit)
    write_csv(OUT / "GATE_PREDICTIONS.csv", gate_rows)
    write_csv(OUT / "OOD_FEATURES.csv", feature_rows)
    write_csv(OUT / "FEATURE_SHIFT.csv", shift_rows)

    comparison = []
    for name in POLICIES_LOCAL:
        subset = [r for r in initial_rows if r["policy"] == name]
        calls = [c for c in call_rows if c["policy"] == name]
        comparison.append({"candidate": name, "questions": len(subset),
                           "initial_valid": sum(r["initial_valid"] for r in subset),
                           "revision_valid": sum(r["revision_valid"] for r in subset),
                           "calls_valid_of_6": sum(r["initial_valid"] + r["revision_valid"] for r in subset),
                           "initial_length": sum(r["base_done_reason"] == "length" for r in subset),
                           "initial_stop": sum(r["base_done_reason"] == "stop" for r in subset),
                           "revision_length": sum(r["revision_reason"] == "length" for r in subset),
                           "revision_stop": sum(r["revision_reason"] == "stop" for r in subset),
                           "initial_accuracy": sum(r["initial_correct"] for r in subset) / len(subset),
                           "revision_accuracy": sum(r["revision_correct"] for r in subset) / len(subset),
                           "mean_call_seconds": float(np.mean([c["latency_seconds"] for c in calls])),
                           "mean_initial_seconds": float(np.mean([c["latency_seconds"] for c in calls if c["phase"] == "initial"])),
                           "mean_revision_seconds": float(np.mean([c["latency_seconds"] for c in calls if c["phase"] == "revision"])),
                           "mean_initial_total_tokens": float(np.mean([c["total_tokens"] for c in calls if c["phase"] == "initial"])),
                           "mean_revision_total_tokens": float(np.mean([c["total_tokens"] for c in calls if c["phase"] == "revision"])),
                           "minimum_free_ram_after_gib": min(c["system_free_ram_after_gib"] for c in calls),
                           "maximum_system_used_percent": max(c["system_used_percent_after"] for c in calls)})
    write_csv(OUT / "PROTOCOL_COMPARISON.csv", comparison)

    by_id = {}
    for r, gate in zip(initial_rows, gate_rows):
        by_id.setdefault(r["example_id"], {})[r["policy"]] = {"gate": gate["ml_gate_recheck_OOD"], "completion": gate["completion_heuristic_recheck"]}
    decision_rows = []
    for qid, data in by_id.items():
        a, b = data["A_thinking_off_1024"], data["B_thinking_on_2048"]
        decision_rows.append({"example_id": qid, "A_gate_recheck": a["gate"], "B_gate_recheck": b["gate"],
                              "gate_disagreement": a["gate"] != b["gate"],
                              "A_completion_recheck": a["completion"], "B_completion_recheck": b["completion"],
                              "completion_disagreement": a["completion"] != b["completion"]})
    write_csv(OUT / "CANDIDATE_DECISION_DIFFERENCES.csv", decision_rows)

    avg_latency = {r["candidate"]: r["mean_call_seconds"] for r in comparison}
    valid_init_total = sum(r["initial_valid"] for r in comparison)
    min_ram = min(c["system_free_ram_after_gib"] for c in call_rows)
    max_used = max(c["system_used_percent_after"] for c in call_rows)
    gate_disagree = sum(r["gate_disagreement"] for r in decision_rows)
    completion_disagree = sum(r["completion_disagreement"] for r in decision_rows)
    comparison_md = "\n".join(["| Candidate | Initial valid | Revision valid | Completion reasons (initial/revision) | Mean call latency | Mean initial/revision total tokens | Initial accuracy* | Revision accuracy* |",
        "|---|---:|---:|---|---:|---:|---:|---:|"] + [f"| {r['candidate']} | {r['initial_valid']}/3 | {r['revision_valid']}/3 | length {r['initial_length']}/{r['revision_length']}; stop {r['initial_stop']}/{r['revision_stop']} | {r['mean_call_seconds']:.1f}s | {r['mean_initial_total_tokens']:.0f} / {r['mean_revision_total_tokens']:.0f} | {r['initial_accuracy']:.0%} | {r['revision_accuracy']:.0%} |" for r in comparison])
    comp = f"""# Protocol comparison and recommendation

**Run:** `v0.3.3_protocol_reliability_development_2026-11-16_01`  
**Status:** 3 prospectively selected development questions; 12/12 calls completed. This check is not final evaluation evidence and does not establish reliability or generalization.

## Observed completion

{comparison_md}

*Accuracy is shown only as descriptive grading evidence and was not used to select a protocol. For both candidates, the three questions were the same. Candidate A produced {sum(r['initial_valid'] for r in comparison if r['candidate'].startswith('A_'))}/3 valid initial answers; Candidate B produced {sum(r['initial_valid'] for r in comparison if r['candidate'].startswith('B_'))}/3. Across both protocols, initial valid extraction was {valid_init_total}/6 and revision extraction was {sum(r['revision_valid'] for r in initial_rows)}/6. Candidate A was faster on mean calls ({avg_latency['A_thinking_off_1024']:.1f}s vs {avg_latency['B_thinking_on_2048']:.1f}s).

## Failure interpretation

The historical pilot had 5/8 initial calls stop at `length` with thinking enabled and a 1,024-token cap. The prior separate thinking-off development sample had 5/5 parseable initial answers at the same cap. In this newly selected paired sample, both A (thinking off, 1,024) and B (thinking on, 2,048) had only 1/3 valid initial answers. Two B initial calls exhausted 2,048 generated tokens; one had no final-channel content because all generated tokens were in the thinking field. A also exhausted 1,024 tokens on two initial calls. This is consistent with a response-length/budget issue and shows material question sensitivity, but the sample cannot attribute failures to thinking mode, token cap, or prompt alone.

## Frozen gate and feature compatibility

All six candidate/question initial records passed the canonical validation and yielded all 19 required `pre-recheck-v2` features. Inputs used only question text, full initial response (including the separate thinking channel where present), parsed initial answer, actual prompt-plus-output token count, actual completion reason, and explicit `finalizer_used=false`. Revised text, keys, correctness, and outcomes were excluded from gate prediction inputs. Candidate A and B gate actions disagreed on {gate_disagree}/3 questions; completion decisions disagreed on {completion_disagree}/3. Gate scores are OOD transfer diagnostics only: the v0.2 gate's SEVRA active-verify prompts, checkpoint identity, finalizer, and generation protocol do not match these Ollama protocols. No protocol identifiers were altered and no production validator was bypassed.

Univariate feature comparisons against frozen v0.2 source training are descriptive only (see `FEATURE_SHIFT.csv`). The protocol comparison does not identify any single cause of shifts.

## Resources

The pre-call baseline was 5.93 GiB available RAM, 266 MiB/4096 MiB GPU memory used, and no loaded Ollama model. During sequential collection, lowest recorded post-call free RAM was {min_ram:.2f} GiB and the maximum system memory utilization was {max_used:.1f}%. At completion, free RAM was {status['resources']['free_ram_gib']:.2f} GiB; GPU telemetry was `{status['resources']['gpu']}`. All calls completed without hitting the 2.0 GiB stop floor. No unrelated processes were terminated and no system configuration was changed.

## Decision

**No final protocol qualifies yet.** Neither candidate showed reliable initial extraction on the fresh paired development questions, and B's larger 2,048-token budget did not reliably produce final answers. Candidate A was faster, but only 1/3 initial answers were valid here. The earlier 5/5 thinking-off result and this 1/3 result are both small, disjoint samples; they should not be pooled as a formal reliability estimate. Do not launch a 30â€“50-question evaluation yet.

The defensible next step is another separately versioned development check after freezing one explicit adjustment (for example, a larger thinking-off token budget) on new questions. That candidate must reach a predeclared initial and revision extraction criterion with acceptable RAM and latency before an evaluation protocol is frozen. Do not choose a setting based on accuracy, and do not change v0.2 weights or threshold.
"""
    (OUT / "PROTOCOL_COMPARISON.md").write_text(comp, encoding="utf-8")

    chosen = {"decision": "NO_PROTOCOL_QUALIFIED", "reason": "Neither tested candidate demonstrated sufficiently reliable initial final-answer extraction on fresh paired development questions.",
              "tested_candidates": protocol["candidates"], "recommended_next_step": "Prospectively freeze one additional development-only candidate (suggested: thinking disabled with a larger explicit output budget) on new non-overlapping questions; assess extraction, latency and resources before freezing final-evaluation protocol.",
              "final_30_to_50_question_evaluation_authorized_or_recommended": False,
              "frozen_gate": "Retain unchanged as OOD transfer policy only; do not describe predictions as protocol validated.",
              "manuscript_or_historical_records_modified": False}
    (OUT / "CHOSEN_PROTOCOL_MANIFEST.json").write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    spec = {"status": "NOT_READY_FOR_FINAL_EVALUATION", "question_count_target_if_later_qualified": "30-50 fresh independent questions",
            "exclusions": "source training/development/test, target adaptation/test, prior live planned/evaluated cohorts, both pilot question sets, and every development question",
            "question_selection": "uniform seeded sample from pinned GSM8K main/train after canonical overlap exclusions; freeze selection before inference; no outcome or gate based selection",
            "paired_policies": ["never recheck", "always recheck", "completion heuristic", "frozen v0.2 ML gate (OOD transfer only)"],
            "gate_features": FEATURE_NAMES, "gate_schema": FEATURE_SCHEMA_VERSION,
            "gate_threshold": bundle["threshold"], "gate_score": "p_HELP - p_HURT", "gate_rule": "strictly greater than threshold",
            "protocol_compatibility": "OOD cross-protocol; artifact expects SEVRA active_verify with unavailable full prompt/checkpoint identity and historical finalizer behavior",
            "launch_command": None, "launch_blocker": "A generation protocol with adequate prospective initial/revision extraction reliability has not yet been demonstrated."}
    (OUT / "FROZEN_EVALUATION_SPEC.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    (OUT / "NEXT_STUDY_LAUNCH.md").write_text("# Next study launch\n\nNo final-evaluation launch command is issued: neither tested protocol met the completion-reliability gate. Freeze and validate a new development-only candidate first. Once qualified, create a new question manifest and protocol hash, then run the approved collector on that fresh study directory; do not reuse this development directory.\n", encoding="utf-8")

    input_paths = [OUT / name for name in ("run_comparison.py", "FROZEN_PROTOCOL.json", "QUESTION_SELECTION.jsonl", "TRUSTED_KEYS.jsonl", "RAW_CALLS.jsonl", "COLLECTION_STATUS.json", "finalize_results.py")]
    outputs = [p for p in OUT.iterdir() if p.is_file() and p.name not in {"PROVENANCE.json"}]
    provenance = {"run": "v0.3.3_protocol_reliability_development_2026-11-16_01", "questions": 3,
                  "completed_calls": 12, "ollama_inference_calls_after_comparison": 0,
                  "artifact_sha256": sha(ARTIFACT), "source_train_sha256": sha(TRAIN),
                  "feature_schema": FEATURE_SCHEMA_VERSION, "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
                  "feature_names": FEATURE_NAMES, "gate_threshold": bundle["threshold"], "gate_harm_weight": bundle["harm_weight"],
                  "gate_class_order": LABELS, "gate_protocol_label": "OOD transfer diagnostic only",
                  "input_sha256": {p.name: sha(p) for p in input_paths},
                  "output_sha256": {p.name: sha(p) for p in outputs if p.name not in {"PROVENANCE.json"}}}
    (OUT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"calls": 12, "candidate_summary": comparison, "minimum_free_ram_gib": min_ram,
                      "gate_A_B_disagreements": gate_disagree, "completion_A_B_disagreements": completion_disagree,
                      "final_protocol": chosen["decision"], "output": str(OUT)}, indent=2))


POLICIES_LOCAL = ["A_thinking_off_1024", "B_thinking_on_2048"]

if __name__ == "__main__":
    main()
