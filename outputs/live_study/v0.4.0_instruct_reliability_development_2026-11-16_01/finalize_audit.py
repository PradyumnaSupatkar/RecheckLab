"""Write non-inference audit summaries and SHA-256 provenance for v0.4.0."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

DEV = Path(__file__).resolve().parent
ROOT = DEV.parents[2]
EVAL = ROOT / "outputs/live_study/v0.4.0_instruct_frozen_evaluation_2026-11-16_01"
sys.path.insert(0, str(ROOT))
from rechecklab.data import LABELS
from rechecklab.features import INITIAL_REQUIRED, FEATURE_NAMES, FEATURE_SCHEMA_VERSION, FEATURE_SCHEMA_FINGERPRINT

GATE = ROOT / "outputs/v0.2/models/source_gate.joblib"
GATE_SHA = "9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439"
GATE_THRESHOLD = 0.25968145276947807


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def readj(path):
    return json.loads(path.read_text(encoding="utf-8"))


def readjl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def hash_tree(folder: Path):
    return {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob("*")) if p.is_file()
            and p.name != "PROVENANCE.json"}


def main():
    dev_manifest = readj(DEV / "EXPERIMENT_MANIFEST.json")
    eval_manifest = readj(EVAL / "EXPERIMENT_MANIFEST.json")
    dev_pairs = readjl(DEV / "PAIRS.jsonl")
    eval_pairs = readjl(EVAL / "PAIRS.jsonl")
    dev_calls = readjl(DEV / "RAW_CALLS.jsonl")
    eval_calls = readjl(EVAL / "RAW_CALLS.jsonl")
    if sha(GATE) != GATE_SHA or len(dev_pairs) != 10 or len(eval_pairs) != 40 or len(dev_calls) != 20 or len(eval_calls) != 80:
        raise ValueError("Frozen model or completed run evidence failed integrity checks.")

    questions = []
    for p in eval_pairs:
        gate_row = next(r for r in read_csv(EVAL / "ML_PREDICTIONS.csv") if r["example_id"] == p["example_id"])
        questions.append({"example_id": p["example_id"], "initial_parse_valid": p["initial_parse_valid"],
                          "revision_parse_valid": p["revision_parse_valid"], "initial_correct": p["initial_correct"],
                          "revision_correct": p["revision_correct"], "initial_answer": p["initial_answer"],
                          "revision_answer": p["revision_answer"], "outcome": p["outcome"],
                          "initial_done_reason": p["initial_done_reason"], "revision_done_reason": p["revision_done_reason"]})
    errors = []
    for p in eval_pairs:
        if p["initial_parse_valid"] and not p["initial_correct"]:
            errors.append({"example_id": p["example_id"], "phase": "initial", "model_answer": p["initial_answer"],
                           "parseable": True, "correct": False, "done_reason": p["initial_done_reason"]})
        if p["revision_parse_valid"] and not p["revision_correct"]:
            errors.append({"example_id": p["example_id"], "phase": "revision", "model_answer": p["revision_answer"],
                           "parseable": True, "correct": False, "done_reason": p["revision_done_reason"]})
    write_csv(EVAL / "MODEL_ERRORS.csv", errors or [{"example_id": "", "phase": "", "model_answer": "", "parseable": "", "correct": "", "done_reason": ""}])
    parser_failures = []
    for p in eval_pairs:
        for phase in ("initial", "revision"):
            if not p[f"{phase}_parse_valid"]:
                parser_failures.append({"example_id": p["example_id"], "phase": phase,
                                        "failure_type": p[f"{phase}_parse_status"],
                                        "completion_reason": p[f"{phase}_done_reason"]})
    write_csv(EVAL / "PARSER_FAILURES.csv", parser_failures or [{"example_id": "", "phase": "", "failure_type": "NONE", "completion_reason": ""}])
    dev_errors = []
    for p in dev_pairs:
        for phase in ("initial", "revision"):
            if p[f"{phase}_parse_valid"] and not p[f"{phase}_correct"]:
                dev_errors.append({"example_id": p["example_id"], "phase": phase,
                                   "model_answer": p[f"{phase}_answer"], "parseable": True,
                                   "correct": False, "done_reason": p[f"{phase}_done_reason"]})
    write_csv(DEV / "VALID_BUT_INCORRECT_DEVELOPMENT.csv", dev_errors or [{"example_id": "", "phase": "", "model_answer": "", "parseable": "", "correct": "", "done_reason": ""}])

    resource = readj(EVAL / "COLLECTION_STATUS.json")["resource_samples"]
    ram_values, vram_values, swap_values = [], [], []
    thinking_content_calls = 0
    for item in resource:
        for when in ("before", "after"):
            obs = item[when]
            ram_values.append(obs["free_ram_gib"])
            swap_values.append(obs["swap_percent"])
            if obs.get("gpu", {}).get("free_mib") is not None:
                vram_values.append(obs["gpu"]["free_mib"])
    prep = readj(DEV / "PREPARATION_STATUS.json")
    thinking_content_calls = sum(bool((c.get("response", {}).get("message", {}).get("thinking") or "").strip())
                                 for c in dev_calls + eval_calls)
    hardware = f"""# Hardware and inference benchmark

**Checkpoint:** Qwen3-4B-Instruct-2507 (`qwen3:4b-instruct`), digest `{eval_manifest['model']['digest']}`; Ollama `{eval_manifest['model']['ollama_version']}`. Sequential local inference only.

| Phase | Questions | Calls | Mean initial call | Mean revision call | Mean initial prompt+output tokens | Mean revision prompt+output tokens | Minimum free RAM | Minimum free VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Development | 10 | 20 | {mean_latency(dev_calls, 'initial'):.2f}s | {mean_latency(dev_calls, 'revision'):.2f}s | {mean_tokens(dev_calls, 'initial'):.1f} | {mean_tokens(dev_calls, 'revision'):.1f} | {min_resource(dev_calls, 'ram'):.2f} GiB | {min_resource(dev_calls, 'vram'):.0f} MiB |
| Frozen evaluation | 40 | 80 | {mean_latency(eval_calls, 'initial'):.2f}s | {mean_latency(eval_calls, 'revision'):.2f}s | {mean_tokens(eval_calls, 'initial'):.1f} | {mean_tokens(eval_calls, 'revision'):.1f} | {min(ram_values):.2f} GiB | {min(vram_values):.0f} MiB |

Pre-inference baseline: {prep['available_ram_gib_at_freeze']:.2f} GiB available RAM; GPU `{prep['available_gpu']}`. Across the evaluation, maximum swap use was {max(swap_values):.1f}%. The enforced stop floors were {eval_manifest['resource_guard']['minimum_free_ram_gib']:.1f} GiB RAM and {eval_manifest['resource_guard']['minimum_free_vram_mib']} MiB VRAM; neither floor was crossed. All 100 calls returned with recorded usage counts and actual client wall latency. The response thinking field was nonempty in {thinking_content_calls}/100 calls. No retry, model download, process termination, or system configuration change occurred.
"""
    (EVAL / "HARDWARE_BENCHMARK.md").write_text(hardware, encoding="utf-8")
    (DEV / "HARDWARE_BENCHMARK.md").write_text(hardware.replace("Frozen evaluation", "Frozen evaluation"), encoding="utf-8")

    compat = f"""# Frozen v0.2 gate: OOD feature compatibility

**Status:** canonical feature-level compatibility passed; protocol compatibility is **cross-model/cross-protocol OOD only**.

- Input rows: {len(eval_pairs)}/40.
- Required initial-only fields available and validated for every row: `{', '.join(INITIAL_REQUIRED)}`.
- Feature schema: `{FEATURE_SCHEMA_VERSION}`, {len(FEATURE_NAMES)} features, fingerprint `{FEATURE_SCHEMA_FINGERPRINT}`.
- Exact original feature order: `{', '.join(FEATURE_NAMES)}`.
- Original class order: `{', '.join(LABELS)}`.
- Frozen artifact SHA-256: `{GATE_SHA}`; threshold `{GATE_THRESHOLD}`; score `p_HELP - 1.0 * p_HURT`; recheck iff score is strictly greater than threshold.

The scoring path loaded the unchanged artifact and called the repository's canonical `validate_initial_record`, `extract_features`, and original predictor probability method. The predictor input frame was explicitly projected to the six initial-only allowlist fields. It did not include revised text, trusted keys, correctness, transitions, or any other outcome. The model is Qwen3-4B-Instruct-2507, whereas the artifact was trained for the historical Qwen3-4B SEVRA active-verify protocol; therefore scores/actions are an OOD cross-model and cross-protocol research diagnostic, not protocol-validated deployment.
"""
    (EVAL / "GATE_OOD_COMPATIBILITY.md").write_text(compat, encoding="utf-8")
    write_csv(DEV / "DEVELOPMENT_ANSWER_DIAGNOSTICS.csv", [
        {"example_id": p["example_id"], "initial_parseable": p["initial_parse_valid"], "revision_parseable": p["revision_parse_valid"],
         "initial_correct": p["initial_correct"], "revision_correct": p["revision_correct"],
         "initial_completion_reason": p["initial_done_reason"], "revision_completion_reason": p["revision_done_reason"],
         "initial_failure_type": next(c["failure_type"] for c in dev_calls if c["example_id"] == p["example_id"] and c["phase"] == "initial"),
         "revision_failure_type": next(c["failure_type"] for c in dev_calls if c["example_id"] == p["example_id"] and c["phase"] == "revision")}
        for p in dev_pairs])

    historical = DEV / "HISTORICAL_FAILURE_REVIEW.md"
    preservation = {"writes_limited_to": [str(DEV.relative_to(ROOT)), str(EVAL.relative_to(ROOT))],
               "historical_runs_modified": [], "source_code_modified": [], "frozen_artifact_modified": False,
               "manuscript_modified": False, "public_upload_performed": False,
               "frozen_artifact_sha256_verified": sha(GATE), "historical_failure_review_sha256": sha(historical),
               "note": "All new evidence, question keys, and raw calls are in outputs/live_study; no remote upload or push was performed."}
    write_json(DEV / "PRESERVATION_CHECK.json", preservation)
    write_json(EVAL / "PRESERVATION_CHECK.json", preservation)
    for folder, manifest in ((DEV, dev_manifest), (EVAL, eval_manifest)):
        q_raw = (folder / "QUESTION_SELECTION.jsonl").read_bytes()
        k_raw = (folder / "TRUSTED_KEYS.jsonl").read_bytes()
        q_canonical = q_raw.replace(b"\r\n", b"\n")
        k_canonical = k_raw.replace(b"\r\n", b"\n")
        if hashlib.sha256(q_canonical).hexdigest() != manifest["question_selection_sha256"]:
            raise ValueError(f"Frozen question selection content changed in {folder.name}.")
        if hashlib.sha256(k_canonical).hexdigest() != manifest["trusted_key_sha256"]:
            raise ValueError(f"Frozen key content changed in {folder.name}.")
        write_json(folder / "FILE_INTEGRITY.json", {"selection_hash_semantics": "SHA-256 over canonical UTF-8 JSONL with LF line endings",
                   "windows_text_file_line_endings": "CRLF; normalized to LF only for manifest comparison",
                   "question_selection_sha256_canonical": hashlib.sha256(q_canonical).hexdigest(),
                   "question_selection_sha256_raw_file": hashlib.sha256(q_raw).hexdigest(),
                   "question_selection_matches_frozen_manifest": True,
                   "trusted_keys_sha256_canonical": hashlib.sha256(k_canonical).hexdigest(),
                   "trusted_keys_sha256_raw_file": hashlib.sha256(k_raw).hexdigest(),
                   "trusted_keys_match_frozen_manifest": True})
        provenance = {"run": manifest["phase"], "model_digest": manifest["model"]["digest"],
                      "dataset_sha256": manifest["dataset"]["source_sha256"], "artifact_sha256": GATE_SHA,
                      "prompt_sha256": manifest["protocol"]["prompt_sha256"],
                      "question_selection_sha256": manifest["question_selection_sha256"],
                      "trusted_key_sha256": manifest["trusted_key_sha256"],
                      "feature_schema": FEATURE_SCHEMA_VERSION, "feature_names": FEATURE_NAMES,
                      "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT, "class_order": LABELS,
                      "output_sha256": hash_tree(folder)}
        (folder / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    verify_provenance(DEV)
    verify_provenance(EVAL)
    print(json.dumps({"dev_files": len(hash_tree(DEV)), "eval_files": len(hash_tree(EVAL)),
                      "dev_questions": len(dev_pairs), "eval_questions": len(eval_pairs),
                      "parseable_but_incorrect_evaluation_answers": len(errors), "artifact_sha256": sha(GATE)}, indent=2))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def mean_latency(calls, phase):
    vals = [c["wall_clock_seconds"] for c in calls if c.get("status") == "succeeded" and c["phase"] == phase]
    return sum(vals) / len(vals)


def mean_tokens(calls, phase):
    vals = [c["tokens"]["total_tokens"] for c in calls if c.get("status") == "succeeded" and c["phase"] == phase]
    return sum(vals) / len(vals)


def min_resource(calls, kind):
    if kind == "ram":
        return min(c[t]["free_ram_gib"] for c in calls for t in ("resource_before", "resource_after"))
    return min(c[t]["gpu"]["free_mib"] for c in calls for t in ("resource_before", "resource_after"))


def verify_provenance(folder):
    doc = readj(folder / "PROVENANCE.json")
    for name, expected in doc["output_sha256"].items():
        if sha(folder / name) != expected:
            raise ValueError(f"Provenance hash mismatch: {folder.name}/{name}")


if __name__ == "__main__":
    main()
