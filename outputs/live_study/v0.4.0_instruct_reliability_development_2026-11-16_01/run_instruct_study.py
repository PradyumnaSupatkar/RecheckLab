"""Prepare, validate, and conditionally execute RecheckLab Instruct study.

Run from repository root: --prepare freezes disjoint 10/40 questions and both
protocol manifests; --run collects the 10-question gate and automatically
collects the 40-question study only when the frozen gate passes.
"""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import psutil

DEV = Path(__file__).resolve().parent
ROOT = DEV.parents[2]
EVAL = ROOT / "outputs/live_study/v0.4.0_instruct_frozen_evaluation_2026-11-16_01"
DATA = ROOT / "outputs/live_study/v0.3.0/raw/gsm8k/740312add88f781978c0658806c59bc2815b9866/train.parquet"
MODEL = "qwen3:4b-instruct"
DIGEST = "0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
INITIAL_PROMPT = (
    "Solve this arithmetic word problem. Use at most three short calculation lines. "
    "End with exactly one final line in this form: FINAL: <number>. Do not write anything after that line."
)
REVISION_PROMPT = (
    "Check the saved answer against the original problem. Keep it if correct; otherwise correct it. "
    "Use at most three short calculation lines. End with exactly one final line in this form: "
    "FINAL: <number>. Do not write anything after that line."
)
PARAMETERS = {
    "context_window": 4096,
    "num_predict": 512,
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "min_p": 0.0,
    "repeat_penalty": 1.0,
    "stop": ["<|im_start|>", "<|im_end|>"],
    "stream": False,
    "think_field": "omitted; use installed Instruct checkpoint's default template",
    "timeout_seconds": 180,
    "retries": 0,
}
FREE_RAM_FLOOR_GIB = 2.0
FREE_VRAM_FLOOR_MIB = 512
INITIAL_SEED_BASE = 40000
REVISION_SEED_BASE = 50000
SELECTION_SEED = 20261116
GATE_PATH = ROOT / "outputs/v0.2/models/source_gate.joblib"
GATE_SHA = "9a92e4e9a91d1a05e0ead0dd638f434838e9b944e406ff01fa3d82ae91e3c439"
GATE_THRESHOLD = 0.25968145276947807


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(text: str) -> str:
    import unicodedata
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return sha_bytes(normalized.encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Ignore only interrupted historical tail records; all complete rows
            # and pinned manifests still contribute exclusions.
            continue
    return rows


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(payload, encoding="utf-8")
    return sha_bytes(payload.encode("utf-8"))


def ollama_tags():
    with urlopen("http://localhost:11434/api/tags", timeout=10) as response:
        return json.load(response).get("models", [])


def model_identity():
    req = Request("http://localhost:11434/api/show", data=json.dumps({"model": MODEL, "verbose": True}).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=15) as response:
        shown = json.load(response)
    version = subprocess.check_output(["ollama", "--version"], text=True, timeout=10).strip()
    return {
        "model": MODEL,
        "digest": next(x.get("digest") for x in ollama_tags() if x.get("name") == MODEL),
        "ollama_version": version,
        "family": shown.get("details", {}).get("family"),
        "parameter_size": shown.get("details", {}).get("parameter_size"),
        "quantization": shown.get("details", {}).get("quantization_level"),
        "fine_tune": shown.get("model_info", {}).get("general.finetune"),
        "model_version": shown.get("model_info", {}).get("general.version"),
        "capabilities": shown.get("capabilities", []),
    }


def selection_exclusions() -> set[str]:
    excluded: set[str] = set()
    for path in (ROOT / "data/prepared").glob("*.parquet"):
        frame = pd.read_parquet(path)
        if "group_id" in frame.columns:
            excluded.update(str(x) for x in frame["group_id"].dropna())
        elif "query" in frame.columns:
            excluded.update(canonical_hash(str(x)) for x in frame["query"].dropna())
    for path in (ROOT / "outputs/v0.2/data/prepared").glob("*.parquet"):
        frame = pd.read_parquet(path)
        if "group_id" in frame.columns:
            excluded.update(str(x) for x in frame["group_id"].dropna())
        elif "query" in frame.columns:
            excluded.update(canonical_hash(str(x)) for x in frame["query"].dropna())
    for folder in (ROOT / "outputs/live_study").glob("v0.3.*"):
        for filename in ("questions.jsonl", "pairs.jsonl", "calls.jsonl", "RAW_CALLS.jsonl", "PREDICTIONS.jsonl"):
            for row in read_jsonl(folder / filename):
                query = row.get("query") or row.get("question")
                if isinstance(query, str) and query.strip():
                    excluded.add(canonical_hash(query))
                group = row.get("group_id")
                if isinstance(group, str) and group:
                    excluded.add(group)
        manifest = folder / "QUESTION_SELECTION.json"
        if manifest.exists():
            doc = read_json(manifest)
            for key in ("selected", "selected_questions"):
                for row in doc.get(key, []):
                    if isinstance(row, dict) and row.get("group_id"):
                        excluded.add(str(row["group_id"]))
    return excluded


def prepare():
    if not DATA.is_file() or not GATE_PATH.is_file():
        raise FileNotFoundError("Pinned GSM8K cache or frozen v0.2 artifact is missing.")
    if sha_file(GATE_PATH) != GATE_SHA:
        raise ValueError("Frozen v0.2 classifier checksum does not match the recorded artifact.")
    identity = model_identity()
    if identity["digest"] != DIGEST:
        raise ValueError(f"Installed Instruct model digest mismatch: expected {DIGEST}, got {identity['digest']}.")
    if identity["family"] != "qwen3" or identity["fine_tune"] != "Instruct" or identity["model_version"] != "2507":
        raise ValueError(f"Installed model identity is not the expected Qwen3-4B-Instruct-2507: {identity}.")
    try:
        active = subprocess.check_output(["ollama", "ps"], text=True, timeout=10)
        loaded = [line for line in active.splitlines()[1:] if line.strip()]
    except Exception as exc:
        raise RuntimeError("Could not verify active Ollama models; refusing to start collection.") from exc
    if loaded:
        raise RuntimeError("Ollama already has an active model. No process was stopped; retry when idle.")
    if DEV.exists() and any(p.name != "run_instruct_study.py" for p in DEV.iterdir()):
        raise FileExistsError("Development output directory already contains files; refusing to overwrite.")
    if EVAL.exists() and any(EVAL.iterdir()):
        raise FileExistsError("Evaluation output directory already contains files; refusing to overwrite.")

    frame = pd.read_parquet(DATA)
    if len(frame) != 7473 or list(frame.columns) != ["question", "answer"]:
        raise ValueError("Pinned GSM8K train cache differs from expected snapshot schema.")
    excluded = selection_exclusions()
    candidates = []
    for index, row in frame.iterrows():
        question = row.get("question")
        if isinstance(question, str) and question.strip():
            query = question.strip()
            group = canonical_hash(query)
            if group not in excluded:
                candidates.append((int(index), query, group))
    if len(candidates) < 50:
        raise ValueError(f"Only {len(candidates)} fresh GSM8K candidates remain; need 50.")
    chosen_idx = np.random.default_rng(SELECTION_SEED).permutation(len(candidates))[:50]
    chosen = [candidates[int(i)] for i in chosen_idx]
    questions = [{"example_id": f"gsm8k:train:{i}", "source_question_id": str(i), "group_id": group,
                  "dataset": "openai/gsm8k", "revision": REVISION, "split": "main/train", "query": query}
                 for i, query, group in chosen]
    keys = []
    for q, (index, _, _) in zip(questions, chosen):
        solution = str(frame.iloc[index]["answer"])
        if "####" not in solution:
            raise ValueError(f"Published GSM8K answer key malformed for {q['example_id']}.")
        keys.append({"example_id": q["example_id"], "answer_type": "number",
                     "gold_answer": solution.rsplit("####", 1)[1].strip()})

    generation = {"initial_num_predict": PARAMETERS["num_predict"], "revision_num_predict": PARAMETERS["num_predict"],
                  **PARAMETERS, "initial_seed": "40000 + source question id", "revision_seed": "50000 + source question id"}
    common_manifest = {
        "study": "RecheckLab independent Instruct-model validation",
        "model": identity,
        "protocol": {"id": "live-study-v0.4.0-qwen3-4b-instruct-2507", "prompts": {"initial": INITIAL_PROMPT, "revision": REVISION_PROMPT},
                     "generation": generation,
                     "prompt_sha256": {"initial": sha_bytes(INITIAL_PROMPT.encode()), "revision": sha_bytes(REVISION_PROMPT.encode())},
                     "parser": "strict-final-line-numeric-v1; last nonempty line must be FINAL: finite decimal/fraction-free numeric literal",
                     "grader": "GSM8K published #### numeric key; exact Decimal equality after comma normalization; no tolerance",
                     "answers_and_gold_excluded_from_gate_features": True,
                     "feature_schema": "pre-recheck-v2", "feature_count": 19,
                     "gate_artifact_sha256": GATE_SHA, "gate_threshold": GATE_THRESHOLD,
                     "gate_score": "p_HELP - p_HURT", "gate_rule": "score > threshold",
                     "gate_status": "cross-model/cross-protocol OOD transfer policy; not protocol-validated"},
        "dataset": {"repository": "openai/gsm8k", "revision": REVISION, "split": "main/train",
                    "source_file": str(DATA.relative_to(ROOT)), "source_sha256": sha_file(DATA),
                    "selection_seed": SELECTION_SEED, "source_candidates": len(frame),
                    "excluded_canonical_groups": len(excluded), "eligible_count": len(candidates),
                    "selection_rule": "Uniform seeded sample from canonical-unique eligible GSM8K train questions after excluding all local v0.2 prepared partitions and all complete v0.3 question, response, prediction, and selection records; question text/hash only; no answer, difficulty, outcomes, or gate predictions used."},
        "resource_guard": {"sequential": True, "minimum_free_ram_gib": FREE_RAM_FLOOR_GIB,
                           "minimum_free_vram_mib": FREE_VRAM_FLOOR_MIB, "timeout_seconds": PARAMETERS["timeout_seconds"],
                           "retries": 0, "stop_on_any_failure_or_guard_breach": True, "never_terminate_other_processes": True},
        "policies": ["never recheck", "always recheck", "completion heuristic", "frozen v0.2 ML gate (cross-model/cross-protocol OOD)"],
        "analysis": {"unit": "question", "accuracy_intervals": "Wilson 95%", "paired_differences": "question bootstrap percentile 95%, 20,000 resamples, fixed seed 20261117", "no_significance_or_generalization_claim_from_small_development_set": True},
        "selection_frozen_before_any_new_inference": True,
    }
    selection_payload = ("\n".join(json.dumps(q, ensure_ascii=False) for q in questions) + "\n").encode()
    key_payload = ("\n".join(json.dumps(k) for k in keys) + "\n").encode()
    for folder, title, subset, key_subset in ((DEV, "development_reliability_gate", questions[:10], keys[:10]),
                                               (EVAL, "untouched_frozen_evaluation", questions[10:], keys[10:])):
        folder.mkdir(parents=True, exist_ok=True)
        qbytes = ("\n".join(json.dumps(q, ensure_ascii=False) for q in subset) + "\n").encode()
        kbytes = ("\n".join(json.dumps(k) for k in key_subset) + "\n").encode()
        write_jsonl(folder / "QUESTION_SELECTION.jsonl", subset)
        write_jsonl(folder / "TRUSTED_KEYS.jsonl", key_subset)
        manifest = {**common_manifest, "phase": title, "sample_size": len(subset),
                    "expected_calls": len(subset) * 2,
                    "question_selection_sha256": sha_bytes(qbytes), "trusted_key_sha256": sha_bytes(kbytes),
                    "question_ids": [q["example_id"] for q in subset]}
        write_json(folder / "EXPERIMENT_MANIFEST.json", manifest)
        write_json(folder / "PRESERVATION_BASELINE.json", {"historical_tree_status": "captured before new inference",
                    "inference_calls_started": 0, "artifact_sha256": GATE_SHA,
                    "historical_input_dirs": ["v0.3.3_independent_pilot_2026-11-16_01", "v0.3.3_development_thinking_off_2026-11-16_01", "v0.3.3_protocol_reliability_development_2026-11-16_01"]})
    write_historical_review()
    write_json(DEV / "PREPARATION_STATUS.json", {"status": "FROZEN_BEFORE_INFERENCE", "development_questions": 10,
               "evaluation_questions": 40, "inference_started": False,
               "all_50_questions_selected_before_development_outcomes": True,
               "question_pool_sha256": sha_bytes(selection_payload), "trusted_keys_pool_sha256": sha_bytes(key_payload),
               "model": identity, "available_ram_gib_at_freeze": resource_sample()["free_ram_gib"],
               "available_gpu": resource_sample()["gpu"]})
    (DEV / "RUN_COMMAND.md").write_text(
        "# Frozen execution command\n\nAfter reviewing the pre-inference manifests, run exactly:\n\n"
        "```powershell\npython outputs/live_study/v0.4.0_instruct_reliability_development_2026-11-16_01/run_instruct_study.py --run\n```\n\n"
        "This executes the 10-question development gate first. It starts the 40-question collection only if the predeclared extraction and resource criteria pass.\n",
        encoding="utf-8")
    print(json.dumps({"status": "FROZEN_BEFORE_INFERENCE", "model": identity, "development_questions": 10,
                      "evaluation_questions": 40, "eligible_count": len(candidates), "excluded_groups": len(excluded),
                      "dev_ids": [q["example_id"] for q in questions[:10]], "eval_ids": [q["example_id"] for q in questions[10:]],
                      "available_ram_gib": resource_sample()["free_ram_gib"], "inference_started": False}, indent=2))


def parse_final(content: str):
    lines = [line.strip() for line in str(content).splitlines() if line.strip()]
    if not lines:
        return None, "missing_final_marker"
    marker = any(re.match(r"^FINAL\s*:", line, re.I) for line in lines)
    if not marker:
        return None, "missing_final_marker"
    match = re.fullmatch(r"FINAL\s*:\s*([-+]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?)", lines[-1], re.I)
    if not match:
        return None, "malformed_final_answer"
    try:
        value = Decimal(match.group(1).replace(",", ""))
        if not value.is_finite():
            return None, "malformed_final_answer"
        return str(value.normalize()), "parseable"
    except InvalidOperation:
        return None, "malformed_final_answer"


def nvidia_stats():
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                      text=True, timeout=5).strip().splitlines()[0]
        name, total, used, utilization = [x.strip() for x in raw.split(",", 3)]
        return {"name": name, "total_mib": int(total), "used_mib": int(used), "free_mib": int(total) - int(used),
                "utilization_percent": int(utilization)}
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__}


def resource_sample():
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {"timestamp_unix": time.time(), "total_ram_gib": round(mem.total / 1024**3, 3),
            "free_ram_gib": round(mem.available / 1024**3, 3), "ram_used_percent": mem.percent,
            "swap_used_gib": round(swap.used / 1024**3, 3), "swap_percent": swap.percent,
            "gpu": nvidia_stats()}


def resource_safe(sample):
    gpu = sample.get("gpu", {})
    return (sample["free_ram_gib"] >= FREE_RAM_FLOOR_GIB and gpu.get("available", True) is not False
            and gpu.get("free_mib", FREE_VRAM_FLOOR_MIB) >= FREE_VRAM_FLOOR_MIB and sample["swap_percent"] < 95)


def call_ollama(payload):
    request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urlopen(request, timeout=PARAMETERS["timeout_seconds"]) as response:
        result = json.load(response)
    result["client_wall_seconds"] = time.perf_counter() - start
    return result


def check_model_still_pinned():
    identity = model_identity()
    if identity["digest"] != DIGEST or identity["fine_tune"] != "Instruct" or identity["model_version"] != "2507":
        raise RuntimeError("Installed model identity changed; refusing calls under a different checkpoint.")


def collect_phase(folder: Path):
    manifest = read_json(folder / "EXPERIMENT_MANIFEST.json")
    questions = read_jsonl(folder / "QUESTION_SELECTION.jsonl")
    keys = {r["example_id"]: r["gold_answer"] for r in read_jsonl(folder / "TRUSTED_KEYS.jsonl")}
    raw_path = folder / "RAW_CALLS.jsonl"
    pairs_path = folder / "PAIRS.jsonl"
    status_path = folder / "COLLECTION_STATUS.json"
    if raw_path.exists() or pairs_path.exists() or status_path.exists():
        raise FileExistsError(f"Refusing to overwrite or resume collection at {folder}.")
    check_model_still_pinned()
    call_count = 0
    pairs = []
    status = {"phase": manifest["phase"], "status": "RUNNING", "calls_completed": 0, "pairs_completed": 0,
              "resource_samples": [], "model_digest": DIGEST}
    write_json(status_path, status)
    for q in questions:
        responses = {}
        for phase, prompt in (("initial", INITIAL_PROMPT), ("revision", REVISION_PROMPT)):
            before = resource_sample()
            if not resource_safe(before):
                status.update(status="STOPPED_RESOURCE_GUARD", stop_reason="RAM/VRAM/swap safety floor", resource=before)
                write_json(status_path, status)
                return status
            user = INITIAL_PROMPT + "\n\n" + q["query"]
            messages = [{"role": "user", "content": user}]
            if phase == "revision":
                messages.extend([{"role": "assistant", "content": responses["initial"].get("message", {}).get("content", "")},
                                 {"role": "user", "content": REVISION_PROMPT}])
            seed = (INITIAL_SEED_BASE if phase == "initial" else REVISION_SEED_BASE) + int(q["source_question_id"])
            payload = {"model": MODEL, "messages": messages, "stream": False,
                       "options": {"num_predict": PARAMETERS["num_predict"], "num_ctx": PARAMETERS["context_window"],
                                   "temperature": PARAMETERS["temperature"], "seed": seed, "top_k": PARAMETERS["top_k"],
                                   "top_p": PARAMETERS["top_p"], "min_p": PARAMETERS["min_p"],
                                   "repeat_penalty": PARAMETERS["repeat_penalty"], "stop": PARAMETERS["stop"]}}
            try:
                result = call_ollama(payload)
                if result.get("done") is not True or "message" not in result:
                    raise RuntimeError("Ollama returned an incomplete response object.")
                responses[phase] = result
                message = result.get("message", {})
                content = message.get("content", "") or ""
                answer, parse_status = parse_final(content)
                token_record = {"prompt_tokens": result.get("prompt_eval_count"), "output_tokens": result.get("eval_count")}
                if token_record["prompt_tokens"] is None or token_record["output_tokens"] is None:
                    raise RuntimeError("Ollama omitted actual prompt/output token counts.")
                token_record["total_tokens"] = int(token_record["prompt_tokens"]) + int(token_record["output_tokens"])
                done_reason = result.get("done_reason", "unknown")
                if answer is None and done_reason == "length" and (message.get("thinking") or ""):
                    failure_type = "reasoning_token_exhaustion"
                elif answer is None and done_reason == "length":
                    failure_type = "length_limit_truncation"
                elif answer is None:
                    failure_type = parse_status
                else:
                    failure_type = "none"
                record = {"example_id": q["example_id"], "phase": phase, "status": "succeeded", "model": MODEL,
                          "model_digest": DIGEST, "request": payload, "response": result,
                          "completion_reason": done_reason, "parsed_answer": answer, "parse_status": parse_status,
                          "failure_type": failure_type, "tokens": token_record,
                          "wall_clock_seconds": result["client_wall_seconds"],
                          "resource_before": before, "resource_after": resource_sample()}
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                call_count += 1
                status["calls_completed"] = call_count
                status["resource_samples"].append({"example_id": q["example_id"], "phase": phase,
                                                   "before": before, "after": record["resource_after"]})
                status["last_call"] = {k: record[k] for k in ("example_id", "phase", "completion_reason", "parse_status", "failure_type", "tokens", "wall_clock_seconds")}
                write_json(status_path, status)
                if not resource_safe(record["resource_after"]):
                    status.update(status="STOPPED_RESOURCE_GUARD", stop_reason="RAM/VRAM/swap safety floor after call")
                    write_json(status_path, status)
                    return status
            except Exception as exc:
                failure = {"example_id": q["example_id"], "phase": phase, "status": "failed",
                           "error_type": type(exc).__name__, "error": str(exc), "request": payload,
                           "resource": resource_sample()}
                with raw_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(failure, ensure_ascii=False) + "\n")
                status.update(status="STOPPED_CALL_FAILURE", failure=failure)
                write_json(status_path, status)
                return status
        if len(responses) != 2:
            status.update(status="INCOMPLETE_PAIR")
            write_json(status_path, status)
            return status
        init_msg, rev_msg = responses["initial"]["message"], responses["revision"]["message"]
        initial_answer, initial_parse = parse_final(init_msg.get("content", ""))
        revision_answer, revision_parse = parse_final(rev_msg.get("content", ""))
        gold = str(keys[q["example_id"]])
        try:
            gold_dec = Decimal(gold.replace(",", ""))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid frozen trusted key for {q['example_id']}") from exc
        init_correct = initial_answer is not None and Decimal(initial_answer) == gold_dec
        revision_correct = revision_answer is not None and Decimal(revision_answer) == gold_dec
        outcome = "HELP" if not init_correct and revision_correct else "HURT" if init_correct and not revision_correct else "NO_CHANGE"
        pair = {"example_id": q["example_id"], "query": q["query"], "initial_answer": initial_answer or "",
                "revision_answer": revision_answer or "", "initial_parse_valid": initial_answer is not None,
                "revision_parse_valid": revision_answer is not None, "initial_parse_status": initial_parse,
                "revision_parse_status": revision_parse,
                "initial_done_reason": responses["initial"].get("done_reason", "unknown"),
                "revision_done_reason": responses["revision"].get("done_reason", "unknown"),
                "initial_tokens": int(responses["initial"]["prompt_eval_count"]) + int(responses["initial"]["eval_count"]),
                "revision_tokens": int(responses["revision"]["prompt_eval_count"]) + int(responses["revision"]["eval_count"]),
                "initial_prompt_tokens": int(responses["initial"]["prompt_eval_count"]),
                "initial_output_tokens": int(responses["initial"]["eval_count"]),
                "revision_prompt_tokens": int(responses["revision"]["prompt_eval_count"]),
                "revision_output_tokens": int(responses["revision"]["eval_count"]),
                "initial_wall_seconds": responses["initial"]["client_wall_seconds"],
                "revision_wall_seconds": responses["revision"]["client_wall_seconds"],
                "initial_correct": bool(init_correct), "revision_correct": bool(revision_correct), "outcome": outcome,
                "gold_answer_used_only_for_offline_scoring": True}
        with pairs_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        pairs.append(pair)
        status["pairs_completed"] = len(pairs)
        status["last_pair"] = {k: pair[k] for k in ("example_id", "initial_parse_valid", "revision_parse_valid", "initial_done_reason", "revision_done_reason", "outcome")}
        write_json(status_path, status)
        print(f"{manifest['phase']} {len(pairs)}/{len(questions)} {q['example_id']}: parse {pair['initial_parse_valid']}/{pair['revision_parse_valid']}; done {pair['initial_done_reason']}/{pair['revision_done_reason']}; free RAM {status['resource_samples'][-1]['after']['free_ram_gib']:.2f} GiB", flush=True)
    status.update(status="COMPLETE", completed_pairs=len(pairs), final_resource=resource_sample())
    write_json(status_path, status)
    return status


def wilson(success: int, total: int, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = success / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def save_reliability_report():
    pairs = read_jsonl(DEV / "PAIRS.jsonl")
    status = read_json(DEV / "COLLECTION_STATUS.json")
    initial = sum(p["initial_parse_valid"] for p in pairs)
    revision = sum(p["revision_parse_valid"] for p in pairs)
    valid_wrong_i = sum(p["initial_parse_valid"] and not p["initial_correct"] for p in pairs)
    valid_wrong_r = sum(p["revision_parse_valid"] and not p["revision_correct"] for p in pairs)
    failures = read_jsonl(DEV / "RAW_CALLS.jsonl")
    failure_counts = {}
    for call in failures:
        kind = call.get("failure_type", "call_failed" if call.get("status") == "failed" else "none")
        failure_counts[kind] = failure_counts.get(kind, 0) + 1
    resources = [x for x in status.get("resource_samples", [])]
    min_ram = min((min(x["before"]["free_ram_gib"], x["after"]["free_ram_gib"]) for x in resources), default=None)
    min_vram = min((min(x["before"].get("gpu", {}).get("free_mib", 999999), x["after"].get("gpu", {}).get("free_mib", 999999)) for x in resources), default=None)
    criterion = status.get("status") == "COMPLETE" and len(pairs) == 10 and initial >= 9 and revision >= 9 and min_ram >= FREE_RAM_FLOOR_GIB and min_vram >= FREE_VRAM_FLOOR_MIB
    write_json(DEV / "RELIABILITY_GATE.json", {"status": "PASS" if criterion else "FAIL", "development_questions": len(pairs),
               "initial_parseable": initial, "revision_parseable": revision, "initial_requirement": "at least 9/10",
               "revision_requirement": "at least 9/10", "resource_requirement_met": bool(min_ram is not None and min_ram >= FREE_RAM_FLOOR_GIB and min_vram is not None and min_vram >= FREE_VRAM_FLOOR_MIB),
               "minimum_free_ram_gib": min_ram, "minimum_free_vram_mib": min_vram,
               "call_failure_count": sum(r.get("status") == "failed" for r in failures),
               "valid_but_incorrect_initial": valid_wrong_i, "valid_but_incorrect_revision": valid_wrong_r,
               "failure_categories": failure_counts, "evaluation_may_start": criterion})
    summary = f"""# Instruct development reliability gate

**Status: {'PASS' if criterion else 'FAIL'}**. This is a parser/reliability gate, not an accuracy threshold.

- Questions: {len(pairs)}/10; calls: {status.get('calls_completed', 0)}/20.
- Initial answers parseable: {initial}/10; revision answers parseable: {revision}/10.
- Parseable but incorrect answers: initial {valid_wrong_i}; revision {valid_wrong_r}.
- Minimum free RAM: {min_ram if min_ram is not None else 'not measured'} GiB (floor {FREE_RAM_FLOOR_GIB} GiB).
- Minimum free VRAM: {min_vram if min_vram is not None else 'not measured'} MiB (floor {FREE_VRAM_FLOOR_MIB} MiB).
- Call/extraction failure categories: `{json.dumps(failure_counts, sort_keys=True)}`.

The 40-question study {'will proceed automatically under the already frozen protocol and untouched selection' if criterion else 'will not run because the predeclared reliability/resource gate did not pass'}.
"""
    (DEV / "DEVELOPMENT_RELIABILITY_RESULTS.md").write_text(summary, encoding="utf-8")
    return criterion


def initial_only_gate(pairs, manifest):
    sys.path.insert(0, str(ROOT))
    from rechecklab.features import INITIAL_REQUIRED, FEATURE_NAMES, FEATURE_SCHEMA_FINGERPRINT, FEATURE_SCHEMA_VERSION, extract_features, validate_initial_record
    from rechecklab.predictor import load_artifact, revision_score, decide_recheck
    bundle = load_artifact(GATE_PATH, expected_sha256=GATE_SHA)
    if bundle["threshold"] != GATE_THRESHOLD or len(FEATURE_NAMES) != 19:
        raise ValueError("Frozen gate threshold or feature contract changed.")
    rows = []
    for p in pairs:
        raw_by_phase = [x for x in read_jsonl(EVAL / "RAW_CALLS.jsonl") if x.get("example_id") == p["example_id"] and x.get("phase") == "initial" and x.get("status") == "succeeded"]
        if len(raw_by_phase) != 1:
            raise ValueError("Expected exactly one saved initial call per evaluation question.")
        m = raw_by_phase[0]["response"].get("message", {})
        thinking, content = m.get("thinking", "") or "", m.get("content", "") or ""
        response = thinking + (("\n\n") if thinking and content else "") + content
        rows.append({"query": p["query"], "base_response": response, "base_answer": p["initial_answer"],
                     "base_actual_tokens": p["initial_tokens"], "base_done_reason": p["initial_done_reason"],
                     "base_finalizer_used": False})
    for row in rows:
        validate_initial_record({k: row[k] for k in INITIAL_REQUIRED})
    frame = pd.DataFrame([{k: row[k] for k in INITIAL_REQUIRED} for row in rows], columns=INITIAL_REQUIRED)
    features = extract_features(frame)
    predictor = bundle["predictor"]
    probs = predictor.probabilities(frame)
    scores = revision_score(probs, bundle["harm_weight"])
    actions = decide_recheck(probs, bundle["threshold"], bundle["harm_weight"])
    result = []
    for i, pair in enumerate(pairs):
        result.append({"example_id": pair["example_id"], "p_HELP": float(probs[i, 0]), "p_NO_CHANGE": float(probs[i, 1]),
                       "p_HURT": float(probs[i, 2]), "score": float(scores[i]), "threshold": bundle["threshold"],
                       "ml_gate_recheck_OOD": bool(actions[i]),
                       "completion_heuristic_recheck": pair["initial_done_reason"] == "length",
                       "feature_schema": FEATURE_SCHEMA_VERSION, "feature_schema_fingerprint": FEATURE_SCHEMA_FINGERPRINT,
                       **{f"feature__{name}": float(value) for name, value in features.iloc[i].to_dict().items()}})
    return result


def evaluate_policy_set():
    pairs = read_jsonl(EVAL / "PAIRS.jsonl")
    if len(pairs) != 40:
        raise ValueError("Evaluation analysis requires exactly forty complete saved response pairs.")
    manifest = read_json(EVAL / "EXPERIMENT_MANIFEST.json")
    predictions = initial_only_gate(pairs, manifest)
    pred_map = {r["example_id"]: r for r in predictions}
    actions = {"Never recheck": lambda p, g: False,
               "Always recheck": lambda p, g: True,
               "Completion heuristic": lambda p, g: p["initial_done_reason"] == "length",
               "Frozen v0.2 ML gate (cross-model/cross-protocol OOD)": lambda p, g: g["ml_gate_recheck_OOD"]}
    policy_rows, question_rows = [], []
    transitions = [p["outcome"] for p in pairs]
    rng = np.random.default_rng(20261117)
    bootstrap_indices = rng.integers(0, len(pairs), size=(20000, len(pairs)))
    never_correct = np.asarray([p["initial_correct"] for p in pairs], dtype=float)
    for name, choose in actions.items():
        selected = np.asarray([bool(choose(p, pred_map[p["example_id"]])) for p in pairs])
        correct = np.asarray([p["revision_correct"] if selected[i] else p["initial_correct"] for i, p in enumerate(pairs)], dtype=float)
        diffs = correct - never_correct
        boot = diffs[bootstrap_indices].mean(axis=1)
        help_mask = np.asarray([x == "HELP" for x in transitions])
        hurt_mask = np.asarray([x == "HURT" for x in transitions])
        nochange_mask = np.asarray([x == "NO_CHANGE" for x in transitions])
        init_tokens = sum(p["initial_tokens"] for p in pairs)
        revision_tokens = sum(p["revision_tokens"] for i, p in enumerate(pairs) if selected[i])
        replay = sum(p["initial_wall_seconds"] + (p["revision_wall_seconds"] if selected[i] else 0.0) for i, p in enumerate(pairs))
        policy_rows.append({"policy": name, "questions": len(pairs), "correct_count": int(correct.sum()),
                            "accuracy": float(correct.mean()), "accuracy_wilson_95_low": wilson(int(correct.sum()), len(pairs))[0],
                            "accuracy_wilson_95_high": wilson(int(correct.sum()), len(pairs))[1],
                            "rechecks": int(selected.sum()), "recheck_rate": float(selected.mean()),
                            "HELP_total": int(help_mask.sum()), "HELP_captured": int((help_mask & selected).sum()),
                            "HELP_missed": int((help_mask & ~selected).sum()),
                            "HURT_total": int(hurt_mask.sum()), "HURT_revision_accepted": int((hurt_mask & selected).sum()),
                            "HURT_avoided": int((hurt_mask & ~selected).sum()),
                            "NO_CHANGE_rechecks": int((nochange_mask & selected).sum()),
                            "mean_prompt_plus_output_tokens_per_question": (init_tokens + revision_tokens) / len(pairs),
                            "mean_initial_tokens_per_question": init_tokens / len(pairs),
                            "mean_selected_revision_tokens_per_question": revision_tokens / len(pairs),
                            "replayed_latency_seconds_total": replay,
                            "replayed_latency_seconds_per_question": replay / len(pairs),
                            "paired_accuracy_difference_vs_never": float(diffs.mean()),
                            "paired_bootstrap_95_low": float(np.quantile(boot, .025)),
                            "paired_bootstrap_95_high": float(np.quantile(boot, .975))})
    for i, p in enumerate(pairs):
        g = pred_map[p["example_id"]]
        row = {k: p[k] for k in ("example_id", "initial_done_reason", "revision_done_reason", "initial_parse_valid", "revision_parse_valid",
                                  "initial_correct", "revision_correct", "outcome", "initial_tokens", "revision_tokens",
                                  "initial_wall_seconds", "revision_wall_seconds")}
        row.update({k: g[k] for k in ("p_HELP", "p_NO_CHANGE", "p_HURT", "score", "threshold", "ml_gate_recheck_OOD", "completion_heuristic_recheck")})
        for policy, choose in actions.items():
            row["action__" + policy] = bool(choose(p, g))
        question_rows.append(row)
    write_csv(EVAL / "ML_PREDICTIONS.csv", predictions)
    write_csv(EVAL / "POLICY_COMPARISON.csv", policy_rows)
    write_csv(EVAL / "QUESTION_LEVEL.csv", question_rows)
    return policy_rows, question_rows


def write_csv(path: Path, rows):
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_report(policy_rows, question_rows):
    pairs = read_jsonl(EVAL / "PAIRS.jsonl")
    calls = read_jsonl(EVAL / "RAW_CALLS.jsonl")
    help_n, hurt_n, nochange_n = [sum(p["outcome"] == label for p in pairs) for label in ("HELP", "HURT", "NO_CHANGE")]
    miss = sum(p["outcome"] == "HELP" and not p["action__Frozen v0.2 ML gate (cross-model/cross-protocol OOD)"] for p in question_rows)
    invalid_initial = sum(not p["initial_parse_valid"] for p in pairs)
    invalid_revision = sum(not p["revision_parse_valid"] for p in pairs)
    total_latency = sum(c.get("wall_clock_seconds", 0) for c in calls if c.get("status") == "succeeded")
    policies = "\n".join(["| Policy | Accuracy (correct/40; Wilson 95% CI) | Rechecks | HELP captured/missed | HURT accepted/avoided | Mean prompt+output tokens/q | Replayed latency/q |",
        "|---|---:|---:|---:|---:|---:|---:|"] + [f"| {r['policy']} | {r['correct_count']}/40 ({r['accuracy']:.1%}; {r['accuracy_wilson_95_low']:.1%}-{r['accuracy_wilson_95_high']:.1%}) | {r['rechecks']}/40 ({r['recheck_rate']:.1%}) | {r['HELP_captured']}/{r['HELP_missed']} | {r['HURT_revision_accepted']}/{r['HURT_avoided']} | {r['mean_prompt_plus_output_tokens_per_question']:.1f} | {r['replayed_latency_seconds_per_question']:.2f}s |" for r in policy_rows])
    report = f"""# Frozen Instruct evaluation results

**Sample:** 40 fresh, pinned GSM8K main/train questions; all selected and frozen before development inference.  
**Checkpoint:** Qwen3-4B-Instruct-2507, `qwen3:4b-instruct`, digest `{DIGEST}`.  
**Gate:** frozen v0.2, unchanged; cross-model/cross-protocol OOD transfer diagnostic.

## Result

- Calls: {len(calls)}; successful recorded calls: {sum(c.get('status') == 'succeeded' for c in calls)}; total measured inference wall time: {total_latency:.1f}s.
- Parse failures: initial {invalid_initial}/40; revision {invalid_revision}/40.
- Question-level transitions: HELP {help_n}, HURT {hurt_n}, NO_CHANGE {nochange_n}.
- All four policies use the same saved initial/revision response pair per question. ML decisions use only the six canonical initial-only inputs; neither correctness, keys, nor revised responses enter the feature frame.

## Policies

{policies}

Selective policy latency is replayed from measured initial-call time plus measured revision-call time only for questions selected by that policy. It is not end-to-end deployed gate latency. All questions received a revision for paired comparison.

The question-level bootstrap uses 20,000 resamples with fixed seed 20261117 for paired accuracy differences versus never recheck. Intervals are descriptive; do not infer significance or generalization from this one 40-question dataset. See `POLICY_COMPARISON.csv`, `QUESTION_LEVEL.csv`, `ML_PREDICTIONS.csv`, and `RAW_CALLS.jsonl` for the complete evidence.
"""
    (EVAL / "EVALUATION_RESULTS.md").write_text(report, encoding="utf-8")


def finish_evaluation():
    status = collect_phase(EVAL)
    if status.get("status") != "COMPLETE":
        write_json(EVAL / "FINAL_STATUS.json", {"status": status.get("status"), "reason": status.get("stop_reason", status.get("failure")),
                                                  "evaluation_complete": False})
        return status
    policy_rows, question_rows = evaluate_policy_set()
    save_evaluation_report(policy_rows, question_rows)
    write_json(EVAL / "FINAL_STATUS.json", {"status": "COMPLETE", "evaluation_complete": True,
                 "questions": 40, "calls": status["calls_completed"], "frozen_gate_ood": True,
                 "artifact_sha256": GATE_SHA, "model_digest": DIGEST})
    return status


def run():
    prep = read_json(DEV / "PREPARATION_STATUS.json")
    if prep.get("status") != "FROZEN_BEFORE_INFERENCE" or prep.get("inference_started") is not False:
        raise ValueError("No valid frozen pre-inference manifests found.")
    dev_status = collect_phase(DEV)
    if dev_status.get("status") != "COMPLETE":
        passed = save_reliability_report()
        write_json(DEV / "FINAL_DECISION.json", {"development_gate": "PASS" if passed else "FAIL",
                   "evaluation_started": False, "reason": dev_status.get("stop_reason", dev_status.get("failure", dev_status.get("status")))})
        return
    passed = save_reliability_report()
    write_json(DEV / "FINAL_DECISION.json", {"development_gate": "PASS" if passed else "FAIL",
               "evaluation_started": bool(passed), "development_status": dev_status.get("status")})
    if not passed:
        print("Development criterion failed. Frozen 40-question evaluation was not started.", flush=True)
        return
    print("Development gate passed. Starting the untouched, preselected 40-question evaluation under the identical frozen protocol.", flush=True)
    eval_status = finish_evaluation()
    if eval_status.get("status") == "COMPLETE":
        print("Frozen 40-question evaluation completed.", flush=True)
    else:
        print(f"Evaluation stopped safely: {eval_status.get('status')}", flush=True)


def write_historical_review():
    text = """# Historical inference-failure review

This review separates response-generation failures from answer-scoring outcomes. Counts below were recalculated from saved raw calls/pair records; frozen studies were not changed.

## Thinking checkpoint: v0.3.3 independent pilot

- Checkpoint digest `359d7dd4...` identifies Qwen3-4B-Thinking-2507, not Instruct. Historical manifest used 1,024 output tokens per call with thinking enabled.
- 7/16 calls ended at the 1,024-token limit while producing a nonempty `thinking` field; the reasoning channel consumed the available generation budget before a parseable final answer. These are reasoning-token exhaustion plus a `length` completion reason.
- 1 additional initial call ended with `stop` but no `FINAL:` marker. That is a completed generation with an extraction/prompt-format failure, not a length truncation.
- No completed response with a `FINAL:` marker had malformed numeric syntax in this sample. All parseable answers were correct: initial 2/8, revisions 6/8; zero valid-but-wrong answers were observed.

## Earlier thinking-disabled check on the same Thinking-only checkpoint

The separate five-question check recorded 5/5 parseable and correct initial answers and 5/5 parseable and correct revisions, with no `length` completions. Its raw log retained call events but not response bodies, so missing-marker versus malformed-answer subtypes cannot be audited from that run. This small disjoint result did not establish checkpoint support for a non-thinking mode.

## Later protocol comparison on the Thinking checkpoint

- Thinking-off at 1,024 tokens: three length-limit truncations without a final answer; parseable calls on the remaining questions were correct.
- Thinking-on at 2,048 tokens: three calls exhausted the budget in the thinking channel before a parseable final answer; observed calls ending in `stop` that parsed were correct.
- No malformed final numeric answers or valid-but-incorrect answers were observed among the saved parsed examples. This comparison is not used to select the new protocol.

## Failure taxonomy for the new Instruct run

The new protocol records `reasoning_token_exhaustion`, `length_limit_truncation`, `missing_final_marker`, `malformed_final_answer`, `parseable/correct`, and `parseable/incorrect` separately. The Instruct model is verified by `general.finetune=Instruct` and version `2507`; no `think` request field is sent.
"""
    (DEV / "HISTORICAL_FAILURE_REVIEW.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", action="store_true")
    group.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
