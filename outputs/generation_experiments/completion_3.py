"""Three-question initial-answer completion experiment; at most six new calls.

Run with the existing environment from the project root:
  .venv/Scripts/python.exe -B outputs/generation_experiments/completion_3.py --execute

A uses archived answers. B/C use one concise prompt and unchanged parser.
Streaming runs in one child process at a time so the parent can monitor RAM
through model loading and inference, and disconnect promptly on resource abort.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
HOST = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"


def stream_worker(request_path, chunks_path, result_path):
    """No keys, grading, dataset loaders or model management in this process."""
    payload = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if (payload["model"] != MODEL or payload["options"]["num_ctx"] != 4096
            or payload["options"]["num_predict"] not in (2048, 3072)
            or len(payload["messages"]) != 1 or set(payload["messages"][0]) != {"role", "content"}):
        raise ValueError("Worker request violates the initial-only completion protocol.")
    started = time.perf_counter()
    visible, thinking, last, error = [], [], None, None
    try:
        request = Request(HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})
        with Path(chunks_path).open("x", encoding="utf-8", buffering=1) as saved:
            with urlopen(request, timeout=600) as response:
                for line in response:
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    saved.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    if chunk.get("error"):
                        raise RuntimeError(chunk["error"])
                    if chunk.get("model") != MODEL:
                        raise RuntimeError("Stream solver identity changed.")
                    message = chunk.get("message", {})
                    visible.append(message.get("content", ""))
                    thinking.append(message.get("thinking", ""))
                    last = chunk
        if last is None or last.get("done") is not True:
            raise RuntimeError("Stream ended without final token accounting.")
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
        if isinstance(exc, HTTPError):
            error["backend_body"] = exc.read(16384).decode("utf-8", errors="replace")
    result = {"wall_seconds": time.perf_counter() - started, "error": error,
              "raw_final_chunk": last, "visible_response": "".join(visible),
              "thinking_response": "".join(thinking)}
    with Path(result_path).open("x", encoding="utf-8") as saved:
        json.dump(result, saved, ensure_ascii=False, allow_nan=False)
        saved.write("\n")
    return 1 if error else 0


# Child workers avoid importing pandas/numpy and their extra memory footprint.
if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    raise SystemExit(stream_worker(*sys.argv[2:]))

import numpy as np
import pandas as pd

from preflight_5 import (append_record, digest, hardware_snapshot, inspect_backend,
                         metadata, timestamp, verify_backend, write_json)
from rechecklab.study_protocol import parse_study_answer, grade_answer

ARCHIVE = ROOT / "outputs/generation_experiments/preflight_5_2026-11-03"
CONCISE_PROMPT = ("Solve the multiple-choice question. Avoid unnecessary explanation. "
                  "Finish with exactly FINAL: X, where X is one A-E letter.")
POSITIONS = (0, 2, 4)
CONTEXT = 4096
MAX_NEW_CALLS = 6
MIN_COLD_START_GIB = 5.0
MIN_WARM_START_GIB = 2.0
MIN_C_START_GIB = 2.5
MIN_B_RAM_FOR_C_GIB = 2.0
CRITICAL_GIB = 1.5
SAMPLE_INTERVAL_SECONDS = 0.5


class MemoryStatus(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                *[(name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended")]]


def memory_sample():
    state = MemoryStatus()
    state.length = ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
        raise RuntimeError("Cannot monitor available RAM; refuse inference.")
    return {"timestamp_utc": timestamp(), "available_ram_gib": state.available / 2**30,
            "memory_load_pct": int(state.load)}


def protected_hashes(run_dir):
    result = {}
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in excluded
                                and Path(parent, d).resolve() != run_dir)
        for name in sorted(files):
            path = Path(parent, name)
            result[path.relative_to(ROOT).as_posix()] = digest(path)
    return result


def historical_diagnosis(run_dir, archived):
    rows = []
    for call in archived:
        if call["valid_final"]:
            continue
        message = call["raw_response"]["message"]
        visible, thinking = message["content"], message.get("thinking", "")
        rows.append({"call_id": call["call_id"], "question_id": call["question_id"],
                     "configuration": call["configuration"], "completion_reason": call["completion_reason"],
                     "generated_tokens": call["generated_tokens"], "thinking_chars": len(thinking),
                     "visible_chars": len(visible), "parser_result_unchanged": parse_study_answer(visible, "choice"),
                     "diagnosis": "output cap before any visible answer; consistent with still thinking" if not visible
                                  else "visible option analysis truncated before FINAL; had left thinking channel"})
    pd.DataFrame(rows).to_csv(run_dir / "historical_failure_diagnosis.csv", index=False)
    write_json(run_dir / "historical_failure_diagnosis.json", {
        "archive_calls_sha256": digest(ARCHIVE / "calls.jsonl"), "invalid_count": len(rows), "failures": rows,
        "complete_visible_FINAL_rejected": 0,
        "caveat": "Old nonstream responses cannot prove the exact final channel token. "
                  "03_C includes an answer-like preference inside partial analysis, not a declared FINAL; do not regrade.",
        "valid_length_case": "04_initial contains FINAL: C, correctly parsed C and graded incorrect.",
        "stop_sequences": "Configured <|im_start|> and <|im_end|> equal model defaults. "
                          "All seven invalid responses report length, not stop; no evidence of direct stop-string termination.",
        "historical_results_changed": False,
    })


def context_check(query, budget):
    message = {"role": "user", "content": CONCISE_PROMPT + "\n\n" + query}
    upper = len(message["content"].encode("utf-8")) + 256
    if upper + budget > CONTEXT:
        raise RuntimeError("Conservative full-request context bound failed; do not increase context.")
    return message, {"input_token_upper_bound": upper, "output_budget": budget,
                     "full_request_upper_bound": upper + budget, "context_window": CONTEXT}


def run_monitored(payload, call_id, run_dir):
    request_path = run_dir / f"request_{call_id}.json"
    chunks_path = run_dir / f"raw_chunks_{call_id}.jsonl"
    result_path = run_dir / f"response_{call_id}.json"
    write_json(request_path, payload)
    samples, abort = [], None
    start = time.perf_counter()
    with (run_dir / f"worker_{call_id}.log").open("x", encoding="utf-8") as worker_log:
        process = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()), "--worker",
                                    str(request_path), str(chunks_path), str(result_path)],
                                   stdout=worker_log, stderr=subprocess.STDOUT,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            while True:
                try:
                    sample = {"call_id": call_id, **memory_sample()}
                    samples.append(sample)
                    append_record(run_dir / "ram_samples.jsonl", sample)
                    if sample["available_ram_gib"] < CRITICAL_GIB:
                        abort = f"Critical RAM: {sample['available_ram_gib']:.3f} GiB < {CRITICAL_GIB} GiB"
                    if time.perf_counter() - start > 660:
                        abort = "Wall-time ceiling exceeded"
                except BaseException as exc:
                    abort = f"Resource monitoring failed: {type(exc).__name__}: {exc}"
                if abort:
                    break
                if process.poll() is not None:
                    break
                time.sleep(SAMPLE_INTERVAL_SECONDS)
        finally:
            # Any parent exception, including interruption between samples, must
            # disconnect the active client before we report/stop the experiment.
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
    result = json.loads(result_path.read_text(encoding="utf-8")) if result_path.exists() else None
    if abort or process.returncode != 0 or result is None or result.get("error"):
        raise MonitoredFailure({"resource_abort": abort, "worker_exit_code": process.returncode,
                               "response": result, "partial_stream_file": chunks_path.name,
                               "minimum_ram_gib": min(s["available_ram_gib"] for s in samples) if samples else None,
                               "cancellation": "Streaming client process terminated on abort; no next request permitted"})
    result["minimum_ram_gib"] = min(sample["available_ram_gib"] for sample in samples)
    return result


class MonitoredFailure(RuntimeError):
    def __init__(self, detail):
        self.detail = detail
        super().__init__(json.dumps(detail, ensure_ascii=False))


def record_metrics(condition, question, response, budget, guard):
    final = response["raw_final_chunk"]
    if final.get("done") is not True or final.get("done_reason") not in ("stop", "length"):
        raise RuntimeError("Invalid stream completion status.")
    for name in ("prompt_eval_count", "eval_count"):
        if isinstance(final.get(name), bool) or not isinstance(final.get(name), int) or final[name] < 0:
            raise RuntimeError("Missing actual final token counts.")
    if (final["eval_count"] > budget or final["prompt_eval_count"] > guard["input_token_upper_bound"]
            or final["prompt_eval_count"] + budget > CONTEXT):
        raise RuntimeError("Actual token counts violate the frozen full-request bounds.")
    visible = response["visible_response"]
    if "<think>" in visible or "</think>" in visible:
        raise RuntimeError("Thinking text leaked into visible channel; stop without rewriting it.")
    parsed = parse_study_answer(visible, "choice")
    return {"condition": condition, "question_id": question["source_question_id"], "source": "new_streamed_initial",
            "budget": budget, "visible_response": visible, "parsed_answer": parsed, "valid_final": parsed is not None,
            "correct": grade_answer(parsed, question["answer_key"], "choice"), "answer_key": question["answer_key"],
            "completion_reason": final["done_reason"], "prompt_tokens": final["prompt_eval_count"],
            "generated_tokens": final["eval_count"], "total_tokens": final["prompt_eval_count"] + final["eval_count"],
            "wall_seconds": response["wall_seconds"], "minimum_ram_gib": response["minimum_ram_gib"],
            "context_check": guard, "thinking_chars": len(response["thinking_response"])}


def summarize(run_dir, records, problem, skip_c):
    rows = []
    for condition in ("A", "B", "C"):
        subset = [record for record in records if record["condition"] == condition]
        rows.append({"condition": condition, "sample_questions": len(subset), "planned_questions": 3,
                     "source": "archived reference, Ollama 0.40.13" if condition == "A" else "new, Ollama frozen in protocol",
                     "valid_FINAL_answers": sum(row["valid_final"] for row in subset),
                     "correct_answers_diagnostic": sum(row["correct"] for row in subset),
                     "length_completions": sum(row["completion_reason"] == "length" for row in subset),
                     "generated_tokens_total": sum(row["generated_tokens"] for row in subset),
                     "actual_tokens_total": sum(row["total_tokens"] for row in subset),
                     "average_generated_tokens": float(np.mean([row["generated_tokens"] for row in subset])) if subset else None,
                     "average_actual_tokens": float(np.mean([row["total_tokens"] for row in subset])) if subset else None,
                     "average_latency_seconds": float(np.mean([row["wall_seconds"] for row in subset])) if subset else None,
                     "ram_observation_method": "archived post-call snapshots" if condition == "A" else "0.5-second monitoring during inference",
                     "lowest_available_ram_gib": min(row["minimum_ram_gib"] for row in subset) if subset else None})
    pd.DataFrame(rows).to_csv(run_dir / "condition_comparison.csv", index=False)
    samples = [json.loads(line) for line in (run_dir / "ram_samples.jsonl").read_text().splitlines()] if (run_dir / "ram_samples.jsonl").exists() else []
    write_json(run_dir / "summary.json", {"status": "STOPPED" if problem else "COMPLETE_WITH_C_SKIPPED" if skip_c else "COMPLETE",
               "problem": problem, "condition_C_skipped_reason": skip_c, "conditions": rows,
               "new_calls_completed": sum(row["condition"] != "A" for row in records),
               "minimum_monitored_ram_gib": min(sample["available_ram_gib"] for sample in samples) if samples else None,
               "diagnostic_only": "Three deliberately selected completion-stress questions; no accuracy conclusion.",
               "reference_caveat": "A is archived on Ollama 0.40.13/nonstream; B and C share the current frozen backend/streaming. "
                                   "A-to-B differences cannot isolate prompt effects from backend changes.",
               "no_rechecks_no_finalizers_no_retries_no_downloads": True, "gate_and_streamlit_unchanged": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--run-name", default="completion_3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not args.execute:
        parser.error("--execute is required for at most six new initial-answer calls")
    if not args.run_name or args.run_name in (".", "..") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.run_name):
        parser.error("run-name must be a simple directory name")
    run_dir = ROOT / "outputs/generation_experiments" / args.run_name
    run_dir.mkdir(exist_ok=False)
    before = protected_hashes(run_dir)
    write_json(run_dir / "original_files_before.json", before)
    archived = [json.loads(line) for line in (ARCHIVE / "calls.jsonl").read_text(encoding="utf-8").splitlines()]
    historical_diagnosis(run_dir, archived)
    old_questions = [json.loads(line) for line in (ARCHIVE / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    keys = {row["source_question_id"]: row["answer_key"] for row in (
        json.loads(line) for line in (ARCHIVE / "trusted_keys.jsonl").read_text(encoding="utf-8").splitlines())}
    questions = [{**old_questions[index], "answer_key": keys[old_questions[index]["source_question_id"]]} for index in POSITIONS]
    records, problem, skip_c, started_calls = [], None, None, 0
    for question in questions:
        old = next(row for row in archived if row["configuration"] == "initial" and row["question_id"] == question["source_question_id"])
        record = {k: old[k] for k in ("question_id", "answer_key", "visible_response", "parsed_answer", "valid_final", "correct",
                  "completion_reason", "prompt_tokens", "generated_tokens", "total_tokens", "wall_seconds")}
        record.update(condition="A", source="archived_initial_reference", budget=2048,
                      minimum_ram_gib=old["hardware_after"]["available_physical_memory_bytes"] / 2**30)
        records.append(record)
        append_record(run_dir / "records.jsonl", record)
    try:
        backend = inspect_backend()
        if backend["installed_model"]["digest"] != json.loads((ARCHIVE / "protocol.json").read_text())["model_digest"]:
            raise RuntimeError("Installed checkpoint differs from required archived Qwen checkpoint.")
        if metadata("/api/ps").get("models"):
            raise RuntimeError("A model is already loaded; refuse to change concurrent model state.")
        for question in questions:
            context_check(question["query"], 3072)
        protocol = {"version": "completion-three-v1", "created_utc": timestamp(), "backend": backend,
                    "model": MODEL, "model_digest": backend["installed_model"]["digest"],
                    "condition_A": "Archived initial prompt/2048 outputs only; no fresh calls",
                    "condition_B": {"prompt": CONCISE_PROMPT, "budget": 2048},
                    "condition_C": {"prompt": CONCISE_PROMPT, "budget": 3072, "conditional_on_resource_guards": True},
                    "sampling": {"seed": 42, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                 "repeat_penalty": 1.0, "thinking": True, "stop": ["<|im_start|>", "<|im_end|>"]},
                    "context_window": CONTEXT, "stream": True,
                    "transport": "Sequential streaming child client; parent monitors RAM during load/inference; abort closes client connection",
                    "resource_limits": {"minimum_cold_start_ram_gib": MIN_COLD_START_GIB,
                        "minimum_warm_start_ram_gib": MIN_WARM_START_GIB, "minimum_C_start_ram_gib": MIN_C_START_GIB,
                        "minimum_B_ram_for_C_gib": MIN_B_RAM_FOR_C_GIB, "critical_abort_ram_gib": CRITICAL_GIB,
                        "sample_interval_seconds": SAMPLE_INTERVAL_SECONDS},
                    "questions": [{k: v for k, v in question.items() if k != "answer_key"} for question in questions],
                    "selection": "Old positions 1,3,5: two failed initials plus one completed control; diagnostic stress sample",
                    "parser": "unchanged rechecklab.study_protocol.parse_study_answer(content, choice)",
                    "reference_version": "Ollama 0.40.13; current backend differs, so A is descriptive reference only",
                    "maximum_new_calls": MAX_NEW_CALLS, "no_retries": True, "no_finalizer": True,
                    "script_sha256": digest(__file__), "helper_sha256": digest(Path(__file__).with_name("preflight_5.py"))}
        write_json(run_dir / "protocol.json", protocol)
        protocol_hash = digest(run_dir / "protocol.json")
        write_json(run_dir / "hardware_before.json", hardware_snapshot())
        b_minimum = float("inf")
        for condition, budget in (("B", 2048), ("C", 3072)):
            if condition == "C" and b_minimum < MIN_B_RAM_FOR_C_GIB:
                skip_c = f"B minimum RAM {b_minimum:.3f} GiB below C eligibility floor {MIN_B_RAM_FOR_C_GIB}"
                break
            for index, question in enumerate(questions, 1):
                available = memory_sample()["available_ram_gib"]
                floor = MIN_COLD_START_GIB if started_calls == 0 else MIN_C_START_GIB if condition == "C" else MIN_WARM_START_GIB
                if available < floor:
                    if condition == "C" and index == 1:
                        skip_c = f"Available RAM {available:.3f} GiB below C start floor {floor}"
                        break
                    raise RuntimeError(f"Available RAM {available:.3f} GiB below call-start floor {floor}")
                verify_backend(backend)
                active = metadata("/api/ps").get("models", [])
                if any(model.get("name") != MODEL for model in active):
                    raise RuntimeError("Another loaded model detected; stop without changing it.")
                if digest(run_dir / "protocol.json") != protocol_hash or started_calls >= MAX_NEW_CALLS:
                    raise RuntimeError("Frozen protocol changed or call ceiling reached.")
                message, guard = context_check(question["query"], budget)
                payload = {"model": MODEL, "messages": [message], "stream": True, "think": True,
                           "options": {"num_ctx": CONTEXT, "num_predict": budget, "seed": 42,
                                       "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                                       "repeat_penalty": 1.0, "stop": ["<|im_start|>", "<|im_end|>"]}}
                call_id = f"{condition}_{index:02d}"
                append_record(run_dir / "events.jsonl", {"event": "started", "call_id": call_id,
                              "question_id": question["source_question_id"], "timestamp_utc": timestamp(), "context_check": guard})
                started_calls += 1
                print(f"[{started_calls}/6] {call_id}, budget={budget}, RAM={available:.2f} GiB, context upper={guard['full_request_upper_bound']}/4096", flush=True)
                response = run_monitored(payload, call_id, run_dir)
                record = record_metrics(condition, question, response, budget, guard)
                active = metadata("/api/ps").get("models", [])
                if len(active) != 1 or active[0].get("digest") != protocol["model_digest"] or active[0].get("context_length") != CONTEXT:
                    raise RuntimeError("Active solver/context differs from the frozen request.")
                record["loaded_model_after"] = active[0]
                records.append(record)
                append_record(run_dir / "records.jsonl", record)
                append_record(run_dir / "events.jsonl", {"event": "completed", "call_id": call_id, "timestamp_utc": timestamp()})
                if condition == "B":
                    b_minimum = min(b_minimum, record["minimum_ram_gib"])
                print(f"{call_id} complete: valid={record['valid_final']}, reason={record['completion_reason']}, "
                      f"generated={record['generated_tokens']}, seconds={record['wall_seconds']:.2f}, min RAM={record['minimum_ram_gib']:.2f} GiB", flush=True)
        verify_backend(backend)
    except BaseException as exc:
        problem = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "failure.json", {"error": problem, "detail": getattr(exc, "detail", None),
                   "no_more_requests": True, "generated_usage_if_interrupted": "Unknown unless final server accounting was received"})
        print("STOPPED: " + problem, flush=True)
    finally:
        summarize(run_dir, records, problem, skip_c)
        after = protected_hashes(run_dir)
        changed = [name for name in before if after.get(name) != before[name]]
        added = sorted(set(after) - set(before))
        write_json(run_dir / "integrity.json", {"original_files_checked": len(before), "changed_or_removed": changed,
                   "new_files_outside_run": added, "all_originals_unchanged": not changed and not added,
                   "includes_previous_generation_experiments": True})
        print(json.dumps({"outputs": str(run_dir), "problem": problem, "C_skipped": skip_c,
                          "new_calls_started": started_calls, "originals_unchanged": not changed and not added}), flush=True)
        if problem or changed or added:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
