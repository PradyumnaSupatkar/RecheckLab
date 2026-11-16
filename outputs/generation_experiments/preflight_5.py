"""Exactly five existing CommonsenseQA training questions, at most 20 LLM calls.

Run from the project root with the existing environment:
  .venv/Scripts/python.exe -B outputs/generation_experiments/preflight_5.py --execute

No downloads, retries, finalizers, model changes, or production-file writes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
EXPERIMENT_ROOT = ROOT / "outputs/generation_experiments"
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from rechecklab.data import question_hash, transition_label
from rechecklab.study_data import benchmark_candidates
from rechecklab.study_protocol import INITIAL_PROMPT, RECHECK_PROMPT, parse_study_answer, grade_answer

HOST = "http://127.0.0.1:11434"
MODEL = "qwen3:4b"
QUESTION_COUNT = 5
MAX_CALLS = 20
CONTEXT = 4096
OUTPUT_BUDGET = 2048
SELECTION_SEED = 46
INITIAL_SEED = 42
RECHECK_SEED = 43
REVISION = "94630fe30dad47192a8546eb75f094926d47e155"
RECHECK_PROMPTS = {
    "A": RECHECK_PROMPT,
    "B": ("Compare the initial answer with all five choices. Check the question's wording, ordinary "
          "real-world meaning, and relevant constraints. Change the answer only if another choice "
          "clearly fits better; otherwise retain it. Explain briefly. End with FINAL: X, replacing "
          "X with one A-E letter."),
    "C": ("First assess all five choices from the question itself, treating the previous answer as "
          "a hypothesis rather than evidence. Identify the best-supported choice, then compare it "
          "with the initial answer and retain or revise accordingly. Explain briefly. End with "
          "FINAL: X, replacing X with one A-E letter."),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def append_record(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def protect_originals():
    result = {}
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    for parent, directories, files in os.walk(ROOT):
        directories[:] = sorted(d for d in directories if d not in excluded
                                and Path(parent, d).resolve() != EXPERIMENT_ROOT)
        for name in sorted(files):
            path = Path(parent, name)
            result[path.relative_to(ROOT).as_posix()] = digest(path)
    return result


def metadata(route, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(HOST + route, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=20) as response:
        result = json.load(response)
    if not isinstance(result, dict) or result.get("error"):
        raise RuntimeError(f"Metadata error on {route}: {result}")
    return result


def inspect_backend():
    version = metadata("/api/version")
    tags = metadata("/api/tags")
    models = [model for model in tags.get("models", []) if model.get("name") == MODEL]
    if len(models) != 1 or not models[0].get("digest"):
        raise RuntimeError("Required installed qwen3:4b is missing; never download automatically.")
    show = metadata("/api/show", {"model": MODEL})
    info = show.get("model_info", {})
    if info.get("general.architecture") != "qwen3" or info.get("tokenizer.ggml.model") != "gpt2":
        raise RuntimeError("Conservative context bound requires the inspected Qwen byte-BPE tokenizer.")
    if show.get("thinking", {}).get("values") != [True]:
        raise RuntimeError("Thinking capability differs from the inspected checkpoint.")
    if "<think>" not in show.get("template", "") or "<|im_start|>" not in show.get("template", ""):
        raise RuntimeError("Unexpected chat template; do not assume its context overhead.")
    return {"ollama_version": version["version"], "installed_model": models[0], "show": show}


def verify_backend(frozen):
    current = inspect_backend()
    if current != frozen:
        raise RuntimeError("Ollama version, model digest or template/settings changed; stopping.")


def hardware_snapshot():
    result = {"timestamp_utc": timestamp()}
    try:
        command = ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                   "--format=csv,noheader"]
        response = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
        result["nvidia_smi"] = response.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        result["gpu_monitor_error"] = str(exc)
    if sys.platform == "win32":
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong),
                        ("total_physical", ctypes.c_ulonglong), ("available_physical", ctypes.c_ulonglong),
                        ("total_page", ctypes.c_ulonglong), ("available_page", ctypes.c_ulonglong),
                        ("total_virtual", ctypes.c_ulonglong), ("available_virtual", ctypes.c_ulonglong),
                        ("available_extended", ctypes.c_ulonglong)]
        state = MemoryStatus()
        state.length = ctypes.sizeof(state)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
            result.update(physical_memory_bytes=state.total_physical,
                          available_physical_memory_bytes=state.available_physical,
                          memory_load_pct=state.load)
    return result


def select_questions(run_dir):
    path = ROOT / f"outputs/live_study/v0.3.0/raw/commonsenseqa/{REVISION}/train.parquet"
    # This is the existing official TRAIN split, never reserved test outcomes.
    candidates = benchmark_candidates("commonsenseqa", "train", pd.read_parquet(path), REVISION)
    prior_ids, prior_groups, manifests = set(), set(), []
    paths = list((ROOT / "outputs/live_study").rglob("questions.jsonl"))
    paths += [p for p in EXPERIMENT_ROOT.rglob("questions.jsonl") if run_dir not in p.parents]
    for manifest in sorted(paths):
        manifests.append({"path": manifest.relative_to(ROOT).as_posix(), "sha256": digest(manifest)})
        for line in manifest.read_text(encoding="utf-8-sig").split("\n"):
            if line.strip():
                row = json.loads(line)
                prior_groups.add(question_hash(row["query"]))
                if row.get("source_dataset") == "tau/commonsense_qa" or row.get("dataset") == "commonsenseqa":
                    prior_ids.add(row["source_question_id"])
    eligible = sorted((row for row in candidates if row["source_question_id"] not in prior_ids
                       and row["group_id"] not in prior_groups), key=lambda row: row["source_question_id"])
    order = np.random.default_rng(SELECTION_SEED).permutation(len(eligible))[:QUESTION_COUNT]
    selected = [eligible[index] for index in order]
    if len(selected) != QUESTION_COUNT or len({row["group_id"] for row in selected}) != QUESTION_COUNT:
        raise AssertionError("Exactly five distinct eligible benchmark questions are required.")
    keys, public = {}, []
    for row in selected:
        key = row["_gold_answer"]
        if key not in tuple("ABCDE"):
            raise AssertionError("Invalid trusted published choice key.")
        keys[row["source_question_id"]] = key
        public.append({k: v for k, v in row.items() if k != "_gold_answer"})
    for row in public:
        append_record(run_dir / "questions.jsonl", row)
        append_record(run_dir / "trusted_keys.jsonl", {"source_question_id": row["source_question_id"],
                                                      "answer_key": keys[row["source_question_id"]]})
    write_json(run_dir / "selection_audit.json", {
        "source_path": path.relative_to(ROOT).as_posix(), "source_sha256": digest(path),
        "source_revision": REVISION, "official_split": "train", "selection_seed": SELECTION_SEED,
        "eligible_questions": len(eligible), "selected_questions": len(public),
        "previous_manifests": manifests, "overlap_prior_ids": 0, "overlap_prior_groups": 0,
        "selection": "Seeded permutation of eligible published IDs sorted lexically; no outcome-based selection.",
        "future_use": "Operational preflight only; exclude these IDs from future development/evaluation.",
    })
    return public, keys


def make_messages(query, configuration, saved_visible=None):
    messages = [{"role": "user", "content": INITIAL_PROMPT + "\n\n" + query}]
    if configuration != "initial":
        if configuration not in RECHECK_PROMPTS or not isinstance(saved_visible, str):
            raise ValueError("A recheck needs a named prompt and the exact initial visible content.")
        messages.extend([{"role": "assistant", "content": saved_visible},
                         {"role": "user", "content": RECHECK_PROMPTS[configuration]}])
    if any(set(message) != {"role", "content"} for message in messages):
        raise AssertionError("Only role/content may enter requests; never keys or thinking fields.")
    return messages


def context_guard(messages):
    # GPT2 byte-BPE yields no more tokens than UTF-8 content bytes. The inspected
    # simple Qwen user/assistant branches have <128 bytes of template overhead
    # per message plus <128 bytes for the generation prefix. No tools/system.
    upper = sum(len(message["content"].encode("utf-8")) for message in messages)
    upper += 128 * len(messages) + 128
    if upper + OUTPUT_BUDGET > CONTEXT:
        raise RuntimeError(f"Cannot prove full request fits: input upper {upper} + output {OUTPUT_BUDGET} > {CONTEXT}.")
    return {"input_token_upper_bound": upper, "output_tokens_reserved": OUTPUT_BUDGET,
            "total_token_upper_bound": upper + OUTPUT_BUDGET, "context_window": CONTEXT,
            "method": "UTF-8 byte-BPE upper bound plus conservative template overhead; no truncation permitted"}


def invoke(messages, seed):
    """Transport deliberately has no question key/correctness argument."""
    payload = {"model": MODEL, "messages": messages, "stream": False, "think": True,
               "options": {"num_predict": OUTPUT_BUDGET, "num_ctx": CONTEXT, "seed": seed,
                           "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                           "repeat_penalty": 1.0, "stop": ["<|im_start|>", "<|im_end|>"]}}
    request = Request(HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                      headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=600) as response:
            raw = json.load(response)
    except HTTPError as exc:
        exc.backend_body = exc.read().decode("utf-8", errors="replace")
        exc.wall_seconds = time.perf_counter() - start
        raise
    except BaseException as exc:
        exc.wall_seconds = time.perf_counter() - start
        raise
    return raw, time.perf_counter() - start


def validate_response(raw, guard):
    if raw.get("error") or raw.get("model") != MODEL or raw.get("done") is not True:
        raise RuntimeError(f"Incomplete response, model mismatch or backend error: {raw.get('error')}")
    message = raw.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise RuntimeError("Missing visible response content.")
    if "thinking" in message and not isinstance(message["thinking"], str):
        raise RuntimeError("Invalid thinking field type.")
    if "<think>" in message["content"] or "</think>" in message["content"]:
        raise RuntimeError("Thinking leaked into visible content; stop rather than silently stripping it.")
    if raw.get("done_reason") not in ("stop", "length"):
        raise RuntimeError("Unrecognized completion reason.")
    for key in ("prompt_eval_count", "eval_count"):
        count = raw.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise RuntimeError(f"Missing or invalid actual token count {key}.")
    if raw["eval_count"] > OUTPUT_BUDGET:
        raise RuntimeError("Backend exceeded the requested generation limit.")
    if raw["prompt_eval_count"] > guard["input_token_upper_bound"]:
        raise RuntimeError("Actual prompt token count exceeds the conservative bound.")
    if raw["prompt_eval_count"] + OUTPUT_BUDGET > CONTEXT:
        raise RuntimeError("Actual prompt leaves insufficient room for the full generation budget.")
    return message["content"]


def validate_parser():
    examples = [("FINAL: A", "A"), ("FINAL: (b).", "B"), ("FINAL ANSWER: E!", "E"),
                ("FINAL: A or B", None), ("The answer is C", None), ("", None)]
    for content, expected in examples:
        if parse_study_answer(content, "choice") != expected:
            raise AssertionError("Existing explicit FINAL parser failed its contract check.")
    if not grade_answer("A", "A", "choice") or grade_answer(None, "A", "choice"):
        raise AssertionError("Offline exact-choice grading failed its contract check.")


def summarize(run_dir, records, problem, started):
    rows = []
    for configuration in ("initial", "A", "B", "C"):
        subset = [row for row in records if row["configuration"] == configuration]
        good = [row for row in subset if row["error"] is None]
        rows.append({"configuration": configuration, "planned_calls": QUESTION_COUNT,
                     "attempted_calls": len(subset), "successful_calls": len(good),
                     "valid_FINAL_answers": sum(row["valid_final"] for row in good),
                     "correct_answers": sum(row["correct"] for row in good),
                     "length_completions": sum(row["completion_reason"] == "length" for row in good),
                     "average_prompt_tokens": float(np.mean([row["prompt_tokens"] for row in good])) if good else None,
                     "average_generated_tokens": float(np.mean([row["generated_tokens"] for row in good])) if good else None,
                     "average_total_tokens": float(np.mean([row["total_tokens"] for row in good])) if good else None,
                     "average_wall_seconds": float(np.mean([row["wall_seconds"] for row in good])) if good else None})
    pairs = []
    for question_id in dict.fromkeys(row["question_id"] for row in records):
        by_config = {row["configuration"]: row for row in records if row["question_id"] == question_id and row["error"] is None}
        if "initial" not in by_config:
            continue
        base = by_config["initial"]
        for configuration in RECHECK_PROMPTS:
            if configuration in by_config:
                final = by_config[configuration]
                pairs.append({"question_id": question_id, "configuration": configuration,
                              "initial_correct": base["correct"], "recheck_correct": final["correct"],
                              "initial_valid": base["valid_final"], "recheck_valid": final["valid_final"],
                              "label": transition_label(base["correct"], final["correct"]),
                              "oracle_correct_diagnostic": base["correct"] or final["correct"],
                              "policy_total_tokens": base["total_tokens"] + final["total_tokens"]})
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(run_dir / "paired_outcomes.csv", index=False)
    for row in rows:
        selected = [pair for pair in pairs if pair["configuration"] == row["configuration"]]
        row.update(HELP=sum(pair["label"] == "HELP" for pair in selected),
                   HURT=sum(pair["label"] == "HURT" for pair in selected),
                   oracle_correct_diagnostic=sum(pair["oracle_correct_diagnostic"] for pair in selected) if selected else None)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    completed = problem is None and len(records) == MAX_CALLS and all(row["error"] is None for row in records)
    summary = {"status": "COMPLETE" if completed else "STOPPED", "planned_questions": QUESTION_COUNT,
               "planned_generation_calls": MAX_CALLS, "attempted_generation_calls": len(records),
               "all_calls_completed_successfully": completed, "problem": problem, "by_configuration": rows,
               "wall_seconds_total": time.perf_counter() - started,
               "diagnostic_only": "Five questions cannot establish an accuracy conclusion.",
               "no_finalizer_no_retries_no_downloads": True, "development_50_not_run": True}
    write_json(run_dir / "summary.json", summary)
    lines = ["# Five-question generation preflight", "", f"Status: {summary['status']}. "
             f"Generation calls attempted: {len(records)}/{MAX_CALLS}.", "",
             "| Configuration | Calls OK | Valid FINAL | Correct | Length stops | Avg actual tokens | Avg seconds |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        tokens = "N/A" if row["average_total_tokens"] is None else f"{row['average_total_tokens']:.1f}"
        seconds = "N/A" if row["average_wall_seconds"] is None else f"{row['average_wall_seconds']:.2f}"
        lines.append(f"| {row['configuration']} | {row['successful_calls']}/5 | {row['valid_FINAL_answers']}/5 | "
                     f"{row['correct_answers']}/5 | {row['length_completions']} | {tokens} | {seconds} |")
    lines.extend(["", "Accuracy counts and paired oracles are diagnostic only. Invalid completed answers "
                  "are scored incorrect. Actual tokens include prompt plus generated tokens, including thinking. "
                  "Only saved visible initial content enters all three independent rechecks.", "",
                  "No existing protocol or artifact was modified; see integrity.json. No 50-question study ran.",
                  "", "Problem: " + (problem or "none"), ""])
    (run_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Authorize at most 20 calls on exactly five questions")
    parser.add_argument("--run-name", default="preflight_5_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    args = parser.parse_args()
    if not args.execute:
        parser.error("Use --execute to run exactly this five-question operational preflight.")
    if not args.run_name or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in args.run_name) or args.run_name in (".", ".."):
        parser.error("run-name must be a simple directory name")
    run_dir = EXPERIMENT_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    before = protect_originals()
    write_json(run_dir / "original_files_before.json", before)
    records, problem, calls_started = [], None, 0
    try:
        validate_parser()
        backend = inspect_backend()
        initially_loaded = metadata("/api/ps")
        if initially_loaded.get("models"):
            raise RuntimeError("Models are already loaded; stop rather than changing their memory state.")
        questions, keys = select_questions(run_dir)
        protocol = {"protocol_version": "generation-preflight-five-v1", "created_utc": timestamp(),
                    "model": MODEL, "model_digest": backend["installed_model"]["digest"], "backend": backend,
                    "initial_prompt": INITIAL_PROMPT, "recheck_prompts": RECHECK_PROMPTS,
                    "generation": {"initial_tokens": OUTPUT_BUDGET, "recheck_tokens": OUTPUT_BUDGET,
                                   "context": CONTEXT, "thinking": True, "temperature": 0.6, "top_p": 0.95,
                                   "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
                                   "initial_seed": INITIAL_SEED, "all_recheck_seeds": RECHECK_SEED,
                                   "stop": ["<|im_start|>", "<|im_end|>"], "stream": False},
                    "scope": {"questions": QUESTION_COUNT, "maximum_generation_calls": MAX_CALLS,
                              "automatic_retries": 0, "finalizer_calls": 0, "call_timeout_seconds": 600},
                    "context_guard": "UTF8 content byte bound +128/message +128 prefix; reserve full 2048 output",
                    "history": "Exact initial message.content only; no thinking field or previous recheck output",
                    "output_format": "Visible FINAL: followed by exactly one A-E letter; optional parentheses/punctuation",
                    "parser": "rechecklab.study_protocol.parse_study_answer(content, choice), explicit-final-v1",
                    "grader": "rechecklab.study_protocol.grade_answer, exact normalized letter, offline trusted keys",
                    "recheck_order": "A then B then C; independent full requests sharing initial visible content",
                    "script_sha256": digest(__file__), "historical_comparison": "Independent of v0.2 public replay and frozen v0.3 studies"}
        write_json(run_dir / "protocol.json", protocol)
        protocol_hash = digest(run_dir / "protocol.json")
        (run_dir / "protocol_hash.txt").write_text(protocol_hash + "\n", encoding="utf-8")
        write_json(run_dir / "hardware_before.json", hardware_snapshot())
        for position, question in enumerate(questions, 1):
            visible_initial = None
            for configuration in ("initial", "A", "B", "C"):
                question_id = question["source_question_id"]
                messages = make_messages(question["query"], configuration, visible_initial)
                if configuration != "initial" and messages[1] != {"role": "assistant", "content": visible_initial}:
                    raise AssertionError("Recheck history differs from the saved visible initial response.")
                guard = context_guard(messages)
                verify_backend(backend)
                if digest(run_dir / "protocol.json") != protocol_hash or calls_started >= MAX_CALLS:
                    raise RuntimeError("Frozen protocol changed or call ceiling reached.")
                call_id = f"{position:02d}_{configuration}"
                seed = INITIAL_SEED if configuration == "initial" else RECHECK_SEED
                request_record = {"call_id": call_id, "question_id": question_id, "configuration": configuration,
                                  "seed": seed, "messages": messages, "context_check": guard}
                append_record(run_dir / "requests.jsonl", request_record)
                append_record(run_dir / "events.jsonl", {"event": "started", "call_id": call_id, "timestamp_utc": timestamp()})
                calls_started += 1
                print(f"[{calls_started}/20] question {position}/5 {configuration}: starting; context upper {guard['total_token_upper_bound']}/4096", flush=True)
                raw, wall, error = None, None, None
                record = {"call_id": call_id, "question_id": question_id, "answer_key": keys[question_id],
                          "configuration": configuration, "recheck_prompt": RECHECK_PROMPTS.get(configuration),
                          "seed": seed, "timestamp_utc": timestamp(), "context_check": guard,
                          "valid_final": False, "correct": False, "parsed_answer": None,
                          "visible_response": None, "raw_response": None, "completion_reason": None,
                          "prompt_tokens": None, "generated_tokens": None, "total_tokens": None,
                          "wall_seconds": None, "error": None}
                try:
                    raw, wall = invoke(messages, seed)
                    record.update(raw_response=raw, wall_seconds=wall)
                    visible = validate_response(raw, guard)
                    parsed = parse_study_answer(visible, "choice")
                    record.update(visible_response=visible, parsed_answer=parsed, valid_final=parsed is not None,
                                  correct=grade_answer(parsed, keys[question_id], "choice"),
                                  completion_reason=raw["done_reason"], prompt_tokens=raw["prompt_eval_count"],
                                  generated_tokens=raw["eval_count"], total_tokens=raw["prompt_eval_count"] + raw["eval_count"],
                                  thinking_chars=len(raw["message"].get("thinking", "")))
                    loaded = metadata("/api/ps")
                    active = [model for model in loaded.get("models", []) if model.get("name") == MODEL]
                    if len(active) != 1 or active[0].get("digest") != protocol["model_digest"]:
                        raise RuntimeError("Loaded solver identity mismatch or runner disappeared.")
                    if active[0].get("context_length") != CONTEXT:
                        raise RuntimeError("Actual runner context differs from the requested 4096.")
                    record["loaded_model_after"] = loaded
                    record["hardware_after"] = hardware_snapshot()
                    if configuration == "initial":
                        visible_initial = visible
                except (Exception, KeyboardInterrupt) as exc:
                    error = {"type": type(exc).__name__, "message": str(exc),
                             "backend_body": getattr(exc, "backend_body", None),
                             "request_outcome_ambiguous": raw is None}
                    record.update(error=error, raw_response=raw,
                                  wall_seconds=wall if wall is not None else getattr(exc, "wall_seconds", None))
                records.append(record)
                append_record(run_dir / "calls.jsonl", record)
                append_record(run_dir / "events.jsonl", {"event": "failed" if error else "completed", "call_id": call_id,
                                                        "timestamp_utc": timestamp(), "error": error})
                if error:
                    raise RuntimeError(f"Stopped after {call_id}: {error['type']}: {error['message']}")
                print(f"[{calls_started}/20] completed: FINAL valid={record['valid_final']}, stop={record['completion_reason']}, "
                      f"tokens={record['total_tokens']}, seconds={record['wall_seconds']:.2f}", flush=True)
        verify_backend(backend)
    except (Exception, KeyboardInterrupt) as exc:
        problem = f"{type(exc).__name__}: {exc}"
        print("STOPPED: " + problem, flush=True)
    finally:
        summary = summarize(run_dir, records, problem, started)
        after = protect_originals()
        changed = [name for name in before if after.get(name) != before[name]]
        added = sorted(set(after) - set(before))
        write_json(run_dir / "integrity.json", {"original_files_checked": len(before),
                   "changed_or_removed_files": changed, "new_files_outside_experiment_directory": added,
                   "all_original_files_unchanged": not changed and not added})
        print(json.dumps({"status": summary["status"], "calls": len(records), "problem": problem,
                          "original_files_unchanged": not changed and not added, "outputs": str(run_dir)}), flush=True)
        if changed or added or problem:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
