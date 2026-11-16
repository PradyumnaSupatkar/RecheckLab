"""Prepare then collect a small paired Qwen3 protocol reliability check.

Run from the repository root with Python 3.12. `--prepare` freezes selection
and protocol before `--collect` permits any inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "outputs/live_study/v0.3.0/raw/gsm8k/740312add88f781978c0658806c59bc2815b9866/train.parquet"
MODEL = "qwen3:4b"
DIGEST = "359d7dd4bcdab3d86b87d73ac27966f4dbb9f5efdfcc75d34a8764a09474fae7"
REVISION = "740312add88f781978c0658806c59bc2815b9866"
INITIAL = "Solve the question independently. Give a brief explanation. Finish with FINAL: followed by only the numeric answer or the option letter, as appropriate."
RECHECK = "Recheck the saved initial response against the same original question. Keep its answer if justified; change it only if you identify an error. Give a brief explanation. Finish with FINAL: followed by only the numeric answer or option letter."
STOP = ["<|im_start|>", "<|im_end|>"]
POLICIES = {
    "A_thinking_off_1024": {"thinking": False, "num_predict": 1024},
    "B_thinking_on_2048": {"thinking": True, "num_predict": 2048},
}
FREE_RAM_FLOOR = 2.0  # GiB; stop before starting another call if below this floor.


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(text: str) -> str:
    import unicodedata
    return sha(" ".join(unicodedata.normalize("NFKC", text).casefold().split()).encode())


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # Some historical append-only logs end with an interrupted partial
            # record; their pinned manifests and every complete row still count.
            continue
    return rows


def selection_exclusions() -> set[str]:
    excluded: set[str] = set()
    known = [
        ROOT / "data/prepared/source_train.parquet", ROOT / "data/prepared/source_development.parquet",
        ROOT / "data/prepared/source_test.parquet", ROOT / "data/prepared/target_adaptation_pool.parquet",
        ROOT / "data/prepared/target_test.parquet",
    ]
    for path in known:
        if path.exists():
            frame = pd.read_parquet(path)
            if "group_id" in frame:
                excluded.update(str(x) for x in frame["group_id"].dropna())
            elif "query" in frame:
                excluded.update(canonical_hash(str(x)) for x in frame["query"])
    for path in (ROOT / "outputs/live_study").glob("v0.3.*"):
        for name in ("questions.jsonl", "pairs.jsonl", "calls.jsonl"):
            for row in read_jsonl(path / name):
                q = row.get("query") or row.get("question")
                if isinstance(q, str) and q.strip():
                    excluded.add(canonical_hash(q))
                group = row.get("group_id")
                if isinstance(group, str):
                    excluded.add(group)
        selection = path / "QUESTION_SELECTION.json"
        if selection.exists():
            doc = read_json(selection)
            for key in ("selected", "selected_questions"):
                for row in doc.get(key, []):
                    if isinstance(row, dict) and row.get("group_id"):
                        excluded.add(str(row["group_id"]))
    return excluded


def prepare():
    if not DATA.is_file():
        raise FileNotFoundError(f"Pinned cached benchmark missing: {DATA.relative_to(ROOT)}")
    frame = pd.read_parquet(DATA)
    if len(frame) != 7473 or list(frame.columns) != ["question", "answer"]:
        raise ValueError("Pinned GSM8K source cache differs from the audited snapshot.")
    excluded = selection_exclusions()
    eligible = [(i, str(row.question).strip()) for i, row in frame.iterrows()
                if isinstance(row.question, str) and row.question.strip()
                and canonical_hash(str(row.question)) not in excluded]
    if len(eligible) < 3:
        raise ValueError("Fewer than three independently eligible questions remain.")
    import numpy as np
    rng = np.random.default_rng(20261118)
    chosen = [eligible[int(k)] for k in rng.permutation(len(eligible))[:3]]
    questions = [{"example_id": f"gsm8k:train:{int(i)}", "source_question_id": str(int(i)),
                  "group_id": canonical_hash(q), "query": q} for i, q in chosen]
    protocol = {
        "study": "v0.3.3 protocol reliability development (not evaluation)",
        "dataset": {"repository": "openai/gsm8k", "revision": REVISION, "split": "main/train",
                    "source_rows": len(frame), "question_selection_seed": 20261118,
                    "selection_rule": "uniform seeded sample from canonical-unique eligible train questions after excluding all local v0.2 prepared cohorts and all saved v0.3 planned/evaluated questions; selection uses question text/hash only, not key, outcome, difficulty, or gate score"},
        "model": {"name": MODEL, "digest": DIGEST, "ollama_version_expected": "0.40.16"},
        "prompts": {"initial": INITIAL, "revision": RECHECK,
                    "initial_messages": "one user: initial prompt + two newlines + question",
                    "revision_messages": "same initial user; assistant: full initial response content; user: revision prompt"},
        "decoding_common": {"context_window": 4096, "temperature": 0.0, "initial_seed": 42, "revision_seed": 43,
                            "top_k": 1, "top_p": 1.0, "min_p": 0.0, "repeat_penalty": 1.0,
                            "stop": STOP, "stream": False, "timeout_seconds": 240, "retries": 0},
        "candidates": POLICIES,
        "parser": "explicit-final-v1: require FINAL marker and finite numeric answer; no extraction from reasoning text",
        "grader": "published GSM8K #### key, decimal exact after comma normalization; score offline only",
        "gate": {"artifact": "outputs/v0.2/models/source_gate.joblib", "threshold": 0.25968145276947807,
                 "score": "p_HELP - p_HURT", "rule": "score > threshold", "role": "OOD transfer diagnostic only; no protocol validation bypass"},
        "resource_guard": {"sequential_calls": True, "minimum_free_ram_before_each_call_gib": FREE_RAM_FLOOR,
                           "stop_on_call_error": True, "no_process_termination": True},
        "call_order": "for each frozen question, policy order A then B, initial then revision; sequential only",
        "outcomes_not_used_for_selection_or_generation": True,
    }
    qbytes = ("\n".join(json.dumps(x, ensure_ascii=False) for x in questions) + "\n").encode()
    key_rows = []
    for q, (idx, _) in zip(questions, chosen):
        answer = str(frame.iloc[idx]["answer"])
        if "####" not in answer:
            raise ValueError(f"Missing published GSM8K key for {q['example_id']}")
        key_rows.append({"example_id": q["example_id"], "gold_answer": answer.rsplit("####", 1)[1].strip()})
    kbytes = ("\n".join(json.dumps(x) for x in key_rows) + "\n").encode()
    (OUT / "QUESTION_SELECTION.jsonl").write_bytes(qbytes)
    (OUT / "TRUSTED_KEYS.jsonl").write_bytes(kbytes)
    protocol["question_manifest_sha256"] = sha(qbytes)
    protocol["trusted_keys_sha256"] = sha(kbytes)
    (OUT / "FROZEN_PROTOCOL.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    status = {"phase": "frozen before inference", "questions": len(questions), "eligible_count": len(eligible),
              "excluded_group_count": len(excluded), "selection_seed": 20261118,
              "question_manifest_sha256": sha(qbytes), "trusted_keys_sha256": sha(kbytes),
              "calls_planned": 12, "inference_started": False}
    (OUT / "PREPARATION_STATUS.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**status, "ids": [q["example_id"] for q in questions]}, indent=2))


def hardware():
    mem = psutil.virtual_memory()
    gpu = "unavailable"
    exe = "nvidia-smi"
    try:
        raw = subprocess.check_output([exe, "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"], text=True, timeout=5).strip()
        gpu = raw
    except Exception as exc:
        gpu = f"unavailable: {type(exc).__name__}"
    return {"free_ram_gib": round(mem.available / 1024**3, 3), "used_ram_percent": mem.percent,
            "gpu": gpu, "timestamp_unix": time.time()}


def invoke(payload: dict):
    request = Request("http://localhost:11434/api/chat", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    with urlopen(request, timeout=240) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    result["client_wall_seconds"] = elapsed
    return result


def parse_final(content: str):
    matches = re.findall(r"FINAL(?:\s+ANSWER)?\s*:\s*([^\n]+)", content, flags=re.I)
    if not matches:
        return None
    value = matches[-1].strip().replace(",", "").rstrip(".")
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
        return None
    from decimal import Decimal, InvalidOperation
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return None


def collect():
    protocol = read_json(OUT / "FROZEN_PROTOCOL.json")
    questions = read_jsonl(OUT / "QUESTION_SELECTION.jsonl")
    keys = {r["example_id"]: r["gold_answer"] for r in read_jsonl(OUT / "TRUSTED_KEYS.jsonl")}
    status_path = OUT / "COLLECTION_STATUS.json"
    raw_path = OUT / "RAW_CALLS.jsonl"
    if raw_path.exists() or status_path.exists():
        raise FileExistsError("Collection output already exists; refusing to overwrite or resume this frozen run.")
    tags = json.load(urlopen("http://localhost:11434/api/tags", timeout=10)).get("models", [])
    found = next((x for x in tags if x.get("name") == MODEL), None)
    if not found:
        raise RuntimeError(f"{MODEL} is not installed locally; no model download is permitted.")
    if found.get("digest") != DIGEST:
        raise RuntimeError("Installed model digest differs from the frozen Qwen3 checkpoint.")
    report = []
    for question in questions:
        for policy, generation in POLICIES.items():
            pair = {"example_id": question["example_id"], "policy": policy, "calls": []}
            previous = None
            for phase in ("initial", "revision"):
                resource = hardware()
                if resource["free_ram_gib"] < FREE_RAM_FLOOR:
                    status = {"status": "STOPPED_RESOURCE_FLOOR", "reason": "free system RAM below 2.0 GiB before next call",
                              "completed_calls": sum(len(p["calls"]) for p in report), "resources": resource,
                              "completed_pairs": report}
                    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(json.dumps(status, indent=2, ensure_ascii=False))
                    return
                prompt = INITIAL if phase == "initial" else RECHECK
                user = prompt + "\n\n" + question["query"]
                messages = [{"role": "user", "content": user}]
                if phase == "revision":
                    messages.extend([{"role": "assistant", "content": previous}, {"role": "user", "content": RECHECK}])
                cfg = protocol["decoding_common"]
                payload = {"model": MODEL, "messages": messages, "stream": False, "think": generation["thinking"],
                           "options": {"num_predict": generation["num_predict"], "num_ctx": cfg["context_window"],
                                       "temperature": cfg["temperature"], "seed": cfg[f"{phase}_seed"],
                                       "top_k": cfg["top_k"], "top_p": cfg["top_p"], "min_p": cfg["min_p"],
                                       "repeat_penalty": cfg["repeat_penalty"], "stop": cfg["stop"]}}
                try:
                    result = invoke(payload)
                    if not result.get("done"):
                        raise RuntimeError("Ollama response did not mark done=true")
                    message = result.get("message", {})
                    content = message.get("content", "") or ""
                    actual = {"prompt_tokens": result.get("prompt_eval_count"), "output_tokens": result.get("eval_count"),
                              "total_tokens": None if result.get("prompt_eval_count") is None or result.get("eval_count") is None
                              else int(result["prompt_eval_count"]) + int(result["eval_count"])}
                    call = {"example_id": question["example_id"], "policy": policy, "phase": phase,
                            "model": MODEL, "model_digest": DIGEST, "ollama_version": "0.40.16",
                            "request": payload, "completion_reason": result.get("done_reason"),
                            "response_message": message, "tokens": actual,
                            "latency_seconds": result["client_wall_seconds"], "server_total_duration_ns": result.get("total_duration"),
                            "server_load_duration_ns": result.get("load_duration"),
                            "server_prompt_eval_duration_ns": result.get("prompt_eval_duration"),
                            "server_eval_duration_ns": result.get("eval_duration"),
                            "resource_before": resource, "resource_after": hardware()}
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(call, ensure_ascii=False) + "\n")
                    pair["calls"].append(call)
                    if phase == "initial":
                        previous = content
                    if call["resource_after"]["free_ram_gib"] < FREE_RAM_FLOOR:
                        status = {"status": "STOPPED_RESOURCE_FLOOR", "reason": "free system RAM fell below 2.0 GiB after call",
                                  "completed_calls": sum(len(p["calls"]) for p in report) + len(pair["calls"]),
                                  "resources": call["resource_after"], "completed_pairs": report + [pair]}
                        status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        print(json.dumps(status, indent=2, ensure_ascii=False))
                        return
                except Exception as exc:
                    failure = {"example_id": question["example_id"], "policy": policy, "phase": phase,
                               "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
                               "resource": hardware(), "request": payload}
                    with raw_path.open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
                    status = {"status": "STOPPED_CALL_FAILURE", "completed_pairs": report,
                              "failed_call": failure, "completed_calls": sum(len(p["calls"]) for p in report)}
                    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                    print(json.dumps(status, indent=2, ensure_ascii=False))
                    return
            init, rev = pair["calls"]
            pair["initial_answer"] = parse_final(init["response_message"].get("content", ""))
            pair["revision_answer"] = parse_final(rev["response_message"].get("content", ""))
            pair["initial_valid"] = pair["initial_answer"] is not None
            pair["revision_valid"] = pair["revision_answer"] is not None
            normalized_key = parse_final("FINAL: " + str(keys[question["example_id"]]))
            pair["initial_correct"] = pair["initial_answer"] == normalized_key if pair["initial_valid"] else False
            pair["revision_correct"] = pair["revision_answer"] == normalized_key if pair["revision_valid"] else False
            report.append(pair)
            (OUT / "COLLECTION_STATUS.json").write_text(json.dumps({"status": "RUNNING", "completed_pairs": report,
                "completed_calls": sum(len(p["calls"]) for p in report), "resources": hardware()}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"{question['example_id']} {policy}: initial={pair['initial_valid']} ({init['completion_reason']}), revision={pair['revision_valid']} ({rev['completion_reason']}); RAM {rev['resource_after']['free_ram_gib']} GiB", flush=True)
    status = {"status": "COMPLETE", "completed_pairs": report, "completed_calls": sum(len(p["calls"]) for p in report),
              "resources": hardware()}
    status_path.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status["status"], "completed_pairs": len(report), "calls": status["completed_calls"], "resources": status["resources"]}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.collect:
        parser.error("choose exactly one of --prepare or --collect")
    prepare() if args.prepare else collect()


if __name__ == "__main__":
    main()
