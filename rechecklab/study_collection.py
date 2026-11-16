"""Question-grouped, deadline-bounded collection with a durable per-call journal."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

from .data import question_hash, transition_label
from .study_protocol import execute_call, grade_answer, load_protocol, normalize_call, protocol_hash, validate_protocol, verify_backend


def now():
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    records = []
    # JSONL uses literal LF. str.splitlines() also splits Unicode NEL/LS/PS
    # characters embedded in real benchmark question strings.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"Invalid durable journal {path}:{line_number}; repair explicitly before resuming.") from exc
    return records


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def audit_safe(value):
    """Retain rejected nonfinite backend metadata as explicit failure evidence."""
    if isinstance(value, dict):
        return {key: audit_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [audit_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return {"invalid_nonfinite_float": repr(value)}
    return value


def validate_manifest(questions, keys):
    if not questions or len({q["example_id"] for q in questions}) != len(questions):
        raise ValueError("Unique preselected question IDs are required.")
    if len({q["group_id"] for q in questions}) != len(questions):
        raise ValueError("Duplicate/overlapping normalized question groups in manifest.")
    keymap = {k["example_id"]: k for k in keys}
    if len(keymap) != len(keys) or set(keymap) != {q["example_id"] for q in questions}:
        raise ValueError("Separate trusted keys must exactly match the frozen questions.")
    for q in questions:
        if any(k in q for k in ("gold_answer", "answerKey", "label", "base_correct", "final_correct")):
            raise ValueError("Question manifest must not contain answer keys or outcomes.")
        if question_hash(q["query"]) != q["group_id"]:
            raise ValueError("Manifest group ID differs from normalized question.")
        if q["partition"] not in ("source_train", "source_development", "source_test", "target_adaptation_pool", "target_test"):
            raise ValueError("Unknown predeclared question partition.")
        grade_answer(None, keymap[q["example_id"]]["gold_answer"], q["answer_type"])
    return keymap


def journal_state(events, questions, digest, protocol=None):
    identities = {q["example_id"]: q for q in questions}
    started, terminals, successes, failures = {}, {}, {}, []
    for event in events:
        if event.get("protocol_hash") != digest:
            raise ValueError("Journal mixes protocols; collection cannot resume.")
        example = event.get("example_id")
        if example not in identities or event.get("group_id") != identities[example]["group_id"]:
            raise ValueError("Journal call does not match frozen manifest.")
        if event.get("phase") not in ("initial", "recheck"):
            raise ValueError("Unknown call phase in journal.")
        cid = event["call_id"]
        if event["status"] == "started":
            if cid in started:
                raise ValueError("Duplicate started call journal event.")
            started[cid] = event
        elif event["status"] in ("succeeded", "failed", "resolved_uncertain", "cancelled_no_request"):
            if cid not in started or cid in terminals:
                raise ValueError("Unmatched or duplicate terminal call journal event.")
            origin = started[cid]
            for key in ("example_id", "group_id", "phase", "seed", "attempt", "session_id", "protocol_hash"):
                if event.get(key) != origin.get(key):
                    raise ValueError(f"Terminal call changed its started {key} identity.")
            terminals[cid] = event
            if event["status"] == "succeeded":
                if protocol is not None:
                    question = identities[example]
                    _, actual = normalize_call(protocol, question["query"], question["answer_type"], event["response"])
                    if event.get("output") != actual or event.get("seed") != protocol["generation"][event["phase"] + "_seed"]:
                        raise ValueError("Saved successful call output/seed disagrees with its actual response or frozen protocol.")
                key = (example, event["phase"])
                if key in successes:
                    raise ValueError("Multiple successful calls for one question/phase.")
                successes[key] = event
            elif event["status"] != "cancelled_no_request":
                failures.append(event)
        else:
            raise ValueError("Unknown journal status.")
    pending = [event for cid, event in started.items() if cid not in terminals]
    return successes, failures, pending


def build_pair(question, key, initial, recheck, protocol, digest):
    base, final = initial["output"], recheck["output"]
    base_correct = grade_answer(base["answer"] or None, key["gold_answer"], question["answer_type"])
    final_correct = grade_answer(final["answer"] or None, key["gold_answer"], question["answer_type"])
    return {**question, "model": protocol["model"], "model_digest": protocol["model_digest"],
            "protocol_hash": digest, "recheck_protocol": "live-study:" + digest, "action": "recheck", "rollout": 0,
            "seed": protocol["generation"]["initial_seed"], "recheck_seed": protocol["generation"]["recheck_seed"],
            "base_call_id": initial["call_id"], "action_call_id": recheck["call_id"],
            "base_response": base["response"], "intervention_response": final["response"],
            "base_answer": base["answer"], "final_answer": final["answer"],
            "base_correct": base_correct, "final_correct": final_correct,
            "label": transition_label(base_correct, final_correct), "answer_text_changed": base["answer"] != final["answer"],
            "base_actual_tokens": base["tokens"], "action_actual_tokens": final["tokens"],
            "base_prompt_eval_count": base["prompt_eval_count"], "base_eval_count": base["eval_count"],
            "action_prompt_eval_count": final["prompt_eval_count"], "action_eval_count": final["eval_count"],
            "base_done_reason": base["done_reason"], "intervention_done_reason": final["done_reason"],
            "base_finalizer_used": False, "action_finalizer_used": False,
            "base_parse_valid": base["parse_valid"], "final_parse_valid": final["parse_valid"],
            "base_wall_seconds": base["wall_seconds"], "action_wall_seconds": final["wall_seconds"],
            "provenance": "Genuine new local Ollama initial response and unconditional offline recheck; key scored offline",
            "timestamp_utc": now()}


def collection_order(questions, smoke=0):
    # IDs are already outcome-independent seeded samples. Interleave domains and
    # partitions to avoid claiming that a time-limited prefix is a full test.
    partitions = (["source_train", "target_adaptation_pool"] if smoke else
                  ["source_train", "source_development", "target_adaptation_pool", "source_test", "target_test"])
    groups = {p: [q for q in questions if q["partition"] == p] for p in partitions}
    ordered = []
    for index in range(max(len(group) for group in groups.values())):
        ordered.extend(groups[p][index] for p in partitions if index < len(groups[p]))
    return ordered[:smoke] if smoke else ordered


def collect_study(root, *, host="http://localhost:11434", minutes=60, smoke=0, retry_failures=False,
                  resolve_uncertain=False, on_progress=None):
    root = Path(root).resolve()
    if isinstance(minutes, bool) or not isinstance(minutes, (int, float)) or not 0 < minutes <= 1440:
        raise ValueError("Collection minutes must be positive and at most 1440.")
    if isinstance(smoke, bool) or not isinstance(smoke, int) or smoke < 0:
        raise ValueError("Smoke count must be a nonnegative integer.")
    questions = read_jsonl(root / "questions.jsonl")
    keymap = validate_manifest(questions, read_jsonl(root / "trusted_keys.jsonl"))
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    expected_hashes = plan.get("manifest_sha256", {})
    for name in ("questions.jsonl", "trusted_keys.jsonl"):
        if expected_hashes.get(name) != hashlib.sha256((root / name).read_bytes()).hexdigest():
            raise ValueError(f"Frozen study manifest hash mismatch: {name}; never silently change preselected questions or keys.")
    if smoke:
        protocol = validate_protocol(json.loads((root / "candidate_protocol.json").read_text(encoding="utf-8")))
        digest = protocol_hash(protocol)
        destination = root / "smoke"
        destination.mkdir(exist_ok=True)
        write_json(destination / "protocol.json", protocol)
        (destination / "protocol_hash.txt").write_text(digest + "\n", encoding="utf-8")
    else:
        protocol, digest = load_protocol(root)
        destination = root
    callpath, pairpath = destination / "calls.jsonl", destination / "pairs.jsonl"
    lock = destination / "collection.lock"
    session = uuid.uuid4().hex
    start = time.monotonic()
    started_at = now()
    deadline = start + minutes * 60
    status = {"status": "PARTIAL", "phase": "smoke" if smoke else "collection", "session_id": session,
              "started_at": started_at, "ended_at": None, "minutes_budget": minutes,
              "protocol_hash": digest, "errors": [], "planned_unique_questions": len(questions),
              "command": f"scripts/collect_live_study.py --study-root {root} --minutes {minutes}"}
    acquired = False
    pairs = []

    def checkpoint(reason=None):
        saved = read_jsonl(pairpath)
        journal = read_jsonl(callpath)
        counts = Counter(q["partition"] for q in saved)
        elapsed = time.monotonic() - start
        total_call_time = sum(e.get("output", {}).get("wall_seconds", 0) for e in journal if e["status"] == "succeeded")
        rate = len(saved) / total_call_time if total_call_time > 0 else None
        status.update(elapsed_seconds=elapsed, completed_pairs=len(saved), completed_unique_questions=len({p["group_id"] for p in saved}),
                      completed_by_partition=dict(counts), class_counts=dict(Counter(p["label"] for p in saved)),
                      call_events=len(journal), successful_calls=sum(e["status"] == "succeeded" for e in journal),
                      failed_calls=sum(e["status"] == "failed" for e in journal),
                      parse_failures=sum(not e["output"]["parse_valid"] for e in journal if e["status"] == "succeeded"),
                      remaining_questions=len(questions) - len(saved), measured_pairs_per_hour=rate * 3600 if rate else None,
                      estimated_remaining_hours=(len(questions) - len(saved)) / (rate * 3600) if rate else None,
                      eta_basis="sum of measured successful call wall times; includes first-call cold load; cache/backend overhead not controlled")
        if reason:
            status["stop_reason"] = reason
        write_json(destination / "collection_status.json", status)

    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(descriptor, json.dumps({"pid": os.getpid(), "session": session, "started_at": started_at}).encode())
        os.close(descriptor)
        acquired = True
        verify_backend(protocol, host)
        events = read_jsonl(callpath)
        successes, failures, pending = journal_state(events, questions, digest, protocol)
        if pending and resolve_uncertain:
            for event in pending:
                append_jsonl(callpath, {**event, "status": "resolved_uncertain", "timestamp_utc": now(),
                                       "error": "Operator explicitly acknowledged uncertain completion; no successful response metadata exists."})
            successes, failures, pending = journal_state(read_jsonl(callpath), questions, digest, protocol)
        if pending:
            raise RuntimeError("Uncertain started call(s) cannot be silently repeated: " + ", ".join(e["call_id"] for e in pending))
        pairs = read_jsonl(pairpath)
        if len({p["example_id"] for p in pairs}) != len(pairs):
            raise ValueError("Duplicate completed pairs in saved output.")
        saved_ids = {p["example_id"] for p in pairs}
        for pair in pairs:
            if pair.get("protocol_hash") != digest:
                raise ValueError("Saved pairs do not match frozen protocol.")
            identity = pair["example_id"]
            question = next((q for q in questions if q["example_id"] == identity), None)
            if question is None or not all((identity, phase) in successes for phase in ("initial", "recheck")):
                raise ValueError("Saved pair lacks its immutable question or two actual successful calls.")
            rebuilt = build_pair(question, keymap[identity], successes[(identity, "initial")], successes[(identity, "recheck")], protocol, digest)
            if any(pair.get(key) != value for key, value in rebuilt.items() if key != "timestamp_utc"):
                raise ValueError("Saved pair differs from actual journal responses, metadata or offline trusted-key grading.")
        # Recover a pair when both calls were durably saved but pair append was interrupted.
        for q in questions:
            identity = q["example_id"]
            if identity not in saved_ids and all((identity, phase) in successes for phase in ("initial", "recheck")):
                append_jsonl(pairpath, build_pair(q, keymap[identity], successes[(identity, "initial")], successes[(identity, "recheck")], protocol, digest))
                saved_ids.add(identity)
        checkpoint()
        for q in collection_order(questions, smoke):
            identity = q["example_id"]
            if identity in saved_ids:
                continue
            for phase in ("initial", "recheck"):
                key = (identity, phase)
                if key in successes:
                    continue
                failed = [e for e in failures if (e["example_id"], e["phase"]) == key]
                if failed and not retry_failures:
                    status["status"] = "BLOCKED"
                    raise RuntimeError(f"Prior failed {phase} call for {identity}; explicit --retry-failures required.")
                if len(failed) > protocol["transport"]["explicit_retry_limit"]:
                    raise RuntimeError(f"Explicit retry limit reached for {identity}/{phase}.")
                remaining = deadline - time.monotonic()
                if remaining < protocol["transport"]["timeout_seconds"]:
                    checkpoint("insufficient session budget for one full bounded call; no request started")
                    return status
                event = {"example_id": identity, "group_id": q["group_id"], "partition": q["partition"],
                         "phase": phase, "call_id": uuid.uuid4().hex, "attempt": len(failed) + 1,
                         "protocol_hash": digest, "timestamp_utc": now(), "session_id": session,
                         "seed": protocol["generation"][phase + "_seed"], "status": "started", "smoke": bool(smoke)}
                append_jsonl(callpath, event)
                checkpoint()
                if on_progress:
                    on_progress(f"{len(saved_ids)}/{len(questions)} pairs | {identity} | {phase}")
                # A slow checkpoint/callback can consume time after the first
                # deadline check. Close the logical journal start without a
                # request rather than leave a fictitious uncertain call.
                remaining = deadline - time.monotonic()
                if remaining < protocol["transport"]["timeout_seconds"]:
                    append_jsonl(callpath, {**event, "status": "cancelled_no_request", "timestamp_utc": now(),
                                           "reason": "Session budget expired during checkpoint/progress; HTTP request never started."})
                    checkpoint("insufficient remaining session budget; request never started")
                    return status
                previous = successes[(identity, "initial")]["output"]["response"] if phase == "recheck" else None
                call_started = time.monotonic()
                try:
                    response, _, output = execute_call(protocol, q["query"], q["answer_type"], phase, previous,
                                                       host=host, timeout=min(remaining, protocol["transport"]["timeout_seconds"]))
                    completed = {**event, "status": "succeeded", "timestamp_utc": now(), "response": response, "output": output}
                    append_jsonl(callpath, completed)
                    successes[key] = completed
                    checkpoint()
                except Exception as exc:
                    failed_event = {**event, "status": "failed", "timestamp_utc": now(),
                        "error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc(),
                        "wall_seconds": time.monotonic() - call_started,
                        "failure_kind": "response_metadata" if hasattr(exc, "response") else "transport_runtime",
                        "reported_response": audit_safe(getattr(exc, "response", None))}
                    append_jsonl(callpath, failed_event)
                    status["errors"].append(failed_event)
                    failures.append(failed_event)
                    status["status"] = "PARTIAL" if time.monotonic() >= deadline else "BLOCKED"
                    checkpoint("call failed; successful initial call preserved, no incorrect label manufactured")
                    return status
            pair = build_pair(q, keymap[identity], successes[(identity, "initial")], successes[(identity, "recheck")], protocol, digest)
            append_jsonl(pairpath, pair)
            saved_ids.add(identity)
            checkpoint()
            if on_progress:
                on_progress(f"Saved {identity}: {pair['label']}; actual tokens {pair['base_actual_tokens']} + {pair['action_actual_tokens']}")
        status["status"] = "COMPLETE" if not smoke and len(saved_ids) == len(questions) else "PARTIAL"
        if smoke:
            status["smoke_status"] = "COMPLETE"
        checkpoint("all selected smoke questions complete" if smoke else "all planned questions complete")
    except Exception as exc:
        status["status"] = "BLOCKED"
        status["errors"].append({"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
        if acquired:
            checkpoint("validation/backend/resume blocked")
        else:
            raise
    finally:
        if acquired:
            status["ended_at"] = now()
            try:
                checkpoint()
            except Exception as exc:
                status["status"] = "BLOCKED"
                status["errors"].append({"error": f"Diagnostic checkpoint failed: {exc}",
                                          "error_type": type(exc).__name__, "traceback": traceback.format_exc()})
                write_json(destination / "collection_status.json", status)
            finally:
                lock.unlink(missing_ok=True)
    return status
