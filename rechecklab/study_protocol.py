"""Frozen local-study protocol shared by offline collection and live inference.

Keys and correctness labels cannot be passed into this module's call API.
Protocol changes require a new version and a separate output directory.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
from pathlib import Path
import re
from urllib.request import Request, urlopen

from .collector import actual_tokens, ollama_chat, response_text
from .features import validate_initial_record

PROTOCOL_VERSION = "live-study-v0.3.2"
PARSER_VERSION = "explicit-final-v1"
GRADER_VERSION = "decimal-exact-choice-v1"
FROZEN_GENERATION = {"initial_token_budget": 1024, "recheck_token_budget": 1024,
                     "context_window": 4096, "thinking": True, "temperature": 0.0,
                     "initial_seed": 42, "recheck_seed": 43, "top_k": 1, "top_p": 1.0,
                     "min_p": 0.0, "repeat_penalty": 1.0,
                     "stop": ["<|im_start|>", "<|im_end|>"], "stream": False}
INITIAL_PROMPT = ("Solve the question independently. Give a brief explanation. "
                  "Finish with FINAL: followed by only the numeric answer or the option letter, as appropriate.")
RECHECK_PROMPT = ("Recheck the saved initial response against the same original question. "
                  "Keep its answer if justified; change it only if you identify an error. "
                  "Give a brief explanation. Finish with FINAL: followed by only the numeric answer or option letter.")


def protocol_hash(protocol):
    raw = json.dumps(protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _get(host, route, payload=None, timeout=15):
    request = Request(host.rstrip("/") + route, data=None if payload is None else json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def inspect_backend(host="http://localhost:11434", model="qwen3:4b"):
    version = _get(host, "/api/version")
    tags = _get(host, "/api/tags")
    installed = [m for m in tags.get("models", []) if m.get("name") == model]
    if len(installed) != 1 or not installed[0].get("digest"):
        raise ValueError(f"Installed {model} checkpoint and digest are required.")
    show = _get(host, "/api/show", {"model": model})
    return {"ollama_version": version.get("version", "unknown"), "installed_model": installed[0],
            "details": show.get("details", {}), "model_info": show.get("model_info", {}),
            "supported_thinking": show.get("thinking", "unknown"),
            "template": show.get("template", "unknown"), "parameters": show.get("parameters", "unknown"),
            "capabilities": show.get("capabilities", "unknown")}


def create_protocol(host="http://localhost:11434", model="qwen3:4b"):
    backend = inspect_backend(host, model)
    protocol = {
        "protocol_version": PROTOCOL_VERSION, "model": model,
        "model_digest": backend["installed_model"]["digest"], "backend": backend,
        "prompts": {"initial": INITIAL_PROMPT, "recheck": RECHECK_PROMPT,
                    "initial_messages": "one user: initial prompt + two newlines + original question",
                    "recheck_messages": "same initial user; assistant: full saved initial response; user: recheck prompt"},
        "generation": {"initial_token_budget": 1024, "recheck_token_budget": 1024,
                       "context_window": 4096, "thinking": True, "temperature": 0.0,
                       "initial_seed": 42, "recheck_seed": 43, "top_k": 1, "top_p": 1.0,
                       "min_p": 0.0, "repeat_penalty": 1.0,
                       "stop": ["<|im_start|>", "<|im_end|>"], "stream": False},
        "finalizer": {"enabled": False, "calls": 0},
        "parser": {"version": PARSER_VERSION, "numeric": "explicit FINAL marker; finite decimal/fraction; comma/currency normalization",
                   "choice": "explicit FINAL marker; single A-E option letter; optional parentheses/punctuation",
                   "open": "explicit FINAL marker only", "unparseable": "record parsing failure; completed call scored incorrect offline"},
        "grader": {"version": GRADER_VERSION, "numeric": "exact Decimal equality; no floating tolerance",
                   "choice": "exact normalized letter equality", "keys": "separate offline file; never in call or feature inputs"},
        "transport": {"timeout_seconds": 120, "automatic_retries": 0,
                      "explicit_retry_limit": 2, "ambiguous_started_calls": "block resume until operator resolves; never silently repeat",
                      "session_budget": "do not start calls after deadline; bound each call by remaining time"},
        "sampling": "one initial call and one separate recheck per question; offline recheck EVERY question",
        "runtime_treatment": "no warm-up call; first-call cold load included; backend caching uncontrolled and actual durations logged",
        "historical_comparison": {"sevra_checkpoint_digest": "unknown", "sevra_prompts": "unknown",
                                  "sevra_context": "unknown", "sevra_initial_budget": 4096,
                                  "sevra_finalizer_max_tokens": 512,
                                  "difference": "new pinned Thinking-2507 local checkpoint, explicit prompts/decoding, 1024-token calls, no finalizer; independent study"},
        "references": ["https://github.com/ollama/ollama/blob/main/docs/api.md",
                       "https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507"],
    }
    return validate_protocol(protocol)


def validate_protocol(protocol):
    if not isinstance(protocol, dict) or protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Unsupported live-study protocol version; use its versioned implementation.")
    for key in ("model", "model_digest"):
        if not isinstance(protocol.get(key), str) or not protocol[key].strip():
            raise ValueError(f"Frozen protocol requires actual {key}.")
    if protocol.get("prompts", {}).get("initial") != INITIAL_PROMPT or protocol.get("prompts", {}).get("recheck") != RECHECK_PROMPT:
        raise ValueError("Prompt changed without a new protocol implementation/version.")
    if protocol.get("parser", {}).get("version") != PARSER_VERSION or protocol.get("grader", {}).get("version") != GRADER_VERSION:
        raise ValueError("Parser/grader changed without a new protocol version.")
    generation = protocol["generation"]
    if generation != FROZEN_GENERATION:
        raise ValueError("Generation settings changed without a new protocol implementation/version.")
    for key in ("initial_token_budget", "recheck_token_budget", "context_window"):
        value = generation.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"Explicit positive integer {key} is required.")
    for key in ("initial_seed", "recheck_seed"):
        if isinstance(generation.get(key), bool) or not isinstance(generation.get(key), int):
            raise ValueError(f"Explicit integer {key} is required.")
    if not isinstance(generation.get("thinking"), bool):
        raise ValueError("Explicit thinking setting is required.")
    for key in ("temperature", "top_p", "min_p", "repeat_penalty"):
        value = generation.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid explicit decoding setting {key}.")
    if not isinstance(generation.get("top_k"), int) or generation["top_k"] < 1 or generation.get("stream") is not False:
        raise ValueError("Explicit top_k and non-streaming calls are required.")
    if not isinstance(generation.get("stop"), list) or not all(isinstance(s, str) and s for s in generation["stop"]):
        raise ValueError("Explicit stop sequences are required.")
    if protocol.get("finalizer") != {"enabled": False, "calls": 0}:
        raise ValueError("This protocol executes no finalizer.")
    transport = protocol["transport"]
    if transport.get("automatic_retries") != 0 or transport.get("explicit_retry_limit") != 2:
        raise ValueError("This protocol uses no automatic inference retries; explicit retries are bounded.")
    if not isinstance(transport.get("timeout_seconds"), (int, float)) or not 1 <= transport["timeout_seconds"] <= 120:
        raise ValueError("Frozen call timeout must be between 1 and 120 seconds.")
    protocol_hash(protocol)  # also rejects nonfinite or non-JSON metadata
    return protocol


def load_protocol(root):
    root = Path(root)
    protocol = validate_protocol(json.loads((root / "protocol.json").read_text(encoding="utf-8")))
    expected = (root / "protocol_hash.txt").read_text(encoding="utf-8").strip()
    if protocol_hash(protocol) != expected:
        raise ValueError("Frozen protocol hash mismatch; refuse collection/inference.")
    return protocol, expected


def freeze_protocol(root, protocol):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate_protocol(protocol)
    path = root / "protocol.json"
    if path.exists():
        existing, digest = load_protocol(root)
        if existing != protocol:
            raise ValueError("Frozen protocol differs; use a new versioned study directory.")
        return digest
    if any((root / name).exists() for name in ("calls.jsonl", "pairs.jsonl")):
        raise ValueError("Cannot freeze over existing calls without a protocol.")
    path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = protocol_hash(protocol)
    (root / "protocol_hash.txt").write_text(digest + "\n", encoding="utf-8")
    return digest


def verify_backend(protocol, host="http://localhost:11434"):
    actual = inspect_backend(host, protocol["model"])
    if actual["installed_model"]["digest"] != protocol["model_digest"]:
        raise ValueError("Installed checkpoint digest differs from frozen protocol.")
    frozen = protocol["backend"]
    for key in ("ollama_version", "template", "parameters", "supported_thinking"):
        if actual[key] != frozen[key]:
            raise ValueError(f"Backend {key} changed; create a new protocol version.")
    return actual


def parse_study_answer(content, answer_type):
    marked = re.findall(r"\bFINAL(?:\s+ANSWER)?\s*:\s*([^\n]+)", content, re.I)
    if not marked:
        return None
    candidate = marked[-1].strip().strip("*` ")
    if answer_type == "open":
        return candidate or None
    if answer_type == "choice":
        match = re.fullmatch(r"\(?([A-E])\)?[.!]?", candidate, re.I)
        return match[1].upper() if match else None
    if answer_type != "number":
        raise ValueError("Study answer_type must be number, choice or open.")
    return normalize_number(candidate)


def normalize_number(candidate):
    candidate = str(candidate).strip().strip("*` ")
    candidate = re.sub(r"^[$£₹€]\s*", "", candidate).rstrip(".")
    numeric = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
    # Thousands separators must be groups of three; do not turn 1,5 into 15.
    if "," in candidate:
        if not re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", candidate):
            return None
        candidate = candidate.replace(",", "")
    try:
        if "/" in candidate:
            parts = candidate.split("/")
            if len(parts) != 2 or not all(re.fullmatch(numeric, p.strip()) for p in parts):
                return None
            denominator = Decimal(parts[1].strip())
            if denominator == 0:
                return None
            value = Decimal(parts[0].strip()) / denominator
        else:
            if not re.fullmatch(numeric, candidate):
                return None
            value = Decimal(candidate)
        return str(value.normalize()) if value.is_finite() else None
    except (InvalidOperation, ArithmeticError):
        return None


def grade_answer(answer, key, answer_type):
    # This function is offline only, outside the generation/feature interface.
    gold = normalize_number(key) if answer_type == "number" else str(key).strip().upper()
    if gold is None or (answer_type == "choice" and (len(gold) != 1 or gold not in "ABCDE")):
        raise ValueError("Invalid trusted benchmark key.")
    return answer is not None and answer == gold


def call_messages(protocol, question, phase, initial_response=None):
    if phase not in ("initial", "recheck") or not isinstance(question, str) or not question.strip():
        raise ValueError("A question and initial/recheck phase are required.")
    messages = [{"role": "user", "content": protocol["prompts"]["initial"] + "\n\n" + question}]
    if phase == "recheck":
        if not isinstance(initial_response, str):
            raise ValueError("Recheck requires the exact saved initial response.")
        messages += [{"role": "assistant", "content": initial_response},
                     {"role": "user", "content": protocol["prompts"]["recheck"]}]
    return messages


def normalize_call(protocol, question, answer_type, response):
    if response.get("model") != protocol["model"] or response.get("done") is not True:
        raise ValueError("Incomplete response or actual solver mismatch.")
    message = response.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Backend did not provide response content.")
    if "thinking" in message and not isinstance(message["thinking"], str):
        raise ValueError("Invalid actual thinking text.")
    if response.get("done_reason") not in ("stop", "length"):
        raise ValueError("Backend did not provide actual stop/length completion status.")
    tokens = actual_tokens(response)
    wall = response.get("wall_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0:
        raise ValueError("Measured call duration is missing or invalid.")
    parsed = parse_study_answer(message["content"], answer_type)
    text = response_text(response)
    record = validate_initial_record({"query": question, "base_response": text,
        "base_answer": parsed or "", "base_actual_tokens": tokens,
        "base_done_reason": response["done_reason"], "base_finalizer_used": False})
    output = {"response": text, "answer": parsed or "", "tokens": tokens,
              "prompt_eval_count": int(response["prompt_eval_count"]), "eval_count": int(response["eval_count"]),
              "done_reason": response["done_reason"], "wall_seconds": wall,
              "finalizer_used": False, "parse_valid": parsed is not None,
              "thinking_chars": len(message.get("thinking", "")), "model": protocol["model"]}
    return record, output


def execute_call(protocol, question, answer_type, phase, initial_response=None, *, host="http://localhost:11434", timeout=None):
    validate_protocol(protocol)
    generation = protocol["generation"]
    limit = protocol["transport"]["timeout_seconds"]
    if timeout is not None:
        limit = min(limit, timeout)
    response = ollama_chat(protocol["model"], call_messages(protocol, question, phase, initial_response), host,
                           generation[phase + "_token_budget"], generation[phase + "_seed"],
                           generation["temperature"], generation["thinking"], timeout=limit,
                           context_window=generation["context_window"],
                           decoding_options={k: generation[k] for k in ("top_k", "top_p", "min_p", "repeat_penalty", "stop")})
    try:
        record, output = normalize_call(protocol, question, answer_type, response)
    except (TypeError, ValueError, KeyError) as exc:
        exc.response = response
        raise
    return response, record, output
