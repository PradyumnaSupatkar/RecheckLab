"""The stable pre-recheck allowlist, shared by training, evaluation, and replay.

Every initial signal must be supplied. An explicitly empty parsed initial
answer means parsing failed. An explicitly 'unknown' completion reason is
represented by its own indicator and is used for missing public-log statuses.
Gold labels, revised text, and future token counts are never read here.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
import pandas as pd

UNCERTAIN = re.compile(r"\b(maybe|perhaps|unsure|uncertain|possibly|might|not sure|i think)\b", re.I)
BACKTRACK = re.compile(r"\b(wait|however|instead|mistake|reconsider|actually|wrong)\b", re.I)
CHECK = re.compile(r"\b(check|verify|verification|confirm|recheck)\b", re.I)
WORDS = re.compile(r"\b\w+\b")

INITIAL_REQUIRED = (
    "query", "base_response", "base_answer", "base_actual_tokens",
    "base_done_reason", "base_finalizer_used",
)
FEATURE_SCHEMA_VERSION = "pre-recheck-v2"
FEATURE_NAMES = [
    "question_log_chars", "question_log_words", "question_numbers",
    "question_math_symbols", "question_marks", "initial_log_chars",
    "initial_log_words", "initial_unique_word_ratio",
    "initial_uncertainty_per_100_words", "initial_backtracking_per_100_words",
    "initial_checking_per_100_words", "initial_number_count",
    "initial_paragraph_count", "initial_question_marks",
    "initial_log_actual_tokens", "initial_finalizer_used",
    "initial_budget_exhausted", "initial_answer_present", "initial_completion_unknown",
]
FEATURE_SCHEMA_FINGERPRINT = hashlib.sha256(json.dumps({
    "version": FEATURE_SCHEMA_VERSION, "features": FEATURE_NAMES,
    "required_inputs": INITIAL_REQUIRED, "missing_data": "reject; explicit unknown completion indicator",
    "completion_values": ["stop", "length", "unknown"], "initial_tokens": "finite nonnegative integer",
}, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def validate_initial_record(row):
    """Validate only the inputs available before a recheck and normalize types."""
    missing = [key for key in INITIAL_REQUIRED if key not in row]
    if missing:
        raise ValueError("Missing required initial signals: " + ", ".join(missing))
    out = {}
    for key in ("query", "base_response", "base_answer", "base_done_reason"):
        value = row[key]
        if value is None or not np.isscalar(value) or pd.isna(value):
            raise ValueError(f"{key} must be supplied; use an explicit empty base_answer for parsing failure.")
        if key != "base_answer" and not isinstance(value, str):
            raise ValueError(f"{key} must be text.")
        out[key] = str(value)
    if not out["query"].strip():
        raise ValueError("query cannot be empty.")
    if not out["base_done_reason"].strip():
        raise ValueError("base_done_reason cannot be empty; supply the recorded completion status or explicit 'unknown'.")
    out["base_done_reason"] = out["base_done_reason"].strip().casefold()
    if out["base_done_reason"] not in ("stop", "length", "unknown"):
        raise ValueError("base_done_reason must be stop, length, or explicit unknown.")
    value = row["base_actual_tokens"]
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("base_actual_tokens must be a finite nonnegative number.")
    try:
        tokens = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError("base_actual_tokens must be a finite nonnegative number.") from exc
    if not math.isfinite(tokens) or tokens < 0 or not tokens.is_integer():
        raise ValueError("base_actual_tokens must be a finite nonnegative integer.")
    out["base_actual_tokens"] = tokens
    value = row["base_finalizer_used"]
    if isinstance(value, (bool, np.bool_)):
        finalizer = bool(value)
    elif isinstance(value, (int, float, np.integer, np.floating)) and value in (0, 1):
        finalizer = bool(value)
    elif isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
        finalizer = value.strip().lower() in ("true", "1")
    else:
        raise ValueError("base_finalizer_used must be an explicit boolean or 0/1.")
    out["base_finalizer_used"] = finalizer
    return out


def extract_one(row):
    row = validate_initial_record(row)
    q, a = row["query"], row["base_response"]
    qw, aw = WORDS.findall(q.lower()), WORDS.findall(a.lower())
    n = max(len(aw), 1)
    values = {
        "question_log_chars": math.log1p(len(q)),
        "question_log_words": math.log1p(len(qw)),
        "question_numbers": len(re.findall(r"\d+(?:\.\d+)?", q)),
        "question_math_symbols": sum(q.count(c) for c in "+-=*/^\\"),
        "question_marks": q.count("?"),
        "initial_log_chars": math.log1p(len(a)),
        "initial_log_words": math.log1p(len(aw)),
        "initial_unique_word_ratio": len(set(aw)) / n,
        "initial_uncertainty_per_100_words": 100 * len(UNCERTAIN.findall(a)) / n,
        "initial_backtracking_per_100_words": 100 * len(BACKTRACK.findall(a)) / n,
        "initial_checking_per_100_words": 100 * len(CHECK.findall(a)) / n,
        "initial_number_count": len(re.findall(r"\d+(?:\.\d+)?", a)),
        "initial_paragraph_count": a.count("\n\n") + 1,
        "initial_question_marks": a.count("?"),
        "initial_log_actual_tokens": math.log1p(row["base_actual_tokens"]),
        "initial_finalizer_used": int(row["base_finalizer_used"]),
        "initial_budget_exhausted": int(row["base_done_reason"] == "length"),
        "initial_answer_present": int(bool(row["base_answer"].strip())),
        "initial_completion_unknown": int(row["base_done_reason"] == "unknown"),
    }
    return {key: values[key] for key in FEATURE_NAMES}


def extract_features(frame):
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Initial inputs must be supplied as a pandas DataFrame.")
    missing = [key for key in INITIAL_REQUIRED if key not in frame.columns]
    if missing:
        raise ValueError("Missing required initial signals: " + ", ".join(missing))
    records = []
    for index, row in zip(frame.index, frame.to_dict("records")):
        try:
            records.append(extract_one(row))
        except ValueError as exc:
            raise ValueError(f"Invalid initial inputs at row {index}: {exc}") from exc
    return pd.DataFrame(records, columns=FEATURE_NAMES, index=frame.index, dtype=float)
