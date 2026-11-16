"""Inspect Ollama or log live functionality checks using supplied questions.

Question JSONL contains example_id/query and optional answer_type/dataset only.
Optional trusted keys live in a separate JSONL file and are scored after calls.
These checks demonstrate execution; they do not establish predictive accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
import json
import math
from pathlib import Path
import re
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.data import question_hash, transition_label
from rechecklab.predictor import ArtifactLoadError

QUESTION_FIELDS = {"example_id", "query", "answer_type", "dataset"}
KEY_FIELDS = {"example_id", "gold_answer", "answer_type"}
ANSWER_TYPES = {"open", "number", "boolean", "choice"}


def read_jsonl(path):
    records = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {number} in {path}.") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Each JSONL line in {path} must be an object.")
        records.append(row)
    if not records:
        raise ValueError(f"No supplied records were found in {path}.")
    return records


def load_questions(path):
    questions, identifiers = [], set()
    for row in read_jsonl(path):
        extra = set(row) - QUESTION_FIELDS
        if extra:
            raise ValueError("Question-only input contains disallowed fields: " + ", ".join(sorted(extra)))
        if "example_id" not in row or not str(row["example_id"]).strip():
            raise ValueError("Every supplied question needs an explicit example_id.")
        identifier = str(row["example_id"])
        if identifier in identifiers:
            raise ValueError("Question input contains duplicate example_id values.")
        identifiers.add(identifier)
        query = row.get("query")
        question_hash(query)  # Strict nonempty text validation; no inference.
        kind = row.get("answer_type", "open")
        if kind not in ANSWER_TYPES:
            raise ValueError("Question answer_type must be open, number, boolean, or choice.")
        questions.append(dict(example_id=identifier, query=query, answer_type=kind,
                              dataset=str(row.get("dataset", "user_supplied_functionality_check"))))
    return questions


def normalize_trusted_answer(value, kind):
    if value is None or isinstance(value, (dict, list)):
        return None
    text = str(value).strip()
    if kind == "choice":
        text = text.upper()
        return text if text in tuple("ABCDE") else None
    if kind == "boolean":
        return "yes" if text.lower() in ("yes", "true") else "no" if text.lower() in ("no", "false") else None
    if kind == "number":
        if isinstance(value, bool) or not text:
            return None
        # The supplied live questions use rupees. Strip only an explicit
        # currency prefix; never search reasoning text for a likely number.
        text = re.sub(r"^(?:₹|Rs\.?)\s*", "", text, flags=re.I)
        if "," in text and not re.fullmatch(r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
            return None
        try:
            number = Decimal(text.replace(",", ""))
            if not number.is_finite():
                return None
            # Decimal.normalize applies the active precision. Keep every
            # supplied digit so distinct long answers cannot round together.
            with localcontext() as context:
                context.prec = max(len(number.as_tuple().digits), 28)
                return str(number.normalize())
        except DecimalException:
            return None
    return None


def load_trusted_keys(path, questions):
    if path is None:
        return {}
    keys = {}
    for row in read_jsonl(path):
        if set(row) - KEY_FIELDS or not {"example_id", "gold_answer", "answer_type"}.issubset(row):
            raise ValueError("Trusted-key rows must contain example_id, gold_answer, and answer_type only.")
        identifier = str(row["example_id"])
        if identifier in keys:
            raise ValueError("Trusted-key input contains duplicate example_id values.")
        kind = row["answer_type"]
        normalized = normalize_trusted_answer(row["gold_answer"], kind)
        if normalized is None:
            raise ValueError(f"Trusted key {identifier} needs a strict number, yes/no, or A-E answer.")
        keys[identifier] = dict(answer_type=kind, normalized_answer=normalized)
    if set(keys) != {row["example_id"] for row in questions}:
        raise ValueError("Trusted-key example_id values must match the supplied question file exactly.")
    for question in questions:
        if question["answer_type"] != "open" and question["answer_type"] != keys[question["example_id"]]["answer_type"]:
            raise ValueError("Trusted-key and question answer_type values disagree.")
    return keys


def score_after_execution(result, trusted_key):
    """Labels describe actual correctness transitions only after a real recheck."""
    unknown = dict(correctness_known=False, initial_correct=None, revised_correct=None,
                   observed_outcome=None, reason="No separate trusted answer key supplied.")
    if trusted_key is None:
        return unknown
    initial = result.get("initial")
    if not initial or result.get("status") in ("initial_failed", "blocked", "input_error"):
        return {**unknown, "reason": "No completed initial call can be scored."}
    kind, expected = trusted_key["answer_type"], trusted_key["normalized_answer"]
    before = normalize_trusted_answer(initial.get("answer"), kind) == expected
    scored = dict(correctness_known=True, initial_correct=bool(before), revised_correct=None,
                  observed_outcome=None, reason="No completed recheck; its outcome remains unobserved.")
    revised = result.get("revised")
    if result.get("status") == "complete" and result.get("rechecked") and revised:
        after = normalize_trusted_answer(revised.get("answer"), kind) == expected
        scored.update(revised_correct=bool(after), observed_outcome=transition_label(before, after),
                      reason="Compared parsed final answers with the separate trusted key after completed calls; missing/invalid parsed answers count incorrect. No semantic grading.")
    return scored


def validated_log_path(path):
    path = Path(path).resolve()
    protected = (ROOT / "data", ROOT / "models", ROOT / "reports", ROOT / "outputs" / "v0.2")
    if any(path == directory.resolve() or directory.resolve() in path.parents for directory in protected):
        raise ValueError("Choose a separate live output path; public data/models/results are preserved.")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ROOT / "outputs" / "v0.2" / "models" / "source_gate.joblib")
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--question-file", type=Path, help="User-supplied question-only JSONL; omit for a status probe")
    parser.add_argument("--key-file", type=Path, help="Optional separate trusted keys; never passed to generation")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "live" / "functionality_checks.jsonl")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--allow-experimental", action="store_true",
                        help="Permit explicitly experimental same-model use of an incompletely released public protocol")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or not 1 <= args.timeout <= 120:
        parser.error("Timeout must be between 1 and 120 seconds.")
    if args.key_file and not args.question_file:
        parser.error("--key-file requires a supplied --question-file.")
    try:
        from rechecklab.live import load_live_artifact, probe_backend, run_live_question

        bundle = load_live_artifact(args.artifact)
        backend = probe_backend(bundle, host=args.host, timeout=min(args.timeout, 8))
        if args.question_file is None:
            print(json.dumps(backend, indent=2, allow_nan=False))
            return 0 if backend.get("ready") else 1
        questions = load_questions(args.question_file)
        keys = load_trusted_keys(args.key_file, questions)
        output = validated_log_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        run_id = uuid.uuid4().hex
        failed = 0
        for index, question in enumerate(questions, 1):
            result = run_live_question(question["query"], bundle, host=args.host, timeout=args.timeout,
                allow_experimental=args.allow_experimental, answer_type=question["answer_type"])
            record = dict(run_id=run_id, attempt_id=uuid.uuid4().hex,
                timestamp_utc=datetime.now(timezone.utc).isoformat(), example_id=question["example_id"],
                group_id=question_hash(question["query"]), question=question,
                result=result, trusted_evaluation=score_after_execution(result, keys.get(question["example_id"])),
                interpretation="Live functionality check, not an accuracy or calibration experiment; every attempt retained.")
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            print(json.dumps(dict(example_id=question["example_id"], status=result.get("status"),
                                  decision=result.get("decision"), error=result.get("error"))))
            failed += result.get("status") != "complete"
        print(f"Logged {len(questions)} supplied-question attempts to {output}.")
        return 1 if failed else 0
    except ArtifactLoadError as exc:
        print(f"Saved predictor loading failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"Live functionality check stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
