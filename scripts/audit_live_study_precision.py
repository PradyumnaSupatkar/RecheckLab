"""Audit saved numeric calls without generation, trusted keys or relabeling."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.study_collection import write_json
from rechecklab.study_protocol import load_protocol
from rechecklab.study_validation import AUDIT_VERSION, PRECISION_LIMITATION, audit_numeric_precision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "outputs" / "live_study" / "v0.3.0")
    args = parser.parse_args()
    root = args.study_root.resolve()
    protocol, digest = load_protocol(root)
    question_bytes = (root / "questions.jsonl").read_bytes()
    plan = json.loads((root / "study_plan.json").read_text(encoding="utf-8"))
    if hashlib.sha256(question_bytes).hexdigest() != plan.get("manifest_sha256", {}).get("questions.jsonl"):
        raise ValueError("Question snapshot differs from its frozen manifest hash.")
    questions = {q["example_id"]: q for q in (json.loads(line) for line in
                 question_bytes.decode("utf-8-sig").split("\n") if line.strip())}
    call_bytes = (root / "calls.jsonl").read_bytes()
    lines = call_bytes.decode("utf-8-sig").split("\n")
    # A concurrent append may expose an incomplete final row. Never treat that
    # fragment as completed evidence; explicitly disclose the snapshot boundary.
    incomplete_tail_excluded = bool(call_bytes) and not call_bytes.endswith(b"\n")
    if incomplete_tail_excluded:
        lines = lines[:-1]
    events = [json.loads(line) for line in lines if line.strip()]
    statuses, phases, failures = Counter(), Counter(), []
    for event in events:
        if event.get("status") != "succeeded":
            continue
        question = questions[event["example_id"]]
        if question["answer_type"] != "number":
            continue
        phases[event["phase"]] += 1
        try:
            evidence = audit_numeric_precision(protocol, event["response"], "number")
            statuses[evidence["status"]] += 1
        except ValueError as exc:
            statuses["PRECISION_LOSS"] += 1
            failures.append({"example_id": event["example_id"], "call_id": event["call_id"],
                             "phase": event["phase"], "error": str(exc)})
    report = {"audit_version": AUDIT_VERSION, "status": "BLOCKED" if failures else "COMPLETE",
              "audited_at_utc": datetime.now(timezone.utc).isoformat(), "study_root": str(root),
              "protocol_hash": digest, "calls_snapshot_sha256": hashlib.sha256(call_bytes).hexdigest(),
              "calls_snapshot_bytes": len(call_bytes), "incomplete_tail_excluded": incomplete_tail_excluded,
              "successful_numeric_calls_checked": sum(phases.values()),
              "checked_calls_by_phase": dict(phases), "status_counts": dict(statuses),
              "precision_loss_calls": len(failures), "failures": failures,
              "frozen_parser_limitation": PRECISION_LIMITATION,
              "trusted_keys_read": False, "llm_calls": 0, "recorded_answers_or_labels_changed": False,
              "command": ".\\.venv\\Scripts\\python.exe scripts\\audit_live_study_precision.py --study-root " + str(root)}
    output = root / "reports" / "numeric_precision_audit.json"
    write_json(output, report)
    console = json.dumps(report, ensure_ascii=False, indent=2) + "\nAudit report: " + str(output) + "\n"
    (root / "reports" / "numeric_precision_audit_console.txt").write_text(console, encoding="utf-8")
    print(console, end="")
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
