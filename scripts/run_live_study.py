"""Run a bounded live-study session and ALWAYS refresh the diagnostic handoff.

Completed calls are resumed; planned targets never shrink to fit a session.
Qwen is fixed: this command trains only the small outcome classifier, if ready.
"""
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rechecklab.study_collection import write_json
from rechecklab.study_report import generate_status


def now():
    return datetime.now(timezone.utc).isoformat()


def merge_context(root, **changes):
    path = root / "run_context.json"
    context = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    context.update(changes)
    write_json(path, context)
    return context


def execute(root, label, arguments):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    log = root / "logs" / f"{label}_{stamp}.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    entry = {"command": subprocess.list2cmdline([sys.executable, "-u", *arguments]),
             "started_at": now(), "ended_at": None, "exit_code": None, "log": str(log)}
    context = merge_context(root)
    history = context.setdefault("commands", [])
    history.append(entry)
    merge_context(root, commands=history, active_processes=[{"pid": "tracked child via run controller", "command": entry["command"]}])
    print(entry["command"], flush=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen([sys.executable, "-u", *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        merge_context(root, active_processes=[{"pid": process.pid, "command": entry["command"], "log": str(log)}])
        try:
            result = process.wait()
        except BaseException:
            # Do not leave an untracked collector after the controller exits.
            process.terminate()
            process.wait(timeout=30)
            raise
    entry.update(ended_at=now(), exit_code=result)
    current = merge_context(root)
    commands = current.get("commands", [])
    for index, saved in enumerate(commands):
        if saved.get("log") == entry["log"]:
            commands[index] = entry
    merge_context(root, commands=commands, active_processes=[])
    print(f"{label}: exit {result}; log {log}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "outputs" / "live_study" / "v0.3.0")
    parser.add_argument("--minutes", type=float, default=60, help="Maximum new collection time; smoke consumes this same budget.")
    parser.add_argument("--smoke", type=int, default=4)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--resolve-uncertain", action="store_true")
    args = parser.parse_args()
    root = args.study_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    context = merge_context(root)
    merge_context(root, started_at=context.get("started_at", now()), latest_session_started_at=now(), ended_at=None,
                  implementation={"status": "IMPLEMENTED", "evidence": "Pinned preparation, shared frozen protocol, durable bounded collection, guarded training/evaluation and diagnostic handoff; execution statuses are separate."},
                  live_integration={"status": "INCOMPLETE", "not_run_reasons": ["Shared live/collection implementation is implemented and tested; no newly trained and independently evaluated study gate exists to activate."]},
                  python=sys.version, packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib", "pyarrow", "matplotlib", "streamlit")},
                  platform=platform.platform(), session_collection_budget_minutes=args.minutes,
                  resume_commands=[r'cd "' + str(ROOT) + '"',
                      r'.\.venv\Scripts\python.exe scripts\run_live_study.py --study-root "' + str(root) + '" --minutes 60'])
    code_hashes = {str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in ("rechecklab", "scripts", "tests") for p in (ROOT / folder).glob("*.py")}
    merge_context(root, code_version={"git": "unknown: workspace has no Git metadata", "source_sha256": code_hashes})
    collection_exit = 0
    collection_seconds = 0
    try:
        if not (root / "questions.jsonl").exists():
            if execute(root, "prepare", ["scripts/prepare_live_study.py", "--study-root", str(root)]):
                raise RuntimeError("Pinned data preparation failed; inspect the captured log.")
        if not (root / "protocol.json").exists():
            if execute(root, "candidate", ["scripts/freeze_live_protocol.py", "--study-root", str(root), "--candidate"]):
                raise RuntimeError("Actual backend inspection failed.")
            smoke_started = time.monotonic()
            if execute(root, "smoke", ["scripts/collect_live_study.py", "--study-root", str(root), "--minutes", str(args.minutes), "--smoke", str(args.smoke)]):
                raise RuntimeError("Smoke failed; no protocol promotion or held-out generation occurred.")
            collection_seconds += time.monotonic() - smoke_started
            if execute(root, "freeze", ["scripts/freeze_live_protocol.py", "--study-root", str(root)]):
                raise RuntimeError("Smoke did not justify freezing this protocol.")
        remaining_minutes = max(0, args.minutes - collection_seconds / 60)
        if remaining_minutes > 0:
            command = ["scripts/collect_live_study.py", "--study-root", str(root), "--minutes", str(remaining_minutes)]
            if args.retry_failures:
                command.append("--retry-failures")
            if args.resolve_uncertain:
                command.append("--resolve-uncertain")
            collection_exit = execute(root, "collection", command)
        # An incomplete collection still deserves an actual guarded training
        # invocation, which records why fitting/evaluation did or did not run.
        training_exit = execute(root, "training", ["scripts/train_live_study.py", "--study-root", str(root)])
        return 2 if collection_exit or training_exit else 0
    except Exception as exc:
        context = merge_context(root)
        errors = context.get("errors", [])
        errors.append({"error": str(exc), "error_type": type(exc).__name__, "traceback": traceback.format_exc(), "timestamp_utc": now()})
        merge_context(root, errors=errors)
        return 2
    finally:
        merge_context(root, ended_at=now(), latest_session_elapsed_seconds=time.monotonic() - started, active_processes=[])
        status = generate_status(ROOT, root)
        print(f"Overall: {status['overall_status']}\nReport: {ROOT / 'reports/live_study/CURRENT_STATUS.md'}\nZIP: {status['diagnostic_bundle']['path']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
