"""Collect genuine pairs under one frozen protocol with a bounded session."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rechecklab.study_collection import collect_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/live_study/v0.3.0"))
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--retry-failures", action="store_true")
    parser.add_argument("--resolve-uncertain", action="store_true")
    args = parser.parse_args()
    status = collect_study(args.study_root, host=args.host, minutes=args.minutes, smoke=args.smoke,
                          retry_failures=args.retry_failures, resolve_uncertain=args.resolve_uncertain,
                          on_progress=lambda message: print(message, flush=True))
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 2 if status["status"] in ("BLOCKED", "FAILED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
