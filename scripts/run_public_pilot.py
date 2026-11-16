"""Run the included public-log pilot; no LLM, key, or GPU required."""
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.data import load_pairs
from rechecklab.experiment import run_experiment
from rechecklab.report import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional paired-outcomes JSONL/CSV/Parquet")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "v0.2",
                        help="Separate run folder; default outputs/v0.2 preserves shipped v0.1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--sizes", default="25,50,100,200,400")
    parser.add_argument("--harm-weight", type=float, default=1.0)
    args = parser.parse_args()
    if bool(args.source) != bool(args.target):
        parser.error("Provide both --source and --target, or neither.")
    if args.repeats < 1 or args.harm_weight <= 0:
        parser.error("Repeats and harm weight must be positive.")
    source = load_pairs(args.source) if args.source else None
    target = load_pairs(args.target) if args.target else None
    meta, policies, adaptation = run_experiment(args.output, source, target, seed=args.seed,
        repeats=args.repeats, sizes=tuple(int(v) for v in args.sizes.split(",")), harm_weight=args.harm_weight)
    render_report(args.output)
    print("Finished:", meta["provenance"])
    print(policies[["evaluation","policy","accuracy_pct","harmful_flip_pct_of_all_questions","average_total_tokens"]].to_string(index=False))
    print("Report:", args.output / "reports" / "Progress_Report.html")


if __name__ == "__main__":
    main()
