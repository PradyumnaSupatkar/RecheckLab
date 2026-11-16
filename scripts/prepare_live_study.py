"""Pin GSM8K/CommonsenseQA and freeze question partitions before collection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.study_data import prepare_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", "--study-root", dest="output", type=Path,
                        default=ROOT / "outputs" / "live_study" / "v0.3.0")
    parser.add_argument("--maths-target", type=int, default=1000)
    parser.add_argument("--target-target", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        plan = prepare_study(args.output, args.maths_target, args.target_target, args.seed)
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        parser.exit(1, f"Study preparation failed: {type(exc).__name__}: {exc}\nPlanned targets were not reduced.\n")
    print(json.dumps({key: plan[key] for key in ("status", "planned_total_by_domain", "planned_questions_by_partition",
        "manifest_sha256", "study_plan_hash", "duplicate_overlap_audit")}, indent=2))
    print("Fixed study plan:", args.output.resolve() / "study_plan.json")
    print("No LLM calls were made; trusted keys are stored separately.")


if __name__ == "__main__":
    main()
