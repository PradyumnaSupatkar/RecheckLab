"""Train/evaluate genuine collected study pairs; does not call the solver."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rechecklab.study_training import train_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", "--root", dest="root", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "live_study" / "v0.3.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repeats", type=int, default=600)
    arguments = parser.parse_args(argv)
    status = train_study(arguments.root, seed=arguments.seed, bootstrap_repeats=arguments.bootstrap_repeats)
    print(json.dumps({"status": status["status"], "training": status["training"], "evaluation": status["evaluation"],
                      "adaptation": status["adaptation"], "status_file": str(arguments.root / "training_status.json")}, indent=2))
    return 1 if status["status"] == "FAILED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
