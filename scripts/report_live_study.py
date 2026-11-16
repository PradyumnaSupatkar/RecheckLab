"""Generate an evidence-based diagnostic handoff for a complete or partial study."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rechecklab.study_report import generate_status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=ROOT / "outputs" / "live_study" / "v0.3.0")
    args = parser.parse_args()
    status = generate_status(ROOT, args.study_root)
    print(f"Overall status: {status['overall_status']}")
    print(ROOT / "reports" / "live_study" / "CURRENT_STATUS.md")
    print(status["diagnostic_bundle"]["path"])


if __name__ == "__main__":
    main()
