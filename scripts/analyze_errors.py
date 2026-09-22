#!/usr/bin/env python3
"""Create traceable error-analysis tables from an evaluation predictions.csv."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.error_analysis import summarize_errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="Per-image predictions.csv exported by an evaluation run")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to the predictions file's directory")
    args = parser.parse_args()

    output_dir = args.output_dir or args.predictions.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    errors, pairs, per_class = summarize_errors(pd.read_csv(args.predictions))
    errors.to_csv(output_dir / "error_examples.csv", index=False)
    pairs.to_csv(output_dir / "error_pairs.csv", index=False)
    per_class.to_csv(output_dir / "error_by_class.csv", index=False)
    print({"evaluated_images": int(per_class.support.sum()), "errors": len(errors), "error_pairs": len(pairs)})


if __name__ == "__main__":
    main()
