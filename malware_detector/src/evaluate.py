"""Evaluate a trained detector against a labeled dataset.

Usage:
    python -m src.evaluate --model models/detector --samples data/test --labels data/test_labels.csv
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from src.dataset import build_dataset
from src.metrics import compute_metrics
from src.model import ModelWrapper, vectors_to_matrix


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the detector on labeled data.")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--samples", required=True, help="Directory of test PE samples")
    parser.add_argument("--labels", required=True, help="CSV file with filename,label columns")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved threshold")
    parser.add_argument("--output", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args(argv)

    try:
        model = ModelWrapper.load(args.model)
    except Exception as e:
        print(f"[evaluate] ERROR loading model: {e}", file=sys.stderr)
        return 2

    print(f"[evaluate] Scanning {args.samples} against labels {args.labels} ...")
    samples, report = build_dataset(args.samples, args.labels)
    print("[evaluate] Dataset report:")
    print(report.summary())

    if len(samples) == 0:
        print("[evaluate] ERROR: no usable labeled samples found.", file=sys.stderr)
        return 2

    X = vectors_to_matrix([s.vector for s in samples])
    y = np.array([s.label for s in samples])

    threshold = args.threshold if args.threshold is not None else model.threshold
    probs = model.predict_proba(X)
    metrics = compute_metrics(y, probs, threshold)

    print(f"[evaluate] Metrics at threshold {threshold:.6f}:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[evaluate] Wrote metrics to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
