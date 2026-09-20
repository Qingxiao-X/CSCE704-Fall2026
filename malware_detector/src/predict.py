"""Predict malware/benign for a single PE file or a directory of files.

Usage:
    python -m src.predict --model models/detector --input sample.exe
    python -m src.predict --model models/detector --input samples/ --output predictions.csv

Samples are only ever statically parsed (src/features.py, via pefile) --
never executed. Output CSV columns: filename,sha256,prediction,malware_probability,status
  - prediction: 0 (benign) / 1 (malware), blank if the file could not be scored
  - status: "ok" for scored files, otherwise a descriptive reason
    (e.g. empty_file, not_a_pe, parse_error, unreadable_file)

For single-file predictions a concise human-readable summary is printed to
stdout; stdout carries no other debug/log output.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import List, Optional, TextIO

from src.features import extract_features
from src.model import ModelWrapper, vectors_to_matrix

CSV_FIELDS = ["filename", "sha256", "prediction", "malware_probability", "status"]


class Row:
    def __init__(self, filename: str, sha256: str = "", prediction: Optional[int] = None,
                 probability: Optional[float] = None, status: str = ""):
        self.filename = filename
        self.sha256 = sha256
        self.prediction = prediction
        self.probability = probability
        self.status = status

    def to_csv_dict(self) -> dict:
        return {
            "filename": self.filename,
            "sha256": self.sha256,
            "prediction": "" if self.prediction is None else str(self.prediction),
            "malware_probability": "" if self.probability is None else f"{self.probability:.6f}",
            "status": self.status,
        }


def score_file(model: ModelWrapper, path: str, filename_label: str) -> Row:
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as e:
        return Row(filename=filename_label, status=f"unreadable_file: {e}")

    result = extract_features(data)
    if result.status != "ok" or result.vector is None:
        return Row(filename=filename_label, sha256=result.sha256, status=f"{result.status}: {result.error}" if result.error else result.status)

    X = vectors_to_matrix([result.vector])
    prob = float(model.predict_proba(X)[0])
    pred = int(prob >= model.threshold)
    return Row(filename=filename_label, sha256=result.sha256, prediction=pred, probability=prob, status="ok")


def write_csv(rows: List[Row], stream: TextIO) -> None:
    writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(row.to_csv_dict())


def print_single_result(row: Row, threshold: float) -> None:
    print(f"file: {row.filename}")
    print(f"sha256: {row.sha256}")
    print(f"status: {row.status}")
    if row.status == "ok":
        label = "MALWARE" if row.prediction == 1 else "BENIGN"
        print(f"prediction: {label} ({row.prediction})")
        print(f"malware_probability: {row.probability:.6f}")
        print(f"threshold: {threshold:.6f}")
    else:
        print("prediction: N/A (file could not be scored)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Predict malware/benign for PE file(s).")
    parser.add_argument("--model", required=True, help="Path to trained model directory")
    parser.add_argument("--input", required=True, help="Path to a single file or a directory")
    parser.add_argument("--output", default=None, help="CSV output path (directory mode; optional)")
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved threshold")
    args = parser.parse_args(argv)

    try:
        model = ModelWrapper.load(args.model)
    except Exception as e:
        print(f"error: failed to load model from {args.model}: {e}", file=sys.stderr)
        return 2
    if args.threshold is not None:
        model.threshold = args.threshold

    if not os.path.exists(args.input):
        print(f"error: input path does not exist: {args.input}", file=sys.stderr)
        return 1

    if os.path.isfile(args.input):
        row = score_file(model, args.input, os.path.basename(args.input))
        print_single_result(row, model.threshold)
        if args.output:
            with open(args.output, "w", newline="") as f:
                write_csv([row], f)
        return 0

    if os.path.isdir(args.input):
        rows: List[Row] = []
        for root, _dirs, files in sorted(os.walk(args.input)):
            for fname in sorted(files):
                full_path = os.path.join(root, fname)
                rel_name = os.path.relpath(full_path, args.input)
                try:
                    rows.append(score_file(model, full_path, rel_name))
                except Exception as e:
                    # A single corrupt/adversarial file must never abort the scan.
                    rows.append(Row(filename=rel_name, status=f"unexpected_error: {type(e).__name__}: {e}"))

        if args.output:
            with open(args.output, "w", newline="") as f:
                write_csv(rows, f)
            print(f"wrote {len(rows)} prediction(s) to {args.output}", file=sys.stderr)
        else:
            write_csv(rows, sys.stdout)
        return 0

    print(f"error: input path is neither a file nor a directory: {args.input}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
