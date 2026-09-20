#!/usr/bin/env bash
# Train the detector. Usage:
#   ./train.sh <samples_dir> <labels_csv> <output_model_dir>
# Defaults to data/train, data/labels.csv, models/detector if omitted.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES="${1:-data/train}"
LABELS="${2:-data/labels.csv}"
OUTPUT="${3:-models/detector}"

python3 -m src.train --samples "$SAMPLES" --labels "$LABELS" --output "$OUTPUT"
