"""Train the PE malware detector.

Usage:
    python -m src.train --samples data/train --labels data/labels.csv --output models/detector

The exact same feature extraction code (src/features.py) used here is used
by src/predict.py and src/evaluate.py -- the trained model bundle stores the
feature schema and refuses to load if it no longer matches, so a training/
inference mismatch fails loudly instead of silently producing garbage scores.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

from src.dataset import build_dataset, stratified_split
from src.features import FEATURE_DIM
from src.metrics import compute_metrics, tune_threshold
from src.model import DEFAULT_SEED, ModelWrapper, TrainingConfig, vectors_to_matrix


def load_config(path: str | None) -> dict:
    defaults = {
        "seed": DEFAULT_SEED,
        "test_size": 0.2,
        "target_fpr": 0.01,
        "backend": None,  # None = auto-select best available
    }
    if path:
        with open(path) as f:
            user_cfg = yaml.safe_load(f) or {}
        defaults.update(user_cfg.get("train", {}))
    return defaults


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train the static PE malware detector.")
    parser.add_argument("--samples", default=None, help="Directory of training PE samples")
    parser.add_argument("--labels", default=None, help="CSV file with filename,label columns")
    parser.add_argument("--output", required=True, help="Output directory for the trained model")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config file")
    parser.add_argument("--test-size", type=float, default=None, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--backend", default=None, choices=["lightgbm", "xgboost", "sklearn_hgb", "sklearn_rf"])
    parser.add_argument("--target-fpr", type=float, default=None, help="Threshold-tuning FPR ceiling")
    parser.add_argument(
        "--ember-data-dir",
        default=None,
        help="Path to the extracted EMBER 2018 raw JSONL directory "
        "(train_features_0..5.jsonl, test_features.jsonl). When given, bypasses "
        "--samples/--labels and trains on the official EMBER dataset instead.",
    )
    parser.add_argument(
        "--ember-vectorized-dir",
        default="data/ember2018_vectorized",
        help="Cache directory for vectorized EMBER features (auto-built from "
        "--ember-data-dir on first use, reused after).",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.test_size is not None:
        cfg["test_size"] = args.test_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.backend is not None:
        cfg["backend"] = args.backend
    if args.target_fpr is not None:
        cfg["target_fpr"] = args.target_fpr

    seed = cfg["seed"]
    np.random.seed(seed)

    if args.ember_data_dir:
        from src.train_ember import run_ember_training

        return run_ember_training(args, cfg)

    if not args.samples or not args.labels:
        print("[train] ERROR: --samples and --labels are required (or use --ember-data-dir).", file=sys.stderr)
        return 2

    print(f"[train] Scanning {args.samples} against labels {args.labels} ...")
    samples, report = build_dataset(args.samples, args.labels)
    print("[train] Dataset report:")
    print(report.summary())

    if report.n_duplicate_hashes:
        print(
            f"[train] WARNING: {report.n_duplicate_hashes} duplicate content hash(es) found; "
            "duplicates will be collapsed to a single representative before splitting so the "
            "same sample cannot land in both train and validation."
        )
    if report.n_label_conflicts:
        print(
            f"[train] WARNING: {report.n_label_conflicts} hash(es) had conflicting labels across "
            "rows; majority label used for the collapsed representative."
        )

    if len(samples) == 0:
        print("[train] ERROR: no usable labeled samples found. Cannot train.", file=sys.stderr)
        return 2

    class_counts = report.class_counts
    if class_counts.get("benign(0)", 0) == 0 or class_counts.get("malware(1)", 0) == 0:
        print(
            "[train] ERROR: dataset must contain both benign and malware examples to train "
            "and to evaluate FPR/TPR.",
            file=sys.stderr,
        )
        return 2

    try:
        train_samples, val_samples = stratified_split(samples, test_size=cfg["test_size"], seed=seed)
    except ValueError as e:
        print(f"[train] ERROR: {e}", file=sys.stderr)
        return 2

    print(f"[train] Train: {len(train_samples)} samples | Validation: {len(val_samples)} samples")

    X_train = vectors_to_matrix([s.vector for s in train_samples])
    y_train = np.array([s.label for s in train_samples])
    X_val = vectors_to_matrix([s.vector for s in val_samples])
    y_val = np.array([s.label for s in val_samples])

    model = ModelWrapper.create(backend=cfg["backend"], seed=seed)
    print(f"[train] Training backend: {model.backend}")
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)

    threshold, satisfied = tune_threshold(y_val, val_probs, target_fpr=cfg["target_fpr"])
    model.threshold = threshold
    metrics = compute_metrics(y_val, val_probs, threshold)
    metrics["target_fpr"] = cfg["target_fpr"]
    metrics["fpr_constraint_satisfied"] = satisfied

    print("[train] Validation metrics at tuned threshold "
          f"({threshold:.6f}, target FPR < {cfg['target_fpr']:.3%}):")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if not satisfied:
        print(
            "[train] WARNING: no threshold on this validation set achieves FPR < "
            f"{cfg['target_fpr']:.3%}. Reporting the lowest-FPR threshold instead. "
            "More/better validation data or model tuning is needed to meet the "
            "WhiteBox checkpoint's FPR < 1% / TPR > 95% requirement."
        )
    elif metrics["recall_tpr"] < 0.95:
        print(
            "[train] NOTE: FPR constraint satisfied but validation TPR "
            f"({metrics['recall_tpr']:.3%}) is below the 95% WhiteBox target."
        )

    config = TrainingConfig(
        backend=model.backend,
        seed=seed,
        test_size=cfg["test_size"],
        target_fpr=cfg["target_fpr"],
        feature_dim=X_train.shape[1],
        n_train=len(train_samples),
        n_val=len(val_samples),
    )
    model.save(args.output, config, metrics)
    print(f"[train] Saved model bundle to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
