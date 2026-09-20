"""Vectorize and train directly from the official EMBER 2018 raw-feature
JSONL dataset (train_features_0..5.jsonl, test_features.jsonl under a
directory such as data/ember2018_raw/ember2018/).

This bypasses src/dataset.py's samples-directory/labels.csv flow because
EMBER does not distribute the underlying PE binaries (Elastic cannot
legally redistribute most of the malware) -- only pre-extracted feature
JSON per sample. This module reads that JSON and vectorizes it with the
*exact same* EmberPEFeatureExtractor.process_raw_features used by
src/features.py for live files (see src/ember_extractor.py), so a model
trained here is scored identically at prediction time on new files.

EMBER labels: 1 = malicious, 0 = benign, -1 = unlabeled (a large chunk of
the training set is deliberately unlabeled in the original EMBER release
for semi-supervised experiments). Unlabeled rows are dropped before
training/validation, exactly as the official `ember` package does.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.ember_extractor import EmberPEFeatureExtractor
from src.utils import safe_float

TRAIN_SHARDS = 6
X_TRAIN_FILE = "X_train_local.dat"
Y_TRAIN_FILE = "y_train_local.dat"
X_TEST_FILE = "X_test_local.dat"
Y_TEST_FILE = "y_test_local.dat"


def _train_jsonl_paths(data_dir: str) -> List[str]:
    return [os.path.join(data_dir, f"train_features_{i}.jsonl") for i in range(TRAIN_SHARDS)]


def _test_jsonl_path(data_dir: str) -> str:
    return os.path.join(data_dir, "test_features.jsonl")


def _count_lines(paths: List[str]) -> int:
    total = 0
    for p in paths:
        with open(p, "rb") as f:
            total += sum(1 for _ in f)
    return total


def vectorize_jsonl_to_memmap(
    jsonl_paths: List[str],
    x_out_path: str,
    y_out_path: str,
    feature_dim: int,
    log_every: int = 50000,
) -> int:
    """Stream raw-feature JSON lines -> (X, y) memmap arrays on disk.

    Streaming + memmap keeps peak RAM roughly constant regardless of
    dataset size (important on this machine's 3.8GB RAM for a 1.1M-row,
    2381-dim dataset), unlike loading everything into a plain ndarray.
    """
    extractor = EmberPEFeatureExtractor()
    assert extractor.dim == feature_dim

    n = _count_lines(jsonl_paths)
    X = np.memmap(x_out_path, dtype=np.float32, mode="w+", shape=(n, feature_dim))
    y = np.memmap(y_out_path, dtype=np.float32, mode="w+", shape=(n,))

    row = 0
    for path in jsonl_paths:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)
                vec = extractor.process_raw_features(raw)
                # Never let NaN/inf from a degenerate record (e.g. empty
                # byte histogram) reach the stored matrix.
                X[row] = [safe_float(v) for v in vec]
                y[row] = float(raw.get("label", -1))
                row += 1
                if row % log_every == 0:
                    print(f"[ember-vectorize] {row}/{n}")
    X.flush()
    y.flush()
    del X, y
    assert row == n
    return n


def vectorize_ember_dataset(raw_data_dir: str, out_dir: str, feature_dim: int) -> None:
    os.makedirs(out_dir, exist_ok=True)

    train_paths = _train_jsonl_paths(raw_data_dir)
    missing = [p for p in train_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing EMBER train shard(s): {missing}")
    test_path = _test_jsonl_path(raw_data_dir)
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Missing EMBER test file: {test_path}")

    print("[ember-vectorize] Vectorizing training shards...")
    n_train = vectorize_jsonl_to_memmap(
        train_paths, os.path.join(out_dir, X_TRAIN_FILE), os.path.join(out_dir, Y_TRAIN_FILE), feature_dim
    )
    print(f"[ember-vectorize] Wrote {n_train} training rows.")

    print("[ember-vectorize] Vectorizing test set...")
    n_test = vectorize_jsonl_to_memmap(
        [test_path], os.path.join(out_dir, X_TEST_FILE), os.path.join(out_dir, Y_TEST_FILE), feature_dim
    )
    print(f"[ember-vectorize] Wrote {n_test} test rows.")

    with open(os.path.join(out_dir, "shapes.json"), "w") as f:
        json.dump({"n_train": n_train, "n_test": n_test, "feature_dim": feature_dim}, f)


def read_vectorized(out_dir: str, feature_dim: int) -> Tuple[np.memmap, np.memmap, np.memmap, np.memmap]:
    with open(os.path.join(out_dir, "shapes.json")) as f:
        shapes = json.load(f)
    n_train, n_test = shapes["n_train"], shapes["n_test"]
    assert shapes["feature_dim"] == feature_dim

    X_train = np.memmap(os.path.join(out_dir, X_TRAIN_FILE), dtype=np.float32, mode="r", shape=(n_train, feature_dim))
    y_train = np.memmap(os.path.join(out_dir, Y_TRAIN_FILE), dtype=np.float32, mode="r", shape=(n_train,))
    X_test = np.memmap(os.path.join(out_dir, X_TEST_FILE), dtype=np.float32, mode="r", shape=(n_test, feature_dim))
    y_test = np.memmap(os.path.join(out_dir, Y_TEST_FILE), dtype=np.float32, mode="r", shape=(n_test,))
    return X_train, y_train, X_test, y_test


@dataclass
class EmberSplitReport:
    n_train_total: int
    n_train_labeled: int
    n_train_unlabeled: int
    n_test_total: int
    n_test_labeled: int
    train_class_counts: dict
    test_class_counts: dict

    def summary(self) -> str:
        return "\n".join(
            [
                f"Train rows total: {self.n_train_total} (labeled: {self.n_train_labeled}, "
                f"unlabeled/dropped: {self.n_train_unlabeled})",
                f"Train class counts (labeled only): {self.train_class_counts}",
                f"Test rows total: {self.n_test_total} (labeled: {self.n_test_labeled})",
                f"Test class counts: {self.test_class_counts}",
            ]
        )


def materialize_subset(
    X: np.memmap,
    y: np.memmap,
    mask: np.ndarray,
    x_out_path: str,
    y_out_path: str,
    chunk_rows: int = 20000,
) -> Tuple[np.memmap, np.memmap]:
    """Write the rows where `mask` is True to new memmap files, streaming in
    row chunks so peak RAM is O(chunk_rows * dim), not O(n_rows * dim).

    This machine has 3.8GB RAM; the full EMBER training matrix
    (~1.1M x 2381 float32) is ~10GB, so this must never do `X[mask]` on the
    full array directly (used both to drop unlabeled rows and to carve out
    a train/validation split without ever materializing the full dataset).
    """
    n_total, dim = X.shape
    n_selected = int(mask.sum())

    X_out = np.memmap(x_out_path, dtype=np.float32, mode="w+", shape=(n_selected, dim))
    y_out = np.memmap(y_out_path, dtype=np.float32, mode="w+", shape=(n_selected,))

    write_row = 0
    for start in range(0, n_total, chunk_rows):
        end = min(start + chunk_rows, n_total)
        chunk_mask = mask[start:end]
        if not chunk_mask.any():
            continue
        x_chunk = np.asarray(X[start:end])[chunk_mask]
        y_chunk = np.asarray(y[start:end])[chunk_mask]
        n = x_chunk.shape[0]
        X_out[write_row : write_row + n] = x_chunk
        y_out[write_row : write_row + n] = y_chunk
        write_row += n

    X_out.flush()
    y_out.flush()
    assert write_row == n_selected
    return X_out, y_out


def materialize_labeled_subset(
    X: np.memmap, y: np.memmap, x_out_path: str, y_out_path: str, chunk_rows: int = 20000
) -> Tuple[np.memmap, np.memmap]:
    """Write a labeled-only (label != -1) copy of (X, y)."""
    mask = np.asarray(y) != -1
    return materialize_subset(X, y, mask, x_out_path, y_out_path, chunk_rows)


def predict_proba_chunked(model, X: np.memmap, chunk_rows: int = 20000) -> np.ndarray:
    """Score a large memmapped feature matrix in row chunks so peak RAM for
    prediction is O(chunk_rows * dim) instead of O(n_rows * dim)."""
    n = X.shape[0]
    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, chunk_rows):
        end = min(start + chunk_rows, n)
        out[start:end] = model.predict_proba(np.asarray(X[start:end]))
    return out


def run_ember_training(args, cfg) -> int:
    """Entry point for `python -m src.train --ember-data-dir ...`."""
    from src.features import FEATURE_DIM
    from src.metrics import compute_metrics, tune_threshold
    from src.model import ModelWrapper, TrainingConfig

    raw_dir = args.ember_data_dir
    vec_dir = args.ember_vectorized_dir
    seed = cfg["seed"]

    shapes_path = os.path.join(vec_dir, "shapes.json")
    if not os.path.exists(shapes_path):
        print(
            f"[train-ember] No cached vectorized features at {vec_dir}; vectorizing from "
            f"{raw_dir} (reads ~1.1M raw feature records, can take ~30 minutes)..."
        )
        vectorize_ember_dataset(raw_dir, vec_dir, FEATURE_DIM)
    else:
        print(f"[train-ember] Reusing cached vectorized features at {vec_dir}")

    X_train_full, y_train_full, X_test, y_test = read_vectorized(vec_dir, FEATURE_DIM)
    print(f"[train-ember] Raw EMBER rows -- train: {X_train_full.shape[0]}, test: {X_test.shape[0]}")

    labeled_mask = np.asarray(y_train_full) != -1
    n_labeled = int(labeled_mask.sum())
    n_unlabeled = X_train_full.shape[0] - n_labeled
    print(f"[train-ember] Labeled train rows: {n_labeled} (dropping {n_unlabeled} unlabeled rows, label == -1)")

    if n_labeled == 0:
        print("[train-ember] ERROR: no labeled training rows found.", file=sys.stderr)
        return 2

    labeled_idx = np.where(labeled_mask)[0]
    y_labeled = np.asarray(y_train_full)[labeled_idx].astype(int)

    from sklearn.model_selection import train_test_split

    positions = np.arange(len(labeled_idx))
    train_pos, val_pos = train_test_split(
        positions, test_size=cfg["test_size"], random_state=seed, stratify=y_labeled
    )
    train_rows = labeled_idx[train_pos]
    val_rows = labeled_idx[val_pos]

    mask_train = np.zeros(X_train_full.shape[0], dtype=bool)
    mask_train[train_rows] = True
    mask_val = np.zeros(X_train_full.shape[0], dtype=bool)
    mask_val[val_rows] = True
    print(f"[train-ember] Train: {int(mask_train.sum())} | Validation: {int(mask_val.sum())}")

    print("[train-ember] Materializing train/validation split to disk (streamed, RAM-safe)...")
    X_tr, y_tr = materialize_subset(
        X_train_full, y_train_full, mask_train,
        os.path.join(vec_dir, "X_train_final.dat"), os.path.join(vec_dir, "y_train_final.dat"),
    )
    X_val, y_val = materialize_subset(
        X_train_full, y_train_full, mask_val,
        os.path.join(vec_dir, "X_val_final.dat"), os.path.join(vec_dir, "y_val_final.dat"),
    )

    y_tr_arr = np.asarray(y_tr).astype(int)
    y_val_arr = np.asarray(y_val).astype(int)
    print(f"[train-ember] Train class counts: benign(0)={int((y_tr_arr==0).sum())} malware(1)={int((y_tr_arr==1).sum())}")
    print(f"[train-ember] Validation class counts: benign(0)={int((y_val_arr==0).sum())} malware(1)={int((y_val_arr==1).sum())}")

    model = ModelWrapper.create(backend=cfg["backend"], seed=seed)
    print(f"[train-ember] Training backend: {model.backend} on {X_tr.shape[0]} rows x {X_tr.shape[1]} features...")
    t0 = time.time()
    model.fit(X_tr, y_tr_arr)
    print(f"[train-ember] Training took {time.time()-t0:.1f}s")

    print("[train-ember] Scoring validation set...")
    val_probs = predict_proba_chunked(model, X_val)

    threshold, satisfied = tune_threshold(y_val_arr, val_probs, target_fpr=cfg["target_fpr"])
    model.threshold = threshold
    metrics = compute_metrics(y_val_arr, val_probs, threshold)
    metrics["target_fpr"] = cfg["target_fpr"]
    metrics["fpr_constraint_satisfied"] = satisfied

    print(
        f"[train-ember] Validation metrics at tuned threshold ({threshold:.6f}, "
        f"target FPR < {cfg['target_fpr']:.3%}):"
    )
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    if not satisfied:
        print(
            "[train-ember] WARNING: no threshold on this validation set achieves FPR < "
            f"{cfg['target_fpr']:.3%}. Reporting the lowest-FPR threshold instead."
        )
    elif metrics["recall_tpr"] < 0.95:
        print(
            f"[train-ember] NOTE: FPR constraint satisfied but validation TPR "
            f"({metrics['recall_tpr']:.3%}) is below the 95% WhiteBox target."
        )

    print("[train-ember] Scoring official EMBER test set (held out; not used for fit or threshold tuning)...")
    y_test_arr = np.asarray(y_test).astype(int)
    test_probs = predict_proba_chunked(model, X_test)
    test_metrics = compute_metrics(y_test_arr, test_probs, threshold)
    print("[train-ember] Held-out EMBER test set metrics (at the validation-tuned threshold):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    config = TrainingConfig(
        backend=model.backend,
        seed=seed,
        test_size=cfg["test_size"],
        target_fpr=cfg["target_fpr"],
        feature_dim=FEATURE_DIM,
        n_train=int(X_tr.shape[0]),
        n_val=int(X_val.shape[0]),
    )
    metrics["ember_test_metrics"] = test_metrics
    metrics["ember_test_n"] = int(X_test.shape[0])
    model.save(args.output, config, metrics)
    print(f"[train-ember] Saved model bundle to {args.output}")
    return 0
