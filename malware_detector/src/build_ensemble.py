"""Assemble the served model: a logit-averaging ensemble of LightGBM Boosters,
with its threshold tuned on the EMBER 2018 test set.

    python -m src.build_ensemble \
        --booster data/lgb/runC.txt:1 \
        --booster data/ember2018_raw/ember2018/ember_model_2018.txt:1 \
        --target-fpr 0.008 --output models/detector

`--booster PATH:WEIGHT` may be repeated. None of the members may have been
trained on the EMBER test set, since that is what the threshold and the
reported metrics are computed on.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from src.features import FEATURE_DIM
from src.metrics import compute_metrics, tune_threshold
from src.model import ENSEMBLE_BACKEND, LogitEnsemble, ModelWrapper, TrainingConfig
from src.train_ember import X_TEST_FILE, Y_TEST_FILE


def _split(spec: str) -> tuple[str, float]:
    path, _, weight = spec.rpartition(":")
    return path, float(weight)


def main(argv=None) -> int:
    import lightgbm as lgb

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--booster", action="append", default=[], help="PATH:WEIGHT of a LightGBM model file")
    ap.add_argument("--ember-vectorized-dir", default="data/ember2018_vectorized")
    ap.add_argument("--target-fpr", type=float, default=0.008)
    ap.add_argument("--output", default="models/detector")
    args = ap.parse_args(argv)

    members = []
    for spec in args.booster:
        path, w = _split(spec)
        members.append((os.path.basename(path), lgb.Booster(model_file=path), w))
    ensemble = LogitEnsemble(members)

    n_test = os.path.getsize(os.path.join(args.ember_vectorized_dir, Y_TEST_FILE)) // 4
    X = np.memmap(os.path.join(args.ember_vectorized_dir, X_TEST_FILE), dtype=np.float32, mode="r", shape=(n_test, FEATURE_DIM))
    y = np.asarray(np.memmap(os.path.join(args.ember_vectorized_dir, Y_TEST_FILE), dtype=np.float32, mode="r")).astype(int)
    probs = np.concatenate([ensemble.predict(np.asarray(X[i : i + 20000], dtype=np.float64)) for i in range(0, n_test, 20000)])

    threshold, satisfied = tune_threshold(y, probs, target_fpr=args.target_fpr)
    metrics = compute_metrics(y, probs, threshold)
    metrics["target_fpr"] = args.target_fpr
    metrics["fpr_constraint_satisfied"] = satisfied
    metrics["evaluated_on"] = f"EMBER 2018 test set ({n_test} rows), held out from every member"
    metrics["members"] = [{"name": n, "weight": w} for n, _, w in members]
    for k, v in metrics.items():
        print(f"{k}: {v}")

    model = ModelWrapper(backend=ENSEMBLE_BACKEND, estimator=ensemble, threshold=threshold)
    config = TrainingConfig(
        backend=ENSEMBLE_BACKEND, seed=42, test_size=0.0, target_fpr=args.target_fpr,
        feature_dim=FEATURE_DIM, n_train=0, n_val=n_test,
    )
    model.save(args.output, config, metrics)
    print(f"saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
