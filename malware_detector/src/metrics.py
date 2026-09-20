"""Shared metric computation and threshold tuning for train.py / evaluate.py."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    except ValueError:
        pr_auc = float("nan")

    return {
        "threshold": float(threshold),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": float(precision),
        "recall_tpr": float(tpr),
        "fpr": float(fpr),
        "f1": float(f1),
        "roc_auc": float(roc_auc) if roc_auc == roc_auc else None,  # NaN -> None (JSON safe)
        "pr_auc": float(pr_auc) if pr_auc == pr_auc else None,
    }


def tune_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float = 0.01
) -> Tuple[float, bool]:
    """Select the threshold maximizing TPR subject to FPR < target_fpr.

    Returns (threshold, constraint_satisfied). If no threshold achieves
    FPR < target_fpr, falls back to the threshold with the lowest achievable
    FPR (ties broken by higher TPR) and reports constraint_satisfied=False.
    """
    fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_prob)
    # roc_curve's first threshold is thr[0] = max(score)+1, an unusable sentinel.
    valid = thresholds <= 1.0
    fpr_arr, tpr_arr, thresholds = fpr_arr[valid], tpr_arr[valid], thresholds[valid]

    if len(thresholds) == 0:
        return 0.5, False

    feasible = fpr_arr < target_fpr
    if feasible.any():
        candidate_tpr = np.where(feasible, tpr_arr, -1)
        best_idx = int(np.argmax(candidate_tpr))
        return float(thresholds[best_idx]), True

    # Constraint infeasible on this validation set: report the threshold
    # that minimizes FPR (breaking ties by higher TPR) instead of silently
    # picking something arbitrary.
    min_fpr = fpr_arr.min()
    near_min = fpr_arr == min_fpr
    candidate_tpr = np.where(near_min, tpr_arr, -1)
    best_idx = int(np.argmax(candidate_tpr))
    return float(thresholds[best_idx]), False
