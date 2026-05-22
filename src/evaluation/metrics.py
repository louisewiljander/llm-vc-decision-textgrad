"""
Evaluation metrics for startup success prediction.

Mirrors the metric set used in Maarouf et al. (2025):
  - Average Precision at K (AP@K, primary metric per Liu et al.)
  - Balanced accuracy
  - Precision, Recall, F1
  - AUROC
  - AUCPR

Probabilities are used for threshold-independent metrics (AUROC, AUCPR, AP@K),
while binary predictions (using a configurable threshold) are used for
accuracy, precision, recall, and F1.
"""
from typing import Optional
import numpy as np
import warnings

try:
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.evaluation.ap_at_k import compute_ap_at_k


def compute_metrics(
    y_true: list,
    y_prob: list,
    threshold: float = 0.5,
    pos_label: int = 1,
) -> dict:
    """
    Compute the full evaluation metric suite.

    Args:
        y_true:    Ground-truth binary labels (0 or 1).
        y_prob:    Model-predicted probabilities for the positive class (0–1).
        threshold: Decision threshold for converting probabilities to binary
                   predictions. Defaults to 0.5.
        pos_label: Which label is the positive class. Defaults to 1.

    Returns:
        Dictionary with keys:
            n, n_positive, n_negative, base_rate,
            auroc, aucpr,
            balanced_accuracy, precision, recall, f1,
            threshold, tp, fp, tn, fn,
            prediction_bias  (mean predicted prob - base rate)
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for metrics. "
            "Install it with: pip install scikit-learn"
        )

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_prob length ({len(y_prob)})"
        )

    if len(y_true) == 0:
        raise ValueError("Empty inputs.")

    y_pred = (y_prob >= threshold).astype(int)

    n = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n - n_positive
    base_rate = n_positive / n

    # Threshold-independent metrics
    auroc = roc_auc_score(y_true, y_prob)
    aucpr = average_precision_score(y_true, y_prob)

    # Threshold-dependent metrics
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*y_pred contains classes not in y_true.*"
        )
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Prediction bias: a positive value means the model over-predicts success
    prediction_bias = float(y_prob.mean()) - base_rate

    # AP@K metrics (primary per Liu et al.)
    ap_metrics = compute_ap_at_k(y_true, y_prob, k_values=[10, 20, 30])

    return {
        "n": n,
        "n_positive": n_positive,
        "n_negative": n_negative,
        "base_rate": round(base_rate, 4),
        # Primary metrics
        "ap_10": ap_metrics["ap_10"],
        "ap_20": ap_metrics["ap_20"],
        "ap_30": ap_metrics["ap_30"],
        # Secondary metrics
        "auroc": round(auroc, 4),
        "aucpr": round(aucpr, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "threshold": threshold,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "prediction_bias": round(prediction_bias, 4),
    }


def print_metrics(metrics: dict, label: str = "Results") -> None:
    """Pretty-print a metrics dictionary."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Dataset:   n={metrics['n']}  "
          f"(+:{metrics['n_positive']}  -:{metrics['n_negative']}  "
          f"base rate: {metrics['base_rate']:.1%})")
    print(f"\n  PRIMARY METRICS (AP@K per Liu et al.):")
    print(f"    AP@10:     {metrics['ap_10']:.4f}")
    print(f"    AP@20:     {metrics['ap_20']:.4f}")
    print(f"    AP@30:     {metrics['ap_30']:.4f}")
    print(f"\n  SECONDARY METRICS:")
    print(f"    AUROC:     {metrics['auroc']:.4f}")
    print(f"    AUCPR:     {metrics['aucpr']:.4f}")
    print(f"    Bal. Acc:  {metrics['balanced_accuracy']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    print(f"\n  THRESHOLD: {metrics['threshold']}")
    print(f"    TP={metrics['tp']}  FP={metrics['fp']}  "
          f"TN={metrics['tn']}  FN={metrics['fn']}")
    bias = metrics['prediction_bias']
    bias_dir = "over-predicts" if bias > 0 else "under-predicts"
    print(f"  Pred. bias:{bias:+.4f}  ({bias_dir} success)")
    print(f"{'='*60}\n")
