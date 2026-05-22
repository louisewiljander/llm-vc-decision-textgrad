"""
Average Precision at K (AP@K) metric for ranked list evaluation.

Follows Liu et al. (2025) definition: AP@K measures the average precision
of relevant items (successful startups) within the top-K recommendations,
providing a VC-focused metric that directly reflects investment utility.

For a ranked list of length N:
  - Sort by predicted probability (descending)
  - For each relevant item at rank i (where i <= K):
    - Precision@i = (# relevant items in top-i) / i
  - AP@K = (sum of precisions where item is relevant) / min(K, # relevant items)

This captures: "Of the top-K companies I'd investigate, how many actually succeeded?"
"""
import numpy as np
from typing import Optional


def compute_ap_at_k(
    y_true: list | np.ndarray,
    y_prob: list | np.ndarray,
    k_values: Optional[list] = None,
) -> dict:
    """
    Compute Average Precision at K for multiple K values.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Model-predicted probabilities (0–1).
        k_values: List of K values to compute AP@K for.
                  Defaults to [10, 20, 30] per Liu et al.

    Returns:
        Dictionary with keys: 'ap_10', 'ap_20', 'ap_30', 'ap_at_k_dict',
        'precisions_at_ranks' (for debugging).
    """
    if k_values is None:
        k_values = [10, 20, 30]

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)

    if len(y_true) != len(y_prob):
        raise ValueError(
            f"y_true length ({len(y_true)}) != y_prob length ({len(y_prob)})"
        )

    # Sort by predicted probability (descending)
    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    # Total number of relevant (positive) items in dataset
    n_relevant_total = y_true.sum()

    # Compute precision at each rank where item is relevant
    precisions_at_ranks = []
    ap_dict = {}

    for k in k_values:
        # Truncate to top-K
        y_true_at_k = y_true_sorted[:k]
        n_relevant_in_k = y_true_at_k.sum()

        if n_relevant_total == 0:
            # No positive examples in dataset — AP@K undefined, set to 0
            ap_at_k = 0.0
        elif n_relevant_in_k == 0:
            # No positive examples in top-K — precision is 0
            ap_at_k = 0.0
        else:
            # Sum precision at each rank where an item is relevant
            precision_sum = 0.0
            for rank, is_relevant in enumerate(y_true_at_k, start=1):
                if is_relevant:
                    precision_at_rank = (y_true_at_k[:rank].sum()) / rank
                    precision_sum += precision_at_rank

            # AP@K = average over min(K, # relevant items)
            ap_at_k = precision_sum / min(k, n_relevant_total)

        ap_dict[f"ap_{k}"] = round(ap_at_k, 4)
        precisions_at_ranks.append((k, ap_at_k, n_relevant_in_k))

    result = {
        "ap_10": ap_dict.get("ap_10", 0.0),
        "ap_20": ap_dict.get("ap_20", 0.0),
        "ap_30": ap_dict.get("ap_30", 0.0),
        "ap_at_k_dict": ap_dict,
        "precisions_at_ranks": precisions_at_ranks,  # For debugging
        "n_relevant_total": int(n_relevant_total),
    }

    return result


def print_ap_at_k(ap_metrics: dict, label: str = "AP@K Metrics") -> None:
    """Pretty-print AP@K metrics."""
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Total relevant items: {ap_metrics['n_relevant_total']}")
    for k, ap in ap_metrics["ap_at_k_dict"].items():
        print(f"  {k.upper()}: {ap:.4f}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Quick test with synthetic data
    # 100 samples: 10 positive, 90 negative
    y_true_test = np.array([1] * 10 + [0] * 90)
    # Perfect ranking: all positives at top
    y_prob_test = np.concatenate([
        np.linspace(1.0, 0.9, 10),  # Top 10 are positive
        np.linspace(0.89, 0.0, 90)  # Bottom 90 are negative
    ])

    metrics = compute_ap_at_k(y_true_test, y_prob_test)
    print_ap_at_k(metrics, "Test: Perfect Ranking (all positives at top)")

    # Test 2: Random ranking
    y_true_random = np.array([1] * 10 + [0] * 90)
    np.random.seed(42)
    y_prob_random = np.random.uniform(0, 1, 100)

    metrics_random = compute_ap_at_k(y_true_random, y_prob_random)
    print_ap_at_k(metrics_random, "Test: Random Ranking")

    # Test 3: Worst ranking (all positives at bottom)
    y_prob_worst = np.concatenate([
        np.linspace(0.89, 0.0, 90),  # Bottom 90 are negative
        np.linspace(1.0, 0.9, 10),   # Top 10 are positive (but predicted low)
    ])
    y_true_worst = np.array([0] * 90 + [1] * 10)

    metrics_worst = compute_ap_at_k(y_true_worst, y_prob_worst)
    print_ap_at_k(metrics_worst, "Test: Worst Ranking (all positives at bottom)")
