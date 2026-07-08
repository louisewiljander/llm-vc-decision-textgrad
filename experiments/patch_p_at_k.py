"""
Patch existing metrics JSON files to add P@K (plain Precision at K).

P@K = (# relevant items in top-K) / K — comparable to Liu et al. (2025),
who report temporal Average P@K (P@K per period averaged over time).
Distinct from AP@K (Average Precision at K, Manning et al. 2008).

Scans results/ablation/runs/ for *_predictions.jsonl files, computes
P@10, P@20, P@30 from the saved predictions, and writes them into the
corresponding *_metrics.json file in-place.

Run from the repo root:
    python experiments/patch_p_at_k.py
    python experiments/patch_p_at_k.py --split val
    python experiments/patch_p_at_k.py --split test  # default
"""
import argparse
import json
from pathlib import Path

import numpy as np

RUNS_DIR = Path(__file__).resolve().parents[1] / "results" / "ablation" / "runs"
K_VALUES = [10, 20, 30]


def compute_p_at_k(pred_file: Path) -> dict:
    rows = []
    with open(pred_file) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    y_true = np.array([r["target"] for r in rows])
    y_prob = np.array([r["probability_float"] for r in rows])

    sorted_indices = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_indices]

    return {f"p_{k}": round(float(y_true_sorted[:k].sum()) / k, 4) for k in K_VALUES}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default=None, choices=["val", "test"],
                        help="Restrict to one split (default: both)")
    args = parser.parse_args()

    splits = [args.split] if args.split else ["val", "test"]
    patched = 0
    skipped = 0

    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        for split in splits:
            for pred_file in run_dir.glob(f"*_{split}_*_predictions.jsonl"):
                metrics_file = pred_file.with_name(
                    pred_file.name.replace("_predictions.jsonl", "_metrics.json")
                )
                if not metrics_file.exists():
                    print(f"  ⚠  No metrics file for {pred_file.name} — skipping")
                    skipped += 1
                    continue

                metrics = json.loads(metrics_file.read_text())
                if "p_10" in metrics:
                    print(f"  ✓ Already has P@K: {metrics_file.parent.name}/{metrics_file.name}")
                    skipped += 1
                    continue

                p_at_k = compute_p_at_k(pred_file)
                metrics.update(p_at_k)
                metrics_file.write_text(json.dumps(metrics, indent=2))
                vals = "  ".join(f"p_{k}={p_at_k[f'p_{k}']:.4f}" for k in K_VALUES)
                print(f"  ✓ Patched {metrics_file.parent.name}/{metrics_file.name}  {vals}")
                patched += 1

    print(f"\nDone — {patched} files patched, {skipped} skipped.")


if __name__ == "__main__":
    main()
