"""
Patch existing metrics JSON files to add weighted_f1.

Scans results/ablation/runs/ for *_predictions.jsonl files,
computes weighted F1 from the saved predictions, and writes
it into the corresponding *_metrics.json file in-place.

Run from the repo root:
    python experiments/patch_weighted_f1.py
    python experiments/patch_weighted_f1.py --split val   # val only
    python experiments/patch_weighted_f1.py --split test  # test only (default)
"""
import argparse
import json
from pathlib import Path

from sklearn.metrics import f1_score

RUNS_DIR = Path(__file__).resolve().parents[1] / "results" / "ablation" / "runs"


def compute_weighted_f1(pred_file: Path) -> float:
    y_true, y_pred = [], []
    with open(pred_file) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            y_true.append(int(rec["target"]))
            y_pred.append(1 if rec.get("decision", "").upper() == "INVEST" else 0)
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


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
                # Derive the corresponding metrics filename
                metrics_file = pred_file.with_name(
                    pred_file.name.replace("_predictions.jsonl", "_metrics.json")
                )
                if not metrics_file.exists():
                    print(f"  ⚠  No metrics file for {pred_file.name} — skipping")
                    skipped += 1
                    continue

                metrics = json.loads(metrics_file.read_text())
                if "weighted_f1" in metrics:
                    print(f"  ✓ Already has weighted_f1: {metrics_file.parent.name}/{metrics_file.name}")
                    skipped += 1
                    continue

                wf1 = compute_weighted_f1(pred_file)
                metrics["weighted_f1"] = round(wf1, 4)
                metrics_file.write_text(json.dumps(metrics, indent=2))
                print(f"  ✓ Patched {metrics_file.parent.name}/{metrics_file.name}  weighted_f1={wf1:.4f}")
                patched += 1

    print(f"\nDone — {patched} files patched, {skipped} skipped.")


if __name__ == "__main__":
    main()
