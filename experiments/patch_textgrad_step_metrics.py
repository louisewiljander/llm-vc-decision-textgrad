"""
patch_textgrad_step_metrics.py
==============================
Patches precision and recall into val_metrics records in metrics_per_step.jsonl
by re-running synthesizer inference on the validation set for each saved prompt.

Run once per TextGrad seed run on Colab (where Ollama is available).

Usage:
    python experiments/patch_textgrad_step_metrics.py \
        --run_dir  /content/drive/MyDrive/.../textgrad/2026-06-21_21-12-56_s42 \
        --cache_dir /content/drive/MyDrive/.../textgrad_validation/cached_assessments \
        --dry_run   # optional: check cache coverage without running inference

The script is idempotent — re-running it on an already-patched file is safe.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests


# ─── OLLAMA ──────────────────────────────────────────────────────────────────

def call_ollama(
    system_prompt: str,
    user_message: str,
    url: str,
    model: str,
    max_retries: int = 3,
) -> str:
    endpoint = f"{url}/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    for attempt in range(max_retries):
        try:
            r = requests.post(endpoint, json=payload, timeout=120)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(5)
            else:
                raise RuntimeError(
                    f"Ollama call failed after {max_retries} attempts: {e}"
                )


# ─── METRICS ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: list, y_prob: list, threshold: float = 0.5) -> dict:
    from sklearn.metrics import (
        roc_auc_score,
        balanced_accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    sorted_idx    = np.argsort(-y_prob)
    y_true_sorted = y_true[sorted_idx]

    return {
        "p_10":              round(float(y_true_sorted[:10].sum()) / 10, 4),
        "p_20":              round(float(y_true_sorted[:20].sum()) / 20, 4),
        "p_30":              round(float(y_true_sorted[:30].sum()) / 30, 4),
        "auroc":             round(float(roc_auc_score(y_true, y_prob)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "precision":         round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall":            round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1":                round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "n_evaluated":       int(len(y_true)),
    }


# ─── EVALUATION ──────────────────────────────────────────────────────────────

def evaluate_prompt_on_val_set(
    prompt: str,
    val_records: list,
    ollama_url: str,
    model: str,
) -> dict:
    """Run synthesizer with the given prompt on all val cached assessments."""
    analyst_order = ["Market", "Business Model", "Feasibility", "Team"]
    y_true, y_prob = [], []
    errors = 0

    for i, rec in enumerate(val_records):
        analyst_report = "\n".join(
            f"{name} Analyst:\n{json.dumps(rec['analyst_assessments'][name], indent=2)}"
            for name in analyst_order
        )
        user_msg = (
            f"Startup: {rec['name']}\n\n"
            f"Profile:\n{rec['startup_profile']}\n\n"
            f"Analyst Reports:\n{analyst_report}\n\n"
            f"Based on the above, give your final investment decision."
        )

        try:
            output = call_ollama(prompt, user_msg, ollama_url, model)
            match  = re.search(r'"probability"\s*:\s*(\d+)', output)
            prob   = int(match.group(1)) / 100.0 if match else 0.5
        except Exception as e:
            print(f"  Warning: failed for {rec['name']}: {e}")
            prob = 0.5
            errors += 1

        y_prob.append(prob)
        y_true.append(int(rec["target"]))

        if (i + 1) % 20 == 0:
            print(f"  [{i + 1}/{len(val_records)}]")

    print(f"  Done — {len(y_true)} evaluated, {errors} errors")
    return compute_metrics(y_true, y_prob)


# ─── CACHE RESOLUTION ────────────────────────────────────────────────────────

def resolve_cache_dir(run_dir: Path, explicit: Path | None) -> Path:
    """Find the cached_assessments directory, checking several candidate locations."""
    candidates = []
    if explicit:
        candidates.append(explicit)
    candidates += [
        run_dir / "cached_assessments",
        run_dir.parent / "cached_assessments",
        run_dir.parent.parent / "cached_assessments",
    ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    raise FileNotFoundError(
        f"Could not find a non-empty cached_assessments directory near {run_dir}.\n"
        f"Tried: {candidates}\n"
        f"Pass --cache_dir explicitly."
    )


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir", required=True,
        help="Path to the seed's textgrad run directory (contains prompt_step_*.txt and metrics_per_step.jsonl).",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="Path to cached_assessments/ folder. Auto-detected from run_dir if omitted.",
    )
    parser.add_argument(
        "--ollama_url", default="http://localhost:11434",
        help="Ollama base URL (default: http://localhost:11434).",
    )
    parser.add_argument(
        "--model", default="glm4:9b",
        help="Model name for Ollama (default: glm4:9b). Omit the 'ollama/' prefix.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Check cache coverage without running inference.",
    )
    args = parser.parse_args()

    run_dir   = Path(args.run_dir)
    cache_dir = resolve_cache_dir(
        run_dir, Path(args.cache_dir) if args.cache_dir else None
    )

    print(f"Run dir:   {run_dir}")
    print(f"Cache dir: {cache_dir}")
    print(f"Model:     {args.model} @ {args.ollama_url}")

    # Load val IDs from data_splits.json
    splits_file = run_dir / "data_splits.json"
    if not splits_file.exists():
        sys.exit(f"ERROR: {splits_file} not found.")
    splits  = json.loads(splits_file.read_text())
    val_ids = [str(i) for i in splits["val_object_ids"]]
    print(f"\nVal set: {len(val_ids)} startups (seed={splits.get('seed')})")

    # Check cache coverage
    missing = [i for i in val_ids if not (cache_dir / f"{i}.json").exists()]
    if missing:
        print(f"WARNING: {len(missing)}/{len(val_ids)} val IDs not in cache.")
        print(f"  First missing: {missing[:5]}")
    else:
        print(f"✓ All {len(val_ids)} val IDs cached.")

    if args.dry_run:
        print("\nDry run complete — no inference run.")
        return

    # Load cached assessments
    val_records = [
        json.loads((cache_dir / f"{oid}.json").read_text())
        for oid in val_ids
    ]

    # Load existing JSONL
    jsonl_path = run_dir / "metrics_per_step.jsonl"
    if not jsonl_path.exists():
        sys.exit(f"ERROR: {jsonl_path} not found.")
    records = [
        json.loads(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]

    # Find steps that have val_metrics
    val_step_indices = [
        (i, r["step"])
        for i, r in enumerate(records)
        if "val_metrics" in r
    ]
    print(f"\nSteps with val_metrics: {[s for _, s in val_step_indices]}")

    # Re-evaluate each step
    for record_idx, step in val_step_indices:
        prompt_file = run_dir / f"prompt_step_{step}.txt"
        if not prompt_file.exists():
            print(f"\nStep {step}: prompt_step_{step}.txt not found — skipping.")
            continue

        # Skip if already patched
        existing = records[record_idx]["val_metrics"]
        if "precision" in existing and "recall" in existing:
            print(f"\nStep {step}: already has precision/recall — skipping.")
            continue

        print(f"\nStep {step}: evaluating {len(val_records)} val startups...")
        prompt  = prompt_file.read_text()
        metrics = evaluate_prompt_on_val_set(
            prompt, val_records, args.ollama_url, args.model
        )

        print(
            f"  precision={metrics['precision']}  "
            f"recall={metrics['recall']}  "
            f"f1={metrics['f1']}  "
            f"p_10={metrics['p_10']}"
        )

        # Patch in place
        records[record_idx]["val_metrics"]["precision"] = metrics["precision"]
        records[record_idx]["val_metrics"]["recall"]    = metrics["recall"]

    # Write back
    jsonl_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n"
    )
    print(f"\n✓ Patched {jsonl_path}")


if __name__ == "__main__":
    main()
