"""
Re-evaluate TextGrad step prompts on the validation set and patch metrics.

The original training logged ap_10, ap_20, ap_30, weighted_f1 per step.
This script re-runs synthesizer inference for each saved prompt_step_N.txt
and rewrites the val_metrics block with p_10, p_20, p_30, f1, auroc, balanced_accuracy.

Typical usage (on Colab with Ollama running):
  # Per-seed run:
  python experiments/patch_textgrad_step_metrics.py \
      --run_dir results/textgrad_validation/runs/2026-06-21_21-12-56_s42

  # Top-level run (cache is in the same dir):
  python experiments/patch_textgrad_step_metrics.py \
      --run_dir results/textgrad_validation

  # All canonical runs at once:
  python experiments/patch_textgrad_step_metrics.py \
      --run_dir results/textgrad_validation/runs/2026-06-21_21-12-56_s42 \
                results/textgrad_validation/runs/2026-06-22_20-41-54_s123

  # Dry run — print what would happen, no writes:
  python experiments/patch_textgrad_step_metrics.py \
      --run_dir results/textgrad_validation --dry_run

Requirements:
  pip install requests numpy scikit-learn
  Ollama must be running with glm4:9b pulled.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.metrics import compute_metrics

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "glm4:9b"
K_VALUES = [10, 20, 30]
ANALYST_NAMES = ["Market", "Business Model", "Feasibility", "Team"]


# ─── INFERENCE ────────────────────────────────────────────────────────────────

def call_synthesizer(
    system_prompt: str,
    startup_name: str,
    startup_profile: str,
    analyst_assessments: dict,
    ollama_url: str,
    model: str,
    timeout: int = 60,
) -> float:
    """
    Run one synthesizer inference call and return probability (0–1).

    Replicates the exact input format used in run_textgrad.py's
    evaluate_synthesizer_on_val_set().

    Returns 0.5 on failure (neutral, so as not to bias ranking).
    """
    analyst_report = "\n".join(
        f"{name} Analyst:\n{json.dumps(analyst_assessments[name], indent=2)}"
        for name in ANALYST_NAMES
        if name in analyst_assessments
    )
    user_message = (
        f"Startup: {startup_name}\n\n"
        f"Profile:\n{startup_profile}\n\n"
        f"Analyst Reports:\n{analyst_report}\n\n"
        f"Based on the above, give your final investment decision."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{ollama_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        output = resp.json()["choices"][0]["message"]["content"]
        match = re.search(r'"probability"\s*:\s*(\d+)', output)
        if match:
            return int(match.group(1)) / 100.0
        else:
            print(f"    ⚠  No probability found in output: {output[:120]!r}")
            return 0.5
    except Exception as exc:
        print(f"    ⚠  Inference error: {exc}")
        return 0.5


# ─── CORE LOGIC PER STEP ──────────────────────────────────────────────────────

def evaluate_step(
    step: int,
    prompt: str,
    val_ids: list[str],
    cache_map: dict,
    ollama_url: str,
    model: str,
) -> dict:
    """
    Run synthesizer inference on all val examples with the given step prompt.
    Returns a metrics dict with p_10, p_20, p_30, f1, balanced_accuracy, auroc.
    """
    y_true, y_prob = [], []

    for i, oid in enumerate(val_ids):
        record = cache_map.get(oid)
        if record is None:
            print(f"    ⚠  No cached assessment for {oid}, skipping")
            continue

        prob = call_synthesizer(
            system_prompt=prompt,
            startup_name=record["name"],
            startup_profile=record["startup_profile"],
            analyst_assessments=record["analyst_assessments"],
            ollama_url=ollama_url,
            model=model,
        )
        y_true.append(record["target"])
        y_prob.append(prob)

        if (i + 1) % 10 == 0:
            print(f"    [{i+1}/{len(val_ids)}] evaluated")

    if len(y_true) < 10:
        raise ValueError(
            f"Too few valid examples ({len(y_true)}) to compute P@10"
        )

    m = compute_metrics(y_true, y_prob, threshold=0.5)
    return {
        "p_10": m["p_10"],
        "p_20": m["p_20"],
        "p_30": m["p_30"],
        "f1": m["f1"],
        "balanced_accuracy": m["balanced_accuracy"],
        "auroc": m["auroc"],
        "n_evaluated": m["n"],
    }


# ─── METRICS_PER_STEP PATCHING ────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def patch_run(
    run_dir: Path,
    cache_dir: Path,
    ollama_url: str,
    model: str,
    dry_run: bool,
) -> None:
    """
    Patch metrics_per_step.jsonl in run_dir with P@K and F1 per step.
    """
    print(f"\n{'='*70}")
    print(f"Run: {run_dir}")
    print(f"{'='*70}")

    # ── Load data_splits.json ─────────────────────────────────────────────────
    splits_file = run_dir / "data_splits.json"
    if not splits_file.exists():
        print(f"  ⚠  No data_splits.json — skipping")
        return
    splits = json.loads(splits_file.read_text())
    val_ids = splits.get("val_object_ids", [])
    print(f"  Seed: {splits.get('seed')}  |  val IDs: {len(val_ids)}")

    if len(val_ids) < 30:
        print(f"  ⚠  Too few val IDs ({len(val_ids)}) for P@30 — skipping")
        return

    # ── Load cached assessments ───────────────────────────────────────────────
    if not cache_dir.exists():
        print(f"  ✗  Cache dir not found: {cache_dir}")
        return

    cache_map = {}
    missing = []
    for oid in val_ids:
        # IDs look like "c:12345"; filenames are "c:12345.json"
        cache_file = cache_dir / f"{oid}.json"
        if cache_file.exists():
            cache_map[oid] = json.loads(cache_file.read_text())
        else:
            missing.append(oid)

    print(f"  Cached: {len(cache_map)}/{len(val_ids)}  |  Missing: {len(missing)}")
    if missing:
        print(f"    Missing IDs: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    # ── Find prompt_step files ────────────────────────────────────────────────
    prompt_files = sorted(run_dir.glob("prompt_step_*.txt"),
                          key=lambda p: int(p.stem.split("_")[-1]))
    if not prompt_files:
        print(f"  ⚠  No prompt_step_*.txt files found — skipping")
        return
    print(f"  Prompt steps found: {[p.stem for p in prompt_files]}")

    # ── Load existing metrics_per_step.jsonl ──────────────────────────────────
    metrics_file = run_dir / "metrics_per_step.jsonl"
    if not metrics_file.exists():
        print(f"  ⚠  No metrics_per_step.jsonl — skipping")
        return
    records = load_jsonl(metrics_file)

    # Build index of existing val_metric records by step
    val_record_idx: dict[int, int] = {}  # step → position in records list
    for i, rec in enumerate(records):
        if "val_metrics" in rec:
            val_record_idx[rec["step"]] = i

    # ── Process each step ────────────────────────────────────────────────────
    for prompt_file in prompt_files:
        step = int(prompt_file.stem.split("_")[-1])

        if step not in val_record_idx:
            print(f"\n  Step {step}: no existing val_metrics record — skipping")
            continue

        rec_idx = val_record_idx[step]
        existing_vm = records[rec_idx]["val_metrics"]

        if "p_10" in existing_vm and "ap_10" not in existing_vm:
            print(f"\n  Step {step}: already patched (has p_10, no ap_10) — skipping")
            continue

        print(f"\n  Step {step}: running inference on {len(cache_map)} val examples ...")
        if dry_run:
            print(f"    [DRY RUN] would call {len(val_ids)} synthesizer inferences")
            continue

        prompt = prompt_file.read_text()

        t0 = time.time()
        new_vm = evaluate_step(
            step=step,
            prompt=prompt,
            val_ids=val_ids,
            cache_map=cache_map,
            ollama_url=ollama_url,
            model=model,
        )
        elapsed = time.time() - t0

        # Preserve n_evaluated and add new metrics
        new_vm_full = {**existing_vm, **new_vm}

        records[rec_idx]["val_metrics"] = new_vm_full
        print(
            f"    ✓ Done in {elapsed:.0f}s  |  "
            f"p_10={new_vm['p_10']:.4f}  p_20={new_vm['p_20']:.4f}  "
            f"p_30={new_vm['p_30']:.4f}  f1={new_vm['f1']:.4f}  "
            f"auroc={new_vm['auroc']:.4f}"
        )

    if not dry_run:
        save_jsonl(metrics_file, records)
        print(f"\n  ✓ Saved patched metrics to {metrics_file}")
    else:
        print(f"\n  [DRY RUN] would write patched metrics to {metrics_file}")


# ─── CACHE DIR RESOLUTION ─────────────────────────────────────────────────────

def resolve_cache_dir(run_dir: Path, explicit: Path | None) -> Path:
    """
    Find cached_assessments relative to run_dir.
    Checks (in order):
      explicit arg
      → run_dir/cached_assessments             (run is top-level)
      → run_dir/../cached_assessments           (run_dir/runs/../)
      → run_dir/../../cached_assessments        (per-seed run inside runs/)
      → repo default

    For per-seed runs the layout is:
      textgrad_validation/
        cached_assessments/      ← here (grandparent of run_dir)
        runs/
          <run_dir>/
    """
    if explicit is not None:
        return explicit

    candidates = [
        run_dir / "cached_assessments",                     # top-level run
        run_dir.parent / "cached_assessments",              # sibling of runs/
        run_dir.parent.parent / "cached_assessments",       # grandparent (per-seed)
        REPO_ROOT / "results" / "textgrad_validation" / "cached_assessments",
    ]

    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            return candidate

    # If none found with files, return the first existing one (even if empty)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Patch TextGrad step metrics with P@K and F1.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run_dir", nargs="+", required=True,
        help="One or more run directories containing prompt_step_*.txt and "
             "metrics_per_step.jsonl. Paths relative to repo root or absolute.",
    )
    parser.add_argument(
        "--cache_dir", default=None,
        help="Path to cached_assessments/ directory. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--ollama_url", default=DEFAULT_OLLAMA_URL,
        help=f"Ollama base URL (default: {DEFAULT_OLLAMA_URL})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Ollama model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Print what would happen without making any API calls or writes.",
    )
    args = parser.parse_args()

    # ── Sanity-check Ollama connection ────────────────────────────────────────
    if not args.dry_run:
        try:
            r = requests.get(f"{args.ollama_url}/api/tags", timeout=5)
            r.raise_for_status()
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(args.model in m for m in models):
                print(
                    f"⚠  Model '{args.model}' not found in Ollama. "
                    f"Available: {models}\n"
                    f"Pull with: ollama pull {args.model}"
                )
                sys.exit(1)
            print(f"✓ Ollama reachable, model '{args.model}' available.\n")
        except Exception as exc:
            print(f"✗ Cannot reach Ollama at {args.ollama_url}: {exc}")
            sys.exit(1)

    # ── Resolve explicit cache_dir ────────────────────────────────────────────
    explicit_cache = Path(args.cache_dir) if args.cache_dir else None

    # ── Process each run dir ──────────────────────────────────────────────────
    for raw_path in args.run_dir:
        run_dir = Path(raw_path)
        if not run_dir.is_absolute():
            run_dir = REPO_ROOT / run_dir
        run_dir = run_dir.resolve()

        if not run_dir.exists():
            print(f"\n⚠  Run dir not found: {run_dir} — skipping")
            continue

        cache_dir = resolve_cache_dir(run_dir, explicit_cache)
        patch_run(
            run_dir=run_dir,
            cache_dir=cache_dir,
            ollama_url=args.ollama_url,
            model=args.model,
            dry_run=args.dry_run,
        )

    print("\nAll done.")


if __name__ == "__main__":
    main()
