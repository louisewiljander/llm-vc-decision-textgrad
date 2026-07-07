"""
End-to-end pipeline. Runs all four ablation conditions and TextGrad optimization.

    python experiments/run_experiments.py
    python experiments/run_experiments.py --exclude 1 2        # skip steps 1 and 2
    python experiments/run_experiments.py --exclude 4 5        # skip TextGrad steps
    python experiments/run_experiments.py --seeds 42 123 456   # run three seeds

Edit the RUN CONFIGURATION block below to control sample sizes and training steps.
For more granular control, call run_ablation.py and run_textgrad.py directly.
"""
import argparse
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data_splits import get_splits
from src.utils.archive import make_run_dir

# Fixed seed for dataset splitting — decoupled from the training/inference seed.
# All experiment seeds evaluate on the identical train/val/test partition.
SPLIT_SEED = 42
PY = sys.executable
ABLATION = str(REPO_ROOT / "experiments" / "run_ablation.py")
TEXTGRAD = str(REPO_ROOT / "experiments" / "run_textgrad.py")
JUDGE    = str(REPO_ROOT / "experiments" / "run_judge_evaluation.py")

# ── GOOGLE DRIVE SYNC ─────────────────────────────────────────────────────────
# Set to your Drive results path when running on Colab; None to skip.
# Can also be set via the DRIVE_RESULTS environment variable.
import os as _os
DRIVE_RESULTS = _os.environ.get("DRIVE_RESULTS", None)

def print_split_verification(seed: int) -> None:
    df_train, df_val, df_test = get_splits(random_state=SPLIT_SEED)
    split_map = {"train": df_train, "val": df_val, "test": df_test}
    df = split_map[SPLIT]
    if SAMPLE is not None:
        df = df.sample(min(SAMPLE, len(df)), random_state=seed)
    invest = df[df["target"] == 1]
    pass_  = df[df["target"] == 0]
    print(f"  Split : {SPLIT}  |  seed={seed}  |  n={len(df)} ({len(invest)} INVEST / {len(pass_)} PASS)")
    print(f"  {'object_id':<14}  {'name':<40}  label")
    print(f"  {'-'*14}  {'-'*40}  -----")
    for _, row in df.sort_values("target", ascending=False).iterrows():
        label = "INVEST" if row["target"] == 1 else "PASS"
        print(f"  {str(row['object_id']):<14}  {str(row['name']):<40}  {label}")


def sync_to_drive(step_label: str) -> None:
    if not DRIVE_RESULTS:
        return
    src = REPO_ROOT / "results"
    dst = Path(DRIVE_RESULTS)
    try:
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"  ✓ Synced results to Drive after: {step_label}")
    except Exception as e:
        print(f"  ⚠  Drive sync failed ({e}) — continuing anyway")
# ──────────────────────────────────────────────────────────────────────────────

# ── RUN CONFIGURATION ─────────────────────────────────────────────────────────
MODEL  = "ollama/glm4:9b"
SPLIT  = "val"    # "train" | "val" | "test"
SAMPLE = None       # rows per ablation condition (None = full split)
N_TRAIN = 3       # TextGrad training steps
N_VAL   = 100      # examples evaluated after each TextGrad step
JUDGE_MODEL      = "groq/llama-3.3-70b-versatile"
N_JUDGE_SAMPLE   = 10      # startups to judge across all three conditions
JUDGE_SLEEP      = 65      # seconds between judge calls (Groq free tier: ~6K TPM)
# ──────────────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Run end-to-end experiment pipeline.")
parser.add_argument(
    "--exclude", nargs="+", type=int, metavar="N",
    help="Step numbers to skip (1-6). E.g. --exclude 1 2",
    default=[],
)
parser.add_argument(
    "--seeds", nargs="+", type=int, metavar="SEED",
    help="One or more random seeds to run (default: 42). Multiple seeds run sequentially. "
         "E.g. --seeds 42 123 456",
    default=[42],
)
parser.add_argument(
    "--split", choices=["val", "test"], default=None,
    help="Override the SPLIT variable (val or test). Defaults to the value in RUN CONFIGURATION.",
)
args = parser.parse_args()
excluded_steps = set(args.exclude)
seeds = args.seeds
if args.split:
    SPLIT = args.split

_sample_args = ["--sample", str(SAMPLE)] if SAMPLE is not None else []

for seed_idx, seed in enumerate(seeds):
    RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_s{seed}"
    multi_seed = len(seeds) > 1
    seed_label = f"Seed {seed_idx + 1}/{len(seeds)} (seed={seed})"
    if multi_seed:
        print(f"\n{'#'*60}\n  {seed_label}\n{'#'*60}")

    print(f"\n{'─'*60}")
    print(f"  Split verification (ablation steps 1–3, 5)")
    print(f"{'─'*60}")
    print_split_verification(seed)
    print(f"{'─'*60}\n")

    # Create run directories upfront so all sub-scripts share the same timestamp.
    # Seed is embedded in RUN_TIMESTAMP (_s{seed}), so runs from different seeds
    # never collide even when writing to the same top-level results/ dirs.
    abl_run_dir = make_run_dir(REPO_ROOT / "results" / "ablation",         timestamp=RUN_TIMESTAMP)
    jdg_run_dir = make_run_dir(REPO_ROOT / "results" / "judge_evaluation", timestamp=RUN_TIMESTAMP)

    # Create a textgrad run dir for this seed.
    # When training (step 4) runs, the script writes its output here.
    # When training is skipped, we create the dir and immediately populate it
    # with this seed's backed-up prompt — so `latest` always points to the
    # correct prompt for the current seed, without clobbering other seeds.
    tg_run_dir = make_run_dir(REPO_ROOT / "results" / "textgrad_validation", timestamp=RUN_TIMESTAMP)
    if 4 in excluded_steps:
        backup = REPO_ROOT / "results" / "textgrad_validation" / f"prompt_seed_{seed}.txt"
        if backup.exists():
            shutil.copy(backup, tg_run_dir / "final_synthesizer_prompt.txt")
            print(f"  ✓ Seeded run dir with prompt for seed {seed} → {tg_run_dir.name}/")
        else:
            print(f"  ⚠  No prompt backup for seed {seed} — TextGrad eval will use whatever is in latest/")

    STEPS = [
        (1, "Step 1/6 — Random baseline",     [PY, ABLATION, "--ablation", "random",   "--split", SPLIT, "--seed", str(seed), "--output_dir", str(abl_run_dir)] + _sample_args),
        (2, "Step 2/6 — Single agent",        [PY, ABLATION, "--ablation", "single",   "--split", SPLIT, "--model", MODEL, "--seed", str(seed), "--output_dir", str(abl_run_dir)] + _sample_args),
        (3, "Step 3/6 — Multi-analyst",       [PY, ABLATION, "--ablation", "multi",    "--split", SPLIT, "--model", MODEL, "--seed", str(seed), "--output_dir", str(abl_run_dir)] + _sample_args),
        (4, "Step 4/6 — TextGrad training",   [PY, TEXTGRAD, "--n_train", str(N_TRAIN), "--n_val", str(N_VAL), "--seed", str(seed), "--output_dir", str(tg_run_dir)]),
        (5, "Step 5/6 — TextGrad evaluation", [PY, ABLATION, "--ablation", "textgrad", "--split", SPLIT, "--model", MODEL, "--seed", str(seed), "--output_dir", str(abl_run_dir)] + _sample_args),
        (6, "Step 6/6 — Judge evaluation",    [PY, JUDGE, "--n_sample", str(N_JUDGE_SAMPLE), "--judge_model", JUDGE_MODEL, "--judge_sleep", str(JUDGE_SLEEP), "--seed", str(seed), "--output_dir", str(jdg_run_dir), "--ablation_dir", str(abl_run_dir)]),
    ]

    for step_num, label, cmd in STEPS:
        if step_num in excluded_steps:
            print(f"\n{'='*60}\n  {label}  [SKIPPED]\n{'='*60}\n")
            continue
        print(f"\n{'='*60}\n  {label}\n{'='*60}\n")

        # Before TextGrad eval (step 5), restore the seed's trained prompt.
        # Always write to the flat location — TextGradSynthesizer checks this
        # first when `latest` doesn't exist, and as a fallback when it does.
        # We avoid touching `latest` to prevent clobbering another seed's folder.
        if step_num == 5:
            backup = REPO_ROOT / "results" / "textgrad_validation" / f"prompt_seed_{seed}.txt"
            if backup.exists():
                flat = REPO_ROOT / "results" / "textgrad_validation" / "final_synthesizer_prompt.txt"
                shutil.copy(backup, flat)
                print(f"  ✓ Restored prompt for seed {seed} → {flat.name}")
            else:
                print(f"  ⚠  No prompt backup found for seed {seed} — using whatever is in latest/")

        result = subprocess.run(cmd, cwd=REPO_ROOT)

        # After TextGrad training (step 4), back up the trained prompt so it
        # isn't overwritten when the next seed's training runs.
        if step_num == 4 and result.returncode == 0:
            src = REPO_ROOT / "results" / "textgrad_validation" / "latest" / "final_synthesizer_prompt.txt"
            if not src.exists():
                src = REPO_ROOT / "results" / "textgrad_validation" / "final_synthesizer_prompt.txt"
            backup = REPO_ROOT / "results" / "textgrad_validation" / f"prompt_seed_{seed}.txt"
            if src.exists():
                shutil.copy(src, backup)
                print(f"  ✓ Prompt backed up to {backup.name}")

        if result.returncode != 0:
            print(f"\n✗ Failed at: {label}" + (f" ({seed_label})" if multi_seed else ""))
            sync_to_drive(f"{label} (partial — failed)")
            sys.exit(result.returncode)
        sync_to_drive(label)

seeds_str = ", ".join(f"s{s}" for s in seeds)
print(f"\n✓ All steps complete. Run timestamp: {RUN_TIMESTAMP}")
print(f"  results/ablation/runs/{RUN_TIMESTAMP}/")
print(f"  results/textgrad_validation/runs/{RUN_TIMESTAMP}/")
print(f"  results/judge_evaluation/runs/{RUN_TIMESTAMP}/")
