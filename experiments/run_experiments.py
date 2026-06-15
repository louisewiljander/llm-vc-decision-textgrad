"""
End-to-end pipeline. Runs all four ablation conditions and TextGrad optimization.

    python experiments/run_experiments.py

Edit the RUN CONFIGURATION block below to control sample sizes and training steps.
For more granular control, call run_ablation.py and run_textgrad.py directly.
"""
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
ABLATION = str(REPO_ROOT / "experiments" / "run_ablation.py")
TEXTGRAD = str(REPO_ROOT / "experiments" / "run_textgrad.py")

# ── RUN CONFIGURATION ─────────────────────────────────────────────────────────
MODEL  = "ollama/glm4:latest"
SPLIT  = "val"    # "train" | "val" | "test"
SAMPLE = 30       # rows per ablation condition (None = full split)
N_TRAIN = 4       # TextGrad training steps
N_VAL   = 30      # examples evaluated after each TextGrad step
# ──────────────────────────────────────────────────────────────────────────────

_sample_args = ["--sample", str(SAMPLE)] if SAMPLE is not None else []

STEPS = [
    ("Step 1/5 — Random baseline",  [PY, ABLATION, "--ablation", "random",   "--split", SPLIT] + _sample_args),
    ("Step 2/5 — Single agent",     [PY, ABLATION, "--ablation", "single",   "--split", SPLIT, "--model", MODEL] + _sample_args),
    ("Step 3/5 — Multi-analyst",    [PY, ABLATION, "--ablation", "multi",    "--split", SPLIT, "--model", MODEL] + _sample_args),
    ("Step 4/5 — TextGrad training",[PY, TEXTGRAD, "--n_train", str(N_TRAIN), "--n_val", str(N_VAL)]),
    ("Step 5/5 — TextGrad evaluation", [PY, ABLATION, "--ablation", "textgrad", "--split", SPLIT, "--model", MODEL] + _sample_args),
]

for label, cmd in STEPS:
    print(f"\n{'='*60}\n  {label}\n{'='*60}\n")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        print(f"\n✗ Failed at: {label}")
        sys.exit(result.returncode)

print("\n✓ All steps complete. Results in results/ablation/ and results/textgrad_validation/\n")
