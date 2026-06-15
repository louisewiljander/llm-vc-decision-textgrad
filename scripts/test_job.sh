#!/bin/bash
#SBATCH --job-name=vc-test
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
# Note: check available partitions with `sinfo` and set one below
##SBATCH --partition=cm4_tiny   # uncomment and adjust to your partition

echo "==== Job started: $(date) ===="
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# ── Environment ──────────────────────────────────────────────────────────────
module load python/3.11   # adjust version based on `module avail python`

REPO_ROOT="$HOME/llm-vc-decision-textgrad"
cd "$REPO_ROOT" || { echo "ERROR: repo not found at $REPO_ROOT"; exit 1; }

source .venv/bin/activate
echo "Python: $(python --version)"

# ── Random baseline (no API call) ────────────────────────────────────────────
echo ""
echo "=== Random baseline (5 rows, no API) ==="
python experiments/run_ablation.py \
    --ablation random \
    --split val \
    --sample 5

echo ""
echo "==== Job finished: $(date) ===="
