#!/bin/bash
#SBATCH --job-name=vc-small-exp
#SBATCH --output=logs/slurm/%j.out
#SBATCH --error=logs/slurm/%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
##SBATCH --partition=cm4_tiny   # uncomment and set after running `sinfo`

echo "==== Job started: $(date) ===="
echo "Node: $(hostname) | Job ID: $SLURM_JOB_ID"

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11   # adjust based on `module avail python`

REPO_ROOT="$HOME/llm-vc-decision-textgrad"
cd "$REPO_ROOT" || { echo "ERROR: repo not found at $REPO_ROOT"; exit 1; }

# Activate virtual environment
source .venv/bin/activate
echo "Python: $(python --version)"

# ── Groq API key check ────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    echo "ERROR: .env not found. Create it with GROQ_API_KEY=gsk_..."
    exit 1
fi
source .env
if [ -z "$GROQ_API_KEY" ]; then
    echo "ERROR: GROQ_API_KEY not set in .env"
    exit 1
fi
echo "✓ GROQ_API_KEY loaded"

# ── Start Ollama server ───────────────────────────────────────────────────────
export OLLAMA_HOME="$HOME/.ollama"
export PATH="$HOME/bin:$PATH"   # Ollama binary location (see setup steps)

echo ""
echo "=== Starting Ollama server ==="
ollama serve &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama ready after ${i}s"
        break
    fi
    sleep 2
done

# Pull model if not already cached
echo ""
echo "=== Pulling GLM-4 model (skips if already cached) ==="
ollama pull glm4:latest
echo "✓ Model ready"

# ── Experiment configuration ──────────────────────────────────────────────────
SAMPLE=5         # rows per ablation condition
N_TRAIN=3        # TextGrad training steps
N_VAL=3          # validation examples per TextGrad step
SPLIT="val"
MODEL="ollama/glm4:latest"

echo ""
echo "=== Configuration: sample=$SAMPLE, n_train=$N_TRAIN, n_val=$N_VAL ==="

# ── Step 1: Random baseline ───────────────────────────────────────────────────
echo ""
echo "=== Step 1/5: Random baseline ==="
python experiments/run_ablation.py \
    --ablation random \
    --split $SPLIT \
    --sample $SAMPLE

# ── Step 2: Single agent ──────────────────────────────────────────────────────
echo ""
echo "=== Step 2/5: Single agent (GLM-4) ==="
python experiments/run_ablation.py \
    --ablation single \
    --model $MODEL \
    --split $SPLIT \
    --sample $SAMPLE

# ── Step 3: Multi-analyst ─────────────────────────────────────────────────────
echo ""
echo "=== Step 3/5: Multi-analyst (GLM-4) ==="
python experiments/run_ablation.py \
    --ablation multi \
    --model $MODEL \
    --split $SPLIT \
    --sample $SAMPLE

# ── Step 4: TextGrad training (GLM-4 forward, Groq backward) ─────────────────
echo ""
echo "=== Step 4/5: TextGrad training ==="
python experiments/run_textgrad.py \
    --n_train $N_TRAIN \
    --n_val $N_VAL \
    --validate_every 1

# ── Step 5: TextGrad ablation (evaluate optimized prompt) ────────────────────
echo ""
echo "=== Step 5/5: TextGrad ablation ==="
python experiments/run_ablation.py \
    --ablation textgrad \
    --model $MODEL \
    --split $SPLIT \
    --sample $SAMPLE

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo ""
echo "=== Shutting down Ollama ==="
kill $OLLAMA_PID 2>/dev/null
wait $OLLAMA_PID 2>/dev/null

echo ""
echo "==== Job finished: $(date) ===="
