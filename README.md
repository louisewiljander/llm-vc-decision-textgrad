# LLM-based Multi-Agent VC Decision Simulation — TextGrad Optimization

An ablation study comparing LLM-based VC decision-making architectures, with prompt optimization via TextGrad. Built on the 2013 Crunchbase snapshot; evaluated against historical exit outcomes.

## Project Goals

- Compare four decision-making architectures (random baseline → single agent → multi-analyst → TextGrad-optimized)
- Optimize the synthesizer's system prompt using TextGrad gradient descent
- Evaluate reasoning quality using an LLM-as-judge grounded in Schumpeter innovation theory
- Produce reproducible results for a thesis comparing LLM agent architectures for startup evaluation

## Architecture

### Four Ablation Conditions

| Condition | Description |
|-----------|-------------|
| `random` | Uniform random predictions — lower bound baseline |
| `single` | Single `InvestorAgent` evaluates each startup end-to-end |
| `multi` | Four specialist analysts (parallel) → fixed `SynthesizerAgent` |
| `textgrad` | Four specialist analysts (parallel) → TextGrad-optimized `TextGradSynthesizer` |

### Multi-Analyst Pipeline (conditions `multi` and `textgrad`)

```
Startup Profile
      │
      ├──────────────────────────────────────────┐
      │ (parallel)                               │
      ▼                                          ▼
Market Analyst   Business Model Analyst   Feasibility Analyst   Team Analyst
      │                │                      │                 │
      └────────────────┴──────────────────────┴─────────────────┘
                                   │
                                   ▼
                        Synthesizer Agent  ← fixed prompt (multi)
                     TextGrad Synthesizer  ← optimized prompt (textgrad)
                                   │
                              INVEST / PASS
                          probability 0–100
```

### TextGrad Optimization

The synthesizer's system prompt is the learnable variable. TextGrad trains it on a set of analyst assessments using an LLM-as-judge loss:

```
Analyst assessments
        │
        ▼
  Synthesizer(system_prompt=tg.Variable, requires_grad=True)
        │
        ▼
  Binary decision + probability
        │
        ▼
  LLM Judge  ← compares decision to ground-truth label, writes textual gradient
        │
        ▼
  TextualGradientDescent → rewrites synthesizer system prompt
```

- **Forward model** (synthesizer): `ollama/glm4:9b` (local, free)
- **Backward model** (gradient generator): `groq/llama-3.3-70b-versatile` (stronger instruction-following)

### Judge Evaluation

Post-hoc qualitative scoring with `groq/llama-3.3-70b-versatile` as judge. Six dimensions, each scored 1–5:

1. Product novelty (Schumpeter type 1)
2. Market opportunity (Schumpeter type 3)
3. Feasibility (Schumpeter type 2)
4. Team quality (Gompers et al. 2020)
5. Reasoning coherence
6. Risk identification

## Quick Start

### 1. Setup

```bash
cd llm-vc-decision-textgrad

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 2. Configure API Keys

```bash
# .env file at repo root
GROQ_API_KEY=gsk_...          # required for judge + TextGrad backward pass
ANTHROPIC_API_KEY=sk-ant-...  # optional — if running Claude models via LiteLLM
```

Ollama must be running locally for `ollama/glm4:9b` (the default forward model):
```bash
ollama serve
ollama pull glm4
```

### 3. Run the Full Pipeline

```bash
# Run all 6 steps: random → single → multi → TextGrad train → TextGrad eval → judge
python experiments/run_experiments.py
```

Edit the `RUN CONFIGURATION` block at the top of `run_experiments.py` to control sample sizes, models, and step counts.

### 4. Run Individual Steps (Choose data split train/val/test as needed)

```bash
# Random baseline
python experiments/run_ablation.py --ablation random --split test

# Single agent (any LiteLLM-compatible model)
python experiments/run_ablation.py --ablation single --model ollama/glm4:9b --split test --sample None

# Multi-analyst pipeline
python experiments/run_ablation.py --ablation multi --model ollama/glm4:9b --split test

# TextGrad training
python experiments/run_textgrad.py --n_train 3 --n_val 100

# TextGrad ablation evaluation (uses optimized prompt from results/)
python experiments/run_ablation.py --ablation textgrad --model ollama/glm4:9b --split test

# LLM-as-judge evaluation
python experiments/run_judge_evaluation.py --n_sample 10 --judge_model groq/llama-3.3-70b-versatile
```

## Evaluation Metrics

Primary metric: **Average Precision at K** (AP@K, per Liu et al. 2026), capturing how well the model ranks successful startups in the top-K predictions.

| Metric | Description |
|--------|-------------|
| AP@10 / AP@20 / AP@30 | Primary: precision of top-K ranked predictions |
| AUROC | Threshold-independent discrimination |
| Balanced accuracy | Corrects for class imbalance |
| Precision / Recall / F1 | At default threshold of 0.5 |
| AUCPR | Area under precision-recall curve |

All metrics are computed by `src/evaluation/metrics.py`.

## Directory Structure

```
├── experiments/
│   ├── run_experiments.py      # Orchestrator: runs all 6 pipeline steps
│   ├── run_ablation.py         # Single ablation condition (random/single/multi/textgrad)
│   ├── run_textgrad.py         # TextGrad synthesizer prompt optimization
│   ├── run_judge_evaluation.py # LLM-as-judge post-hoc evaluation
│   └── legacy/                 # Archived older scripts

├── src/
│   ├── agents/
│   │   ├── base_agent.py               # LiteLLM wrapper with caching
│   │   ├── single_agent.py             # InvestorAgent (single condition)
│   │   ├── market_analyst.py           # Specialist: market & sector
│   │   ├── business_model_analyst.py   # Specialist: revenue model & scalability
│   │   ├── feasibility_analyst.py      # Specialist: product & execution
│   │   ├── team_analyst.py             # Specialist: founding team
│   │   ├── synthesizer.py              # Fixed synthesizer (multi condition)
│   │   └── textgrad_synthesizer.py     # Optimized synthesizer (textgrad condition)
│   ├── evaluation/
│   │   ├── metrics.py          # AP@K, AUROC, balanced accuracy, F1, AUCPR
│   │   └── ap_at_k.py          # AP@K implementation
│   ├── prompts/
│   │   └── templates.py        # Startup profile formatter
│   └── utils/
│       ├── llm_client.py       # Anthropic client 
│       ├── litellm_client.py   # LiteLLM client (used by agents)
│       ├── data_splits.py      # Reproducible train/val/test splits
│       ├── archive.py          # Result archiving utility
│       └── logging.py          # Log analysis utilities
├── notebooks/
│   ├── data_processing.ipynb           # Crunchbase data pipeline
│   ├── agent_data_quality_audit.ipynb  # Data quality checks
│   ├── analysis.ipynb                  # Results analysis
│   ├── output_overview.ipynb           # Predictions overview 
│   ├── reasoning_explorer.ipynb        # Qualitative reasoning inspection (WIP)
│   └── textgrad_visualization.ipynb    # TextGrad prompt evolution plots (WIP)
├── results/
│   ├── ablation/               # Per-condition predictions, metrics, run info
│   ├── textgrad_validation/    # TextGrad training logs, prompts, cached assessments
│   ├── judge_evaluation/       # LLM-as-judge scores
│   ├── baseline/               # Legacy baseline results
│   ├── metrics/
│   └── logs/
├── scripts/
│   ├── run_vc_experiment.sh            # Shell wrapper for experiment runs
│   ├── test_job.sh                     # Smoke test script
│   ├── recover_judge_output.py         # Recover partial judge results
│   └── split_objects_by_entity_type.py # Data preprocessing utility
├── pyproject.toml
└── requirements.txt
```

## Output Files

Each ablation run writes three files to `results/ablation/`:

| File | Contents |
|------|----------|
| `{condition}_{split}_{model}_predictions.jsonl` | Per-startup predictions + analyst assessments |
| `{condition}_{split}_{model}_metrics.json` | Full metric suite |
| `{condition}_{split}_{model}_run_info.json` | Run metadata, timing, cache stats |

TextGrad writes to `results/textgrad_validation/`:

| File | Contents |
|------|----------|
| `final_synthesizer_prompt.txt` | Optimized prompt (loaded by TextGradSynthesizer) |
| `metrics_per_step.jsonl` | Per-step training + validation metrics |
| `prompt_evolution.json` | Prompt length + metrics at each validated step |
| `prompt_step_N.txt` | Checkpoint prompts for resume |
| `cached_assessments/` | Pre-computed analyst assessments (JSON per startup) |

Judge evaluation writes to `results/judge_evaluation/`:

| File | Contents |
|------|----------|
| `judge_scores_{timestamp}.jsonl` | Per-startup, per-condition scores on 6 dimensions |
| `judge_summary_{timestamp}.json` | Mean scores by condition |
| `judge_scores_incremental.jsonl` | Incremental file written during run (crash-safe) |

## Resuming Interrupted Runs

TextGrad training saves a checkpoint after each step. Resume with:

```bash
python experiments/run_textgrad.py --resume_from_step 3
```

## Running on Colab

See `notebooks/colab_experiment.ipynb` and `scripts/run_vc_experiment.sh`. The pipeline supports Google Drive sync — set `DRIVE_RESULTS` in `run_experiments.py`.

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Model Selection

Any LiteLLM-compatible model string works for the analyst and synthesizer agents. The judge and TextGrad backward pass are tested with `groq/llama-3.3-70b-versatile`.

```bash
# Local Ollama
--model ollama/glm4:9b

# Claude via Anthropic
--model claude-haiku-4-5-20251001

# Groq
--model groq/llama-3.3-70b-versatile
```

## References

- Maarouf et al. (2025) — startup success prediction metric framework
- Wang et al. (2025) — LLM over-prediction bias in VC settings
- Gompers et al. (2020) — VC decision criteria, team quality emphasis
- Yuksekgonul et al. (2025) — TextGrad: Automatic Differentiation via Text
- Liu et al. — AP@K as primary ranking metric

## License

MIT License — see LICENSE file for details.
