# Architecture Overview

## System Overview

The project compares four agent architectures for binary VC investment prediction on the 2013 Crunchbase dataset. A 6-step pipeline runs all conditions and evaluates them quantitatively (metrics) and qualitatively (LLM-as-judge).

```
┌────────────────────────────────────────────────────────────────────┐
│                        PIPELINE LAYER                              │
│                    run_experiments.py                              │
├────────────────────────────────────────────────────────────────────┤
│  Step 1: random baseline    → results/ablation/random_*           │
│  Step 2: single agent       → results/ablation/single_*           │
│  Step 3: multi-analyst      → results/ablation/multi_*            │
│  Step 4: TextGrad training  → results/textgrad_validation/        │
│  Step 5: textgrad eval      → results/ablation/textgrad_*         │
│  Step 6: judge evaluation   → results/judge_evaluation/           │
└────────────────────────────────────────────────────────────────────┘
```

---

## Ablation Conditions

### Condition 1: Random Baseline

Uniform random probability draw. No LLM calls. Establishes a lower-bound reference.

### Condition 2: Single Agent

```
Startup Profile
      │
      ▼
┌──────────────────────────┐
│   InvestorAgent          │
│   (single_agent.py)      │
│                          │
│ • 2013-era VC framing    │
│ • Explicit base rate     │
│   calibration (57%)      │
│ • SHAP-ranked eval dims  │
└──────────────────────────┘
      │
   JSON: {decision, probability_float, reasoning}
```

### Condition 3: Multi-Analyst (fixed synthesizer)

```
Startup Profile
      │
      ├─────────────────────────────────────────────────────────────┐
      │              (ThreadPoolExecutor, max_workers=4)            │
      ▼                    ▼                   ▼                   ▼
MarketAnalyst   BusinessModelAnalyst   FeasibilityAnalyst   TeamAnalyst
(sector,         (revenue model,        (product viability,  (founder quality,
 geography,       scalability,           execution,           credentials,
 timing)          capital efficiency)    traction)            composition)
      │                    │                   │                   │
      └────────────────────┴───────────────────┴───────────────────┘
                                     │
                           4 assessment dicts
                           {decision, confidence, rationale}
                                     │
                                     ▼
                           ┌──────────────────┐
                           │  SynthesizerAgent │
                           │  (fixed prompt)   │
                           └──────────────────┘
                                     │
                          JSON: {decision, probability,
                                 num_promising, conflicts, reasoning}
```

### Condition 4: TextGrad (optimized synthesizer)

Same multi-analyst structure, but `SynthesizerAgent` is replaced with `TextGradSynthesizer`, which loads the optimized prompt from `results/textgrad_validation/final_synthesizer_prompt.txt`.

---

## TextGrad Optimization

TextGrad treats the synthesizer's system prompt as a differentiable variable and updates it via textual gradient descent (Yuksekgonul et al. 2025).

```
PHASE 1 — Data Preparation
──────────────────────────
Pre-compute analyst assessments for all training examples.
Cache to results/textgrad_validation/cached_assessments/{object_id}.json
(Avoids re-running analysts on every training step.)

PHASE 2 — Training Loop (one step per training example)
────────────────────────────────────────────────────────

  synthesizer_prompt = tg.Variable(SYNTHESIZER_SYSTEM_PROMPT, requires_grad=True)

  For each training example:
  ┌─────────────────────────────────────────────────────────────────┐
  │  1. Forward pass                                                │
  │     synthesizer_model(cached assessments) → JSON output        │
  │                                                                 │
  │  2. Compute loss                                                │
  │     LLM Judge (backward_engine) compares decision to           │
  │     ground-truth label → textual feedback                      │
  │                                                                 │
  │  3. Backward pass                                               │
  │     loss.backward() → textual gradient on synthesizer_prompt   │
  │                                                                 │
  │  4. Optimizer step                                              │
  │     TextualGradientDescent rewrites synthesizer_prompt         │
  │     (falls back to manual guidance if IndexError)              │
  │                                                                 │
  │  5. Post-processing                                             │
  │     ensure_output_format_preserved() — TextGrad may drop the   │
  │     OUTPUT FORMAT section; this reinstates it if missing.      │
  │                                                                 │
  │  6. Validation (every N steps)                                  │
  │     Evaluate on val set → AP@10/20/30, AUROC, balanced acc.    │
  │                                                                 │
  │  7. Save checkpoint                                             │
  │     results/textgrad_validation/prompt_step_{step}.txt         │
  └─────────────────────────────────────────────────────────────────┘

PHASE 3 — Save Results
──────────────────────
  final_synthesizer_prompt.txt   ← loaded by TextGradSynthesizer
  metrics_per_step.jsonl
  prompt_evolution.json
```

**Models used in TextGrad:**

| Role | Model | Rationale |
|------|-------|-----------|
| Forward (synthesizer) | `ollama/glm4:latest` | Local, cheap; the model being optimized |
| Backward (gradient generator) | `groq/llama-3.3-70b-versatile` | Stronger instruction-following for gradient quality |

---

## LLM-as-Judge Evaluation

Post-hoc qualitative evaluation across the three LLM conditions (single, multi, textgrad). A shared sample of startups is scored on 6 dimensions.

```
For each (startup, condition) pair:

  Startup Profile
       +
  Analyst Reports (if multi/textgrad)
       +
  Synthesizer Output (decision, probability, reasoning)
       │
       ▼
  Judge LLM (groq/llama-3.3-70b-versatile)
       │
       ▼
  6 scores (1–5) + justifications:
    1. product_novelty      — Schumpeter type 1
    2. market_opportunity   — Schumpeter type 3
    3. feasibility          — Schumpeter type 2
    4. team_quality         — Gompers et al. 2020
    5. reasoning_coherence
    6. risk_identification
```

Rate-limit handling: 65-second sleep between judge calls (Groq free tier). Incremental writes to `judge_scores_incremental.jsonl` survive crashes; `--resume_from` skips already-scored pairs.

---

## Agent Layer

All agents extend `BaseAgent` (`src/agents/base_agent.py`), which wraps a LiteLLM client with prompt caching.

```
BaseAgent
├── InvestorAgent          single_agent.py     — end-to-end VC evaluation
├── MarketAnalyst          market_analyst.py   — sector / geography / timing
├── BusinessModelAnalyst   business_model_analyst.py
├── FeasibilityAnalyst     feasibility_analyst.py
├── TeamAnalyst            team_analyst.py
├── SynthesizerAgent       synthesizer.py      — fixed prompt synthesizer
└── TextGradSynthesizer    textgrad_synthesizer.py — loads optimized prompt
```

Each specialist outputs:
```json
{
  "decision": "PROMISING" | "NOT_PROMISING",
  "confidence": 0-100,
  "rationale": "..."
}
```

The synthesizer aggregates the four specialist outputs:
```json
{
  "decision": "INVEST" | "PASS",
  "probability": 0-100,
  "num_promising": 0-4,
  "num_not_promising": 0-4,
  "avg_confidence": 0.0-100.0,
  "conflicts": "...",
  "reasoning": "..."
}
```

---

## Evaluation Metrics

Primary metric: **AP@K** (Average Precision at K), following Liu et al. Measures how precisely successful startups are ranked in the top-K predictions.

```
compute_metrics(y_true, y_prob) → {
    # Primary
    "ap_10": ...,  "ap_20": ...,  "ap_30": ...,

    # Secondary
    "auroc": ...,  "aucpr": ...,
    "balanced_accuracy": ...,
    "precision": ...,  "recall": ...,  "f1": ...,

    # Diagnostics
    "base_rate": ...,  "prediction_bias": ...,
    "tp": ...,  "fp": ...,  "tn": ...,  "fn": ...
}
```

Per-sector metrics are also computed for conditions with ≥3 examples per sector.

---

## Data Flow

```
raw Crunchbase CSVs (2013 snapshot)
        │
        ▼
notebooks/data_processing.ipynb
  • Feature engineering
  • Team aggregation
  • Anonymization
  • Leakage detection
  • University ranking integration
        │
        ▼
data/processed/*.parquet
        │
        ▼
src/utils/data_splits.py
  get_splits(random_state=42) → df_train, df_val, df_test
        │
        ├── df_train → TextGrad training examples
        ├── df_val   → TextGrad validation + ablation evaluation
        └── df_test  → Final ablation evaluation
```

---

## Results Structure

```
results/
├── ablation/
│   ├── {condition}_{split}_{model}_predictions.jsonl
│   ├── {condition}_{split}_{model}_metrics.json
│   ├── {condition}_{split}_{model}_sector_metrics.json
│   ├── {condition}_{split}_{model}_run_info.json
│   └── archive/          ← previous runs auto-archived
├── textgrad_validation/
│   ├── final_synthesizer_prompt.txt
│   ├── metrics_per_step.jsonl
│   ├── prompt_evolution.json
│   ├── prompt_step_N.txt          ← one per training step
│   ├── data_splits.json
│   └── cached_assessments/
│       └── {object_id}.json       ← one per startup
└── judge_evaluation/
    ├── judge_scores_{timestamp}.jsonl
    ├── judge_summary_{timestamp}.json
    └── judge_scores_incremental.jsonl
```

---

For implementation details, see the inline docstrings in each file. Key entry points:
- [experiments/run_experiments.py](experiments/run_experiments.py) — full pipeline
- [experiments/run_ablation.py](experiments/run_ablation.py) — individual conditions
- [experiments/run_textgrad.py](experiments/run_textgrad.py) — TextGrad training
- [experiments/run_judge_evaluation.py](experiments/run_judge_evaluation.py) — LLM-as-judge
- [src/evaluation/metrics.py](src/evaluation/metrics.py) — metric definitions
