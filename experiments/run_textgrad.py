"""
TextGrad Optimization for Synthesizer System Prompt
====================================================
Goal: Optimize the synthesizer's system prompt using TextGrad on a small validation set.

Architecture:
  [Pre-cached multi-analyst assessments]
        │
        ▼
  [Synthesizer]  ←── TextGrad variable (requires_grad=True)  ← OPTIMIZED
        │
        ▼
  [Binary decision: invest / pass]
        │
        ▼
  [LLM Judge]  ←── compares to ground truth label, generates gradient
        │
        ▼
  [TextualGradientDescent]  ←── rewrites synthesizer prompt
                                (or manual refinement if TextGrad fails)

Per training step:
  1. Load cached analyst assessments (pre-computed, no re-runs)
  2. Forward pass: synthesizer(assessments) → decision + probability
  3. Compute loss: LLM judge vs ground truth
  4. Backward: loss.backward() → textual gradient
  5. Optimizer step: optimizer.step() → rewrite prompt
  6. Validate: eval on val set → log AP@10, AP@20, AP@30, balanced_accuracy, weighted_f1, auroc

Run with:
  python experiments/run_textgrad.py --n_train 5 --n_val 5 --seed 42
  
  # Fast MVP (validate only every 5 steps):
  python experiments/run_textgrad.py --n_train 5 --n_val 5 --validate_every 5
  
  # Ultra-fast (validate only at the end):
  python experiments/run_textgrad.py --n_train 5 --n_val 5 --validate_every 999
"""

import sys
import json
import time
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import textgrad as tg
from textgrad.engine_experimental.litellm import LiteLLMEngine
from textgrad.optimizer import TextualGradientDescent

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.market_analyst import MarketAnalyst
from src.agents.business_model_analyst import BusinessModelAnalyst
from src.agents.feasibility_analyst import FeasibilityAnalyst
from src.agents.team_analyst import TeamAnalyst
from src.agents.synthesizer import SynthesizerAgent, SYNTHESIZER_SYSTEM_PROMPT
from src.evaluation.metrics import compute_metrics, print_metrics
from src.prompts.templates import format_startup_profile
from src.utils.data_splits import get_splits
from src.utils.archive import make_run_dir

RESULTS_DIR = Path("results/textgrad_validation")
EXPERIMENTS_LOG_DIR = Path(__file__).resolve().parent / "textgrad logs"
MODEL_NAME = "ollama/glm4:latest"              # Forward model: synthesizer during TextGrad training + analyst assessments
BACKWARD_MODEL_NAME = "groq/llama-3.3-70b-versatile"    # Backward model: generates textual gradients (needs stronger instruction-following than forward)

# Configure logging to experiments/textgrad logs/
EXPERIMENTS_LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = EXPERIMENTS_LOG_DIR / f"{datetime.utcnow().strftime('%Y-%m-%d_%H-%M-%S')}.jsonl"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


# ─── PHASE 1: DATA PREPARATION & CACHING ─────────────────────────────────────

def check_cache_exists(
    df_examples: pd.DataFrame,
    cache_dir: Optional[Path] = None,
) -> bool:
    """
    Check if all assessments for examples are already cached.
    
    Args:
        df_examples: DataFrame with startup examples
        cache_dir: Directory where cached assessments are stored
    
    Returns:
        True if all cache files exist, False otherwise
    """
    if cache_dir is None:
        cache_dir = RESULTS_DIR / "cached_assessments"
    
    if not cache_dir.exists():
        return False
    
    for _, row in df_examples.iterrows():
        object_id = str(row.get("object_id"))
        cache_file = cache_dir / f"{object_id}.json"
        if not cache_file.exists():
            return False
    
    return True


def load_cached_assessments(
    df_examples: pd.DataFrame,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Load pre-computed assessments from cache.
    
    Args:
        df_examples: DataFrame with startup examples
        cache_dir: Directory where cached assessments are stored
    
    Returns:
        Dict mapping object_id → cached assessment record
    """
    if cache_dir is None:
        cache_dir = RESULTS_DIR / "cached_assessments"
    
    cached_assessments = {}
    
    for _, row in df_examples.iterrows():
        object_id = str(row.get("object_id"))
        cache_file = cache_dir / f"{object_id}.json"
        
        with open(cache_file, "r") as f:
            cache_record = json.load(f)
        
        cached_assessments[object_id] = cache_record
    
    return cached_assessments


def cache_multi_analyst_assessments(
    df_examples: pd.DataFrame,
    model: str = MODEL_NAME,
    cache_dir: Optional[Path] = None,
) -> dict:
    """
    Pre-compute and cache multi-agent assessments for examples.
    
    Args:
        df_examples: DataFrame with startup examples
        model: Model to use for agents
        cache_dir: Directory to save cached assessments
    
    Returns:
        Dict mapping object_id → cached assessment record
    """
    if cache_dir is None:
        cache_dir = RESULTS_DIR / "cached_assessments"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Initialize agents
    market_analyst = MarketAnalyst(model=model)
    biz_analyst = BusinessModelAnalyst(model=model)
    feasibility_analyst = FeasibilityAnalyst(model=model)
    team_analyst = TeamAnalyst(model=model)

    analysts = [market_analyst, biz_analyst, feasibility_analyst, team_analyst]
    analyst_names = ["Market", "Business Model", "Feasibility", "Team"]

    cached_assessments = {}

    print(f"Pre-computing and caching analyst assessments for {len(df_examples)} examples...\n")
    start_time = time.time()

    for i, (idx, row) in enumerate(df_examples.iterrows()):
        object_id = str(row.get("object_id"))
        profile = format_startup_profile(row)

        # Run 4 analysts
        assessments = []
        for analyst, analyst_name in zip(analysts, analyst_names):
            try:
                assessment = analyst.evaluate(profile)
            except Exception as e:
                assessment = {"parse_error": True, "error": str(e)}
            assessments.append(assessment)

        # Cache the assessments
        cache_record = {
            "object_id": object_id,
            "name": row.get("name"),
            "target": int(row.get("target")),
            "category_code": row.get("category_code"),
            "startup_profile": profile,
            "analyst_assessments": {
                name: assessment for name, assessment in zip(analyst_names, assessments)
            },
        }

        # Save to file
        cache_file = cache_dir / f"{object_id}.json"
        with open(cache_file, "w") as f:
            json.dump(cache_record, f, indent=2)

        cached_assessments[object_id] = cache_record

        if (i + 1) % 2 == 0 or (i + 1) == len(df_examples):
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(df_examples) - i - 1) / rate if rate > 0 else 0
            print(
                f"  [{i+1}/{len(df_examples)}]  "
                f"elapsed: {elapsed:.0f}s  "
                f"est. remaining: {remaining:.0f}s"
            )

    print(f"Cached assessments saved to {cache_dir}/\n")
    return cached_assessments


# ─── PHASE 2: TEXTGRAD TRAINING LOOP ──────────────────────────────────────────

def ensure_output_format_preserved(prompt: str) -> str:
    """
    Ensure the OUTPUT FORMAT section is present in the prompt.
    
    TextGrad optimizer may remove it during rewriting, so this post-processes
    the optimized prompt to guarantee the format instructions are present.
    
    Args:
        prompt: Optimized prompt from TextGrad
    
    Returns:
        Prompt with output format section guaranteed to be present
    """
    output_format_section = """
OUTPUT FORMAT:
Respond with valid JSON only — no markdown, no preamble:

{
  "decision": "INVEST" or "PASS",
  "probability": <integer 0-100, your estimated probability of successful exit>,
  "num_promising": <integer 0-4, number of analysts who said PROMISING>,
  "num_not_promising": <integer 0-4>,
  "avg_confidence": <float 0-100, average of the four analyst confidences>,
  "conflicts": "<string describing any high-confidence disagreements, or 'None'>",
  "reasoning": "<2-3 sentences synthesizing the four assessments and explaining the decision>"
}"""
    
    if "OUTPUT FORMAT:" not in prompt:
        prompt = prompt.rstrip() + output_format_section
    
    return prompt


def make_loss_fn(correct_label: str, engine: LiteLLMEngine) -> tg.TextLoss:
    """
    Create LLM-as-judge loss function.
    
    Args:
        correct_label: "INVEST" or "PASS"
        engine: LLM engine for judge
    
    Returns:
        TextLoss object
    """
    eval_prompt = (
        f"The correct investment decision for this startup is: '{correct_label.upper()}'. "
        f"The synthesizer responds with a JSON object containing a 'decision' field. "
        f"Check if the 'decision' field equals '{correct_label.upper()}'. "
        f"If YES: confirm that the decision matches correctly. "
        f"If NO: explain specifically what the SYNTHESIZER'S SYSTEM PROMPT should say "
        f"differently to produce decision: '{correct_label.upper()}'. "
        f"Focus on improving the synthesis logic and decision rules, not the analyst assessments."
    )
    return tg.TextLoss(eval_system_prompt=eval_prompt, engine=engine)


def evaluate_synthesizer_on_val_set(
    synthesizer_model: tg.BlackboxLLM,
    cached_val_assessments: dict,
) -> dict:
    """
    Evaluate current synthesizer prompt on validation set.
    
    Args:
        synthesizer_model: TextGrad synthesizer model
        cached_val_assessments: Dict of cached assessments
    
    Returns:
        Dict with metrics: ap_10, ap_20, ap_30, balanced_accuracy, weighted_f1, auroc
    """
    y_true = []
    y_prob = []

    for object_id, cache_record in cached_val_assessments.items():
        # Get analyst assessments
        analyst_assessments = [
            cache_record["analyst_assessments"]["Market"],
            cache_record["analyst_assessments"]["Business Model"],
            cache_record["analyst_assessments"]["Feasibility"],
            cache_record["analyst_assessments"]["Team"],
        ]

        # Prepare input for synthesizer
        profile = cache_record["startup_profile"]
        analyst_names = ["Market", "Business Model", "Feasibility", "Team"]
        
        analyst_report = "\n".join([
            f"{name} Analyst:\n{json.dumps(a, indent=2)}"
            for name, a in zip(analyst_names, analyst_assessments)
        ])

        synthesizer_input = tg.Variable(
            value=(
                f"Startup: {cache_record['name']}\n\n"
                f"Profile:\n{profile}\n\n"
                f"Analyst Reports:\n{analyst_report}\n\n"
                f"Based on the above, give your final investment decision."
            ),
            requires_grad=False,
            role_description="combined startup description and analyst reports",
        )

        # Forward pass
        try:
            synthesizer_output = synthesizer_model(synthesizer_input)
            # Parse probability from output (assume JSON format)
            output_text = synthesizer_output.value
            match = re.search(r'"probability"\s*:\s*(\d+)', output_text)
            if match:
                prob = int(match.group(1)) / 100.0
            else:
                prob = 0.5

            y_prob.append(prob)
            y_true.append(cache_record["target"])
        except Exception as e:
            print(f"    Warning: Synthesizer failed for {object_id}: {e}")
            continue

    if len(y_true) == 0:
        raise ValueError("No valid validation examples evaluated")

    # Compute metrics
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)

    return {
        "ap_10": metrics.get("ap_10"),
        "ap_20": metrics.get("ap_20"),
        "ap_30": metrics.get("ap_30"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "weighted_f1": metrics.get("f1"),
        "auroc": metrics.get("auroc"),
        "n_evaluated": metrics.get("n"),
    }


def run_textgrad_optimization(
    n_train: int = 5,
    n_val: int = 5,
    seed: int = 42,
    validate_every: int = 1,
    resume_from_step: Optional[int] = None,
    output_dir: Path = RESULTS_DIR,
) -> None:
    """
    Main TextGrad optimization loop.

    Args:
        n_train: Number of training examples
        n_val: Number of validation examples
        seed: Random seed for reproducibility
        validate_every: Validate every N steps (1=every step, 5=every 5th step, etc.)
        resume_from_step: If set, load prompt from prompt_step_N.txt and skip steps 0..N.
                          Useful for resuming after a crash or rate-limit failure.
                          Example: --resume_from_step 3 resumes from after step 3.
    """
    print("=" * 70)
    print("TextGrad Synthesizer Optimization")
    print("=" * 70)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ─── Load data ────────────────────────────────────────────────────────────
    print("\n[PHASE 1: Data Preparation & Caching]")
    print(f"\nLoading {n_train} training + {n_val} validation examples...\n")

    df_train, df_val, _ = get_splits(random_state=seed)

    # Limit to n_train and n_val
    df_train = df_train.head(n_train)
    df_val = df_val.head(n_val)

    print(f"✓ Loaded {len(df_train)} training examples")
    print(f"✓ Loaded {len(df_val)} validation examples\n")

    # Save split info
    split_info = {
        "n_train": len(df_train),
        "n_val": len(df_val),
        "train_object_ids": df_train["object_id"].tolist(),
        "val_object_ids": df_val["object_id"].tolist(),
        "seed": seed,
    }
    with open(output_dir / "data_splits.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # ─── Cache assessments ─────────────────────────────────────────────────────
    print("Checking cache for multi-agent assessments...\n")

    cache_dir = output_dir / "cached_assessments"
    train_cache_ok = check_cache_exists(df_train, cache_dir)
    val_cache_ok = check_cache_exists(df_val, cache_dir)

    if train_cache_ok and val_cache_ok:
        print(f"✓ Cache found for all {len(df_train)} training examples")
        print(f"✓ Cache found for all {len(df_val)} validation examples\n")
        print("Loading assessments from cache...\n")
        train_assessments = load_cached_assessments(df_train, cache_dir)
        val_assessments = load_cached_assessments(df_val, cache_dir)
    else:
        print("Cache missing or incomplete. Computing assessments...\n")
        if not train_cache_ok:
            print("  - Computing training assessments...")
            train_assessments = cache_multi_analyst_assessments(df_train, model=MODEL_NAME)
        else:
            print("  - Loading training assessments from cache...")
            train_assessments = load_cached_assessments(df_train, cache_dir)
        
        if not val_cache_ok:
            print("  - Computing validation assessments...")
            val_assessments = cache_multi_analyst_assessments(df_val, model=MODEL_NAME)
        else:
            print("  - Loading validation assessments from cache...")
            val_assessments = load_cached_assessments(df_val, cache_dir)

    # ─── Initialize TextGrad ──────────────────────────────────────────────────
    print("\n[PHASE 2: TextGrad Training Loop]\n")
    print("Initializing TextGrad components...\n")

    # Option A (Yuksekgonul et al. 2025): separate forward and backward engines.
    # Forward (synthesizer): weaker/cheaper local model — the model being optimized.
    # Backward (gradient generator): stronger local model — generates textual gradients.
    forward_engine  = LiteLLMEngine(MODEL_NAME, cache=True)
    backward_engine = LiteLLMEngine(BACKWARD_MODEL_NAME, cache=True)
    tg.set_backward_engine(backward_engine, override=True)

    # Load starting prompt — either from a checkpoint or from the default
    if resume_from_step is not None:
        checkpoint_file = output_dir / f"prompt_step_{resume_from_step}.txt"
        if not checkpoint_file.exists():
            raise FileNotFoundError(
                f"Cannot resume: {checkpoint_file} not found. "
                f"Available steps: {sorted(int(f.stem.split('_')[-1]) for f in output_dir.glob('prompt_step_*.txt'))}"
            )
        starting_prompt = checkpoint_file.read_text()
        print(f"✓ Resuming from step {resume_from_step} ({checkpoint_file.name}, {len(starting_prompt)} chars)\n")
    else:
        starting_prompt = SYNTHESIZER_SYSTEM_PROMPT

    synthesizer_prompt = tg.Variable(
        value=starting_prompt,
        requires_grad=True,
        role_description="system prompt for the investment synthesizer",
    )

    print(f"Forward model  (synthesizer): {MODEL_NAME}")
    print(f"Backward model (gradients):   {BACKWARD_MODEL_NAME}")
    print(f"Starting prompt length: {len(synthesizer_prompt.value)} chars\n")

    synthesizer_model = tg.BlackboxLLM(engine=forward_engine, system_prompt=synthesizer_prompt)

    optimizer = TextualGradientDescent(
        parameters=[synthesizer_prompt],
        engine=backward_engine,
        new_variable_tags=["<improved_variable>", "</improved_variable>"],
        constraints=[
            "Output must have clear INVEST or PASS decision in JSON format.",
            "ALWAYS include this OUTPUT FORMAT section at the end, exactly as specified: OUTPUT FORMAT: Respond with valid JSON only — no markdown, no preamble: { \"decision\": \"INVEST\" or \"PASS\", \"probability\": <integer 0-100>, \"num_promising\": <integer 0-4>, \"num_not_promising\": <integer 0-4>, \"avg_confidence\": <float 0-100>, \"conflicts\": \"<string>\", \"reasoning\": \"<2-3 sentences>\" }",
        ],
    )

    # ─── Training loop ─────────────────────────────────────────────────────────
    print("=" * 70)
    print("Starting TextGrad training loop...")
    print(f"Validation every {validate_every} step(s)")
    print("=" * 70)

    metrics_log = []
    prompts_log = []
    validation_steps = set()  # Track which steps we validate

    for step, (idx, row) in enumerate(df_train.iterrows()):
        # Skip steps already completed before the resume point
        if resume_from_step is not None and step <= resume_from_step:
            print(f"  Skipping step {step} (already completed before resume point)")
            continue

        print(f"\n{'─' * 70}")
        print(f"Training Step {step + 1}/{len(df_train)}")
        print(f"Example: {row.get('name')}")
        print(f"Ground truth: {'INVEST' if row.get('target') == 1 else 'PASS'}")

        object_id = str(row.get("object_id"))
        cache_record = train_assessments[object_id]

        # Get analyst assessments
        analyst_assessments = [
            cache_record["analyst_assessments"]["Market"],
            cache_record["analyst_assessments"]["Business Model"],
            cache_record["analyst_assessments"]["Feasibility"],
            cache_record["analyst_assessments"]["Team"],
        ]

        # ── Forward pass ──────────────────────────────────────────────────────
        profile = cache_record["startup_profile"]
        analyst_names = ["Market", "Business Model", "Feasibility", "Team"]
        
        analyst_report = "\n".join([
            f"{name} Analyst:\n{json.dumps(a, indent=2)}"
            for name, a in zip(analyst_names, analyst_assessments)
        ])

        synthesizer_input = tg.Variable(
            value=(
                f"Startup: {cache_record['name']}\n\n"
                f"Profile:\n{profile}\n\n"
                f"Analyst Reports:\n{analyst_report}\n\n"
                f"Based on the above, give your final investment decision."
            ),
            requires_grad=False,
            role_description="combined startup description and analyst reports",
        )

        print("\n  [Forward pass...]")
        synthesizer_output = synthesizer_model(synthesizer_input)
        print(f"  Synthesizer output: {synthesizer_output.value[:150]}...")

        # ── Compute loss ──────────────────────────────────────────────────────
        correct_label = "INVEST" if row.get("target") == 1 else "PASS"
        loss_fn = make_loss_fn(correct_label, backward_engine)

        print("\n  [Computing loss...]")
        loss = loss_fn(synthesizer_output)
        print(f"  Judge feedback: {loss.value[:150]}...")

        # Parse synthesizer output to check correctness
        pred_decision = None
        pred_prob = None
        try:
            output_text = synthesizer_output.value.strip()
            if output_text.startswith("```"):
                output_text = output_text.split("```")[1]
                if output_text.startswith("json"):
                    output_text = output_text[4:]
                output_text = output_text.rsplit("```", 1)[0]
            result = json.loads(output_text.strip())
            pred_decision = result.get("decision", "UNKNOWN")
            pred_prob = result.get("probability", 50)
            pred_correct = (pred_decision == correct_label)
        except (json.JSONDecodeError, IndexError):
            pred_decision = "PARSE_ERROR"
            pred_prob = 50
            pred_correct = False

        # ── Backward pass + optimizer step (with rate-limit retry) ───────────
        # Groq TPD cap (100K tokens/day) can hit mid-backward-pass. If it does,
        # sleep until the limit resets and retry the whole backward+step pair.
        _MAX_RL_RETRIES = 5
        _RL_SLEEP_S     = 90   # seconds to wait on 429 before retrying
        for _rl_attempt in range(_MAX_RL_RETRIES):
            try:
                print("  [Backward pass...]")
                loss.backward()
                break   # success — exit retry loop
            except Exception as _e:
                _msg = str(_e)
                if "rate_limit" in _msg.lower() or "429" in _msg or "RateLimitError" in type(_e).__name__:
                    if _rl_attempt < _MAX_RL_RETRIES - 1:
                        import time as _time
                        print(f"\n  ⚠ Groq rate limit hit during backward pass "
                              f"(attempt {_rl_attempt + 1}/{_MAX_RL_RETRIES}). "
                              f"Sleeping {_RL_SLEEP_S}s before retry...")
                        _time.sleep(_RL_SLEEP_S)
                        # Reset gradients so we can redo the backward cleanly
                        optimizer.zero_grad()
                    else:
                        print(f"\n  ✗ Rate limit still active after {_MAX_RL_RETRIES} retries. Aborting.")
                        raise
                else:
                    raise   # non-rate-limit error — propagate immediately

        # Capture prompt-level gradient before zero_grad clears it
        prompt_gradient_text = None
        if synthesizer_prompt.gradients:
            prompt_gradient_text = next(iter(synthesizer_prompt.gradients)).value

        # Store prompt before update
        prompt_before_len = len(synthesizer_prompt.value)

        # ── Optimizer step ────────────────────────────────────────────────────
        print("  [Optimizer step...]")
        try:
            optimizer.step()
            # Ensure output format is preserved (TextGrad sometimes removes it)
            synthesizer_prompt.value = ensure_output_format_preserved(synthesizer_prompt.value)
            print(f"\n  [Updated prompt]:\n  '{synthesizer_prompt.value[:150]}...'")
        except IndexError as e:
            # TextGrad optimizer formatting failed - use manual refinement instead
            print(f"\n  ⚠ TextGrad optimizer failed: {str(e)[:100]}...")
            print("  Using manual prompt refinement as fallback.\n")
            
            # Manually refine the prompt by appending guidance
            if "INVEST" in correct_label:
                guidance = "\n\nWhen analysts agree on PROMISING signals across market, business model, feasibility, and team, lean toward INVEST."
            else:
                guidance = "\n\nWhen any analyst raises red flags or confidence is low, lean toward PASS."
            
            synthesizer_prompt.value += guidance
            print(f"  [Manual guidance added to prompt]")

        # Log per-step metrics
        prompt_after_len = len(synthesizer_prompt.value)
        step_metrics = {
            "step": step,
            "training_example_id": object_id,
            "training_example_name": cache_record["name"],
            "ground_truth": correct_label,
            "predicted_decision": pred_decision,
            "predicted_probability": pred_prob,
            "prediction_correct": pred_correct,
            "judge_feedback_excerpt": loss.value[:100],
            "judge_feedback_full": loss.value,
            "prompt_gradient": prompt_gradient_text,
            "prompt_length_before": prompt_before_len,
            "prompt_length_after": prompt_after_len,
            "prompt_length_change": prompt_after_len - prompt_before_len,
            "timestamp": datetime.utcnow().isoformat(),
        }
        metrics_log.append(step_metrics)
        # Flush immediately so a crash mid-run doesn't lose completed steps
        with open(output_dir / "metrics_per_step.jsonl", "a") as f:
            f.write(json.dumps(step_metrics) + "\n")

        print(f"\n  [Step {step + 1} Metrics]")
        print(f"    Prediction: {pred_decision} (prob: {pred_prob}%) {'✓' if pred_correct else '✗'}")
        print(f"    Prompt length: {prompt_before_len} → {prompt_after_len} (Δ {prompt_after_len - prompt_before_len:+d})")

        optimizer.zero_grad()

        # ── Validation & logging ──────────────────────────────────────────────
        # Only validate every N steps or at the end
        should_validate = (step % validate_every == 0) or (step == len(df_train) - 1)
        
        if should_validate:
            print("\n  [Validating on val set...]")
            val_metrics = evaluate_synthesizer_on_val_set(synthesizer_model, val_assessments)

            print(f"\n  Validation Metrics:")
            print(f"    AP@10: {val_metrics['ap_10']:.4f}")
            print(f"    AP@20: {val_metrics['ap_20']:.4f}")
            print(f"    AP@30: {val_metrics['ap_30']:.4f}")
            print(f"    Balanced Accuracy: {val_metrics['balanced_accuracy']:.4f}")
            print(f"    Weighted F1: {val_metrics['weighted_f1']:.4f}")
            print(f"    AUROC: {val_metrics['auroc']:.4f}")

            # Log metrics
            metrics_record = {
                "step": step,
                "training_example_id": object_id,
                "training_example_name": cache_record["name"],
                "timestamp": datetime.utcnow().isoformat(),
                "val_metrics": val_metrics,
            }
            metrics_log.append(metrics_record)
            with open(output_dir / "metrics_per_step.jsonl", "a") as f:
                f.write(json.dumps(metrics_record) + "\n")
            validation_steps.add(step)
        else:
            print(f"\n  [Skipping validation (validating every {validate_every} steps)]")

        # Log prompt
        prompts_log.append({
            "step": step,
            "prompt": synthesizer_prompt.value,
            "prompt_length": len(synthesizer_prompt.value),
        })

        # Save intermediate results
        with open(output_dir / f"prompt_step_{step}.txt", "w") as f:
            f.write(synthesizer_prompt.value)

    # ─── Phase 3: Compile results ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("TEXTGRAD OPTIMIZATION COMPLETE")
    print("=" * 70)

    # Rewrite metrics log cleanly at end of successful run (deduplicates any incremental writes)
    with open(output_dir / "metrics_per_step.jsonl", "w") as f:
        for record in metrics_log:
            f.write(json.dumps(record) + "\n")

    # Save final metrics
    val_records = [r for r in metrics_log if "val_metrics" in r]
    final_metrics = val_records[-1]["val_metrics"]
    with open(output_dir / "final_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    # Save final prompt
    with open(output_dir / "final_synthesizer_prompt.txt", "w") as f:
        f.write(synthesizer_prompt.value)

    # Save prompt evolution
    # Create a mapping from step number to metrics (only validated steps have metrics)
    metrics_by_step = {record["step"]: record["val_metrics"] for record in val_records}

    with open(output_dir / "prompt_evolution.json", "w") as f:
        json.dump(
            {
                "steps": [
                    {
                        "step": p["step"],
                        "prompt_length": p["prompt_length"],
                        "metrics": metrics_by_step[p["step"]],
                    }
                    for p in prompts_log
                    if p["step"] in metrics_by_step  # Only include validated steps
                ]
            },
            f,
            indent=2,
        )

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - metrics_per_step.jsonl: Metrics after each training step")
    print(f"  - final_metrics.json: Final validation metrics")
    print(f"  - final_synthesizer_prompt.txt: Optimized prompt")
    print(f"  - prompt_evolution.json: Prompt evolution across steps")
    print(f"  - prompt_step_*.txt: Intermediate prompts")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TextGrad optimization for synthesizer system prompt"
    )
    parser.add_argument(
        "--n_train",
        type=int,
        default=5,
        help="Number of training examples (default: 5)",
    )
    parser.add_argument(
        "--n_val",
        type=int,
        default=5,
        help="Number of validation examples (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--validate_every",
        type=int,
        default=1,
        help="Validate every N training steps (1=every step, 5=every 5th, default: 1). Use validate_every=5 to speed up 5x or validate_every=999 for final only.",
    )
    parser.add_argument(
        "--resume_from_step",
        type=int,
        default=None,
        help=(
            "Resume from a saved checkpoint. Loads prompt_step_N.txt and skips steps 0..N. "
            "Example: --resume_from_step 3 continues from after step 3. "
            "Use with the same --n_train and --seed as the original run."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write results (default: results/textgrad_validation)",
    )

    args = parser.parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    elif args.resume_from_step is not None:
        latest = RESULTS_DIR.resolve() / "latest"
        if not latest.exists():
            raise FileNotFoundError("No previous run to resume from — run without --resume_from_step first.")
        output_dir = latest.resolve()
    else:
        output_dir = make_run_dir(RESULTS_DIR.resolve())

    run_textgrad_optimization(
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
        validate_every=args.validate_every,
        resume_from_step=args.resume_from_step,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()