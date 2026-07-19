"""
Ablation study: Compare four conditions:
1. Random baseline: Uniform random {0, 1} predictions
2. Single LLM agent: Baseline InvestorAgent
3. Multi-analyst: 4 specialists + fixed synthesizer (with parallelization)
4. TextGrad: 4 specialists + TextGrad-optimized synthesizer

Usage
-----
    python experiments/run_ablation.py --ablation random --split test
    python experiments/run_ablation.py --ablation single --model ollama/glm4:latest --sample 50
    python experiments/run_ablation.py --ablation multi --model ollama/glm4:latest --split val
    python experiments/run_ablation.py --ablation textgrad --model ollama/glm4:latest --split test

Output
------
    results/ablation/{ablation}_{split}_{model}_predictions.jsonl
    results/ablation/{ablation}_{split}_{model}_metrics.json
    results/ablation/{ablation}_{split}_{model}_run_info.json
"""
import argparse
import json
import time
import random
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.single_agent import InvestorAgent
from src.agents.market_analyst import MarketAnalyst
from src.agents.business_model_analyst import BusinessModelAnalyst
from src.agents.feasibility_analyst import FeasibilityAnalyst
from src.agents.team_analyst import TeamAnalyst
from src.agents.synthesizer import SynthesizerAgent
from src.agents.textgrad_synthesizer import TextGradSynthesizer
from src.evaluation.metrics import compute_metrics, print_metrics
from src.prompts.templates import format_startup_profile
from src.utils.data_splits import get_splits, get_temporal_splits
from src.utils.archive import make_run_dir

RESULTS_DIR = Path("results/ablation")

# Fixed seed for dataset splitting — decoupled from the training/inference seed
# so that all experiment seeds evaluate on the identical train/val/test partition.
# Only the model-level randomness (synthesizer temperature, TextGrad trajectory)
# varies across seeds. 
SPLIT_SEED = 42



def run_random_baseline(
    df_eval,
    split_name: str,
    random_state: int = 42,
    threshold: float = 0.5,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Random baseline: Uniform random predictions.

    Args:
        df_eval: DataFrame to evaluate
        split_name: Name of the split (for file naming)
        random_state: Seed for reproducibility
        threshold: Decision threshold

    Returns:
        Metrics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(random_state)
    np.random.seed(random_state)

    predictions_path = output_dir / f"random_{split_name}_predictions.jsonl"
    results = []

    print(f"Running random baseline on {len(df_eval)} rows...\n")
    start_time = time.time()

    with open(predictions_path, "w") as f_out:
        for i, (idx, row) in enumerate(df_eval.iterrows()):
            # Random probability
            prob = random.uniform(0, 1)
            decision = "INVEST" if prob >= threshold else "PASS"

            record = {
                "object_id": row.get("object_id"),
                "name": row.get("name"),
                "target": int(row["target"]),
                "category_code": row.get("category_code"),
                "probability_float": prob,
                "decision": decision,
                "reasoning": "Random baseline (no evaluation)",
                "parse_error": False,
            }

            results.append(record)
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 50 == 0 or (i + 1) == len(df_eval):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(df_eval) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1}/{len(df_eval)}]  "
                    f"elapsed: {elapsed:.0f}s  "
                    f"est. remaining: {remaining:.0f}s"
                )

    elapsed_total = time.time() - start_time

    # Compute metrics
    y_true = [r["target"] for r in results]
    y_prob = [r["probability_float"] for r in results]

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    print_metrics(metrics, label="Random Baseline")

    # Save outputs
    with open(output_dir / f"random_{split_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablation": "random",
        "split": split_name,
        "model": "random",
        "n_evaluated": int(len(results)),
        "elapsed_seconds": float(round(elapsed_total, 1)),
        "threshold": float(threshold),
        "random_state": int(random_state),
    }

    with open(output_dir / f"random_{split_name}_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Random baseline saved to {output_dir}/")

    return metrics


def run_single_agent(
    df_eval,
    split_name: str,
    model: str = "claude-haiku-4-5-20251001",
    random_state: int = 42,
    threshold: float = 0.5,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Single LLM agent baseline: InvestorAgent.

    Args:
        df_eval: DataFrame to evaluate
        split_name: Name of the split (for file naming)
        model: Model to use
        random_state: Seed for reproducibility
        threshold: Decision threshold

    Returns:
        Metrics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.replace("/", "_").replace(".", "_")
    predictions_path = output_dir / f"single_{split_name}_{model_name}_predictions.jsonl"
    results = []
    errors = 0

    agent = InvestorAgent(use_cache=True, model=model)

    print(f"Running single agent ({model}) on {len(df_eval)} rows...\n")
    start_time = time.time()

    with open(predictions_path, "w") as f_out:
        for i, (idx, row) in enumerate(df_eval.iterrows()):
            profile = format_startup_profile(row)
            response = agent.evaluate_startup(profile)

            record = {
                "object_id": row.get("object_id"),
                "name": row.get("name"),
                "target": int(row["target"]),
                "category_code": row.get("category_code"),
                "probability_float": response.get("probability_float", 0.5),
                "decision": response.get("decision"),
                "reasoning": response.get("reasoning"),
                "parse_error": response.get("parse_error", False),
            }

            if response.get("parse_error"):
                errors += 1

            results.append(record)
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 10 == 0 or (i + 1) == len(df_eval):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(df_eval) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1}/{len(df_eval)}]  "
                    f"elapsed: {elapsed:.0f}s  "
                    f"est. remaining: {remaining:.0f}s  "
                    f"parse errors: {errors}"
                )

    elapsed_total = time.time() - start_time

    # Compute metrics
    y_true = [r["target"] for r in results]
    y_prob = [r["probability_float"] for r in results]

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    print_metrics(metrics, label=f"Single Agent — {model}")

    # Save outputs
    with open(output_dir / f"single_{split_name}_{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cache_stats = agent.llm_client.get_cache_stats()
    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablation": "single",
        "split": split_name,
        "model": model,
        "n_evaluated": int(len(results)),
        "n_parse_errors": int(errors),
        "elapsed_seconds": float(round(elapsed_total, 1)),
        "threshold": float(threshold),
        "random_state": int(random_state),
        "cache_stats": cache_stats,
    }

    with open(output_dir / f"single_{split_name}_{model_name}_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Single agent results saved to {output_dir}/")
    if cache_stats.get("total_cost_usd", 0) > 0:
        print(f"  API cost: ${cache_stats['total_cost_usd']:.4f}  "
              f"(est. cache savings: ${cache_stats.get('estimated_savings_usd', 0):.4f})")

    return metrics


def _evaluate_multi_analyst(row_data):
    """Helper function to evaluate a single startup with 4 analysts in parallel."""
    idx, row, analysts, profile = row_data

    # Run 4 analysts in parallel
    assessments = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(analysts[0].evaluate, profile): 0,  # Market
            executor.submit(analysts[1].evaluate, profile): 1,  # Business Model
            executor.submit(analysts[2].evaluate, profile): 2,  # Feasibility
            executor.submit(analysts[3].evaluate, profile): 3,  # Team
        }

        for future in as_completed(futures):
            analyst_idx = futures[future]
            try:
                assessment = future.result()
                assessments.append((analyst_idx, assessment))
            except Exception as e:
                assessments.append((analyst_idx, {"parse_error": True, "raw_error": str(e)}))

    # Sort assessments by analyst index
    assessments.sort(key=lambda x: x[0])
    assessment_list = [a[1] for a in assessments]

    return (idx, row, profile, assessment_list)


def run_multi_analyst(
    df_eval,
    split_name: str,
    model: str = "claude-haiku-4-5-20251001",
    random_state: int = 42,
    threshold: float = 0.5,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    Multi-analyst pipeline: 4 specialists (in parallel) + synthesizer.

    Args:
        df_eval: DataFrame to evaluate
        split_name: Name of the dataset split
        model: Model to use for all agents
        random_state: Seed for reproducibility
        threshold: Decision threshold

    Returns:
        Metrics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.replace("/", "_").replace(".", "_")
    predictions_path = output_dir / f"multi_{split_name}_{model_name}_predictions.jsonl"
    results = []
    errors = 0

    # Initialize agents
    market_analyst = MarketAnalyst(model=model)
    biz_analyst = BusinessModelAnalyst(model=model)
    feasibility_analyst = FeasibilityAnalyst(model=model)
    team_analyst = TeamAnalyst(model=model)
    synthesizer = SynthesizerAgent(model=model)

    analysts = [market_analyst, biz_analyst, feasibility_analyst, team_analyst]
    analyst_names = ["Market", "Business Model", "Feasibility", "Team"]

    print(f"Running multi-analyst pipeline ({model}) on {len(df_eval)} rows...\n")
    start_time = time.time()

    with open(predictions_path, "w") as f_out:
        # Prepare data for parallel evaluation
        eval_data = [
            (idx, row, analysts, format_startup_profile(row))
            for idx, row in df_eval.iterrows()
        ]

        for i, row_data in enumerate(eval_data):
            idx, row, profile, assessments = _evaluate_multi_analyst(row_data)

            # Get synthesizer decision
            synthesizer_response = synthesizer.synthesize(profile, assessments)

            # Count analysts with parse errors
            n_parse_errors = sum(1 for a in assessments if a.get("parse_error"))

            record = {
                "object_id": row.get("object_id"),
                "name": row.get("name"),
                "target": int(row["target"]),
                "category_code": row.get("category_code"),
                "probability_float": synthesizer_response.get("probability", 50) / 100.0,
                "decision": synthesizer_response.get("decision"),
                "reasoning": synthesizer_response.get("reasoning"),
                "parse_error": synthesizer_response.get("parse_error", False),
                "analyst_assessments": {
                    name: {
                        "decision": a.get("decision"),
                        "confidence": a.get("confidence"),
                        "rationale": a.get("rationale"),
                        "parse_error": a.get("parse_error", False),
                    }
                    for name, a in zip(analyst_names, assessments)
                },
                "synthesizer_num_promising": synthesizer_response.get("num_promising"),
                "synthesizer_conflicts": synthesizer_response.get("conflicts"),
            }

            if synthesizer_response.get("parse_error"):
                errors += 1

            results.append(record)
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 5 == 0 or (i + 1) == len(df_eval):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(df_eval) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1}/{len(df_eval)}]  "
                    f"elapsed: {elapsed:.0f}s  "
                    f"est. remaining: {remaining:.0f}s  "
                    f"synthesizer errors: {errors}"
                )

    elapsed_total = time.time() - start_time

    # Compute metrics
    y_true = [r["target"] for r in results]
    y_prob = [r["probability_float"] for r in results]

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    print_metrics(metrics, label=f"Multi-Analyst Pipeline — {model}")

    # Save outputs
    with open(output_dir / f"multi_{split_name}_{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Aggregate cache stats from all agents
    cache_stats = market_analyst.llm_client.get_cache_stats()
    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablation": "multi",
        "split": split_name,
        "model": model,
        "n_evaluated": int(len(results)),
        "n_synthesizer_errors": int(errors),
        "elapsed_seconds": float(round(elapsed_total, 1)),
        "threshold": float(threshold),
        "random_state": int(random_state),
        "analysts": analyst_names,
        "cache_stats": cache_stats,
    }

    with open(output_dir / f"multi_{split_name}_{model_name}_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Multi-analyst results saved to {output_dir}/")
    if cache_stats.get("total_cost_usd", 0) > 0:
        print(f"  API cost: ${cache_stats['total_cost_usd']:.4f}  "
              f"(est. cache savings: ${cache_stats.get('estimated_savings_usd', 0):.4f})")

    return metrics


def run_textgrad_multi_analyst(
    df_eval,
    split_name: str,
    model: str = "ollama/glm4:latest",
    random_state: int = 42,
    threshold: float = 0.5,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """
    TextGrad condition: 4 specialists (in parallel) + TextGrad-optimized synthesizer.

    Loads the optimized prompt from results/textgrad_validation/final_synthesizer_prompt.txt.
    Falls back to the default synthesizer prompt if that file does not exist.

    Args:
        df_eval: DataFrame to evaluate
        split_name: Name of the dataset split
        model: Model to use for all agents
        random_state: Seed for reproducibility
        threshold: Decision threshold

    Returns:
        Metrics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    model_name = model.replace("/", "_").replace(".", "_")
    predictions_path = output_dir / f"textgrad_{split_name}_{model_name}_predictions.jsonl"
    results = []
    errors = 0

    # Initialize agents
    market_analyst = MarketAnalyst(model=model)
    biz_analyst = BusinessModelAnalyst(model=model)
    feasibility_analyst = FeasibilityAnalyst(model=model)
    team_analyst = TeamAnalyst(model=model)
    synthesizer = TextGradSynthesizer(model=model)

    analysts = [market_analyst, biz_analyst, feasibility_analyst, team_analyst]
    analyst_names = ["Market", "Business Model", "Feasibility", "Team"]

    print(f"Running TextGrad multi-analyst pipeline ({model}) on {len(df_eval)} rows...\n")
    start_time = time.time()

    with open(predictions_path, "w") as f_out:
        eval_data = [
            (idx, row, analysts, format_startup_profile(row))
            for idx, row in df_eval.iterrows()
        ]

        for i, row_data in enumerate(eval_data):
            idx, row, profile, assessments = _evaluate_multi_analyst(row_data)

            synthesizer_response = synthesizer.synthesize(profile, assessments)

            n_parse_errors = sum(1 for a in assessments if a.get("parse_error"))

            record = {
                "object_id": row.get("object_id"),
                "name": row.get("name"),
                "target": int(row["target"]),
                "category_code": row.get("category_code"),
                "probability_float": synthesizer_response.get("probability", 50) / 100.0,
                "decision": synthesizer_response.get("decision"),
                "reasoning": synthesizer_response.get("reasoning"),
                "parse_error": synthesizer_response.get("parse_error", False),
                "analyst_assessments": {
                    name: {
                        "decision": a.get("decision"),
                        "confidence": a.get("confidence"),
                        "rationale": a.get("rationale"),
                        "parse_error": a.get("parse_error", False),
                    }
                    for name, a in zip(analyst_names, assessments)
                },
                "synthesizer_num_promising": synthesizer_response.get("num_promising"),
                "synthesizer_conflicts": synthesizer_response.get("conflicts"),
            }

            if synthesizer_response.get("parse_error"):
                errors += 1

            results.append(record)
            f_out.write(json.dumps(record) + "\n")

            if (i + 1) % 5 == 0 or (i + 1) == len(df_eval):
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = (len(df_eval) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i+1}/{len(df_eval)}]  "
                    f"elapsed: {elapsed:.0f}s  "
                    f"est. remaining: {remaining:.0f}s  "
                    f"synthesizer errors: {errors}"
                )

    elapsed_total = time.time() - start_time

    y_true = [r["target"] for r in results]
    y_prob = [r["probability_float"] for r in results]

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    print_metrics(metrics, label=f"TextGrad Multi-Analyst — {model}")

    with open(output_dir / f"textgrad_{split_name}_{model_name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cache_stats = market_analyst.llm_client.get_cache_stats()
    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "ablation": "textgrad",
        "split": split_name,
        "model": model,
        "n_evaluated": int(len(results)),
        "n_synthesizer_errors": int(errors),
        "elapsed_seconds": float(round(elapsed_total, 1)),
        "threshold": float(threshold),
        "random_state": int(random_state),
        "analysts": analyst_names,
        "cache_stats": cache_stats,
    }

    with open(output_dir / f"textgrad_{split_name}_{model_name}_run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"TextGrad results saved to {output_dir}/")
    if cache_stats.get("total_cost_usd", 0) > 0:
        print(f"  API cost: ${cache_stats['total_cost_usd']:.4f}  "
              f"(est. cache savings: ${cache_stats.get('estimated_savings_usd', 0):.4f})")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study: random baseline vs. single agent vs. multi-analyst vs. textgrad"
    )
    parser.add_argument(
        "--ablation",
        choices=["random", "single", "multi", "textgrad"],
        required=True,
        help="Which ablation condition to run",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ollama/glm4:9b",
        help="Model to use (ignored for random baseline). "
        "Examples: ollama/glm4:9b, claude-haiku-4-5-20251001, ollama/mistral, ollama/qwen",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Which data split to evaluate (default: test)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Sample N rows for a quick test (e.g. --sample 50)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for binary metrics (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to write results (default: results/ablation)",
    )
    parser.add_argument(
        "--temporal",
        action="store_true",
        default=False,
        help=(
            "Use temporal (first-funding-year) splits instead of random stratified splits. "
            "Pools: train<=2008 / val=2009 / test>=2010. Class ratios (50/50 val, 10/90 test) "
            "are still enforced by sampling within each temporal pool."
        ),
    )
    parser.add_argument(
        "--temporal_train_end",
        type=int,
        default=2008,
        help="Temporal split: train pool upper year, inclusive (default: 2008)",
    )
    parser.add_argument(
        "--temporal_val_start",
        type=int,
        default=2009,
        help="Temporal split: val pool lower year, inclusive (default: 2009)",
    )
    parser.add_argument(
        "--temporal_val_end",
        type=int,
        default=2009,
        help="Temporal split: val pool upper year, inclusive (default: 2009)",
    )

    args = parser.parse_args()
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_run_dir(RESULTS_DIR.resolve())

    # Load splits — always use SPLIT_SEED (not the training seed) so all
    # experiment seeds evaluate on the same train/val/test partition.
    if args.temporal:
        df_train, df_val, df_test = get_temporal_splits(
            train_end_year=args.temporal_train_end,
            val_start_year=args.temporal_val_start,
            val_end_year=args.temporal_val_end,
            random_state=SPLIT_SEED,
        )
    else:
        df_train, df_val, df_test = get_splits(random_state=SPLIT_SEED)
    split_map = {"train": df_train, "val": df_val, "test": df_test}
    df_eval = split_map[args.split]

    if args.sample is not None:
        df_eval = df_eval.sample(
            min(args.sample, len(df_eval)), random_state=args.seed
        )
        print(f"\nUsing random sample of {len(df_eval)} rows from '{args.split}' split.")
    else:
        print(f"\nEvaluating on full '{args.split}' split ({len(df_eval)} rows).")

    # Run appropriate ablation condition
    if args.ablation == "random":
        metrics = run_random_baseline(df_eval, split_name=args.split, random_state=args.seed, threshold=args.threshold, output_dir=output_dir)
    elif args.ablation == "single":
        metrics = run_single_agent(
            df_eval, split_name=args.split, model=args.model, random_state=args.seed, threshold=args.threshold, output_dir=output_dir
        )
    elif args.ablation == "multi":
        metrics = run_multi_analyst(
            df_eval, split_name=args.split, model=args.model, random_state=args.seed, threshold=args.threshold, output_dir=output_dir
        )
    elif args.ablation == "textgrad":
        metrics = run_textgrad_multi_analyst(
            df_eval, split_name=args.split, model=args.model, random_state=args.seed, threshold=args.threshold, output_dir=output_dir
        )

    print(f"\n✓ Ablation '{args.ablation}' completed successfully")


if __name__ == "__main__":
    main()
