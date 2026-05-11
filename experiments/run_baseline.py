"""
Baseline experiment: single InvestorAgent evaluated against the test split.

Usage
-----
    python experiments/run_baseline.py              # full test set (10/90)
    python experiments/run_baseline.py --sample 30  # quick smoke test
    python experiments/run_baseline.py --split val  # evaluate on val set (50/50)

Output
------
  results/baseline/predictions.jsonl   — one JSON line per startup
  results/baseline/metrics.json        — aggregate metric scores
  results/baseline/run_info.json       — metadata (timestamp, cost, config)
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
import sys
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.agents.investor import InvestorAgent
from src.evaluation.metrics import compute_metrics, print_metrics
from src.prompts.templates import format_startup_profile
from src.utils.data_splits import get_splits

RESULTS_DIR = Path("results/baseline")


class _MockLLMClient:
    model = "mock-investor"

    def get_cache_stats(self) -> dict:
        return {
            "total_calls": 0,
            "total_input_tokens": 0,
            "cache_created_tokens": 0,
            "cache_read_tokens": 0,
            "cache_usage_percentage": 0.0,
            "total_cost_usd": 0.0,
            "estimated_savings_usd": 0.0,
        }


class _MockInvestorAgent:
    def __init__(self) -> None:
        self.llm_client = _MockLLMClient()

    def evaluate_startup(self, startup_profile: str) -> dict:
        text = startup_profile.lower()

        score = 45
        if "top university" in text or "top university alumni" in text:
            score += 8
        if "team size" in text:
            score += 5
        if "total funding" in text:
            score += 7
        if "relationships" in text:
            score += 4
        if "milestones" in text:
            score += 3
        if "pre-funding" in text:
            score -= 6

        funding_match = re.search(r"TOTAL FUNDING:\s*\$([\d,]+)", startup_profile)
        if funding_match:
            funding_value = int(funding_match.group(1).replace(",", ""))
            if funding_value >= 10_000_000:
                score += 6
            elif funding_value == 0:
                score -= 5

        score = max(5, min(95, score))
        decision = "INVEST" if score >= 50 else "PASS"

        return {
            "decision": decision,
            "probability": score,
            "probability_raw": score,
            "probability_float": score / 100.0,
            "market_assessment": "Mock assessment based on simple heuristics.",
            "team_assessment": "Mock assessment based on team-related fields in the profile.",
            "funding_assessment": "Mock assessment based on funding-related fields in the profile.",
            "key_risks": ["Offline smoke test mode", "Heuristic scoring only"],
            "reasoning": "Deterministic offline mock response used to validate the pipeline without API access.",
        }


def run_baseline(
    split: str = "test",
    sample: int | None = None,
    random_state: int = 42,
    threshold: float = 0.5,
    offline_smoke: bool = False,
) -> dict:
    """
    Run the baseline investor agent on the specified data split.

    Args:
        split:        "train", "val", or "test".
        sample:       If set, randomly sample this many rows (for quick tests).
        random_state: Seed — must match get_splits() to avoid data leakage.
        threshold:    Decision threshold for binary classification metrics.
        offline_smoke: If True, use a deterministic mock agent and skip API calls.

    Returns:
        Metrics dictionary.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # -- Load splits ---------------------------------------------------------
    df_train, df_val, df_test = get_splits(random_state=random_state)
    split_map = {"train": df_train, "val": df_val, "test": df_test}
    df_eval = split_map[split]

    if sample is not None:
        df_eval = df_eval.sample(
            min(sample, len(df_eval)), random_state=random_state
        )
        print(f"\nUsing random sample of {len(df_eval)} rows from '{split}' split.")
    else:
        print(f"\nEvaluating on full '{split}' split ({len(df_eval)} rows).")

    # -- Initialise agent ----------------------------------------------------
    agent = _MockInvestorAgent() if offline_smoke else InvestorAgent(use_cache=True)

    # -- Evaluation loop -----------------------------------------------------
    predictions_path = RESULTS_DIR / "predictions.jsonl"
    results = []
    errors = 0

    print(f"Running evaluations...\n")
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

    # -- Compute metrics -----------------------------------------------------
    y_true = [r["target"] for r in results]
    y_prob = [r["probability_float"] for r in results]

    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    print_metrics(metrics, label=f"Baseline Investor Agent — {split} split")

    # -- Save outputs --------------------------------------------------------
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    cache_stats = agent.llm_client.get_cache_stats()
    run_info = {
        "timestamp": datetime.utcnow().isoformat(),
        "split": split,
        "n_evaluated": len(results),
        "n_parse_errors": errors,
        "elapsed_seconds": round(elapsed_total, 1),
        "threshold": threshold,
        "random_state": random_state,
        "model": agent.llm_client.model,
        "cache_stats": cache_stats,
    }

    with open(RESULTS_DIR / "run_info.json", "w") as f:
        json.dump(run_info, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}/")
    print(f"  API cost: ${cache_stats.get('total_cost_usd', 0):.4f}  "
          f"(est. cache savings: ${cache_stats.get('estimated_savings_usd', 0):.4f})")
    if offline_smoke:
        print("  Mode: offline smoke test (no Anthropic API calls)")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the baseline investor agent experiment."
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="test",
        help="Which data split to evaluate (default: test).",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Sample N rows for a quick test (e.g. --sample 30).",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Decision threshold for binary metrics (default: 0.5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed — must be consistent across all experiments (default: 42).",
    )
    parser.add_argument(
        "--offline-smoke", action="store_true",
        help="Run with a deterministic mock investor and no API calls.",
    )
    args = parser.parse_args()

    run_baseline(
        split=args.split,
        sample=args.sample,
        random_state=args.seed,
        threshold=args.threshold,
        offline_smoke=args.offline_smoke,
    )
