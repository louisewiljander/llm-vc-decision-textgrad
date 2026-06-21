"""
Post-hoc qualitative evaluation using LLM-as-a-judge.

Evaluates reasoning quality across ablation conditions (single agent, multi-analyst,
TextGrad) on a shared sample of startups. Grounded in innovation theory (Schumpeter's types of innovation) and VC evaluation criteria.

Dimensions:
  1. Product novelty       (Schumpeter type 1)
  2. Market opportunity    (Schumpeter type 3)
  3. Feasibility           (Schumpeter type 2)
  4. Team quality
  5. Reasoning coherence
  6. Risk identification

Usage:
    python experiments/run_judge_evaluation.py --n_sample 5 --judge_model groq/llama-3.3-70b-versatile
    python experiments/run_judge_evaluation.py --n_sample 5 --judge_model groq/llama-3.3-70b-versatile
    python experiments/run_judge_evaluation.py --single path/to/single.jsonl --multi path/to/multi.jsonl --textgrad path/to/textgrad.jsonl
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from litellm import completion

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.prompts.templates import format_startup_profile
from src.utils.data_splits import get_splits
from src.utils.archive import make_run_dir

RESULTS_DIR  = REPO_ROOT / "results" / "judge_evaluation"
ABLATION_DIR = REPO_ROOT / "results" / "ablation"

DEFAULT_JUDGE_MODEL = "groq/llama-3.3-70b-versatile"

# ─── Rubrics ──────────────────────────────────────────────────────────────────

RUBRICS = """
DIMENSION RUBRICS (score each 1–5):

1. PRODUCT NOVELTY (Schumpeter type 1 — introduction of a new good or new quality of a good)
   Does the reasoning assess what is new or distinctive about the startup's product or service?
   5 = Clearly identifies the specific differentiator and explains why it is novel
   4 = Acknowledges novelty with some specificity but does not fully explain the differentiator
   3 = Mentions the product but does not assess novelty
   2 = Product description is generic/paraphrased with no evaluative content
   1 = No assessment of the product

2. MARKET OPPORTUNITY (Schumpeter type 3 — exploitation of a new market)
   Does the reasoning evaluate the size, timing, or accessibility of the target market?
   5 = Assesses market opportunity with specificity (sector context, timing, addressable demand)
   4 = Addresses market opportunity with some nuance but relies partly on generic claims
   3 = Mentions the market but does not evaluate the opportunity
   2 = Market reference is a restatement of the category with no evaluation
   1 = No market assessment

3. FEASIBILITY (Schumpeter type 2 — new method of production; operational/technical viability)
   Does the reasoning evaluate whether the startup can realistically execute its proposition?
   5 = Explicitly assesses execution viability with specific evidence; identifies feasibility risks or strengths
   4 = Addresses feasibility without specific evidence from the analyst reports
   3 = Mentions execution or traction without evaluating feasibility
   2 = Feasibility is implied but not assessed
   1 = No feasibility assessment

4. TEAM QUALITY (Gompers et al. 2020 - VCs heavily weighs founding team quality in investment decisions)
   Does the reasoning evaluate the founding team's ability to execute?
   5 = Assesses team quality with specificity (credentials, experience, composition); connects team to thesis
   4 = Mentions team quality with some reasoning but lacks specificity
   3 = References the team without evaluating quality
   2 = Team is mentioned only in passing
   1 = No team assessment

5. REASONING COHERENCE
   Does the final decision follow logically from the inputs provided?
   5 = Decision fully consistent with inputs; reasoning explicitly connects inputs to conclusion; no logical gaps
   4 = Decision consistent with inputs; minor gaps in reasoning chain
   3 = Decision is plausible but does not clearly derive from the inputs
   2 = Decision is inconsistent with or unexplained by the inputs
   1 = Decision contradicts the inputs with no justification

6. RISK IDENTIFICATION
   Does the reasoning surface meaningful risks present in the analyst reports (or startup profile for single agent)?
   5 = Identifies specific risks; explains how risks were weighed in the decision
   4 = Identifies risks but does not fully explain how they affected the decision
   3 = Mentions risks generically without connecting to specific signals
   2 = Risks are underweighted or overlooked despite being present in the inputs
   1 = No risk identification
"""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of venture capital investment reasoning.
You evaluate the quality of an analyst's reasoning about a startup investment decision.
You score reasoning on 6 dimensions using the provided rubrics.
You must respond with valid JSON only — no markdown, no preamble."""


def build_judge_prompt(
    startup_profile: str,
    analyst_assessments: dict | None,
    decision_output: dict,
    condition: str,
) -> str:
    """Build the judge input prompt for a single startup + condition."""

    sections = [f"STARTUP PROFILE:\n{startup_profile}\n"]

    if analyst_assessments:
        analyst_text = "\nANALYST REPORTS:\n"
        for name, assessment in analyst_assessments.items():
            if assessment.get("parse_error"):
                analyst_text += f"\n{name} Analyst: [parse error]\n"
            else:
                analyst_text += (
                    f"\n{name} Analyst:\n"
                    f"  Decision: {assessment.get('decision', 'N/A')}\n"
                    f"  Confidence: {assessment.get('confidence', 'N/A')}\n"
                    f"  Rationale: {assessment.get('rationale', 'N/A')}\n"
                )
        sections.append(analyst_text)

    sections.append(
        f"\nSYNTHESIZER OUTPUT ({condition.upper()}):\n"
        f"  Decision: {decision_output.get('decision', 'N/A')}\n"
        f"  Probability: {decision_output.get('probability_float', 'N/A')}\n"
        f"  Reasoning: {decision_output.get('reasoning', 'N/A')}\n"
    )

    sections.append(RUBRICS)
    sections.append(
        "\nScore the synthesizer's reasoning on each dimension. "
        "Respond with valid JSON only:\n"
        "{\n"
        '  "product_novelty": {"score": <1-5>, "justification": "<one sentence>"},\n'
        '  "market_opportunity": {"score": <1-5>, "justification": "<one sentence>"},\n'
        '  "feasibility": {"score": <1-5>, "justification": "<one sentence>"},\n'
        '  "team_quality": {"score": <1-5>, "justification": "<one sentence>"},\n'
        '  "reasoning_coherence": {"score": <1-5>, "justification": "<one sentence>"},\n'
        '  "risk_identification": {"score": <1-5>, "justification": "<one sentence>"}\n'
        "}"
    )

    return "\n".join(sections)


DIMENSIONS = [
    "product_novelty", "market_opportunity", "feasibility",
    "team_quality", "reasoning_coherence", "risk_identification",
]


def _extract_scores_via_regex(text: str) -> dict:
    """Fallback: extract just the numeric scores via regex when JSON parse fails."""
    import re
    result = {}
    for dim in DIMENSIONS:
        match = re.search(rf'"{dim}"[^{{]*"score"\s*:\s*([1-5])', text)
        if not match:
            match = re.search(rf'{dim}[^{{]*score[^\d]*([1-5])', text)
        score = int(match.group(1)) if match else 3  # default to mid-point
        result[dim] = {"score": score, "justification": "(parse error — score only)"}
    return result


def call_judge(prompt: str, model: str, retries: int = 3) -> dict:
    """Call the judge LLM and parse the response. Retries on rate limit errors."""
    import time
    for attempt in range(retries):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            text = response.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.rsplit("```", 1)[0]
            try:
                return json.loads(text.strip())
            except json.JSONDecodeError:
                print(f"    JSON parse error — falling back to regex score extraction")
                return _extract_scores_via_regex(text)
        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                wait = 15 * (attempt + 1)
                print(f"    Rate limit hit — waiting {wait}s before retry {attempt + 1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Judge call failed after {retries} retries")


def load_predictions(path: Path) -> dict:
    """Load predictions from a jsonl file, keyed by object_id."""
    preds = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[str(rec["object_id"])] = rec
    return preds


def find_prediction_file(condition: str, ablation_dir: Path) -> Path | None:
    """Auto-discover the most recent predictions file for a condition."""
    # Check the given dir directly (works for both explicit run dirs and latest/)
    matches = sorted(
        ablation_dir.glob(f"{condition}_val_*_predictions.jsonl"),
        key=lambda p: p.stat().st_mtime,
    )
    if matches:
        return matches[-1]
    # New structure: search runs/ subdirs newest-first
    runs_dir = ablation_dir / "runs"
    if runs_dir.exists():
        for run_subdir in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            matches = sorted(run_subdir.glob(f"{condition}_val_*_predictions.jsonl"))
            if matches:
                return matches[-1]
    # Legacy fallback: old archive/ subdirs
    archive_dir = ablation_dir / "archive"
    if archive_dir.exists():
        for archive in sorted(archive_dir.iterdir(), reverse=True):
            matches = sorted(archive.glob(f"{condition}_val_*_predictions.jsonl"))
            if matches:
                return matches[-1]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc qualitative evaluation using LLM-as-judge"
    )
    parser.add_argument("--n_sample", type=int, default=10,
                        help="Number of startups to evaluate (default: 10)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--judge_model", type=str, default=DEFAULT_JUDGE_MODEL,
                        help=f"Judge model (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument("--single", type=str, default=None,
                        help="Path to single agent predictions jsonl")
    parser.add_argument("--multi", type=str, default=None,
                        help="Path to multi-analyst predictions jsonl")
    parser.add_argument("--textgrad", type=str, default=None,
                        help="Path to textgrad predictions jsonl")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to an existing judge scores JSONL — skip already-evaluated (startup, condition) pairs")
    parser.add_argument("--judge_sleep", type=float, default=0,
                        help="Seconds to sleep between judge calls (default: 0). Set to 65 for Groq free tier rate limits.")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to write results (default: new timestamped run under results/judge_evaluation/runs/)")
    parser.add_argument("--ablation_dir", type=str, default=None,
                        help="Directory containing ablation prediction files (default: results/ablation/latest)")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = make_run_dir(RESULTS_DIR)

    if args.ablation_dir:
        ablation_dir = Path(args.ablation_dir)
    else:
        latest = ABLATION_DIR / "latest"
        ablation_dir = latest.resolve() if latest.exists() else ABLATION_DIR

    # ─── Load resume state ─────────────────────────────────────────────────────
    # Keys are (object_id, condition) OR (name, condition) for recovered records
    # where object_id is unknown.
    already_done: set[tuple[str, str]] = set()
    prior_results: list[dict] = []
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            with open(resume_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        condition = rec["condition"]
                        oid = rec.get("object_id")
                        if oid is not None:
                            already_done.add((str(oid), condition))
                        name = rec.get("name")
                        if name:
                            already_done.add((name, condition))  # fallback key
                        prior_results.append(rec)
            print(f"Resuming from {resume_path} — {len(already_done)} done keys.\n")
        else:
            print(f"⚠  --resume_from path not found: {resume_path}\n")

    # ─── Locate prediction files ───────────────────────────────────────────────
    print("Locating prediction files...\n")

    paths = {
        "single":   Path(args.single) if args.single else find_prediction_file("single", ablation_dir),
        "multi":    Path(args.multi) if args.multi else find_prediction_file("multi", ablation_dir),
        "textgrad": Path(args.textgrad) if args.textgrad else find_prediction_file("textgrad", ablation_dir),
    }

    for name, path in paths.items():
        if path is None or not path.exists():
            print(f"✗ Could not find predictions for '{name}'. "
                  f"Run the ablation first or pass --{name} <path>.")
            sys.exit(1)
        print(f"  {name:10s}: {path}")

    # ─── Load predictions ──────────────────────────────────────────────────────
    preds = {name: load_predictions(path) for name, path in paths.items()}

    # Find startups present in all three conditions
    common_ids = set(preds["single"]) & set(preds["multi"]) & set(preds["textgrad"])
    print(f"\n{len(common_ids)} startups found in all three conditions.")

    if len(common_ids) < args.n_sample:
        print(f"⚠  Only {len(common_ids)} common startups — using all of them.")
        args.n_sample = len(common_ids)

    # ─── Load dataset for startup profiles ────────────────────────────────────
    print("Loading dataset...\n")
    df_train, df_val, df_test = get_splits(random_state=args.seed)
    df_all = pd.concat([df_train, df_val, df_test])
    df_all["object_id"] = df_all["object_id"].astype(str)
    profile_map = {
        row["object_id"]: format_startup_profile(row)
        for _, row in df_all.iterrows()
        if row["object_id"] in common_ids
    }

    # Sample
    import random
    random.seed(args.seed)
    sample_ids = random.sample(sorted(common_ids), args.n_sample)

    # ─── Run judge ─────────────────────────────────────────────────────────────
    print(f"Running judge ({args.judge_model}) on {args.n_sample} startups × 3 conditions...\n")
    print("=" * 70)

    results = list(prior_results)  # seed with any resumed results
    dimensions = DIMENSIONS
    incremental_path = output_dir / "judge_scores_incremental.jsonl"

    for object_id in sample_ids:
        startup_name = preds["single"][object_id].get("name", object_id)
        target = preds["single"][object_id].get("target")
        profile = profile_map.get(object_id, "Profile not available")

        conditions_todo = [
            c for c in ["single", "multi", "textgrad"]
            if (str(object_id), c) not in already_done
            and (startup_name, c) not in already_done
        ]
        if not conditions_todo:
            print(f"\nStartup: {startup_name} — already done, skipping.")
            continue

        print(f"\nStartup: {startup_name} (target={'INVEST' if target == 1 else 'PASS'})")
        print("-" * 70)

        for condition in conditions_todo:
            import time
            if args.judge_sleep > 0:
                time.sleep(args.judge_sleep)  # e.g. 65s for Groq free tier (~6K TPM)
            pred = preds[condition][object_id]
            analyst_assessments = pred.get("analyst_assessments")

            prompt = build_judge_prompt(
                startup_profile=profile,
                analyst_assessments=analyst_assessments,
                decision_output=pred,
                condition=condition,
            )

            try:
                scores = call_judge(prompt, args.judge_model)
                record = {
                    "object_id": object_id,
                    "name": startup_name,
                    "target": target,
                    "condition": condition,
                    "decision": pred.get("decision"),
                    **{f"{dim}_score": scores[dim]["score"] for dim in dimensions},
                    **{f"{dim}_justification": scores[dim]["justification"] for dim in dimensions},
                    "total_score": sum(scores[dim]["score"] for dim in dimensions),
                }
                results.append(record)
                already_done.add((str(object_id), condition))
                # Append to incremental file so progress survives a crash
                with open(incremental_path, "a") as _f:
                    _f.write(json.dumps(record) + "\n")

                score_str = "  ".join(
                    f"{d.split('_')[0][:4]}={scores[d]['score']}" for d in dimensions
                )
                print(f"  [{condition:9s}] decision={pred.get('decision'):4s}  {score_str}  total={record['total_score']}")

            except Exception as e:
                print(f"  [{condition:9s}] ERROR: {e}")

    # ─── Save & summarise ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)

    if not results:
        print("\n⚠  No results — no common startups were evaluated.")
        return

    df_results = pd.DataFrame(results)
    output_path = output_dir / "judge_scores.jsonl"
    df_results.to_json(output_path, orient="records", lines=True)
    print(f"\nResults saved to: {output_path}")

    # Summary table
    print("\n── Average scores by condition ──────────────────────────────")
    score_cols = [f"{d}_score" for d in dimensions]
    summary = (
        df_results.groupby("condition")[score_cols + ["total_score"]]
        .mean()
        .round(2)
        .rename(columns={f"{d}_score": d.replace("_", " ") for d in dimensions})
    )
    order = ["single", "multi", "textgrad"]
    summary = summary.reindex([c for c in order if c in summary.index])
    print(summary.to_string())

    summary_path = output_dir / "judge_summary.json"
    summary.to_json(summary_path, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
