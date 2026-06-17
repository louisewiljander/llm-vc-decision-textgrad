"""
Reconstruct judge evaluation results from printed Colab cell output.

The judge prints each startup in the format:
  Startup: <name> (target=INVEST|PASS)
  [single   ] decision=X  prod=N  mark=N  feas=N  team=N  reas=N  risk=N  total=N
  [multi    ] ...
  [textgrad ] ...

This script parses that text and writes a judge_scores_recovered.jsonl to
results/judge_evaluation/ so the file can be used as --resume_from input
for a subsequent run of run_judge_evaluation.py.

Usage:
    python scripts/recover_judge_output.py --input output.txt
    python scripts/recover_judge_output.py  # reads from stdin

Then resume the judge eval from where it left off:
    python experiments/run_judge_evaluation.py \
        --n_sample 30 --seed 42 \
        --resume_from results/judge_evaluation/judge_scores_recovered.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results" / "judge_evaluation"

DIM_KEYS = {
    "prod": "product_novelty",
    "mark": "market_opportunity",
    "feas": "feasibility",
    "team": "team_quality",
    "reas": "reasoning_coherence",
    "risk": "risk_identification",
}


def parse_output(text: str) -> list[dict]:
    records = []

    startup_pattern = re.compile(
        r"Startup:\s+(.+?)\s+\(target=(INVEST|PASS)\)"
    )
    condition_pattern = re.compile(
        r"\[(\w+)\s*\]\s+decision=(\S+)"
        r"\s+prod=(\d+)\s+mark=(\d+)\s+feas=(\d+)"
        r"\s+team=(\d+)\s+reas=(\d+)\s+risk=(\d+)"
        r"\s+total=(\d+)"
    )

    current_name = None
    current_target = None

    for line in text.splitlines():
        m = startup_pattern.search(line)
        if m:
            current_name = m.group(1).strip()
            current_target = 1 if m.group(2) == "INVEST" else 0
            continue

        m = condition_pattern.search(line)
        if m and current_name is not None:
            condition = m.group(1).strip().lower()
            decision = m.group(2)
            prod, mark, feas, team, reas, risk, total = (
                int(m.group(i)) for i in range(3, 10)
            )
            record = {
                "object_id": None,   # unknown from print output
                "name": current_name,
                "target": current_target,
                "condition": condition,
                "decision": decision,
                "product_novelty_score": prod,
                "market_opportunity_score": mark,
                "feasibility_score": feas,
                "team_quality_score": team,
                "reasoning_coherence_score": reas,
                "risk_identification_score": risk,
                "total_score": total,
                # Justifications unavailable from print output
                "product_novelty_justification": "(recovered from print output)",
                "market_opportunity_justification": "(recovered from print output)",
                "feasibility_justification": "(recovered from print output)",
                "team_quality_justification": "(recovered from print output)",
                "reasoning_coherence_justification": "(recovered from print output)",
                "risk_identification_justification": "(recovered from print output)",
            }
            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None,
                        help="Path to text file with printed cell output (default: stdin)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path (default: results/judge_evaluation/judge_scores_recovered.jsonl)")
    args = parser.parse_args()

    if args.input:
        text = Path(args.input).read_text()
    else:
        print("Paste the cell output below, then press Ctrl-D (or Ctrl-Z on Windows):",
              file=sys.stderr)
        text = sys.stdin.read()

    records = parse_output(text)

    if not records:
        print("No records found — check the input format.", file=sys.stderr)
        sys.exit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / "judge_scores_recovered.jsonl"

    with open(out_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(records)} records to {out_path}")

    # Print summary
    from collections import Counter
    ctr = Counter((r["name"], r["condition"]) for r in records)
    startups = sorted({r["name"] for r in records})
    conditions = ["single", "multi", "textgrad"]
    print(f"\nRecovered {len(startups)} startups × conditions:")
    for name in startups:
        row = "  " + name + ": " + ", ".join(
            c for c in conditions if (name, c) in ctr
        )
        print(row)

    print(f"\nTo resume the judge run, add this flag to run_judge_evaluation.py:")
    print(f"  --resume_from {out_path}")

    # Note: object_ids are unknown from print output.
    # The resume logic in run_judge_evaluation.py matches on (object_id, condition),
    # but object_ids are None here. We need to match by name instead.
    incomplete = [name for name in startups
                  if not all((name, c) in ctr for c in conditions)]
    if incomplete:
        print(f"\n⚠  Incomplete startups (partial data): {incomplete}")
        print("   These will be re-evaluated on the next run.")


if __name__ == "__main__":
    main()
