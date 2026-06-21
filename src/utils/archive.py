"""
Utility functions for managing experimental results.
"""
from pathlib import Path
from datetime import datetime


def make_run_dir(base_dir: Path, timestamp: str | None = None) -> Path:
    """
    Create a timestamped run directory and update the `latest` symlink.

    New runs go to base_dir/runs/TIMESTAMP/. The base_dir/latest symlink always
    points to the most recently created run, so code that doesn't care about a
    specific run can just resolve `base_dir/latest`.

    Args:
        base_dir:  Top-level results directory (e.g. results/textgrad_validation).
        timestamp: Optional explicit timestamp; defaults to current local time.

    Returns:
        Path to the new run directory.
    """
    ts = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = base_dir / "runs" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    latest = base_dir / "latest"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(Path("runs") / ts)  # relative symlink

    print(f"✓ Run directory: {run_dir}/")
    return run_dir


def archive_old_results(results_dir: Path) -> None:
    """
    Deprecated — use make_run_dir() for new code.

    Moves existing result files to results_dir/archive/TIMESTAMP/ to prevent
    accidental overwrites.
    """
    if not results_dir.exists():
        return

    existing_files = (
        list(results_dir.glob("*.jsonl")) +
        list(results_dir.glob("*.json")) +
        list(results_dir.glob("*.txt"))
    )

    if not existing_files:
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = results_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)

    for file_path in existing_files:
        if file_path.is_file():
            file_path.rename(archive_dir / file_path.name)

    print(f"✓ Archived previous results to: {archive_dir}/\n")
