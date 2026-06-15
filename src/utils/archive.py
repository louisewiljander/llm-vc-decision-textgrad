"""
Utility functions for managing experimental results and preventing overwrites.
"""
from pathlib import Path
from datetime import datetime


def archive_old_results(results_dir: Path) -> None:
    """
    Archive existing results to timestamped subdirectory before running new experiment.
    
    This prevents accidental overwriting of previous experimental runs. All existing
    result files (*.jsonl, *.json, *.txt) are moved to results_dir/archive/TIMESTAMP/.
    
    Args:
        results_dir: Path to results directory to check for existing files
    
    Example:
        >>> from src.utils.archive import archive_old_results
        >>> from pathlib import Path
        >>> archive_old_results(Path("results/ablation"))
        ✓ Archived previous results to: results/ablation/archive/2026-06-02_14-30-45/
    """
    if not results_dir.exists():
        return
    
    # Check if there are any existing results files
    existing_files = (
        list(results_dir.glob("*.jsonl")) + 
        list(results_dir.glob("*.json")) + 
        list(results_dir.glob("*.txt"))
    )
    
    if not existing_files:
        return
    
    # Create timestamped archive directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    archive_dir = results_dir / "archive" / timestamp
    archive_dir.mkdir(parents=True, exist_ok=True)
    
    # Move existing files to archive
    for file_path in existing_files:
        if file_path.is_file():
            file_path.rename(archive_dir / file_path.name)
    
    print(f"✓ Archived previous results to: {archive_dir}/\n")
