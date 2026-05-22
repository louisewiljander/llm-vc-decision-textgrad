"""
Dataset splitting for the LLM-VC-TextGrad experiments.

Split design
------------
- Test  (10/90): 300 rows — 30 success + 270 failure.
  Reflects realistic VC success ratios for final evaluation reporting.
  See Wang et al. (2025) for the motivation.

- Val   (50/50): 200 rows — 100 success + 100 failure.
  Balanced for TextGrad optimisation; the gradient signal should see equal
  examples of both classes.

- Train (50/50): All remaining rows, balanced by undersampling the majority
  class (success). Unused success rows are discarded.

Resulting sizes (with default random_state=42):
  Test:  300  (10% positive)
  Val:   200  (50% positive)
  Train: ~1558 (50% positive)

All splits are deterministic given the same random_state. Call
get_splits() to obtain the three DataFrames.
"""
from pathlib import Path
import pandas as pd

# Resolve paths relative to repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()  # Navigate from src/utils/ to repo root
DATA_PATH = str(_REPO_ROOT / "data" / "processed" / "companies_clean.parquet")
DEFAULT_TEST_CONTAMINATION_EXCLUSIONS = str(_REPO_ROOT / "data" / "processed" / "test_contamination_exclusions.csv")


def _load_contamination_exclusions(
    exclusion_path: str | Path | None,
) -> set[str]:
    """Load object IDs that should be excluded from the test split."""
    path = Path(exclusion_path) if exclusion_path is not None else Path(DEFAULT_TEST_CONTAMINATION_EXCLUSIONS)

    if not path.exists():
        return set()

    exclusions = pd.read_csv(path)
    if "object_id" not in exclusions.columns:
        raise ValueError(
            f"Contamination exclusion file {path} must contain an 'object_id' column."
        )

    return {str(value) for value in exclusions["object_id"].dropna().tolist()}


def get_splits(
    random_state: int = 42,
    test_size: int = 300,
    test_positive_rate: float = 0.10,
    val_size: int = 200,
    contamination_exclusion_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split the processed dataset into train / val / test.

    Args:
        random_state:       Seed for all random sampling — fix this for
                            reproducibility across baseline and TextGrad runs.
        test_size:          Total number of rows in the test set.
        test_positive_rate: Fraction of test rows that are positive (success).
        val_size:           Total number of rows in the val set (will be
                            balanced 50/50 regardless of natural distribution).
        contamination_exclusion_path:
                            Optional CSV containing object_id values to exclude
                            from the test split after contamination probing.

    Returns:
        (df_train, df_val, df_test) — three non-overlapping DataFrames.
    """
    df = pd.read_parquet(DATA_PATH)

    success = df[df["target"] == 1]
    failure = df[df["target"] == 0]

    # ------------------------------------------------------------------ Test
    n_test_pos = round(test_size * test_positive_rate)
    n_test_neg = test_size - n_test_pos

    test_pos = success.sample(n=n_test_pos, random_state=random_state)
    test_neg = failure.sample(n=n_test_neg, random_state=random_state)
    df_test = pd.concat([test_pos, test_neg]).sample(
        frac=1, random_state=random_state
    )

    # Remaining pool after reserving the initial test sample.
    remaining_pos = success.drop(index=test_pos.index)
    remaining_neg = failure.drop(index=test_neg.index)

    excluded_test_ids = _load_contamination_exclusions(contamination_exclusion_path)
    n_test_excluded = 0
    if excluded_test_ids:
        if "object_id" not in df_test.columns:
            print(
                "Warning: contamination exclusions were found, but object_id is missing; "
                "skipping test-set contamination filtering."
            )
        else:
            contaminated_mask = df_test["object_id"].astype(str).isin(excluded_test_ids)
            n_test_excluded = int(contaminated_mask.sum())

            if n_test_excluded:
                contaminated = df_test.loc[contaminated_mask].copy()
                df_test = df_test.loc[~contaminated_mask].copy()

                n_replace_pos = int((contaminated["target"] == 1).sum())
                n_replace_neg = n_test_excluded - n_replace_pos

                available_pos = remaining_pos.loc[
                    ~remaining_pos["object_id"].astype(str).isin(excluded_test_ids)
                ]
                available_neg = remaining_neg.loc[
                    ~remaining_neg["object_id"].astype(str).isin(excluded_test_ids)
                ]

                if len(available_pos) < n_replace_pos or len(available_neg) < n_replace_neg:
                    raise ValueError(
                        "Not enough non-contaminated rows remain to rebuild the test split."
                    )

                replacement_pos = available_pos.sample(
                    n=n_replace_pos, random_state=random_state
                ) if n_replace_pos else available_pos.iloc[0:0]
                replacement_neg = available_neg.sample(
                    n=n_replace_neg, random_state=random_state
                ) if n_replace_neg else available_neg.iloc[0:0]

                df_test = pd.concat([df_test, replacement_pos, replacement_neg]).sample(
                    frac=1, random_state=random_state
                )

                remaining_pos = remaining_pos.drop(index=replacement_pos.index)
                remaining_neg = remaining_neg.drop(index=replacement_neg.index)

                print(
                    f"  Contamination probe exclusions applied: {n_test_excluded} test rows "
                    f"removed and replaced."
                )

    # ------------------------------------------------------------------- Val
    n_val_each = val_size // 2
    val_pos = remaining_pos.sample(n=n_val_each, random_state=random_state)
    val_neg = remaining_neg.sample(n=n_val_each, random_state=random_state)
    df_val = pd.concat([val_pos, val_neg]).sample(
        frac=1, random_state=random_state
    )

    remaining_pos = remaining_pos.drop(index=val_pos.index)
    remaining_neg = remaining_neg.drop(index=val_neg.index)

    # ----------------------------------------------------------------- Train
    # Undersample majority (success) to balance
    n_train_each = min(len(remaining_pos), len(remaining_neg))
    train_pos = remaining_pos.sample(n=n_train_each, random_state=random_state)
    train_neg = remaining_neg.sample(n=n_train_each, random_state=random_state)
    df_train = pd.concat([train_pos, train_neg]).sample(
        frac=1, random_state=random_state
    )

    n_discarded = len(remaining_pos) - n_train_each

    _print_summary(df_train, df_val, df_test, n_discarded)

    return df_train, df_val, df_test


def _print_summary(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_discarded: int,
) -> None:
    """Print a compact split summary."""
    print("Dataset splits:")
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        n_pos = (df["target"] == 1).sum()
        n_neg = (df["target"] == 0).sum()
        print(
            f"  {name:<6} {len(df):>5} rows  "
            f"{n_pos:>4}+ / {n_neg:>4}-  "
            f"({df['target'].mean():.0%} positive)"
        )
    print(f"  (Success rows discarded from train to enforce balance: {n_discarded})")


if __name__ == "__main__":
    train, val, test = get_splits()
