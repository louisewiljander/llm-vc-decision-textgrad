"""
Dataset splitting for the LLM-VC-TextGrad experiments.

Target variable
---------------
`target` = 1 if the company received subsequent funding within 365 days of its
first funding date (derived from funding_rounds.csv), else 0. This is a
funding-behaviour signal observable within a 1-year horizon. Companies first funded
in 2013 are excluded from the dataset because their 1-year follow-on window
extends past the 2013 Crunchbase snapshot.

Split design
------------
Temporal boundaries (defaults):
  Train pool : first_funding_at <= 2008  (or null)
  Val pool   : first_funding_at == 2009
  Test pool  : first_funding_at >= 2010

- Test  (natural ~23%): 300 rows drawn from the test pool. The class
  distribution is natural — no forced ratio — reflecting the ~22.77%
  success rate of the full dataset.

- Val   (50/50): 200 rows — 100 success + 100 failure. Balanced for
  TextGrad optimisation; the gradient signal should see equal examples of
  both classes. Drawn from the year band between train_end_year and
  test_min_year.

- Train (50/50): All remaining rows in the train pool, balanced by
  undersampling the majority class. Unused majority-class rows are
  discarded.

All splits are deterministic given the same random_state.
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
    val_size: int = 200,
    train_end_year: int | None = 2008,
    test_min_year: int | None = 2010,
    contamination_exclusion_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split the processed dataset into train / val / test.

    Three temporally ordered pools are defined by first_funding_at year:
      Train pool : <= train_end_year  (or null first_funding_at)
      Val pool   : (train_end_year, test_min_year) exclusive
      Test pool  : >= test_min_year

    Test uses the natural class distribution. Val is balanced 50/50 for
    TextGrad optimisation. Train is balanced 50/50 by undersampling.

    Args:
        random_state:       Seed for all random sampling — fix this for
                            reproducibility across baseline and TextGrad runs.
        test_size:          Total number of rows in the test set.
        val_size:           Total number of rows in the val set (balanced 50/50).
        train_end_year:     Inclusive upper year for the train pool. Companies
                            with first_funding_at <= this year (or null) go to
                            train. Set None alongside test_min_year=None to
                            disable all temporal constraints.
        test_min_year:      Inclusive lower year for the test pool. The val pool
                            spans (train_end_year, test_min_year) exclusively.
                            Set None to disable temporal constraints.
        contamination_exclusion_path:
                            Optional CSV containing object_id values to exclude
                            from the test split after contamination probing.

    Returns:
        (df_train, df_val, df_test) — three non-overlapping DataFrames.

    Raises:
        ValueError: If any pool is too small for the requested sample sizes.
    """
    df = pd.read_parquet(DATA_PATH)

    # ------------------------------------------------------------------ Pools
    if test_min_year is not None and train_end_year is not None:
        if train_end_year >= test_min_year:
            raise ValueError(
                f"train_end_year ({train_end_year}) must be < test_min_year ({test_min_year})."
            )
        funding_year = pd.to_datetime(df["first_funding_at"], errors="coerce").dt.year
        test_pool  = df[funding_year >= test_min_year]
        val_pool   = df[(funding_year > train_end_year) & (funding_year < test_min_year)]
        train_pool = df[(funding_year <= train_end_year) | funding_year.isna()]
        late_remainder = pd.DataFrame(columns=df.columns)
    else:
        # No temporal constraints — sample test randomly, val/train from remainder.
        test_pool  = df
        val_pool   = None   # determined after test sampling below
        train_pool = None
        late_remainder = pd.DataFrame(columns=df.columns)

    # ------------------------------------------------------------------ Test
    if test_min_year is not None and train_end_year is not None:
        if len(test_pool) < test_size:
            raise ValueError(
                f"Test pool (first_funding_at >= {test_min_year}) has only "
                f"{len(test_pool)} rows — insufficient for test_size={test_size}."
            )
        df_test = test_pool.sample(n=test_size, random_state=random_state)
    else:
        df_test = df.sample(n=test_size, random_state=random_state)
        remaining = df.drop(index=df_test.index)
        val_pool   = remaining
        train_pool = remaining

    # ----------- Contamination exclusions (optional, test set only) ----------
    excluded_test_ids = _load_contamination_exclusions(contamination_exclusion_path)
    if excluded_test_ids:
        if "object_id" not in df_test.columns:
            print(
                "Warning: contamination exclusions found but object_id column missing; "
                "skipping test-set contamination filtering."
            )
        else:
            contaminated_mask = df_test["object_id"].astype(str).isin(excluded_test_ids)
            n_test_excluded = int(contaminated_mask.sum())

            if n_test_excluded:
                contaminated = df_test.loc[contaminated_mask].copy()
                df_test = df_test.loc[~contaminated_mask].copy()

                # Replacements come from the test pool (same temporal cohort).
                replacement_pool = test_pool.drop(index=df_test.index)
                available = replacement_pool.loc[
                    ~replacement_pool["object_id"].astype(str).isin(excluded_test_ids)
                ]

                if len(available) < n_test_excluded:
                    raise ValueError(
                        "Not enough non-contaminated rows remain to rebuild the test split."
                    )

                replacements = available.sample(n=n_test_excluded, random_state=random_state)
                df_test = pd.concat([df_test, replacements]).sample(
                    frac=1, random_state=random_state
                )

                print(
                    f"  Contamination probe exclusions applied: {n_test_excluded} test rows "
                    f"removed and replaced."
                )

    # ------------------------------------------------------------------- Val
    # Val is sampled from the year band (train_end_year, test_min_year).
    val_pos_pool = val_pool[val_pool["target"] == 1]
    val_neg_pool = val_pool[val_pool["target"] == 0]

    n_val_each = val_size // 2
    if len(val_pos_pool) < n_val_each or len(val_neg_pool) < n_val_each:
        raise ValueError(
            f"Val pool has {len(val_pos_pool)} positives and "
            f"{len(val_neg_pool)} negatives — insufficient for val_size={val_size} at 50/50."
        )

    val_pos = val_pos_pool.sample(n=n_val_each, random_state=random_state)
    val_neg = val_neg_pool.sample(n=n_val_each, random_state=random_state)
    df_val = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=random_state)

    # ----------------------------------------------------------------- Train
    # Train pool is strictly <= train_end_year (late remainder is not used).
    # Undersample the majority class to balance.
    train_pool = train_pool.drop(index=[i for i in df_val.index if i in train_pool.index])
    remaining_pos = train_pool[train_pool["target"] == 1]
    remaining_neg = train_pool[train_pool["target"] == 0]

    n_train_each = min(len(remaining_pos), len(remaining_neg))
    train_pos = remaining_pos.sample(n=n_train_each, random_state=random_state)
    train_neg = remaining_neg.sample(n=n_train_each, random_state=random_state)
    df_train = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=random_state)

    n_discarded = (len(remaining_pos) + len(remaining_neg)) - (2 * n_train_each)

    _print_summary(df_train, df_val, df_test, n_discarded, test_min_year)

    return df_train, df_val, df_test


def _print_summary(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    n_discarded: int,
    test_min_year: int | None = None,
) -> None:
    """Print a compact split summary."""
    test_label = f"Test (first_funding>={test_min_year})" if test_min_year else "Test"
    print("Dataset splits:")
    for name, split in [("Train", df_train), ("Val", df_val), (test_label, df_test)]:
        n_pos = (split["target"] == 1).sum()
        n_neg = (split["target"] == 0).sum()
        print(
            f"  {name:<32} {len(split):>5} rows  "
            f"{n_pos:>4}+ / {n_neg:>4}-  "
            f"({split['target'].mean():.1%} positive)"
        )
    print(f"  (Rows discarded from train to enforce balance: {n_discarded})")



if __name__ == "__main__":
    # Default: test from first_funding_at >= 2010, natural class distribution
    train, val, test = get_splits()
