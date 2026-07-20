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
  Train pool : 2005 <= first_funding_at <= 2008
  Val pool   : first_funding_at == 2009
  Test pool  : first_funding_at >= 2010

- Test  (natural): 300 rows drawn from the test pool. The class
  distribution is natural — no forced ratio — reflecting the positive rate
  of the test cohort.

- Val   (50/50): 200 rows — 100 success + 100 failure. Balanced for
  TextGrad optimisation; the gradient signal should see equal examples of
  both classes. Drawn from the year band between train_end_year and
  test_min_year.

- Train (50/50): All remaining rows in the train pool, balanced by
  undersampling the majority class. Unused majority-class rows are
  discarded.

All splits are deterministic given the same random_state.

Contamination exclusions
------------------------
Companies for which an LLM contamination probe returned high confidence of
identification are excluded from the test set. The exclusion list is hardcoded below as
TEST_CONTAMINATION_EXCLUSION_IDS. It is versioned here rather than in a data
file so it survives a fresh clone.
"""
from pathlib import Path
import pandas as pd

# Resolve paths relative to repo root
_REPO_ROOT = Path(__file__).parent.parent.parent.resolve()
DATA_PATH = str(_REPO_ROOT / "data" / "processed" / "companies_clean.parquet")

# Manually curated contamination exclusions from the pre-run probe.
# These object_ids were identified by an LLM probe as likely recognisable
# from their anonymised descriptions and are excluded from the test set.
TEST_CONTAMINATION_EXCLUSION_IDS: frozenset[str] = frozenset({
    'c:63233', 'c:142810', 'c:170379', 'c:143872', 'c:240150', 'c:179637', 'c:224653', 'c:47870'
})


def get_splits(
    random_state: int = 42,
    test_size: int = 300,
    val_size: int = 200,
    train_min_year: int | None = 2005,
    train_end_year: int | None = 2008,
    test_min_year: int | None = 2010,
    exclude_contaminated: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split the processed dataset into train / val / test.

    Three temporally ordered pools are defined by first_funding_at year:
      Train pool : train_min_year <= first_funding_at <= train_end_year
      Val pool   : (train_end_year, test_min_year) exclusive
      Test pool  : >= test_min_year

    Test uses the natural class distribution. Val is balanced 50/50 for
    TextGrad optimisation. Train is balanced 50/50 by undersampling.

    Args:
        random_state:       Seed for all random sampling — fix this for
                            reproducibility across baseline and TextGrad runs.
        test_size:          Total number of rows in the test set.
        val_size:           Total number of rows in the val set (balanced 50/50).
        train_min_year:     Inclusive lower year for the train pool. Companies
                            with first_funding_at < this year are excluded as
                            data-quality outliers. Default: 2005.
        train_end_year:     Inclusive upper year for the train pool. Set None
                            alongside test_min_year=None to disable temporal
                            constraints entirely.
        test_min_year:      Inclusive lower year for the test pool. The val pool
                            spans (train_end_year, test_min_year) exclusively.
                            Set None to disable temporal constraints.
        exclude_contaminated:
                            If True (default), remove TEST_CONTAMINATION_EXCLUSION_IDS
                            from the test set. Removed rows are not replaced, so the
                            returned test set may be smaller than test_size.

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
        # Train pool: train_min_year <= first_funding_at <= train_end_year.
        # Null first_funding_at rows are absent — the notebook drops them before saving.
        train_mask = (funding_year <= train_end_year)
        if train_min_year is not None:
            train_mask = train_mask & (funding_year >= train_min_year)
        train_pool = df[train_mask]
    else:
        # No temporal constraints — sample test randomly, val/train from remainder.
        test_pool  = df
        val_pool   = None
        train_pool = None

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

    # ----------- Contamination exclusions (test set only) --------------------
    if exclude_contaminated and TEST_CONTAMINATION_EXCLUSION_IDS:
        contaminated_mask = df_test["object_id"].astype(str).isin(
            TEST_CONTAMINATION_EXCLUSION_IDS
        )
        n_excluded = int(contaminated_mask.sum())

        if n_excluded:
            df_test = df_test.loc[~contaminated_mask].copy()
            print(f"  Contamination exclusions applied: {n_excluded} test rows removed (test_size now {len(df_test)}).")

    # ------------------------------------------------------------------- Val
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
    train, val, test = get_splits()
