"""
Transaction filtering utilities.

Provides helpers to remove counterflow transactions (a single tx_hash that
routes in both WETH→USDC and USDC→WETH directions) and to enforce a minimum
USDC-equivalent trade size.
"""

import pandas as pd


def is_counterflow(directions: list[str]) -> bool:
    """Return True if both swap directions appear in the direction list.

    A transaction is considered counterflow when it simultaneously routes
    WETH→USDC and USDC→WETH (i.e. it is a two-leg trade that partially
    offsets itself).

    Args:
        directions: Collection of direction strings for a single transaction's
                    legs (e.g. ``['WETH->USDC', 'USDC->WETH']``).

    Returns:
        ``True`` if both ``'WETH->USDC'`` and ``'USDC->WETH'`` appear in the
        list, ``False`` otherwise.
    """
    direction_set = set(directions)
    return "WETH->USDC" in direction_set and "USDC->WETH" in direction_set


def filter_counterflow(df: pd.DataFrame) -> pd.DataFrame:
    """Remove all legs of any transaction that routes in both directions.

    Groups the DataFrame by ``tx_hash`` and identifies transactions where
    at least one leg goes WETH→USDC *and* at least one leg goes USDC→WETH.
    All rows belonging to such transactions are dropped entirely.

    Args:
        df: DataFrame with at least columns ``tx_hash`` (str) and
            ``direction`` (str, values ``'WETH->USDC'`` or ``'USDC->WETH'``).

    Returns:
        A filtered copy of ``df`` with counterflow transactions removed.

    Raises:
        KeyError: If required columns are missing from ``df``.
    """
    for col in ("tx_hash", "direction"):
        if col not in df.columns:
            raise KeyError(f"Required column '{col}' not found in DataFrame.")

    counterflow_hashes = (
        df.groupby("tx_hash")["direction"]
        .apply(lambda dirs: is_counterflow(dirs.tolist()))
        .pipe(lambda s: s[s].index)
    )

    return df[~df["tx_hash"].isin(counterflow_hashes)].copy()


def filter_min_usdc(
    df: pd.DataFrame,
    min_usdc: float = 100.0,
) -> pd.DataFrame:
    """Remove transactions below the minimum USDC-equivalent trade size.

    Rows where ``usdc_equiv_size < min_usdc`` are dropped.

    Args:
        df:       DataFrame with at least a ``usdc_equiv_size`` (numeric)
                  column.
        min_usdc: Minimum USDC-equivalent size threshold (default 100.0).

    Returns:
        A filtered copy of ``df`` keeping only rows where
        ``usdc_equiv_size >= min_usdc``.

    Raises:
        KeyError: If ``'usdc_equiv_size'`` column is missing from ``df``.
    """
    if "usdc_equiv_size" not in df.columns:
        raise KeyError("Required column 'usdc_equiv_size' not found in DataFrame.")

    return df[df["usdc_equiv_size"] >= min_usdc].copy()


def apply_all_filters(
    df: pd.DataFrame,
    min_usdc: float = 100.0,
) -> tuple[pd.DataFrame, dict]:
    """Apply counterflow and minimum-size filters in sequence.

    Counterflow filtering is applied first (it is the more aggressive
    structural filter), followed by the USDC size threshold.

    Args:
        df:       Raw transaction DataFrame.  Must contain columns
                  ``tx_hash``, ``direction``, and ``usdc_equiv_size``.
        min_usdc: Minimum USDC-equivalent trade size (default 100.0).

    Returns:
        A 2-tuple ``(filtered_df, filter_stats)`` where:

        - ``filtered_df`` is the cleaned DataFrame (copy).
        - ``filter_stats`` is a dict with keys:

          - ``n_before`` (int): row count before any filtering.
          - ``n_counterflow_removed`` (int): rows removed by counterflow filter.
          - ``n_min_usdc_removed`` (int): rows removed by size filter.
          - ``n_after`` (int): row count in the returned DataFrame.
    """
    n_before = len(df)

    after_counterflow = filter_counterflow(df)
    n_counterflow_removed = n_before - len(after_counterflow)

    after_size = filter_min_usdc(after_counterflow, min_usdc=min_usdc)
    n_min_usdc_removed = len(after_counterflow) - len(after_size)

    n_after = len(after_size)

    filter_stats = {
        "n_before": n_before,
        "n_counterflow_removed": n_counterflow_removed,
        "n_min_usdc_removed": n_min_usdc_removed,
        "n_after": n_after,
    }

    return after_size, filter_stats
