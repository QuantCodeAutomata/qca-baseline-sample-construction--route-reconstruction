"""Experiment 4 — Trade-Size Heterogeneity.

Stratifies transactions into USDC-size bins and compares FVO/G-FVO shortfalls
across bins to characterise how routing quality varies with trade size.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import POOL_IDS
except ImportError:
    POOL_IDS = [0, 1, 2, 3]

from exp.exp2_benchmarks import run_experiment as _exp2_run

_RESULTS_DIR = Path("/workspace/project/results")

# Bin definitions: [lower, upper) in USDC
_BIN_EDGES = [100.0, 1_000.0, 10_000.0, 100_000.0, np.inf]
_BIN_LABELS = ["[100, 1K)", "[1K, 10K)", "[10K, 100K)", "[100K, ∞)"]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def assign_size_bins(df: pd.DataFrame, size_col: str = "usdc_equiv_size") -> pd.DataFrame:
    """Add a 'size_bin' column based on USDC-equivalent trade size.

    Bins:
        [100, 1 000)     → '[100,1K)'
        [1 000, 10 000)  → '[1K,10K)'
        [10 000, 100 000)→ '[10K,100K)'
        [100 000, ∞)     → '[100K+)'

    Rows outside [100, ∞) receive NaN and are excluded by downstream functions.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a USDC-equivalent size column.
    size_col : str
        Name of the USDC-size column.

    Returns
    -------
    pd.DataFrame
        Copy of df with 'size_bin' column added.
    """
    df = df.copy()
    df["size_bin"] = pd.cut(
        df[size_col],
        bins=_BIN_EDGES,
        labels=_BIN_LABELS,
        right=False,
    ).astype(str)
    # pd.cut returns NaN for values outside the range; label them explicitly
    df.loc[df["size_bin"] == "nan", "size_bin"] = np.nan
    return df


def compute_bin_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate shortfall statistics by size bin.

    Accepts DataFrames from exp2's compute_shortfalls (which contain
    shortfall_fvo / shortfall_fvo_bps) or from standalone fixtures that use
    the suffixed forms shortfall_fvo_usdc / shortfall_fvo_bps.

    Parameters
    ----------
    df : pd.DataFrame
        Enriched route-level DataFrame with shortfall and size_bin columns.

    Returns
    -------
    pd.DataFrame
        One row per size_bin with columns: size_bin, count, total_volume_usdc,
        mean_fvo_bps, median_fvo_bps, p25_fvo_bps, p75_fvo_bps,
        mean_gfvo_bps, median_gfvo_bps.
    """
    df = df.dropna(subset=["size_bin"]).copy()

    # Resolve USDC shortfall columns: prefer bare name, fall back to _usdc suffix
    def _resolve_col(base: str, alt: str) -> str | None:
        if base in df.columns:
            return base
        if alt in df.columns:
            df[base] = df[alt]
            return base
        return None

    fvo_usdc_col = _resolve_col("shortfall_fvo", "shortfall_fvo_usdc")
    gfvo_usdc_col = _resolve_col("shortfall_gfvo", "shortfall_gfvo_usdc")

    def p25(x): return float(np.percentile(x, 25))
    def p75(x): return float(np.percentile(x, 75))

    base_agg: dict = {
        "count": ("usdc_equiv_size", "count"),
        "total_volume_usdc": ("usdc_equiv_size", "sum"),
        "mean_fvo_bps": ("shortfall_fvo_bps", "mean"),
        "median_fvo_bps": ("shortfall_fvo_bps", "median"),
        "p25_fvo_bps": ("shortfall_fvo_bps", p25),
        "p75_fvo_bps": ("shortfall_fvo_bps", p75),
        "mean_gfvo_bps": ("shortfall_gfvo_bps", "mean"),
        "median_gfvo_bps": ("shortfall_gfvo_bps", "median"),
    }
    if fvo_usdc_col:
        base_agg["mean_fvo_usdc"] = (fvo_usdc_col, "mean")
        base_agg["total_fvo_usdc"] = (fvo_usdc_col, "sum")
    if gfvo_usdc_col:
        base_agg["mean_gfvo_usdc"] = (gfvo_usdc_col, "mean")
        base_agg["total_gfvo_usdc"] = (gfvo_usdc_col, "sum")

    agg = (
        df.groupby("size_bin", observed=True)
        .agg(**base_agg)
        .reset_index()
    )

    # Preserve natural bin order
    bin_order = {label: i for i, label in enumerate(_BIN_LABELS)}
    agg["_order"] = agg["size_bin"].map(bin_order)
    agg = agg.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return agg


def compute_tail_contribution(df: pd.DataFrame, top_pct: float = 0.01) -> dict:
    """Compute the fraction of total losses attributable to the top top_pct loss events.

    Accepts both bare column names (shortfall_fvo, shortfall_gfvo) and the
    _usdc-suffixed variants (shortfall_fvo_usdc, shortfall_gfvo_usdc).

    Parameters
    ----------
    df : pd.DataFrame
        Enriched route-level DataFrame with FVO/G-FVO shortfall columns.
    top_pct : float
        Fraction of events defining the "tail" (default 0.01 = top 1%).

    Returns
    -------
    dict
        Keys: top_pct, n_tail_events, fvo_tail_share, gfvo_tail_share,
              tail_fvo_loss_usdc, total_fvo_loss_usdc,
              tail_gfvo_loss_usdc, total_gfvo_loss_usdc.
    """
    n_total = len(df)
    n_tail = max(1, int(np.ceil(n_total * top_pct)))

    def _get_col(df: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
        if primary in df.columns:
            return df[primary]
        if fallback in df.columns:
            return df[fallback]
        return pd.Series(0.0, index=df.index)

    fvo_series = _get_col(df, "shortfall_fvo", "shortfall_fvo_usdc")
    gfvo_series = _get_col(df, "shortfall_gfvo", "shortfall_gfvo_usdc")

    # FVO tail
    fvo_sorted = fvo_series.sort_values(ascending=False)
    total_fvo = float(fvo_sorted.sum())
    tail_fvo = float(fvo_sorted.iloc[:n_tail].sum())

    # G-FVO tail
    gfvo_sorted = gfvo_series.sort_values(ascending=False)
    total_gfvo = float(gfvo_sorted.sum())
    tail_gfvo = float(gfvo_sorted.iloc[:n_tail].sum())

    fvo_tail_share = tail_fvo / total_fvo if total_fvo > 0.0 else 0.0
    gfvo_tail_share = tail_gfvo / total_gfvo if total_gfvo > 0.0 else 0.0

    return {
        "top_pct": top_pct,
        "n_tail_events": n_tail,
        "tail_fvo_loss_usdc": tail_fvo,
        "total_fvo_loss_usdc": total_fvo,
        "fvo_tail_share": fvo_tail_share,
        "tail_gfvo_loss_usdc": tail_gfvo,
        "total_gfvo_loss_usdc": total_gfvo,
        "gfvo_tail_share": gfvo_tail_share,
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_synthetic: int = 50_000, save_results: bool = True) -> dict:
    """Run the full Experiment 4 pipeline.

    1. Run Exp2 to obtain benchmark shortfalls.
    2. Assign size bins.
    3. Compute per-bin and tail statistics.

    Returns
    -------
    dict
        Contains keys: bin_stats (pd.DataFrame), tail_stats (dict), enriched_df.
    """
    print(f"[Exp4] Running benchmark computation (n={n_synthetic:,}) …")
    exp2_results = _exp2_run(n_synthetic=n_synthetic, save_results=False)
    enriched_df = exp2_results["enriched_df"]

    enriched_df = assign_size_bins(enriched_df)
    bin_stats = compute_bin_statistics(enriched_df)
    tail_stats = compute_tail_contribution(enriched_df)

    print("[Exp4] Per-bin shortfall statistics:")
    print(bin_stats[["size_bin", "count", "mean_fvo_bps", "mean_gfvo_bps"]]
          .to_string(index=False))

    print(f"\n[Exp4] Tail contribution (top {tail_stats['top_pct']*100:.0f}%):")
    print(f"  FVO : {tail_stats['fvo_tail_share']*100:.1f}% of total loss "
          f"({tail_stats['tail_fvo_loss_usdc']:,.0f} USDC)")
    print(f"  GFVO: {tail_stats['gfvo_tail_share']*100:.1f}% of total loss "
          f"({tail_stats['tail_gfvo_loss_usdc']:,.0f} USDC)")

    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        bin_stats.to_csv(_RESULTS_DIR / "exp4_size_bins.csv", index=False)
        print(f"[Exp4] Results saved to {_RESULTS_DIR}")

    return {
        "bin_stats": bin_stats,
        "tail_stats": tail_stats,
        "enriched_df": enriched_df,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Exp4 complete ===")
    print(results["bin_stats"])
