"""Experiment 1 — Baseline Sample Construction.

Reconstructs the WETH-USDC same-pair routing sample and computes per-transaction
concentration metrics (activation rate, entropy, HHI).
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import POOL_IDS, ROUTER_ADDRESSES
except ImportError:
    POOL_IDS = [0, 1, 2, 3]
    ROUTER_ADDRESSES = {
        "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "UniswapV3Router",
        "0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B": "UniswapUniversalRouter",
        "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F": "SushiswapRouter",
    }

from src.synthetic_data import generate_synthetic_dataset

try:
    from src.concentration import compute_activation_rate, compute_entropy, compute_hhi
except ImportError:
    def compute_activation_rate(x: np.ndarray, num_pools: int = 4) -> float:
        """Fraction of num_pools with non-zero flow share."""
        return float(np.sum(np.asarray(x) > 0)) / num_pools

    def compute_entropy(x: np.ndarray) -> float:
        """Raw Shannon entropy of flow shares (nats)."""
        x = np.asarray(x)
        x = x[x > 0]
        return float(-np.sum(x * np.log(x)))

    def compute_hhi(x: np.ndarray) -> float:
        """Herfindahl-Hirschman Index (sum of squared shares)."""
        return float(np.sum(np.asarray(x) ** 2))


try:
    from src.router_labels import get_router_label as label_router
except ImportError:
    def label_router(address: str) -> str:
        return ROUTER_ADDRESSES.get(str(address), "unlabeled")


try:
    from src.filters import filter_min_usdc as apply_filters
except ImportError:
    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        """Retain transactions with USDC-equivalent size >= 100."""
        return df[df["usdc_equiv_size"] >= 100.0].copy()


_RESULTS_DIR = Path("/workspace/project/results")
_N_POOLS = len(POOL_IDS)
_Q_COLS = [f"q_pool{i}" for i in POOL_IDS]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_route_level_dataset(swap_events: pd.DataFrame) -> pd.DataFrame:
    """Group pool-level swap events by transaction to produce route-level rows.

    For each transaction aggregates q_j per pool, determines the dominant
    direction, detects counterflow legs, computes Q_total and flow shares x_j.

    Parameters
    ----------
    swap_events : pd.DataFrame
        Pool-level swap events with columns tx_hash, block_number, tx_index,
        pool_id, direction, amount_in, amount_out, recipient (to_address).

    Returns
    -------
    pd.DataFrame
        Route-level DataFrame with columns: tx_hash, block_number, tx_index,
        to_address, direction, q_pool0..q_pool3, Q_total, usdc_equiv_size,
        x_pool0..x_pool3, has_counterflow, router_label.
    """
    groups = swap_events.groupby("tx_hash", sort=False)

    records = []
    for tx_hash, grp in groups:
        # Dominant direction (most common across legs)
        direction = grp["direction"].mode().iloc[0]

        # Counterflow: any leg with opposite direction?
        has_counterflow = bool((grp["direction"] != direction).any())

        # Sum amount_in per pool (handles multiple logs per pool)
        q_per_pool = grp.groupby("pool_id")["amount_in"].sum()
        q_vals = np.array([q_per_pool.get(pid, 0.0) for pid in POOL_IDS])

        Q_total = q_vals.sum()
        if Q_total <= 0.0:
            continue

        x_vals = q_vals / Q_total

        # USDC-equivalent size
        if direction == "WETH->USDC":
            usdc_size = Q_total * 3_000.0  # approximate conversion
        else:
            usdc_size = Q_total

        row_meta = grp.iloc[0]
        to_addr = str(row_meta.get("recipient", row_meta.get("to_address", "")))

        record = {
            "tx_hash": tx_hash,
            "block_number": int(row_meta["block_number"]),
            "tx_index": int(row_meta["tx_index"]),
            "to_address": to_addr,
            "direction": direction,
            "Q_total": Q_total,
            "usdc_equiv_size": usdc_size,
            "has_counterflow": has_counterflow,
        }
        for i, pid in enumerate(POOL_IDS):
            record[f"q_pool{pid}"] = q_vals[i]
            record[f"x_pool{pid}"] = x_vals[i]

        record["router_label"] = label_router(to_addr)
        records.append(record)

    return pd.DataFrame(records).reset_index(drop=True)


def compute_concentration_metrics(routes_df: pd.DataFrame) -> pd.DataFrame:
    """Add activation_rate, entropy, hhi columns to routes_df.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route-level DataFrame with x_pool0..x_pool3 share columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns appended in-place.
    """
    x_share_cols = [f"x_pool{pid}" for pid in POOL_IDS]
    df = routes_df.copy()

    X = df[x_share_cols].values  # shape (n, 4)

    df["activation_rate"] = [compute_activation_rate(X[i]) for i in range(len(df))]
    df["entropy"] = [compute_entropy(X[i]) for i in range(len(df))]
    df["hhi"] = [compute_hhi(X[i]) for i in range(len(df))]

    return df


def compute_summary_statistics(routes_df: pd.DataFrame) -> dict:
    """Compute dataset-level summary statistics.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route-level DataFrame with concentration metric columns.

    Returns
    -------
    dict
        Keys: n_transactions, total_usdc_volume, avg_trade_size,
              mean_activation_rate, mean_entropy, mean_hhi.
    """
    return {
        "n_transactions": len(routes_df),
        "total_usdc_volume": float(routes_df["usdc_equiv_size"].sum()),
        "avg_trade_size": float(routes_df["usdc_equiv_size"].mean()),
        "mean_activation_rate": float(routes_df["activation_rate"].mean()),
        "mean_entropy": float(routes_df["entropy"].mean()),
        "mean_hhi": float(routes_df["hhi"].mean()),
    }


def compute_router_statistics(routes_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concentration metrics by router label.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route-level DataFrame with router_label and concentration columns.

    Returns
    -------
    pd.DataFrame
        One row per router label with columns: count, total_volume,
        avg_size, mean_activation, mean_entropy, mean_hhi.
    """
    agg = (
        routes_df.groupby("router_label")
        .agg(
            count=("tx_hash", "count"),
            total_volume=("usdc_equiv_size", "sum"),
            avg_size=("usdc_equiv_size", "mean"),
            mean_activation=("activation_rate", "mean"),
            mean_entropy=("entropy", "mean"),
            mean_hhi=("hhi", "mean"),
        )
        .reset_index()
        .sort_values("total_volume", ascending=False)
    )
    return agg


def run_experiment(n_synthetic: int = 50_000, save_results: bool = True) -> dict:
    """Run the full Experiment 1 pipeline.

    1. Generate (or load) synthetic WETH-USDC swap data.
    2. Apply size filters.
    3. Build route-level dataset from pool swap events.
    4. Compute concentration metrics.
    5. Aggregate summary and router-level statistics.
    6. Optionally save CSVs to results/.

    Parameters
    ----------
    n_synthetic : int
        Number of synthetic transactions to generate.
    save_results : bool
        Write output CSVs to /workspace/project/results/.

    Returns
    -------
    dict
        Contains keys: summary, router_stats, routes_df.
    """
    print(f"[Exp1] Generating {n_synthetic:,} synthetic transactions …")
    _ds = generate_synthetic_dataset(n_transactions=n_synthetic)
    transactions_df = _ds["transactions"]
    swap_events_df = _ds["swap_events"]

    # Filter out micro-trades
    transactions_df = apply_filters(transactions_df)
    print(f"[Exp1] After filters: {len(transactions_df):,} transactions")

    # Use transactions_df directly (already route-level from generator)
    # Build route-level from swap events (demonstrates the pipeline)
    routes_df = build_route_level_dataset(swap_events_df)
    routes_df = apply_filters(routes_df)

    # Concentration metrics
    routes_df = compute_concentration_metrics(routes_df)

    # Statistics
    summary = compute_summary_statistics(routes_df)
    router_stats = compute_router_statistics(routes_df)

    print("[Exp1] Summary statistics:")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df = pd.DataFrame([summary])
        summary_df.to_csv(_RESULTS_DIR / "exp1_summary.csv", index=False)
        router_stats.to_csv(_RESULTS_DIR / "exp1_router_stats.csv", index=False)
        print(f"[Exp1] Results saved to {_RESULTS_DIR}")

    return {
        "n_transactions": summary["n_transactions"],
        "total_usdc_volume": summary["total_usdc_volume"],
        "avg_trade_size": summary["avg_trade_size"],
        "mean_activation_rate": summary["mean_activation_rate"],
        "mean_entropy": summary["mean_entropy"],
        "mean_hhi": summary["mean_hhi"],
        "summary": summary,
        "router_stats": router_stats,
        "routes_df": routes_df,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Exp1 complete ===")
    print(results["summary"])
