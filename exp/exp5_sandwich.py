"""Experiment 5 — Sandwich Attack Detection.

Tags transactions with a within-block bracketing heuristic and compares
SCO/FVO/G-FVO shortfalls between sandwiched and non-sandwiched routes.
"""

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import POOL_IDS
except ImportError:
    POOL_IDS = [0, 1, 2, 3]

from src.synthetic_data import generate_synthetic_dataset
from exp.exp2_benchmarks import (
    compute_shortfalls,
    apply_filters,
    _get_pool_states_for_block,
)

try:
    from exp.exp1_baseline import build_route_level_dataset, compute_concentration_metrics
except ImportError:
    from exp.exp1_baseline import build_route_level_dataset, compute_concentration_metrics

_RESULTS_DIR = Path("/workspace/project/results")
_OPPOSITE = {"WETH->USDC": "USDC->WETH", "USDC->WETH": "WETH->USDC"}


# ---------------------------------------------------------------------------
# Sandwich detection helpers
# ---------------------------------------------------------------------------

def compute_counterparty_key(row: pd.Series) -> str:
    """Return a cluster key for grouping potential MEV-bot addresses.

    Uses the sender address as the cluster identifier. In practice, more
    sophisticated address-clustering (e.g., funding graph analysis) would be
    applied; this simplified version is suitable for synthetic data.

    Parameters
    ----------
    row : pd.Series
        A swap event row with a 'sender' field.

    Returns
    -------
    str
        Cluster key string.
    """
    return str(row.get("sender", "unknown"))


def tag_sandwiched_transactions(swap_events: pd.DataFrame) -> pd.DataFrame:
    """Apply within-block bracketing heuristic to detect sandwich victims.

    For each block, events are sorted by (tx_index, log_index). A transaction V
    is flagged as sandwiched if there exists a cluster C and pool P such that:
      1. An earlier transaction F in the same block from cluster C makes a swap
         on pool P in the direction OPPOSITE to V's direction on P.
      2. A later transaction B in the same block from cluster C makes a swap on
         pool P in the SAME direction as F (opposite to V).

    Accepts either ``pool_id`` or ``pool_address`` as the pool identifier column.

    Parameters
    ----------
    swap_events : pd.DataFrame
        Pool-level swap events with columns: tx_hash, block_number, tx_index,
        log_index, pool_id OR pool_address, direction, sender.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with three new columns:
        cluster_key (str), sandwich_flag (bool), sandwich_reason (str).
    """
    df = swap_events.copy()

    # Resolve the pool identifier column
    pool_col = "pool_id" if "pool_id" in df.columns else "pool_address"

    df["cluster_key"] = df.apply(compute_counterparty_key, axis=1)
    df["sandwich_flag"] = False
    df["sandwich_reason"] = ""

    for block_num, block_df in df.groupby("block_number", sort=False):
        block_df = block_df.sort_values(["tx_index", "log_index"])
        events = block_df.to_dict("records")
        n = len(events)

        for vi, victim in enumerate(events):
            victim_pool = victim[pool_col]
            victim_dir = victim["direction"]
            opposite_dir = _OPPOSITE.get(victim_dir, "")

            # Search for front-run: earlier event, same pool, opposite direction
            front_run = None
            for fi in range(vi):
                cand = events[fi]
                if (
                    cand[pool_col] == victim_pool
                    and cand["direction"] == opposite_dir
                    and cand["tx_hash"] != victim["tx_hash"]
                ):
                    front_run = cand
                    break

            if front_run is None:
                continue

            # Search for back-run: later event, same pool, same attacker cluster.
            # The back-run moves in the OPPOSITE direction to the front-run
            # (i.e. the attacker reverses their position, matching the victim direction).
            fr_cluster = front_run["cluster_key"]
            back_run_dir = _OPPOSITE.get(front_run["direction"], "")
            for bi in range(vi + 1, n):
                cand = events[bi]
                if (
                    cand[pool_col] == victim_pool
                    and cand["direction"] == back_run_dir
                    and cand["cluster_key"] == fr_cluster
                    and cand["tx_hash"] != victim["tx_hash"]
                ):
                    # Victim is sandwiched
                    idx = df.index[
                        (df["block_number"] == block_num)
                        & (df["tx_hash"] == victim["tx_hash"])
                        & (df[pool_col] == victim_pool)
                    ]
                    df.loc[idx, "sandwich_flag"] = True
                    df.loc[idx, "sandwich_reason"] = (
                        f"front={front_run['tx_hash'][:10]}, "
                        f"back={cand['tx_hash'][:10]}, "
                        f"pool={victim_pool}"
                    )
                    break

    return df.reset_index(drop=True)


def aggregate_sandwich_flags(swap_events: pd.DataFrame) -> pd.DataFrame:
    """Roll per-leg sandwich flags up to transaction level.

    A transaction is considered sandwiched if any of its pool-level legs were
    flagged by the bracketing heuristic.

    Parameters
    ----------
    swap_events : pd.DataFrame
        Pool-level swap events enriched by tag_sandwiched_transactions.

    Returns
    -------
    pd.DataFrame
        One row per tx_hash with columns: tx_hash, sandwich_flag.
    """
    return (
        swap_events.groupby("tx_hash", sort=False)["sandwich_flag"]
        .any()
        .reset_index()
        .rename(columns={"sandwich_flag": "sandwich_flag"})
    )


# ---------------------------------------------------------------------------
# Shortfall comparison
# ---------------------------------------------------------------------------

def compare_sandwiched_shortfalls(routes_df: pd.DataFrame) -> dict:
    """Compare SCO/FVO/G-FVO shortfalls between sandwiched and non-sandwiched routes.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route-level DataFrame with shortfall columns and sandwich_flag column.

    Returns
    -------
    dict
        Keys: 'full', 'non_sandwiched', 'sandwiched'. Each value is a dict
        with mean, median, p25, p75 for shortfall_sco_bps, shortfall_fvo_bps,
        shortfall_gfvo_bps.
    """
    def _stats(subset: pd.DataFrame, label: str) -> dict:
        result = {"label": label, "n": len(subset)}
        for col in ["shortfall_sco_bps", "shortfall_fvo_bps", "shortfall_gfvo_bps"]:
            if col not in subset.columns:
                continue
            s = subset[col].dropna()
            result[f"mean_{col}"] = float(s.mean()) if len(s) > 0 else np.nan
            result[f"median_{col}"] = float(s.median()) if len(s) > 0 else np.nan
            result[f"p25_{col}"] = float(np.percentile(s, 25)) if len(s) > 0 else np.nan
            result[f"p75_{col}"] = float(np.percentile(s, 75)) if len(s) > 0 else np.nan
        return result

    if "sandwich_flag" in routes_df.columns:
        flag = routes_df["sandwich_flag"].astype(bool)
        sandwiched = routes_df[flag]
        non_sandwiched = routes_df[~flag]
    else:
        sandwiched = routes_df.iloc[0:0]
        non_sandwiched = routes_df

    return {
        "full": _stats(routes_df, "full"),
        "non_sandwiched": _stats(non_sandwiched, "non_sandwiched"),
        "sandwiched": _stats(sandwiched, "sandwiched"),
    }


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_synthetic: int = 50_000, save_results: bool = True) -> dict:
    """Run the full Experiment 5 pipeline.

    1. Generate synthetic data.
    2. Tag sandwiched transactions via bracketing heuristic.
    3. Run benchmark shortfall computation (Exp2).
    4. Merge sandwich flags into route-level data.
    5. Compare shortfalls between sandwiched and non-sandwiched routes.

    Returns
    -------
    dict
        Summary dict with full/non_sandwiched/sandwiched shortfall statistics.
    """
    print(f"[Exp5] Generating {n_synthetic:,} synthetic transactions …")
    _ds = generate_synthetic_dataset(n_transactions=n_synthetic)
    transactions_df = _ds["transactions"]
    swap_events_df = _ds["swap_events"]
    pool_states_by_block = _ds["pool_states_by_block"]

    # Tag sandwich victims at pool-event level
    print("[Exp5] Tagging sandwiched transactions …")
    tagged_events = tag_sandwiched_transactions(swap_events_df)
    tx_flags = aggregate_sandwich_flags(tagged_events)

    n_sandwich = int(tx_flags["sandwich_flag"].sum())
    print(f"[Exp5] Sandwiched transactions detected: {n_sandwich:,} "
          f"({n_sandwich / len(tx_flags) * 100:.2f}%)")

    # Build route-level dataset and compute benchmarks
    routes_df = build_route_level_dataset(swap_events_df)
    routes_df = apply_filters(routes_df)
    routes_df = compute_concentration_metrics(routes_df)

    routes_df = compute_shortfalls(routes_df, pool_states_by_block)

    # Merge sandwich flags
    routes_df = routes_df.merge(tx_flags, on="tx_hash", how="left")
    routes_df["sandwich_flag"] = routes_df["sandwich_flag"].fillna(False)

    # Compare shortfalls
    comparison = compare_sandwiched_shortfalls(routes_df)

    print("[Exp5] Shortfall comparison (mean FVO bps):")
    for group, stats in comparison.items():
        val = stats.get("mean_shortfall_fvo_bps", np.nan)
        n = stats.get("n", 0)
        print(f"  {group:20s}: n={n:6,}  mean_fvo_bps={val:.4f}")

    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_rows = []
        for group, stats in comparison.items():
            row = {"group": group}
            row.update(stats)
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(_RESULTS_DIR / "exp5_sandwich.csv", index=False)
        print(f"[Exp5] Results saved to {_RESULTS_DIR}")

    return {
        "summary": comparison,
        "n_sandwiched": n_sandwich,
        "routes_df": routes_df,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Exp5 complete ===")
    for group, stats in results["summary"].items():
        print(f"  {group}: {stats}")
