"""Experiment 3 — State-Staleness Analysis.

Measures how much FVO output degrades when routing algorithms use pool state
from N blocks before the transaction's execution block (stale state).
"""

import bisect
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from src.config import POOL_IDS
except ImportError:
    POOL_IDS = [0, 1, 2, 3]

from src.synthetic_data import generate_synthetic_dataset
from exp.exp2_benchmarks import (
    compute_fvo,
    _get_pool_states_for_block,
    apply_filters,
)

import bisect as _bisect


def _nearest_block_state(pool_states_by_block: dict, target_block: int):
    """Return pool-state dict for the nearest stored block <= target_block."""
    sorted_blks = sorted(pool_states_by_block.keys())
    idx = _bisect.bisect_right(sorted_blks, target_block) - 1
    if idx < 0:
        idx = 0
    return pool_states_by_block[sorted_blks[idx]]

from exp.exp1_baseline import build_route_level_dataset, compute_concentration_metrics

_RESULTS_DIR = Path("/workspace/project/results")
_Q_COLS = [f"q_pool{i}" for i in POOL_IDS]
_WETH_PRICE_USDC = 3_000.0
_DEFAULT_N_VALUES = list(range(11))  # N = 0, 1, …, 10


# ---------------------------------------------------------------------------
# Block-state helpers
# ---------------------------------------------------------------------------

def get_block_end_state(pool_states_by_block: dict, block: int, pool_id: int):
    """Return pool state for pool_id at the end of `block`, or None if block missing.

    Parameters
    ----------
    pool_states_by_block : dict
        {block_num: list_or_dict_of_pool_states}
    block : int
        Target block number.
    pool_id : int
        Pool index (0-3).

    Returns
    -------
    dict or None
        Pool state dict, or None if block not in pool_states_by_block.
    """
    if block not in pool_states_by_block:
        return None
    return pool_states_by_block[block][pool_id]


def get_stale_state(
    pool_states_by_block: dict,
    block: int,
    N: int,
    exec_time_states=None,
):
    """Return the pool states N blocks before `block`.

    N=0 returns exec_time_states (the execution-time state list/dict).
    N>0 returns pool_states_by_block[block - N], or None if unavailable.

    Parameters
    ----------
    pool_states_by_block : dict
        {block_num: list_or_dict_of_pool_states}
    block : int
        Execution block number.
    N : int
        Staleness level (0 = execution-time state).
    exec_time_states : list or dict or None
        Pre-transaction execution-time pool states for this transaction.

    Returns
    -------
    list or dict or None
        Pool states for the stale block, or None if unavailable.
    """
    if N == 0:
        return exec_time_states
    stale_block = block - N
    if stale_block not in pool_states_by_block:
        return None
    return pool_states_by_block[stale_block]


# ---------------------------------------------------------------------------
# FVO at a given state snapshot
# ---------------------------------------------------------------------------

def compute_fvo_at_state(Q: float, pool_states, direction: str) -> float:
    """Return the FVO output for Q traded across the 4-pool universe.

    Parameters
    ----------
    Q : float
        Total input quantity. Must be positive.
    pool_states : list or dict of pool state dicts (length 4).
    direction : str
        'WETH->USDC' or 'USDC->WETH'.

    Returns
    -------
    float
        FVO output value.

    Raises
    ------
    ValueError
        If Q <= 0.
    """
    if Q <= 0.0:
        raise ValueError(f"Q must be positive, got {Q}")
    # Normalise to list for compute_fvo
    if isinstance(pool_states, dict):
        states_list = [pool_states[pid] for pid in POOL_IDS]
    else:
        states_list = list(pool_states)
    _, fvo_output = compute_fvo(Q, states_list, direction)
    return fvo_output


# ---------------------------------------------------------------------------
# Staleness analysis
# ---------------------------------------------------------------------------

def analyze_staleness(
    routes_df,
    pool_states_by_block: dict,
    exec_time_states: dict = None,
    n_values: list = None,
) -> dict:
    """Compute mean FVO output for each staleness level N.

    Parameters
    ----------
    routes_df : pd.DataFrame or list of dict
        Route-level data with block_number, Q_total, direction / direction_z.
    pool_states_by_block : dict
        {block_num: list_or_dict_of_pool_states}.
    exec_time_states : dict or None
        {block_num: pool_states} for execution-time states (N=0).
        When None, uses pool_states_by_block as approximation.
    n_values : list of int or None
        Staleness levels. Defaults to [0, 1, ..., 10].

    Returns
    -------
    dict
        {N: {"mean_fvo_output": float, "n_obs": int}} for each N.
    """
    if n_values is None:
        n_values = _DEFAULT_N_VALUES
    if exec_time_states is None:
        exec_time_states = pool_states_by_block

    fvo_by_n: dict = {N: [] for N in n_values}

    def _get(row, *keys, default=None):
        for k in keys:
            if isinstance(row, dict):
                if k in row:
                    return row[k]
            elif hasattr(row, k):
                return getattr(row, k)
        return default

    rows_iter = (
        routes_df
        if not isinstance(routes_df, pd.DataFrame)
        else [row._asdict() for row in routes_df.itertuples()]
    )

    for row in rows_iter:
        block = int(_get(row, "block_number", default=0))
        Q = float(_get(row, "Q_total", default=0.0))
        direction = str(_get(row, "direction_z", "direction", default="WETH->USDC"))
        if Q <= 0.0:
            continue

        # Execution-time state: nearest stored block <= block
        exec_st = _nearest_block_state(pool_states_by_block, block)

        for N in n_values:
            if N == 0:
                stale_st = exec_st
            else:
                stale_st = _nearest_block_state(pool_states_by_block, block - N)
            if stale_st is None:
                continue
            try:
                fvo_out = compute_fvo_at_state(Q, stale_st, direction)
                fvo_by_n[N].append(fvo_out)
            except Exception:
                pass

    result = {}
    for N in n_values:
        vals = fvo_by_n[N]
        result[N] = {
            "mean_fvo_output": float(np.mean(vals)) if vals else float("nan"),
            "n_obs": len(vals),
        }
    return result


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_synthetic: int = 50_000, save_results: bool = True) -> dict:
    """Run the full Experiment 3 pipeline.

    1. Generate synthetic data with block-level pool states.
    2. Build route-level dataset.
    3. Analyse mean FVO output for N = 0 … 10 blocks of staleness.
    4. Summarise N=0 vs N=1 difference.

    Returns
    -------
    dict
        Flattened summary with n_transactions, mean FVO values, and diff.
    """
    print(f"[Exp3] Generating {n_synthetic:,} synthetic transactions …")
    _ds = generate_synthetic_dataset(n_transactions=n_synthetic)
    transactions_df = _ds["transactions"]
    swap_events_df = _ds["swap_events"]
    pool_states_by_block = _ds["pool_states_by_block"]

    routes_df = build_route_level_dataset(swap_events_df)
    routes_df = apply_filters(routes_df)
    routes_df = compute_concentration_metrics(routes_df)
    print(f"[Exp3] Analysing staleness for {len(routes_df):,} routes …")

    staleness_result = analyze_staleness(routes_df, pool_states_by_block)

    # Build a per-N DataFrame from the dict result
    per_n_rows = [
        {"N": N, "mean_fvo_output": staleness_result[N]["mean_fvo_output"],
         "n_obs": staleness_result[N]["n_obs"]}
        for N in sorted(staleness_result.keys())
    ]
    per_n = pd.DataFrame(per_n_rows)

    fvo_n0 = staleness_result[0]["mean_fvo_output"]
    fvo_n1 = staleness_result[1]["mean_fvo_output"]
    diff_n0_n1 = (
        (fvo_n0 - fvo_n1) if (not np.isnan(fvo_n0) and not np.isnan(fvo_n1)) else float("nan")
    )

    summary = {
        "n_transactions": len(routes_df),
        "mean_fvo_at_N0": fvo_n0,
        "mean_fvo_at_N1": fvo_n1,
        "mean_fvo_diff_N0_minus_N1": diff_n0_n1,
    }

    print("[Exp3] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        per_n.to_csv(_RESULTS_DIR / "exp3_staleness_per_n.csv", index=False)
        print(f"[Exp3] Results saved to {_RESULTS_DIR}")

    return {
        "n_transactions": summary["n_transactions"],
        "mean_fvo_at_N0": summary["mean_fvo_at_N0"],
        "mean_fvo_at_N1": summary["mean_fvo_at_N1"],
        "mean_fvo_diff_N0_minus_N1": summary["mean_fvo_diff_N0_minus_N1"],
        "summary": summary,
        "per_n": per_n,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Exp3 complete ===")
    print(results["summary"])
