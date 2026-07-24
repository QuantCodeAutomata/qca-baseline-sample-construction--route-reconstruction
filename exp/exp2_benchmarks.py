"""Experiment 2 — Execution-Time Optimal Routing Benchmarks.

Computes SCO (Set-Constrained Optimal), FVO (Full-Venue Optimal), and
G-FVO (Gas-adjusted Full-Venue Optimal) benchmarks for each observed
transaction, then measures shortfall relative to the realized route.
"""

from itertools import combinations
from pathlib import Path
from typing import Optional

import bisect
import numpy as np
import pandas as pd
from scipy.optimize import brentq

try:
    from src.config import POOL_IDS, POOL_PROTOCOLS, GAS_COSTS_USDC
except ImportError:
    POOL_IDS = [0, 1, 2, 3]
    POOL_PROTOCOLS = {0: "v2", 1: "v3", 2: "v3", 3: "v3"}
    GAS_COSTS_USDC = [5.0, 3.0, 3.0, 3.0]

from src.synthetic_data import generate_synthetic_dataset

try:
    from src.amm_math import v2_total_output
except ImportError:
    def v2_total_output(q: float, reserve_in: float, reserve_out: float, fee_bps: int = 30) -> float:
        """Uniswap V2 constant-product output."""
        if q <= 0.0:
            return 0.0
        k = 10_000 - fee_bps
        return q * k * reserve_out / (reserve_in * 10_000 + q * k)

# V3 output uses a local implementation with a direction-aware interface,
# avoiding the incompatible token_in_is_token0 boolean of src.amm_math.
_Q96 = float(2 ** 96)


def _v3_output_local(
    q: float, sqrt_price_x96: float, liquidity: float, fee_bps: int, direction: str
) -> float:
    """Single-range V3 output; fee deducted on effective input."""
    if q <= 0.0:
        return 0.0
    P = sqrt_price_x96 / _Q96
    f = 1.0 - fee_bps / 10_000.0
    q_eff = q * f
    if direction == "WETH->USDC":
        return liquidity * q_eff * P * P / (liquidity + q_eff * P)
    return liquidity * q_eff / (liquidity * P * P + q_eff * P)


try:
    from src.concentration import compute_activation_rate, compute_entropy, compute_hhi
except ImportError:
    def compute_activation_rate(x, num_pools=4): return float(np.sum(x > 0)) / num_pools
    def compute_entropy(x):
        x = np.asarray(x)[np.asarray(x) > 0]; return float(-np.sum(x * np.log(x)))
    def compute_hhi(x): return float(np.sum(np.asarray(x) ** 2))

try:
    from src.router_labels import get_router_label as label_router
except ImportError:
    def label_router(addr: str) -> str:
        _ROUTERS = {
            "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "UniswapV3Router",
            "0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B": "UniswapUniversalRouter",
            "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F": "SushiswapRouter",
        }
        return _ROUTERS.get(str(addr), "unlabeled")

try:
    from src.filters import filter_min_usdc as apply_filters
except ImportError:
    def apply_filters(df): return df[df["usdc_equiv_size"] >= 100.0].copy()

_RESULTS_DIR = Path("/workspace/project/results")
_N_POOLS = len(POOL_IDS)
_Q_COLS = [f"q_pool{i}" for i in POOL_IDS]
_WETH_PRICE_USDC = 3_000.0


# ---------------------------------------------------------------------------
# AMM wrappers
# ---------------------------------------------------------------------------

def compute_v2_output(q: float, pool_state: dict, direction: str) -> float:
    """V2 output for quantity q in direction WETH->USDC or USDC->WETH."""
    fee_bps = pool_state["fee_bps"]
    if direction == "WETH->USDC":
        return v2_total_output(q, pool_state["reserve_weth"], pool_state["reserve_usdc"], fee_bps)
    else:
        return v2_total_output(q, pool_state["reserve_usdc"], pool_state["reserve_weth"], fee_bps)


def compute_v3_output(q: float, pool_state: dict, direction: str) -> float:
    """V3 output for quantity q using single-range approximation."""
    return _v3_output_local(
        q,
        pool_state["sqrt_price_x96"],
        pool_state["liquidity"],
        pool_state["fee_bps"],
        direction,
    )


def compute_pool_output(q: float, pool_id: int, pool_state: dict, direction: str) -> float:
    """Route to V2 or V3 output based on pool protocol."""
    if q <= 0.0:
        return 0.0
    protocol = POOL_PROTOCOLS.get(pool_id, pool_state.get("protocol", "v3"))
    if protocol == "v2":
        return compute_v2_output(q, pool_state, direction)
    return compute_v3_output(q, pool_state, direction)


def total_output_for_allocation(
    allocation: np.ndarray, pool_states: list, direction: str
) -> float:
    """Sum AMM outputs across 4 pools for a given allocation vector."""
    total = 0.0
    for i, pid in enumerate(POOL_IDS):
        total += compute_pool_output(float(allocation[i]), pid, pool_states[i], direction)
    return total


# ---------------------------------------------------------------------------
# Spot price and inverse marginal rate helpers
# ---------------------------------------------------------------------------

def _spot_price(pool_id: int, pool_state: dict, direction: str) -> float:
    """Marginal output rate at q=0 (spot price, in output/input units)."""
    f = 1.0 - pool_state["fee_bps"] / 10_000.0
    protocol = POOL_PROTOCOLS.get(pool_id, pool_state.get("protocol", "v3"))
    if protocol == "v2":
        if direction == "WETH->USDC":
            return f * pool_state["reserve_usdc"] / pool_state["reserve_weth"]
        return f * pool_state["reserve_weth"] / pool_state["reserve_usdc"]
    else:
        _Q96 = 2 ** 96
        P = pool_state["sqrt_price_x96"] / _Q96
        if direction == "WETH->USDC":
            return f * P * P
        return f / (P * P)


def _inv_marginal_rate(pool_id: int, pool_state: dict, direction: str, lam: float) -> float:
    """Input quantity q such that marginal_rate(q) == lam. Returns 0 if pool is inactive."""
    _Q96 = 2 ** 96
    f = 1.0 - pool_state["fee_bps"] / 10_000.0
    spot = _spot_price(pool_id, pool_state, direction)
    if lam >= spot:
        return 0.0

    protocol = POOL_PROTOCOLS.get(pool_id, pool_state.get("protocol", "v3"))

    if protocol == "v2":
        if direction == "WETH->USDC":
            R_in = pool_state["reserve_weth"]
            R_out = pool_state["reserve_usdc"]
        else:
            R_in = pool_state["reserve_usdc"]
            R_out = pool_state["reserve_weth"]
        q = (np.sqrt(f * R_out * R_in / lam) - R_in) / f
    else:
        P = pool_state["sqrt_price_x96"] / _Q96
        L = pool_state["liquidity"]
        if direction == "WETH->USDC":
            # q = L*(P*sqrt(f/lam) - 1) / (f*P)
            q = L * (P * np.sqrt(f / lam) - 1.0) / (f * P)
        else:
            # q = L*(sqrt(f/lam) - P) / f
            q = L * (np.sqrt(f / lam) - P) / f

    return max(0.0, float(q))


# ---------------------------------------------------------------------------
# Optimal allocation via marginal price equalization
# ---------------------------------------------------------------------------

def gas_free_optimal_allocation(
    Q: float,
    pool_states: list,
    direction: str,
    active_set: Optional[set] = None,
    tol: float = 1e-10,
) -> np.ndarray:
    """Gas-free optimal allocation via marginal price equalization (bisection).

    Finds the allocation vector q* that maximizes total AMM output subject to
    sum(q*) = Q, with each q_j >= 0, over the active pool set.

    Parameters
    ----------
    Q : float
        Total input quantity (in WETH for WETH->USDC, in USDC for USDC->WETH).
    pool_states : list of dict
        Pool state dicts ordered by POOL_IDS (length 4).
    direction : str
        'WETH->USDC' or 'USDC->WETH'.
    active_set : set of int or None
        Pool indices to include. None uses all 4 pools.
    tol : float
        Bisection tolerance for brentq.

    Returns
    -------
    np.ndarray
        Allocation array of shape (4,).
    """
    if active_set is None:
        active_set = set(POOL_IDS)

    active_pools = [i for i in POOL_IDS if i in active_set]

    if Q <= 0.0 or not active_pools:
        return np.zeros(_N_POOLS)

    spots = [_spot_price(pid, pool_states[pid], direction) for pid in active_pools]
    max_spot = max(spots)

    if max_spot <= 0.0:
        return np.zeros(_N_POOLS)

    def total_q(lam: float) -> float:
        return sum(_inv_marginal_rate(pid, pool_states[pid], direction, lam) for pid in active_pools)

    def excess(lam: float) -> float:
        return total_q(lam) - Q

    # At lam → max_spot (from below): total_q → 0, excess → -Q < 0
    lam_high = max_spot * (1.0 - 1e-12)
    # At lam → 0: total_q → ∞, excess → +∞; find a bracket point
    lam_low = max_spot * 1e-9

    # Ensure lower bracket gives positive excess
    if excess(lam_low) <= 0.0:
        # Q exceeds total pool capacity; all flow to highest-spot pool
        alloc = np.zeros(_N_POOLS)
        best_pid = active_pools[int(np.argmax(spots))]
        alloc[best_pid] = Q
        return alloc

    best_pid = active_pools[int(np.argmax(spots))]

    # For very high-liquidity pools q_j(lambda) stays enormous until lambda is
    # within a few ULP of spot_j. Narrow lam_high toward max_spot iteratively.
    if excess(lam_high) >= 0.0:
        gap = max_spot - lam_high
        found = False
        for _ in range(64):
            gap *= 0.5
            if gap < max_spot * 2.3e-16:  # below float64 machine epsilon
                break
            lam_high += gap
            if excess(lam_high) < 0.0:
                found = True
                break
        if not found:
            # Float64 precision exhausted: optimal λ* ≈ max_spot ⟹ all Q in
            # the highest-spot pool (negligible price impact for any Q).
            alloc = np.zeros(_N_POOLS)
            alloc[best_pid] = Q
            return alloc

    lam_star = brentq(excess, lam_low, lam_high, xtol=tol, maxiter=200)

    alloc = np.zeros(_N_POOLS)
    for pid in active_pools:
        alloc[pid] = _inv_marginal_rate(pid, pool_states[pid], direction, lam_star)

    # Rescale to enforce sum = Q exactly (numerical precision).
    # If total ≈ 0 (lam_star collapsed to max_spot in float64), fall back to best pool.
    total = alloc.sum()
    if total <= 0.0:
        alloc = np.zeros(_N_POOLS)
        alloc[best_pid] = Q
    else:
        alloc *= Q / total

    return alloc


# ---------------------------------------------------------------------------
# Benchmark computations
# ---------------------------------------------------------------------------

def compute_sco(
    Q: float,
    realized_alloc: np.ndarray,
    pool_states: list,
    direction: str,
) -> tuple:
    """SCO: gas-free optimal restricted to the realized activated pool set.

    Parameters
    ----------
    Q : float
        Total input quantity.
    realized_alloc : np.ndarray
        Observed allocation vector (length 4).
    pool_states : list
        Pool state dicts (length 4).
    direction : str

    Returns
    -------
    tuple[np.ndarray, float]
        (sco_allocation, sco_output)
    """
    active_set = {i for i in POOL_IDS if realized_alloc[i] > 0.0}
    if not active_set:
        active_set = set(POOL_IDS)
    sco_alloc = gas_free_optimal_allocation(Q, pool_states, direction, active_set)
    sco_output = total_output_for_allocation(sco_alloc, pool_states, direction)
    return sco_alloc, sco_output


def compute_fvo(
    Q: float,
    pool_states: list,
    direction: str,
) -> tuple:
    """FVO: gas-free optimal over the full 4-pool universe.

    Returns
    -------
    tuple[np.ndarray, float]
        (fvo_allocation, fvo_output)
    """
    fvo_alloc = gas_free_optimal_allocation(Q, pool_states, direction)
    fvo_output = total_output_for_allocation(fvo_alloc, pool_states, direction)
    return fvo_alloc, fvo_output


def compute_gfvo(
    Q: float,
    pool_states: list,
    direction: str,
    gas_costs_usdc: list,
) -> tuple:
    """G-FVO: enumerate all 15 non-empty subsets, pick best net output.

    For each subset S:
        Q_eff = Q - gas_cost(S) expressed in input units (approx via price).
        Skip if Q_eff <= 0.
        Compute gas-free optimal over S for full Q.
        Net output = total_output - gas_cost_usdc(S).

    Returns
    -------
    tuple[np.ndarray, float]
        (gfvo_allocation, gfvo_net_output)  — net output is in USDC.
    """
    # Gas costs in input-token units
    if direction == "WETH->USDC":
        gas_in_input = [c / _WETH_PRICE_USDC for c in gas_costs_usdc]
    else:
        gas_in_input = list(gas_costs_usdc)  # already in USDC

    best_net = -np.inf
    best_alloc = np.zeros(_N_POOLS)

    for r in range(1, _N_POOLS + 1):
        for subset in combinations(POOL_IDS, r):
            S = set(subset)
            gas_input_S = sum(gas_in_input[j] for j in S)
            Q_eff = Q - gas_input_S
            if Q_eff <= 0.0:
                continue

            alloc_S = gas_free_optimal_allocation(Q, pool_states, direction, S)
            raw_output = total_output_for_allocation(alloc_S, pool_states, direction)

            # Gas cost subtracted from output (in USDC)
            gas_usdc_S = sum(gas_costs_usdc[j] for j in S)
            if direction == "USDC->WETH":
                # raw_output is in WETH; convert to USDC for net comparison
                net_output = raw_output * _WETH_PRICE_USDC - gas_usdc_S
            else:
                net_output = raw_output - gas_usdc_S

            if net_output > best_net:
                best_net = net_output
                best_alloc = alloc_S

    # Fallback: if all subsets infeasible (gas > budget for every non-empty subset),
    # route all Q to the single pool with the best spot price, ignoring gas.
    if best_net == -np.inf:
        spots = [_spot_price(pid, pool_states[pid], direction) for pid in POOL_IDS]
        best_pid = int(np.argmax(spots))
        best_alloc = np.zeros(_N_POOLS)
        best_alloc[best_pid] = Q
        raw_out = total_output_for_allocation(best_alloc, pool_states, direction)
        gas_usdc_S = gas_costs_usdc[best_pid]
        if direction == "USDC->WETH":
            best_net = raw_out * _WETH_PRICE_USDC - gas_usdc_S
        else:
            best_net = raw_out - gas_usdc_S

    return best_alloc, float(best_net)


def compute_realized_output(
    realized_alloc: np.ndarray, pool_states: list, direction: str
) -> float:
    """Evaluate the realized route under current pool states."""
    return total_output_for_allocation(realized_alloc, pool_states, direction)


# ---------------------------------------------------------------------------
# Block-state lookup
# ---------------------------------------------------------------------------

def _get_pool_states_for_block(pool_states_by_block: dict, block: int) -> dict:
    """Return pool state dict for the largest stored block <= `block`."""
    sorted_blocks = sorted(pool_states_by_block.keys())
    idx = bisect.bisect_right(sorted_blocks, block) - 1
    if idx < 0:
        idx = 0
    return pool_states_by_block[sorted_blocks[idx]]


# ---------------------------------------------------------------------------
# Shortfall computation over all transactions
# ---------------------------------------------------------------------------

def compute_shortfalls(
    routes_df: pd.DataFrame, pool_states_by_block: dict
) -> pd.DataFrame:
    """Compute SCO/FVO/G-FVO shortfalls for each transaction.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Route-level DataFrame with q_pool0..q_pool3, Q_total, direction,
        block_number, usdc_equiv_size columns.
    pool_states_by_block : dict
        {block_num: {pool_id: state_dict}} from generate_pool_states.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with columns:
        sco_output, fvo_output, gfvo_output, realized_output_evaluated,
        shortfall_sco, shortfall_fvo, shortfall_gfvo  (USDC),
        shortfall_sco_bps, shortfall_fvo_bps, shortfall_gfvo_bps.
    """
    df = routes_df.copy()

    sco_outputs, fvo_outputs, gfvo_outputs, realized_outputs = [], [], [], []

    for _, row in df.iterrows():
        block = int(row["block_number"])
        direction = str(row["direction"])
        Q = float(row["Q_total"])
        realized_alloc = np.array([float(row[c]) for c in _Q_COLS])

        block_state = _get_pool_states_for_block(pool_states_by_block, block)
        pool_states = [block_state[pid] for pid in POOL_IDS]

        _, sco_out = compute_sco(Q, realized_alloc, pool_states, direction)
        _, fvo_out = compute_fvo(Q, pool_states, direction)
        _, gfvo_net = compute_gfvo(Q, pool_states, direction, list(GAS_COSTS_USDC))
        realized_out = compute_realized_output(realized_alloc, pool_states, direction)

        sco_outputs.append(sco_out)
        fvo_outputs.append(fvo_out)
        gfvo_outputs.append(gfvo_net)
        realized_outputs.append(realized_out)

    df["sco_output"] = sco_outputs
    df["fvo_output"] = fvo_outputs
    df["gfvo_output"] = gfvo_outputs
    df["realized_output_evaluated"] = realized_outputs

    # Convert to USDC if direction is USDC->WETH (output is in WETH)
    def _to_usdc(col_series, direction_series):
        out = col_series.copy()
        mask = direction_series == "USDC->WETH"
        out[mask] = out[mask] * _WETH_PRICE_USDC
        return out

    dir_col = df["direction"]
    sco_usdc = _to_usdc(df["sco_output"], dir_col)
    fvo_usdc = _to_usdc(df["fvo_output"], dir_col)
    realized_usdc = _to_usdc(df["realized_output_evaluated"], dir_col)
    ref_size = df["usdc_equiv_size"].clip(lower=1.0)

    df["shortfall_sco"] = (sco_usdc - realized_usdc).clip(lower=0.0)
    df["shortfall_fvo"] = (fvo_usdc - realized_usdc).clip(lower=0.0)
    df["shortfall_gfvo"] = (df["gfvo_output"] - realized_usdc).clip(lower=0.0)

    df["shortfall_sco_bps"] = df["shortfall_sco"] / ref_size * 10_000.0
    df["shortfall_fvo_bps"] = df["shortfall_fvo"] / ref_size * 10_000.0
    df["shortfall_gfvo_bps"] = df["shortfall_gfvo"] / ref_size * 10_000.0

    return df


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(n_synthetic: int = 50_000, save_results: bool = True) -> dict:
    """Run the full Experiment 2 pipeline.

    1. Generate synthetic data.
    2. Build route-level dataset.
    3. Compute SCO/FVO/G-FVO benchmarks.
    4. Aggregate shortfall statistics.

    Returns
    -------
    dict
        Summary statistics and full enriched DataFrame.
    """
    from exp.exp1_baseline import build_route_level_dataset, compute_concentration_metrics

    print(f"[Exp2] Generating {n_synthetic:,} synthetic transactions …")
    _ds = generate_synthetic_dataset(n_transactions=n_synthetic)
    transactions_df = _ds["transactions"]
    swap_events_df = _ds["swap_events"]
    pool_states_by_block = _ds["pool_states_by_block"]

    routes_df = build_route_level_dataset(swap_events_df)
    routes_df = apply_filters(routes_df)
    routes_df = compute_concentration_metrics(routes_df)
    print(f"[Exp2] Computing benchmarks for {len(routes_df):,} routes …")

    enriched_df = compute_shortfalls(routes_df, pool_states_by_block)

    summary = {
        "n_transactions": len(enriched_df),
        "mean_shortfall_sco_bps": float(enriched_df["shortfall_sco_bps"].mean()),
        "mean_shortfall_fvo_bps": float(enriched_df["shortfall_fvo_bps"].mean()),
        "mean_shortfall_gfvo_bps": float(enriched_df["shortfall_gfvo_bps"].mean()),
        "median_shortfall_sco_bps": float(enriched_df["shortfall_sco_bps"].median()),
        "median_shortfall_fvo_bps": float(enriched_df["shortfall_fvo_bps"].median()),
        "median_shortfall_gfvo_bps": float(enriched_df["shortfall_gfvo_bps"].median()),
        "total_shortfall_fvo_usdc": float(enriched_df["shortfall_fvo"].sum()),
        "total_shortfall_gfvo_usdc": float(enriched_df["shortfall_gfvo"].sum()),
    }

    print("[Exp2] Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:,.4f}" if isinstance(v, float) else f"  {k}: {v:,}")

    if save_results:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        enriched_df.to_csv(_RESULTS_DIR / "exp2_shortfalls.csv", index=False)
        pd.DataFrame([summary]).to_csv(_RESULTS_DIR / "exp2_summary.csv", index=False)
        print(f"[Exp2] Results saved to {_RESULTS_DIR}")

    return {
        "n_transactions": summary["n_transactions"],
        "mean_shortfall_sco_bps": summary["mean_shortfall_sco_bps"],
        "mean_shortfall_fvo_bps": summary["mean_shortfall_fvo_bps"],
        "mean_shortfall_gfvo_bps": summary["mean_shortfall_gfvo_bps"],
        "summary": summary,
        "enriched_df": enriched_df,
    }


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Exp2 complete ===")
    print(results["summary"])
