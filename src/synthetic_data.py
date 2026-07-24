"""Synthetic WETH-USDC swap data generator for testing when real blockchain data is unavailable.

Target statistics from the paper:
    n_transactions  : ~2,978,019  (default n=50_000 for fast runs)
    total USDC vol  : ~1.2042e11
    avg trade size  : ~40,437 USDC
    activation_rate : ~0.283
    entropy         : ~0.048
    HHI             : ~0.971
"""

import numpy as np
import pandas as pd

try:
    from src.config import (
        POOL_IDS,
        POOL_ADDRESSES,
        POOL_PROTOCOLS,
        POOL_FEE_BPS,
        GAS_COSTS_USDC,
        ROUTER_ADDRESSES,
    )
except ImportError:
    POOL_IDS = [0, 1, 2, 3]
    POOL_ADDRESSES = {
        0: "0xB4e16d0168e52d35CaCD2c6185b44281Ec28C9Dc",
        1: "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        2: "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D5",
        3: "0x7BeA39867e4169DBe237d55C8242a8f2fcDcc387",
    }
    POOL_PROTOCOLS = {0: "v2", 1: "v3", 2: "v3", 3: "v3"}
    POOL_FEE_BPS = {0: 30, 1: 5, 2: 30, 3: 100}
    GAS_COSTS_USDC = [5.0, 3.0, 3.0, 3.0]
    ROUTER_ADDRESSES = {
        "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "UniswapV3Router",
        "0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B": "UniswapUniversalRouter",
        "0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F": "SushiswapRouter",
    }

_BLOCK_START = 19_500_000
_BLOCK_END = 23_000_000
_WETH_PRICE_USDC = 3_000.0
_Q96 = 2 ** 96

# Log-normal parameters for USDC trade sizes: mean ~40,437, heavy tail
# E[X] = exp(mu + sigma^2/2) = 40437  →  mu = log(40437) - sigma^2/2
_LN_SIGMA = 2.0
_LN_MU = np.log(40_437.0) - _LN_SIGMA ** 2 / 2.0

# Multi-pool routing mix (must sum to 1): 2-pool, 3-pool, 4-pool
_MULTI_POOL_MIX = [0.72, 0.20, 0.06, 0.02]  # single, 2-, 3-, 4-pool fractions
# Dirichlet concentration parameter for multi-pool share generation (alpha=0.1 → HHI≈0.971)
_DIR_ALPHA = 0.1

# Base V2 reserve levels
_V2_RESERVE_WETH = 5_000.0  # WETH
_V2_RESERVE_USDC = _V2_RESERVE_WETH * _WETH_PRICE_USDC  # USDC

# V3 liquidity by pool id (higher liquidity for lower fee tier)
_V3_BASE_LIQUIDITY = {1: 5e22, 2: 2e22, 3: 5e21}

_ROUTER_ADDR_LIST = list(ROUTER_ADDRESSES.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sqrt_price_x96_from_usdc_per_weth(price: float) -> float:
    """Compute sqrt_price_x96 for a given USDC/WETH price."""
    return float(np.sqrt(price)) * _Q96


def _approx_v2_output(q_in: float, reserve_in: float, reserve_out: float, fee_bps: int) -> float:
    """Uniswap V2 constant-product output."""
    f = 1.0 - fee_bps / 10_000.0
    return reserve_out * q_in * f / (reserve_in + q_in * f)


def _approx_v3_output(
    q_in: float, sqrt_price_x96: float, liquidity: float, fee_bps: int, direction: str
) -> float:
    """Simplified V3 output using single-range approximation."""
    P = sqrt_price_x96 / _Q96  # sqrt(USDC/WETH)
    f = 1.0 - fee_bps / 10_000.0
    q_eff = q_in * f
    if direction == "WETH->USDC":
        # WETH in, USDC out: Δy = L·q_eff·P²/(L + q_eff·P)
        return liquidity * q_eff * P * P / (liquidity + q_eff * P)
    else:
        # USDC in, WETH out: Δx = L·q_eff/(L·P² + q_eff·P)
        return liquidity * q_eff / (liquidity * P * P + q_eff * P)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pool_states(n_blocks: int = 1000, seed: int = 42) -> dict:
    """Generate synthetic pool states for each of n_blocks sampled blocks.

    V2 pool: reserve_weth, reserve_usdc with small random walk.
    V3 pools: sqrt_price_x96, liquidity, fee_bps, ticks (1 tick range).

    Returns
    -------
    dict
        {block_num: {pool_id: state_dict}}
    """
    rng = np.random.default_rng(seed)

    block_step = max(1, (_BLOCK_END - _BLOCK_START) // n_blocks)
    blocks = [_BLOCK_START + i * block_step for i in range(n_blocks)]

    price = _WETH_PRICE_USDC
    v2_weth = _V2_RESERVE_WETH
    v2_usdc = _V2_RESERVE_USDC

    states: dict = {}
    for block in blocks:
        # Price random walk (0.05% vol per block step)
        price *= np.exp(rng.normal(0.0, 0.0005))
        price = max(price, 100.0)

        # V2 reserves: maintain approximate AMM invariant under price drift
        k = v2_weth * v2_usdc
        v2_weth = np.sqrt(k / price)
        v2_usdc = np.sqrt(k * price)

        sqrt_px96 = _sqrt_price_x96_from_usdc_per_weth(price)

        block_states: dict = {}

        # Pool 0: Uniswap V2
        block_states[0] = {
            "protocol": "v2",
            "reserve_weth": float(v2_weth),
            "reserve_usdc": float(v2_usdc),
            "fee_bps": POOL_FEE_BPS[0],
        }

        # Pools 1-3: Uniswap V3 (fee tiers 5, 30, 100 bps)
        for pid in [1, 2, 3]:
            liq = _V3_BASE_LIQUIDITY[pid] * rng.uniform(0.95, 1.05)
            block_states[pid] = {
                "protocol": "v3",
                "sqrt_price_x96": float(sqrt_px96),
                "liquidity": float(liq),
                "fee_bps": POOL_FEE_BPS[pid],
                "ticks": [(-887_272, 887_272, float(liq))],
            }

        states[block] = block_states

    return states


def generate_transactions(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    """Generate n synthetic transaction-level routes.

    Routing mix:
        72.0% single-pool  → HHI = 1.0
        20.0% 2-pool       → Dirichlet(alpha) split
         6.0% 3-pool
         2.0% 4-pool
    Dirichlet alpha=0.1 → expected overall HHI ≈ 0.971.

    Returns
    -------
    pd.DataFrame
        Columns: tx_hash, block_number, tx_index, timestamp, to_address,
                 direction, q_pool0..q_pool3, Q_total, usdc_equiv_size.
    """
    rng = np.random.default_rng(seed)

    # ---- USDC trade sizes -----------------------------------------------
    usdc_sizes = rng.lognormal(mean=_LN_MU, sigma=_LN_SIGMA, size=n)
    usdc_sizes = np.clip(usdc_sizes, 100.0, None)  # floor at $100 (paper filter threshold)

    # ---- Directions ---------------------------------------------------------
    directions = rng.choice(
        ["WETH->USDC", "USDC->WETH"], size=n, p=[0.55, 0.45]
    )

    # ---- Routing type -------------------------------------------------------
    route_types = rng.choice([1, 2, 3, 4], size=n, p=_MULTI_POOL_MIX)

    # ---- Block numbers & tx indices -----------------------------------------
    n_unique_blocks = 500
    block_choices = np.sort(
        rng.integers(_BLOCK_START, _BLOCK_END, size=n_unique_blocks)
    )
    block_numbers = rng.choice(block_choices, size=n)
    # Assign tx_index within block (0-based, sorted later)
    tx_indices = rng.integers(0, 300, size=n)
    timestamps = (block_numbers - _BLOCK_START) * 12 + 1_711_900_000  # ~12s per block

    # ---- Router addresses ---------------------------------------------------
    addr_pool = _ROUTER_ADDR_LIST + ["unlabeled"] * 3
    to_addresses = rng.choice(addr_pool, size=n)

    # ---- Allocations --------------------------------------------------------
    q_pools = np.zeros((n, 4))

    for i in range(n):
        k = int(route_types[i])  # number of active pools
        usdc = usdc_sizes[i]
        direction = directions[i]

        # Total input quantity in the input token
        if direction == "WETH->USDC":
            Q_total = usdc / _WETH_PRICE_USDC
        else:
            Q_total = usdc

        if k == 1:
            # Single pool: all flow in one random pool
            chosen = int(rng.integers(0, 4))
            q_pools[i, chosen] = Q_total
        else:
            # Multi-pool: Dirichlet-distributed shares over k random pools
            chosen_pools = rng.choice(4, size=k, replace=False)
            shares = rng.dirichlet(np.full(k, _DIR_ALPHA))
            for j, pid in enumerate(chosen_pools):
                q_pools[i, pid] = Q_total * shares[j]

    Q_totals = q_pools.sum(axis=1)

    # ---- Assemble DataFrame -------------------------------------------------
    # Generate tx hashes as pairs of 32-bit values to stay within int64 bounds
    hi = rng.integers(0, 2 ** 31, size=n)
    lo = rng.integers(0, 2 ** 31, size=n)
    tx_hashes = [f"0x{hi[i]:08x}{lo[i]:08x}" for i in range(n)]

    df = pd.DataFrame(
        {
            "tx_hash": tx_hashes,
            "block_number": block_numbers,
            "tx_index": tx_indices,
            "timestamp": timestamps,
            "to_address": to_addresses,
            "direction": directions,
            "q_pool0": q_pools[:, 0],
            "q_pool1": q_pools[:, 1],
            "q_pool2": q_pools[:, 2],
            "q_pool3": q_pools[:, 3],
            "Q_total": Q_totals,
            "usdc_equiv_size": usdc_sizes,
        }
    )

    return df.reset_index(drop=True)


def generate_pool_swap_events(transactions_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Expand transaction-level routes into pool-level swap events.

    One row per (transaction, active pool) pair.

    Returns
    -------
    pd.DataFrame
        Columns: tx_hash, block_number, tx_index, log_index, pool_id,
                 pool_address, direction, amount_in, amount_out,
                 sender, recipient.
    """
    rng = np.random.default_rng(seed)

    q_cols = ["q_pool0", "q_pool1", "q_pool2", "q_pool3"]
    rows = []
    log_counter: dict = {}

    sender_pool = (
        [f"0x{rng.integers(0, 2**40):010x}" for _ in range(20)]
    )

    for _, tx in transactions_df.iterrows():
        block = int(tx["block_number"])
        tx_idx = int(tx["tx_index"])
        direction = tx["direction"]
        block_log_key = (block, tx_idx)
        log_idx = log_counter.get(block_log_key, 0)

        for pid in POOL_IDS:
            q_in = float(tx[q_cols[pid]])
            if q_in <= 0.0:
                continue

            protocol = POOL_PROTOCOLS[pid]
            fee_bps = POOL_FEE_BPS[pid]

            if protocol == "v2":
                if direction == "WETH->USDC":
                    amount_out = _approx_v2_output(
                        q_in, _V2_RESERVE_WETH, _V2_RESERVE_USDC, fee_bps
                    )
                else:
                    amount_out = _approx_v2_output(
                        q_in, _V2_RESERVE_USDC, _V2_RESERVE_WETH, fee_bps
                    )
            else:
                sqrt_px96 = _sqrt_price_x96_from_usdc_per_weth(_WETH_PRICE_USDC)
                liq = _V3_BASE_LIQUIDITY[pid]
                amount_out = _approx_v3_output(
                    q_in, sqrt_px96, liq, fee_bps, direction
                )

            sender = sender_pool[int(rng.integers(0, len(sender_pool)))]
            recipient = tx["to_address"]

            rows.append(
                {
                    "tx_hash": tx["tx_hash"],
                    "block_number": block,
                    "tx_index": tx_idx,
                    "log_index": log_idx,
                    "pool_id": pid,
                    "pool_address": POOL_ADDRESSES[pid],
                    "direction": direction,
                    "amount_in": q_in,
                    "amount_out": float(amount_out),
                    "sender": sender,
                    "recipient": recipient,
                }
            )
            log_idx += 1

        log_counter[block_log_key] = log_idx

    return pd.DataFrame(rows).reset_index(drop=True)


def generate_synthetic_dataset(
    n_transactions: int = 50_000,
    seed: int = 42,
    *,
    n: int = None,
) -> dict:
    """Generate a complete synthetic dataset ready for analysis.

    Parameters
    ----------
    n_transactions : int
        Number of transactions to generate.
    seed : int
        Random seed.
    n : int, optional
        Alias for n_transactions (overrides it when supplied).

    Returns
    -------
    dict
        Keys: transactions (pd.DataFrame), swap_events (pd.DataFrame),
        pool_states_by_block (dict), routes (pd.DataFrame with concentration
        metrics pre-computed).
    """
    if n is not None:
        n_transactions = n

    transactions_df = generate_transactions(n=n_transactions, seed=seed)
    swap_events_df = generate_pool_swap_events(transactions_df, seed=seed)
    pool_states = generate_pool_states(n_blocks=1000, seed=seed)

    # Build route-level dataset with flow shares and concentration metrics
    routes_df = transactions_df.copy()

    # Alias direction column
    routes_df["direction_z"] = routes_df["direction"]

    # Compute flow shares x_pool0..3
    Q_total = routes_df["Q_total"].clip(lower=1e-30)
    for pid in POOL_IDS:
        routes_df[f"x_pool{pid}"] = routes_df[f"q_pool{pid}"] / Q_total

    # Concentration metrics inline (avoids circular import)
    X = routes_df[[f"x_pool{pid}" for pid in POOL_IDS]].values

    def _activation_rate(x: np.ndarray) -> float:
        return float(np.sum(x > 0)) / 4.0

    def _entropy(x: np.ndarray) -> float:
        pos = x[x > 0]
        return float(-np.sum(pos * np.log(pos)))

    def _hhi(x: np.ndarray) -> float:
        return float(np.sum(x ** 2))

    routes_df["activation_rate"] = [_activation_rate(X[i]) for i in range(len(routes_df))]
    routes_df["entropy"] = [_entropy(X[i]) for i in range(len(routes_df))]
    routes_df["hhi"] = [_hhi(X[i]) for i in range(len(routes_df))]

    return {
        "transactions": transactions_df,
        "swap_events": swap_events_df,
        "pool_states_by_block": pool_states,
        "routes": routes_df,
    }
