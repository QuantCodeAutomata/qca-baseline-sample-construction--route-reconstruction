"""
Exact AMM output functions for Uniswap V2 (constant-product) and
Uniswap V3 (concentrated liquidity, single- and multi-tick).

All floating-point V3 prices are stored as plain Python floats using the
same unit as Uniswap's sqrtPriceX96 (i.e. sqrt_price = sqrtPriceX96 / 2**96
is NOT pre-divided here; callers pass raw sqrtPriceX96 values as floats and
the functions work in those units internally).
"""

import math
import copy
from typing import Optional


# ---------------------------------------------------------------------------
# Uniswap V2 – constant-product with proportional fee
# ---------------------------------------------------------------------------

def v2_get_amount_out(
    amount_in: int,
    reserve_in: int,
    reserve_out: int,
    fee_bps: int = 30,
) -> int:
    """Return the integer output amount for an exact-input V2 swap.

    Formula (integer arithmetic, floor division):
        amount_out = amount_in_with_fee * reserve_out
                     / (reserve_in * 10000 + amount_in_with_fee)
    where amount_in_with_fee = amount_in * (10000 - fee_bps).

    Args:
        amount_in:   Exact input token amount (integer, raw units).
        reserve_in:  Pool reserve of the input token (integer, raw units).
        reserve_out: Pool reserve of the output token (integer, raw units).
        fee_bps:     Fee in basis points (default 30 → 0.30 %).

    Returns:
        Output amount as a non-negative integer (floor division).

    Raises:
        ValueError: If any argument is non-positive.
    """
    if amount_in <= 0 or reserve_in <= 0 or reserve_out <= 0:
        raise ValueError("amount_in, reserve_in, and reserve_out must be positive.")

    k = 10_000 - fee_bps
    numerator = amount_in * k * reserve_out
    denominator = reserve_in * 10_000 + amount_in * k
    return numerator // denominator


def v2_marginal_output(
    q: float,
    reserve_in: float,
    reserve_out: float,
    fee_bps: int = 30,
) -> float:
    """Marginal output rate d(output)/dq for a Uniswap V2 pool at trade size q.

    Analytical derivative of the V2 output function with respect to q:

        d(output)/dq = (10000 - fee_bps) * reserve_out * reserve_in * 10000
                       / (reserve_in * 10000 + q * (10000 - fee_bps))^2

    Args:
        q:           Trade size (input token amount, float).
        reserve_in:  Pool reserve of the input token.
        reserve_out: Pool reserve of the output token.
        fee_bps:     Fee in basis points.

    Returns:
        Marginal output rate (dimensionless ratio, always > 0).
    """
    k = 10_000 - fee_bps
    denom = reserve_in * 10_000 + q * k
    return k * reserve_out * reserve_in * 10_000 / (denom * denom)


def v2_inverse_marginal_price(
    lam: float,
    reserve_in: float,
    reserve_out: float,
    fee_bps: int = 30,
) -> float:
    """Return q ≥ 0 such that v2_marginal_output(q, ...) == lam.

    Analytical inverse derived by solving the marginal-output equation:

        q = (sqrt(reserve_out * reserve_in * (10000 - fee_bps) / (10000 * lam))
             - reserve_in) * 10000 / (10000 - fee_bps)

    Clamped to [0, ∞).

    Args:
        lam:         Target marginal output rate (must be positive).
        reserve_in:  Pool reserve of the input token.
        reserve_out: Pool reserve of the output token.
        fee_bps:     Fee in basis points.

    Returns:
        Trade size q ≥ 0.

    Raises:
        ValueError: If lam ≤ 0.
    """
    if lam <= 0:
        raise ValueError("lam must be positive.")

    k = 10_000 - fee_bps
    inner = reserve_out * reserve_in * k / (10_000 * lam)
    q = (math.sqrt(inner) - reserve_in) * 10_000 / k
    return max(0.0, q)


def v2_total_output(
    q: float,
    reserve_in: float,
    reserve_out: float,
    fee_bps: int = 30,
) -> float:
    """Exact output for a trade of size q on a Uniswap V2 pool.

    Uses the continuous form of the V2 formula:

        output = q * (10000 - fee_bps) * reserve_out
                 / (reserve_in * 10000 + q * (10000 - fee_bps))

    Args:
        q:           Trade size (input token amount, float).
        reserve_in:  Pool reserve of the input token.
        reserve_out: Pool reserve of the output token.
        fee_bps:     Fee in basis points.

    Returns:
        Output amount (float).
    """
    k = 10_000 - fee_bps
    numerator = q * k * reserve_out
    denominator = reserve_in * 10_000 + q * k
    return numerator / denominator


# ---------------------------------------------------------------------------
# Uniswap V3 – concentrated liquidity, single tick range
# ---------------------------------------------------------------------------

#: Q96 scaling factor used by Uniswap V3's sqrtPriceX96 representation.
Q96: float = float(2**96)


def v3_get_amount_out_single_range(
    amount_in: float,
    liquidity: float,
    sqrt_price: float,
    token_in_is_token0: bool,
    fee_bps: int,
) -> tuple[float, float]:
    """Exact output for a V3 swap fully contained within one tick range.

    All prices are in raw sqrtPriceX96 units (i.e. ``sqrt(P) * 2**96`` cast
    to float).  Liquidity ``L`` must be in the matching Uniswap V3 unit
    (same convention as the on-chain ``liquidity`` slot).

    Q = 2**96

    Token0-in (e.g. WETH → USDC, price decreases):
        amount_in_eff  = amount_in * (10000 - fee_bps) / 10000
        new_sqrt_price = L * Q * sqrt_price / (L * Q + amount_in_eff * sqrt_price)
        amount_out     = L * (sqrt_price - new_sqrt_price) / Q       # token1

    Token1-in (e.g. USDC → WETH, price increases):
        amount_in_eff  = amount_in * (10000 - fee_bps) / 10000
        new_sqrt_price = sqrt_price + amount_in_eff * Q / L
        amount_out     = L * Q * (1/sqrt_price - 1/new_sqrt_price)   # token0

    Args:
        amount_in:          Exact input token amount (float, raw units).
        liquidity:          Active liquidity (Uniswap V3 ``liquidity`` slot).
        sqrt_price:         Current sqrtPriceX96 (float).
        token_in_is_token0: True if input token is token0 (WETH).
        fee_bps:            Pool fee in basis points.

    Returns:
        ``(amount_out, new_sqrt_price)``: Output amount and updated sqrtPriceX96.

    Raises:
        ValueError: If amount_in, liquidity, or sqrt_price are non-positive.
    """
    if amount_in <= 0 or liquidity <= 0 or sqrt_price <= 0:
        raise ValueError("amount_in, liquidity, and sqrt_price must be positive.")

    f = (10_000 - fee_bps) / 10_000
    L = liquidity
    p = sqrt_price
    q_amt = amount_in * f  # effective amount after fee

    if token_in_is_token0:
        # Numerically stable: compute delta directly to avoid p - new_p cancellation
        # new_sqrt_price = L*Q*p / (L*Q + q_amt*p)
        # amount_out = L * (p - new_sqrt_price) / Q = L * q_amt * p^2 / (Q * (L*Q + q_amt*p))
        denom = L * Q96 + q_amt * p
        new_sqrt_price = L * Q96 * p / denom
        amount_out = L * q_amt * p * p / (Q96 * denom)
    else:
        # Numerically stable: avoid 1/p - 1/new_p cancellation when delta_sqrt << p.
        # new_sqrt_price = p + q_amt*Q/L (delta_sqrt = q_amt*Q/L)
        # amount_out = L*Q*(1/p - 1/new_p) = Q^2 * q_amt / (p * new_sqrt_price)
        delta_sqrt = q_amt * Q96 / L
        new_sqrt_price = p + delta_sqrt
        amount_out = Q96 * Q96 * q_amt / (p * new_sqrt_price)

    return amount_out, new_sqrt_price


def v3_marginal_output(
    q: float,
    liquidity: float,
    sqrt_price: float,
    token_in_is_token0: bool,
    fee_bps: int,
) -> float:
    """Marginal output rate d(output)/dq for a single-range V3 pool at trade size q.

    Q = 2**96, f = (10000 - fee_bps) / 10000.

    Token0-in:
        d(output)/dq = L² · f · sqrt_price²
                       / (L · Q + f · q · sqrt_price)²

    Token1-in:
        d(output)/dq = Q² · f · L²
                       / (L · sqrt_price + f · q · Q)²

    Args:
        q:                  Trade size (float, raw input token units).
        liquidity:          Active liquidity (Uniswap V3 ``liquidity`` slot).
        sqrt_price:         Current sqrtPriceX96 (float).
        token_in_is_token0: True if input token is token0 (WETH).
        fee_bps:            Pool fee in basis points.

    Returns:
        Marginal output rate (float, always > 0 within the tick range).
    """
    f = (10_000 - fee_bps) / 10_000
    L = liquidity
    p = sqrt_price

    if token_in_is_token0:
        denom = L * Q96 + f * q * p
        return L * L * f * p * p / (denom * denom)
    else:
        denom = L * p + f * q * Q96
        return Q96 * Q96 * f * L * L / (denom * denom)


def v3_inverse_marginal_price(
    lam: float,
    liquidity: float,
    sqrt_price: float,
    token_in_is_token0: bool,
    fee_bps: int,
) -> float:
    """Return q ≥ 0 such that ``v3_marginal_output(q, ...) == lam``.

    Analytical inverse of ``v3_marginal_output``.

    Q = 2**96, f = (10000 - fee_bps) / 10000.

    Token0-in:
        q = L · (sqrt_price · sqrt(f / lam) − Q) / (f · sqrt_price)

    Token1-in:
        q = L · (Q · sqrt(f / lam) − sqrt_price) / (f · Q)

    Clamped to [0, ∞); returns 0 when ``lam`` is at or above the at-zero
    marginal rate (i.e. the pool cannot produce that marginal rate).

    Args:
        lam:                Target marginal output rate (must be positive).
        liquidity:          Active liquidity (Uniswap V3 ``liquidity`` slot).
        sqrt_price:         Current sqrtPriceX96 (float).
        token_in_is_token0: True if input token is token0 (WETH).
        fee_bps:            Pool fee in basis points.

    Returns:
        Trade size q ≥ 0.

    Raises:
        ValueError: If lam ≤ 0.
    """
    if lam <= 0:
        raise ValueError("lam must be positive.")

    f = (10_000 - fee_bps) / 10_000
    L = liquidity
    p = sqrt_price

    if token_in_is_token0:
        inner = f / lam
        if inner < 0:
            return 0.0
        q = L * (p * math.sqrt(inner) - Q96) / (f * p)
    else:
        inner = f / lam
        if inner < 0:
            return 0.0
        q = L * (Q96 * math.sqrt(inner) - p) / (f * Q96)

    return max(0.0, q)


def v3_total_output(
    q: float,
    liquidity: float,
    sqrt_price: float,
    token_in_is_token0: bool,
    fee_bps: int,
) -> float:
    """Exact output for a trade of size q fully within a single V3 tick range.

    Q = 2**96, f = (10000 - fee_bps) / 10000.

    Token0-in:
        output = L · f · q · sqrt_price² / (Q · (L · Q + f · q · sqrt_price))

    Token1-in:
        output = Q² · f · q · L / (sqrt_price · (L · sqrt_price + f · q · Q))

    Args:
        q:                  Trade size (float, raw input token units).
        liquidity:          Active liquidity (Uniswap V3 ``liquidity`` slot).
        sqrt_price:         Current sqrtPriceX96 (float).
        token_in_is_token0: True if input token is token0 (WETH).
        fee_bps:            Pool fee in basis points.

    Returns:
        Output amount (float, raw output token units).
    """
    f = (10_000 - fee_bps) / 10_000
    L = liquidity
    p = sqrt_price

    if token_in_is_token0:
        return L * f * q * p * p / (Q96 * (L * Q96 + f * q * p))
    else:
        return Q96 * Q96 * f * q * L / (p * (L * p + f * q * Q96))


# ---------------------------------------------------------------------------
# Uniswap V3 – multi-tick swap simulation
# ---------------------------------------------------------------------------

def v3_simulate_swap(
    amount_in: float,
    pool_state: dict,
    token_in_is_token0: bool,
) -> tuple[float, dict]:
    """Simulate an exact-input Uniswap V3 swap across one or more tick ranges.

    The simulation reproduces the Uniswap V3 on-chain swap loop: it processes
    input token amounts step by step, crossing tick boundaries and updating
    active liquidity via ``liquidity_net`` until the full ``amount_in`` is
    consumed.

    Pool state schema::

        {
            "sqrt_price_x96": float,   # current sqrtPriceX96
            "liquidity": float,        # active liquidity at current price
            "fee_bps": int,            # pool fee in basis points
            "ticks": [                 # list of initialised tick objects
                {
                    "tick": int,                  # tick index
                    "sqrt_price_x96": float,      # sqrtPriceX96 at this tick
                    "liquidity_net": float,        # Δliquidity when crossed
                },
                ...
            ],
        }

    Tick-crossing conventions (matching Uniswap V3):
    - Token0-in (price decreasing): ticks below current price are considered,
      visited in descending order. ``liquidity_net`` is *subtracted* on cross.
    - Token1-in (price increasing): ticks above current price are considered,
      visited in ascending order. ``liquidity_net`` is *added* on cross.

    If liquidity in a range drops to zero or below, that range is skipped and
    no further ticks are available; the remaining input is returned unswapped.

    Args:
        amount_in:          Total exact-input amount (float, raw token units).
        pool_state:         Current pool state dict (see schema above). The
                            original dict is **not** mutated.
        token_in_is_token0: True if the input token is token0 (e.g. WETH).

    Returns:
        (amount_out, new_pool_state): Total output received and an updated
        copy of ``pool_state`` reflecting the post-swap price and liquidity.

    Raises:
        ValueError: If ``amount_in`` is non-positive or pool state is invalid.
    """
    if amount_in <= 0:
        raise ValueError("amount_in must be positive.")

    state = copy.deepcopy(pool_state)
    sqrt_price = state["sqrt_price_x96"]
    liquidity = state["liquidity"]
    fee_bps = state["fee_bps"]
    ticks: list[dict] = state["ticks"]
    f = (10_000 - fee_bps) / 10_000

    # Sort tick boundaries relevant to the current direction.
    if token_in_is_token0:
        # Price decreases → visit ticks below current price, high to low.
        relevant = sorted(
            [t for t in ticks if t["sqrt_price_x96"] < sqrt_price],
            key=lambda t: t["sqrt_price_x96"],
            reverse=True,
        )
    else:
        # Price increases → visit ticks above current price, low to high.
        relevant = sorted(
            [t for t in ticks if t["sqrt_price_x96"] > sqrt_price],
            key=lambda t: t["sqrt_price_x96"],
        )

    amount_remaining = amount_in
    total_out = 0.0

    for tick_obj in relevant:
        if liquidity <= 0:
            break

        sqrt_price_next = tick_obj["sqrt_price_x96"]

        # Effective input amount needed to move price to the tick boundary.
        # Token0-in (from SqrtPriceMath getAmount0Delta):
        #   amount_in_eff = L * Q * (1/sqrt_price_next - 1/sqrt_price)
        # Token1-in (from SqrtPriceMath getAmount1Delta):
        #   amount_in_eff = L * (sqrt_price_next - sqrt_price) / Q
        if token_in_is_token0:
            amount_in_eff_to_tick = liquidity * Q96 * (
                1.0 / sqrt_price_next - 1.0 / sqrt_price
            )
        else:
            amount_in_eff_to_tick = liquidity * (sqrt_price_next - sqrt_price) / Q96

        # Gross input (before fee deduction) to reach the boundary.
        amount_in_gross_to_tick = amount_in_eff_to_tick / f

        if amount_remaining <= amount_in_gross_to_tick:
            # Swap finishes within this range; no tick crossing needed.
            amount_in_eff = amount_remaining * f
            if token_in_is_token0:
                new_sqrt_price = liquidity * Q96 * sqrt_price / (
                    liquidity * Q96 + amount_in_eff * sqrt_price
                )
                step_out = liquidity * (sqrt_price - new_sqrt_price) / Q96
            else:
                new_sqrt_price = sqrt_price + amount_in_eff * Q96 / liquidity
                step_out = liquidity * Q96 * (1.0 / sqrt_price - 1.0 / new_sqrt_price)

            total_out += step_out
            sqrt_price = new_sqrt_price
            amount_remaining = 0.0
            break

        # Consume the entire range up to the tick boundary.
        if token_in_is_token0:
            step_out = liquidity * (sqrt_price - sqrt_price_next) / Q96
        else:
            step_out = liquidity * Q96 * (1.0 / sqrt_price - 1.0 / sqrt_price_next)

        total_out += step_out
        amount_remaining -= amount_in_gross_to_tick
        sqrt_price = sqrt_price_next

        # Apply tick crossing: update active liquidity.
        liq_net = tick_obj["liquidity_net"]
        if token_in_is_token0:
            liquidity -= liq_net   # moving down crosses the tick in reverse
        else:
            liquidity += liq_net

    # If input remains and no more ticks exist, consume it at the current price
    # within an unbounded range (best-effort; liquidity guard applies).
    if amount_remaining > 0 and liquidity > 0:
        amount_in_eff = amount_remaining * f
        if token_in_is_token0:
            new_sqrt_price = liquidity * Q96 * sqrt_price / (
                liquidity * Q96 + amount_in_eff * sqrt_price
            )
            step_out = liquidity * (sqrt_price - new_sqrt_price) / Q96
        else:
            new_sqrt_price = sqrt_price + amount_in_eff * Q96 / liquidity
            step_out = liquidity * Q96 * (1.0 / sqrt_price - 1.0 / new_sqrt_price)

        total_out += step_out
        sqrt_price = new_sqrt_price

    # Write updated state back.
    state["sqrt_price_x96"] = sqrt_price
    state["liquidity"] = liquidity

    return total_out, state
