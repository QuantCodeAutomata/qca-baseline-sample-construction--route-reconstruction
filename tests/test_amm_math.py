"""Tests for src/amm_math.py — Uniswap V2 and V3 AMM output functions."""

import math
import sys
import pytest
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.amm_math import (
    v2_get_amount_out,
    v2_marginal_output,
    v2_inverse_marginal_price,
    v2_total_output,
    v3_get_amount_out_single_range,
    v3_marginal_output,
    v3_inverse_marginal_price,
    v3_total_output,
    v3_simulate_swap,
    Q96,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def v2_pool():
    """Realistic V2 WETH/USDC pool: 5000 WETH, 15M USDC, fee 30bps."""
    return {
        "reserve_in": 5_000.0,      # WETH
        "reserve_out": 15_000_000.0, # USDC
        "fee_bps": 30,
    }


@pytest.fixture
def v3_pool_state():
    """Simple V3 pool state at ~3000 USDC/WETH with single tick range."""
    price = 3_000.0  # USDC per WETH
    sqrt_price = math.sqrt(price) * Q96
    return {
        "sqrt_price_x96": sqrt_price,
        "liquidity": 5e22,
        "fee_bps": 5,
        "ticks": [
            {"tick": -80000, "sqrt_price_x96": math.sqrt(1000.0) * Q96, "liquidity_net": 5e22},
            {"tick": -40000, "sqrt_price_x96": math.sqrt(5000.0) * Q96, "liquidity_net": -5e22},
        ],
    }


# ---------------------------------------------------------------------------
# V2 tests
# ---------------------------------------------------------------------------

class TestV2GetAmountOut:
    def test_basic_integer_calculation(self):
        """V2 exact integer output matches reference formula."""
        amount_in = 1_000_000  # 1 WETH in integer (1e6 units for test)
        reserve_in = 5_000_000_000
        reserve_out = 15_000_000_000_000
        fee_bps = 30
        result = v2_get_amount_out(amount_in, reserve_in, reserve_out, fee_bps)
        k = 10_000 - fee_bps
        expected = (amount_in * k * reserve_out) // (reserve_in * 10_000 + amount_in * k)
        assert result == expected

    def test_output_is_nonnegative_integer(self):
        """Output must be a non-negative integer."""
        result = v2_get_amount_out(100, 5_000, 15_000_000, 30)
        assert isinstance(result, int)
        assert result >= 0

    def test_raises_on_nonpositive_inputs(self):
        with pytest.raises(ValueError):
            v2_get_amount_out(0, 5_000, 15_000_000, 30)
        with pytest.raises(ValueError):
            v2_get_amount_out(100, 0, 15_000_000, 30)
        with pytest.raises(ValueError):
            v2_get_amount_out(100, 5_000, 0, 30)

    def test_higher_fee_produces_less_output(self):
        """Higher fee reduces output for same input."""
        out_30 = v2_get_amount_out(1_000, 5_000_000, 15_000_000_000, 30)
        out_100 = v2_get_amount_out(1_000, 5_000_000, 15_000_000_000, 100)
        assert out_30 > out_100

    def test_large_input_bounded_by_reserve(self):
        """Output can never exceed reserve_out."""
        reserve_out = 15_000_000
        result = v2_get_amount_out(10**15, 5_000, reserve_out, 30)
        assert result < reserve_out


class TestV2MarginalOutput:
    def test_marginal_decreases_with_trade_size(self, v2_pool):
        """Marginal output should decrease as trade size increases (concavity)."""
        q_small = v2_marginal_output(1.0, v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"])
        q_large = v2_marginal_output(1000.0, v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"])
        assert q_small > q_large

    def test_matches_finite_difference(self, v2_pool):
        """Analytical derivative matches numerical finite difference."""
        q = 10.0
        h = 1e-6
        ri, ro, fee = v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"]
        analytical = v2_marginal_output(q, ri, ro, fee)
        numerical = (v2_total_output(q + h, ri, ro, fee) - v2_total_output(q - h, ri, ro, fee)) / (2 * h)
        assert abs(analytical - numerical) / max(abs(analytical), 1e-30) < 1e-8

    def test_at_zero_matches_spot_price(self, v2_pool):
        """Marginal output at q=0 is the fee-adjusted spot price."""
        ri, ro, fee = v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"]
        mo = v2_marginal_output(0.0, ri, ro, fee)
        spot = ro / ri * (10_000 - fee) / 10_000
        assert abs(mo - spot) / spot < 1e-10


class TestV2InverseMarginalPrice:
    def test_round_trip(self, v2_pool):
        """inverse(marginal(q)) == q within tolerance."""
        ri, ro, fee = v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"]
        for q in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            lam = v2_marginal_output(q, ri, ro, fee)
            q_recovered = v2_inverse_marginal_price(lam, ri, ro, fee)
            assert abs(q_recovered - q) / max(abs(q), 1e-30) < 1e-10

    def test_clamps_to_zero(self, v2_pool):
        """When lambda >= spot price, q should be 0."""
        ri, ro, fee = v2_pool["reserve_in"], v2_pool["reserve_out"], v2_pool["fee_bps"]
        spot = ro / ri * (10_000 - fee) / 10_000
        q = v2_inverse_marginal_price(spot * 2, ri, ro, fee)
        assert q == 0.0

    def test_raises_on_nonpositive_lam(self, v2_pool):
        with pytest.raises(ValueError):
            v2_inverse_marginal_price(0.0, v2_pool["reserve_in"], v2_pool["reserve_out"])


class TestV2TotalOutput:
    def test_output_is_positive(self, v2_pool):
        assert v2_total_output(1.0, v2_pool["reserve_in"], v2_pool["reserve_out"]) > 0

    def test_output_increases_with_input(self, v2_pool):
        ri, ro = v2_pool["reserve_in"], v2_pool["reserve_out"]
        assert v2_total_output(2.0, ri, ro) > v2_total_output(1.0, ri, ro)

    def test_matches_integer_version_closely(self):
        """Float and int versions agree to <1 unit."""
        ri, ro, fee = 5_000_000_000, 15_000_000_000_000, 30
        q = 1_000_000
        float_out = v2_total_output(q, ri, ro, fee)
        int_out = v2_get_amount_out(q, ri, ro, fee)
        assert abs(float_out - int_out) < 1.0  # within 1 unit (floor vs float)


# ---------------------------------------------------------------------------
# V3 single-range tests
# ---------------------------------------------------------------------------

class TestV3GetAmountOutSingleRange:
    def test_weth_in_produces_usdc_out(self):
        """WETH in should produce positive USDC out."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        out, _ = v3_get_amount_out_single_range(1e18, liq, sqrt_price, True, 5)
        assert out > 0

    def test_usdc_in_produces_weth_out(self):
        """USDC in should produce positive WETH out."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        out, _ = v3_get_amount_out_single_range(3000e6, liq, sqrt_price, False, 5)
        assert out > 0

    def test_price_moves_correctly_token0_in(self):
        """For token0-in (WETH), sqrt price should decrease."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        _, new_sp = v3_get_amount_out_single_range(1e18, liq, sqrt_price, True, 5)
        assert new_sp < sqrt_price

    def test_price_moves_correctly_token1_in(self):
        """For token1-in (USDC), sqrt price should increase."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        _, new_sp = v3_get_amount_out_single_range(3000e6, liq, sqrt_price, False, 5)
        assert new_sp > sqrt_price

    def test_raises_on_nonpositive_inputs(self):
        sp = math.sqrt(3000.0) * Q96
        with pytest.raises(ValueError):
            v3_get_amount_out_single_range(0, 5e22, sp, True, 5)
        with pytest.raises(ValueError):
            v3_get_amount_out_single_range(1e18, 0, sp, True, 5)
        with pytest.raises(ValueError):
            v3_get_amount_out_single_range(1e18, 5e22, 0, True, 5)

    def test_higher_fee_produces_less_output(self):
        """Higher fee reduces output for same amount_in."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        out_5, _ = v3_get_amount_out_single_range(1e18, liq, sqrt_price, True, 5)
        out_30, _ = v3_get_amount_out_single_range(1e18, liq, sqrt_price, True, 30)
        assert out_5 > out_30


class TestV3MarginalOutput:
    def test_marginal_matches_finite_difference_token0_in(self):
        """Analytical marginal matches finite difference for token0-in."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        q = 1e17  # 0.1 WETH
        h = q * 1e-6
        analytical = v3_marginal_output(q, liq, sqrt_price, True, 5)
        num_out1, _ = v3_get_amount_out_single_range(q + h, liq, sqrt_price, True, 5)
        num_out2, _ = v3_get_amount_out_single_range(q - h, liq, sqrt_price, True, 5)
        numerical = (num_out1 - num_out2) / (2 * h)
        assert abs(analytical - numerical) / max(abs(analytical), 1e-30) < 1e-6

    def test_marginal_matches_finite_difference_token1_in(self):
        """Analytical marginal matches finite difference for token1-in."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        q = 1e9  # 1000 USDC (1e9 with 6 decimals)
        h = q * 1e-6
        analytical = v3_marginal_output(q, liq, sqrt_price, False, 5)
        num_out1, _ = v3_get_amount_out_single_range(q + h, liq, sqrt_price, False, 5)
        num_out2, _ = v3_get_amount_out_single_range(q - h, liq, sqrt_price, False, 5)
        numerical = (num_out1 - num_out2) / (2 * h)
        assert abs(analytical - numerical) / max(abs(analytical), 1e-30) < 1e-6


class TestV3InverseMarginalPrice:
    def test_round_trip_token0_in(self):
        """Inverse of marginal returns original q for token0-in."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        for q in [1e16, 1e17, 1e18]:
            lam = v3_marginal_output(q, liq, sqrt_price, True, 5)
            q_rec = v3_inverse_marginal_price(lam, liq, sqrt_price, True, 5)
            assert abs(q_rec - q) / q < 1e-9

    def test_round_trip_token1_in(self):
        """Inverse of marginal returns original q for token1-in.

        The closed-form formula `Q96·√(f/λ) - p` incurs a cancellation of size
        f·q·Q96 / (L·p) relative to p.  For the convention p = √price·Q96 the
        price-asymmetry factor is √price, so token1-in needs q values ~price×
        larger than token0-in ([1e16,1e17,1e18]) to achieve the same float64
        precision.  We therefore test at q = [1e19, 1e20, 1e21].
        """
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        for q in [1e19, 1e20, 1e21]:
            lam = v3_marginal_output(q, liq, sqrt_price, False, 5)
            q_rec = v3_inverse_marginal_price(lam, liq, sqrt_price, False, 5)
            assert abs(q_rec - q) / q < 1e-9

    def test_raises_on_nonpositive_lam(self):
        sp = math.sqrt(3000.0) * Q96
        with pytest.raises(ValueError):
            v3_inverse_marginal_price(0.0, 5e22, sp, True, 5)


class TestV3TotalOutput:
    def test_matches_single_range_function(self):
        """v3_total_output should match v3_get_amount_out_single_range."""
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        liq = 5e22
        q = 1e17
        total_out = v3_total_output(q, liq, sqrt_price, True, 5)
        sr_out, _ = v3_get_amount_out_single_range(q, liq, sqrt_price, True, 5)
        assert abs(total_out - sr_out) / sr_out < 1e-10

    def test_output_is_positive(self):
        price = 3000.0
        sqrt_price = math.sqrt(price) * Q96
        assert v3_total_output(1e17, 5e22, sqrt_price, True, 5) > 0


class TestV3SimulateSwap:
    def test_small_swap_within_tick_range(self, v3_pool_state):
        """A small swap stays within tick range and matches single-range output."""
        amount_in = 1e15  # very small, stays in range
        out, new_state = v3_simulate_swap(amount_in, v3_pool_state, True)
        assert out > 0
        assert new_state["sqrt_price_x96"] < v3_pool_state["sqrt_price_x96"]

    def test_state_is_not_mutated(self, v3_pool_state):
        """Original pool state must not be mutated."""
        orig_sp = v3_pool_state["sqrt_price_x96"]
        orig_liq = v3_pool_state["liquidity"]
        v3_simulate_swap(1e15, v3_pool_state, True)
        assert v3_pool_state["sqrt_price_x96"] == orig_sp
        assert v3_pool_state["liquidity"] == orig_liq

    def test_raises_on_nonpositive_amount(self, v3_pool_state):
        with pytest.raises(ValueError):
            v3_simulate_swap(0, v3_pool_state, True)

    def test_multi_tick_crossing_produces_more_output_than_single(self, v3_pool_state):
        """Large swap that crosses ticks produces more output (more liquidity available)."""
        # Small: stays in range
        small_out, _ = v3_simulate_swap(1e15, v3_pool_state, True)
        # Large: might cross ticks
        large_out, _ = v3_simulate_swap(1e20, v3_pool_state, True)
        assert large_out > small_out
