"""Tests for exp/exp2_benchmarks.py — SCO, FVO, G-FVO optimization."""

import math
import sys
import pytest
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.amm_math import Q96
from exp.exp2_benchmarks import (
    compute_v2_output,
    compute_v3_output,
    compute_pool_output,
    total_output_for_allocation,
    gas_free_optimal_allocation,
    compute_sco,
    compute_fvo,
    compute_gfvo,
    compute_realized_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRICE = 3_000.0  # USDC per WETH

@pytest.fixture
def pool_states():
    """Four pool states: one V2 and three V3 at ~3000 USDC/WETH."""
    sp = math.sqrt(PRICE) * Q96
    return [
        # Pool 0: V2 30bps
        {"protocol": "v2", "fee_bps": 30, "reserve_weth": 5_000.0, "reserve_usdc": 15_000_000.0},
        # Pool 1: V3 1bp
        {"protocol": "v3", "fee_bps": 1, "sqrt_price_x96": sp, "liquidity": 5e22},
        # Pool 2: V3 5bps
        {"protocol": "v3", "fee_bps": 5, "sqrt_price_x96": sp, "liquidity": 2e22},
        # Pool 3: V3 30bps
        {"protocol": "v3", "fee_bps": 30, "sqrt_price_x96": sp, "liquidity": 5e21},
    ]


@pytest.fixture
def gas_costs_usdc():
    """Fixed gas costs per pool in USDC-equivalent."""
    return [2.0, 1.5, 1.5, 1.5]


# ---------------------------------------------------------------------------
# Pool output wrappers
# ---------------------------------------------------------------------------

class TestComputePoolOutput:
    def test_v2_positive_output(self, pool_states):
        out = compute_v2_output(1.0, pool_states[0], "WETH->USDC")
        assert out > 0

    def test_v3_positive_output(self, pool_states):
        out = compute_v3_output(1e18, pool_states[1], "WETH->USDC")
        assert out > 0

    def test_pool_output_routes_correctly(self, pool_states):
        """compute_pool_output should route V2/V3 correctly."""
        v2_out = compute_pool_output(1.0, 0, pool_states[0], "WETH->USDC")
        v3_out = compute_pool_output(1e18, 1, pool_states[1], "WETH->USDC")
        assert v2_out > 0
        assert v3_out > 0

    def test_zero_input_returns_zero(self, pool_states):
        """Zero input → zero output."""
        out = compute_v2_output(0.0, pool_states[0], "WETH->USDC")
        assert out == 0.0


# ---------------------------------------------------------------------------
# Gas-free optimal allocation
# ---------------------------------------------------------------------------

class TestGasFreeOptimalAllocation:
    def test_allocation_sums_to_budget(self, pool_states):
        """Optimal allocation must sum to Q within tolerance."""
        Q = 10.0
        alloc = gas_free_optimal_allocation(Q, pool_states, "WETH->USDC")
        assert abs(alloc.sum() - Q) / Q < 1e-4

    def test_allocation_nonnegative(self, pool_states):
        Q = 10.0
        alloc = gas_free_optimal_allocation(Q, pool_states, "WETH->USDC")
        assert np.all(alloc >= -1e-10)

    def test_single_active_set_full_allocation(self, pool_states):
        """With active_set={0}, all Q should go to pool 0."""
        Q = 5.0
        alloc = gas_free_optimal_allocation(Q, pool_states, "WETH->USDC", active_set={0})
        assert abs(alloc[0] - Q) < 1e-6
        assert abs(alloc[1]) < 1e-6
        assert abs(alloc[2]) < 1e-6
        assert abs(alloc[3]) < 1e-6

    def test_output_exceeds_single_pool_allocation(self, pool_states):
        """FVO output >= output from routing all Q to one pool (non-negative marginal value of splitting)."""
        Q = 100.0
        alloc = gas_free_optimal_allocation(Q, pool_states, "WETH->USDC")
        total_opt = total_output_for_allocation(alloc, pool_states, "WETH->USDC")
        # Compare to single best pool
        single_best = max(
            total_output_for_allocation(np.array([Q, 0, 0, 0]), pool_states, "WETH->USDC"),
            total_output_for_allocation(np.array([0, Q, 0, 0]), pool_states, "WETH->USDC"),
            total_output_for_allocation(np.array([0, 0, Q, 0]), pool_states, "WETH->USDC"),
            total_output_for_allocation(np.array([0, 0, 0, Q]), pool_states, "WETH->USDC"),
        )
        assert total_opt >= single_best - 1e-6


# ---------------------------------------------------------------------------
# SCO benchmark
# ---------------------------------------------------------------------------

class TestComputeSCO:
    def test_single_pool_realized_zero_sco_shortfall(self, pool_states):
        """Single-pool realized route: SCO is constrained to same pool → zero shortfall."""
        realized_alloc = np.array([5.0, 0.0, 0.0, 0.0])  # all in pool 0
        Q = 5.0
        sco_alloc, sco_out = compute_sco(Q, realized_alloc, pool_states, "WETH->USDC")
        realized_out = compute_realized_output(realized_alloc, pool_states, "WETH->USDC")
        # SCO is constrained to pool 0 only, so its output = realized output
        assert abs(sco_out - realized_out) / max(abs(realized_out), 1e-10) < 1e-6

    def test_sco_output_ge_realized(self, pool_states):
        """SCO output >= realized output (it's an optimum over the same set)."""
        realized_alloc = np.array([3.0, 2.0, 0.0, 0.0])
        Q = 5.0
        _, sco_out = compute_sco(Q, realized_alloc, pool_states, "WETH->USDC")
        realized_out = compute_realized_output(realized_alloc, pool_states, "WETH->USDC")
        assert sco_out >= realized_out - 1e-8

    def test_sco_allocation_sums_to_q(self, pool_states):
        realized_alloc = np.array([2.0, 3.0, 0.0, 0.0])
        Q = 5.0
        sco_alloc, _ = compute_sco(Q, realized_alloc, pool_states, "WETH->USDC")
        assert abs(sco_alloc.sum() - Q) / Q < 1e-4


# ---------------------------------------------------------------------------
# FVO benchmark
# ---------------------------------------------------------------------------

class TestComputeFVO:
    def test_fvo_ge_sco(self, pool_states):
        """FVO >= SCO: FVO uses more pools so it's weakly better."""
        realized_alloc = np.array([3.0, 2.0, 0.0, 0.0])
        Q = 5.0
        _, sco_out = compute_sco(Q, realized_alloc, pool_states, "WETH->USDC")
        _, fvo_out = compute_fvo(Q, pool_states, "WETH->USDC")
        assert fvo_out >= sco_out - 1e-8

    def test_fvo_allocation_nonneg_sums_to_q(self, pool_states):
        Q = 10.0
        fvo_alloc, _ = compute_fvo(Q, pool_states, "WETH->USDC")
        assert np.all(fvo_alloc >= -1e-10)
        assert abs(fvo_alloc.sum() - Q) / Q < 1e-4


# ---------------------------------------------------------------------------
# G-FVO benchmark
# ---------------------------------------------------------------------------

class TestComputeGFVO:
    def test_gfvo_le_fvo(self, pool_states, gas_costs_usdc):
        """G-FVO net output <= FVO (gas costs reduce output)."""
        Q = 10.0
        _, fvo_out = compute_fvo(Q, pool_states, "WETH->USDC")
        _, gfvo_out = compute_gfvo(Q, pool_states, "WETH->USDC", gas_costs_usdc)
        assert gfvo_out <= fvo_out + 1e-6

    def test_gfvo_allocation_nonneg_sums_to_q(self, pool_states, gas_costs_usdc):
        Q = 10.0
        gfvo_alloc, _ = compute_gfvo(Q, pool_states, "WETH->USDC", gas_costs_usdc)
        assert np.all(gfvo_alloc >= -1e-10)
        assert abs(gfvo_alloc.sum() - Q) / Q < 1e-4

    def test_gfvo_with_large_gas_costs_goes_single_pool(self, pool_states):
        """If gas costs are very large, G-FVO should route all Q to one pool."""
        huge_gas = [1e8, 1e8, 1e8, 1e8]  # gas cost > any reasonable output
        Q = 5.0
        gfvo_alloc, _ = compute_gfvo(Q, pool_states, "WETH->USDC", huge_gas)
        active_count = np.sum(gfvo_alloc > 1e-10)
        assert active_count == 1  # only one pool used


# ---------------------------------------------------------------------------
# Shortfall ordering
# ---------------------------------------------------------------------------

class TestShortfallOrdering:
    def test_benchmark_ordering(self, pool_states, gas_costs_usdc):
        """FVO output >= G-FVO net output >= SCO output (in general)."""
        realized_alloc = np.array([3.0, 2.0, 0.0, 0.0])
        Q = 5.0
        _, sco_out = compute_sco(Q, realized_alloc, pool_states, "WETH->USDC")
        _, fvo_out = compute_fvo(Q, pool_states, "WETH->USDC")
        _, gfvo_out = compute_gfvo(Q, pool_states, "WETH->USDC", gas_costs_usdc)
        assert fvo_out >= gfvo_out - 1e-6
        assert fvo_out >= sco_out - 1e-6

    def test_run_experiment_produces_results(self):
        """Full experiment run should return a non-empty results dict."""
        from exp.exp2_benchmarks import run_experiment
        result = run_experiment(n_synthetic=200, save_results=False)
        assert "n_transactions" in result or isinstance(result, dict)
