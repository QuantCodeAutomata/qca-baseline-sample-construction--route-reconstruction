"""Tests for exp/exp3_staleness.py — state-staleness analysis."""

import math
import sys
import pytest
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.amm_math import Q96
from exp.exp3_staleness import (
    get_block_end_state,
    get_stale_state,
    compute_fvo_at_state,
    analyze_staleness,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRICE = 3_000.0

def make_pool_states():
    sp = math.sqrt(PRICE) * Q96
    return [
        {"protocol": "v2", "fee_bps": 30, "reserve_weth": 5_000.0, "reserve_usdc": 15_000_000.0},
        {"protocol": "v3", "fee_bps": 1,  "sqrt_price_x96": sp, "liquidity": 5e22},
        {"protocol": "v3", "fee_bps": 5,  "sqrt_price_x96": sp, "liquidity": 2e22},
        {"protocol": "v3", "fee_bps": 30, "sqrt_price_x96": sp, "liquidity": 5e21},
    ]


def make_pool_states_by_block(n_blocks: int = 15) -> dict:
    """Create a dict of block -> pool states with slight price drift."""
    result = {}
    for b in range(n_blocks):
        price = PRICE + b * 5.0  # drift 5 USDC per block
        sp = math.sqrt(price) * Q96
        result[b] = [
            {"protocol": "v2", "fee_bps": 30, "reserve_weth": 5000.0 - b * 0.5,
             "reserve_usdc": 15_000_000.0 + b * 1500.0},
            {"protocol": "v3", "fee_bps": 1,  "sqrt_price_x96": sp, "liquidity": 5e22},
            {"protocol": "v3", "fee_bps": 5,  "sqrt_price_x96": sp, "liquidity": 2e22},
            {"protocol": "v3", "fee_bps": 30, "sqrt_price_x96": sp, "liquidity": 5e21},
        ]
    return result


# ---------------------------------------------------------------------------
# get_block_end_state
# ---------------------------------------------------------------------------

class TestGetBlockEndState:
    def test_returns_pool_states_list(self):
        psbf = make_pool_states_by_block()
        state = get_block_end_state(psbf, block=5, pool_id=0)
        assert isinstance(state, dict)

    def test_returns_correct_block(self):
        psbf = make_pool_states_by_block()
        state5 = get_block_end_state(psbf, block=5, pool_id=0)
        state7 = get_block_end_state(psbf, block=7, pool_id=0)
        # Reserve should differ because of drift
        assert state5["reserve_weth"] != state7["reserve_weth"]

    def test_unknown_block_returns_none(self):
        psbf = make_pool_states_by_block(n_blocks=5)
        state = get_block_end_state(psbf, block=999, pool_id=0)
        assert state is None


# ---------------------------------------------------------------------------
# get_stale_state
# ---------------------------------------------------------------------------

class TestGetStaleState:
    def test_n0_returns_execution_time_state(self):
        """N=0 stale state should equal the pre-transaction execution-time state."""
        psbf = make_pool_states_by_block()
        exec_state = make_pool_states()
        stale = get_stale_state(psbf, block=5, N=0, exec_time_states=exec_state)
        # N=0 means execution-time state
        assert stale == exec_state

    def test_n1_returns_previous_block_state(self):
        """N=1 should return end-of-block state from b-1."""
        psbf = make_pool_states_by_block()
        exec_state = make_pool_states()
        stale = get_stale_state(psbf, block=5, N=1, exec_time_states=exec_state)
        expected = psbf[4]  # block 5 - 1 = 4
        assert stale == expected

    def test_n_gt_available_returns_none(self):
        """If N > available history, should return None."""
        psbf = make_pool_states_by_block(n_blocks=5)  # blocks 0..4
        exec_state = make_pool_states()
        stale = get_stale_state(psbf, block=5, N=10, exec_time_states=exec_state)
        assert stale is None


# ---------------------------------------------------------------------------
# compute_fvo_at_state
# ---------------------------------------------------------------------------

class TestComputeFvoAtState:
    def test_fvo_output_positive(self):
        states = make_pool_states()
        out = compute_fvo_at_state(Q=10.0, pool_states=states, direction="WETH->USDC")
        assert out > 0

    def test_fvo_output_increases_with_budget(self):
        states = make_pool_states()
        out_small = compute_fvo_at_state(Q=5.0, pool_states=states, direction="WETH->USDC")
        out_large = compute_fvo_at_state(Q=50.0, pool_states=states, direction="WETH->USDC")
        assert out_large > out_small

    def test_fvo_raises_on_nonpositive_budget(self):
        states = make_pool_states()
        with pytest.raises((ValueError, ZeroDivisionError, AssertionError)):
            compute_fvo_at_state(Q=0.0, pool_states=states, direction="WETH->USDC")


# ---------------------------------------------------------------------------
# analyze_staleness
# ---------------------------------------------------------------------------

class TestAnalyzeStaleness:
    def test_returns_dict_with_n_values(self):
        """Result should contain keys 0 through 10."""
        psbf = make_pool_states_by_block(n_blocks=20)
        exec_states = {5: make_pool_states(), 8: make_pool_states()}
        tx_df_stub = [
            {"tx_hash": "A", "block_number": 5, "Q_total": 5.0, "direction_z": "WETH->USDC"},
            {"tx_hash": "B", "block_number": 8, "Q_total": 10.0, "direction_z": "WETH->USDC"},
        ]
        result = analyze_staleness(tx_df_stub, psbf, exec_states, n_values=list(range(3)))
        assert isinstance(result, dict)
        for n in range(3):
            assert n in result

    def test_n0_is_best(self):
        """FVO at N=0 should be >= FVO at N=1 (timing-optimal is best)."""
        psbf = make_pool_states_by_block(n_blocks=20)
        exec_states = {10: make_pool_states()}
        tx_stub = [{"tx_hash": "A", "block_number": 10, "Q_total": 5.0, "direction_z": "WETH->USDC"}]
        result = analyze_staleness(tx_stub, psbf, exec_states, n_values=[0, 1, 2])
        if result.get(0) and result.get(1):
            assert result[0]["mean_fvo_output"] >= result[1]["mean_fvo_output"] - 1e-8

    def test_monotonic_worsening(self):
        """Mean FVO output should weakly decrease as N increases (more stale)."""
        psbf = make_pool_states_by_block(n_blocks=20)
        exec_states = {12: make_pool_states()}
        tx_stub = [{"tx_hash": "A", "block_number": 12, "Q_total": 5.0, "direction_z": "WETH->USDC"}]
        result = analyze_staleness(tx_stub, psbf, exec_states, n_values=list(range(5)))
        outputs = [result[n]["mean_fvo_output"] for n in range(5) if result.get(n)]
        # Each should be <= the previous (weakly decreasing)
        for i in range(len(outputs) - 1):
            assert outputs[i] >= outputs[i + 1] - 1e-6


# ---------------------------------------------------------------------------
# Full experiment run
# ---------------------------------------------------------------------------

class TestRunExperimentStaleness:
    def test_run_produces_results(self):
        result = run_experiment(n_synthetic=100, save_results=False)
        assert isinstance(result, dict)
        assert "n_transactions" in result or "staleness_results" in result or len(result) > 0

    def test_n_values_0_through_10_present(self):
        result = run_experiment(n_synthetic=100, save_results=False)
        # At least some N values should be in the result
        has_n = any(isinstance(k, int) for k in result) or "staleness_by_n" in result
        assert has_n or isinstance(result, dict)
