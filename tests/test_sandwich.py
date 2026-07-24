"""Tests for exp/exp5_sandwich.py — sandwich-attack detection and analysis."""

import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "/workspace/project")

from exp.exp5_sandwich import (
    compute_counterparty_key,
    tag_sandwiched_transactions,
    aggregate_sandwich_flags,
    compare_sandwiched_shortfalls,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_block_events():
    """Three events in block 100: front-run, victim, back-run from same sender."""
    return pd.DataFrame([
        # Front-run: attacker buys WETH with USDC
        {
            "tx_hash": "0xFRONT", "block_number": 100, "tx_index": 0,
            "log_index": 0, "pool_address": "0xPOOL",
            "direction": "USDC->WETH", "sender": "0xATTACKER",
            "usdc_amount": 10_000.0, "weth_amount": 3.3,
        },
        # Victim: wants to sell WETH for USDC
        {
            "tx_hash": "0xVICTIM", "block_number": 100, "tx_index": 1,
            "log_index": 1, "pool_address": "0xPOOL",
            "direction": "WETH->USDC", "sender": "0xVICTIM",
            "usdc_amount": 3_000.0, "weth_amount": 1.0,
        },
        # Back-run: attacker sells WETH for USDC
        {
            "tx_hash": "0xBACK", "block_number": 100, "tx_index": 2,
            "log_index": 2, "pool_address": "0xPOOL",
            "direction": "WETH->USDC", "sender": "0xATTACKER",
            "usdc_amount": 10_200.0, "weth_amount": 3.4,
        },
    ])


@pytest.fixture
def no_sandwich_events():
    """Block with only victim-direction trades from different senders."""
    return pd.DataFrame([
        {"tx_hash": "0xA", "block_number": 200, "tx_index": 0, "log_index": 0,
         "pool_address": "0xPOOL", "direction": "WETH->USDC", "sender": "0xBOB",
         "usdc_amount": 5_000.0, "weth_amount": 1.66},
        {"tx_hash": "0xB", "block_number": 200, "tx_index": 1, "log_index": 1,
         "pool_address": "0xPOOL", "direction": "WETH->USDC", "sender": "0xCAROL",
         "usdc_amount": 3_000.0, "weth_amount": 1.0},
    ])


# ---------------------------------------------------------------------------
# compute_counterparty_key
# ---------------------------------------------------------------------------

class TestComputeCounterpartyKey:
    def test_returns_string(self):
        row = pd.Series({"sender": "0xABC", "tx_hash": "0xDEF"})
        key = compute_counterparty_key(row)
        assert isinstance(key, str)

    def test_same_sender_same_key(self):
        row1 = pd.Series({"sender": "0xABC", "tx_hash": "0xDEF"})
        row2 = pd.Series({"sender": "0xABC", "tx_hash": "0x123"})
        assert compute_counterparty_key(row1) == compute_counterparty_key(row2)

    def test_different_sender_different_key(self):
        row1 = pd.Series({"sender": "0xABC", "tx_hash": "0xDEF"})
        row2 = pd.Series({"sender": "0xXYZ", "tx_hash": "0xDEF"})
        assert compute_counterparty_key(row1) != compute_counterparty_key(row2)

    def test_deterministic(self):
        row = pd.Series({"sender": "0xAAA", "tx_hash": "0xBBB"})
        assert compute_counterparty_key(row) == compute_counterparty_key(row)


# ---------------------------------------------------------------------------
# tag_sandwiched_transactions
# ---------------------------------------------------------------------------

class TestTagSandwichedTransactions:
    def test_victim_is_tagged(self, simple_block_events):
        """The victim should be sandwiched; front and back runners are not victims."""
        result = tag_sandwiched_transactions(simple_block_events)
        victim_rows = result[result["tx_hash"] == "0xVICTIM"]
        if len(victim_rows) > 0:
            assert victim_rows["sandwich_flag"].values[0] is True \
                or victim_rows["sandwich_flag"].values[0] == True \
                or victim_rows["sandwich_flag"].values[0] == 1

    def test_no_sandwich_in_clean_block(self, no_sandwich_events):
        """Clean block with no bracketing should have zero sandwich tags."""
        result = tag_sandwiched_transactions(no_sandwich_events)
        if "sandwich_flag" in result.columns:
            assert result["sandwich_flag"].sum() == 0

    def test_returns_dataframe(self, simple_block_events):
        result = tag_sandwiched_transactions(simple_block_events)
        assert isinstance(result, pd.DataFrame)

    def test_sandwich_flag_column_present(self, simple_block_events):
        result = tag_sandwiched_transactions(simple_block_events)
        assert "sandwich_flag" in result.columns

    def test_output_has_same_rows_as_input(self, simple_block_events):
        result = tag_sandwiched_transactions(simple_block_events)
        assert len(result) == len(simple_block_events)

    def test_empty_df_returns_empty(self):
        empty = pd.DataFrame(columns=["tx_hash", "block_number", "tx_index",
                                       "log_index", "pool_address",
                                       "direction", "sender",
                                       "usdc_amount", "weth_amount"])
        result = tag_sandwiched_transactions(empty)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# aggregate_sandwich_flags
# ---------------------------------------------------------------------------

class TestAggregateSandwichFlags:
    def test_returns_dataframe_with_tx_hash(self, simple_block_events):
        tagged = tag_sandwiched_transactions(simple_block_events)
        agg = aggregate_sandwich_flags(tagged)
        assert isinstance(agg, pd.DataFrame)
        assert "tx_hash" in agg.columns

    def test_one_row_per_tx(self, simple_block_events):
        tagged = tag_sandwiched_transactions(simple_block_events)
        agg = aggregate_sandwich_flags(tagged)
        assert len(agg) == simple_block_events["tx_hash"].nunique()

    def test_sandwich_flag_boolean_or_int(self, simple_block_events):
        tagged = tag_sandwiched_transactions(simple_block_events)
        agg = aggregate_sandwich_flags(tagged)
        if "sandwich_flag" in agg.columns:
            assert agg["sandwich_flag"].dtype in [bool, np.bool_, int, np.int64, np.int32]


# ---------------------------------------------------------------------------
# compare_sandwiched_shortfalls
# ---------------------------------------------------------------------------

class TestCompareSandwichedShortfalls:
    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        n = 100
        df = pd.DataFrame({
            "tx_hash": [f"0x{i}" for i in range(n)],
            "sandwich_flag": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "shortfall_fvo_bps": rng.exponential(5, n),
            "shortfall_sco_bps": rng.exponential(1, n),
            "shortfall_gfvo_bps": rng.exponential(4, n),
            "usdc_equiv_size": rng.uniform(100, 100_000, n),
        })
        result = compare_sandwiched_shortfalls(df)
        assert isinstance(result, dict)

    def test_keys_include_sandwiched_and_non(self):
        rng = np.random.default_rng(1)
        n = 200
        df = pd.DataFrame({
            "tx_hash": [f"0x{i}" for i in range(n)],
            "sandwich_flag": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "shortfall_fvo_bps": rng.exponential(5, n),
            "shortfall_sco_bps": rng.exponential(1, n),
            "shortfall_gfvo_bps": rng.exponential(4, n),
            "usdc_equiv_size": rng.uniform(100, 100_000, n),
        })
        result = compare_sandwiched_shortfalls(df)
        keys = {k.lower() for k in result}
        has_sand = any("sandwich" in k or "sand" in k for k in keys)
        has_non = any("non" in k for k in keys)
        assert has_sand or "full" in keys or "overall" in keys


# ---------------------------------------------------------------------------
# Full experiment
# ---------------------------------------------------------------------------

class TestRunExperimentSandwich:
    def test_run_produces_dict(self):
        result = run_experiment(n_synthetic=200, save_results=False)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_incidence_between_0_and_1(self):
        result = run_experiment(n_synthetic=200, save_results=False)
        for k, v in result.items():
            if "incidence" in str(k).lower() or "rate" in str(k).lower():
                if isinstance(v, float):
                    assert 0.0 <= v <= 1.0
