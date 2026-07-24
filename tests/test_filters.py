"""Tests for src/filters.py — transaction filtering logic."""

import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.filters import (
    is_counterflow,
    filter_counterflow,
    filter_min_usdc,
    apply_all_filters,
)


# ---------------------------------------------------------------------------
# Helper: build a simple DataFrame
# ---------------------------------------------------------------------------

def make_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# is_counterflow
# ---------------------------------------------------------------------------

class TestIsCounterflow:
    def test_single_direction_not_counterflow(self):
        assert is_counterflow(["WETH->USDC", "WETH->USDC"]) is False

    def test_both_directions_is_counterflow(self):
        assert is_counterflow(["WETH->USDC", "USDC->WETH"]) is True

    def test_empty_list(self):
        assert is_counterflow([]) is False

    def test_usdc_to_weth_only(self):
        assert is_counterflow(["USDC->WETH", "USDC->WETH"]) is False

    def test_multiple_same_direction(self):
        assert is_counterflow(["WETH->USDC"] * 5) is False

    def test_mixed_three_directions(self):
        assert is_counterflow(["WETH->USDC", "WETH->USDC", "USDC->WETH"]) is True


# ---------------------------------------------------------------------------
# filter_counterflow
# ---------------------------------------------------------------------------

class TestFilterCounterflow:
    def test_removes_counterflow_tx(self):
        """Transactions with both directions should be entirely removed."""
        df = make_df([
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0},
            {"tx_hash": "A", "direction": "USDC->WETH", "usdc_equiv_size": 200.0},  # counterflow
            {"tx_hash": "B", "direction": "WETH->USDC", "usdc_equiv_size": 100.0},
        ])
        filtered = filter_counterflow(df)
        assert "A" not in filtered["tx_hash"].values
        assert "B" in filtered["tx_hash"].values

    def test_clean_df_unchanged(self):
        """DataFrame with no counterflow should be unchanged."""
        df = make_df([
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0},
            {"tx_hash": "B", "direction": "USDC->WETH", "usdc_equiv_size": 100.0},
        ])
        filtered = filter_counterflow(df)
        assert len(filtered) == len(df)

    def test_all_counterflow_removed(self):
        """All transactions that are counterflow must be removed."""
        df = make_df([
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 100.0},
            {"tx_hash": "A", "direction": "USDC->WETH", "usdc_equiv_size": 100.0},
            {"tx_hash": "B", "direction": "WETH->USDC", "usdc_equiv_size": 100.0},
            {"tx_hash": "B", "direction": "USDC->WETH", "usdc_equiv_size": 100.0},
        ])
        filtered = filter_counterflow(df)
        assert len(filtered) == 0

    def test_multileg_same_direction_kept(self):
        """Multiple legs in same direction should be kept."""
        df = make_df([
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 100.0},
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0},
        ])
        filtered = filter_counterflow(df)
        assert len(filtered) == 2


# ---------------------------------------------------------------------------
# filter_min_usdc
# ---------------------------------------------------------------------------

class TestFilterMinUsdc:
    def test_removes_below_threshold(self):
        df = make_df([
            {"tx_hash": "A", "usdc_equiv_size": 50.0},
            {"tx_hash": "B", "usdc_equiv_size": 200.0},
        ])
        filtered = filter_min_usdc(df, min_usdc=100.0)
        assert "A" not in filtered["tx_hash"].values
        assert "B" in filtered["tx_hash"].values

    def test_exactly_threshold_is_kept(self):
        df = make_df([{"tx_hash": "A", "usdc_equiv_size": 100.0}])
        filtered = filter_min_usdc(df, min_usdc=100.0)
        assert len(filtered) == 1

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"tx_hash": [], "usdc_equiv_size": []})
        filtered = filter_min_usdc(df)
        assert len(filtered) == 0

    def test_default_threshold_is_100(self):
        df = make_df([
            {"tx_hash": "A", "usdc_equiv_size": 99.9},
            {"tx_hash": "B", "usdc_equiv_size": 100.0},
        ])
        filtered = filter_min_usdc(df)
        assert len(filtered) == 1
        assert filtered["tx_hash"].values[0] == "B"


# ---------------------------------------------------------------------------
# apply_all_filters
# ---------------------------------------------------------------------------

class TestApplyAllFilters:
    def test_returns_tuple(self):
        df = make_df([{"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0}])
        result = apply_all_filters(df)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_stats_dict_keys(self):
        df = make_df([{"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0}])
        _, stats = apply_all_filters(df)
        assert "n_before" in stats
        assert "n_after" in stats

    def test_combined_filters_applied(self):
        """Both filters should be applied: counterflow removal AND min USDC."""
        df = make_df([
            {"tx_hash": "A", "direction": "WETH->USDC", "usdc_equiv_size": 200.0},  # kept
            {"tx_hash": "B", "direction": "WETH->USDC", "usdc_equiv_size": 50.0},   # removed (size)
            {"tx_hash": "C", "direction": "WETH->USDC", "usdc_equiv_size": 300.0},  # counterflow
            {"tx_hash": "C", "direction": "USDC->WETH", "usdc_equiv_size": 300.0},  # counterflow
        ])
        filtered, stats = apply_all_filters(df)
        assert "A" in filtered["tx_hash"].values
        assert "B" not in filtered["tx_hash"].values
        assert "C" not in filtered["tx_hash"].values
        assert stats["n_before"] == 4
        assert stats["n_after"] == 1
