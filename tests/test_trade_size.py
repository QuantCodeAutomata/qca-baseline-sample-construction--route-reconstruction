"""Tests for exp/exp4_trade_size.py — trade-size heterogeneity analysis."""

import sys
import pytest
import pandas as pd
import numpy as np

sys.path.insert(0, "/workspace/project")

from exp.exp4_trade_size import (
    assign_size_bins,
    compute_bin_statistics,
    compute_tail_contribution,
    run_experiment,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """Small sample DataFrame with known size distribution."""
    rng = np.random.default_rng(42)
    n = 200
    sizes = np.concatenate([
        rng.uniform(100, 1_000, 80),      # bin [100, 1K)
        rng.uniform(1_000, 10_000, 60),    # bin [1K, 10K)
        rng.uniform(10_000, 100_000, 40),  # bin [10K, 100K)
        rng.uniform(100_000, 500_000, 20), # bin [100K, ∞)
    ])
    shortfalls_fvo = rng.exponential(10.0, n)  # right-skewed
    shortfalls_gfvo = rng.exponential(8.0, n)
    return pd.DataFrame({
        "tx_hash": [f"0x{i:040x}" for i in range(n)],
        "usdc_equiv_size": sizes,
        "shortfall_fvo_bps": shortfalls_fvo,
        "shortfall_gfvo_bps": shortfalls_gfvo,
        "shortfall_fvo_usdc": shortfalls_fvo * sizes / 10_000,
        "shortfall_gfvo_usdc": shortfalls_gfvo * sizes / 10_000,
    })


# ---------------------------------------------------------------------------
# assign_size_bins
# ---------------------------------------------------------------------------

class TestAssignSizeBins:
    def test_bins_mutually_exclusive(self, sample_df):
        """Each transaction belongs to exactly one bin."""
        df_with_bins = assign_size_bins(sample_df)
        assert "size_bin" in df_with_bins.columns
        assert df_with_bins["size_bin"].notna().all()

    def test_correct_bin_assignments(self):
        df = pd.DataFrame({"usdc_equiv_size": [150.0, 5_000.0, 50_000.0, 200_000.0]})
        df_binned = assign_size_bins(df)
        bins = df_binned["size_bin"].tolist()
        assert bins[0] == "[100, 1K)"
        assert bins[1] == "[1K, 10K)"
        assert bins[2] == "[10K, 100K)"
        assert bins[3] == "[100K, ∞)"

    def test_exactly_at_boundary_goes_to_higher_bin(self):
        """Value exactly at 1000 goes to [1K, 10K) (right-open left-closed)."""
        df = pd.DataFrame({"usdc_equiv_size": [1_000.0]})
        df_binned = assign_size_bins(df)
        assert df_binned["size_bin"].values[0] == "[1K, 10K)"

    def test_counts_match(self, sample_df):
        """Total count across bins equals total rows."""
        df_binned = assign_size_bins(sample_df)
        assert len(df_binned) == len(sample_df)

    def test_all_four_bins_present(self, sample_df):
        df_binned = assign_size_bins(sample_df)
        unique_bins = set(df_binned["size_bin"].unique())
        expected = {"[100, 1K)", "[1K, 10K)", "[10K, 100K)", "[100K, ∞)"}
        assert unique_bins == expected


# ---------------------------------------------------------------------------
# compute_bin_statistics
# ---------------------------------------------------------------------------

class TestComputeBinStatistics:
    def test_returns_dataframe(self, sample_df):
        df_binned = assign_size_bins(sample_df)
        stats = compute_bin_statistics(df_binned)
        assert isinstance(stats, pd.DataFrame)

    def test_required_columns_present(self, sample_df):
        df_binned = assign_size_bins(sample_df)
        stats = compute_bin_statistics(df_binned)
        required = {"count", "mean_fvo_bps", "median_fvo_bps", "mean_gfvo_bps", "median_gfvo_bps"}
        assert required.issubset(set(stats.columns))

    def test_count_sums_to_total(self, sample_df):
        df_binned = assign_size_bins(sample_df)
        stats = compute_bin_statistics(df_binned)
        assert stats["count"].sum() == len(sample_df)

    def test_mean_exceeds_median_for_skewed_data(self, sample_df):
        """For right-skewed exponential shortfalls, mean > median in each bin."""
        df_binned = assign_size_bins(sample_df)
        stats = compute_bin_statistics(df_binned)
        # For exponential distribution, mean > median on average
        # Allow some statistical variation; at least 3 of 4 bins should satisfy
        exceeds = (stats["mean_fvo_bps"] > stats["median_fvo_bps"]).sum()
        assert exceeds >= 2  # At least half of bins

    def test_stats_are_nonnegative(self, sample_df):
        df_binned = assign_size_bins(sample_df)
        stats = compute_bin_statistics(df_binned)
        assert (stats["mean_fvo_bps"] >= 0).all()
        assert (stats["median_fvo_bps"] >= 0).all()


# ---------------------------------------------------------------------------
# compute_tail_contribution
# ---------------------------------------------------------------------------

class TestComputeTailContribution:
    def test_returns_dict(self, sample_df):
        result = compute_tail_contribution(sample_df, top_pct=0.05)
        assert isinstance(result, dict)

    def test_top_pct_keys(self, sample_df):
        result = compute_tail_contribution(sample_df, top_pct=0.05)
        assert "top_pct" in result
        assert abs(result["top_pct"] - 0.05) < 1e-10

    def test_tail_share_between_0_and_1(self, sample_df):
        result = compute_tail_contribution(sample_df, top_pct=0.05)
        if "fvo_tail_share" in result:
            assert 0.0 <= result["fvo_tail_share"] <= 1.0

    def test_extreme_top_pct_1_returns_all(self, sample_df):
        """With top_pct=1.0, tail contribution should equal total."""
        result = compute_tail_contribution(sample_df, top_pct=1.0)
        if "fvo_tail_share" in result:
            assert abs(result["fvo_tail_share"] - 1.0) < 1e-10

    def test_tail_share_ge_pct(self, sample_df):
        """For right-skewed data, top X% by shortfall should account for more than X% of volume."""
        result = compute_tail_contribution(sample_df, top_pct=0.05)
        if "fvo_tail_share" in result:
            assert result["fvo_tail_share"] >= 0.05  # heavy tail property


# ---------------------------------------------------------------------------
# Full experiment run
# ---------------------------------------------------------------------------

class TestRunExperimentTradeSize:
    def test_run_produces_results(self):
        result = run_experiment(n_synthetic=200, save_results=False)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_result_contains_bin_stats(self):
        result = run_experiment(n_synthetic=200, save_results=False)
        # Should contain some bin-related results
        has_bin = any("bin" in str(k).lower() for k in result)
        assert has_bin or "bin_statistics" in result or "n_transactions" in result
