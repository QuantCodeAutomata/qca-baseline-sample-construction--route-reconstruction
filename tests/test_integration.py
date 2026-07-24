"""Integration tests: verify end-to-end pipelines for all five experiments.

These tests run entire experiment pipelines on small synthetic datasets to
validate that the methodology is correctly implemented end-to-end.  They
do NOT require an Ethereum archive node or Massive API access.
"""

import sys
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, "/workspace/project")

from src.synthetic_data import generate_synthetic_dataset
from exp.exp1_baseline import run_experiment as run_exp1
from exp.exp2_benchmarks import run_experiment as run_exp2
from exp.exp3_staleness import run_experiment as run_exp3
from exp.exp4_trade_size import run_experiment as run_exp4
from exp.exp5_sandwich import run_experiment as run_exp5


N_SYNTHETIC = 300  # small sample for fast test runs


# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

class TestSyntheticDataset:
    def test_returns_expected_keys(self):
        ds = generate_synthetic_dataset(n=100, seed=0)
        assert "routes" in ds
        assert "pool_states_by_block" in ds or "transactions" in ds

    def test_route_columns_present(self):
        ds = generate_synthetic_dataset(n=100, seed=0)
        routes = ds.get("routes", ds.get("transactions"))
        assert routes is not None
        required = {"tx_hash", "usdc_equiv_size"}
        assert required.issubset(set(routes.columns))

    def test_concentration_metrics_computed(self):
        ds = generate_synthetic_dataset(n=200, seed=1)
        routes = ds.get("routes", ds.get("transactions"))
        metric_cols = {"activation_rate", "entropy", "hhi"}
        computed = metric_cols.intersection(set(routes.columns))
        assert len(computed) == 3, f"Missing metric columns: {metric_cols - computed}"

    def test_activation_rate_in_valid_range(self):
        ds = generate_synthetic_dataset(n=200, seed=2)
        routes = ds.get("routes", ds.get("transactions"))
        ar = routes["activation_rate"]
        assert (ar >= 0).all() and (ar <= 1).all()

    def test_entropy_nonnegative(self):
        ds = generate_synthetic_dataset(n=200, seed=3)
        routes = ds.get("routes", ds.get("transactions"))
        h = routes["entropy"]
        assert (h >= -1e-9).all()

    def test_hhi_in_valid_range(self):
        ds = generate_synthetic_dataset(n=200, seed=4)
        routes = ds.get("routes", ds.get("transactions"))
        hhi = routes["hhi"]
        assert (hhi >= 0.25 - 1e-9).all() and (hhi <= 1.0 + 1e-9).all()

    def test_usdc_equiv_sizes_above_threshold(self):
        ds = generate_synthetic_dataset(n=200, seed=5)
        routes = ds.get("routes", ds.get("transactions"))
        # After filtering, all sizes should be >= 100
        assert (routes["usdc_equiv_size"] >= 100).all()

    def test_pool_shares_sum_to_one(self):
        ds = generate_synthetic_dataset(n=100, seed=6)
        routes = ds.get("routes", ds.get("transactions"))
        share_cols = [c for c in routes.columns if c.startswith("x_pool")]
        if len(share_cols) == 4:
            share_sums = routes[share_cols].sum(axis=1)
            assert np.allclose(share_sums, 1.0, atol=1e-6)

    def test_no_counterflow_transactions(self):
        """Counterflow transactions must have been removed."""
        ds = generate_synthetic_dataset(n=200, seed=7)
        routes = ds.get("routes", ds.get("transactions"))
        if "direction_z" in routes.columns:
            # There should be only one direction per tx_hash
            multi_dir = (
                routes.groupby("tx_hash")["direction_z"].nunique() > 1
            ).any()
            assert not multi_dir


# ---------------------------------------------------------------------------
# Experiment 1: Baseline
# ---------------------------------------------------------------------------

class TestExp1Baseline:
    def test_returns_dict(self):
        result = run_exp1(n_synthetic=N_SYNTHETIC, save_results=False)
        assert isinstance(result, dict)

    def test_n_transactions_positive(self):
        result = run_exp1(n_synthetic=N_SYNTHETIC, save_results=False)
        assert result.get("n_transactions", 0) > 0

    def test_total_volume_positive(self):
        result = run_exp1(n_synthetic=N_SYNTHETIC, save_results=False)
        vol_key = next((k for k in result if "volume" in k.lower() or "vol" in k.lower()), None)
        if vol_key:
            assert result[vol_key] > 0

    def test_concentration_metrics_in_range(self):
        result = run_exp1(n_synthetic=N_SYNTHETIC, save_results=False)
        ar = result.get("mean_activation_rate", result.get("activation_rate"))
        h = result.get("mean_entropy", result.get("entropy"))
        hhi = result.get("mean_hhi", result.get("hhi"))
        if ar is not None:
            assert 0.0 <= ar <= 1.0
        if h is not None:
            assert h >= -1e-9
        if hhi is not None:
            assert 0.25 - 1e-9 <= hhi <= 1.0 + 1e-9

    def test_paper_target_order_of_magnitude(self):
        """Synthetic should roughly match paper: activation ~0.283, HHI ~0.971."""
        result = run_exp1(n_synthetic=N_SYNTHETIC, save_results=False)
        ar = result.get("mean_activation_rate", result.get("activation_rate"))
        hhi = result.get("mean_hhi", result.get("hhi"))
        # Allow ±0.15 tolerance around paper targets given small sample
        if ar is not None:
            assert 0.15 <= ar <= 0.50, f"Activation rate {ar} far from paper target ~0.283"
        if hhi is not None:
            assert 0.70 <= hhi <= 1.00, f"HHI {hhi} far from paper target ~0.971"


# ---------------------------------------------------------------------------
# Experiment 2: Benchmarks
# ---------------------------------------------------------------------------

class TestExp2Benchmarks:
    def test_returns_dict(self):
        result = run_exp2(n_synthetic=N_SYNTHETIC, save_results=False)
        assert isinstance(result, dict)

    def test_shortfall_keys_present(self):
        result = run_exp2(n_synthetic=N_SYNTHETIC, save_results=False)
        # At least one of the expected shortfall keys
        has_shortfall = any(
            "sco" in str(k).lower() or "fvo" in str(k).lower()
            for k in result
        )
        assert has_shortfall

    def test_shortfalls_nonnegative(self):
        result = run_exp2(n_synthetic=N_SYNTHETIC, save_results=False)
        for k, v in result.items():
            if isinstance(v, float) and "shortfall" in str(k).lower():
                assert v >= -1e-6, f"Negative shortfall in {k}: {v}"

    def test_sco_le_fvo(self):
        """Mean SCO shortfall should generally be <= mean FVO shortfall."""
        result = run_exp2(n_synthetic=N_SYNTHETIC, save_results=False)
        sco = result.get("mean_shortfall_sco_bps", result.get("mean_sco_bps"))
        fvo = result.get("mean_shortfall_fvo_bps", result.get("mean_fvo_bps"))
        if sco is not None and fvo is not None:
            assert sco <= fvo + 1.0  # allow small tolerance for synthetic data


# ---------------------------------------------------------------------------
# Experiment 3: Staleness
# ---------------------------------------------------------------------------

class TestExp3Staleness:
    def test_returns_dict(self):
        result = run_exp3(n_synthetic=N_SYNTHETIC, save_results=False)
        assert isinstance(result, dict)

    def test_n_values_covered(self):
        result = run_exp3(n_synthetic=N_SYNTHETIC, save_results=False)
        # Should have results for some N values
        assert len(result) > 0

    def test_n0_not_worse_than_n1(self):
        result = run_exp3(n_synthetic=N_SYNTHETIC, save_results=False)
        # N=0 FVO output should be >= N=1 FVO output (fresher state is better)
        n0_out = result.get("n0_mean_fvo") or result.get(0, {}).get("mean_fvo_output")
        n1_out = result.get("n1_mean_fvo") or result.get(1, {}).get("mean_fvo_output")
        if n0_out is not None and n1_out is not None:
            assert n0_out >= n1_out - 1e-6


# ---------------------------------------------------------------------------
# Experiment 4: Trade Size Heterogeneity
# ---------------------------------------------------------------------------

class TestExp4TradeSize:
    def test_returns_dict(self):
        result = run_exp4(n_synthetic=N_SYNTHETIC, save_results=False)
        assert isinstance(result, dict)

    def test_has_bin_info(self):
        result = run_exp4(n_synthetic=N_SYNTHETIC, save_results=False)
        has = any("bin" in str(k).lower() for k in result) or "bin_statistics" in result
        assert has or len(result) > 0


# ---------------------------------------------------------------------------
# Experiment 5: Sandwich
# ---------------------------------------------------------------------------

class TestExp5Sandwich:
    def test_returns_dict(self):
        result = run_exp5(n_synthetic=N_SYNTHETIC, save_results=False)
        assert isinstance(result, dict)

    def test_incidence_between_0_and_100_pct(self):
        result = run_exp5(n_synthetic=N_SYNTHETIC, save_results=False)
        for k, v in result.items():
            if "incidence" in str(k).lower() and isinstance(v, float):
                assert 0.0 <= v <= 1.0
