"""Tests for src/concentration.py — activation rate, entropy, HHI."""

import sys
import pytest
import numpy as np

sys.path.insert(0, "/workspace/project")

from src.concentration import (
    compute_activation_rate,
    compute_entropy,
    compute_hhi,
    compute_all_metrics,
    compute_flow_shares,
)


# ---------------------------------------------------------------------------
# compute_flow_shares
# ---------------------------------------------------------------------------

class TestComputeFlowShares:
    def test_shares_sum_to_one(self):
        q = np.array([10.0, 30.0, 60.0, 0.0])
        x = compute_flow_shares(q)
        assert abs(x.sum() - 1.0) < 1e-12

    def test_shares_nonnegative(self):
        q = np.array([5.0, 0.0, 15.0, 0.0])
        x = compute_flow_shares(q)
        assert np.all(x >= 0)

    def test_single_pool_share_is_one(self):
        q = np.array([0.0, 0.0, 0.0, 100.0])
        x = compute_flow_shares(q)
        assert abs(x[3] - 1.0) < 1e-12
        assert x[:3].sum() < 1e-12

    def test_uniform_shares(self):
        q = np.array([25.0, 25.0, 25.0, 25.0])
        x = compute_flow_shares(q)
        assert np.allclose(x, 0.25)

    def test_zero_input_returns_zeros(self):
        q = np.zeros(4)
        x = compute_flow_shares(q)
        assert np.all(x == 0)


# ---------------------------------------------------------------------------
# compute_activation_rate
# ---------------------------------------------------------------------------

class TestComputeActivationRate:
    def test_single_pool_active(self):
        """One active pool out of four → rate = 0.25."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        rate = compute_activation_rate(x, num_pools=4)
        assert abs(rate - 0.25) < 1e-12

    def test_two_pools_active(self):
        x = np.array([0.5, 0.5, 0.0, 0.0])
        rate = compute_activation_rate(x, num_pools=4)
        assert abs(rate - 0.5) < 1e-12

    def test_all_pools_active(self):
        x = np.array([0.25, 0.25, 0.25, 0.25])
        rate = compute_activation_rate(x, num_pools=4)
        assert abs(rate - 1.0) < 1e-12

    def test_no_pool_active(self):
        x = np.zeros(4)
        rate = compute_activation_rate(x, num_pools=4)
        assert rate == 0.0

    def test_rate_in_valid_range(self):
        """Activation rate must be in [0, 1]."""
        for k in range(5):
            x = np.zeros(4)
            x[:k] = 0.25
            rate = compute_activation_rate(x, num_pools=4)
            assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# compute_entropy
# ---------------------------------------------------------------------------

class TestComputeEntropy:
    def test_single_pool_entropy_is_zero(self):
        """Entropy of a degenerate distribution is 0."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        h = compute_entropy(x)
        assert abs(h) < 1e-12

    def test_uniform_entropy_is_log_n(self):
        """Entropy of uniform over 4 is log(4)."""
        x = np.array([0.25, 0.25, 0.25, 0.25])
        h = compute_entropy(x)
        assert abs(h - np.log(4)) < 1e-10

    def test_entropy_nonnegative(self):
        """Entropy is always non-negative."""
        for _ in range(20):
            q = np.abs(np.random.default_rng(42).normal(0, 1, 4))
            x = compute_flow_shares(q) if q.sum() > 0 else np.zeros(4)
            h = compute_entropy(x)
            assert h >= -1e-12

    def test_zero_times_log_zero_is_zero(self):
        """0 * log(0) should be treated as 0, not NaN."""
        x = np.array([0.5, 0.0, 0.5, 0.0])
        h = compute_entropy(x)
        assert not np.isnan(h)
        assert h == pytest.approx(-2 * 0.5 * np.log(0.5), abs=1e-12)

    def test_two_equal_pools_entropy(self):
        """Entropy of [0.5, 0.5, 0, 0] = log(2)."""
        x = np.array([0.5, 0.5, 0.0, 0.0])
        h = compute_entropy(x)
        assert abs(h - np.log(2)) < 1e-10


# ---------------------------------------------------------------------------
# compute_hhi
# ---------------------------------------------------------------------------

class TestComputeHHI:
    def test_single_pool_hhi_is_one(self):
        """HHI of a pure single-pool route = 1."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        hhi = compute_hhi(x)
        assert abs(hhi - 1.0) < 1e-12

    def test_uniform_hhi_is_one_over_n(self):
        """HHI of uniform = 1/n."""
        x = np.array([0.25, 0.25, 0.25, 0.25])
        hhi = compute_hhi(x)
        assert abs(hhi - 0.25) < 1e-12

    def test_hhi_in_valid_range(self):
        """HHI must be in [1/n, 1] for valid shares."""
        for _ in range(20):
            q = np.abs(np.random.default_rng(0).normal(0, 1, 4)) + 0.01
            x = compute_flow_shares(q)
            hhi = compute_hhi(x)
            assert 0.25 - 1e-10 <= hhi <= 1.0 + 1e-10

    def test_hhi_is_sum_of_squares(self):
        x = np.array([0.6, 0.4, 0.0, 0.0])
        expected = 0.6**2 + 0.4**2
        assert abs(compute_hhi(x) - expected) < 1e-12


# ---------------------------------------------------------------------------
# compute_all_metrics
# ---------------------------------------------------------------------------

class TestComputeAllMetrics:
    def test_keys_present(self):
        x = np.array([1.0, 0.0, 0.0, 0.0])
        m = compute_all_metrics(x)
        assert "activation_rate" in m
        assert "entropy" in m
        assert "hhi" in m

    def test_single_pool_metrics(self):
        """Single-pool route: activation=0.25, entropy=0, hhi=1."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        m = compute_all_metrics(x)
        assert abs(m["activation_rate"] - 0.25) < 1e-12
        assert abs(m["entropy"]) < 1e-12
        assert abs(m["hhi"] - 1.0) < 1e-12

    def test_values_are_floats(self):
        x = np.array([0.5, 0.5, 0.0, 0.0])
        m = compute_all_metrics(x)
        assert isinstance(m["activation_rate"], float)
        assert isinstance(m["entropy"], float)
        assert isinstance(m["hhi"], float)

    def test_paper_target_single_pool_consistency(self):
        """Validate paper claim: single-pool → kappa=0.25, entropy=0, HHI=1."""
        x = np.array([1.0, 0.0, 0.0, 0.0])
        m = compute_all_metrics(x)
        assert m["activation_rate"] == pytest.approx(0.25)
        assert m["entropy"] == pytest.approx(0.0, abs=1e-12)
        assert m["hhi"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Mathematical consistency: entropy <= log(active_pools)
# ---------------------------------------------------------------------------

class TestMathematicalProperties:
    def test_entropy_bounded_by_log_active_pools(self):
        """Shannon entropy is maximized at uniform over active set."""
        rng = np.random.default_rng(123)
        for _ in range(50):
            q = rng.exponential(1.0, 4)
            q[rng.integers(0, 4, size=2)] = 0  # some zeros
            total = q.sum()
            if total == 0:
                continue
            x = q / total
            h = compute_entropy(x)
            k = np.sum(x > 0)
            max_h = np.log(k) if k > 0 else 0
            assert h <= max_h + 1e-10

    def test_hhi_ge_activation_squared_over_num_active(self):
        """By Cauchy-Schwarz, HHI >= 1/k where k = active pools."""
        rng = np.random.default_rng(456)
        for _ in range(50):
            q = rng.exponential(1.0, 4)
            total = q.sum()
            x = q / total
            hhi = compute_hhi(x)
            k = max(1, np.sum(x > 0))
            assert hhi >= 1.0 / k - 1e-10
