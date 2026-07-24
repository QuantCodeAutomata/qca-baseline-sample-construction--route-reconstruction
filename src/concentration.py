"""
Concentration metrics for order-flow share vectors.

All metrics operate on a normalised share vector x ∈ ℝ^N where x_j ≥ 0
and sum(x) ≈ 1.  Use ``compute_flow_shares`` to convert raw flow volumes
to a valid share vector before calling the metric functions.
"""

import math

import numpy as np


def compute_activation_rate(x: np.ndarray, num_pools: int = 4) -> float:
    """Fraction of pools that receive positive order flow.

    κ = #{j : x_j > 0} / num_pools

    Args:
        x:         1-D array of non-negative flow shares.
        num_pools: Total number of pools in the universe (denominator).

    Returns:
        Activation rate κ ∈ [0, 1].
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x > 0)) / num_pools


def compute_entropy(x: np.ndarray) -> float:
    """Shannon entropy (natural log) of the flow-share distribution.

    H = -∑_j x_j · ln(x_j),  with  0 · ln(0) ≡ 0.

    A higher value indicates more dispersed routing; H = 0 when all flow
    is concentrated in a single pool.

    Args:
        x: 1-D array of non-negative flow shares.

    Returns:
        Shannon entropy H ≥ 0 (nats).
    """
    x = np.asarray(x, dtype=float)
    # Mask zero entries to avoid log(0).
    nonzero = x > 0
    h = -np.sum(x[nonzero] * np.log(x[nonzero]))
    return float(h)


def compute_hhi(x: np.ndarray) -> float:
    """Herfindahl–Hirschman Index of the flow-share distribution.

    HHI = ∑_j x_j²

    HHI = 1 when all flow goes to one pool; HHI = 1/N for uniform routing
    across N pools.

    Args:
        x: 1-D array of non-negative flow shares.

    Returns:
        HHI ∈ (0, 1].
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x ** 2))


def compute_all_metrics(x: np.ndarray, num_pools: int = 4) -> dict:
    """Compute all concentration metrics for a flow-share vector.

    Validates that the input is a non-negative vector that sums to
    approximately 1 before computing metrics.

    Args:
        x:         1-D array of non-negative flow shares summing to ~1.
        num_pools: Total number of pools (used for activation rate).

    Returns:
        Dict with keys:
            - ``activation_rate`` (float): fraction of pools with x_j > 0.
            - ``entropy`` (float): Shannon entropy H (nats).
            - ``hhi`` (float): Herfindahl–Hirschman Index.

    Raises:
        ValueError: If any element of x is negative or sum(x) is not ≈ 1.
    """
    x = np.asarray(x, dtype=float)

    if np.any(x < 0):
        raise ValueError("All elements of x must be non-negative.")

    total = float(np.sum(x))
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Share vector must sum to approximately 1, got {total:.6f}."
        )

    return {
        "activation_rate": compute_activation_rate(x, num_pools=num_pools),
        "entropy": compute_entropy(x),
        "hhi": compute_hhi(x),
    }


def compute_flow_shares(q: np.ndarray) -> np.ndarray:
    """Normalise a raw flow-volume vector to a share vector.

    x_j = q_j / ∑_j q_j

    If the total volume is zero the function returns a zero vector of the
    same shape rather than raising a division-by-zero error.

    Args:
        q: 1-D array of non-negative raw flow volumes.

    Returns:
        1-D array of flow shares with the same shape as q.  Sums to 1
        unless the total is zero, in which case a zero vector is returned.
    """
    q = np.asarray(q, dtype=float)
    total = float(np.sum(q))
    if total == 0.0:
        return np.zeros_like(q)
    return q / total
