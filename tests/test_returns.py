"""
Tests for view-based return generation.
"""
import numpy as np
from src.core.returns import calculate_view_returns

def test_calculate_view_returns_scalar():
    """Test with scalar inputs."""
    res = calculate_view_returns(0.08, 0.05)
    assert np.isclose(res, 0.13)

def test_calculate_view_returns_array():
    """Test with array inputs."""
    baseline = 0.08
    alphas = np.array([0.05, -0.02, 0.0])
    expected = np.array([0.13, 0.06, 0.08])
    res = calculate_view_returns(baseline, alphas)
    np.testing.assert_allclose(res, expected)

def test_negative_baseline():
    """Test with negative market baseline (bear market view)."""
    res = calculate_view_returns(-0.05, 0.10)
    assert np.isclose(res, 0.05)