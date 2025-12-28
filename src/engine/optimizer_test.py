"""
Tests for the Mean-Variance Optimizer.

Covering:
1. Mathematical correctness (Tangency portfolio logic).
2. Constraints (Bounds).
3. Edge cases (Zero returns/volatility).
"""

import pytest
import numpy as np
from src.engine.optimizer import find_tangency_portfolio


@pytest.fixture
def sample_data():
    """
    Creates a simple 2-asset universe for testing.

    Asset A: High Return (15%), High Vol (20%) -> Variance 0.04
    Asset B: Low Return (5%), Low Vol (5%)     -> Variance 0.0025
    Correlation: 0.0 (Uncorrelated)
    """
    mu = np.array([0.15, 0.05])
    sigma = np.array([[0.04, 0.00], [0.00, 0.0025]])
    return mu, sigma


def test_find_tangency_portfolio_math(sample_data):
    """
    Verifies the optimizer finds the mathematically correct Tangency Portfolio.
    """
    mu, sigma = sample_data
    rf = 0.02  # 2% Risk Free Rate

    result = find_tangency_portfolio(mu, sigma, risk_free_rate=rf)

    # --- 1. Check Weights Sum to 1.0 (Fully Invested) ---
    assert np.isclose(np.sum(result["weights"]), 1.0)

    # --- 2. Mathematical Logic Check ---
    # Formula for weight ratio with 0 correlation: (Excess Return) / Variance
    # Score A = (0.15 - 0.02) / 0.04   = 3.25
    # Score B = (0.05 - 0.02) / 0.0025 = 12.0
    #
    # Total Score = 15.25
    # Expected Weight A = 3.25 / 15.25 ≈ 21.3%
    # Expected Weight B = 12.0 / 15.25 ≈ 78.7%

    assert np.isclose(result["weights"][1], 0.7868, atol=0.001)

    # --- 3. Check Sharpe Ratio Calculation ---
    exp_ret = (0.2131 * 0.15) + (0.7869 * 0.05)
    exp_var = (0.2131**2 * 0.04) + (0.7869**2 * 0.0025)
    exp_vol = np.sqrt(exp_var)
    exp_sharpe = (exp_ret - rf) / exp_vol

    assert np.isclose(result["sharpe_ratio"], exp_sharpe, atol=0.01)


def test_bounds_list(sample_data):
    """Test the bounds interface."""
    mu, sigma = sample_data

    # Force Asset 0 to be between 40% and 50%
    # Force Asset 1 to be between 50% and 60%
    bounds = [(0.4, 0.5), (0.5, 0.6)]

    result = find_tangency_portfolio(mu, sigma, risk_free_rate=0.02, bounds=bounds)

    w = result["weights"]
    assert 0.4 <= w[0] <= 0.5
    assert 0.5 <= w[1] <= 0.6
    assert np.isclose(np.sum(w), 1.0)


def test_zero_returns():
    """
    Edge Case: When expected returns == risk_free_rate (0.0),
    the Sharpe Ratio is 0.0 for ALL portfolios.
    """
    mu = np.array([0.0, 0.0])
    # Diagonal covariance (independent assets)
    sigma = np.eye(2) * 0.01

    # rf = 0.0 implies Excess Return is 0.0 everywhere.
    result = find_tangency_portfolio(mu, sigma, risk_free_rate=0.0)

    # 1. Check Weights: Should be Equal Weights (0.5, 0.5)
    # Since gradient is zero (flat objective), solver stays at initial guess
    np.testing.assert_allclose(result["weights"], [0.5, 0.5], atol=1e-6)

    # 2. Check Sharpe: Should be exactly 0.0
    assert np.isclose(result["sharpe_ratio"], 0.0)

    # 3. Check Return: 0.0
    assert np.isclose(result["expected_return"], 0.0)
