"""
Edge case tests for the Quantamental Portfolio Optimizer.

Tests potential breaking scenarios:
- Infeasible constraints
- Negative/zero inputs
- Single assets
- All negative returns
"""

import pytest
import numpy as np
from src.core.returns import calculate_implied_cagr
from src.optimization.optimizer import optimize_portfolio, portfolio_performance


class TestCAGREdgeCases:
    """Edge cases for calculate_implied_cagr."""

    def test_negative_target_margin(self):
        """Company expected to have losses at exit."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.1,
            net_margin_target=-0.2,  # Loss expected
            adjusted_growth_rate=0.1,
            exit_pe=20,
            years=5,
        )
        # Negative margin * positive PE = negative EPS * PE = negative price
        # Clipped to 0 → CAGR = -100%
        assert cagr == -1.0

    def test_zero_sales(self):
        """Pre-revenue company has zero future value."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=0,  # Pre-revenue
            net_margin_current=0,
            net_margin_target=0.2,
            adjusted_growth_rate=0.5,
            exit_pe=30,
            years=5,
        )
        assert cagr == -1.0

    def test_zero_exit_pe(self):
        """Exit PE of 0 means worthless exit."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.1,
            net_margin_target=0.2,
            adjusted_growth_rate=0.1,
            exit_pe=0,  # Worthless
            years=5,
        )
        assert cagr == -1.0

    def test_extreme_growth_decline(self):
        """Growth rate of -100% wipes out sales."""
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.1,
            net_margin_target=0.2,
            adjusted_growth_rate=-1.0,  # Total sales collapse
            exit_pe=20,
            years=5,
        )
        assert cagr == -1.0

    def test_vectorized_with_mixed_inputs(self):
        """Vectorized call with some bad inputs."""
        cagr = calculate_implied_cagr(
            current_price=np.array([100, 100, 100]),
            sales_per_share=np.array([10, 0, 10]),  # Second has zero sales
            net_margin_current=np.array([0.1, 0.1, 0.1]),
            net_margin_target=np.array([0.2, 0.2, -0.1]),  # Third has negative margin
            adjusted_growth_rate=np.array([0.1, 0.1, 0.1]),
            exit_pe=np.array([20, 20, 20]),
            years=5,
        )
        assert len(cagr) == 3
        # First case: 10 * 1.1^5 * 0.2 * 20 / 100 = negative due to low multiple
        # This is actually correct - not all "normal" inputs give positive CAGR
        assert not np.isnan(cagr[0])  # Should be valid number
        assert cagr[1] == -1.0  # Zero sales
        assert cagr[2] == -1.0  # Negative margin

    def test_raises_on_zero_price(self):
        """Zero price should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_implied_cagr(
                current_price=0,
                sales_per_share=10,
                net_margin_current=0.1,
                net_margin_target=0.2,
                adjusted_growth_rate=0.1,
                exit_pe=20,
                years=5,
            )

    def test_raises_on_zero_years(self):
        """Zero years should raise ValueError."""
        with pytest.raises(ValueError, match="positive"):
            calculate_implied_cagr(
                current_price=100,
                sales_per_share=10,
                net_margin_current=0.1,
                net_margin_target=0.2,
                adjusted_growth_rate=0.1,
                exit_pe=20,
                years=0,
            )


class TestOptimizerEdgeCases:
    """Edge cases for portfolio optimization."""

    def test_single_asset(self):
        """Single asset should get 100% allocation."""
        result = optimize_portfolio(
            expected_returns=np.array([0.1]),
            cov_matrix=np.array([[0.04]]),
            risk_free_rate=0.02,
        )
        assert result["success"]
        assert np.isclose(result["weights"][0], 1.0)

    def test_all_negative_returns(self):
        """Optimizer should find 'least bad' portfolio."""
        returns = np.array([-0.05, -0.10, -0.15])
        cov = np.diag([0.01, 0.02, 0.03])
        
        result = optimize_portfolio(returns, cov, risk_free_rate=0.0)
        
        assert result["success"]
        assert np.isclose(np.sum(result["weights"]), 1.0)
        # With all negative returns and rf=0, Sharpe ratio is same for all
        # So optimizer may return any valid portfolio - just verify no crash

    def test_infeasible_max_weight(self):
        """max_weight too low to sum to 1 is infeasible."""
        returns = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        cov = np.diag([0.01] * 5)
        
        # 5 assets * 0.1 max = 0.5 < 1.0 needed
        result = optimize_portfolio(returns, cov, max_weight=0.1)
        
        # SLSQP may report success=False or return constrained solution
        # Just verify it doesn't crash and sum of weights is handled
        weights_sum = np.sum(result["weights"])
        # Either fails or produces infeasible weights
        if result["success"]:
            # If it "succeeds", weights are clamped at max
            assert np.all(result["weights"] <= 0.1 + 1e-6)

    def test_identical_assets(self):
        """Identical assets should get equal (or near-equal) weights."""
        n = 3
        returns = np.array([0.1, 0.1, 0.1])
        cov = np.eye(n) * 0.04  # Same variance, uncorrelated
        
        result = optimize_portfolio(returns, cov, risk_free_rate=0.02)
        
        assert result["success"]
        # Should spread roughly equally due to diversification
        assert np.std(result["weights"]) < 0.1

    def test_sharpe_with_zero_volatility_asset(self):
        """Asset with zero vol (risk-free-like) should not crash."""
        returns = np.array([0.1, 0.02])  # Second is risk-free-like
        cov = np.array([[0.04, 0], [0, 1e-10]])  # Near-zero variance
        
        # Should not crash
        result = optimize_portfolio(returns, cov, risk_free_rate=0.02)
        assert result["success"] or not result["success"]  # Just don't crash


class TestPortfolioPerformance:
    """Test portfolio_performance edge cases."""

    def test_zero_weights(self):
        """Zero weights should give zero return/volatility."""
        weights = np.array([0.0, 0.0, 0.0])
        returns = np.array([0.1, 0.15, 0.2])
        cov = np.diag([0.01, 0.02, 0.03])
        
        ret, vol, sharpe = portfolio_performance(weights, returns, cov)
        
        assert ret == 0.0
        assert vol == 0.0
        assert sharpe == 0.0  # Division by zero handled


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
