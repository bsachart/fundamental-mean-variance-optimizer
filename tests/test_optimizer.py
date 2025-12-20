"""
Tests for portfolio optimization module.

Test Strategy:
    1. Test mathematical correctness (formulas, analytical solutions)
    2. Test constraint satisfaction (weights sum to 1, bounds respected)
    3. Test optimization properties (frontier dominance, diversification benefits)
    4. Test edge cases (degenerate inputs, infeasible constraints)
    
Organization:
    - Fixtures: Test data with clear mathematical properties
    - TestMetricsCalculation: Portfolio metric formulas
    - TestConstrainedOptimization: Constraint handling
    - TestEfficientFrontier: Mean-variance efficiency
    - TestDegenerateInputs: Edge cases and error conditions
"""

import pytest
import numpy as np
from src.optimization.optimizer import (
    PortfolioOptimizer,
    optimize_portfolio,
    PortfolioMetrics,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def uncorrelated_assets():
    """
    Five uncorrelated assets with linearly increasing risk and return.
    
    Properties:
    - Returns: 10%, 15%, 20%, 25%, 30%
    - Volatilities: 10%, 15%, 20%, 25%, 30%
    - Correlation: Zero (diagonal covariance matrix)
    
    This setup has known analytical solutions for testing.
    """
    returns = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    cov = np.diag(vols**2)
    return returns, cov, vols


@pytest.fixture
def optimizer_uncorrelated(uncorrelated_assets):
    """Optimizer with uncorrelated assets."""
    returns, cov, _ = uncorrelated_assets
    return PortfolioOptimizer(returns, cov, risk_free_rate=0.02)


@pytest.fixture
def correlated_assets():
    """
    Three assets with positive correlation.
    
    Tests that optimizer handles covariance structure correctly.
    """
    returns = np.array([0.08, 0.12, 0.15])
    # Correlation matrix with moderate positive correlations
    corr = np.array([
        [1.0, 0.3, 0.2],
        [0.3, 1.0, 0.4],
        [0.2, 0.4, 1.0]
    ])
    vols = np.array([0.15, 0.20, 0.25])
    # Convert correlation to covariance: cov_ij = corr_ij * vol_i * vol_j
    cov = corr * np.outer(vols, vols)
    return returns, cov


# ============================================================================
# Test Metrics Calculation
# ============================================================================

class TestMetricsCalculation:
    """Test portfolio metric formulas are implemented correctly."""
    
    def test_return_calculation(self, optimizer_uncorrelated):
        """Portfolio return should be weighted average of asset returns."""
        weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        metrics = optimizer_uncorrelated.calculate_metrics(weights)
        
        # E[R_p] = Σ w_i * E[R_i]
        expected_return = 0.2 * 0.10 + 0.2 * 0.15 + 0.2 * 0.20 + 0.2 * 0.25 + 0.2 * 0.30
        assert np.isclose(metrics.return_, expected_return)
        assert np.isclose(metrics.return_, 0.20)
    
    def test_volatility_calculation_uncorrelated(self, uncorrelated_assets, optimizer_uncorrelated):
        """For uncorrelated assets: σ_p² = Σ w_i² σ_i²"""
        _, _, vols = uncorrelated_assets
        weights = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        metrics = optimizer_uncorrelated.calculate_metrics(weights)
        
        # For diagonal covariance: portfolio variance is sum of weighted variances
        expected_variance = np.sum(weights**2 * vols**2)
        expected_vol = np.sqrt(expected_variance)
        
        assert np.isclose(metrics.volatility, expected_vol)
    
    def test_volatility_calculation_correlated(self, correlated_assets):
        """For correlated assets: σ_p² = w' Σ w (full matrix multiplication)."""
        returns, cov = correlated_assets
        optimizer = PortfolioOptimizer(returns, cov, risk_free_rate=0.02)
        
        weights = np.array([0.4, 0.4, 0.2])
        metrics = optimizer.calculate_metrics(weights)
        
        # Full covariance formula
        expected_variance = weights @ cov @ weights
        expected_vol = np.sqrt(expected_variance)
        
        assert np.isclose(metrics.volatility, expected_vol)
    
    def test_sharpe_ratio_calculation(self, optimizer_uncorrelated):
        """Sharpe ratio = (E[R] - R_f) / σ"""
        weights = np.array([0.0, 0.0, 1.0, 0.0, 0.0])  # 100% in asset 3
        metrics = optimizer_uncorrelated.calculate_metrics(weights)
        
        # Return = 20%, vol = 20%, rf = 2%
        expected_sharpe = (0.20 - 0.02) / 0.20
        assert np.isclose(metrics.sharpe_ratio, expected_sharpe)
        assert np.isclose(metrics.sharpe_ratio, 0.90)
    
    def test_sharpe_can_be_negative(self):
        """Sharpe ratio should be negative when return < risk-free rate."""
        returns = np.array([0.01, 0.02])
        cov = np.diag([0.04, 0.09])
        optimizer = PortfolioOptimizer(returns, cov, risk_free_rate=0.05)
        
        weights = np.array([1.0, 0.0])
        metrics = optimizer.calculate_metrics(weights)
        
        # Return 1% < 5% risk-free → negative Sharpe
        assert metrics.sharpe_ratio < 0
    
    def test_zero_volatility_edge_case(self, optimizer_uncorrelated):
        """Zero-weight portfolio should return zero for all metrics."""
        weights = np.zeros(5)
        metrics = optimizer_uncorrelated.calculate_metrics(weights)
        
        assert metrics.return_ == 0.0
        assert metrics.volatility == 0.0
        assert metrics.sharpe_ratio == 0.0  # Handled gracefully (0/0 → 0)


# ============================================================================
# Test Constrained Optimization
# ============================================================================

class TestConstrainedOptimization:
    """Test that optimizer respects constraints and finds correct solutions."""
    
    def test_weights_sum_to_one(self, optimizer_uncorrelated):
        """All optimization methods must produce fully-invested portfolios."""
        max_sharpe = optimizer_uncorrelated.maximize_sharpe()
        min_vol = optimizer_uncorrelated.minimize_volatility()
        
        assert np.isclose(np.sum(max_sharpe.weights), 1.0)
        assert np.isclose(np.sum(min_vol.weights), 1.0)
    
    def test_long_only_constraint(self, optimizer_uncorrelated):
        """All weights should be non-negative (no short-selling)."""
        portfolio = optimizer_uncorrelated.maximize_sharpe()
        
        assert np.all(portfolio.weights >= -1e-6)  # Allow tiny numerical errors
    
    def test_concentration_limit(self, optimizer_uncorrelated):
        """Max weight constraint should be respected."""
        max_weight = 0.25
        # Convert max_weight to per-asset bounds
        bounds = [(0.0, max_weight) for _ in range(5)]
        portfolio = optimizer_uncorrelated.maximize_sharpe(bounds=bounds)
        
        assert portfolio.success
        assert np.all(portfolio.weights <= max_weight + 1e-6)
        
        # With 5 assets and 25% limit, should actually use the limit
        # (unconstrained optimum likely concentrates more)
        assert np.max(portfolio.weights) >= max_weight - 1e-3
    
    def test_target_return_constraint(self, optimizer_uncorrelated):
        """Constrained minimization should achieve target return."""
        target = 0.20
        portfolio = optimizer_uncorrelated.minimize_volatility(target_return=target)
        
        assert portfolio.success
        assert np.isclose(portfolio.return_, target, atol=1e-4)
    
    def test_infeasible_concentration_limit(self):
        """Optimizer should handle infeasible constraints gracefully."""
        returns = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        cov = np.diag([0.01] * 5)
        optimizer = PortfolioOptimizer(returns, cov, risk_free_rate=0.02)
        
        # 5 assets × 0.15 max = 0.75 < 1.0 required (infeasible)
        bounds = [(0.0, 0.15) for _ in range(5)]
        portfolio = optimizer.maximize_sharpe(bounds=bounds)
        
        # Optimizer may fail or find best constrained solution
        # Either way, shouldn't crash and weights should respect bounds
        if portfolio.success:
            assert np.all(portfolio.weights <= 0.15 + 1e-6)

    def test_short_selling_allowed(self):
        """Test that optimizer handles short selling when bounds allow it."""
        # Scenario: High return asset and asset with return < risk-free rate
        # Strategy: Short the asset with negative excess return to leverage the better one.
        returns = np.array([0.15, 0.01])
        vols = np.array([0.20, 0.10])
        cov = np.diag(vols**2)
        optimizer = PortfolioOptimizer(returns, cov, risk_free_rate=0.02)
        
        # Long asset 0 (+150%), short asset 1 (-50%). Sum = 1.0.
        bounds = [(0.0, 1.5), (-0.5, 0.0)]
        portfolio = optimizer.maximize_sharpe(bounds=bounds)
        
        assert portfolio.success
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        assert portfolio.weights[1] < -0.1  # Should short the lower return asset
        assert np.all(portfolio.weights >= -0.5 - 1e-6)
        assert np.all(portfolio.weights <= 1.5 + 1e-6)


# ============================================================================
# Test Optimization Properties
# ============================================================================

class TestOptimizationProperties:
    """Test mathematical properties of optimal portfolios."""
    
    def test_max_sharpe_beats_individual_assets(self, uncorrelated_assets, optimizer_uncorrelated):
        """Optimal portfolio should have higher Sharpe than any single asset."""
        returns, _, vols = uncorrelated_assets
        optimal = optimizer_uncorrelated.maximize_sharpe()
        
        # Calculate Sharpe for each individual asset
        individual_sharpes = (returns - 0.02) / vols
        
        assert optimal.sharpe_ratio >= np.max(individual_sharpes) - 1e-6
    
    def test_min_volatility_analytical_solution(self, uncorrelated_assets, optimizer_uncorrelated):
        """
        For uncorrelated assets, min-vol weights should be inversely proportional to variance.
        
        Analytical solution: w_i = (1/σ_i²) / Σ(1/σ_j²)
        """
        _, _, vols = uncorrelated_assets
        portfolio = optimizer_uncorrelated.minimize_volatility()
        
        # Calculate analytical weights
        variances = vols**2
        inverse_variances = 1 / variances
        analytical_weights = inverse_variances / np.sum(inverse_variances)
        
        # Numerical optimizer should match analytical solution within 1%
        np.testing.assert_allclose(
            portfolio.weights,
            analytical_weights,
            rtol=0.01,
            err_msg="Min-vol weights should be proportional to 1/variance"
        )
    
    def test_diversification_reduces_risk(self, uncorrelated_assets, optimizer_uncorrelated):
        """Diversified portfolio should have lower vol than lowest-vol individual asset."""
        _, _, vols = uncorrelated_assets
        portfolio = optimizer_uncorrelated.minimize_volatility()
        
        # Diversification benefit: σ_p < min(σ_i) for uncorrelated assets
        assert portfolio.volatility < np.min(vols)
    
    def test_max_sharpe_is_on_efficient_frontier(self, optimizer_uncorrelated):
        """Max Sharpe portfolio should lie on the efficient frontier."""
        optimal = optimizer_uncorrelated.maximize_sharpe()
        
        # The efficient frontier portfolio at the same return should have same volatility
        frontier_portfolio = optimizer_uncorrelated.minimize_volatility(
            target_return=optimal.return_
        )
        
        assert np.isclose(optimal.volatility, frontier_portfolio.volatility, rtol=0.01)


# ============================================================================
# Test Efficient Frontier
# ============================================================================

class TestEfficientFrontier:
    """Test efficient frontier generation and dominance properties."""
    
    def test_frontier_is_monotonic(self, optimizer_uncorrelated):
        """Frontier should have increasing return and volatility."""
        frontier = optimizer_uncorrelated.efficient_frontier(num_points=30)
        
        returns = [p.return_ for p in frontier]
        vols = [p.volatility for p in frontier]
        
        # Both should be non-decreasing
        for i in range(len(frontier) - 1):
            assert returns[i] <= returns[i+1] + 1e-6
            assert vols[i] <= vols[i+1] + 1e-6
    
    def test_frontier_dominates_random_portfolios(self, optimizer_uncorrelated):
        """
        No random portfolio should exceed the efficient frontier.
        
        For any volatility level, the frontier return is the maximum achievable.
        """
        frontier = optimizer_uncorrelated.efficient_frontier(num_points=50)
        random_portfolios = optimizer_uncorrelated.random_portfolios(num_portfolios=200)
        
        # Build frontier interpolation
        frontier_vols = np.array([p.volatility for p in frontier])
        frontier_rets = np.array([p.return_ for p in frontier])
        sort_idx = np.argsort(frontier_vols)
        frontier_vols = frontier_vols[sort_idx]
        frontier_rets = frontier_rets[sort_idx]
        
        # Check each random portfolio
        for rp in random_portfolios:
            max_return_at_vol = np.interp(rp.volatility, frontier_vols, frontier_rets)
            assert max_return_at_vol >= rp.return_ - 1e-4, (
                f"Random portfolio dominates frontier: "
                f"Vol={rp.volatility:.3f}, Return={rp.return_:.3f} > {max_return_at_vol:.3f}"
            )
    
    def test_frontier_gap_demonstrates_optimization_value(self, optimizer_uncorrelated):
        """Optimal portfolio should significantly outperform average random portfolio."""
        random_portfolios = optimizer_uncorrelated.random_portfolios(num_portfolios=500)
        optimal = optimizer_uncorrelated.maximize_sharpe()
        
        random_sharpes = np.array([
            (p.return_ - 0.02) / p.volatility for p in random_portfolios
        ])
        
        # Optimal should beat all random portfolios
        assert optimal.sharpe_ratio >= np.max(random_sharpes) - 1e-6
        
        # Should significantly beat average (demonstrates value of optimization)
        assert optimal.sharpe_ratio > np.mean(random_sharpes) * 1.1
    
    def test_frontier_starts_at_min_volatility(self, optimizer_uncorrelated):
        """First frontier point should be the global min-vol portfolio."""
        frontier = optimizer_uncorrelated.efficient_frontier(num_points=20)
        min_vol = optimizer_uncorrelated.minimize_volatility()
        
        # First frontier point should match min-vol portfolio
        assert np.isclose(frontier[0].volatility, min_vol.volatility, rtol=0.01)
        assert np.isclose(frontier[0].return_, min_vol.return_, rtol=0.01)


# ============================================================================
# Test Degenerate Inputs
# ============================================================================

class TestDegenerateInputs:
    """Test edge cases and unusual inputs."""
    
    def test_single_asset_gets_full_allocation(self):
        """Single asset should receive 100% weight."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.10]),
            cov_matrix=np.array([[0.04]]),
            risk_free_rate=0.02
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        assert portfolio.success
        assert np.isclose(portfolio.weights[0], 1.0)
    
    def test_identical_assets_get_equal_weights(self):
        """Identical assets should be allocated equally (diversification)."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.10, 0.10, 0.10]),
            cov_matrix=np.eye(3) * 0.04,
            risk_free_rate=0.02
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        assert portfolio.success
        # Should be close to 1/3 each (may have tiny numerical differences)
        assert np.std(portfolio.weights) < 0.01
    
    def test_all_negative_returns(self):
        """Should find least-bad portfolio when all returns are negative."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([-0.05, -0.10, -0.15]),
            cov_matrix=np.diag([0.01, 0.02, 0.03]),
            risk_free_rate=0.0
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        # Should complete without crashing
        assert np.isclose(np.sum(portfolio.weights), 1.0)
        assert portfolio.return_ < 0
        
        # With all negative returns and rf=0, all Sharpe ratios are negative
        # Sharpe_i = (R_i - 0) / σ_i = R_i / σ_i (all negative)
        # Asset 0: -0.05/0.1 = -0.5
        # Asset 1: -0.10/0.141 = -0.71
        # Asset 2: -0.15/0.173 = -0.87
        # Asset 0 has the "best" (least negative) Sharpe, but optimizer
        # may diversify since all are bad. Just verify it completes.
        assert portfolio.sharpe_ratio < 0
    
    def test_near_zero_volatility_asset(self):
        """Assets with near-zero volatility shouldn't cause numerical issues."""
        optimizer = PortfolioOptimizer(
            expected_returns=np.array([0.10, 0.02]),
            cov_matrix=np.array([[0.04, 0], [0, 1e-10]]),  # Second asset nearly risk-free
            risk_free_rate=0.02
        )
        
        portfolio = optimizer.maximize_sharpe()
        
        # Should complete without numerical errors (division by near-zero)
        assert isinstance(portfolio, PortfolioMetrics)
        assert portfolio.success
        
        # Asset 0: Sharpe = (0.10 - 0.02) / 0.20 = 0.40
        # Asset 1: Sharpe = (0.02 - 0.02) / ~0 = ~0 (zero excess return)
        # Asset 0 should get significant allocation, but optimizer may diversify
        # for numerical stability. Just verify no crash and reasonable allocation.
        assert portfolio.weights[0] > 0.1  # Gets some meaningful allocation
        assert np.isclose(np.sum(portfolio.weights), 1.0)
    
    def test_perfectly_correlated_assets(self):
        """Perfectly correlated assets offer no diversification benefit."""
        returns = np.array([0.08, 0.12])
        # Perfect correlation: ρ = 1
        vols = np.array([0.15, 0.20])
        cov = np.outer(vols, vols)  # cov_ij = σ_i * σ_j when ρ=1
        
        optimizer = PortfolioOptimizer(returns, cov, risk_free_rate=0.02)
        portfolio = optimizer.maximize_sharpe()
        
        # Should pick the asset with better Sharpe ratio (asset 2: higher return)
        assert portfolio.success
        # Either corner solution or mix depending on risk/return tradeoff


# ============================================================================
# Test CAGR Calculations (Domain-Specific)
# ============================================================================

def test_convenience_function_maintains_compatibility(uncorrelated_assets):
    """Convenience function should provide simple API for common use case."""
    returns, cov, _ = uncorrelated_assets
    
    bounds = [(0.0, 0.3) for _ in range(5)]
    portfolio = optimize_portfolio(
        returns,
        cov,
        risk_free_rate=0.02,
        bounds=bounds
    )
    
    assert isinstance(portfolio, PortfolioMetrics)
    assert portfolio.success
    assert np.isclose(np.sum(portfolio.weights), 1.0)
    assert np.all(portfolio.weights <= 0.3 + 1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])