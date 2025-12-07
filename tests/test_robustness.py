
import pytest
import numpy as np
import pandas as pd
from src.optimization.optimizer import (
    optimize_portfolio,
    calculate_efficient_frontier,
    generate_random_portfolios,
    portfolio_performance,
)

@pytest.fixture
def sample_data():
    """
    Creates sample expected returns and covariance matrix for testing.
    """
    # 5 assets
    expected_returns = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    
    # Simple diagonal covariance for uncorrelated assets (easier to reason about)
    # Volatilities: 10%, 15%, 20%, 25%, 30%
    vols = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    cov_matrix = np.diag(vols**2)
    
    return expected_returns, cov_matrix

def test_max_weight_constraint(sample_data):
    """
    Verifies that the optimizer respects the max_weight constraint.
    """
    expected_returns, cov_matrix = sample_data
    max_weight = 0.25
    
    result = optimize_portfolio(
        expected_returns, 
        cov_matrix, 
        max_weight=max_weight
    )
    
    assert result['success']
    weights = result['weights']
    
    # Check 1: Sum to 1
    assert np.isclose(np.sum(weights), 1.0)
    
    # Check 2: No weight exceeds max_weight (with small tolerance)
    assert np.all(weights <= max_weight + 1e-6)
    
    # Check 3: Weights are non-negative
    assert np.all(weights >= -1e-6)

def test_efficient_frontier_dominance(sample_data):
    """
    Verifies that the Efficient Frontier dominates random portfolios.
    For any random portfolio, there should be a point on the frontier 
    with <= volatility and >= return (or strictly better).
    
    Since the frontier is discrete points, we check that all random portfolios
    lie 'below/right' of the interpolated frontier curve.
    """
    expected_returns, cov_matrix = sample_data
    
    # 1. Generate Frontier
    frontier_points = calculate_efficient_frontier(expected_returns, cov_matrix, num_points=50)
    frontier_df = pd.DataFrame(frontier_points)
    
    # 2. Generate Random Portfolios
    random_portfolios = generate_random_portfolios(100, expected_returns, cov_matrix)
    random_df = pd.DataFrame(random_portfolios)
    
    # 3. Check Dominance
    # For each random portfolio, find the frontier point with the closest volatility
    # The frontier point's return should be >= random portfolio's return
    
    for _, rp in random_df.iterrows():
        rp_vol = rp['volatility']
        rp_ret = rp['return']
        
        # Find closest frontier point by volatility
        # Note: Frontier is sorted by volatility? Usually yes if starting from Min Vol.
        # Let's find the frontier point with vol <= rp_vol (closest from left)
        # Or just find the one with closest volatility.
        
        # Better check: For a given volatility, the frontier return is the MAX return.
        # So if we interpolate the frontier return at rp_vol, it should be >= rp_ret.
        
        # Simple check: Find frontier point with slightly higher volatility
        # Its return must be >= rp_ret.
        
        # Let's use numpy interp
        frontier_vols = frontier_df['volatility'].values
        frontier_rets = frontier_df['return'].values
        
        # Ensure sorted for interp
        sort_idx = np.argsort(frontier_vols)
        frontier_vols = frontier_vols[sort_idx]
        frontier_rets = frontier_rets[sort_idx]
        
        # Interpolate max return for this volatility
        max_theoretical_return = np.interp(rp_vol, frontier_vols, frontier_rets)
        
        # Allow small numerical error margin
        assert max_theoretical_return >= rp_ret - 1e-4, \
            f"Random portfolio (Vol={rp_vol:.2%}, Ret={rp_ret:.2%}) exceeds frontier (MaxRet={max_theoretical_return:.2%})"

def test_frontier_gap_explanation(sample_data):
    """
    Explicitly tests that there IS a gap, i.e., random portfolios are suboptimal.
    This confirms the user's observation is expected behavior.
    """
    expected_returns, cov_matrix = sample_data
    
    # Generate many random portfolios
    random_portfolios = generate_random_portfolios(500, expected_returns, cov_matrix)
    random_df = pd.DataFrame(random_portfolios)
    
    # Calculate Optimal Portfolio (Max Sharpe)
    opt_result = optimize_portfolio(expected_returns, cov_matrix)
    opt_sharpe = opt_result['sharpe']
    
    # Calculate Sharpe for all random portfolios
    random_sharpes = (random_df['return'] - 0.02) / random_df['volatility']
    
    # The optimal Sharpe should be higher than ANY random portfolio's Sharpe
    max_random_sharpe = np.max(random_sharpes)
    
    assert opt_sharpe >= max_random_sharpe, "Optimizer failed to find better Sharpe than random guessing"
    
    # Verify there is a significant gap on average
    # (Not strictly a unit test, but validates the 'cloud' visual)
    avg_random_sharpe = np.mean(random_sharpes)
    assert opt_sharpe > avg_random_sharpe * 1.1, "Optimal portfolio should be significantly better than average random portfolio"
