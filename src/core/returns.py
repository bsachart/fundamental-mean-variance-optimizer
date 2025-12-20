"""
Return generation models for portfolio optimization.
"""
import numpy as np
import numpy.typing as npt
from typing import Union

def calculate_view_returns(
    market_baseline: float,
    alpha_deltas: Union[npt.NDArray[np.float64], list[float]]
) -> npt.NDArray[np.float64]:
    """
    Calculates expected returns based on a market baseline and asset-specific alpha.
    
    Logic: mu = mu_market + delta_alpha
    
    Args:
        market_baseline: The general expected return of the market (e.g., 0.08).
        alpha_deltas: Array of outperformance/underperformance views per asset.
        
    Returns:
        npt.NDArray[np.float64]: Vector of expected returns.
    """
    alphas = np.asarray(alpha_deltas, dtype=np.float64)
    return market_baseline + alphas