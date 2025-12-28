"""
Portfolio Optimizer - Mean-Variance Optimization engine.

Philosophy:
    - Pure Solver: Wraps scipy.optimize to solve mathematical objectives.
    - Explicit Inputs: No default risk-free rates.
"""

import numpy as np
import scipy.optimize as sco
from typing import TypedDict, List, Tuple, Optional


class PortfolioMetrics(TypedDict):
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    cash_weight: float


def find_tangency_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> PortfolioMetrics:
    """
    Solves for the Tangency Portfolio (Maximum Sharpe Ratio).
    """
    n_assets = len(expected_returns)

    if bounds is None:
        bounds = [(0.0, 1.0)] * n_assets

    def negative_sharpe(w):
        ret = w @ expected_returns
        vol = np.sqrt(w @ cov_matrix @ w)
        if vol < 1e-8:
            return 0.0
        return -(ret - risk_free_rate) / vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    x0 = np.ones(n_assets) / n_assets

    result = sco.minimize(
        negative_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        tol=1e-8,
    )

    weights = result.x if result.success else x0

    p_ret = weights @ expected_returns
    p_vol = np.sqrt(weights @ cov_matrix @ weights)
    sharpe = (p_ret - risk_free_rate) / p_vol if p_vol > 1e-8 else 0.0

    return {
        "weights": weights,
        "expected_return": p_ret,
        "volatility": p_vol,
        "sharpe_ratio": sharpe,
        "cash_weight": 0.0,
    }
