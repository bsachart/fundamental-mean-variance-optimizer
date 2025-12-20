"""
Module for Mean-Variance Optimization (MVO).

Philosophy:
    - "Strategic programming": Design the optimizer to be extensible (e.g., adding constraints later).
    - Maximize Sharpe Ratio is the default objective.
    - Robustness: Handle edge cases where risk-free rate > portfolio return (though rare in long-only equity).

Reference:
    - Modern Portfolio Theory (Markowitz).
"""

from dataclasses import dataclass
import numpy as np
import scipy.optimize as sco
import numpy.typing as npt
from typing import Dict, Tuple, Optional, List


@dataclass
class PortfolioMetrics:
    """
    Portfolio performance metrics.
    """
    return_: float
    volatility: float
    sharpe_ratio: float
    weights: npt.NDArray[np.float64]
    success: bool = True
    message: str = ""


class PortfolioOptimizer:
    """
    Deep module for portfolio optimization.
    """
    
    def __init__(
        self,
        expected_returns: npt.NDArray[np.float64],
        cov_matrix: npt.NDArray[np.float64],
        risk_free_rate: float = 0.02,
    ):
        self.expected_returns = expected_returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(expected_returns)
    
    def calculate_metrics(self, weights: npt.NDArray[np.float64]) -> PortfolioMetrics:
        ret = weights @ self.expected_returns
        vol = np.sqrt(weights @ self.cov_matrix @ weights)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return PortfolioMetrics(ret, vol, sharpe, weights)
    
    def maximize_sharpe(self, bounds: Optional[List[Tuple[float, float]]] = None) -> PortfolioMetrics:
        """
        Find portfolio with maximum Sharpe ratio.
        
        Args:
            bounds: Optional list of (min, max) weights per asset. 
                    If None, defaults to long-only (0, 1).
        """
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.num_assets)]

        def objective(weights):
            metrics = self.calculate_metrics(weights)
            return -metrics.sharpe_ratio
        
        result = self._optimize(objective=objective, bounds=bounds)
        
        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics
    
    def minimize_volatility(
        self, 
        bounds: Optional[List[Tuple[float, float]]] = None,
        target_return: Optional[float] = None
    ) -> PortfolioMetrics:
        """
        Find minimum volatility portfolio.
        """
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.num_assets)]

        def objective(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)
        
        result = self._optimize(
            objective=objective,
            bounds=bounds,
            target_return=target_return,
        )
        
        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics
    
    def efficient_frontier(
        self, 
        bounds: Optional[List[Tuple[float, float]]] = None,
        num_points: int = 20
    ) -> List[PortfolioMetrics]:
        """
        Calculate efficient frontier.
        """
        if bounds is None:
            bounds = [(0.0, 1.0) for _ in range(self.num_assets)]

        min_vol = self.minimize_volatility(bounds=bounds)
        min_return = min_vol.return_
        max_return = np.max(self.expected_returns)
        
        if min_return >= max_return:
            return [min_vol]
        
        target_returns = np.linspace(min_return, max_return, num_points)
        frontier = []
        
        for target in target_returns:
            portfolio = self.minimize_volatility(bounds=bounds, target_return=target)
            if portfolio.success:
                frontier.append(portfolio)
        
        return frontier
    
    def random_portfolios(self, num_portfolios: int = 1000) -> List[PortfolioMetrics]:
        """
        Generate random portfolios.
        Note: Simple Dirichlet doesn't handle arbitrary bounds perfectly,
        but for visualization it's usually fine or we can just clip.
        For this simplified version, we'll keep Dirichlet as a proxy.
        """
        portfolios = []
        alphas = [0.1, 0.3, 0.5, 1.0, 2.0]
        n_per_alpha = num_portfolios // len(alphas)
        
        for alpha in alphas:
            for _ in range(n_per_alpha):
                weights = np.random.dirichlet(np.ones(self.num_assets) * alpha)
                portfolios.append(self.calculate_metrics(weights))
        
        return portfolios
    
    def _optimize(
        self,
        objective,
        bounds: List[Tuple[float, float]],
        target_return: Optional[float] = None,
    ) -> sco.OptimizeResult:
        """
        Internal optimization routine.
        """
        # Initial guess: midpoint of bounds if possible, else 1/N
        initial_weights = []
        for low, high in bounds:
            if low <= 1.0/self.num_assets <= high:
                initial_weights.append(1.0/self.num_assets)
            else:
                initial_weights.append((low + high) / 2.0)
        
        initial_weights = np.array(initial_weights)
        # Normalize to sum to 1 if possible
        if np.sum(initial_weights) != 0:
            initial_weights = initial_weights / np.sum(initial_weights)

        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]
        
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda x, t=target_return: x @ self.expected_returns - t
            })
        
        return sco.minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=tuple(bounds),
            constraints=constraints,
        )


def optimize_portfolio(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.02,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> PortfolioMetrics:
    optimizer = PortfolioOptimizer(expected_returns, cov_matrix, risk_free_rate)
    return optimizer.maximize_sharpe(bounds)