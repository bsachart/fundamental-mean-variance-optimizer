"""
Module for Mean-Variance Optimization (MVO).

Philosophy:
    - Optimize over risky assets only (standard MPT approach)
    - Cash allocation determined separately based on risk preferences
    - Maximize Sharpe Ratio finds the tangency portfolio
    - Target volatility approach for final allocation

Reference:
    - Modern Portfolio Theory (Markowitz, 1952)
    - Capital Market Line (Tobin, 1958)
"""


from dataclasses import dataclass
import numpy as np
import scipy.optimize as sco
import numpy.typing as npt
from typing import Dict, Tuple, Optional, List

# Constants
ZERO_VOL_THRESHOLD = 1e-8      # Numerical zero for volatility checks
MIN_SHARPE_DENOMINATOR = 1e-6  # Minimum volatility for Sharpe calculation


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
        self.num_assets = len(self.expected_returns)

    def _clean_bounds(
        self, bounds: Optional[List[Tuple[float, float]]]
    ) -> List[Tuple[float, float]]:
        """Ensures bounds match the number of assets."""
        if bounds is None:
            return [(0.0, 1.0) for _ in range(self.num_assets)]
        
        if len(bounds) != self.num_assets:
            raise ValueError(
                f"Bounds length ({len(bounds)}) must match number of assets ({self.num_assets})"
            )
        
        return list(bounds)

    def calculate_metrics(self, weights: npt.NDArray[np.float64]) -> PortfolioMetrics:
        ret = weights @ self.expected_returns
        vol = np.sqrt(weights @ self.cov_matrix @ weights)
        
        # Determine Sharpe with handling for edge cases
        if vol > MIN_SHARPE_DENOMINATOR:
            sharpe = (ret - self.risk_free_rate) / vol
        else:
            sharpe = 0.0

        return PortfolioMetrics(ret, vol, sharpe, weights)

    def maximize_sharpe(
        self, bounds: Optional[List[Tuple[float, float]]] = None
    ) -> PortfolioMetrics:
        """
        Find portfolio with maximum Sharpe ratio.

        Args:
            bounds: Optional list of (min, max) weights per asset.
                    If None, defaults to long-only (0, 1).
        """
        clean_bounds = self._clean_bounds(bounds)

        def objective(weights):
            ret = weights @ self.expected_returns
            vol = np.sqrt(weights @ self.cov_matrix @ weights)
            
            # Handle zero volatility edge case
            if vol < ZERO_VOL_THRESHOLD:
                return 0.0
            
            # Standard Sharpe ratio: excess return per unit risk
            sharpe = (ret - self.risk_free_rate) / vol
            return -sharpe  # Negative because we minimize

        result = self._optimize(objective=objective, bounds=clean_bounds)

        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics

    def minimize_volatility(
        self,
        bounds: Optional[List[Tuple[float, float]]] = None,
        target_return: Optional[float] = None,
    ) -> PortfolioMetrics:
        """
        Find minimum volatility portfolio.
        """
        clean_bounds = self._clean_bounds(bounds)

        def objective(weights):
            return np.sqrt(weights @ self.cov_matrix @ weights)

        result = self._optimize(
            objective=objective,
            bounds=clean_bounds,
            target_return=target_return,
        )

        metrics = self.calculate_metrics(result.x)
        metrics.success = result.success
        metrics.message = result.message
        return metrics

    def efficient_frontier(
        self, bounds: Optional[List[Tuple[float, float]]] = None, num_points: int = 20
    ) -> List[PortfolioMetrics]:
        """
        Calculate efficient frontier.
        """
        clean_bounds = self._clean_bounds(bounds)

        min_vol = self.minimize_volatility(bounds=clean_bounds)
        min_return = min_vol.return_
        max_return = np.max(self.expected_returns)

        if min_return >= max_return:
            return [min_vol]

        target_returns = np.linspace(min_return, max_return, num_points)
        frontier = []

        for target in target_returns:
            portfolio = self.minimize_volatility(
                bounds=clean_bounds, target_return=target
            )
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
    
    def allocate_with_cash(
        self,
        tangency_portfolio: PortfolioMetrics,
        target_volatility: float,
        asset_names: Optional[List[str]] = None
    ) -> Dict:
        """
        Allocate between tangency portfolio and cash to achieve target volatility.
        
        Args:
            tangency_portfolio: The optimal risky portfolio (from maximize_sharpe)
            target_volatility: Desired portfolio volatility (e.g., 0.10 for 10%)
            asset_names: Optional list of asset names for the output
        
        Returns:
            Dictionary containing:
                - 'risky_fraction': fraction invested in risky portfolio
                - 'cash_fraction': fraction in cash
                - 'final_weights': weights including cash
                - 'final_return': expected return of final portfolio
                - 'final_volatility': volatility of final portfolio
                - 'final_sharpe': Sharpe ratio of final portfolio
                - 'allocation_table': dict mapping asset names to weights
        """
        if target_volatility < 0:
            raise ValueError("Target volatility must be non-negative")
        
        # Calculate fraction to invest in risky portfolio
        if tangency_portfolio.volatility > ZERO_VOL_THRESHOLD:
            risky_fraction = target_volatility / tangency_portfolio.volatility
        else:
            risky_fraction = 1.0 if target_volatility > 0 else 0.0

        # Cap at 100% (no leverage allowed)
        risky_fraction = min(risky_fraction, 1.0)
        cash_fraction = 1.0 - risky_fraction
        
        # Scale tangency portfolio weights
        final_risky_weights = tangency_portfolio.weights * risky_fraction
        
        # Calculate final portfolio metrics
        final_return = (
            risky_fraction * tangency_portfolio.return_ + 
            cash_fraction * self.risk_free_rate
        )
        final_volatility = risky_fraction * tangency_portfolio.volatility
        
        if final_volatility > ZERO_VOL_THRESHOLD:
            final_sharpe = (final_return - self.risk_free_rate) / final_volatility
        else:
            final_sharpe = 0.0
        
        # Create allocation table
        allocation_table = {}
        if asset_names:
            for i, name in enumerate(asset_names):
                allocation_table[name] = final_risky_weights[i]
        else:
            for i in range(len(final_risky_weights)):
                allocation_table[f"Asset_{i}"] = final_risky_weights[i]
        
        allocation_table["CASH"] = cash_fraction
        
        return {
            'risky_fraction': risky_fraction,
            'cash_fraction': cash_fraction,
            'final_weights': np.append(final_risky_weights, cash_fraction),
            'final_return': final_return,
            'final_volatility': final_volatility,
            'final_sharpe': final_sharpe,
            'allocation_table': allocation_table
        }

    def _optimize(
        self,
        objective,
        bounds: List[Tuple[float, float]],
        target_return: Optional[float] = None,
    ) -> sco.OptimizeResult:
        """
        Internal optimization routine.
        """
        # Start with equal weights, adjusted for bounds
        initial_weights = np.ones(self.num_assets) / self.num_assets
        
        # Adjust if bounds don't allow equal weights
        for i, (low, high) in enumerate(bounds):
            initial_weights[i] = np.clip(initial_weights[i], low, high)
        
        # Renormalize
        if initial_weights.sum() > 0:
            initial_weights = initial_weights / initial_weights.sum()

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - 1}]

        if target_return is not None:
            constraints.append(
                {
                    "type": "eq",
                    "fun": lambda x, t=target_return: x @ self.expected_returns - t,
                }
            )

        # Run optimization
        result = sco.minimize(
            objective,
            initial_weights,
            method="SLSQP",
            bounds=tuple(bounds),
            constraints=constraints,
        )

        return result


def optimize_portfolio(
    expected_returns: npt.NDArray[np.float64],
    cov_matrix: npt.NDArray[np.float64],
    risk_free_rate: float = 0.02,
    bounds: Optional[List[Tuple[float, float]]] = None,
) -> PortfolioMetrics:
    optimizer = PortfolioOptimizer(
        expected_returns, cov_matrix, risk_free_rate
    )
    return optimizer.maximize_sharpe(bounds)
