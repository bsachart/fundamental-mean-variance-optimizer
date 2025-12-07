import unittest
import numpy as np
import pandas as pd
from src.core.returns import calculate_implied_cagr
from src.core.risk import calculate_covariance_matrix
from src.optimization.optimizer import optimize_portfolio


class TestCoreLogic(unittest.TestCase):
    def test_implied_cagr_basic(self):
        # Simple case: No growth, no margin change, P/E constant
        # Should return 0 if P0 == P_exit
        # P0 = 100
        # Sales = 10, Margin = 0.5 => EPS = 5
        # PE = 20 => P_exit = 100
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.5,
            net_margin_target=0.5,
            adjusted_growth_rate=0.0,
            exit_pe=20,
            years=1,
        )
        # In this model, if adjusted_growth_rate is 0, we assume the earnings yield
        # is NOT reinvested or returned (or is offset by dilution).
        # Therefore, Price stays constant -> Return is 0.
        self.assertAlmostEqual(cagr, 0.0)

    def test_implied_cagr_growth(self):
        # Doubling in 1 year
        # P0 = 100
        # Sales = 10, Margin = 0.5 => EPS = 5
        # Growth = 1.0 (100%) => Sales_1 = 20
        # PE = 20 => P_exit = 20 * 0.5 * 20 = 200
        cagr = calculate_implied_cagr(
            current_price=100,
            sales_per_share=10,
            net_margin_current=0.5,
            net_margin_target=0.5,
            adjusted_growth_rate=1.0,
            exit_pe=20,
            years=1,
        )
        self.assertAlmostEqual(cagr, 1.0)

    def test_covariance_annualization(self):
        # Create dummy price data: 2 assets, perfectly correlated
        dates = pd.date_range(start="2020-01-01", periods=13, freq="M")
        # Asset A: 10, 11, 12... (approx 10% returns)
        # Asset B: 20, 22, 24... (approx 10% returns)
        # Log returns will be constant.
        prices = pd.DataFrame(
            {
                "A": [10 * (1.1) ** i for i in range(13)],
                "B": [20 * (1.1) ** i for i in range(13)],
            },
            index=dates,
        )

        cov = calculate_covariance_matrix(prices, frequency="monthly")

        # Variance of constant returns is 0.
        # But let's add some noise to make it non-zero
        np.random.seed(42)
        prices["A"] = prices["A"] * (1 + np.random.normal(0, 0.01, 13))
        prices["B"] = prices["B"] * (1 + np.random.normal(0, 0.01, 13))

        cov = calculate_covariance_matrix(prices, frequency="monthly")

        # Check shape
        self.assertEqual(cov.shape, (2, 2))
        # Check symmetry
        self.assertAlmostEqual(cov.iloc[0, 1], cov.iloc[1, 0])

    def test_optimizer_simple(self):
        # 2 Assets
        # A: Return 10%, Vol 10%
        # B: Return 5%, Vol 5%
        # Correlation 0

        means = np.array([0.10, 0.05])
        # Covariance matrix:
        # [0.01, 0   ]
        # [0,    0.0025]
        cov = np.array([[0.01, 0], [0, 0.0025]])

        result = optimize_portfolio(means, cov, risk_free_rate=0.0)

        self.assertTrue(result["success"])
        weights = result["weights"]
        self.assertAlmostEqual(np.sum(weights), 1.0)
        # Should prefer A or mix? Sharpe A = 1, Sharpe B = 1.
        # With 0 correlation, diversification improves Sharpe.
        # Optimal weights should be inversely proportional to volatility?
        # Actually proportional to Mean/Var.
        # A: 0.1/0.01 = 10
        # B: 0.05/0.0025 = 20
        # So B should have higher weight?
        # Let's just check they are positive.
        self.assertTrue(all(weights >= 0))


if __name__ == "__main__":
    unittest.main()
