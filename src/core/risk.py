"""
Module for calculating Risk (Covariance) from historical price data.

Philosophy:
    - While we use fundamentals for returns, we trust the market for risk.
    - Historical price covariance is the best proxy for near-term correlations.
    - "Garbage in, garbage out": We must ensure proper annualization of the covariance matrix.

Reference:
    - "A Philosophy of Software Design": Define errors out of existence (e.g. by enforcing frequency inputs).
"""

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Literal

# Type alias for supported frequencies
Frequency = Literal["daily", "weekly", "monthly"]

ANNUALIZATION_FACTORS = {"daily": 252, "weekly": 52, "monthly": 12}


def calculate_covariance_matrix(
    prices: pd.DataFrame, frequency: Frequency = "monthly"
) -> pd.DataFrame:
    """
    Calculates the annualized covariance matrix from historical prices.

    Steps:
    1. Computes log returns: ln(P_t / P_{t-1}).
       Log returns are preferred for time-series aggregation and statistical properties.
    2. Computes the sample covariance matrix of these returns.
    3. Annualizes the matrix by multiplying by the frequency factor (e.g., 12 for monthly).

    Args:
        prices: A DataFrame where index is Datetime and columns are Tickers.
                Values are adjusted closing prices.
        frequency: The sampling frequency of the data ('daily', 'weekly', 'monthly').
                   Defaults to 'monthly'.

    Returns:
        A DataFrame representing the annualized covariance matrix (N x N).

    Raises:
        ValueError: If `prices` is empty or has fewer than 2 rows.
        ValueError: If `frequency` is invalid.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError(
            "Price history must contain at least 2 data points to compute returns."
        )

    if frequency not in ANNUALIZATION_FACTORS:
        raise ValueError(
            f"Invalid frequency: {frequency}. Must be one of {list(ANNUALIZATION_FACTORS.keys())}"
        )

    # 1. Compute Log Returns
    # We use log returns: r_t = ln(P_t) - ln(P_{t-1})
    # This is mathematically more robust than simple arithmetic returns for chaining,
    # though for single-period MVO, arithmetic returns are often used.
    # However, in continuous time finance, log returns are standard for covariance estimation.
    # For small time steps (daily/weekly), log returns approx arithmetic returns.
    log_returns = np.log(prices / prices.shift(1))

    # Drop the first NaN row created by the shift
    log_returns = log_returns.dropna()

    if log_returns.empty:
        raise ValueError("Not enough data to compute covariance after dropping NaNs.")

    # 2. Compute Sample Covariance
    # ddof=1 for unbiased estimator (standard in pandas/numpy)
    covariance = log_returns.cov()

    # 3. Annualize
    factor = ANNUALIZATION_FACTORS[frequency]
    annualized_covariance = covariance * factor

    return annualized_covariance


def calculate_correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix from historical prices.
    Correlation is time-invariant (scale-free), so annualization is not needed.

    Args:
        prices: A DataFrame where index is Datetime and columns are Tickers.

    Returns:
        A DataFrame representing the correlation matrix (N x N).
    """
    # Use log returns for consistency with covariance
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.corr()
