"""
Module for calculating Risk (Covariance) from historical price data and implied volatility.

Philosophy:
    - Standard historical covariance is the base for correlation structure.
    - Implied Volatility (IV) provides forward-looking risk scaling.
    - We blend IV and historical vol to reduce "noise" while staying reactive.
"""

import numpy as np
import pandas as pd
import numpy.typing as npt
from typing import Literal, Union, Optional

# Type alias for supported frequencies
Frequency = Literal["daily", "weekly", "monthly"]

ANNUALIZATION_FACTORS = {"daily": 252, "weekly": 52, "monthly": 12}


def calculate_covariance_matrix(
    prices: pd.DataFrame, frequency: Frequency = "monthly"
) -> pd.DataFrame:
    """
    Calculates the annualized historical covariance matrix.
    """
    if prices.empty or len(prices) < 2:
        raise ValueError("Price history must contain at least 2 data points.")

    if frequency not in ANNUALIZATION_FACTORS:
        raise ValueError(f"Invalid frequency: {frequency}")

    # Compute Log Returns
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if log_returns.empty:
        raise ValueError("Not enough data to compute returns.")

    # Compute Sample Covariance and Annualize
    factor = ANNUALIZATION_FACTORS[frequency]
    return log_returns.cov() * factor


def calculate_hybrid_covariance(
    prices: pd.DataFrame,
    implied_vols: Union[npt.NDArray[np.float64], pd.Series],
    anchor_vols: Optional[Union[float, npt.NDArray[np.float64], pd.Series]] = None,
    w_iv: float = 0.5,
    frequency: Frequency = "monthly"
) -> pd.DataFrame:
    """
    Constructs a Hybrid Covariance Matrix by blending Forward-Looking IV 
    with Historical Volatility, while preserving historical correlations.
    
    Logic:
    1. Calculate historical covariance (Sigma_H).
    2. Extract historical correlation matrix (R) and historical vols (sigma_H).
    3. Blend vols: sigma_hybrid = w * IV + (1 - w) * sigma_H.
    4. Reconstruct Sigma_hybrid = diag(sigma_hybrid) * R * diag(sigma_hybrid).
    """
    # 1. Base Historical Covariance
    hist_cov = calculate_covariance_matrix(prices, frequency)
    tickers = hist_cov.columns
    
    # 2. Extract Components
    hist_vols = np.sqrt(np.diag(hist_cov))
    # Correlation matrix R = D^-1 * Sigma * D^-1 where D = diag(sigma)
    inv_vols = 1.0 / np.where(hist_vols == 0, 1e-8, hist_vols)
    corr_matrix = np.diag(inv_vols) @ hist_cov.values @ np.diag(inv_vols)
    
    # 3. Blend Volatilities
    ivs = np.asarray(implied_vols, dtype=np.float64)
    if len(ivs) != len(hist_vols):
        raise ValueError("Implied vols must match the number of assets.")
    
    # Use provided anchor (e.g. Market IV) or fallback to Hist Vol
    vols_for_blending = anchor_vols if anchor_vols is not None else hist_vols
    hybrid_vols = w_iv * ivs + (1.0 - w_iv) * vols_for_blending
    
    # 4. Reconstruct Covariance
    hybrid_cov_values = np.diag(hybrid_vols) @ corr_matrix @ np.diag(hybrid_vols)
    
    return pd.DataFrame(hybrid_cov_values, index=tickers, columns=tickers)


def calculate_correlation_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix from historical prices.
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns.corr()
