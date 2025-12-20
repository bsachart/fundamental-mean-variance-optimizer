"""
Module for loading and validating input data.

Philosophy:
    - Validate inputs rigorously at the boundary.
    - Focus on minimalist, precise data structures.
"""

import pandas as pd
import numpy as np
import warnings
import yfinance as yf
from typing import Tuple, Literal, List

Frequency = Literal["daily", "weekly", "monthly"]


def infer_frequency(df: pd.DataFrame) -> Frequency:
    """
    Infers the frequency of the DatetimeIndex.
    """
    if len(df) < 2:
        return "monthly"

    diff = df.index.to_series().diff().mean()
    days = diff.days

    if days is None:
        return "monthly"

    if days <= 4:
        return "daily"
    elif days <= 10:
        return "weekly"
    else:
        return "monthly"


def load_and_validate_prices(file_path_or_buffer) -> pd.DataFrame:
    """
    Loads historical price data from a CSV.
    """
    try:
        df = pd.read_csv(file_path_or_buffer, index_col=0, parse_dates=True)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    if df.empty:
        raise ValueError("Price data is empty.")

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            raise ValueError("First column must be Date.")

    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    return df_numeric.sort_index()


def fetch_prices_from_yfinance(
    tickers: List[str], period: str = "2y", interval: str = "1mo"
) -> pd.DataFrame:
    """
    Downloads historical adjusted closing prices from Yahoo Finance.
    
    Args:
        tickers: List of ticker symbols.
        period: Time period (e.g., '1y', '2y', '5y').
        interval: Data interval ('1d', '1wk', '1mo').
    """
    if not tickers:
        raise ValueError("Ticker list cannot be empty.")

    try:
        # auto_adjust=True for split/dividend adjusted prices
        df = yf.download(tickers, period=period, interval=interval, progress=False, auto_adjust=True)
        
        # Handle yfinance multi-index vs single-index quirks
        if isinstance(df.columns, pd.MultiIndex):
            # If multiple tickers, columns are (Column, Ticker)
            # We want just the 'Close' prices for each ticker
            df = df['Close']
        elif 'Close' in df.columns:
            # If single ticker, columns might just be Column names
            df = df[['Close']]
            df.columns = [tickers[0]]
        
        if df.empty:
            raise ValueError(f"No data returned for tickers: {tickers}")
            
        return df.sort_index()
        
    except Exception as e:
        raise ValueError(f"Failed to fetch data from Yahoo Finance: {e}")


def load_and_validate_asset_metrics(file_path_or_buffer) -> pd.DataFrame:
    """
    Loads asset-specific metrics (Implied Vol, Alpha Delta, Constraints).

    Expected Format:
        - Index: Ticker
        - Columns:
            - 'Implied Volatility (%)': percentage (e.g. 30.0)
            - 'Alpha Delta (%)': percentage (e.g. 2.0)
            - 'Custom Return (%)': percentage (e.g. 8.0)
            - 'Constraint': 'Long', 'Short', or 'Both'
    """
    try:
        df = pd.read_csv(file_path_or_buffer, index_col=0)
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")

    # Required for the core logic if uploaded
    possible_numeric = ["Implied Volatility (%)", "Alpha Delta (%)", "Custom Return (%)"]
    
    # Check for missing basic columns
    required_columns = ["Implied Volatility (%)", "Alpha Delta (%)"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        # Fallback to legacy names if they exist and convert to %
        legacy_map = {
            "Implied Volatility": "Implied Volatility (%)",
            "Alpha Delta": "Alpha Delta (%)",
            "Custom Return": "Custom Return (%)"
        }
        found_legacy = False
        for old, new in legacy_map.items():
            if old in df.columns:
                df[new] = df[old] * 100.0
                found_legacy = True
        
        if not found_legacy:
            raise ValueError(f"Missing required columns: {required_columns}")

    # Validate numeric types
    for col in possible_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isnull().any():
                raise ValueError(f"Column '{col}' contains non-numeric data.")
        elif col == "Custom Return (%)":
            # Default if missing but requested by UX
            df[col] = 8.0

    # Default constraint to 'Both' if missing
    if "Constraint" not in df.columns:
        df["Constraint"] = "Both"
    else:
        valid_constraints = ["Long", "Short", "Both"]
        df["Constraint"] = df["Constraint"].str.capitalize()
        invalid = df[~df["Constraint"].isin(valid_constraints)]
        if not invalid.empty:
            raise ValueError(f"Invalid constraints found for: {invalid.index.tolist()}. Must be Long, Short, or Both.")

    return df


def align_tickers(
    prices: pd.DataFrame, metrics: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns Price and Metric DataFrames to common tickers.
    """
    common_tickers = prices.columns.intersection(metrics.index)

    if len(common_tickers) == 0:
        raise ValueError("No common tickers found.")

    if len(common_tickers) < len(metrics.index):
        missing = metrics.index.difference(prices.columns)
        warnings.warn(f"Dropping tickers with missing price history: {missing.tolist()}")

    return prices[common_tickers], metrics.loc[common_tickers]

