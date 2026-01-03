"""
Data loader for portfolio universe (Pandas version for browser deployment).

Loads and aligns price history and asset metrics into a unified Universe.
"""

import pandas as pd
from typing import Union, List
from pathlib import Path
from dataclasses import dataclass
import warnings
from io import StringIO

# Types
FileInput = Union[str, Path, pd.DataFrame, StringIO]


@dataclass(frozen=True)
class Universe:
    prices: pd.DataFrame
    metrics: pd.DataFrame
    tickers: List[str]


class DataValidationError(Exception):
    """Raised when data structure is fundamentally broken."""

    pass


def load_universe(price_source: FileInput, metric_source: FileInput) -> Universe:
    """
    Load and align price and metric data into a unified portfolio universe.
    """
    prices = _load_prices(price_source)
    metrics = _load_metrics(metric_source)

    # Intersection logic
    price_tickers = set(prices.columns) - {"date"}
    metric_tickers = set(metrics["ticker"].values)
    common = sorted(list(price_tickers.intersection(metric_tickers)))

    if not common:
        raise DataValidationError(
            f"No overlapping tickers found. Prices: {list(price_tickers)[:3]}..., Metrics: {list(metric_tickers)[:3]}..."
        )

    return Universe(
        prices=prices[["date"] + common].sort_values("date").reset_index(drop=True),
        metrics=metrics[metrics["ticker"].isin(common)]
        .sort_values("ticker")
        .reset_index(drop=True),
        tickers=common,
    )


def _read_df(source: FileInput) -> pd.DataFrame:
    """Standardizes input reading (String/Path/Buffer -> DataFrame)."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    try:
        # StringIO or Path string
        df = pd.read_csv(source)
        # normalize columns immediately to lower case
        df.columns = df.columns.str.lower().str.strip()
        return df
    except Exception as e:
        raise DataValidationError(f"Could not read source: {e}")


def _load_prices(source: FileInput) -> pd.DataFrame:
    df = _read_df(source)

    # Normalize 'date' column
    date_col = next((c for c in df.columns if "date" in c), None)
    if not date_col:
        date_col = df.columns[0]
    df = df.rename(columns={date_col: "date"})

    # CRITICAL: Ensure all ticker columns are LOWERCASE to match metrics
    # This must happen BEFORE we use the columns for any logic
    df.columns = [c.lower() if c != "date" else c for c in df.columns]

    # Convert date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Convert all ticker columns to numeric
    for col in df.columns:
        if col != "date":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Forward fill then Backward fill
    df = df.sort_values("date").ffill().bfill()

    return df


def _load_metrics(source: FileInput) -> pd.DataFrame:
    df = _read_df(source)

    # Required columns check
    # We allow loose matching (trimming spaces is done in _read_df)
    required = {"ticker", "expected_return", "implied_volatility"}
    missing = required - set(df.columns)

    # Try to alias common alternatives if missing
    if missing:
        aliases = {
            "symbol": "ticker",
            "return": "expected_return",
            "er": "expected_return",
            "vol": "implied_volatility",
            "iv": "implied_volatility",
        }
        df = df.rename(columns=aliases)
        missing = required - set(df.columns)

    if missing:
        raise DataValidationError(f"Metrics missing required columns: {missing}")

    # Convert to numeric
    df["expected_return"] = pd.to_numeric(df["expected_return"], errors="coerce")
    df["implied_volatility"] = pd.to_numeric(df["implied_volatility"], errors="coerce")

    # Provide defaults for weights
    if "min_weight" not in df.columns:
        df["min_weight"] = 0.0
    else:
        df["min_weight"] = pd.to_numeric(df["min_weight"], errors="coerce").fillna(0.0)

    if "max_weight" not in df.columns:
        df["max_weight"] = 1.0
    else:
        df["max_weight"] = pd.to_numeric(df["max_weight"], errors="coerce").fillna(1.0)

    # Clean strings and force LOWERCASE
    # This aligns with the lowercase columns in _load_prices
    df["ticker"] = df["ticker"].astype(str).str.strip().str.lower()

    # Validation
    if (df["implied_volatility"] <= 0).any():
        # Soft handle: clip to 1% instead of crashing
        warnings.warn("Found non-positive volatility. Clipping to 0.01.")
        df["implied_volatility"] = df["implied_volatility"].clip(lower=0.01)

    return df
