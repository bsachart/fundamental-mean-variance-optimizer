"""
Data loader for portfolio universe (Polars version).

Loads and aligns price history and asset metrics into a unified Universe.
"""

import polars as pl
from typing import Union, List, Set
from pathlib import Path
from dataclasses import dataclass
import warnings

FileInput = Union[str, Path, pl.DataFrame]


@dataclass(frozen=True)
class Universe:
    prices: pl.DataFrame
    metrics: pl.DataFrame
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
    metric_tickers = set(metrics["ticker"].to_list())
    common = sorted(list(price_tickers.intersection(metric_tickers)))

    if not common:
        raise DataValidationError(
            "No overlapping tickers found between prices and metrics."
        )

    return Universe(
        prices=prices.select(["date"] + common).sort("date"),
        metrics=metrics.filter(pl.col("ticker").is_in(common)).sort("ticker"),
        tickers=common,
    )


# --- Internal Helpers ---


def _read_df(source: FileInput) -> pl.DataFrame:
    """Standardizes input reading."""
    if isinstance(source, pl.DataFrame):
        return source
    try:
        return pl.read_csv(source, try_parse_dates=True)
    except Exception as e:
        raise DataValidationError(f"Could not read source: {e}")


def _normalize_columns(df: pl.DataFrame, structural_cols: Set[str]) -> pl.DataFrame:
    """Lowercases structural columns; warns if extra columns exist."""
    rename_map = {c: c.lower() for c in df.columns if c.lower() in structural_cols}
    df = df.rename(rename_map)

    # Identify extra columns that aren't structural and aren't (presumably) tickers
    extras = [
        c
        for c in df.columns
        if c.lower() not in structural_cols and c not in structural_cols
    ]
    if extras and "ticker" in df.columns:  # Only warn for metrics, not price matrix
        warnings.warn(f"Ignoring extra columns: {extras}")

    return df


def _load_prices(source: FileInput) -> pl.DataFrame:
    df = _read_df(source)

    # Normalize 'date' column
    if df.columns[0].lower() != "date":
        raise DataValidationError(
            f"Price data must start with 'date' column. Found: {df.columns[0]}"
        )

    df = df.rename({df.columns[0]: "date"})

    # Force types
    return df.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.col(pl.Float64).exclude("date"),  # Cast all others to float
        ]
    ).drop_nulls(subset=["date"])


def _load_metrics(source: FileInput) -> pl.DataFrame:
    df = _read_df(source)
    structural = {
        "ticker",
        "expected_return",
        "implied_volatility",
        "min_weight",
        "max_weight",
    }
    df = _normalize_columns(df, structural)

    # Required Columns check
    required = {"ticker", "expected_return", "implied_volatility"}
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(f"Metrics missing required columns: {missing}")

    # Provide defaults for weights
    df = df.with_columns(
        [
            pl.col("expected_return").cast(pl.Float64),
            pl.col("implied_volatility").cast(pl.Float64),
            # If weights don't exist, create them. If they do, cast them.
            pl.col("min_weight").cast(pl.Float64)
            if "min_weight" in df.columns
            else pl.lit(0.0).alias("min_weight"),
            pl.col("max_weight").cast(pl.Float64)
            if "max_weight" in df.columns
            else pl.lit(1.0).alias("max_weight"),
        ]
    )

    # Validation
    if (df["implied_volatility"] <= 0).any():
        raise DataValidationError("Implied volatility must be positive.")

    if (df["min_weight"] > df["max_weight"]).any():
        raise DataValidationError("Detected min_weight > max_weight.")

    return df
