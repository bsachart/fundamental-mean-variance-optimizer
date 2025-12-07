"""
Generate test data for the Quantamental Portfolio Optimizer.

This module downloads historical price data from Yahoo Finance and generates
mock fundamental data for portfolio optimization testing.

Output files:
    - {interval}_open_prices.csv: Historical open prices
    - fundamentals.csv: Mock fundamental metrics
"""

from typing import Dict, List, Literal
import yfinance as yf
import polars as pl
import numpy as np
from pathlib import Path

# Type alias for valid intervals
Interval = Literal["1d", "5d", "1wk", "1mo", "3mo"]

# Default configuration
TICKERS: List[str] = [
    "MSFT",
    "AAPL",
    "GOOG",
    "VIRT",
    "JPM",
    "INTC",
    "AMZN",
    "CROX",
    "NVDA",
    "C",
    "LULU",
    "BCC",
    "HOOD",
    "SOFI",
    "ACGL",
    "VLO",
    "MPC",
    "NVO",
    "PFE",
]

OUTPUT_DIR = "./tmp"
DEFAULT_PERIOD = "3y"
DEFAULT_INTERVAL: Interval = "1mo"

# Fundamental generation parameters
PRICE_TO_SALES_RANGE = (0.2, 0.8)
MARGIN_RANGE = (0.05, 0.25)
MARGIN_CHANGE_RANGE = (0.9, 1.2)
GROWTH_RATE_RANGE = (0.02, 0.15)
EXIT_PE_RANGE = (10.0, 30.0)
DEFAULT_PRICE = 100.0
RANDOM_SEED = 42


def generate_test_datasets(
    tickers: List[str],
    output_dir: str,
    period: str = DEFAULT_PERIOD,
    interval: Interval = DEFAULT_INTERVAL,
) -> None:
    """
    Generate complete test datasets for portfolio optimization.

    Downloads historical prices and generates mock fundamentals, handling all
    complexity of data fetching, format conversion, and file I/O internally.

    Args:
        tickers: List of ticker symbols
        output_dir: Directory for output files
        period: Historical period (e.g., "3y", "5y")
        interval: Data frequency (e.g., "1mo", "1wk", "1d")
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    prices = _download_and_save_prices(tickers, output_path, period, interval)
    prices_filled = prices.fill_null(strategy="forward")
    _generate_and_save_fundamentals(tickers, prices_filled, output_path)

    print(f"\n✅ Complete: {len(tickers)} tickers, {period} @ {interval}")


def _download_and_save_prices(
    tickers: List[str],
    output_dir: Path,
    period: str,
    interval: Interval,
) -> pl.DataFrame:
    """Download price data from Yahoo Finance and save to CSV."""
    print(f"Downloading prices ({interval}, {period})...")

    raw_data = yf.download(tickers, interval=interval, period=period, progress=False)

    # Extract Open prices (handles both single and multi-ticker cases)
    if hasattr(raw_data.columns, "levels"):
        open_prices = raw_data["Open"]
    else:
        open_prices = raw_data[["Open"]]

    open_prices = open_prices.dropna(how="all")
    prices_pl = pl.from_pandas(open_prices.reset_index())

    filename = f"{interval}_open_prices.csv"
    prices_pl.write_csv(output_dir / filename)
    print(f"✅ Saved {len(prices_pl)} rows to {filename}")

    return prices_pl


def _generate_and_save_fundamentals(
    tickers: List[str],
    prices: pl.DataFrame,
    output_dir: Path,
) -> None:
    """Generate mock fundamental data based on last prices."""
    print("Generating fundamentals...")

    np.random.seed(RANDOM_SEED)

    last_prices = _extract_last_prices(prices)
    records = [_create_fundamental_record(ticker, last_prices) for ticker in tickers]

    df = pl.DataFrame(records)
    df.write_csv(output_dir / "fundamentals.csv")
    print(f"✅ Saved fundamentals for {len(tickers)} tickers")


def _extract_last_prices(prices: pl.DataFrame) -> Dict[str, float]:
    """Extract most recent price for each ticker."""
    last_row = prices.tail(1)
    price_columns = [col for col in last_row.columns if col != "Date"]
    return {col: last_row[col][0] for col in price_columns}


def _create_fundamental_record(
    ticker: str,
    last_prices: Dict[str, float],
) -> Dict[str, float]:
    """
    Create realistic mock fundamental metrics.

    Generates statistically reasonable values for:
    - Sales/Share (P/S ratio 1.25-5x)
    - Current/Target Margin (5-25%)
    - Growth Rate (2-15%)
    - Exit PE (10-30x)
    """
    price = last_prices.get(ticker)
    if price is None or np.isnan(price):
        price = DEFAULT_PRICE

    sales_per_share = price * np.random.uniform(*PRICE_TO_SALES_RANGE)
    current_margin = np.random.uniform(*MARGIN_RANGE)
    target_margin = current_margin * np.random.uniform(*MARGIN_CHANGE_RANGE)
    growth_rate = np.random.uniform(*GROWTH_RATE_RANGE)
    exit_pe = np.random.uniform(*EXIT_PE_RANGE)

    return {
        "Ticker": ticker,
        "Current Price": float(price),
        "Sales/Share": float(sales_per_share),
        "Current Margin": float(current_margin),
        "Target Margin": float(target_margin),
        "Adjusted Growth Rate": float(growth_rate),
        "Exit PE": float(exit_pe),
    }


if __name__ == "__main__":
    generate_test_datasets(TICKERS, OUTPUT_DIR)
