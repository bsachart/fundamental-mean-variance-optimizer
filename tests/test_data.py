import pytest
import pandas as pd
import numpy as np
from src.core.data import load_and_validate_asset_metrics, fetch_prices_from_yfinance
from io import StringIO

def test_load_and_validate_asset_metrics_with_custom_return():
    csv_content = """Ticker,Implied Volatility (%),Alpha Delta (%),Custom Return (%),Constraint
AAPL,25.0,2.0,12.0,Long
GOOG,30.0,1.0,8.0,Both
"""
    df = load_and_validate_asset_metrics(StringIO(csv_content))
    assert len(df) == 2
    assert df.loc["AAPL", "Custom Return (%)"] == 12.0
    assert df.loc["GOOG", "Constraint"] == "Both"

def test_load_and_validate_asset_metrics_defaults():
    csv_content = """Ticker,Implied Volatility (%),Alpha Delta (%)
AAPL,25.0,2.0
"""
    df = load_and_validate_asset_metrics(StringIO(csv_content))
    assert df.loc["AAPL", "Custom Return (%)"] == 8.0
    assert df.loc["AAPL", "Constraint"] == "Both"

def test_fetch_prices_invalid_ticker():
    with pytest.raises(ValueError):
        fetch_prices_from_yfinance(["INVALID_TICKER_123456"], period="1d")

# Note: We won't run live yfinance tests in the automated suite to avoid CI flakiness,
# but we've verified the structure.
