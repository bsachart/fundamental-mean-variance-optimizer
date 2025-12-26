import yfinance as yf
import polars as pl
import pandas as pd
import os
from dataclasses import dataclass
from typing import List, Literal, Optional

@dataclass
class TickerDef:
    ticker: str
    implied_vol: float       # Decimal (e.g., 0.25)
    constraint: Literal["Long", "Short", "Both"] = "Long"
   
    # Mode A: Simple
    custom_return: Optional[float] = None  # Decimal (e.g., 0.12)
   
    # Mode B: Fundamental
    current_sales: Optional[float] = None   # Units (e.g., Billions)
    current_npm: Optional[float] = None     # Decimal
    organic_growth: Optional[float] = None  # Decimal (Includes SBC; excludes buybacks/dividends)
    terminal_npm: Optional[float] = None    # Decimal
    exit_pe: Optional[float] = None         # Multiple
    n_years: int = 5

def calculate_expected_return(t: TickerDef, current_mc: float) -> float:
    """
    Logic hidden from the user.
    If fundamental metrics are provided, it runs the simulation.
    Otherwise, it defaults to the custom_return.
    """
    # Mode B: Fundamental Simulation
    if all(v is not None for v in [t.current_sales, t.current_npm, t.organic_growth, t.terminal_npm, t.exit_pe]):
        if current_mc <= 0:
            return 0.0
       
        sales = t.current_sales
        for year in range(1, t.n_years + 1):
            # Linear margin ramp
            fraction = year / t.n_years
            annual_npm = t.current_npm + (t.terminal_npm - t.current_npm) * fraction
           
            # Sales growth includes the reinvestment 'boost' from profit margins
            sales *= (1 + t.organic_growth + annual_npm)

        terminal_mc = (sales * t.terminal_npm) * t.exit_pe
        price_ratio = max(terminal_mc / current_mc, 0.0001)
        return (price_ratio ** (1 / t.n_years)) - 1
   
    # Mode A: Simple Return
    return t.custom_return if t.custom_return is not None else 0.0

def main():
    # 1. Configuration (All rates as decimals)
    universe_definitions = [
        # Example Mode B (Fundamental)
        TickerDef(
            ticker="GOOG",
            implied_vol=0.28,
            current_sales=307.0,
            current_npm=0.24,
            organic_growth=0.10,
            terminal_npm=0.26,
            exit_pe=22.0
        ),
        # Example Mode A (Simple)
        TickerDef(
            ticker="CROX",
            implied_vol=0.35,
            custom_return=0.15
        ),
        TickerDef(
            ticker="NVO",
            implied_vol=0.30,
            current_sales=35.0,
            current_npm=0.34,
            organic_growth=0.15,
            terminal_npm=0.38,
            exit_pe=30.0
        ),
    ]

    tmp_dir = "/home/bog/Documents/Efficient Portolio Construction/tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tickers = [t.ticker for t in universe_definitions]

    # 2. Fetch Market Data
    print(f"Fetching data for: {tickers}")
    try:
        data = yf.download(tickers, period="2y", interval="1wk", auto_adjust=True, progress=False)
        prices = data["Open"] if isinstance(data.columns, pd.MultiIndex) else data
       
        # Fetch current Market Cap for Mode B tickers (Sales are in Billions, so MC must be too)
        market_caps = {}
        for t_def in universe_definitions:
            if t_def.current_sales is not None:
                info = yf.Ticker(t_def.ticker).info
                market_caps[t_def.ticker] = info.get('marketCap', 0) / 1e9
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # 3. Process Metrics
    metrics_list = []
    for t in universe_definitions:
        current_mc = market_caps.get(t.ticker, 0.0)
        expected_mu = calculate_expected_return(t, current_mc)
       
        metrics_list.append({
            "Ticker": t.ticker,
            "Implied Volatility": t.implied_vol, # Saved as decimal (0.28)
            "Custom Return": expected_mu,        # Saved as decimal (0.15)
            "Constraint": t.constraint
        })
        print(f"Calculated {t.ticker}: Return = {expected_mu:.4f}")

    # 4. Save Outputs
    # Save Prices
    df_prices = pl.from_pandas(prices.reset_index())
    df_prices.write_csv(os.path.join(tmp_dir, "universe.csv"))
   
    # Save Metrics
    df_metrics = pl.DataFrame(metrics_list)
    df_metrics.write_csv(os.path.join(tmp_dir, "metrics.csv"))

    print(f"Success. Files saved in {tmp_dir}")

if __name__ == "__main__":
    main()