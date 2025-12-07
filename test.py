import polars as pl
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
import yfinance as yf


TICKERS = [
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
    "F",
    "ASTS",
    "ACGL",
    "VLO",
    "MPC",
    "NVO",
    "PFE",
]

# Download monthly prices
monthly_df = yf.download(TICKERS, interval="1mo", period="5y")

# Keep only Close
close_df = monthly_df["Close"]

# Save to CSV
close_path = "./tmp/monthly_close_prices.csv"
close_df.to_csv(close_path)

print(f"✅ Saved to {close_path}")

NUM_YEARS = 2
NUM_MONTS = NUM_YEARS * 12


@dataclass
class TickerData:
    name: str
    monthly: pl.DataFrame
    yearly: pl.DataFrame


def add_returns(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns((pl.col(col).pct_change().alias("Return")))


def compute_cov_corr(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    tickers = df.columns
    cov_matrix = pl.DataFrame(
        {
            t1: [df.select(pl.cov(pl.col(t1), pl.col(t2))).item() for t2 in tickers]
            for t1 in tickers
        }
    )
    corr_matrix = pl.DataFrame(
        {
            t1: [df.select(pl.corr(pl.col(t1), pl.col(t2))).item() for t2 in tickers]
            for t1 in tickers
        }
    )
    return cov_matrix, corr_matrix


def concat_returns(
    data_dict: dict[str, TickerData], attr: str, n_tail: int
) -> pl.DataFrame:
    return pl.concat(
        [
            getattr(data, attr)
            .tail(n_tail)
            .select("Return")
            .rename({"Return": data.name})
            for data in data_dict.values()
        ],
        how="horizontal",
    )


# --- Read CSV and yearly aggregation ---
df = pl.read_csv("./tmp/monthly_close_prices.csv").with_columns(
    pl.col("Date").str.to_datetime(format="%Y-%m-%d")
)[:-1]

yearly_df = df.group_by_dynamic(
    "Date", every="12mo", closed="right", label="right", start_by="datapoint"
).agg(pl.all().last())

# --- Build TickerData dict ---
ticker_data = {
    ticker: TickerData(
        name=ticker,
        monthly=add_returns(df.tail(NUM_MONTS + 2).select(["Date", ticker]), ticker),
        yearly=add_returns(yearly_df.select(["Date", ticker]), ticker),
    )
    for ticker in df.columns
    if ticker != "Date"
}

# --- Prepare monthly & yearly returns ---
monthly_returns_df = concat_returns(ticker_data, "monthly", NUM_MONTS)
yearly_returns_df = concat_returns(ticker_data, "yearly", NUM_YEARS)

# --- Compute covariance & correlation ---
cov_month, corr_month = compute_cov_corr(monthly_returns_df)
scaled_cov_month = cov_month * 12
cov_year, corr_year = compute_cov_corr(yearly_returns_df)

tickers = monthly_returns_df.columns

# --- Plot all heatmaps in one figure ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.heatmap(
    cov_month.to_numpy(),
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    xticklabels=tickers,
    yticklabels=tickers,
    ax=axes[0, 0],
)
axes[0, 0].set_title("Monthly Covariance")

sns.heatmap(
    corr_month.to_numpy(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=tickers,
    yticklabels=tickers,
    vmin=-1,
    vmax=1,
    ax=axes[0, 1],
)
axes[0, 1].set_title("Monthly Correlation")

sns.heatmap(
    cov_year.to_numpy(),
    annot=True,
    fmt=".4f",
    cmap="coolwarm",
    xticklabels=tickers,
    yticklabels=tickers,
    ax=axes[1, 0],
)
axes[1, 0].set_title("Yearly Covariance")

sns.heatmap(
    corr_year.to_numpy(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    xticklabels=tickers,
    yticklabels=tickers,
    vmin=-1,
    vmax=1,
    ax=axes[1, 1],
)
axes[1, 1].set_title("Yearly Correlation")

plt.tight_layout()
plt.show()
