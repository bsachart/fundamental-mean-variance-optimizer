"""
Streamlit UI for the Quantamental Portfolio Optimizer.

Philosophy:
    - "Incremental complexity": Simple default view, advanced options hidden.
    - "Quantamental": clearly separate Fundamental inputs (Returns) from Market inputs (Risk).
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import sys
import os
import base64
import io

# Add project root to path to allow imports
sys.path.append(os.getcwd())

from src.core.data import (
    load_and_validate_prices,
    load_and_validate_fundamentals,
    align_tickers,
    infer_frequency,
)
from src.core.returns import calculate_implied_cagr
from src.core.risk import calculate_covariance_matrix, calculate_correlation_matrix
from src.optimization import optimizer
from src.ui import charts
from streamlit_local_storage import LocalStorage

st.set_page_config(page_title="Quantamental Optimizer", layout="wide")

st.title("Quantamental Portfolio Optimizer")
st.markdown(r"""
**Philosophy:**
*   **Returns ($\mu$):** Derived from *Fundamentals* (Implied CAGR).
*   **Risk ($\Sigma$):** Derived from *Market History* (Covariance).
""")

# --- Sidebar: Inputs ---
# --- Persistence Setup (Client-Side LocalStorage) ---
localS = LocalStorage()

def save_to_local_storage(key, uploaded_file):
    """Save uploaded file to browser LocalStorage (Base64 encoded)."""
    if uploaded_file is not None:
        # Check size (LocalStorage limit is usually 5MB, base64 adds 33% overhead)
        # Limit to ~3.5MB to be safe
        if uploaded_file.size > 3.5 * 1024 * 1024:
            st.sidebar.warning(f"File {uploaded_file.name} is too large for browser storage (>3.5MB). It will not be saved.")
            return

        bytes_data = uploaded_file.getvalue()
        b64_str = base64.b64encode(bytes_data).decode()
        # We must provide a unique key for the component instance to avoid StreamlitDuplicateElementKey
        localS.setItem(key, b64_str, key=f"set_{key}")

def load_from_local_storage(key, filename):
    """Load file from LocalStorage and convert to BytesIO."""
    b64_str = localS.getItem(key)
    if b64_str:
        try:
            bytes_data = base64.b64decode(b64_str)
            file_obj = io.BytesIO(bytes_data)
            file_obj.name = filename # Pandas needs a name sometimes
            return file_obj
        except Exception as e:
            st.sidebar.error(f"Failed to load {filename} from storage: {e}")
    return None

st.sidebar.header("1. Data Inputs")

# Check for cached files in LocalStorage
# Note: LocalStorage component might take a moment to sync on first load
cached_price = load_from_local_storage("price_csv", "price_history.csv")
cached_fund = load_from_local_storage("fund_csv", "fundamentals.csv")

price_file = st.sidebar.file_uploader("Upload Price History (CSV)", type=["csv"])
fund_file = st.sidebar.file_uploader("Upload Fundamentals (CSV)", type=["csv"])

# Logic: Use uploaded file if present, else use cached file
if price_file:
    save_to_local_storage("price_csv", price_file)
elif cached_price:
    st.sidebar.info("Using cached prices from browser storage.")
    price_file = cached_price

if fund_file:
    save_to_local_storage("fund_csv", fund_file)
elif cached_fund:
    st.sidebar.info("Using cached fundamentals from browser storage.")
    fund_file = cached_fund

if st.sidebar.button("Clear Browser Storage"):
    localS.deleteAll()
    st.rerun()

st.sidebar.header("2. Optimization Parameters")
investment_horizon = st.sidebar.slider("Investment Horizon (Years)", 1, 15, 5)
risk_free_rate = (
    st.sidebar.number_input("Risk Free Rate (%)", value=4.0, step=0.1) / 100.0
)
max_weight = st.sidebar.slider("Max Weight per Asset", 0.0, 1.0, 0.2, 0.05)

# --- Main Area ---

# Initialize session state for data if not present
if 'prices' not in st.session_state:
    st.session_state['prices'] = None
if 'fundamentals' not in st.session_state:
    st.session_state['fundamentals'] = None
if 'aligned_tickers' not in st.session_state:
    st.session_state['aligned_tickers'] = None

# Load data from file ONLY if session_state is empty (preserves Forecast Editor edits)
if price_file and st.session_state['prices'] is None:
    try:
        prices_raw = load_and_validate_prices(price_file)
        st.session_state['prices'] = prices_raw
        st.success(f"Loaded Price History for {len(prices_raw.columns)} assets.")
    except Exception as e:
        st.error(f"Error processing price data: {e}")

if fund_file and st.session_state['fundamentals'] is None:
    try:
        funds_raw = load_and_validate_fundamentals(fund_file)
        st.session_state['fundamentals'] = funds_raw
    except Exception as e:
        st.error(f"Error processing fundamental data: {e}")

# Process Data if available
if st.session_state['prices'] is not None and st.session_state['fundamentals'] is not None:
    try:
        # Align Data
        # We re-align every time to ensure consistency, or check if already aligned?
        # Re-aligning is cheap enough.
        prices_aligned, funds_aligned = align_tickers(st.session_state['prices'], st.session_state['fundamentals'])
        tickers = prices_aligned.columns.tolist()
        st.session_state['aligned_tickers'] = tickers
        
        # Update session state with aligned data so Editor sees the same set
        # But wait, Editor might want to see all fundamentals? 
        # Let's keep 'fundamentals' as the raw loaded data, and work with aligned here.
        # Actually, for the Editor to be useful, it should edit the data that is used here.
        # So we should probably update 'fundamentals' in session state to be the aligned version if we want strict consistency,
        # OR just use the aligned version for calculations.
        # Let's use the aligned version for calculations here.

        st.success(
            f"Aligned {len(tickers)} assets with Fundamentals: {', '.join(tickers)}"
        )
        
        detected_freq = infer_frequency(prices_aligned)
        cov_matrix = calculate_covariance_matrix(prices_aligned, frequency=detected_freq)
        corr_matrix = calculate_correlation_matrix(prices_aligned)

        # --- Section 1: Historical Analysis ---
        with st.expander("Historical Analysis", expanded=False):
            col_hist1, col_hist2 = st.columns(2)

            with col_hist1:
                st.subheader("Performance")
                normalized_prices = ((prices_aligned / prices_aligned.iloc[0]) - 1) * 100
                df_long = normalized_prices.reset_index().melt(
                    "Date", var_name="Ticker", value_name="Return"
                )
                selection = alt.selection_point(fields=["Ticker"], bind="legend")
                chart = (
                    alt.Chart(df_long)
                    .mark_line(point=True)
                    .encode(
                        x="Date:T",
                        y=alt.Y("Return:Q", title="Return (%)"),
                        color="Ticker:N",
                        opacity=alt.condition(selection, alt.value(1), alt.value(0.2)),
                        tooltip=["Date", "Ticker", alt.Tooltip("Return", format=".2f")],
                    )
                    .add_params(selection)
                    .interactive()
                )
                st.altair_chart(chart, width="stretch")

            with col_hist2:
                st.subheader("Correlations")
                st.dataframe(
                    corr_matrix.style.background_gradient(
                        cmap="coolwarm", vmin=-1, vmax=1
                    ).format("{:.2f}"),
                    width="stretch",
                )

        # 4. Calculate Returns (Implied CAGR)
        if "Current Price" in funds_aligned.columns:
            P0 = funds_aligned["Current Price"]
        else:
            P0 = prices_aligned.iloc[-1]

        # Calculate CAGR for each asset (Vectorized)
        expected_returns = calculate_implied_cagr(
            current_price=P0.values,
            sales_per_share=funds_aligned["Sales/Share"].values,
            net_margin_current=funds_aligned["Current Margin"].values,
            net_margin_target=funds_aligned["Target Margin"].values,
            adjusted_growth_rate=funds_aligned["Adjusted Growth Rate"].values,
            exit_pe=funds_aligned["Exit PE"].values,
            years=investment_horizon,
        )

        # --- Section 2: Fundamental Analysis (Editable) ---
        st.header("2. Fundamental Analysis")
        
        # Create editable DataFrame with fundamental inputs + calculated Implied CAGR
        edit_df = funds_aligned[["Current Price", "Sales/Share", "Current Margin", "Target Margin", "Adjusted Growth Rate", "Exit PE"]].copy()
        edit_df["Implied CAGR"] = expected_returns
        
        # Calculate Current PE if possible
        try:
            current_eps = funds_aligned["Sales/Share"] * funds_aligned["Current Margin"]
            current_pe = P0.values / current_eps.replace(0, np.nan)
            edit_df["Current PE"] = current_pe
        except Exception:
            pass
        
        # Column configuration for proper formatting and to disable Implied CAGR editing
        column_config = {
            "Current Price": st.column_config.NumberColumn("Current Price", format="$%.2f", help="Current stock price"),
            "Sales/Share": st.column_config.NumberColumn("Sales/Share", format="$%.2f", help="Sales per share (TTM)"),
            "Current Margin": st.column_config.NumberColumn("Current Margin", format="%.4f", min_value=-1.0, max_value=1.0, help="Current net profit margin (decimal)"),
            "Target Margin": st.column_config.NumberColumn("Target Margin", format="%.4f", min_value=-1.0, max_value=1.0, help="Expected margin at exit (decimal)"),
            "Adjusted Growth Rate": st.column_config.NumberColumn("Growth Rate", format="%.4f", min_value=-1.0, max_value=2.0, help="Annual sales growth rate (decimal)"),
            "Exit PE": st.column_config.NumberColumn("Exit PE", format="%.1f", min_value=0, help="Expected P/E ratio at exit"),
            "Implied CAGR": st.column_config.NumberColumn("Implied CAGR", format="%.4f", disabled=True, help="Calculated implied return (not editable)"),
            "Current PE": st.column_config.NumberColumn("Current PE", format="%.1f", disabled=True, help="Current P/E ratio (calculated)"),
        }
        
        # Compact header with sort control
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown("### Edit Assumptions")
        with cols[1]:
            sort_options = ["↓ Ticker", "↑ Ticker"] + [f"↓ {col}" for col in edit_df.columns] + [f"↑ {col}" for col in edit_df.columns]
            sort_by = st.selectbox("Sort", sort_options, label_visibility="collapsed")
        
        # Apply sorting (↓ = ascending/A→Z, ↑ = descending/Z→A)
        if sort_by == "↓ Ticker":
            edit_df = edit_df.sort_index(ascending=True)
        elif sort_by == "↑ Ticker":
            edit_df = edit_df.sort_index(ascending=False)
        elif sort_by.startswith("↓ "):
            col_name = sort_by[2:]
            if col_name in edit_df.columns:
                edit_df = edit_df.sort_values(by=col_name, ascending=True)
        elif sort_by.startswith("↑ "):
            col_name = sort_by[2:]
            if col_name in edit_df.columns:
                edit_df = edit_df.sort_values(by=col_name, ascending=False)
        
        edited_df = st.data_editor(
            edit_df,
            column_config=column_config,
            use_container_width=True,
            key="fundamentals_editor"
        )
        
        # Sync edits back to session state (update the original fundamentals)
        editable_cols = ["Current Price", "Sales/Share", "Current Margin", "Target Margin", "Adjusted Growth Rate", "Exit PE"]
        original_vals = funds_aligned[editable_cols]
        edited_vals = edited_df[editable_cols]
        
        if not original_vals.equals(edited_vals):
            # Update session state fundamentals with edited values
            for col in editable_cols:
                st.session_state['fundamentals'].loc[edited_df.index, col] = edited_df[col]
            st.toast("Changes saved! Re-run optimization to apply.", icon="✅")
            st.rerun()
        
        # Export functionality
        from datetime import datetime
        
        def convert_df_to_csv(df: pd.DataFrame) -> bytes:
            return df.to_csv().encode('utf-8')
        
        csv_data = convert_df_to_csv(edited_df)
        filename = f"fundamentals_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        st.download_button(
            label="📥 Export CSV",
            data=csv_data,
            file_name=filename,
            mime="text/csv",
            help="Download your forecasts"
        )
        
        # Margin Interpolation Display
        with st.expander("📈 Margin Trajectory (Linear Interpolation)", expanded=False):
            st.caption(f"Shows the assumed linear progression from Current Margin to Target Margin over {investment_horizon} years.")
            
            # Create interpolation data
            years = list(range(investment_horizon + 1))
            margin_data = []
            
            for ticker in edited_df.index:
                current = edited_df.loc[ticker, "Current Margin"]
                target = edited_df.loc[ticker, "Target Margin"]
                for year in years:
                    # Linear interpolation: M_t = M_0 + (M_N - M_0) * (t / N)
                    margin_t = current + (target - current) * (year / investment_horizon)
                    margin_data.append({"Ticker": ticker, "Year": year, "Margin": margin_t})
            
            margin_df = pd.DataFrame(margin_data)
            
            # Create chart
            margin_chart = alt.Chart(margin_df).mark_line(point=True).encode(
                x=alt.X("Year:O", title="Year", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Margin:Q", title="Net Margin", axis=alt.Axis(format=".1%")),
                color="Ticker:N",
                tooltip=[
                    "Ticker",
                    "Year",
                    alt.Tooltip("Margin:Q", format=".2%", title="Net Margin")
                ]
            ).properties(height=300).interactive()
            
            st.altair_chart(margin_chart, use_container_width=True)

        # 5. Optimize
        if st.button("Run Optimization"):
            result = optimizer.optimize_portfolio(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix.values,
                risk_free_rate=risk_free_rate,
                max_weight=max_weight,
            )

            if result["success"]:
                st.divider()
                st.header("Optimal Portfolio")

                # Metrics
                m1, m2, m3 = st.columns(3)
                m1.metric("Expected Return", f"{result['return']:.2%}")
                m2.metric("Volatility", f"{result['volatility']:.2%}")
                m3.metric("Sharpe Ratio", f"{result['sharpe']:.2f}")

                # Weights
                # Calculate volatility for each asset in the portfolio
                asset_vols = np.sqrt(np.diag(cov_matrix.values))
                
                # Create comprehensive Portfolio Composition table
                weights_df = pd.DataFrame(
                    {
                        "Ticker": tickers, 
                        "Weight": result["weights"],
                        "Implied Return": expected_returns,
                        "Volatility": asset_vols
                    }
                ).sort_values(by="Weight", ascending=False)

                # Filter out small weights
                weights_df = weights_df[weights_df["Weight"] > 0.001]

                # Portfolio Allocation: Single comprehensive table (Ousterhout: reduce complexity, single source of truth)
                st.subheader("Portfolio Allocation")
                
                # Merge with fundamental data
                # We can use display_df (transposed) or funds_aligned
                # Let's use funds_aligned for raw values and format them
                
                comp_df = weights_df.set_index("Ticker").copy()
                
                # Add fundamental columns
                fund_cols = ["Current Margin", "Target Margin", "Exit PE"]
                for col in fund_cols:
                    if col in funds_aligned.columns:
                        comp_df[col] = funds_aligned.loc[comp_df.index, col]
                
                # Add Current PE if we calculated it
                if "Current PE" in display_df.columns:
                     comp_df["Current PE"] = display_df.loc[comp_df.index, "Current PE"]

                # Rename columns for clarity
                rename_map = {
                    "Target Margin": f"Target Margin (Yr {investment_horizon})",
                    "Exit PE": f"Exit PE (Yr {investment_horizon})"
                }
                comp_df = comp_df.rename(columns=rename_map)
                
                # Format and display
                st.dataframe(
                    comp_df.style.format({
                        "Weight": "{:.2%}",
                        "Implied Return": "{:.2%}",
                        "Volatility": "{:.2%}",
                        "Current Margin": "{:.2%}",
                        f"Target Margin (Yr {investment_horizon})": "{:.2%}",
                        "Current PE": "{:.1f}",
                        f"Exit PE (Yr {investment_horizon})": "{:.1f}"
                    }).background_gradient(cmap="Blues", subset=["Weight"]),
                    width="stretch"
                )

                # --- Efficient Frontier Plot ---
                
                # 1. Calculate Frontier
                frontier_data = optimizer.calculate_efficient_frontier(expected_returns, cov_matrix.values, num_points=50)
                
                # 2. Generate Random Portfolios (Feasible Set)
                # We pass a copy of frontier data to the generator if needed, but the generator doesn't modify it in place
                # so we can just pass it.
                random_portfolios = optimizer.generate_random_portfolios(
                    1500, 
                    expected_returns, 
                    cov_matrix.values,
                    frontier_points=frontier_data
                )


                combined_chart = charts.plot_efficient_frontier(
                    frontier_points=frontier_data,
                    random_portfolios=random_portfolios,
                    optimal_portfolio=result,
                    tickers=tickers,
                    asset_returns=expected_returns,
                    asset_vols=asset_vols
                )
                
                st.altair_chart(combined_chart, width="stretch")

            else:
                st.error(f"Optimization failed: {result['message']}")

    except Exception as e:
        st.error(f"Error processing fundamental data: {e}")
        st.exception(e)

if not price_file:
    st.info("Please upload Price History CSV to begin.")

    with st.expander("See Expected CSV Formats"):
        st.markdown("""
        **Price History CSV:**
        *   Index: Date
        *   Columns: Tickers
        *   Values: Adjusted Close Prices
        
        **Fundamentals CSV:**
        *   Index: Ticker
        *   Columns: `Sales/Share`, `Current Margin`, `Target Margin`, `Adjusted Growth Rate`, `Exit PE`, `Current Price` (Optional)
        """)
