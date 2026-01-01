"""
Hybrid Quantamental Optimizer - Stlite Web Application

Philosophy (Ousterhout):
- Thin presentation layer over deep engine modules
- Single-page interface with progressive disclosure
- Stateful session management for smooth UX
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO

# UI Components (Ousterhout: Information Hiding through modular UI)
from components.uploader import render_file_uploader
from components.optimizer_panel import render_optimizer_controls
from components.results_display import render_results

# Deep modules (Ousterhout: Complex implementation, simple interface)
from src.engine.portfolio_engine import (
    optimize_portfolio,
    target_portfolio,
    generate_cml,
)
from src.engine.risk import RiskModel

# --- Page Configuration ---
st.set_page_config(
    page_title="Hybrid Quantamental Optimizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Custom Styling ---
st.markdown(
    """
<style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1 {
        background: linear-gradient(90deg, #00C853, #00E5FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .stAlert {
        border-radius: 8px;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""",
    unsafe_allow_html=True,
)

# --- Session State Initialization ---
if "optimization_complete" not in st.session_state:
    st.session_state.optimization_complete = False
if "tangency_portfolio" not in st.session_state:
    st.session_state.tangency_portfolio = None
if "universe_data" not in st.session_state:
    st.session_state.universe_data = None

# --- Header ---
st.title("Hybrid Quantamental Optimizer")
st.markdown("> *Modern Portfolio Theory evolved for the fundamental investor*")
st.markdown("---")

# --- Step 1: File Upload ---
st.header("📁 Step 1: Upload Data")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Price History (CSV)**")
    with st.expander("ℹ️ Format Guide"):
        st.code(
            """date,AAPL,GOOG,MSFT
2023-01-31,150.23,105.44,280.50
2023-02-28,152.11,108.22,285.33""",
            language="csv",
        )

    prices_file = st.file_uploader(
        "Upload prices.csv",
        type=["csv"],
        key="prices_upload",
        label_visibility="collapsed",
    )

with col2:
    st.markdown("**Asset Metrics (CSV)**")
    with st.expander("ℹ️ Format Guide"):
        st.code(
            """ticker,expected_return,implied_volatility,min_weight,max_weight
AAPL,0.12,0.25,0.0,1.0
GOOG,0.15,0.28,0.0,1.0""",
            language="csv",
        )

    metrics_file = st.file_uploader(
        "Upload metrics.csv",
        type=["csv"],
        key="metrics_upload",
        label_visibility="collapsed",
    )

# --- Step 2: Optimizer Controls ---
if prices_file and metrics_file:
    st.markdown("---")
    st.header("⚙️ Step 2: Configure Optimization")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        risk_free_rate = (
            st.slider(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=10.0,
                value=4.0,
                step=0.25,
                format="%.2f%%",
            )
            / 100.0
        )

    with col_b:
        risk_model_str = st.selectbox(
            "Risk Model", ["Forward-Looking", "Historical"], index=0
        )
        risk_model = (
            RiskModel.FORWARD_LOOKING
            if risk_model_str == "Forward-Looking"
            else RiskModel.HISTORICAL
        )

    with col_c:
        st.markdown("&nbsp;")  # Spacing
        optimize_button = st.button(
            "🚀 Run Optimization", type="primary", use_container_width=True
        )

    # --- Run Optimization ---
    if optimize_button:
        with st.spinner("Optimizing portfolio..."):
            try:
                # Read uploaded files
                prices_file.seek(0)
                metrics_file.seek(0)

                # Call deep module (single function does it all)
                tangency = optimize_portfolio(
                    price_source=StringIO(prices_file.getvalue().decode("utf-8")),
                    metric_source=StringIO(metrics_file.getvalue().decode("utf-8")),
                    risk_free_rate=risk_free_rate,
                    risk_model=risk_model,
                    annualization_factor=252
                    if risk_model == RiskModel.HISTORICAL
                    else None,
                )

                # Generate efficient frontier
                cml_points = generate_cml(
                    tangency_portfolio=tangency,
                    risk_free_rate=risk_free_rate,
                    num_points=30,
                )

                # Save to session state
                st.session_state.tangency_portfolio = tangency
                st.session_state.cml_points = cml_points
                st.session_state.risk_free_rate = risk_free_rate
                st.session_state.optimization_complete = True

                # Extract data for charts (re-read for metadata)
                metrics_file.seek(0)
                import polars as pl

                metrics_df = pl.read_csv(
                    StringIO(metrics_file.getvalue().decode("utf-8"))
                )

                st.session_state.universe_data = {
                    "tickers": tangency["tickers"],
                    "asset_returns": metrics_df["expected_return"].to_numpy(),
                    "asset_vols": metrics_df["implied_volatility"].to_numpy()
                    if "implied_volatility" in metrics_df.columns
                    else np.zeros(len(tangency["tickers"])),
                }

                st.success("✅ Optimization complete!")
                st.rerun()

            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                st.exception(e)

# --- Step 3: Results Display ---
if st.session_state.optimization_complete:
    st.markdown("---")
    st.header("📊 Step 3: Results & Allocation")

    tangency = st.session_state.tangency_portfolio

    # Display tangency metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Max Sharpe Return", f"{tangency['expected_return']:.2%}")
    with col2:
        st.metric("Max Sharpe Volatility", f"{tangency['volatility']:.2%}")
    with col3:
        st.metric("Sharpe Ratio", f"{tangency['sharpe_ratio']:.2f}")

    st.markdown("---")
    st.subheader("🎯 Target Allocation (Capital Market Line)")

    # Interactive volatility slider
    max_vol = tangency["volatility"]
    default_target = min(0.10, max_vol)

    target_vol = st.slider(
        "Target Portfolio Volatility",
        min_value=0.0,
        max_value=float(max_vol),
        value=float(default_target),
        step=0.005,
        format="%.1f%%",
        help="Slide to navigate the Capital Market Line",
    )

    # Calculate target portfolio
    final_portfolio = target_portfolio(
        tangency_portfolio=tangency,
        target_volatility=target_vol,
        risk_free_rate=st.session_state.risk_free_rate,
    )

    # Display final metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Expected Return", f"{final_portfolio['expected_return']:.2%}")
    with col_b:
        st.metric("Volatility", f"{final_portfolio['volatility']:.2%}")
    with col_c:
        st.metric("Sharpe Ratio", f"{final_portfolio['sharpe_ratio']:.2f}")
    with col_d:
        st.metric("Cash Allocation", f"{final_portfolio['cash_weight']:.2%}")

    # Chart
    st.markdown("---")
    render_results(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=st.session_state.cml_points,
        universe_data=st.session_state.universe_data,
        rf_rate=st.session_state.risk_free_rate,
    )

else:
    # Instructions when no data uploaded
    if not prices_file or not metrics_file:
        st.info("👆 Upload both CSV files to begin optimization")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.9rem;'>"
    "Built with ❤️ using Stlite | "
    "<a href='https://github.com/yourusername/portfolio-optimizer' target='_blank'>View Source</a>"
    "</div>",
    unsafe_allow_html=True,
)
