"""
Results Display Component - Unified Data Model
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Dict, List, Any


def render_results(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    rf_rate: float,
):
    """Render the chart and allocation table."""
    chart = _create_chart(tangency, final_portfolio, cml_points, rf_rate)
    st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Final Allocation Breakdown")

    alloc_df = _create_allocation_df(final_portfolio, tangency)
    st.dataframe(
        alloc_df.style.format(
            {"Weight": "{:.2%}", "Expected Return": "{:.2%}", "Volatility": "{:.2%}"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    risky_pct = (1 - _safe_float(final_portfolio.get("cash_weight"))) * 100
    cash_pct = _safe_float(final_portfolio.get("cash_weight")) * 100
    st.info(
        f"💡 **Allocation Strategy**: {risky_pct:.1f}% in optimized risky portfolio, "
        f"{cash_pct:.1f}% in cash to achieve {_safe_float(final_portfolio.get('volatility')):.1%} target volatility."
    )

    # Download Results Button
    csv_data = _create_results_csv(final_portfolio, tangency)
    st.download_button(
        label="📥 Download Results (CSV)",
        data=csv_data,
        file_name="portfolio_optimization_results.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _safe_float(val: Any) -> float:
    """Convert to native Python float, defaulting to 0.0 on error."""
    try:
        if val is None:
            return 0.0
        if hasattr(val, "item"):
            val = val.item()
        f = float(val)
        return f if np.isfinite(f) else 0.0
    except (TypeError, ValueError, OverflowError):
        return 0.0


def _create_chart(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    rf_rate: float,
) -> alt.Chart:
    """
    Create chart using a unified data model.

    Design: One DataFrame, two mark layers (lines + points).
    The 'Category' column drives color encoding and auto-generates the legend.
    """
    t_vol = _safe_float(tangency.get("volatility"))
    t_ret = _safe_float(tangency.get("expected_return"))
    rf = _safe_float(rf_rate)

    # --- BUILD UNIFIED DATA ---
    rows = []

    # 1. Efficient Frontier (line)
    for p in cml_points:
        rows.append(
            {
                "x": _safe_float(p.get("volatility")),
                "y": _safe_float(p.get("expected_return")),
                "Category": "Efficient Frontier",
                "MarkType": "line",
                "Label": "",
            }
        )

    # 2. CML - Capital Market Line (line from RF to Tangency)
    rows.append(
        {"x": 0.0, "y": rf, "Category": "CML", "MarkType": "line", "Label": "Risk Free"}
    )
    rows.append(
        {
            "x": t_vol,
            "y": t_ret,
            "Category": "CML",
            "MarkType": "line",
            "Label": "Tangency",
        }
    )

    # 3. Individual Assets (points)
    tickers = tangency.get("tickers", [])
    asset_rets = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])

    for i, ticker in enumerate(tickers):
        if i < len(asset_vols) and i < len(asset_rets):
            rows.append(
                {
                    "x": _safe_float(asset_vols[i]),
                    "y": _safe_float(asset_rets[i]),
                    "Category": "Assets",
                    "MarkType": "point",
                    "Label": str(ticker),
                }
            )

    # 4. Max Sharpe Portfolio (special point)
    rows.append(
        {
            "x": t_vol,
            "y": t_ret,
            "Category": "Max Sharpe",
            "MarkType": "point",
            "Label": "Max Sharpe",
        }
    )

    # 5. Target Portfolio (special point)
    rows.append(
        {
            "x": _safe_float(final_portfolio.get("volatility")),
            "y": _safe_float(final_portfolio.get("expected_return")),
            "Category": "Target Portfolio",
            "MarkType": "point",
            "Label": "Target",
        }
    )

    df = pd.DataFrame(rows)

    # Ensure no empty DataFrame
    if df.empty:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_point()

    # --- CHART CONFIGURATION ---
    color_scale = alt.Scale(
        domain=[
            "Efficient Frontier",
            "CML",
            "Assets",
            "Max Sharpe",
            "Target Portfolio",
        ],
        range=["#00E676", "#FFC107", "#00BCD4", "#FF5252", "#FFEB3B"],
    )

    x_axis = alt.X(
        "x:Q",
        title="Volatility (Risk)",
        axis=alt.Axis(format="%"),
    )
    y_axis = alt.Y("y:Q", title="Expected Return", axis=alt.Axis(format="%"))

    # --- LAYER 1: LINES ---
    df_lines = df[df["MarkType"] == "line"].copy()

    # Frontier line
    df_frontier = df_lines[df_lines["Category"] == "Efficient Frontier"]
    frontier = (
        alt.Chart(df_frontier)
        .mark_line(size=3)
        .encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color(
                "Category:N", scale=color_scale, legend=alt.Legend(title="Legend")
            ),
            order="x:Q",
        )
    )

    # CML line (dashed)
    df_cml = df_lines[df_lines["Category"] == "CML"]
    cml = (
        alt.Chart(df_cml)
        .mark_line(size=2, strokeDash=[5, 5])
        .encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color("Category:N", scale=color_scale),
            order="x:Q",
        )
    )

    # --- LAYER 2: POINTS ---
    df_points = df[df["MarkType"] == "point"].copy()

    # Assets (circles)
    df_assets = df_points[df_points["Category"] == "Assets"]
    assets = (
        alt.Chart(df_assets)
        .mark_circle(size=100, opacity=0.8)
        .encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color("Category:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("Label:N", title="Ticker"),
                alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                alt.Tooltip("y:Q", format=".2%", title="Return"),
            ],
        )
    )

    # Max Sharpe (star)
    df_sharpe = df_points[df_points["Category"] == "Max Sharpe"]
    sharpe = (
        alt.Chart(df_sharpe)
        .mark_point(shape="cross", size=200, filled=True)
        .encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color("Category:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("Label:N", title="Portfolio"),
                alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                alt.Tooltip("y:Q", format=".2%", title="Return"),
            ],
        )
    )

    # Target Portfolio (diamond)
    df_target = df_points[df_points["Category"] == "Target Portfolio"]
    target = (
        alt.Chart(df_target)
        .mark_point(shape="diamond", size=200, filled=True)
        .encode(
            x=x_axis,
            y=y_axis,
            color=alt.Color("Category:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("Label:N", title="Portfolio"),
                alt.Tooltip("x:Q", format=".2%", title="Volatility"),
                alt.Tooltip("y:Q", format=".2%", title="Return"),
            ],
        )
    )

    # Reference lines at origin (visible when panning)
    x_zero = (
        alt.Chart(pd.DataFrame({"x": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )
    y_zero = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(y="y:Q")
    )

    return (
        (x_zero + y_zero + frontier + cml + assets + sharpe + target)
        .properties(title="Efficient Frontier & Capital Market Line", height=500)
        .interactive()
    )


def _create_allocation_df(
    final_portfolio: Dict, universe_context: Dict
) -> pd.DataFrame:
    """Create allocation table DataFrame."""
    tickers = universe_context.get("tickers", [])
    weights = final_portfolio.get("weights", [])
    returns = universe_context.get("asset_returns", [])
    vols = universe_context.get("asset_vols", [])

    if weights is None or len(weights) == 0:
        return pd.DataFrame()

    rows = []

    cash_w = _safe_float(final_portfolio.get("cash_weight"))
    if cash_w > 0.0001:
        rows.append(
            {
                "Asset": "CASH",
                "Weight": cash_w,
                "Expected Return": 0.0,
                "Volatility": 0.0,
            }
        )

    for i, ticker in enumerate(tickers):
        w = _safe_float(weights[i])
        if abs(w) > 0.0001:
            rows.append(
                {
                    "Asset": str(ticker),
                    "Weight": w,
                    "Expected Return": _safe_float(returns[i])
                    if i < len(returns)
                    else 0.0,
                    "Volatility": _safe_float(vols[i]) if i < len(vols) else 0.0,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Weight", ascending=False, key=abs)
    return df


def _create_results_csv(final_portfolio: Dict, tangency: Dict) -> str:
    """Generate a CSV summary of optimization results."""
    import io
    import csv

    tickers = tangency.get("tickers", [])
    rows = []

    # Header info
    rows.append(["Optimization Results"])
    rows.append(
        ["Target Volatility", f"{_safe_float(final_portfolio.get('volatility')):.2%}"]
    )
    rows.append([])

    # Portfolio metrics
    rows.append(["Portfolio Metrics"])
    rows.append(
        [
            "Expected Return",
            f"{_safe_float(final_portfolio.get('expected_return')):.2%}",
        ]
    )
    rows.append(["Volatility", f"{_safe_float(final_portfolio.get('volatility')):.2%}"])
    rows.append(
        ["Sharpe Ratio", f"{_safe_float(final_portfolio.get('sharpe_ratio')):.2f}"]
    )
    rows.append(
        ["Cash Weight", f"{_safe_float(final_portfolio.get('cash_weight')):.2%}"]
    )
    rows.append([])

    # Allocations
    rows.append(["Asset", "Weight", "Expected Return", "Volatility"])

    cash_w = _safe_float(final_portfolio.get("cash_weight"))
    if cash_w > 0.0001:
        rows.append(["CASH", f"{cash_w:.4f}", "0.0000", "0.0000"])

    weights = final_portfolio.get("weights", [])
    asset_returns = tangency.get("asset_returns", [])
    asset_vols = tangency.get("asset_vols", [])

    for i, ticker in enumerate(tickers):
        w = _safe_float(weights[i]) if i < len(weights) else 0.0
        if abs(w) > 0.0001:
            ret = _safe_float(asset_returns[i]) if i < len(asset_returns) else 0.0
            vol = _safe_float(asset_vols[i]) if i < len(asset_vols) else 0.0
            rows.append([ticker, f"{w:.4f}", f"{ret:.4f}", f"{vol:.4f}"])

    # Convert to CSV string
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerows(rows)
    return output.getvalue()
