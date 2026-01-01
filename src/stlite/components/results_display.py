"""
Results Display Component

Philosophy (Ousterhout):
- Single function interface to render complex visualizations
- Hides Altair configuration details
- Provides clean DataFrame formatting
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
    universe_data: Dict,
    rf_rate: float,
):
    """
    Render the complete results section with chart and allocation table.

    Deep Module: Hides all Altair complexity behind a simple interface.
    """
    # Generate chart
    chart = _create_frontier_chart(
        tangency=tangency,
        final_portfolio=final_portfolio,
        cml_points=cml_points,
        universe_data=universe_data,
        rf_rate=rf_rate,
    )

    st.altair_chart(chart, use_container_width=True)

    # Generate allocation table
    st.markdown("---")
    st.subheader("📋 Final Allocation Breakdown")

    alloc_df = _create_allocation_table(
        final_portfolio=final_portfolio, universe_data=universe_data
    )

    st.dataframe(
        alloc_df.style.format(
            {"Weight": "{:.2%}", "Expected Return": "{:.2%}", "Volatility": "{:.2%}"}
        ).background_gradient(cmap="RdYlGn", subset=["Weight"], vmin=-0.2, vmax=0.2),
        use_container_width=True,
        hide_index=True,
    )

    # Summary text
    risky_pct = (1 - final_portfolio["cash_weight"]) * 100
    cash_pct = final_portfolio["cash_weight"] * 100

    st.info(
        f"💡 **Allocation Strategy**: {risky_pct:.1f}% in optimized risky portfolio, "
        f"{cash_pct:.1f}% in cash to achieve {final_portfolio['volatility']:.1%} target volatility."
    )


def _create_frontier_chart(
    tangency: Dict,
    final_portfolio: Dict,
    cml_points: List[Dict],
    universe_data: Dict,
    rf_rate: float,
) -> alt.Chart:
    """Create the efficient frontier visualization."""

    # Prepare data
    frontier_df = pd.DataFrame(
        [
            {
                "Volatility": p["volatility"],
                "Return": p["expected_return"],
                "Type": "Efficient Frontier",
            }
            for p in cml_points
        ]
    )

    # CML line (from RF to Tangency)
    cml_df = pd.DataFrame(
        [
            {"Volatility": 0.0, "Return": rf_rate, "Type": "CML"},
            {
                "Volatility": tangency["volatility"],
                "Return": tangency["expected_return"],
                "Type": "CML",
            },
        ]
    )

    # Individual assets
    assets_df = pd.DataFrame(
        {
            "Volatility": universe_data["asset_vols"],
            "Return": universe_data["asset_returns"],
            "Ticker": universe_data["tickers"],
            "Type": "Assets",
        }
    )

    # Optimal point
    optimal_df = pd.DataFrame(
        [
            {
                "Volatility": tangency["volatility"],
                "Return": tangency["expected_return"],
                "Type": "Max Sharpe",
            }
        ]
    )

    # Target point
    target_df = pd.DataFrame(
        [
            {
                "Volatility": final_portfolio["volatility"],
                "Return": final_portfolio["expected_return"],
                "Type": "Target Portfolio",
            }
        ]
    )

    # Color scale
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

    # Base chart
    base = alt.Chart().encode(
        x=alt.X(
            "Volatility:Q",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(format="%", title="Volatility (Risk)"),
        ),
        y=alt.Y(
            "Return:Q",
            scale=alt.Scale(zero=False),
            axis=alt.Axis(format="%", title="Expected Return"),
        ),
        color=alt.Color("Type:N", scale=color_scale, legend=alt.Legend(title="")),
    )

    # Layers
    frontier_line = base.mark_line(size=3).encode(alt.datum(frontier_df))

    cml_line = base.mark_line(size=2, strokeDash=[5, 5]).encode(alt.datum(cml_df))

    assets_points = base.mark_circle(size=100, opacity=0.8).encode(
        alt.datum(assets_df), tooltip=["Ticker:N", "Return:Q", "Volatility:Q"]
    )

    optimal_star = base.mark_point(shape="star", size=300, filled=True).encode(
        alt.datum(optimal_df), tooltip=["Type:N", "Return:Q", "Volatility:Q"]
    )

    target_diamond = base.mark_point(shape="diamond", size=250, filled=True).encode(
        alt.datum(target_df), tooltip=["Type:N", "Return:Q", "Volatility:Q"]
    )

    # Combine
    chart = (
        alt.layer(cml_line, frontier_line, assets_points, optimal_star, target_diamond)
        .properties(
            width="container",
            height=500,
            title="Efficient Frontier & Capital Market Line",
        )
        .configure_view(strokeWidth=0)
        .configure_title(fontSize=18, anchor="start")
    )

    return chart


def _create_allocation_table(
    final_portfolio: Dict, universe_data: Dict
) -> pd.DataFrame:
    """Create formatted allocation table."""

    tickers = universe_data["tickers"]
    weights = final_portfolio["weights"]
    returns = universe_data["asset_returns"]
    vols = universe_data["asset_vols"]

    # Build rows
    rows = []

    # Add cash if present
    if final_portfolio["cash_weight"] > 0.0001:
        rows.append(
            {
                "Asset": "CASH",
                "Weight": final_portfolio["cash_weight"],
                "Expected Return": 0.0,  # Assuming RF rate is captured elsewhere
                "Volatility": 0.0,
            }
        )

    # Add risky assets
    for i, ticker in enumerate(tickers):
        if abs(weights[i]) > 0.0001:
            rows.append(
                {
                    "Asset": ticker,
                    "Weight": weights[i],
                    "Expected Return": returns[i],
                    "Volatility": vols[i],
                }
            )

    df = pd.DataFrame(rows)
    df = df.sort_values("Weight", ascending=False, key=abs)

    return df
