"""
Module for generating Altair charts for the Portfolio Optimizer.

Philosophy:
    - "Deep Module": Hides the complexity of Altair configuration behind a simple interface.
    - "Information Hiding": The main app doesn't need to know about mark_circle, encode, etc.
    - "Define Errors Out of Existence": Accept raw data types and handle formatting internally.
"""

import altair as alt
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import numpy.typing as npt

def _format_top_holdings(weights: npt.NDArray[np.float64], tickers: List[str]) -> str:
    """Helper to format top holdings string."""
    sorted_idx = np.argsort(weights)[::-1]
    top_3 = sorted_idx[:3]
    return ", ".join([f"{tickers[i]}: {weights[i]:.0%}" for i in top_3 if weights[i] > 0.01])

def _format_composition(weights: npt.NDArray[np.float64], tickers: List[str]) -> str:
    """Helper to format full composition string for tooltips."""
    sorted_indices = np.argsort(weights)[::-1]
    composition_lines = []
    for idx in sorted_indices:
        if weights[idx] > 0.001:
            composition_lines.append(f"{tickers[idx]}: {weights[idx]:.1%}")
    return "\n".join(composition_lines)

def plot_efficient_frontier(
    frontier_points: List[Dict[str, Any]],
    random_portfolios: List[Dict[str, Any]],
    optimal_portfolio: Dict[str, Any],
    tickers: List[str],
    asset_returns: npt.NDArray[np.float64],
    asset_vols: npt.NDArray[np.float64],
) -> alt.VConcatChart:
    """
    Generates the combined Efficient Frontier and Asset Allocation chart.

    Args:
        frontier_points: List of dicts with 'return', 'volatility', 'weights'.
        random_portfolios: List of dicts with 'return', 'volatility', 'weights'.
        optimal_portfolio: Dict with 'return', 'volatility', 'weights', 'sharpe'.
        tickers: List of asset ticker symbols.
        asset_returns: Array of expected returns for each asset.
        asset_vols: Array of volatilities for each asset.

    Returns:
        An Altair VConcatChart containing both plots.
    """
    
    # --- Data Preparation (Internal Logic) ---

    # 1. Prepare Frontier Data
    # We need to process this carefully to avoid modifying the input list in place if it's reused elsewhere,
    # but here we are consuming it.
    frontier_data = []
    transition_data = []
    
    for item in frontier_points:
        ret = item['return']
        vol = item['volatility']
        w = item['weights']
        
        # For Frontier Plot
        frontier_data.append({
            'Return': ret,
            'Volatility': vol,
            'Top Holdings': _format_top_holdings(w, tickers)
        })
        
        # For Transition Map
        comp_str = _format_composition(w, tickers)
        for i, ticker in enumerate(tickers):
            if w[i] > 0.001: # Filter small weights
                transition_data.append({
                    "Volatility": vol,
                    "Return": ret,
                    "Ticker": ticker,
                    "Weight": w[i],
                    "Composition": comp_str
                })

    frontier_df = pd.DataFrame(frontier_data)
    transition_df = pd.DataFrame(transition_data)

    # 2. Prepare Random Portfolios Data
    random_data = []
    for item in random_portfolios:
        random_data.append({
            'Return': item['return'],
            'Volatility': item['volatility'],
            'Top Holdings': _format_top_holdings(item['weights'], tickers)
        })
    random_df = pd.DataFrame(random_data)

    # 3. Prepare Individual Assets Data
    assets_df = pd.DataFrame({
        "Ticker": tickers,
        "Return": asset_returns,
        "Volatility": asset_vols,
        "Type": "Asset",
        "Top Holdings": [f"{t}: 100%" for t in tickers]
    })

    # 4. Prepare Optimal Portfolio Data
    optimal_df = pd.DataFrame([{
        "Return": optimal_portfolio["return"],
        "Volatility": optimal_portfolio["volatility"],
        "Type": "Max Sharpe",
        "Ticker": "Optimal",
        "Top Holdings": _format_top_holdings(optimal_portfolio["weights"], tickers)
    }])

    # --- Chart Generation ---

    # Common Axis Config
    axis_config = {
        "titleFontSize": 14,
        "labelFontSize": 12,
        "titlePadding": 10
    }

    # Chart 1: Efficient Frontier

    # A. Feasible Set (Cloud)
    cloud_chart = alt.Chart(random_df).mark_circle(size=40, color='gray', opacity=0.3).encode(
        x=alt.X('Volatility:Q', axis=alt.Axis(format='%', title='Volatility (Risk)', **axis_config)),
        y=alt.Y('Return:Q', axis=alt.Axis(format='%', title='Expected Return', **axis_config)),
        tooltip=[
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Volatility:Q', format='.2%'), 
            alt.Tooltip('Top Holdings:N')
        ]
    )
    
    # B. Frontier Line
    frontier_chart = alt.Chart(frontier_df).mark_line(color='#00E676', size=4).encode(
        x='Volatility:Q',
        y='Return:Q',
        tooltip=[
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Volatility:Q', format='.2%'), 
            alt.Tooltip('Top Holdings:N')
        ]
    )
    
    # C. Individual Assets
    assets_layer = alt.Chart(assets_df).mark_circle(size=150, color='#00BCD4', opacity=1).encode(
        x='Volatility:Q',
        y='Return:Q',
        tooltip=[
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Volatility:Q', format='.2%'), 
            alt.Tooltip('Top Holdings:N')
        ]
    ) + alt.Chart(assets_df).mark_text(align='left', dx=12, dy=-12, color='white', fontSize=13, fontWeight='bold').encode(
        x='Volatility:Q',
        y='Return:Q',
        text='Ticker'
    )
    
    # D. Optimal Portfolio
    optimal_chart = alt.Chart(optimal_df).mark_point(shape='diamond', size=300, filled=True).encode(
        x='Volatility:Q',
        y='Return:Q',
        color=alt.value('#FF5252'),
        tooltip=[
            alt.Tooltip('Type'), 
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Volatility:Q', format='.2%'), 
            alt.Tooltip('Top Holdings:N')
        ]
    )
    
    frontier_combined = (cloud_chart + frontier_chart + assets_layer + optimal_chart).properties(
        height=500,
        title=alt.TitleParams(text='Efficient Frontier', fontSize=18)
    )
    
    # Chart 2: Transition Map (Asset Allocation)
    
    nearest = alt.selection_point(nearest=True, on='mouseover', fields=['Volatility'], empty=False)

    base = alt.Chart(transition_df).encode(
        x=alt.X('Volatility:Q', axis=alt.Axis(format='%', title='Volatility (Risk)', **axis_config))
    )

    area = base.mark_area().encode(
        y=alt.Y('Weight:Q', stack='normalize', axis=alt.Axis(format='%', title='Allocation', **axis_config)),
        color=alt.Color('Ticker:N', legend=alt.Legend(orient='bottom', columns=8, title=None, labelFontSize=12, symbolSize=150)),
        tooltip=[
            'Ticker',
            alt.Tooltip('Weight:Q', format='.2%'),
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Composition:N', title='Full Portfolio')
        ]
    )

    rule = base.mark_rule(color='white', strokeWidth=2).encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0)),
        tooltip=[
            alt.Tooltip('Return:Q', format='.2%'),
            alt.Tooltip('Volatility:Q', format='.2%'),
            alt.Tooltip('Composition:N', title='Full Portfolio')
        ]
    ).add_params(nearest)

    transition_chart = (area + rule).properties(
        height=350,
        title=alt.TitleParams(text="Asset Allocation vs. Risk", fontSize=18)
    )
    
    combined_chart = alt.vconcat(frontier_combined, transition_chart).resolve_scale(x='shared')
    
    return combined_chart
