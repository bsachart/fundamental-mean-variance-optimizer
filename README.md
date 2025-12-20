# Hybrid Quantamental Optimizer

A portfolio construction tool designed to fix the "rear-view mirror" bias of Modern Portfolio Theory (MPT).

Standard optimizers fail because they rely on historical averages for future returns and historical volatility for future risk. This project explicitly **decouples** the estimation of returns, the modeling of risk, and the mathematical optimization.

This allows the user to inject **Forward-Looking** data—using Market Views for returns and Options Market pricing for risk—into a robust mathematical framework.

---

## The Core Philosophy: Decoupled Architecture

The application is built on three independent engines.

### 1. The Return Engine (Forecasting $\mu$)
We do not rely on past price performance to predict future returns.

#### Currently Supported: View-Based Alpha
*   **Logic:** $\mu_{asset} = \mu_{market} + \delta_{alpha}$
*   **Inputs:**
    *   **Market Baseline:** The general expected return of the equity risk premium (e.g., 8%).
    *   **Asset Delta:** User-defined "Alpha" views (e.g., "I expect NVDA to outperform the market by 5%").
*   **Decoupled Mode:** Users can also provide raw Expected Returns directly, bypassing the View-Based logic.

#### Fundamental Implied CAGR (In Development)
Best for long-term fundamental investors. We derive an **Implied Annual Return** by simulating the business fundamentals over an $N$-year holding period.
*   **Logic:** $ \mu = (P_{exit} / P_{current})^{1/N} - 1 $
*   **Inputs:**
    *   **Sales Growth:** Organic growth adjusted for buybacks/dilution.
    *   **Margin Ramp:** Interpolation from current margins to target maturity margins.
    *   **Exit Valuation:** A realistic Terminal P/E based on the company's maturity profile.

### 2. The Risk Engine (Forecasting $\Sigma$)
Historical covariance is useful for correlation, but it is slow to react to changing volatility regimes. We use a **Hybrid Volatility** model that anchors risk to the options market.

We construct the covariance matrix using a blend of **Implied Volatility (Forward-Looking)** and **Historical Volatility (Mean-Reverting)**, while preserving structural historical correlations.

$$ \sigma_{forecast} = w \cdot IV_{current} + (1 - w) \cdot \sigma_{historical} $$

*   **Implied Volatility (IV):** Derived from current option pricing.
*   **Historical Correlation:** Used to determine how assets move relative to one another.

### 3. The Optimization Engine (The Solver)
Once $\mu$ (Returns) and $\Sigma$ (Risk) are calculated, they are passed to the solver. This engine is agnostic to *how* the inputs were generated; it cares only about the math.

**Key Features:**
*   **Asset-Specific Constraints:** Define directional limits per asset (Long/Short/Both).
*   **Efficient Frontier Calculation:** Visualizes the Optimal Sharpe Ratio portfolio against the feasible region of random portfolios.

---

## Workflow

1.  **Configure Assets:** Select your universe and define directional constraints (Long/Short/Both).
2.  **Generate Returns:** Use the *View-Based Model* (Market + Alpha) to populate the expected return vector.
3.  **Calibrate Risk:** Ingest current Implied Volatility data to construct the Hybrid Covariance Matrix.
4.  **Optimize:** The Streamlit interface solves for the Maximum Sharpe Ratio weights and visualizes the Efficient Frontier.

---

## Disclaimer

*This tool is for educational and research purposes only. It does not constitute financial advice. The "View-Based Alpha" and "Hybrid Risk" models are theoretical frameworks and do not guarantee future performance.*