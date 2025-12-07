# Fundamental Mean-Variance Optimizer

A "quantamental" portfolio construction tool that marries classic Mean-Variance Optimization (MVO) with forward-looking fundamental data.

Standard MPT is often broken because it drives forward-looking models using backward-looking return data. This tool fixes that by replacing historical average returns with a fundamentally derived **Implied Annual Return (CAGR)**, while keeping the empirically robust historical covariance matrix for risk management.

It trusts **fundamentals** to tell us where prices will eventually go, and **market prices** to tell us how bumpy the ride will be.

---

## Core Philosophy: The Hybrid Model

We are building a "Fundamental Tie-Broken" optimizer. We use the best predictive source for each half of the MVO equation:

| Component | Standard MPT Input | Our "Fundamental" Input | Why? |
| :--- | :--- | :--- | :--- |
| **Expected Return ($\mu$)** | Historical Average Returns | **Implied CAGR** | Past returns are terrible predictors of future performance. Fundamental growth and exit valuations are more reliable anchors of value. |
| **Risk Matrix ($\Sigma$)** | Historical Price Covariance | **Historical Price Covariance** | Fundamentals don't capture daily market microstructure. Recent price history is the best proxy for near-term correlations and volatility. |

### The Horizon Mismatch (Feature, not bug)
You will notice a deliberate horizon mismatch in our inputs:
*   **Returns ($\mu$)** are based on an $N$-year forward outlook (long-term destination).
*   **Risk ($\Sigma$)** is based on an $M$-year historical lookback (short-term path variance).

This is intentional. We want a portfolio deemed attractive over the long haul ($N$ years), but structured robustly enough to survive the near-term correlation shocks ($M$ years history) required to get there.

---

## Methodology

### 1. The Fundamental Engine (Calculating $\mu$)
We avoid the complexity of discounted cash flow (DCF) models which require guessing dividend payout ratios and tax rates. Instead, we use a **"Total Reinvestment"** model.

We assume all cash generated during the holding period is reinvested to fuel growth. Therefore, the investor's return comes entirely from the **Terminal Exit Value**.

For every asset, we simulate the P&L to find the Exit Price ($P_N$) and solve for the CAGR:

1.  **Forecast Sales ($S_t$):** Grow Sales per Share by the input growth rate $g$.
2.  **Ramp Margins ($M_t$):** Linearly interpolate Net Margins from Current ($M_0$) to Target ($M_N$) to capture efficiency improvements.
3.  **Determine Exit Price:** Apply a realistic Exit PE to the final year's earnings.
    $$ P_N = (S_N \times M_N) \times PE_{\text{exit}} $$
4.  **Solve for $\mu$:**
    $$ \mu = \left( \frac{P_N}{P_0} \right)^{\frac{1}{N}} - 1 $$

### 2. The Risk Engine (Calculating $\Sigma$)
We use standard historical price returns over a lookback window $M$.
*   Compute log returns from monthly/weekly price data.
*   Calculate the covariance matrix.
*   **Crucial:** Annualize the matrix to match the time units of our annualized $\mu$.

### 3. The Optimizer
We feed $\mu$ and $\Sigma$ into a standard quadratic solver (`scipy.optimize`) to find the weight vector $\mathbf{w}$ that maximizes the Sharpe Ratio, subject to standard long-only constraints ($w_i \geq 0$).

---

## Implementation Details

### Tech Stack
*   **Core Logic:** Python (NumPy/Pandas) for vectorized array operations.
*   **Optimization:** `scipy.optimize` (SLSQP method).
*   **Frontend:** Streamlit for rapid parameter adjustment.

### Input Engineering: The Guardrails
MVO is notoriously sensitive to garbage inputs ("garbage in, hyper-leveraged garbage out"). Because our model is simple, the **inputs must be precise**.

We rely on **Input Engineering** to handle complex corporate actions (Dividends, Buybacks, Dilution) without complicating the math.

#### 1. The "Net Effective Growth" Rule
Do not simply input the "Revenue Growth" rate. You must adjust $g$ to reflect the per-share reality:

$$ g_{\text{input}} \approx g_{\text{organic}} + \text{Yield} + \text{Buybacks} - \text{Dilution} $$

| Scenario | Organic Revenue Growth | Adjustments | **Input $g$** |
| :--- | :--- | :--- | :--- |
| **Cash Cow (Coke)** | 3% | +3% Div, +1% Buyback | **7%** |
| **Tech Diluter (SaaS)** | 30% | -3% SBC Dilution | **27%** |
| **Aggressive Cannibal** | 5% | +5% Buyback | **10%** |

#### 2. The Terminal P/E Guardrails
The Exit Price accounts for 100% of the return in this model. Small changes here create wild swings in $\mu$.
**The Golden Rule:** You cannot have high Growth ($g$) AND a high Exit Multiple ($PE_{exit}$) simultaneously.
*   *Why?* A high multiple implies the market expects *future* high growth after Year $N$. If the company is mature enough to be sold, its multiple should contract.

| Scenario Preset | Assumption for Year $N$ | Recommended Exit PE |
| :--- | :--- | :--- |
| **"Distressed/Deep Value"** | Company is currently hated but will survive. | **8x - 10x** |
| **"Mature Cash Cow"** | Zero growth, pure dividend play. | **12x - 15x** |
| **"Standard Compounder"** | Good business, standard growth (10%). | **18x - 20x** |
| **"Hyper Growth"** | *Danger Zone.* Cap this rigorously. | **Max 25x** |

> **Warning:** If your calculated $\mu$ exceeds 25-30% for a large-cap stock, check your Exit PE. You are likely "Double Counting" (assuming high growth continues forever). **When in doubt, fade the multiple toward 20x.**