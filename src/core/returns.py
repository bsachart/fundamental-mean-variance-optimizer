"""
Module for calculating Implied Annual Returns (CAGR) using a fundamental "Total Reinvestment" model.

Philosophy:
    - Historical returns are poor predictors of the future.
    - We use fundamental forecasts (Sales Growth, Margin Expansion, Exit Multiples) to derive
      an implied annualized return.
    - We assume a "Total Reinvestment" model: all cash flows (dividends, buybacks) are
      reinvested into the company to fuel growth, or equivalently, the investor's return
      is derived solely from the difference between Entry Price and Exit Price.

Reference:
    - "A Philosophy of Software Design": Deep modules, simple interfaces.
"""

import numpy as np
import numpy.typing as npt


def calculate_implied_cagr(
    current_price: float | npt.NDArray[np.float64],
    sales_per_share: float | npt.NDArray[np.float64],
    net_margin_current: float | npt.NDArray[np.float64],
    net_margin_target: float | npt.NDArray[np.float64],
    adjusted_growth_rate: float | npt.NDArray[np.float64],
    exit_pe: float | npt.NDArray[np.float64],
    years: int = 5,
) -> float | npt.NDArray[np.float64]:
    """
    Calculates the Implied Compound Annual Growth Rate (CAGR) for an asset.

    The model projects the future stock price based on fundamental drivers:
    1. Sales Growth: Sales per share grows at `adjusted_growth_rate`.
    2. Margin Expansion: Net margin interpolates from `net_margin_current` to `net_margin_target`.
    3. Exit Valuation: Applies `exit_pe` to the projected Earnings Per Share (EPS) in year `years`.

    Formula:
        Future EPS = (Sales_0 * (1 + g)^N) * Margin_N
        Exit Price = Future EPS * Exit PE
        CAGR = (Exit Price / Current Price)^(1/N) - 1

    Args:
        current_price: Current stock price (P_0).
        sales_per_share: Current Sales Per Share (S_0).
        net_margin_current: Current Net Profit Margin (decimal, e.g., 0.20 for 20%).
        net_margin_target: Target Net Profit Margin in year N (decimal).
        adjusted_growth_rate: Annualized adjusted growth rate of Sales Per Share (decimal).
                             This is the "Net Effective Growth" which accounts for
                             organic growth + buybacks - dilution + dividend yield reinvestment.
        exit_pe: Assumed Price-to-Earnings ratio at exit (Year N).
        years: Investment horizon in years (N).

    Returns:
        The implied annualized return (CAGR) as a decimal (e.g., 0.15 for 15%).
        Returns -1.0 (or close to it) if the investment goes to zero.

    Raises:
        ValueError: If `years` is <= 0.
        ValueError: If `current_price` is <= 0.
    """
    if years <= 0:
        raise ValueError("Investment horizon (years) must be positive.")

    # Ensure inputs are numpy arrays for vectorized operations if they are lists
    # If scalars are passed, numpy operations will handle them correctly.
    P0 = np.asarray(current_price)
    S0 = np.asarray(sales_per_share)
    M0 = np.asarray(net_margin_current)
    MN = np.asarray(net_margin_target)
    g = np.asarray(adjusted_growth_rate)
    PE_exit = np.asarray(exit_pe)

    if np.any(P0 <= 0):
        raise ValueError("Current price must be positive.")

    # 1. Forecast Sales in Year N
    # S_N = S_0 * (1 + g)^N
    S_N = S0 * np.power(1 + g, years)

    # 2. Determine Margins in Year N
    # We simply use the target margin. The interpolation path doesn't matter for the *final*
    # exit price, only the destination matters for the point-to-point CAGR.
    # (If we were doing DCF, the path would matter).
    M_N = MN

    # 3. Calculate Future EPS
    # EPS_N = S_N * M_N
    EPS_N = S_N * M_N

    # 4. Determine Exit Price
    # P_N = EPS_N * PE_exit
    P_N = EPS_N * PE_exit

    # Handle cases where P_N might be negative (if margins are negative)
    # In a long-only context, max loss is -100%.
    # We'll clip P_N at 0 for the CAGR calculation to avoid complex numbers.
    P_N = np.maximum(P_N, 0.0)

    # 5. Solve for CAGR
    # CAGR = (P_N / P_0)^(1/N) - 1
    # We use a small epsilon for P0 to avoid division by zero if not caught above,
    # though the check P0 <= 0 handles it.

    cagr = np.power(P_N / P0, 1.0 / years) - 1.0

    return cagr
