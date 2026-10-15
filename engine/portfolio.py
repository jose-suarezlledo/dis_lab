"""Fixed-weight portfolios: target weights, drift between rebalances, trading costs.

This is the simplest backtest there is (no signals), and the seed of the full
backtest engine. Conventions:

- Constituents are compared on their common dates only (joined by date, no filling),
  so a late-starting asset moves the portfolio's start date instead of silently
  earning 0% before it existed.
- Between rebalances the weights drift with prices (buy and hold).
- A rebalance happens on the last common trading day of each month / quarter / year.
  The final date of the sample is never a rebalance (its period is not over).
- Costs are charged on traded value: cost = sum(|target w - drifted w|) * cost_bps / 10,000,
  as a fraction of NAV. `cost_bps` is the one-way cost per unit traded (spread +
  commission + slippage). The initial purchase is free, as in metrics.turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import align, to_returns, years_between

__all__ = ["REBALANCE_FREQUENCIES", "PortfolioResult", "fixed_weight_portfolio"]

# label -> pandas Period alias (None = never rebalance, weights drift forever)
REBALANCE_FREQUENCIES = {"none": None, "monthly": "M", "quarterly": "Q", "yearly": "Y"}


@dataclass(frozen=True)
class PortfolioResult:
    nav: pd.Series  # portfolio value, net of costs, starting at start_value
    weights: pd.DataFrame  # end-of-day weights (after any rebalance that day)
    turnover: pd.Series  # one-way turnover at each rebalance date
    costs: pd.Series  # cost paid at each rebalance, as a fraction of NAV
    annual_turnover: float  # average one-way turnover per calendar year
    annual_cost: float  # average cost per calendar year, as a fraction of NAV


def _rebalance_mask(index: pd.DatetimeIndex, freq: str | None) -> np.ndarray:
    """True on the last date of each period (never on the final date of the sample)."""
    if freq is None:
        return np.zeros(len(index), dtype=bool)
    period = index.to_period(freq)
    mask = np.zeros(len(index), dtype=bool)
    mask[:-1] = period[:-1] != period[1:]
    return mask


def fixed_weight_portfolio(
    prices: pd.DataFrame,
    weights: dict[str, float],
    rebalance: str = "monthly",
    cost_bps: float = 0.0,
    start_value: float = 100.0,
    name: str = "Portfolio",
) -> PortfolioResult:
    """NAV of a portfolio held at fixed target weights, rebalanced periodically.

    weights: {ticker: weight}, long-only, summing to 1 (e.g. {"SPY": 0.6, "AGG": 0.4}).
    rebalance: "none", "monthly", "quarterly" or "yearly".
    """
    if rebalance not in REBALANCE_FREQUENCIES:
        raise ValueError(f"rebalance must be one of {list(REBALANCE_FREQUENCIES)}")
    if not weights:
        raise ValueError("weights is empty")
    missing = [t for t in weights if t not in prices.columns]
    if missing:
        raise KeyError(f"no prices for {missing}")
    w = pd.Series(weights, dtype=float)
    if (w < 0).any():
        raise ValueError("weights must be non-negative (long-only)")
    if abs(w.sum() - 1) > 1e-6:
        raise ValueError(f"weights must sum to 1, got {w.sum():.4f}")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")

    levels = align(*[prices[t] for t in w.index])
    if len(levels) < 3:
        raise ValueError("the constituents share fewer than 3 dates")
    rets = to_returns(levels).to_numpy()
    target = w.to_numpy()
    rebal = _rebalance_mask(levels.index, REBALANCE_FREQUENCIES[rebalance])
    cost_rate = cost_bps / 10_000

    n = len(levels)
    nav = np.empty(n)
    w_hist = np.empty((n, len(target)))
    turn = np.zeros(n)
    cost = np.zeros(n)
    holdings = target * start_value
    nav[0], w_hist[0] = start_value, target
    for t in range(1, n):
        holdings = holdings * (1 + rets[t - 1])
        value = holdings.sum()
        if rebal[t]:
            drifted = holdings / value
            traded = np.abs(target - drifted).sum()
            turn[t] = 0.5 * traded
            cost[t] = traded * cost_rate
            value *= 1 - cost[t]
            holdings = target * value
        nav[t] = value
        w_hist[t] = holdings / value

    idx = levels.index
    years = years_between(idx[0], idx[-1])
    turnover = pd.Series(turn, index=idx)[rebal]
    costs = pd.Series(cost, index=idx)[rebal]
    return PortfolioResult(
        nav=pd.Series(nav, index=idx, name=name),
        weights=pd.DataFrame(w_hist, index=idx, columns=w.index),
        turnover=turnover,
        costs=costs,
        annual_turnover=float(turnover.sum() / years),
        annual_cost=float(costs.sum() / years),
    )
