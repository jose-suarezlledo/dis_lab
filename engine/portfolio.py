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

__all__ = ["REBALANCE_FREQUENCIES", "PortfolioResult", "fixed_weight_portfolio", "cash_from_rate",
           "vol_target_portfolio"]

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


# --------------------------------------------------------------------------- volatility targeting


def cash_from_rate(rate_pct: pd.Series, index: pd.DatetimeIndex, start_value: float = 100.0) -> pd.Series:
    """Value of cash earning a quoted annual rate (in %, e.g. FRED DTB3), on the given dates.

    Each period earns the rate known at the *start* of the period, accrued on calendar
    days / 360 (money-market convention). Gaps in the rate are filled with the last
    published value, which is what an investor would have known.
    """
    rate = rate_pct.sort_index().reindex(rate_pct.index.union(index)).ffill().reindex(index)
    if rate.isna().any():
        raise ValueError("the rate series starts after the first date")
    days = index.to_series().diff().dt.days.fillna(0).to_numpy()
    growth = 1 + rate.shift(1).fillna(0).to_numpy() / 100 * days / 360
    return pd.Series(start_value * np.cumprod(growth), index=index, name="Cash")


def vol_target_portfolio(
    risky: pd.Series,
    cash: pd.Series,
    target_vol: float = 0.10,
    lookback: int = 63,
    max_weight: float = 1.0,
    cost_bps: float = 0.0,
    lag: int = 1,
    start_value: float = 100.0,
    name: str = "Vol target",
) -> PortfolioResult:
    """Hold the risky asset at weight min(target_vol / realized vol, max_weight), the rest in cash.

    Realized vol = standard deviation of the last `lookback` daily returns x sqrt(252),
    measured at the close. With lag=1 (the default and the only honest choice) the weight
    set at today's close earns tomorrow's return. lag=0 uses today's volatility to trade
    today's return: look-ahead bias, kept only to show how much it flatters a backtest.
    Rebalanced daily to the target weight; cost = |change in weight| x cost_bps.
    The strategy starts once the first weight is available (lookback + lag days in).
    """
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if not 0 < max_weight:
        raise ValueError("max_weight must be positive")
    levels = align(risky.rename("risky"), cash.rename("cash"))
    r = levels.pct_change(fill_method=None)
    vol = r["risky"].rolling(lookback, min_periods=lookback).std() * np.sqrt(252)
    signal = (target_vol / vol).clip(upper=max_weight)  # known at the close of each day
    held = signal.shift(lag)  # weight held over each day's return
    first = held.first_valid_index()
    if first is None:
        raise ValueError("not enough data for the lookback")
    held, r = held.loc[first:], r.loc[first:]
    traded = (held - held.shift(1)).abs().fillna(0.0)  # entering at the first weight is free
    cost = traded * cost_bps / 10_000
    ret = held * r["risky"] + (1 - held) * r["cash"] - cost
    ret.iloc[0] = 0.0  # the start date is the base of the NAV
    w = held
    nav = start_value * (1 + ret).cumprod()
    idx = nav.index
    years = years_between(idx[0], idx[-1])
    return PortfolioResult(
        nav=nav.rename(name),
        weights=pd.DataFrame({risky.name or "risky": w, "cash": 1 - w}, index=idx),
        turnover=traded[traded > 0],
        costs=cost[cost > 0],
        annual_turnover=float(traded.sum() / years),
        annual_cost=float(cost.sum() / years),
    )
