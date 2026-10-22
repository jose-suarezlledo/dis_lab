"""Factor returns: long-short factor series (Ken French library) and their history.

Factor returns here are *monthly excess returns* of zero-investment portfolios
(e.g. HML = cheap stocks minus expensive stocks). They are not investable as such:
no trading costs, no shorting fees, and frictionless rebalancing. ETFs that "harvest" a factor are long-only, costed versions.

    from engine import data, factors
    f = data.load_french()
    factors.decade_table(f[["HML", "MOM"]])
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["growth", "annualized", "rolling_annualized", "decade_table", "drawdowns", "worst_drawdown"]


def growth(returns: pd.DataFrame | pd.Series, start: float = 1.0) -> pd.DataFrame | pd.Series:
    """Growth of `start` from periodic returns; NaN before each series begins."""
    g = (1 + returns).cumprod() * start
    return g


def annualized(returns: pd.DataFrame | pd.Series, ppy: int = 12) -> pd.Series | float:
    """Geometric annualized return over the non-missing observations of each column."""
    if isinstance(returns, pd.Series):
        r = returns.dropna()
        return float((1 + r).prod() ** (ppy / len(r)) - 1) if len(r) else np.nan
    return returns.apply(lambda c: annualized(c, ppy))


def rolling_annualized(returns: pd.DataFrame | pd.Series, years: int = 10, ppy: int = 12):
    """Rolling geometric annualized return over `years` (full windows only)."""
    n = years * ppy
    logs = np.log1p(returns)
    return np.expm1(logs.rolling(n, min_periods=n).sum() * ppy / n)


def decade_table(returns: pd.DataFrame, ppy: int = 12, min_share: float = 0.5) -> pd.DataFrame:
    """Annualized return of each column in each calendar decade (rows: '1930s'...).

    A decade with less than `min_share` of its observations (e.g. the 1920s from
    July 1926, or the current decade) is labelled with its actual years instead.
    """
    rows = {}
    for dec, block in returns.groupby((returns.index.year // 10) * 10):
        first, last = block.index[0].year, block.index[-1].year
        label = f"{dec}s" if len(block) >= min_share * 10 * ppy and first == dec and last == dec + 9             else f"{first}–{last}"
        rows[label] = annualized(block, ppy)
    return pd.DataFrame(rows).T


def drawdowns(returns: pd.DataFrame | pd.Series):
    """Drawdown path of the growth line (0 at a new high, negative below it)."""
    g = growth(returns)
    return g / g.cummax() - 1


def worst_drawdown(returns: pd.Series) -> dict:
    """Depth, peak, trough and recovery date (None if not yet recovered) of the worst drawdown."""
    r = returns.dropna()
    g = growth(r)
    dd = g / g.cummax() - 1
    trough = dd.idxmin()
    peak = g.loc[:trough].idxmax()
    after = g.loc[trough:]
    rec = after[after >= g.loc[peak]]
    return {
        "depth": float(dd.min()),
        "peak": peak,
        "trough": trough,
        "recovered": rec.index[0] if len(rec) else None,
    }
