"""Stylized facts of returns: what real return series look like, versus textbook ones.

Session 1 material, reused whenever a model assumes normal, independent returns.
Every function takes a price / NAV level series, like engine.metrics.

- Fat tails: far more extreme days than a normal distribution with the same mean
  and volatility predicts (`tail_counts`, excess kurtosis in `return_stats`).
- Volatility clustering: returns barely autocorrelate, but their size does
  (`autocorrelation(..., absolute=True)`).
- Volatility drag: the average return is not what you earn; CAGR is lower
  (`return_stats`: arithmetic mean vs CAGR).
- A few days drive the result (`missing_days_effect`).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .metrics import _clean, _ppy, cagr, to_returns, years_between

__all__ = ["return_stats", "normal_tail_probability", "tail_counts", "autocorrelation",
           "missing_days_effect"]


def return_stats(nav: pd.Series, ppy: int | None = None) -> pd.Series:
    """Moments of periodic returns, plus the arithmetic-mean vs CAGR gap."""
    s = _clean(nav)
    r = to_returns(s)
    p = _ppy(s, ppy)
    arith = float(r.mean() * p)
    g = cagr(s)
    return pd.Series(
        {
            "periods": len(r),
            "periods_per_year": p,
            "mean": float(r.mean()),
            "std": float(r.std(ddof=1)),
            "skew": float(r.skew()),
            "excess_kurtosis": float(r.kurt()),  # pandas: Fisher, normal = 0
            "best": float(r.max()),
            "worst": float(r.min()),
            "best_date": r.idxmax().date(),
            "worst_date": r.idxmin().date(),
            "worst_in_sigmas": float((r.min() - r.mean()) / r.std(ddof=1)),
            "arith_mean_ann": arith,
            "cagr": g,
            "vol_drag": arith - g,
        },
        name=nav.name,
    )


def normal_tail_probability(k: float) -> float:
    """P(Z < -k) for a standard normal Z."""
    return 0.5 * math.erfc(k / math.sqrt(2))


def tail_counts(nav: pd.Series, sigmas: tuple[float, ...] = (2, 3, 4, 5)) -> pd.DataFrame:
    """Days with a loss beyond k standard deviations: actual vs normal-predicted.

    Uses the series' own mean and standard deviation. "Expected once every" is
    in years, from the number of periods per year.
    """
    s = _clean(nav)
    r = to_returns(s)
    mu, sd = r.mean(), r.std(ddof=1)
    p = _ppy(s, None)
    rows = []
    for k in sigmas:
        prob = normal_tail_probability(k)
        expected = prob * len(r)
        rows.append(
            {
                "sigmas": k,
                "threshold": mu - k * sd,
                "actual": int((r < mu - k * sd).sum()),
                "normal_expected": expected,
                "normal_once_every_years": (1 / prob) / p,
            }
        )
    return pd.DataFrame(rows).set_index("sigmas")


def autocorrelation(nav: pd.Series, lags: int = 20, absolute: bool = False) -> pd.Series:
    """Autocorrelation of returns (or of |returns|) at lags 1..lags."""
    r = to_returns(_clean(nav))
    if absolute:
        r = r.abs()
    return pd.Series({k: float(r.autocorr(k)) for k in range(1, lags + 1)}, name=nav.name)


def missing_days_effect(nav: pd.Series, n: int = 10) -> pd.Series:
    """CAGR if you had sat in cash (0% return) on the n best / n worst days.

    The classic "don't miss the best days" chart shows only the first number. The
    best and worst days cluster together (often in the same weeks), so missing
    both is the realistic comparison.
    """
    s = _clean(nav)
    r = to_returns(s)
    if len(r) <= 2 * n:
        raise ValueError(f"need more than {2 * n} periods")
    years = years_between(s.index[0], s.index[-1])
    best = r.nlargest(n).index
    worst = r.nsmallest(n).index

    def growth(x: pd.Series) -> float:
        return float(np.prod(1 + x.to_numpy()) ** (1 / years) - 1)

    return pd.Series(
        {
            "all_days": growth(r),
            "missed_best": growth(r.where(~r.index.isin(best), 0.0)),
            "missed_worst": growth(r.where(~r.index.isin(worst), 0.0)),
            "missed_both": growth(r.where(~r.index.isin(best.union(worst)), 0.0)),
        },
        name=nav.name,
    )
