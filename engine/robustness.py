"""Robustness tests for a rules-based strategy: is the backtest a result or an accident?

Four questions, each a function:
- parameter_grid:      how do the *neighbours* of the chosen parameters do? (data snooping)
- rebalance_day_luck:  does the result survive trading on another day of the month? (timing luck)
- split_periods:       in-sample vs out-of-sample, or before vs after publication (decay)
- subperiod_table:     calendar-year or episode returns against the benchmarks (regime dependence)

Each takes a *rule*: a function that maps period-end prices to a NAV series, so the same tests
work for any strategy built on `engine.signals` + `engine.backtest` or anything else.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from . import metrics as m

__all__ = ["parameter_grid", "nth_trading_day", "rebalance_day_luck", "split_periods", "subperiod_table"]

Rule = Callable[..., pd.Series]


def parameter_grid(rule: Rule, grid: dict[str, Iterable], rf: float = 0.0, **fixed) -> pd.DataFrame:
    """Run `rule(**params, **fixed)` for every combination in `grid`; one row per combination.

    Columns: the parameters, then cagr, vol, sharpe, max_dd, worst_year. Combinations that
    raise (e.g. not enough history) are reported with NaN metrics rather than stopping the sweep.
    Pivot the result (e.g. `.pivot(index="lookback", columns="top_n", values="sharpe")`) for a heatmap.
    """
    names = list(grid)
    rows = []
    for combo in itertools.product(*[list(grid[k]) for k in names]):
        params = dict(zip(names, combo))
        row = dict(params)
        try:
            nav = rule(**params, **fixed)
            row |= {"cagr": m.cagr(nav), "vol": m.annualized_vol(nav), "sharpe": m.sharpe(nav, rf),
                    "max_dd": m.max_drawdown(nav), "worst_year": m.worst_year(nav), "error": ""}
        except Exception as exc:  # noqa: BLE001 - a failed cell is information, not a crash
            row |= {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan, "max_dd": np.nan, "worst_year": np.nan,
                    "error": type(exc).__name__}
        rows.append(row)
    return pd.DataFrame(rows)


def nth_trading_day(daily: pd.DataFrame, k: int = 0) -> pd.DataFrame:
    """Sample daily prices on the k-th trading day of each month (k=0 first day, k=-1 last day).

    A 'monthly' strategy is usually run on month-ends; running it on another day of the month
    is the same rule on the same data, so any difference in results is luck, not skill.
    Months shorter than k+1 trading days use their last day.
    """
    if daily.empty:
        raise ValueError("daily is empty")
    idx = daily.index
    key = idx.to_period("M")
    picks = []
    for _, group in pd.Series(idx, index=idx).groupby(key):
        g = group.index
        picks.append(g[k] if -len(g) <= k < len(g) else g[-1])
    return daily.loc[pd.DatetimeIndex(picks)]


def rebalance_day_luck(
    daily: pd.DataFrame,
    rule: Callable[[pd.DataFrame], pd.Series],
    days: Iterable[int] = (0, 5, 10, 15, -1),
    rf: float = 0.0,
) -> pd.DataFrame:
    """Run `rule(monthly_prices)` with the month sampled on each of `days`; metrics per variant.

    The last variant (-1 = month-end) is the usual backtest. The spread across rows is the
    part of the result that depends on an arbitrary calendar choice.
    """
    rows = {}
    for k in days:
        monthly = nth_trading_day(daily, k)
        nav = rule(monthly)
        label = "month-end" if k == -1 else f"trading day {k + 1}"
        rows[label] = {"cagr": m.cagr(nav), "vol": m.annualized_vol(nav), "sharpe": m.sharpe(nav, rf),
                       "max_dd": m.max_drawdown(nav), "worst_year": m.worst_year(nav),
                       "start": nav.index[0].date(), "end": nav.index[-1].date()}
    return pd.DataFrame(rows).T


def split_periods(nav: pd.Series, split: str | pd.Timestamp, rf: float = 0.0,
                  labels: tuple[str, str] = ("Before", "After")) -> pd.DataFrame:
    """Metrics before and after a date: in-sample vs hold-out, or pre- vs post-publication.

    The 'after' segment is rebased at the split date so both segments start at a known level.
    """
    s = nav.dropna()
    t = pd.Timestamp(split)
    before = s.loc[:t]
    after = s.loc[t:]
    out = {}
    for lab, seg in ((labels[0], before), (labels[1], after)):
        if len(seg) < 3:
            out[lab] = pd.Series({"start": np.nan, "end": np.nan, "cagr": np.nan, "vol": np.nan, "sharpe": np.nan,
                                  "max_dd": np.nan, "worst_year": np.nan})
            continue
        seg = seg / seg.iloc[0] * 100
        out[lab] = pd.Series({"start": seg.index[0].date(), "end": seg.index[-1].date(), "cagr": m.cagr(seg),
                              "vol": m.annualized_vol(seg), "sharpe": m.sharpe(seg, rf), "max_dd": m.max_drawdown(seg),
                              "worst_year": m.worst_year(seg)})
    return pd.DataFrame(out)


def subperiod_table(navs: dict[str, pd.Series], episodes: dict[str, tuple[str, str]] | None = None) -> pd.DataFrame:
    """Total return of several series over each stress episode (rows), one column per series."""
    return pd.DataFrame({k: m.episode_returns(v, episodes) for k, v in navs.items()})
