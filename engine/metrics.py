"""Performance and risk metrics: the single source of truth for every number in DIS.

Definitions follow root CLAUDE.md section 7. Every function takes a price / NAV
*level* series (not returns) unless its name says otherwise, so that drawdowns
are always measured on the NAV path and CAGR always uses calendar time.

Conventions
-----------
- Returns are simple close-to-close returns of the level series given
  (use adjusted / total-return closes for ETFs and funds).
- CAGR  = (End / Start) ** (1 / years) - 1, years = calendar days / 365.25.
- Annualized volatility = std(periodic returns, ddof=1) * sqrt(periods per year).
- "Annualized return" in Sharpe, Sortino and Calmar is the CAGR.
- Sharpe  = (CAGR - rf) / annualized volatility. rf is an annual rate, always reported.
- Sortino = (CAGR - rf) / downside deviation, where downside deviation uses
  min(r - target, 0) over *all* periods (not only the negative ones), target = rf
  per period, annualized with sqrt(periods per year).
- Max drawdown = min(NAV / running peak - 1), measured on the NAV path.
- Time under water = longest stretch, in calendar days, from a peak until the NAV
  gets back to that peak (or until the last date, if it has not recovered yet).
- Calmar = CAGR / |Max drawdown|.

Missing data is never turned into a 0% return: leading NaNs (before inception)
are dropped per series, and comparisons are made on the common date range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
TRADING_DAYS = 252
_EPS = 1e-12  # a volatility below this is treated as zero (ratio reported as NaN)

__all__ = [
    "periods_per_year",
    "to_returns",
    "years_between",
    "cagr",
    "annualized_vol",
    "sharpe",
    "downside_deviation",
    "sortino",
    "drawdown_series",
    "max_drawdown",
    "time_under_water",
    "calmar",
    "calendar_year_returns",
    "pct_positive_years",
    "worst_year",
    "beta",
    "alpha",
    "tracking_error",
    "information_ratio",
    "correlation",
    "capture_ratios",
    "turnover",
    "rolling_cagr",
    "rolling_vol",
    "align",
    "summary",
    "summary_table",
    "METRIC_LABELS",
]


# --------------------------------------------------------------------------- helpers


def _clean(nav: pd.Series) -> pd.Series:
    """Drop NaNs (pre-inception and gaps) and check the series is usable."""
    if not isinstance(nav, pd.Series):
        raise TypeError("expected a pandas Series of price / NAV levels")
    s = nav.dropna().astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("the series index must be a DatetimeIndex")
    if not s.index.is_monotonic_increasing:
        s = s.sort_index()
    if len(s) < 2:
        raise ValueError(f"need at least 2 observations, got {len(s)}")
    if (s <= 0).any():
        raise ValueError("price / NAV levels must be strictly positive")
    return s


def periods_per_year(index: pd.DatetimeIndex) -> int:
    """Infer the sampling frequency from the median gap between dates.

    Daily (trading days) -> 252, weekly -> 52, monthly -> 12, quarterly -> 4,
    yearly -> 1. Calendar-daily data (weekends included) -> 365.
    """
    if len(index) < 3:
        raise ValueError("need at least 3 dates to infer the frequency")
    gap = pd.Series(index).diff().dt.days.median()
    if gap <= 1:
        # Distinguish trading-day data from calendar-day data by weekend presence.
        has_weekends = (pd.DatetimeIndex(index).dayofweek >= 5).mean() > 0.2
        return 365 if has_weekends else TRADING_DAYS
    if gap <= 4:
        return TRADING_DAYS
    if gap <= 10:
        return 52
    if gap <= 45:
        return 12
    if gap <= 120:
        return 4
    return 1


def to_returns(nav: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Simple periodic returns from levels. NaN prices give NaN returns, never 0."""
    return nav.pct_change(fill_method=None).iloc[1:]


def years_between(start: pd.Timestamp, end: pd.Timestamp) -> float:
    return (pd.Timestamp(end) - pd.Timestamp(start)).days / DAYS_PER_YEAR


def _ppy(s: pd.Series, ppy: int | None) -> int:
    return ppy if ppy is not None else periods_per_year(s.index)


def _rf_per_period(rf: float, ppy: int) -> float:
    return (1 + rf) ** (1 / ppy) - 1


# --------------------------------------------------------------------------- return & risk


def cagr(nav: pd.Series) -> float:
    """Compound annual growth rate, using calendar days / 365.25."""
    s = _clean(nav)
    years = years_between(s.index[0], s.index[-1])
    if years <= 0:
        raise ValueError("series spans zero calendar days")
    return (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1


def annualized_vol(nav: pd.Series, ppy: int | None = None) -> float:
    s = _clean(nav)
    r = to_returns(s)
    return r.std(ddof=1) * np.sqrt(_ppy(s, ppy))


def sharpe(nav: pd.Series, rf: float = 0.0, ppy: int | None = None) -> float:
    """(CAGR - rf) / annualized volatility. rf is an annual rate (0.02 = 2%)."""
    vol = annualized_vol(nav, ppy)
    return np.nan if vol < _EPS else (cagr(nav) - rf) / vol


def downside_deviation(nav: pd.Series, target: float = 0.0, ppy: int | None = None) -> float:
    """Annualized downside deviation below an annual target rate, over all periods."""
    s = _clean(nav)
    p = _ppy(s, ppy)
    r = to_returns(s)
    shortfall = np.minimum(r - _rf_per_period(target, p), 0.0)
    return float(np.sqrt((shortfall**2).mean()) * np.sqrt(p))


def sortino(nav: pd.Series, rf: float = 0.0, ppy: int | None = None) -> float:
    dd = downside_deviation(nav, target=rf, ppy=ppy)
    return np.nan if dd < _EPS else (cagr(nav) - rf) / dd


def drawdown_series(nav: pd.Series) -> pd.Series:
    """NAV / running peak - 1 (0 at new highs, negative under water)."""
    s = _clean(nav)
    return s / s.cummax() - 1


def max_drawdown(nav: pd.Series) -> float:
    """Worst peak-to-trough loss on the NAV path, as a negative number."""
    return float(drawdown_series(nav).min())


@dataclass(frozen=True)
class UnderWater:
    days: int  # calendar days of the longest stretch below a previous peak
    start: pd.Timestamp | None  # date of the peak that started it
    end: pd.Timestamp | None  # recovery date (or last date, if ongoing)
    recovered: bool


def time_under_water(nav: pd.Series) -> UnderWater:
    """Longest stretch, in calendar days, spent below a previous peak."""
    s = _clean(nav)
    peak = s.cummax()
    under = s < peak
    best = UnderWater(0, None, None, True)
    if not under.any():
        return best
    # Each under-water episode starts at the last peak date before it.
    episode_id = (~under).cumsum()
    for _, grp in s[under].groupby(episode_id[under]):
        peak_date = s.index[s.index.get_loc(grp.index[0]) - 1]
        after = s.index[s.index > grp.index[-1]]
        recovered = len(after) > 0
        end_date = after[0] if recovered else s.index[-1]
        days = (end_date - peak_date).days
        if days > best.days:
            best = UnderWater(days, peak_date, end_date, recovered)
    return best


def calmar(nav: pd.Series) -> float:
    mdd = max_drawdown(nav)
    return np.nan if mdd == 0 else cagr(nav) / abs(mdd)


# --------------------------------------------------------------------------- calendar years


def calendar_year_returns(nav: pd.Series) -> pd.DataFrame:
    """Return per calendar year, with a flag for partial years.

    Year Y uses the last level of Y over the last level of Y-1. The first year
    starts from the first available level instead, and is partial unless the
    series starts in the last week of the previous year. The last year is partial
    if the series ends before 24 December.
    """
    s = _clean(nav)
    year_end = s.groupby(s.index.year).last()
    base = year_end.shift(1)
    base.iloc[0] = s.iloc[0]
    out = pd.DataFrame({"return": year_end / base - 1, "partial": False})
    out.index.name = "year"

    def last_week(d: pd.Timestamp) -> bool:
        return d.month == 12 and d.day >= 24

    if last_week(s.index[0]):
        out = out.iloc[1:]  # a few-day stub; its year-end level is the base of the next year
    else:
        out.iloc[0, out.columns.get_loc("partial")] = True
    if len(out) and not last_week(s.index[-1]):
        out.iloc[-1, out.columns.get_loc("partial")] = True
    return out


def pct_positive_years(nav: pd.Series) -> float:
    """Share of *full* calendar years with a positive return (NaN if none)."""
    yr = calendar_year_returns(nav)
    full = yr.loc[~yr["partial"], "return"]
    return np.nan if full.empty else float((full > 0).mean())


def worst_year(nav: pd.Series) -> float:
    """Worst *full* calendar-year return (NaN if none)."""
    yr = calendar_year_returns(nav)
    full = yr.loc[~yr["partial"], "return"]
    return np.nan if full.empty else float(full.min())


# --------------------------------------------------------------------------- vs benchmark


def align(*navs: pd.Series, how: str = "inner") -> pd.DataFrame:
    """Join level series on their dates and keep the common range only.

    Dates are matched by index (never by row position). With how="inner" a date
    is kept only if every series has a price, so a holiday in one market turns
    into a multi-day return in the next row instead of a fake 0% return.
    """
    df = pd.concat(navs, axis=1, join=how)
    return df.dropna(how="any")


def _paired_returns(nav: pd.Series, bench: pd.Series) -> tuple[pd.Series, pd.Series, int]:
    both = align(_clean(nav).rename("p"), _clean(bench).rename("b"))
    if len(both) < 3:
        raise ValueError("series and benchmark share fewer than 3 dates")
    r = to_returns(both)
    return r["p"], r["b"], periods_per_year(both.index)


def beta(nav: pd.Series, bench: pd.Series) -> float:
    rp, rb, _ = _paired_returns(nav, bench)
    return float(rp.cov(rb) / rb.var())


def alpha(nav: pd.Series, bench: pd.Series, rf: float = 0.0) -> float:
    """Jensen's alpha, annualized: (mean excess - beta * mean bench excess) * ppy."""
    rp, rb, p = _paired_returns(nav, bench)
    rfp = _rf_per_period(rf, p)
    b = rp.cov(rb) / rb.var()
    return float(((rp - rfp).mean() - b * (rb - rfp).mean()) * p)


def tracking_error(nav: pd.Series, bench: pd.Series) -> float:
    rp, rb, p = _paired_returns(nav, bench)
    return float((rp - rb).std(ddof=1) * np.sqrt(p))


def information_ratio(nav: pd.Series, bench: pd.Series) -> float:
    """(CAGR - benchmark CAGR) / tracking error, both on the common date range."""
    both = align(_clean(nav).rename("p"), _clean(bench).rename("b"))
    te = tracking_error(both["p"], both["b"])
    return np.nan if te < _EPS else (cagr(both["p"]) - cagr(both["b"])) / te


def correlation(nav: pd.Series, bench: pd.Series) -> float:
    """Correlation of periodic *returns* (never of price levels)."""
    rp, rb, _ = _paired_returns(nav, bench)
    return float(rp.corr(rb))


def capture_ratios(nav: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """(upside capture, downside capture): mean return in benchmark up / down
    periods divided by the benchmark's mean return in those same periods."""
    rp, rb, _ = _paired_returns(nav, bench)
    up, down = rb > 0, rb < 0
    up_cap = rp[up].mean() / rb[up].mean() if up.any() else np.nan
    down_cap = rp[down].mean() / rb[down].mean() if down.any() else np.nan
    return float(up_cap), float(down_cap)


# --------------------------------------------------------------------------- implementation


def turnover(weights: pd.DataFrame) -> float:
    """Average annual one-way turnover from a weights history.

    One-way turnover at each rebalance = 0.5 * sum(|w_t - w_{t-1}|), so that
    selling 10% of A to buy 10% of B counts as 10%. The initial purchase is not
    counted. Annualized by calendar years spanned.
    """
    w = weights.sort_index().fillna(0.0)
    if len(w) < 2:
        return 0.0
    per_rebalance = 0.5 * w.diff().abs().sum(axis=1).iloc[1:]
    years = years_between(w.index[0], w.index[-1])
    return float(per_rebalance.sum() / years) if years > 0 else np.nan


# --------------------------------------------------------------------------- rolling


def rolling_cagr(nav: pd.Series, years: float = 3.0) -> pd.Series:
    """CAGR over a trailing window of `years` calendar years, at each date."""
    s = _clean(nav)
    window = pd.Timedelta(days=round(years * DAYS_PER_YEAR))
    idx = s.index
    j = idx.searchsorted(idx - window)  # first date inside the trailing window
    span = (idx - idx[j]).days.to_numpy() / DAYS_PER_YEAR
    values = s.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (values / values[j]) ** (1 / span) - 1
    enough_history = idx - window >= idx[0]
    return pd.Series(out[enough_history], index=idx[enough_history], name=nav.name)


def rolling_vol(nav: pd.Series, years: float = 3.0, ppy: int | None = None) -> pd.Series:
    s = _clean(nav)
    p = _ppy(s, ppy)
    window = max(int(round(years * p)), 2)
    return to_returns(s).rolling(window).std(ddof=1).dropna() * np.sqrt(p)


# --------------------------------------------------------------------------- summary


METRIC_LABELS = {
    "start": "Start date",
    "end": "End date",
    "years": "Years",
    "cagr": "CAGR",
    "vol": "Volatility (ann.)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_dd": "Max drawdown",
    "tuw_days": "Longest time under water (days)",
    "calmar": "Calmar",
    "pct_pos_years": "% positive years",
    "worst_year": "Worst calendar year",
    "beta": "Beta",
    "alpha": "Alpha (ann.)",
    "correlation": "Correlation",
    "tracking_error": "Tracking error",
    "info_ratio": "Information ratio",
    "up_capture": "Upside capture",
    "down_capture": "Downside capture",
    "rf": "Risk-free rate used",
}


def summary(
    nav: pd.Series,
    bench: pd.Series | None = None,
    rf: float = 0.0,
    ppy: int | None = None,
) -> pd.Series:
    """All section 7 metrics for one series (plus benchmark-relative ones)."""
    s = _clean(nav)
    tuw = time_under_water(s)
    out = {
        "start": s.index[0].date(),
        "end": s.index[-1].date(),
        "years": years_between(s.index[0], s.index[-1]),
        "cagr": cagr(s),
        "vol": annualized_vol(s, ppy),
        "sharpe": sharpe(s, rf, ppy),
        "sortino": sortino(s, rf, ppy),
        "max_dd": max_drawdown(s),
        "tuw_days": tuw.days,
        "calmar": calmar(s),
        "pct_pos_years": pct_positive_years(s),
        "worst_year": worst_year(s),
    }
    if bench is not None:
        up, down = capture_ratios(s, bench)
        out |= {
            "beta": beta(s, bench),
            "alpha": alpha(s, bench, rf),
            "correlation": correlation(s, bench),
            "tracking_error": tracking_error(s, bench),
            "info_ratio": information_ratio(s, bench),
            "up_capture": up,
            "down_capture": down,
        }
    out["rf"] = rf
    return pd.Series(out, name=nav.name)


def summary_table(
    prices: pd.DataFrame,
    benchmark: str | None = None,
    rf: float = 0.0,
    common_window: bool = True,
) -> pd.DataFrame:
    """Metrics for every column of a wide price table, one column per asset.

    With common_window=True (the default) all series are first cut to the dates
    they all share, so that the comparison is like for like.
    """
    df = align(*[prices[c] for c in prices.columns]) if common_window else prices
    bench = df[benchmark] if benchmark is not None else None
    cols = {c: summary(df[c], None if c == benchmark else bench, rf) for c in df.columns}
    return pd.DataFrame(cols)
