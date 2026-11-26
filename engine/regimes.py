"""Market regimes from macro data (Session 5): what was known when, how regimes are labelled, and what they are worth.

Four tools, each answering a question an investor asks every month:
- regime labels (threshold rules) and returns by regime: "which box are we in, and what did assets do in that box?"
- the regime monitor (nearest neighbours): "what happened the last times the economy looked like today?"
- the signal scorecard: "this recession signal just flashed; what is its record?"
- the portfolio weather report: "how does my portfolio behave in each regime and crisis?"
plus a macro filter that switches a Session 4 trend rule to cash when a macro signal says 'risk off'.

Real-time discipline. Macro data is published with a lag and revised. `macro_panel` dates every series at the end of the
month it describes; `known_at` shifts each one by its publication lag, so that row t holds only what an investor could
have read at the close of month t. Thresholds ("above its average") can be computed on the full sample (look-ahead,
kept on purpose to show its effect), an expanding window, or a rolling window.

Long-history assets (monthly, decimal returns): US stocks = Fama-French market (MKT_RF + RF, 1926-), cash = RF,
factors HML/SMB/MOM (1926-) and RMW/CMA (1963-), and 10-year Treasuries approximated from the constant-maturity yield
GS10 (1953-) as carry minus duration × yield change, a standard approximation labelled as such wherever it is shown.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import data
from . import metrics as m

__all__ = ["PUB_LAG", "macro_panel", "known_at", "long_returns", "threshold_regimes", "QUADRANT_NAMES", "returns_by_regime",
           "regime_rotation_targets", "label_agreement", "standardize", "nearest_months", "analog_outcomes",
           "analog_track_record", "SIGNALS", "signal_flags", "signal_scorecard", "LONG_EPISODES", "weather_report",
           "risk_on_filter", "apply_filter", "stock_bond_correlation", "realtime_vs_revised_payrolls"]

# Publication lag in months: the value for month t is first published during month t + lag.
PUB_LAG = {"IP_YOY": 1, "CPI_YOY": 1, "UNRATE": 1, "UNRATE_CHG12": 1, "PAYEMS_YOY": 1, "PAYEMS_YOY_FIRST": 1, "SAHM": 1, "USREC": 12,
           "T10Y2Y": 0, "T10Y3M": 0, "SLOPE": 0, "DGS2": 0, "GS10": 0, "TB3MS": 0, "CREDIT": 0, "BAA10Y": 0}
LABELS = {"IP_YOY": "Industrial production, % y/y", "CPI_YOY": "Consumer prices, % y/y", "UNRATE": "Unemployment rate, %",
          "UNRATE_CHG12": "Unemployment rate, change over 12 months (pts)", "PAYEMS_YOY": "Payrolls, % y/y",
          "SAHM": "Sahm rule indicator (real time)", "T10Y2Y": "10y − 2y Treasury yield (pts)", "T10Y3M": "10y − 3m Treasury yield (pts)",
          "SLOPE": "10y − 3m yield, monthly (pts)", "DGS2": "2-year Treasury yield, %", "GS10": "10-year Treasury yield, %",
          "TB3MS": "3-month T-bill, %", "CREDIT": "Baa corporate − 10y Treasury yield (pts)", "BAA10Y": "Baa − 10y spread, daily (pts)",
          "USREC": "NBER recession (1 = in recession)"}


def _month_end(s: pd.Series, how: str = "last") -> pd.Series:
    s = s.dropna()
    out = s.resample("ME").last() if how == "last" else s.resample("ME").mean()
    return out


def macro_panel(data_dir=None) -> pd.DataFrame:
    """Monthly macro panel, one row per month-end, each value dated at the end of the month it describes (revised data)."""
    f = lambda sid: data.load_fred(sid, data_dir)  # noqa: E731
    out = {}
    ip = _month_end(f("INDPRO"))
    out["IP_YOY"] = 100 * ip.pct_change(12, fill_method=None)
    cpi = _month_end(f("CPIAUCSL"))
    out["CPI_YOY"] = 100 * cpi.pct_change(12, fill_method=None)
    un = _month_end(f("UNRATE"))
    out["UNRATE"] = un
    out["UNRATE_CHG12"] = un - un.shift(12)
    pay = _month_end(f("PAYEMS"))
    out["PAYEMS_YOY"] = 100 * pay.pct_change(12, fill_method=None)
    try:  # payrolls growth as first published (ALFRED initial releases; payrolls are never rebased, so the ratio is meaningful)
        first = _month_end(f("ALFRED_PAYEMS_first"))
        out["PAYEMS_YOY_FIRST"] = 100 * (first / first.shift(12) - 1)
    except KeyError:
        pass
    out["SAHM"] = _month_end(f("SAHMREALTIME"))
    out["USREC"] = _month_end(f("USREC"))
    for sid in ("T10Y2Y", "T10Y3M", "DGS2", "BAA10Y"):
        out[sid] = _month_end(f(sid))
    gs10, tb = _month_end(f("GS10")), _month_end(f("TB3MS"))
    out["GS10"], out["TB3MS"] = gs10, tb
    out["SLOPE"] = gs10 - tb
    out["CREDIT"] = _month_end(f("BAA")) - gs10
    df = pd.DataFrame(out).sort_index()
    df.index.name = "date"
    return df


def known_at(panel: pd.DataFrame, lags: dict[str, int] | None = None) -> pd.DataFrame:
    """Shift each column by its publication lag: row t = what could be read at the close of month t."""
    lags = PUB_LAG if lags is None else lags
    return pd.DataFrame({c: panel[c].shift(lags.get(c, 0)) for c in panel.columns}, index=panel.index)


def long_returns(data_dir=None, start: str = "1953-04-30") -> pd.DataFrame:
    """Monthly decimal returns of long-history assets: US stocks, 10y Treasuries (approx.), cash, factors."""
    fr = data.load_french(data_dir=data_dir)
    fr.index = fr.index + pd.offsets.MonthEnd(0)
    y = _month_end(data.load_fred("GS10", data_dir)) / 100
    duration = 7.5  # approximate modified duration of a 10-year par bond at typical yields
    bond = (y.shift(1) / 12 - duration * (y - y.shift(1))).rename("US 10y Treasuries (approx.)")
    out = pd.DataFrame({
        "US stocks": fr["MKT_RF"] + fr["RF"],
        "US 10y Treasuries (approx.)": bond,
        "Cash (T-bills)": fr["RF"],
        "Value − growth (HML)": fr["HML"],
        "Small − big (SMB)": fr["SMB"],
        "Momentum (MOM)": fr["MOM"],
        "Profitability (RMW)": fr["RMW"],
        "Investment (CMA)": fr["CMA"],
    })
    return out.loc[start:]


# --------------------------------------------------------------------------- threshold regimes

QUADRANT_NAMES = {"A": "Growth up, inflation up", "B": "Growth down, inflation up", "C": "Growth down, inflation down",
                  "D": "Growth up, inflation down"}


def _threshold(x: pd.Series, mode: str, window: int) -> pd.Series:
    if mode == "full":
        return pd.Series(x.mean(), index=x.index)  # look-ahead: uses the whole sample, future included
    # min_periods below the window so that one unpublished month (e.g. US CPI for Oct 2025, never collected) does not
    # blank the average for the next `window` months
    if mode == "expanding":
        return x.expanding(min_periods=window).mean()
    if mode == "rolling":
        return x.rolling(window, min_periods=max(2, int(0.8 * window))).mean()
    raise ValueError("mode must be 'full', 'expanding' or 'rolling'")


def threshold_regimes(growth: pd.Series, inflation: pd.Series, mode: str = "rolling", window: int = 36) -> pd.Series:
    """Label each month A-D by whether growth and inflation are above their average (the 2025 spreadsheet's rule).

    A = both above, B = growth below / inflation above, C = both below, D = growth above / inflation below.
    The series should already be 'known_at' (publication lags applied). NaN until the threshold exists.
    """
    g, i = growth.astype(float), inflation.astype(float)
    tg, ti = _threshold(g, mode, window), _threshold(i, mode, window)
    up_g, up_i = g >= tg, i >= ti
    lab = np.where(up_g & up_i, "A", np.where(~up_g & up_i, "B", np.where(~up_g & ~up_i, "C", "D")))
    valid = g.notna() & i.notna() & tg.notna() & ti.notna()
    return pd.Series(lab, index=g.index).where(valid)


def label_agreement(a: pd.Series, b: pd.Series) -> float:
    both = pd.concat([a, b], axis=1).dropna()
    return float((both.iloc[:, 0] == both.iloc[:, 1]).mean()) if len(both) else np.nan


def returns_by_regime(returns: pd.DataFrame, regimes: pd.Series, next_month: bool = True) -> pd.DataFrame:
    """Annualized mean return, volatility and hit rate of each asset by regime.

    next_month=True (honest): the regime known at the close of month t is matched with the returns of month t+1.
    next_month=False matches it with the same month's returns: what a 'returns by regime' chart usually shows.
    """
    lab = regimes.shift(1) if next_month else regimes
    df = returns.join(lab.rename("_regime"), how="inner").dropna(subset=["_regime"])
    rows = []
    for reg, g in df.groupby("_regime"):
        for col in returns.columns:
            r = g[col].dropna()
            if len(r) < 3:
                continue
            rows.append({"regime": reg, "asset": col, "months": len(r), "ann_return": float((1 + r.mean()) ** 12 - 1),
                         "ann_vol": float(r.std() * np.sqrt(12)), "hit_rate": float((r > 0).mean())})
    return pd.DataFrame(rows)


def regime_rotation_targets(returns: pd.DataFrame, regimes: pd.Series, mode: str = "expanding", min_months: int = 60) -> pd.DataFrame:
    """The 2025 spreadsheet's rotation: in each regime, weights proportional to each asset's average return in that regime.

    mode="full": averages over the whole sample (the spreadsheet; look-ahead). mode="expanding": only months before t.
    Negative averages get zero weight; if all are negative, equal weight. Weights are decided at t (regime of t) and are
    meant to be held over t+1 (signal_backtest with lag=1).
    """
    cols = list(returns.columns)
    lab = regimes.reindex(returns.index)
    out = pd.DataFrame(np.nan, index=returns.index, columns=cols)
    full_means = {reg: returns[lab.shift(1) == reg].mean() for reg in "ABCD"} if mode == "full" else None
    for i, t in enumerate(returns.index):
        reg = lab.iloc[i]
        if pd.isna(reg) or i < min_months:
            continue
        if mode == "full":
            mu = full_means[reg]
        else:
            past_lab = lab.iloc[:i].shift(1)  # regime of s-1 matched with returns of s, for s < t
            mu = returns.iloc[:i][past_lab == reg].mean()
        mu = mu.clip(lower=0)
        w = pd.Series(1 / len(cols), index=cols) if (mu.sum() <= 0 or mu.isna().any()) else mu / mu.sum()
        out.loc[t] = w.to_numpy()
    return out.dropna(how="all")


# --------------------------------------------------------------------------- nearest neighbours (regime monitor)


def standardize(features: pd.DataFrame, expanding: bool = False, min_periods: int = 120) -> pd.DataFrame:
    """z-scores; expanding=True uses only data up to each date (needed for honest backtests)."""
    if expanding:
        mu = features.expanding(min_periods=min_periods).mean()
        sd = features.expanding(min_periods=min_periods).std()
        return (features - mu) / sd
    return (features - features.mean()) / features.std()


def _pick(dist: pd.Series, k: int, gap_months: int) -> list[pd.Timestamp]:
    chosen: list[pd.Timestamp] = []
    for t in dist.sort_values().index:
        if all(abs((t.year - c.year) * 12 + t.month - c.month) > gap_months for c in chosen):
            chosen.append(t)
            if len(chosen) == k:
                break
    return chosen


def nearest_months(z: pd.DataFrame, date: pd.Timestamp, k: int = 10, gap_months: int = 6, horizon: int = 12,
                   weights: dict[str, float] | None = None) -> pd.DataFrame:
    """The k past months most similar to `date` (Euclidean distance on z-scores), at least `gap_months` apart,
    and at least `horizon` months before `date` (so that their outcomes are known). Returns date and distance."""
    z = z.dropna()
    if date not in z.index:
        raise KeyError(f"{pd.Timestamp(date).date()} has incomplete features")
    w = np.ones(z.shape[1]) if weights is None else np.array([weights.get(c, 1.0) for c in z.columns])
    target = z.loc[date].to_numpy()
    cand = z.loc[: pd.Timestamp(date) - pd.offsets.MonthEnd(horizon)]
    if cand.empty:
        raise ValueError("no candidate months before the date")
    dist = pd.Series(np.sqrt((((cand.to_numpy() - target) ** 2) * w).sum(axis=1)), index=cand.index)
    picks = _pick(dist, k, gap_months)
    return pd.DataFrame({"date": picks, "distance": dist.loc[picks].to_numpy()})


def _forward(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Cumulative return over the next `horizon` months, dated at the start (month t -> months t+1..t+h)."""
    g = np.log1p(returns)
    fwd = g.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
    return np.expm1(fwd)


def analog_outcomes(returns: pd.DataFrame, analogs: pd.DataFrame, horizon: int = 12) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward `horizon`-month returns after each analog month, and a summary against all months (the base rate)."""
    fwd = _forward(returns, horizon)
    per = fwd.reindex(analogs["date"]).set_index(analogs["date"])
    summ = pd.DataFrame({
        "analogs: median": per.median(), "analogs: % positive": (per > 0).mean(), "analogs: worst": per.min(),
        "analogs: best": per.max(), "all months: median": fwd.median(), "all months: % positive": (fwd > 0).mean(),
        "analogs with data": per.notna().sum()})
    return per, summ


def analog_track_record(z_expanding: pd.DataFrame, returns: pd.Series, horizon: int = 12, k: int = 10, gap_months: int = 6,
                        start: str = "1975-01-31") -> pd.DataFrame:
    """Honest test of the regime monitor: at each month t, using only data known at t (expanding z-scores; analogs whose
    outcomes were already known), predict the next `horizon`-month return by the analogs' median. Returns prediction and
    outcome per month."""
    fwd = _forward(returns.to_frame(), horizon).iloc[:, 0]
    z = z_expanding.dropna()
    rows = []
    for t in z.loc[start:].index:
        if pd.isna(fwd.get(t)):
            continue
        cand_end = t - pd.offsets.MonthEnd(horizon)
        cand = z.loc[:cand_end]
        if len(cand) < 3 * k:
            continue
        target = z.loc[t].to_numpy()
        dist = pd.Series(np.sqrt(((cand.to_numpy() - target) ** 2).sum(axis=1)), index=cand.index)
        picks = _pick(dist, k, gap_months)
        past_fwd = fwd.reindex(picks)
        rows.append({"date": t, "predicted": float(past_fwd.median()), "realized": float(fwd.loc[t]),
                     "base_rate_so_far": float((fwd.loc[:cand_end] > 0).mean())})
    out = pd.DataFrame(rows).set_index("date")
    return out


# --------------------------------------------------------------------------- signal scorecard

SIGNALS = {
    "curve_10y3m": ("Yield curve inverted (10y − 3m < 0, monthly average)", "SLOPE", "lt", 0.0),
    "curve_10y2y": ("Yield curve inverted (10y − 2y < 0)", "T10Y2Y", "lt", 0.0),
    "sahm": ("Sahm rule (unemployment's 3-month average ≥ 0.5 pts above its 12-month low)", "SAHM", "ge", 0.5),
    "ip_negative": ("Industrial production falling over 12 months", "IP_YOY", "lt", 0.0),
    "credit_wide": ("Credit spread (Baa − 10y) up ≥ 1 pt from its 12-month low", "CREDIT_JUMP", "ge", 1.0),
}


def signal_flags(panel_known: pd.DataFrame) -> pd.DataFrame:
    """Boolean flags (NaN when the input is missing), one column per SIGNALS entry, on data known at each month."""
    p = panel_known.copy()
    p["CREDIT_JUMP"] = p["CREDIT"] - p["CREDIT"].rolling(12, min_periods=12).min()
    out = {}
    for key, (_, col, op, thr) in SIGNALS.items():
        x = p[col]
        flag = (x < thr) if op == "lt" else (x >= thr)
        out[key] = flag.astype(float).where(x.notna())
    return pd.DataFrame(out, index=p.index)


def signal_scorecard(flags: pd.Series, usrec: pd.Series, stocks: pd.Series, within: int = 24, quiet: int = 12) -> pd.DataFrame:
    """Each trigger of a signal (first month on after `quiet` months off): did an NBER recession start within `within`
    months, after how many months, and what did US stocks return over the next 12 months?

    usrec: the NBER recession indicator with today's hindsight (recession dates are announced months later; the
    scorecard is a historical judgment, not a real-time signal).
    """
    f = flags.dropna()
    rec = usrec.reindex(f.index.union(usrec.index)).ffill()
    starts = rec[(rec == 1) & (rec.shift(1) == 0)].index
    fwd12 = _forward(stocks.to_frame(), 12).iloc[:, 0]
    rows = []
    off_run = quiet
    for t, v in f.items():
        if v == 1 and off_run >= quiet:
            in_rec = bool(rec.get(t, 0) == 1)
            nxt = [s for s in starts if t < s <= t + pd.offsets.MonthEnd(within)]
            lead = None
            if nxt:
                s = nxt[0]
                lead = (s.year - t.year) * 12 + s.month - t.month
            rows.append({"trigger": t, "already in recession": in_rec, "recession followed": bool(nxt) and not in_rec,
                         "lead (months)": lead, "stocks next 12m": fwd12.get(t, np.nan)})
        off_run = 0 if v == 1 else off_run + 1
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- portfolio weather report

LONG_EPISODES = {
    "1973–74 oil shock and bear market": ("1973-01-11", "1974-10-03"),
    "1980–82 Volcker recessions": ("1980-11-28", "1982-08-12"),
    "1987 crash": ("1987-08-25", "1987-12-04"),
    "1990 Gulf War recession": ("1990-07-16", "1990-10-11"),
    **m.STRESS_EPISODES,
}


def weather_report(port_returns: pd.Series, regimes: pd.Series, corr_regime: pd.Series | None = None) -> dict[str, pd.DataFrame]:
    """How a portfolio behaved in each regime (regime known at t, return of t+1), in each correlation regime, and in crises."""
    r = port_returns.dropna()
    out = {}
    tab = returns_by_regime(r.to_frame("Portfolio"), regimes, next_month=True)
    out["by_regime"] = tab.drop(columns="asset").set_index("regime")
    if corr_regime is not None:
        out["by_correlation"] = returns_by_regime(r.to_frame("Portfolio"), corr_regime, next_month=True).drop(columns="asset").set_index("regime")
    nav = 100 * (1 + r).cumprod()
    nav = pd.concat([pd.Series([100.0], index=[nav.index[0] - pd.offsets.MonthEnd(1)]), nav])
    eps = {}
    for name, (a, b) in LONG_EPISODES.items():
        a_, b_ = pd.Timestamp(a) - pd.offsets.MonthEnd(1), pd.Timestamp(b) + pd.offsets.MonthEnd(0)
        if nav.index[0] <= a_ and nav.index[-1] >= b_:
            eps[name] = float(nav.asof(b_) / nav.asof(a_) - 1)
        else:
            eps[name] = np.nan
    out["episodes"] = pd.Series(eps, name="Portfolio").to_frame()
    return out


def stock_bond_correlation(stocks: pd.Series, bonds: pd.Series, window: int = 36) -> pd.Series:
    """Rolling correlation of monthly stock and bond returns (known at the close of each month)."""
    return stocks.rolling(window, min_periods=window).corr(bonds)


# --------------------------------------------------------------------------- macro filter for a trend rule


def risk_on_filter(panel_known: pd.DataFrame, kind: str, mode: str = "rolling", window: int = 12) -> pd.Series:
    """True = risk on. kind: 'curve' (10y − 3m ≥ 0), 'sahm' (< 0.5), 'growth' (IP growth ≥ its average),
    'credit' (Baa spread ≤ its average), 'payrolls' (payroll growth ≥ its average, revised data) and 'payrolls_first'
    (the same with payrolls as first published: the real-time version).
    mode for the averages: 'rolling' (window months), 'expanding', or 'full' (look-ahead)."""
    p = panel_known
    if kind == "curve":
        x = p["SLOPE"]
        on = x >= 0
    elif kind == "sahm":
        x = p["SAHM"]
        on = x < 0.5
    elif kind == "growth":
        x = p["IP_YOY"]
        on = x >= _threshold(x, mode, window)
    elif kind == "credit":
        x = p["CREDIT"]
        on = x <= _threshold(x, mode, window)
    elif kind in ("payrolls", "payrolls_first"):
        x = p["PAYEMS_YOY" if kind == "payrolls" else "PAYEMS_YOY_FIRST"]
        on = x >= _threshold(x, mode, window)
    else:
        raise ValueError("kind must be 'curve', 'sahm', 'growth', 'credit', 'payrolls' or 'payrolls_first'")
    return on.where(x.notna()).astype("float").rename(kind)


def apply_filter(targets: pd.DataFrame, risk_on: pd.Series, risky: list[str], safe: str = "CASH") -> pd.DataFrame:
    """When the filter says risk off at the close of t, move the weight of the `risky` assets in that month's target
    to `safe`. Missing filter values leave the target unchanged."""
    out = targets.copy()
    if safe not in out.columns:
        out[safe] = 0.0
    flag = risk_on.reindex(out.index)
    off = flag == 0
    moved = out.loc[off, [c for c in risky if c in out.columns]].sum(axis=1)
    out.loc[off, [c for c in risky if c in out.columns]] = 0.0
    out.loc[off, safe] = out.loc[off, safe] + moved
    return out


# --------------------------------------------------------------------------- real-time vs revised


def realtime_vs_revised_payrolls(data_dir=None) -> pd.DataFrame:
    """US payrolls (thousands) as first published (ALFRED) vs as known today, and the cumulative revision."""
    first = _month_end(data.load_fred("ALFRED_PAYEMS_first", data_dir))
    latest = _month_end(data.load_fred("PAYEMS", data_dir))
    df = pd.DataFrame({"first_release": first, "latest": latest}).dropna()
    df["revision"] = df["latest"] - df["first_release"]
    df["yoy_first"] = 100 * (df["first_release"] / df["first_release"].shift(12) - 1)
    df["yoy_latest"] = 100 * df["latest"].pct_change(12, fill_method=None)
    return df


@dataclass(frozen=True)
class _Unused:  # keeps dataclass import meaningful for future result types
    pass
