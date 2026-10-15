"""Automatic "what could be wrong here" checks, mapped to the bias checklist.

The checklist is root CLAUDE.md section 7 (also student-facing). These checks
cover what can be detected from a set of return series alone; strategy-specific
checks (look-ahead in signals, parameter snooping) come with the backtest engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .metrics import (
    STRESS_EPISODES,
    cagr,
    correlation,
    sharpe,
    sharpe_standard_error,
    years_between,
)

__all__ = ["CHECKLIST", "Check", "performance_checks"]

CHECKLIST = {
    1: "Look-ahead",
    2: "Survivorship / selection",
    3: "Overfitting / data snooping",
    4: "Costs and implementation",
    5: "Benchmark",
    6: "Sanity",
    7: "Regime dependence",
}

SHARPE_ALARM = 1.5
CAGR_ALARM = 0.25
MIN_YEARS = 10
MIN_BENCH_CORR = 0.6


@dataclass(frozen=True)
class Check:
    severity: str  # "warn": look at this before believing the numbers; "info": keep in mind
    item: int  # bias checklist item, 1-7
    title: str
    detail: str


def _pct(x: float) -> str:
    return f"{x:.1%}"


def performance_checks(
    levels: pd.DataFrame,
    benchmark: str | None = None,
    inception: dict[str, pd.Timestamp] | None = None,
    listing_currency: dict[str, str] | None = None,
    display_currency: str | None = None,
    freq: str = "daily",
    rf: float = 0.0,
    portfolio: dict | None = None,
    asset_class: dict[str, str] | None = None,
) -> list[Check]:
    """Checks for a comparison of level series on their common window.

    levels: aligned levels (common dates), one column per series, at `freq`.
    inception: first available date of each series (and portfolio constituent) before alignment.
    listing_currency: listing currency per column (portfolio columns may be absent).
    display_currency: None if prices are shown as listed, else "USD" / "EUR".
    portfolio: {"name", "rebalance", "cost_bps"} if a portfolio is among the columns.
    """
    out: list[Check] = []
    idx = levels.index
    start, end = idx[0], idx[-1]
    years = years_between(start, end)
    cols = list(levels.columns)
    asset_class = asset_class or {}
    listing_currency = listing_currency or {}

    # 7. Regime dependence: does the window contain a real bear market?
    covered = [k for k, (a, b) in STRESS_EPISODES.items() if start <= pd.Timestamp(a) and end >= pd.Timestamp(b)]
    if years < MIN_YEARS:
        out.append(Check("warn", 7, f"Short history: {years:.1f} years",
                         "A few years of data say little about long-run return or risk. "
                         "Volatility and drawdowns estimated in calm years understate what can happen."))
    if not covered:
        out.append(Check("warn", 7, "No major sell-off in this window",
                         "The window contains none of: " + ", ".join(STRESS_EPISODES) + ". "
                         "Max drawdown and volatility here are a best case."))
    else:
        out.append(Check("info", 7, "Sell-offs inside this window: " + ", ".join(covered),
                         "See the stress-episode table (Calendar years tab): does the ranking "
                         "of these series hold in every episode, or only on average?"))

    # Window cut by the youngest series.
    if inception:
        firsts = {c: pd.Timestamp(d) for c, d in inception.items()}  # may include portfolio constituents
        if firsts:
            youngest = max(firsts, key=firsts.get)
            oldest = min(firsts.values())
            lost = years_between(oldest, firsts[youngest])
            if lost >= 1:
                out.append(Check("info", 2, f"The window starts in {start:%b %Y} because of {youngest}",
                                 f"{youngest} only has data from {firsts[youngest]:%d %b %Y}. To compare like "
                                 f"for like, every series is cut to the common dates, so up to {lost:.0f} years "
                                 "of history are dropped. Remove it to see the longer record."))

    # 6. Sanity: results too good to be true.
    srs = {c: sharpe(levels[c], rf) for c in cols}
    cagrs = {c: cagr(levels[c]) for c in cols}
    for c in cols:
        if srs[c] > SHARPE_ALARM:
            out.append(Check("warn", 6, f"{c}: Sharpe {srs[c]:.2f} is above {SHARPE_ALARM}",
                             "Very few investable assets sustain this. Assume a bug or a lucky window "
                             "until proven otherwise: check the data and try other start dates."))
        if cagrs[c] > CAGR_ALARM:
            out.append(Check("warn", 6, f"{c}: CAGR {_pct(cagrs[c])}",
                             "Far above what broad stock markets have delivered (roughly 7-10% a year "
                             "over long periods). Before extrapolating, ask what would have to stay true, "
                             "and whether you would have picked it at the start of the window."))

    # 2. Selection with hindsight (always true when you pick assets today).
    special = [c for c in cols if asset_class.get(c) == "Crypto" or cagrs[c] > CAGR_ALARM]
    detail = ("You chose these series today, knowing how they did. The universe also contains "
              "only funds that still exist (closed or merged funds are gone: survivorship).")
    if special:
        detail += f" This matters most for {', '.join(special)}: few people held them through the whole window."
    out.append(Check("info", 2, "Picked with hindsight", detail))

    # Currency mix.
    ccy = {listing_currency[c] for c in cols if c in listing_currency}
    if display_currency is None and len(ccy) > 1:
        out.append(Check("warn", 4, "Mixed currencies: " + " and ".join(sorted(ccy)),
                         "Series are shown in their listing currency, so the comparison mixes asset "
                         "returns with EUR/USD moves. Pick one currency (EUR for a euro investor)."))
    elif display_currency is not None and any(listing_currency.get(c) not in (None, display_currency) for c in cols):
        out.append(Check("info", 4, f"Converted to {display_currency}, unhedged",
                         "Returns include currency moves, as an investor who does not hedge FX sees them. "
                         "Hedging removes most of that FX risk, at a cost close to the interest-rate gap."))

    # Asynchronous closes (EU vs US listings, 7-day crypto) bias daily co-movement down.
    exchanges = {"EU" if listing_currency.get(c) == "EUR" else "US" for c in cols if c in listing_currency}
    has_crypto = any(asset_class.get(c) == "Crypto" for c in cols)
    if freq == "daily" and (len(exchanges) > 1 or has_crypto):
        out.append(Check("warn", 5, "Different closing times on daily data",
                         "European listings close hours before New York, and crypto trades at weekends. "
                         "Daily correlation and beta are understated. Switch to weekly or monthly returns."))

    # 5. Benchmark fit.
    if benchmark is not None and benchmark in cols:
        for c in cols:
            if c == benchmark:
                continue
            rho = correlation(levels[c], levels[benchmark])
            if rho < MIN_BENCH_CORR:
                out.append(Check("warn", 5, f"{benchmark} is a poor benchmark for {c} (correlation {rho:.2f})",
                                 f"Beta, alpha, tracking error and information ratio of {c} versus {benchmark} "
                                 "say little. Use a benchmark from the same asset class, or what you would "
                                 "otherwise hold (e.g. a 60/40 fund such as AOR or VBINX)."))
        if asset_class.get(benchmark) in ("US equity", "Global equity"):
            out.append(Check("info", 5, f"Benchmark {benchmark} is 100% equity",
                             "Beating stocks is not the only bar. Also compare with the simple portfolio "
                             "you would otherwise own, e.g. 60/40 (AOR, VBINX) or an equal-weight mix."))

    # 3. Is the Sharpe gap more than noise?
    if len(cols) > 1:
        top = max(srs.values())
        se = sharpe_standard_error(top, years)
        gap = top - min(srs.values())
        verdict = "within" if gap < 2 * se else "beyond"
        out.append(Check("info", 3, f"Sharpe ratios carry about ±{2 * se:.2f} of noise here",
                         f"Over {years:.1f} years a Sharpe of {top:.2f} has a standard error of about {se:.2f} "
                         f"(Lo, 2002). The gap between the best and worst Sharpe ({gap:.2f}) is {verdict} "
                         "two standard errors."))

    # 4. Costs and implementation.
    if portfolio is not None:
        if portfolio.get("rebalance") != "none" and portfolio.get("cost_bps", 0) == 0:
            out.append(Check("warn", 4, f"{portfolio.get('name', 'Portfolio')}: rebalanced at zero cost",
                             "Every rebalance pays spreads and commissions. Try 5-20 bps per trade and "
                             "see how much of the edge survives."))
    out.append(Check("info", 4, "Prices are net of fund fees only",
                     "Adjusted ETF/fund prices already deduct the fund's expense ratio (TER), but not "
                     "broker commissions, bid-ask spreads, FX conversion fees or taxes."))
    if freq != "daily":
        out.append(Check("info", 7, f"{freq.capitalize()} data hides intra-period lows",
                         "Max drawdown and time under water measured on weekly or monthly closes miss "
                         "the worst days (e.g. March 2020). Compare with the daily view."))

    # 1. Look-ahead: not applicable to buy-and-hold measurement.
    out.append(Check("info", 1, "No trading signals here",
                     "Look-ahead bias needs a rule that trades on information (signal at t, trade at t+1). "
                     "It becomes the first thing to check from Session 4 on."))

    order = {"warn": 0, "info": 1}
    return sorted(out, key=lambda c: (order[c.severity], c.item))
