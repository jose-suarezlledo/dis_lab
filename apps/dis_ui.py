"""Shared look and words for every DIS app: palette, number formats, metric glossary.

Numbers never come from here (they come from engine/); this module only decides
how they are shown and explained.
"""

from __future__ import annotations

import math

import pandas as pd

DISCLAIMER = (
    "Educational tool for the IESE MBA elective *Data-driven Investment Strategies*. "
    "Past performance does not predict future results. Nothing here is investment advice."
)

# Categorical palette, fixed order (validated for colour-vision deficiency on adjacent pairs).
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BENCHMARK_COLOR = "#898781"  # neutral grey, dashed: readable on light and dark surfaces
GRID_COLOR = "rgba(137,135,129,0.25)"
DIVERGING_SCALE = [[0.0, "#e34948"], [0.5, "#f0efec"], [1.0, "#2a78d6"]]  # -1 red, 0 grey, +1 blue


def assign_colors(series: list[str], state: dict) -> dict[str, str]:
    """Stable colour per series: a series keeps its colour while it stays selected."""
    taken = {k: v for k, v in state.items() if k in series}
    free = [i for i in range(len(SERIES_COLORS)) if i not in taken.values()]
    for s in series:
        if s not in taken and free:
            taken[s] = free.pop(0)
    state.clear()
    state.update(taken)
    return {s: SERIES_COLORS[i] for s, i in taken.items()}


# --------------------------------------------------------------------------- formats


def pct(x, digits: int = 1) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}%}"


def num(x, digits: int = 2) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


PCT_ROWS = {"cagr", "vol", "max_dd", "pct_pos_years", "worst_year", "alpha", "tracking_error", "rf",
            "annual_turnover", "annual_cost"}
RATIO_ROWS = {"sharpe", "sortino", "calmar", "beta", "correlation", "info_ratio", "up_capture",
              "down_capture", "sharpe_se"}


def format_metric(key: str, value) -> str:
    if key in PCT_ROWS:
        return pct(value)
    if key in RATIO_ROWS:
        return num(value)
    if key == "years":
        return num(value, 1)
    if key == "tuw_days":
        return "–" if pd.isna(value) else f"{int(value):,} days ({value / 365.25:.1f} y)"
    return str(value)


# --------------------------------------------------------------------------- glossary


# key -> (label, what it is in plain English, the trap / why it matters)
GLOSSARY = {
    "cagr": ("CAGR", "The constant yearly growth rate that turns the start value into the end value.",
             "It hides the path: two series with the same CAGR can have very different drawdowns."),
    "vol": ("Volatility", "How much returns swing around their average, scaled to one year.",
            "It treats upside and downside swings alike, and it is low in calm years right before a crash."),
    "sharpe": ("Sharpe ratio", "Return above the risk-free rate per unit of volatility: (CAGR − rf) / vol.",
               "Estimated with a lot of noise: over 10 years ±0.3 or more. Small gaps mean nothing."),
    "sortino": ("Sortino ratio", "Like Sharpe, but only counts swings below the risk-free rate as risk.",
                "Flattering for strategies that sell insurance: few small losses, then a rare huge one."),
    "max_dd": ("Max drawdown", "The worst fall from a previous high to a later low, on the value path.",
               "The number that makes people sell at the bottom. Ask: could I have held through this?"),
    "tuw_days": ("Longest time under water", "The longest stretch spent below a previous high, until recovered.",
                 "Often years. Recovery takes longer than the fall: −50% needs +100% to get back."),
    "calmar": ("Calmar ratio", "CAGR divided by the size of the max drawdown.",
               "Depends on one single worst episode, so it moves a lot with the window chosen."),
    "pct_pos_years": ("% positive years", "Share of full calendar years with a gain.",
                      "Partial first and last years are excluded."),
    "worst_year": ("Worst calendar year", "The worst full calendar-year return.",
                   "Drawdowns can be deeper than the worst calendar year (they cross year ends)."),
    "beta": ("Beta", "How much the series tends to move when the benchmark moves 1%.",
             "Only meaningful if the benchmark actually explains the series (look at correlation)."),
    "alpha": ("Alpha (Jensen's)", "Return left over after paying for the benchmark exposure (beta), per year.",
              "Positive alpha over one window is often luck, a missing risk factor, or the wrong benchmark."),
    "correlation": ("Correlation", "How closely returns move together, from −1 to +1. Computed on returns, never prices.",
                    "It is not stable: it often rises in crises, exactly when you need diversification."),
    "tracking_error": ("Tracking error", "Volatility of the return difference versus the benchmark.",
                       "High tracking error means you will lag the benchmark, sometimes for years."),
    "info_ratio": ("Information ratio", "Extra return over the benchmark per unit of tracking error.",
                   "The active manager's Sharpe ratio. Above 0.5 sustained is rare."),
    "up_capture": ("Upside capture", "Average return when the benchmark is up, as a share of the benchmark's.", ""),
    "down_capture": ("Downside capture", "Average return when the benchmark is down, as a share of the benchmark's.",
                     "Below 100% means it falls less than the benchmark on down days."),
    "sharpe_se": ("Sharpe ± (1 s.e.)", "Approximate standard error of the Sharpe ratio (Lo, 2002).",
                  "Two Sharpe ratios closer than about 2 standard errors are statistically indistinguishable."),
    "annual_turnover": ("Turnover (per year)", "Share of the portfolio traded per year to rebalance (one-way).", ""),
    "annual_cost": ("Trading cost (per year)", "Average yearly cost of rebalancing, at the cost assumed.", ""),
}


def glossary_table(keys: list[str]) -> pd.DataFrame:
    rows = [GLOSSARY[k] for k in keys if k in GLOSSARY]
    return pd.DataFrame(rows, columns=["Metric", "What it is", "Watch out"])
