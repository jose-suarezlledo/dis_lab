"""Signal-driven backtest: target weights decided at the close of one period, held over the next.

This is the engine behind every rules-based strategy in the course (TAA in Session 4, regimes
in Session 5, robustness in Session 7). It does one thing: turn a table of *target weights*
(from `engine.signals`, or from a regime model) into a NAV, honestly.

The honest timeline, with lag=1 (the default):
    close of month t:  prices known -> signal -> target weights w_t
    month t+1:         the portfolio holds w_t and earns the returns of t+1
    close of t+1:      new targets; the trade from the drifted weights to w_{t+1} pays costs
lag=0 makes w_t earn the return of month t itself: the look-ahead bias that at least 3 of 10
teams had in the 2025 assignment. It is kept as an option so that students can see its size.

Costs: cost = Σ|w_target − w_drifted| × cost_bps / 10,000, charged when the trade happens
(the first position is free, as in `engine.portfolio`). Turnover is one-way (half the traded
value). Missing returns are never 0: holding an asset with no price raises an error, and
`missing="zero"` exists only to reproduce the spreadsheet trap on purpose.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import years_between
from .signals import CASH

__all__ = ["BacktestResult", "signal_backtest", "cash_returns_from_rate", "trace"]


@dataclass(frozen=True)
class BacktestResult:
    nav: pd.Series  # strategy value, net of costs, starting at start_value
    returns: pd.Series  # periodic strategy returns, net of costs (first period = 0)
    targets: pd.DataFrame  # the weights decided at each date (as given)
    held: pd.DataFrame  # the weights that actually earned each period's return
    drifted: pd.DataFrame  # the weights just before each rebalance (after the period's drift)
    turnover: pd.Series  # one-way turnover at each rebalance (fraction of NAV)
    costs: pd.Series  # cost paid at each rebalance (fraction of NAV)
    asset_returns: pd.DataFrame  # the periodic returns the held weights were applied to
    lag: int
    annual_turnover: float
    annual_cost: float
    trades: int  # number of periods in which something was traded


def cash_returns_from_rate(rate_pct: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Periodic return of cash earning a quoted annual rate (e.g. FRED DTB3, in %), on `index`.

    Each period earns the rate known at its start, accrued on calendar days / 360.
    """
    from .portfolio import cash_from_rate

    level = cash_from_rate(rate_pct, index)
    return level.pct_change(fill_method=None).fillna(0.0).rename(CASH)


def signal_backtest(
    prices: pd.DataFrame,
    targets: pd.DataFrame,
    lag: int = 1,
    cost_bps: float = 0.0,
    cash_returns: pd.Series | None = None,
    missing: str = "exclude",
    start_value: float = 100.0,
    name: str = "Strategy",
) -> BacktestResult:
    """NAV of a strategy that holds `targets` shifted by `lag` periods, with trading costs.

    prices:       period-end prices (same index as `targets`, or a superset of it).
    targets:      weights per date; rows sum to 1 (a CASH column is allowed, see `cash_returns`).
    lag:          1 = decide at the close of t, hold over t+1 (honest); 0 = look-ahead.
    cost_bps:     one-way cost per unit traded, on the value traded.
    cash_returns: periodic return of the CASH column (needed if targets send weight to CASH).
    missing:      "exclude" raises if a held asset has no return; "zero" treats it as 0% (the trap).
    """
    if lag not in (0, 1, 2, 3):
        raise ValueError("lag must be 0 (look-ahead), 1 (honest) or a small positive integer")
    if cost_bps < 0:
        raise ValueError("cost_bps must be non-negative")
    if missing not in ("exclude", "zero"):
        raise ValueError("missing must be 'exclude' or 'zero'")
    if targets.empty:
        raise ValueError("targets is empty: the signal never had enough history to decide")
    sums = targets.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-6):
        bad = targets.index[~np.isclose(sums, 1.0, atol=1e-6)]
        raise ValueError(f"target weights must sum to 1 on every date; first offender {bad[0].date()}")
    if (targets < -1e-12).any().any():
        raise ValueError("target weights must be long-only")

    assets = [c for c in targets.columns if c != CASH]
    missing_cols = [c for c in assets if c not in prices.columns]
    if missing_cols:
        raise KeyError(f"no prices for {missing_cols}")
    rets = prices[assets].pct_change(fill_method=None)
    if CASH in targets.columns:
        uses_cash = bool((targets[CASH] > 1e-12).any())
        if cash_returns is None:
            if uses_cash:
                raise ValueError("targets use CASH: pass cash_returns")
            rets[CASH] = 0.0  # the column exists but never holds anything
        else:
            rets[CASH] = cash_returns.reindex(rets.index)
            if uses_cash and rets[CASH].isna().any():
                first_bad = rets.index[rets[CASH].isna()][0]
                raise ValueError(f"cash_returns missing on {first_bad.date()}")
    rets = rets[targets.columns]

    held = targets.shift(lag).reindex(rets.index)
    first = held.dropna(how="all").index
    first = first[(held.loc[first].sum(axis=1) > 0)]
    if len(first) == 0:
        raise ValueError("no period is both decided and investable")
    start = first[0]
    held = held.loc[start:].fillna(0.0)
    rets = rets.loc[start:]
    r = rets.to_numpy(dtype=float)
    w = held.to_numpy(dtype=float)
    if missing == "zero":
        r = np.nan_to_num(r, nan=0.0)
    else:
        exposed = (w > 1e-12) & np.isnan(r)
        if exposed.any():
            i, j = np.argwhere(exposed)[0]
            raise ValueError(
                f"held {held.columns[j]} on {held.index[i].date()} but it has no price; "
                "exclude late-starting or dead assets from the signal instead of treating them as 0%"
            )
        r = np.nan_to_num(r, nan=0.0)  # unheld assets with NaN returns do not matter

    n, k = w.shape
    cost_rate = cost_bps / 10_000
    port_ret = np.zeros(n)
    cost = np.zeros(n)
    turn = np.zeros(n)
    drifted = np.zeros((n, k))
    traded_flag = np.zeros(n, dtype=bool)
    nav = np.empty(n)
    nav[0] = start_value
    prev_end = w[0].copy()  # the position at the start of the sample: entering it is free
    drifted[0] = w[0]
    for t in range(1, n):
        # trade from the weights we ended period t-1 with (prev_end) to the new held weights w[t]
        traded = np.abs(w[t] - prev_end).sum()
        if traded > 1e-10:
            turn[t] = 0.5 * traded
            cost[t] = traded * cost_rate
            traded_flag[t] = True
        drifted[t] = prev_end
        gross = float((w[t] * r[t]).sum())
        port_ret[t] = (1 + gross) * (1 - cost[t]) - 1
        nav[t] = nav[t - 1] * (1 + port_ret[t])
        end_val = w[t] * (1 + r[t])
        prev_end = end_val / end_val.sum() if end_val.sum() > 0 else w[t]

    idx = held.index
    years = years_between(idx[0], idx[-1])
    turnover = pd.Series(turn, index=idx)[traded_flag]
    costs = pd.Series(cost, index=idx)[traded_flag]
    return BacktestResult(
        nav=pd.Series(nav, index=idx, name=name),
        returns=pd.Series(port_ret, index=idx, name=name),
        targets=targets,
        held=held,
        drifted=pd.DataFrame(drifted, index=idx, columns=held.columns),
        turnover=turnover,
        costs=costs,
        asset_returns=pd.DataFrame(r, index=idx, columns=held.columns),
        lag=lag,
        annual_turnover=float(turnover.sum() / years) if years > 0 else np.nan,
        annual_cost=float(costs.sum() / years) if years > 0 else np.nan,
        trades=int(traded_flag.sum()),
    )


def trace(result: BacktestResult, date: str | pd.Timestamp, score: pd.DataFrame | None = None) -> pd.DataFrame:
    """One period of the backtest, line by line, for the glass-box view.

    Rows = assets; columns = the score that decided the weights (if given), the target decided
    `lag` periods earlier, the weight actually held, the asset's return over the period, and
    its contribution. The last row totals the contribution, cost and net strategy return.
    """
    t = pd.Timestamp(date)
    if t not in result.held.index:
        raise KeyError(f"{t.date()} is not a period of this backtest")
    pos = result.held.index.get_loc(t)
    decided_on = result.targets.index[result.targets.index <= t]
    decided_on = decided_on[-(result.lag + 1)] if len(decided_on) > result.lag else None
    cols = {}
    if score is not None and decided_on is not None and decided_on in score.index:
        cols[f"score at {decided_on.date()}"] = score.loc[decided_on].reindex(result.held.columns)
    cols["weight held"] = result.held.iloc[pos]
    cols["asset return"] = result.asset_returns.iloc[pos]
    cols["contribution"] = result.held.iloc[pos] * result.asset_returns.iloc[pos]
    df = pd.DataFrame(cols)
    total = {c: np.nan for c in df.columns}
    total["weight held"] = df["weight held"].sum()
    total["contribution"] = df["contribution"].sum()
    df.loc["Total"] = total
    df.loc["Cost paid"] = {**{c: np.nan for c in df.columns}, "contribution": -float(result.costs.get(t, 0.0))}
    df.loc["Strategy return"] = {**{c: np.nan for c in df.columns}, "contribution": float(result.returns.iloc[pos])}
    return df
