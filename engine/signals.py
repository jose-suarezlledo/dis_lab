"""Trend and momentum signals, and the target weights they imply (Session 4: tactical asset allocation).

Every function works on a wide table of period-end prices (one column per asset, usually
month-ends) and returns a table on the same index. A signal computed on the row dated *t*
uses prices up to and including *t*, so it is known at the close of *t*. Whether it earns
the return of *t* (look-ahead) or of *t+1* (honest) is decided later, in `backtest.signal_backtest`.

Conventions
- NaN means "not available": an asset that does not have `lookback` prices yet has a NaN
  score and is never selected (it is not a 0). `missing="zero"` reproduces the classic
  spreadsheet error for teaching purposes only.
- The reserved column name "CASH" stands for the risk-free asset. Targets may send weight
  to it; the backtest fills it with the cash return it is given.
- Weights are long-only and sum to 1 on every date with a decision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["CASH", "momentum", "score_13612w", "excess_momentum", "sma", "above_sma",
           "top_n_weights", "dual_momentum_weights", "gem_weights", "sma_weights", "equal_weights"]

CASH = "CASH"


# --------------------------------------------------------------------------- scores


def momentum(prices: pd.DataFrame, lookback: int = 12) -> pd.DataFrame:
    """Total return over the last `lookback` periods: P_t / P_{t-lookback} - 1 (NaN until available)."""
    if lookback < 1:
        raise ValueError("lookback must be >= 1")
    return prices.pct_change(lookback, fill_method=None)


def score_13612w(prices: pd.DataFrame) -> pd.DataFrame:
    """Keller's 13612W momentum: (12·r1 + 4·r3 + 2·r6 + 1·r12) / 4, with r_k the k-month return.

    Each horizon is weighted so that it counts the same per month (12 × 1 month = 4 × 3 months...),
    which makes the score react fast to the latest month. Requires 12 periods of history.
    """
    return (12 * momentum(prices, 1) + 4 * momentum(prices, 3) + 2 * momentum(prices, 6) + momentum(prices, 12)) / 4


def excess_momentum(prices: pd.DataFrame, cash_returns: pd.Series, lookback: int = 12) -> pd.DataFrame:
    """Momentum minus the cash return over the same window (absolute momentum vs T-bills).

    `cash_returns` are the risk-free asset's periodic returns on the same index.
    """
    cash_growth = (1 + cash_returns.reindex(prices.index)).rolling(lookback).apply(np.prod, raw=True) - 1
    return momentum(prices, lookback).sub(cash_growth, axis=0)


def sma(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """Simple moving average of the last `window` period-end prices (the current one included)."""
    if window < 2:
        raise ValueError("window must be >= 2")
    return prices.rolling(window, min_periods=window).mean()


def above_sma(prices: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """1.0 when the price closes above its moving average, 0.0 below, NaN before the average exists."""
    avg = sma(prices, window)
    return (prices > avg).astype(float).where(avg.notna())


# --------------------------------------------------------------------------- weights


def _validate(prices: pd.DataFrame) -> None:
    if not isinstance(prices, pd.DataFrame) or prices.shape[1] == 0:
        raise TypeError("prices must be a DataFrame with one column per asset")
    if CASH in prices.columns:
        raise ValueError(f"'{CASH}' is reserved for the risk-free asset; rename that column")


def top_n_weights(
    score: pd.DataFrame,
    n: int = 1,
    absolute: pd.DataFrame | None = None,
    safe: str = CASH,
    missing: str = "exclude",
) -> pd.DataFrame:
    """Equal weight in the `n` assets with the highest score; a pick fails the absolute filter -> `safe`.

    score:    one score per asset and date (higher is better); NaN = not available.
    absolute: optional table on the same shape; a selected asset whose value is <= 0 is replaced
              by `safe` (e.g. momentum minus cash return, or plain momentum for a zero bound).
    safe:     column that receives the weight of failed picks: "CASH" or an asset in `score`. On dates
              when that asset has no score yet (not listed long enough), the weight goes to CASH instead.
    missing:  "exclude" (default) ignores assets without a score on that date;
              "zero" treats a missing score as 0 and lets it be selected: the spreadsheet trap.
    Returns target weights (always with a CASH column), one row per date on which at least `n` assets had a score.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if missing not in ("exclude", "zero"):
        raise ValueError("missing must be 'exclude' or 'zero'")
    if safe != CASH and safe not in score.columns:
        raise KeyError(f"safe asset {safe!r} is not in the universe")
    s = score.fillna(0.0) if missing == "zero" else score
    a = None if absolute is None else (absolute.fillna(0.0) if missing == "zero" else absolute.reindex_like(score))
    w = pd.DataFrame(0.0, index=score.index, columns=list(score.columns) + [CASH])
    decided = np.zeros(len(score), dtype=bool)
    for i, (t, row) in enumerate(s.iterrows()):
        avail = row.dropna()
        if len(avail) < n:
            continue
        safe_here = safe if (safe == CASH or not pd.isna(row[safe])) else CASH
        picks = avail.sort_values(ascending=False, kind="mergesort").index[:n]
        for p in picks:
            ok = True if a is None else bool(a.at[t, p] > 0) if not pd.isna(a.at[t, p]) else False
            w.at[t, p if ok else safe_here] += 1.0 / n
        decided[i] = True
    return w.loc[decided]


def dual_momentum_weights(
    prices: pd.DataFrame,
    lookback: int | str = 12,
    top_n: int = 1,
    absolute: str | None = "cash",
    cash_returns: pd.Series | None = None,
    safe: str = CASH,
    missing: str = "exclude",
) -> pd.DataFrame:
    """Antonacci-style dual momentum: relative momentum picks the Top-N, absolute momentum vetoes.

    lookback:  months of the simple momentum score, or "13612W" for Keller's weighted score.
    absolute:  "cash" (pick must beat T-bills over the lookback; needs `cash_returns`),
               "zero" (pick must have a positive score), or None (no veto: pure relative momentum).
    safe:      where vetoed picks go ("CASH" or a bond column).
    """
    _validate(prices)
    if isinstance(lookback, str):
        if lookback.upper() != "13612W":
            raise ValueError("lookback must be a number of periods or '13612W'")
        score, lb = score_13612w(prices), 12
    else:
        score, lb = momentum(prices, int(lookback)), int(lookback)
    if absolute is None:
        abs_table = None
    elif absolute == "zero":
        abs_table = score
    elif absolute == "cash":
        if cash_returns is None:
            raise ValueError("absolute='cash' needs cash_returns")
        abs_table = excess_momentum(prices, cash_returns, lb) if not isinstance(lookback, str) else \
            score.sub(((1 + cash_returns.reindex(prices.index)).rolling(12).apply(np.prod, raw=True) - 1), axis=0)
    else:
        raise ValueError("absolute must be 'cash', 'zero' or None")
    return top_n_weights(score, top_n, abs_table, safe, missing)


def gem_weights(
    prices: pd.DataFrame,
    us: str,
    intl: str,
    bonds: str,
    cash_returns: pd.Series,
    lookback: int = 12,
) -> pd.DataFrame:
    """Antonacci's Global Equities Momentum (2013): one decision a month, three possible holdings.

    If the US index beat T-bills over the lookback, hold whichever of US / international equities
    rose more; otherwise hold bonds. Note the asymmetry: the veto looks at the US index only.
    """
    _validate(prices)
    for c in (us, intl, bonds):
        if c not in prices.columns:
            raise KeyError(f"{c!r} not in prices")
    mom = momentum(prices[[us, intl]], lookback)
    ex_us = excess_momentum(prices[[us]], cash_returns, lookback)[us]
    valid = mom.notna().all(axis=1) & ex_us.notna() & prices[bonds].notna()
    w = pd.DataFrame(0.0, index=prices.index, columns=[us, intl, bonds])
    risk_on = ex_us > 0
    pick_us = mom[us] >= mom[intl]
    w.loc[valid & risk_on & pick_us, us] = 1.0
    w.loc[valid & risk_on & ~pick_us, intl] = 1.0
    w.loc[valid & ~risk_on, bonds] = 1.0
    return w.loc[valid]


def sma_weights(prices: pd.DataFrame, window: int = 10, safe: str = CASH, missing: str = "exclude") -> pd.DataFrame:
    """Faber's rule (2007): hold each asset only when it closes above its 10-month average, else cash.

    Each asset gets 1/N of the portfolio (N = assets with an average available on that date);
    an asset below its average sends its share to `safe`.
    """
    _validate(prices)
    if safe != CASH and safe not in prices.columns:
        raise KeyError(f"safe asset {safe!r} is not in the universe")
    sig = above_sma(prices, window)
    if missing == "zero":
        sig = sig.fillna(0.0)
    elif missing != "exclude":
        raise ValueError("missing must be 'exclude' or 'zero'")
    n_avail = sig.notna().sum(axis=1)
    w = pd.DataFrame(0.0, index=prices.index, columns=list(prices.columns) + [CASH])
    share = sig.div(n_avail.where(n_avail > 0), axis=0).fillna(0.0)
    w[prices.columns] = share
    to_safe = (1.0 - share.sum(axis=1)).where(n_avail > 0, 0.0)
    if safe == CASH:
        w[CASH] = to_safe
    else:  # the safe asset must itself be available on that date; otherwise the share goes to CASH
        safe_ok = prices[safe].notna()
        w[safe] = w[safe] + to_safe.where(safe_ok, 0.0)
        w[CASH] = to_safe.where(~safe_ok, 0.0)
    return w.loc[n_avail > 0]


def equal_weights(prices: pd.DataFrame, missing: str = "exclude") -> pd.DataFrame:
    """1/N over the assets that exist on each date (the no-skill benchmark for a Top-N rule).

    "Exists" = has a return on that date (a price today and one period earlier), so that the
    weights can be held over the same period (lag=0) as well as the next one.
    """
    _validate(prices)
    avail = prices.notna() & prices.shift(1).notna()
    n = avail.sum(axis=1)
    if missing == "zero":
        w = pd.DataFrame(1.0 / prices.shape[1], index=prices.index, columns=prices.columns)
    else:
        w = avail.astype(float).div(n.where(n > 0), axis=0).fillna(0.0)
    return w.loc[n > 0]
