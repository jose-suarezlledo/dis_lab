"""Signals, the lagged backtest and the robustness tests, on synthetic data with known answers."""

import numpy as np
import pandas as pd
import pytest

from engine import metrics as m
from engine import robustness as rb
from engine import signals as sg
from engine.backtest import cash_returns_from_rate, signal_backtest, trace


def month_ends(n=60, start="2015-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def prices_two(n=60):
    """A rises 1% a month, B falls 1% a month, both from 100."""
    idx = month_ends(n)
    a = 100 * 1.01 ** np.arange(n)
    b = 100 * 0.99 ** np.arange(n)
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def zero_cash(idx):
    return pd.Series(0.0, index=idx, name=sg.CASH)


# --------------------------------------------------------------------------- signals


def test_momentum_is_the_lookback_return_and_nan_before():
    p = prices_two(20)
    mom = sg.momentum(p, 12)
    assert mom.iloc[:12].isna().all().all()
    assert mom["A"].iloc[12] == pytest.approx(1.01**12 - 1)
    with pytest.raises(ValueError):
        sg.momentum(p, 0)


def test_13612w_hand_value():
    p = prices_two(20)
    s = sg.score_13612w(p)
    r = lambda k: 1.01**k - 1  # noqa: E731
    assert s["A"].iloc[12] == pytest.approx((12 * r(1) + 4 * r(3) + 2 * r(6) + r(12)) / 4)


def test_top_n_picks_the_riser_and_excludes_late_starters():
    p = prices_two(30)
    p.loc[: p.index[14], "B"] = np.nan  # B lists in month 15
    score = sg.momentum(p, 12)
    w = sg.top_n_weights(score, n=1)
    assert w.index[0] == p.index[12]  # A alone is enough for Top-1
    assert (w["A"] == 1.0).all()
    w2 = sg.top_n_weights(score, n=2)
    assert w2.index[0] == p.index[15 + 12]  # Top-2 waits until B (listed in month 15) has 12 months of prices
    assert set(w2.columns) == {"A", "B", sg.CASH}  # CASH is always there, even when nothing is sent to it


def test_missing_zero_is_the_spreadsheet_trap():
    p = prices_two(30)
    p.loc[: p.index[14], "B"] = np.nan
    score = sg.momentum(p, 12)
    w = sg.top_n_weights(score, n=2, missing="zero")
    assert w.index[0] == p.index[0]  # every date is 'decided': missing scores count as 0
    assert w.loc[p.index[12], "B"] == pytest.approx(0.5)  # B is selected before it was listed


def test_absolute_filter_sends_failed_picks_to_safe():
    idx = month_ends(40)
    p = pd.DataFrame({"A": 100 * 0.99 ** np.arange(40), "B": 100 * 0.98 ** np.arange(40), "BND": 100.0}, index=idx)
    w = sg.dual_momentum_weights(p[["A", "B"]], lookback=12, top_n=1, absolute="zero")
    assert (w[sg.CASH] == 1.0).all()  # both fall, so everything goes to cash
    w2 = sg.dual_momentum_weights(p, lookback=12, top_n=1, absolute="zero", safe="BND")
    assert (w2["BND"] == 1.0).all()
    # a safe asset that is not yet available: the vetoed weight goes to CASH until it is
    late = p.copy()
    late.loc[: idx[14], "BND"] = np.nan
    w3 = sg.dual_momentum_weights(late, lookback=12, top_n=1, absolute="zero", safe="BND")
    assert (w3.loc[: idx[14 + 12 - 1], sg.CASH] == 1.0).all()  # BND has no 12-month score yet -> CASH
    assert w3["BND"].iloc[-1] == 1.0
    with pytest.raises(KeyError):
        sg.dual_momentum_weights(p, safe="NOPE", absolute="zero")
    with pytest.raises(ValueError):
        sg.dual_momentum_weights(p, absolute="cash")  # needs cash_returns


def test_gem_hand_example():
    idx = month_ends(16)
    us = 100 * 1.01 ** np.arange(16)
    intl = 100 * 1.02 ** np.arange(16)
    bonds = np.full(16, 100.0)
    p = pd.DataFrame({"US": us, "INTL": intl, "BND": bonds}, index=idx)
    cash = zero_cash(idx)
    w = sg.gem_weights(p, "US", "INTL", "BND", cash)
    assert w.index[0] == idx[12]
    assert (w["INTL"] == 1.0).all()  # US beats cash (risk on), international rose more
    cash_high = pd.Series(0.05, index=idx)  # 5% a month: US does not beat cash
    w2 = sg.gem_weights(p, "US", "INTL", "BND", cash_high)
    assert (w2["BND"] == 1.0).all()


def test_sma_weights_split_between_risers_and_cash():
    p = prices_two(30)
    w = sg.sma_weights(p, window=10)
    assert w.index[0] == p.index[9]
    np.testing.assert_allclose(w["A"], 0.5)  # A is above its average, B below -> half in A, half in cash
    np.testing.assert_allclose(w[sg.CASH], 0.5)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_equal_weights_follow_availability():
    p = prices_two(30)
    p.loc[: p.index[14], "B"] = np.nan
    w = sg.equal_weights(p)
    assert w.index[0] == p.index[1]  # an asset 'exists' once it has a return
    assert w.loc[p.index[1], "A"] == 1.0
    assert w.loc[p.index[15], "A"] == 1.0  # B's first price, no return yet
    assert w.loc[p.index[20], "A"] == 0.5


# --------------------------------------------------------------------------- backtest


def test_lag_one_holds_last_months_target():
    p = prices_two(30)
    # a target that switches to B in the last month: with lag=1 it never earns B's return
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=p.index)
    w.iloc[-1] = [0.0, 1.0]
    res = signal_backtest(p, w, lag=1)
    assert res.lag == 1
    assert res.held.iloc[-1]["A"] == 1.0
    np.testing.assert_allclose(res.nav.iloc[-1] / res.nav.iloc[0], p["A"].iloc[-1] / p["A"].iloc[1])


def test_look_ahead_with_perfect_foresight_is_absurd():
    rng = np.random.default_rng(0)
    idx = month_ends(120)
    r = rng.normal(0.005, 0.04, size=(120, 2))
    p = pd.DataFrame(100 * np.cumprod(1 + r, axis=0), index=idx, columns=["A", "B"])
    rets = p.pct_change(fill_method=None)
    foresight = (rets["A"] > rets["B"]).astype(float)
    w = pd.DataFrame({"A": foresight, "B": 1 - foresight}, index=idx).iloc[1:]
    honest = signal_backtest(p, w, lag=1)
    cheat = signal_backtest(p, w, lag=0)
    assert m.sharpe(cheat.nav) > 3 * max(m.sharpe(honest.nav), 0.1)  # the same 'signal', one month early


def test_costs_match_traded_value_and_first_position_is_free():
    p = prices_two(30)
    w = pd.DataFrame({"A": 1.0, "B": 0.0}, index=p.index)
    w.iloc[10:] = [0.0, 1.0]  # one switch, decided in month 10, executed for month 11
    res = signal_backtest(p, w, lag=1, cost_bps=100)
    assert res.trades == 1
    assert res.turnover.index[0] == p.index[11]
    assert res.turnover.iloc[0] == pytest.approx(1.0)  # sold all of A, bought all of B: one-way 100%
    assert res.costs.iloc[0] == pytest.approx(2.0 * 0.01)
    free = signal_backtest(p, w, lag=1, cost_bps=0)
    assert res.nav.iloc[-1] == pytest.approx(free.nav.iloc[-1] * (1 - 0.02))


def test_drift_is_accounted_for_in_turnover():
    p = prices_two(4)
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=p.index)
    res = signal_backtest(p, w, lag=1, cost_bps=100)
    # after one month A is 50*1.01, B 50*0.99 -> weights 0.505/0.495 -> trade 0.01 -> one-way 0.005
    assert res.turnover.iloc[0] == pytest.approx(0.005, rel=1e-6)


def test_holding_an_asset_without_a_price_raises_unless_the_trap_is_requested():
    p = prices_two(30)
    p.loc[: p.index[14], "B"] = np.nan
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=p.index)
    with pytest.raises(ValueError, match="no price"):
        signal_backtest(p, w, lag=1)
    res = signal_backtest(p, w, lag=1, missing="zero")
    assert res.nav.iloc[-1] > 0  # runs, with B 'earning' 0% while it did not exist


def test_cash_column_needs_cash_returns_and_uses_them():
    p = prices_two(30)
    w = pd.DataFrame({"A": 0.0, sg.CASH: 1.0}, index=p.index)
    with pytest.raises(ValueError, match="cash_returns"):
        signal_backtest(p, w)
    cash = pd.Series(0.01, index=p.index)
    res = signal_backtest(p, w, cash_returns=cash)
    np.testing.assert_allclose(res.returns.iloc[1:], 0.01)


def test_cash_returns_from_rate_matches_money_market_convention():
    idx = pd.to_datetime(["2024-01-31", "2024-02-29", "2024-03-31"])
    rate = pd.Series([3.6, 3.6, 3.6], index=idx)  # 3.6% a year
    c = cash_returns_from_rate(rate, idx)
    assert c.iloc[0] == 0.0
    assert c.iloc[1] == pytest.approx(0.036 * 29 / 360)
    assert c.iloc[2] == pytest.approx(0.036 * 31 / 360)


def test_weights_must_sum_to_one():
    p = prices_two(10)
    bad = pd.DataFrame({"A": 0.6, "B": 0.6}, index=p.index)
    with pytest.raises(ValueError, match="sum to 1"):
        signal_backtest(p, bad)


def test_trace_adds_up():
    p = prices_two(30)
    score = sg.momentum(p, 12)
    w = sg.top_n_weights(score, 1)
    res = signal_backtest(p, w, lag=1, cost_bps=10)
    t = res.held.index[5]
    tr = trace(res, t, score)
    assert tr.loc["Total", "weight held"] == pytest.approx(1.0)
    assert tr.loc["Strategy return", "contribution"] == pytest.approx(res.returns.loc[t])
    assert any(c.startswith("score at") for c in tr.columns)


# --------------------------------------------------------------------------- robustness


def test_parameter_grid_reports_every_combination_and_errors_as_nan():
    p = prices_two(40)

    def rule(lookback, top_n):
        w = sg.top_n_weights(sg.momentum(p, lookback), top_n)
        return signal_backtest(p, w, lag=1).nav

    g = rb.parameter_grid(rule, {"lookback": [3, 6, 60], "top_n": [1, 2]})
    assert len(g) == 6
    assert g.loc[g.lookback == 60, "sharpe"].isna().all()  # not enough history: NaN, not a crash
    assert g.loc[g.lookback == 3, "error"].eq("").all()


def test_nth_trading_day_sampler():
    idx = pd.bdate_range("2024-01-01", "2024-03-31")
    d = pd.DataFrame({"A": np.arange(len(idx), dtype=float)}, index=idx)
    first = rb.nth_trading_day(d, 0)
    last = rb.nth_trading_day(d, -1)
    assert list(first.index.day) == [1, 1, 1]
    assert list(last.index.day) == [31, 29, 29]
    far = rb.nth_trading_day(d, 40)  # longer than any month: falls back to the last day
    assert far.equals(last)


def test_rebalance_day_luck_runs_a_rule_per_day():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2010-01-01", periods=2600)
    p = pd.DataFrame({"A": 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 2600))}, index=idx)

    def rule(monthly):
        w = sg.sma_weights(monthly, 10)
        return signal_backtest(monthly, w, cash_returns=pd.Series(0.0, index=monthly.index)).nav

    table = rb.rebalance_day_luck(p, rule, days=(0, 10, -1))
    assert list(table.index) == ["trading day 1", "trading day 11", "month-end"]
    assert table["cagr"].notna().all()


def test_split_periods_rebases_the_second_segment():
    p = prices_two(60)
    tab = rb.split_periods(p["A"], p.index[30], labels=("In-sample", "Out-of-sample"))
    assert set(tab.columns) == {"In-sample", "Out-of-sample"}
    assert tab.loc["cagr", "In-sample"] == pytest.approx(tab.loc["cagr", "Out-of-sample"], rel=1e-2)  # month lengths differ
