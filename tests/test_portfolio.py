"""Fixed-weight portfolios: drift, rebalancing, costs, alignment."""

import numpy as np
import pandas as pd
import pytest

from engine import metrics as m
from engine.portfolio import fixed_weight_portfolio


def two_assets(n=300, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    a = 100 * 1.001 ** np.arange(n)  # steady riser
    b = 100 * np.ones(n)  # flat
    return pd.DataFrame({"A": a, "B": b}, index=idx)


def test_single_asset_portfolio_equals_the_asset():
    p = two_assets()
    res = fixed_weight_portfolio(p, {"A": 1.0}, rebalance="monthly", cost_bps=10)
    np.testing.assert_allclose(res.nav.to_numpy(), p["A"].to_numpy())
    assert res.annual_turnover == pytest.approx(0.0)
    assert res.annual_cost == pytest.approx(0.0)


def test_buy_and_hold_drifts_and_matches_hand_value():
    p = two_assets()
    res = fixed_weight_portfolio(p, {"A": 0.5, "B": 0.5}, rebalance="none")
    expected = 50 * p["A"] / 100 + 50 * p["B"] / 100
    np.testing.assert_allclose(res.nav.to_numpy(), expected.to_numpy())
    assert res.weights["A"].iloc[-1] > 0.5  # the winner drifted up
    assert res.turnover.empty


def test_monthly_rebalance_on_last_day_of_month():
    p = two_assets()
    res = fixed_weight_portfolio(p, {"A": 0.6, "B": 0.4}, rebalance="monthly")
    month_ends = res.turnover.index
    last_days = p.index.to_series().groupby(p.index.to_period("M")).max()
    assert set(month_ends) == set(last_days.iloc[:-1])  # the unfinished last month is not rebalanced
    np.testing.assert_allclose(res.weights.loc[month_ends, "A"], 0.6)


def test_rebalancing_has_turnover_and_costs_reduce_nav():
    p = two_assets()
    free = fixed_weight_portfolio(p, {"A": 0.5, "B": 0.5}, "monthly", cost_bps=0)
    costly = fixed_weight_portfolio(p, {"A": 0.5, "B": 0.5}, "monthly", cost_bps=50)
    assert free.annual_turnover > 0
    assert costly.nav.iloc[-1] < free.nav.iloc[-1]
    # cost at each rebalance = 2 x one-way turnover x 50 bps
    np.testing.assert_allclose(costly.costs.to_numpy(), 2 * costly.turnover.to_numpy() * 0.005)


def test_one_rebalance_by_hand():
    idx = pd.to_datetime(["2024-01-30", "2024-01-31", "2024-02-01"])
    p = pd.DataFrame({"A": [100.0, 200.0, 200.0], "B": [100.0, 100.0, 100.0]}, index=idx)
    res = fixed_weight_portfolio(p, {"A": 0.5, "B": 0.5}, "monthly", cost_bps=100)
    # 31 Jan: A doubled -> 100 + 50 = 150, weights 2/3, 1/3 -> trade 1/6 + 1/6 = 1/3 of NAV
    assert res.turnover.loc["2024-01-31"] == pytest.approx(1 / 6)
    assert res.nav.loc["2024-01-31"] == pytest.approx(150 * (1 - (1 / 3) * 0.01))


def test_late_starter_moves_the_start_instead_of_earning_zero():
    p = two_assets()
    p.loc[p.index[:50], "B"] = np.nan
    res = fixed_weight_portfolio(p, {"A": 0.5, "B": 0.5})
    assert res.nav.index[0] == p.index[50]


@pytest.mark.parametrize("weights", [{"A": 0.5, "B": 0.4}, {"A": 1.2, "B": -0.2}, {}])
def test_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        fixed_weight_portfolio(two_assets(), weights)


def test_rejects_unknown_ticker():
    with pytest.raises(KeyError):
        fixed_weight_portfolio(two_assets(), {"A": 0.5, "Z": 0.5})


def test_rebalanced_60_40_sits_between_its_parts():
    p = two_assets()
    res = fixed_weight_portfolio(p, {"A": 0.6, "B": 0.4}, "quarterly")
    assert m.cagr(p["B"]) < m.cagr(res.nav) < m.cagr(p["A"])


# --------------------------------------------------------------------------- volatility targeting

from engine.portfolio import cash_from_rate, vol_target_portfolio  # noqa: E402


def test_cash_from_rate_accrues_known_rate():
    idx = pd.to_datetime(["2020-01-01", "2020-01-31", "2020-03-01"])
    rate = pd.Series([3.6, 7.2], index=pd.to_datetime(["2019-12-31", "2020-01-31"]))
    c = cash_from_rate(rate, idx)
    assert c.iloc[0] == 100
    assert c.iloc[1] == pytest.approx(100 * (1 + 0.036 * 30 / 360))  # rate known at the start
    assert c.iloc[2] == pytest.approx(c.iloc[1] * (1 + 0.072 * 30 / 360))


def _risky_and_cash(vols, seed=1):
    rng = np.random.default_rng(seed)
    r = np.concatenate([rng.standard_normal(n) * v / np.sqrt(252) for n, v in vols])
    idx = pd.bdate_range("2010-01-01", periods=len(r))
    risky = pd.Series(100 * np.cumprod(1 + r), index=idx, name="X")
    cash = pd.Series(100.0, index=idx)
    return risky, cash


def test_vol_target_scales_down_in_turbulent_regime():
    risky, cash = _risky_and_cash([(300, 0.08), (300, 0.40)])
    res = vol_target_portfolio(risky, cash, target_vol=0.10, lookback=63)
    w = res.weights["X"]
    assert w.iloc[200] == pytest.approx(1.0)  # calm: vol < target -> capped at 100%
    assert 0.15 < w.iloc[-1] < 0.40  # turbulent: ~0.10 / 0.40
    assert w.max() <= 1.0 and res.nav.index[0] == risky.index[64]  # 63 returns end on day 63; held from day 64


def test_vol_target_lag_is_one_day():
    risky, cash = _risky_and_cash([(200, 0.15)])
    res = vol_target_portfolio(risky, cash, target_vol=0.10, lookback=20)
    r = risky.pct_change()
    sig = (0.10 / (r.rolling(20).std() * np.sqrt(252))).clip(upper=1.0)
    t = res.nav.index[5]
    prev = res.nav.index[4]
    expected = res.nav.loc[prev] * (1 + sig.shift(1).loc[t] * r.loc[t])
    assert res.nav.loc[t] == pytest.approx(expected)


def test_look_ahead_flatters_the_backtest():
    # Same-day weights know today's turbulence: lag=0 should not be worse on risk-adjusted terms.
    risky, cash = _risky_and_cash([(250, 0.10), (60, 0.60), (250, 0.10), (60, 0.60), (250, 0.10)], seed=3)
    honest = vol_target_portfolio(risky, cash, target_vol=0.10, lookback=10, lag=1)
    cheat = vol_target_portfolio(risky, cash, target_vol=0.10, lookback=10, lag=0)
    from engine.metrics import annualized_vol
    common = honest.nav.index.intersection(cheat.nav.index)
    assert annualized_vol(cheat.nav.loc[common]) < annualized_vol(honest.nav.loc[common])


def test_vol_target_costs_reduce_nav():
    risky, cash = _risky_and_cash([(300, 0.08), (300, 0.40)])
    free = vol_target_portfolio(risky, cash, cost_bps=0)
    costly = vol_target_portfolio(risky, cash, cost_bps=20)
    assert costly.nav.iloc[-1] < free.nav.iloc[-1]
    assert costly.annual_cost > 0 and costly.annual_turnover == pytest.approx(free.annual_turnover)
