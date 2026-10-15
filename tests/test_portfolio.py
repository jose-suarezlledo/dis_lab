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
