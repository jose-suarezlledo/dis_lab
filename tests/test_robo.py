"""Reference robo: questionnaire, model portfolios, backtest with fees, goal simulation."""

import numpy as np
import pandas as pd
import pytest

from engine import robo
from engine.metrics import cagr

ASSETS = list(robo.ASSETS)


def answers(**over):
    base = {q.id: 0 for q in robo.QUESTIONS}
    base.update(over)
    return base


def test_profile_is_the_lower_of_tolerance_and_capacity():
    bold_but_poor = answers(loss=3, swings=3, experience=3, horizon=0, share=0, income=0, need=0)
    p = robo.risk_profile(bold_but_poor)
    assert p.tolerance == 10 and p.capacity < 2 and p.profile == 1 and p.binding == "capacity"
    rich_but_timid = answers(loss=0, swings=0, experience=0, horizon=4, share=3, income=3, need=0)
    q = robo.risk_profile(rich_but_timid)
    assert q.capacity == 10 and q.profile == 1 and q.binding == "tolerance"


def test_profile_is_clamped_and_need_is_flagged():
    top = answers(loss=3, swings=3, experience=3, horizon=4, share=3, income=3, need=3)
    assert robo.risk_profile(top).profile == 10
    greedy = answers(loss=1, swings=1, experience=1, horizon=1, share=1, income=1, need=3)
    p = robo.risk_profile(greedy)
    assert p.need_exceeds_profile and "does not raise the risk" in p.note


def test_questionnaire_validation():
    with pytest.raises(KeyError):
        robo.risk_profile({"loss": 0})
    with pytest.raises(ValueError):
        robo.risk_profile(answers(loss=9))


def test_model_portfolios_sum_to_one_and_equity_rises_with_profile():
    t = robo.MODEL_PORTFOLIOS
    np.testing.assert_allclose(t.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    eq = [robo.equity_share(robo.model_weights(p)) for p in t.index]
    assert eq == sorted(eq) and eq[5] == pytest.approx(0.60) and eq[-1] == pytest.approx(0.95)
    assert "XEON.DE" in robo.model_weights(1) and "XEON.DE" not in robo.model_weights(6)


def _prices(n=800, seed=2):
    idx = pd.bdate_range("2018-01-01", periods=n)
    rng = np.random.default_rng(seed)
    vols = {"IWDA.AS": 0.15, "IS3N.DE": 0.20, "EUNH.DE": 0.05, "XG7S.DE": 0.04, "XEON.DE": 0.001}
    return pd.DataFrame({t: 100 * np.cumprod(1 + rng.standard_normal(n) * v / np.sqrt(252) + 0.0002)
                         for t, v in vols.items()}, index=idx)


def test_backtest_fee_and_rebalance_modes():
    px = _prices()
    w = robo.model_weights(6)
    free = robo.robo_backtest(px, w, fee=0.0, rebalance="monthly", cost_bps=0)
    paid = robo.robo_backtest(px, w, fee=0.01, rebalance="monthly", cost_bps=0)
    assert cagr(paid.nav) == pytest.approx(cagr(free.nav) - 0.01, abs=0.0015)
    band = robo.robo_backtest(px, w, fee=0.0, rebalance="band", band=0.05, cost_bps=10)
    assert band.band == 0.05 and len(band.gross.turnover) < len(free.gross.turnover)
    hold = robo.robo_backtest(px, w, rebalance="none")
    assert hold.gross.turnover.empty
    with pytest.raises(ValueError):
        robo.robo_backtest(px, w, rebalance="weekly")


def _monthly_returns(n=240, seed=5):
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    r = {"IWDA.AS": rng.standard_normal(n) * 0.045 + 0.006,
         "IS3N.DE": rng.standard_normal(n) * 0.06 + 0.005,
         "EUNH.DE": rng.standard_normal(n) * 0.012 + 0.002,
         "XG7S.DE": rng.standard_normal(n) * 0.010 + 0.002,
         "XEON.DE": np.full(n, 0.0)}
    return pd.DataFrame(r, index=idx)


def test_bootstrap_shapes_and_values_come_from_history():
    mr = _monthly_returns()
    paths = robo.bootstrap_paths(mr, months=36, n_sims=50, block=12, seed=1)
    assert paths.shape == (50, 36, 5)
    hist = set(np.round(mr["IWDA.AS"].to_numpy(), 12))
    assert all(np.round(v, 12) in hist for v in paths[0, :, 0])


def test_goal_simulation_cash_goal_is_certain_and_table_is_complete():
    mr = _monthly_returns()
    cash = {"XEON.DE": 1.0}
    goals = [robo.Goal("House", 60_000, 60_000, 3, cash),
             robo.Goal("Retirement", 40_000, 100_000, 20, robo.model_weights(9))]
    sim = robo.goal_simulation(mr, goals, profile_weights=robo.model_weights(6), n_sims=300, seed=0)
    t = sim.table
    assert set(t.index.get_level_values(0)) == {"Goal-based", "Profile-based"}
    assert t.loc[("Goal-based", "House"), "p_met"] == 1.0  # cash at 0% keeps exactly 60,000
    assert 0 <= sim.p_all_met["Goal-based"] <= 1 and 0 <= sim.p_all_met["Profile-based"] <= 1
    assert sim.equity_at_start["Profile-based"] == pytest.approx(0.60)
    assert sim.equity_at_start["Goal-based"] == pytest.approx(0.4 * 0.90)
    assert sim.months == 240


def test_profile_based_withdraws_the_first_goal():
    mr = _monthly_returns()
    goals = [robo.Goal("Now", 50_000, 50_000, 1 / 12, {"XEON.DE": 1.0}),
             robo.Goal("Later", 50_000, 50_000, 2, {"XEON.DE": 1.0})]
    sim = robo.goal_simulation(mr, goals, profile_weights={"XEON.DE": 1.0}, n_sims=10)
    t = sim.table
    # all in cash: the pot is 100,000, pays 50,000 after a month, and still has 50,000 at year 2
    assert t.loc[("Profile-based", "Now"), "median_wealth"] == pytest.approx(100_000)
    assert t.loc[("Profile-based", "Later"), "median_wealth"] == pytest.approx(50_000)
    assert sim.p_all_met["Profile-based"] == 1.0


def test_aggregate_weights_is_money_weighted():
    goals = [robo.Goal("A", 75, 1, 1, {"XEON.DE": 1.0}), robo.Goal("B", 25, 1, 1, {"IWDA.AS": 1.0})]
    agg = robo.aggregate_weights(goals)
    assert agg == {"XEON.DE": 0.75, "IWDA.AS": 0.25}
