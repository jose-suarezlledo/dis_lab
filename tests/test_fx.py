"""Currency views: local, unhedged, hedged (carry = rate differential)."""

import numpy as np
import pandas as pd
import pytest

from engine import fx
from engine.metrics import cagr


def _local(n=500, seed=0):
    idx = pd.bdate_range("2020-01-01", periods=n)
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.standard_normal(n) * 0.01), index=idx, name="X")


def _rates(usd, eur, idx):
    first = idx[0] - pd.Timedelta(days=1)
    return {"USD": pd.Series([usd], index=[first]), "EUR": pd.Series([eur], index=[first])}


def test_equal_rates_hedged_equals_local():
    s = _local()
    h = fx.hedged_levels(s, rates=_rates(2.0, 2.0, s.index))
    np.testing.assert_allclose(h.to_numpy(), s.to_numpy())


def test_hedging_pays_when_home_rate_is_higher_and_costs_when_lower():
    s = _local()
    pays = fx.hedged_levels(s, rates=_rates(1.0, 4.0, s.index))
    costs = fx.hedged_levels(s, rates=_rates(4.0, 1.0, s.index))
    assert pays.iloc[-1] > s.iloc[-1] > costs.iloc[-1]
    # 3 points of carry over ~2 years, money-market accrual
    assert cagr(costs) == pytest.approx(cagr(s) - 0.03, abs=0.004)


def test_hedged_frame_keeps_shape():
    s = _local()
    df = pd.DataFrame({"A": s, "B": s * 2})
    h = fx.hedged_levels(df, rates=_rates(2.0, 3.0, s.index))
    assert h.shape == df.shape and list(h.columns) == ["A", "B"]


def test_currency_views_unhedged_follows_fx_and_hedged_ignores_it():
    s = _local()
    usd_per_eur = pd.Series(np.linspace(1.0, 1.25, len(s)), index=s.index)  # euro rises 25%
    v = fx.currency_views(s, rates=_rates(2.0, 2.0, s.index), fx=usd_per_eur)
    assert list(v.columns) == ["USD (local)", "EUR unhedged", "EUR hedged"]
    np.testing.assert_allclose(v.iloc[0].to_numpy(), 100)
    # unhedged: a rising euro hurts the euro investor by exactly the FX move
    assert v["EUR unhedged"].iloc[-1] / v["USD (local)"].iloc[-1] == pytest.approx(1 / 1.25)
    np.testing.assert_allclose(v["EUR hedged"].to_numpy(), v["USD (local)"].to_numpy())


def test_rejects_same_currency():
    with pytest.raises(ValueError):
        fx.hedged_levels(_local(), "EUR", "EUR")


def test_real_snapshot_world_equity_views():
    """MSCI World (URTH) since 2016: the dollar cushioned 2022 for a euro investor."""
    from engine import data

    urth = data.load_prices("URTH")["URTH"].loc["2016":]
    v = fx.currency_views(urth)
    yr = v.resample("YE").last().pct_change()
    r22 = yr.loc["2022"].iloc[0]
    assert r22["EUR unhedged"] > r22["USD (local)"] > r22["EUR hedged"] - 0.03
    # Hedging a dollar asset into euros cost carry over the window (US rates above euro rates).
    assert cagr(v["EUR hedged"]) < cagr(v["USD (local)"])
