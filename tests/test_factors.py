"""Factor data (Ken French parser), factor history helpers, correlation matrix, down markets."""

import numpy as np
import pandas as pd
import pytest

from engine import data, factors, metrics

FRENCH_SAMPLE = """This file was created using the 202608 CRSP database.
Some notes about the factors.

Missing data are indicated by -99.99 or -999.

,Mkt-RF,SMB,HML,RF
192607,    2.89,   -2.55,   -2.39,    0.22
192608,    2.64,   -1.14,    3.81,    0.25
192609,  -99.99,   -1.36,    0.05,    0.23

  Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
1927,   29.47,   -2.46,   -3.75,    3.12
"""


def test_parse_french_csv_monthly_block_only():
    df = data.parse_french_csv(FRENCH_SAMPLE)
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 3  # the annual block is ignored
    assert df.index[0] == pd.Timestamp("1926-07-31")
    assert df.loc["1926-08-31", "HML"] == pytest.approx(0.0381)
    assert np.isnan(df.loc["1926-09-30", "Mkt-RF"])  # -99.99 = missing


def test_cached_french_factors():
    f = data.load_french()
    assert list(f.columns) == list(data.FRENCH_FACTORS)
    assert f.index[0] == pd.Timestamp("1926-07-31")
    assert f["RMW"].first_valid_index() == pd.Timestamp("1963-07-31")
    # Known episode: the momentum crash of spring 2009.
    assert f.loc["2009-04-30", "MOM"] < -0.30
    # Monthly decimals, not percent.
    assert f["MKT_RF"].abs().max() < 0.5


def test_growth_and_annualized():
    r = pd.Series([0.10, -0.10, 0.10, -0.10], index=pd.date_range("2000-01-31", periods=4, freq="ME"))
    g = factors.growth(r)
    assert g.iloc[-1] == pytest.approx(1.1 * 0.9 * 1.1 * 0.9)
    # 4 months -> annualize with ppy=12
    assert factors.annualized(r) == pytest.approx((1.1 * 0.9) ** 6 - 1)
    df = pd.DataFrame({"a": r, "b": [np.nan, 0.01, 0.01, 0.01]}, index=r.index)
    ann = factors.annualized(df)
    assert ann["b"] == pytest.approx(1.01**12 - 1)  # NaN start skipped, not treated as 0%


def test_rolling_annualized_full_windows_only():
    idx = pd.date_range("2000-01-31", periods=36, freq="ME")
    r = pd.Series(0.01, index=idx)
    roll = factors.rolling_annualized(r, years=2)
    assert roll.iloc[:23].isna().all()
    assert roll.iloc[23] == pytest.approx(1.01**12 - 1)


def test_decade_table_labels_partial_decades():
    idx = pd.date_range("1926-07-31", "1950-12-31", freq="ME")
    r = pd.DataFrame({"x": 0.01}, index=idx)
    t = factors.decade_table(r)
    assert list(t.index) == ["1926–1929", "1930s", "1940s", "1950–1950"]
    assert t.loc["1930s", "x"] == pytest.approx(1.01**12 - 1)


def test_worst_drawdown_and_recovery():
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    r = pd.Series([0.10, -0.50, 0.20, 0.50, 0.30, 0.0], index=idx)
    w = factors.worst_drawdown(r)
    assert w["depth"] == pytest.approx(-0.5)
    assert w["peak"] == idx[0] and w["trough"] == idx[1]
    # 1.1 -> 0.55 -> 0.66 -> 0.99 -> 1.287: back above 1.1 in month 5
    assert w["recovered"] == idx[4]


def test_correlation_matrix_returns_vs_levels():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=2500)
    a = 100 * np.exp(np.cumsum(0.0006 + 0.01 * rng.standard_normal(2500)))
    b = 100 * np.exp(np.cumsum(0.0006 + 0.01 * rng.standard_normal(2500)))
    lv = pd.DataFrame({"a": a, "b": b}, index=idx)
    on_ret = metrics.correlation_matrix(lv, "returns")
    on_lvl = metrics.correlation_matrix(lv, "levels")
    assert on_ret.loc["a", "a"] == pytest.approx(1.0)
    assert abs(on_ret.loc["a", "b"]) < 0.1  # independent returns
    assert on_lvl.loc["a", "b"] > 0.5  # two upward trends look "correlated"
    with pytest.raises(ValueError):
        metrics.correlation_matrix(lv, "prices")


def test_down_market_table():
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    lv = pd.DataFrame(
        {"bench": [100, 90, 99, 89.1, 98.01], "def": [100, 95, 99.75, 94.76, 99.5]}, index=idx
    )
    t = metrics.down_market_table(lv, "bench")
    assert t.attrs["down_periods"] == 2 and t.attrs["up_periods"] == 2
    assert t.loc["bench", "avg_when_bench_down"] == pytest.approx(-0.10)
    assert t.loc["def", "avg_when_bench_down"] == pytest.approx(-0.05, abs=1e-3)
    assert t.loc["def", "beat_bench_when_down"] == 1.0
