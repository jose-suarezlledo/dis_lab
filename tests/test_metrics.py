"""Metric definitions checked against hand-computable cases (root CLAUDE.md section 7)."""

import numpy as np
import pandas as pd
import pytest

from engine import metrics as m


def nav_from_returns(r, start="2015-01-01", freq="B", base=100.0):
    idx = pd.date_range(start, periods=len(r) + 1, freq=freq)
    return pd.Series(base * np.concatenate([[1.0], np.cumprod(1 + np.asarray(r))]), index=idx)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# --------------------------------------------------------------------------- frequency & returns


@pytest.mark.parametrize(
    "freq, expected",
    [("B", 252), ("D", 365), ("W-FRI", 52), ("ME", 12), ("QE", 4), ("YE", 1)],
)
def test_periods_per_year(freq, expected):
    idx = pd.date_range("2010-01-01", periods=60, freq=freq)
    assert m.periods_per_year(idx) == expected


def test_returns_keep_nan_instead_of_zero():
    s = pd.Series([np.nan, np.nan, 100, 110], index=pd.date_range("2020-01-01", periods=4))
    r = m.to_returns(s)
    assert r.isna().sum() == 2  # pre-inception stays missing, never 0%
    assert r.iloc[-1] == pytest.approx(0.10)


# --------------------------------------------------------------------------- CAGR


def test_cagr_uses_calendar_days_over_365_25():
    s = pd.Series([100.0, 200.0], index=pd.to_datetime(["2000-01-01", "2010-01-01"]))
    years = 3653 / 365.25
    assert m.cagr(s) == pytest.approx(2 ** (1 / years) - 1)


def test_cagr_ignores_pre_inception_nans():
    s = pd.Series(
        [np.nan, 100.0, 121.0],
        index=pd.to_datetime(["1990-01-01", "2000-01-01", "2002-01-01"]),
    )
    years = 731 / 365.25
    assert m.cagr(s) == pytest.approx(1.21 ** (1 / years) - 1)


def test_rejects_non_positive_levels():
    s = pd.Series([100.0, 0.0, 50.0], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        m.cagr(s)


# --------------------------------------------------------------------------- volatility, Sharpe, Sortino


def test_vol_is_std_of_returns_times_sqrt_ppy(rng):
    r = rng.normal(0.0004, 0.01, 1000)
    nav = nav_from_returns(r)
    assert m.annualized_vol(nav) == pytest.approx(np.std(r, ddof=1) * np.sqrt(252))


def test_vol_monthly_uses_sqrt_12(rng):
    r = rng.normal(0.005, 0.04, 120)
    nav = nav_from_returns(r, freq="ME")
    assert m.annualized_vol(nav) == pytest.approx(np.std(r, ddof=1) * np.sqrt(12))


def test_sharpe_is_cagr_minus_rf_over_vol(rng):
    nav = nav_from_returns(rng.normal(0.0004, 0.01, 1000))
    expected = (m.cagr(nav) - 0.03) / m.annualized_vol(nav)
    assert m.sharpe(nav, rf=0.03) == pytest.approx(expected)


def test_sharpe_nan_with_zero_vol():
    nav = nav_from_returns([0.001] * 100)
    assert m.annualized_vol(nav) == pytest.approx(0, abs=1e-12)
    assert np.isnan(m.sharpe(nav))
    assert np.isnan(m.sortino(nav))  # no down periods at all


def test_downside_deviation_counts_all_periods():
    r = np.array([0.02, -0.01, 0.03, -0.02])
    nav = nav_from_returns(r, freq="ME")
    expected = np.sqrt(np.mean(np.minimum(r, 0) ** 2)) * np.sqrt(12)
    assert m.downside_deviation(nav) == pytest.approx(expected)
    assert m.sortino(nav) == pytest.approx(m.cagr(nav) / expected)


# --------------------------------------------------------------------------- drawdowns


@pytest.fixture
def bumpy():
    levels = [100, 120, 90, 110, 130, 80, 100]
    return pd.Series(levels, index=pd.date_range("2020-01-31", periods=7, freq="ME"), dtype=float)


def test_max_drawdown_on_nav_path(bumpy):
    assert m.max_drawdown(bumpy) == pytest.approx(80 / 130 - 1)


def test_max_drawdown_is_not_worst_single_return():
    nav = nav_from_returns([-0.10, -0.10, -0.10, 0.05])
    assert m.max_drawdown(nav) == pytest.approx(0.9**3 - 1)  # -27.1%, not -10%


def test_drawdown_series_zero_at_new_highs(bumpy):
    dd = m.drawdown_series(bumpy)
    assert dd.loc["2020-05-31"] == 0
    assert dd.loc["2020-03-31"] == pytest.approx(90 / 120 - 1)


def test_time_under_water_recovered_episode():
    idx = pd.date_range("2020-01-01", periods=7, freq="D")
    nav = pd.Series([100, 110, 100, 105, 111, 90, 95], index=idx, dtype=float)
    tuw = m.time_under_water(nav)
    # Peak 110 on day 1, back above it on day 4 -> 3 days (the ongoing one is 2 days).
    assert tuw.days == 3
    assert tuw.start == idx[1] and tuw.end == idx[4]
    assert tuw.recovered


def test_time_under_water_ongoing_episode_counts_to_last_date():
    idx = pd.date_range("2020-01-01", periods=6, freq="D")
    nav = pd.Series([100, 105, 106, 90, 95, 99], index=idx, dtype=float)
    tuw = m.time_under_water(nav)
    assert tuw.days == 3 and not tuw.recovered and tuw.end == idx[-1]


def test_time_under_water_zero_for_monotone_series():
    assert m.time_under_water(nav_from_returns([0.01] * 20)).days == 0


def test_calmar(bumpy):
    assert m.calmar(bumpy) == pytest.approx(m.cagr(bumpy) / abs(80 / 130 - 1))


# --------------------------------------------------------------------------- calendar years


def test_calendar_years_full_and_partial():
    nav = pd.Series(
        [100, 110, 120, 108, 120],
        index=pd.to_datetime(["2019-12-31", "2020-06-30", "2020-12-31", "2021-12-31", "2022-06-30"]),
        dtype=float,
    )
    yr = m.calendar_year_returns(nav)
    assert list(yr.index) == [2020, 2021, 2022]
    assert yr.loc[2020, "return"] == pytest.approx(0.20)
    assert yr.loc[2021, "return"] == pytest.approx(-0.10)
    assert list(yr["partial"]) == [False, False, True]
    assert m.pct_positive_years(nav) == pytest.approx(0.5)
    assert m.worst_year(nav) == pytest.approx(-0.10)


def test_calendar_years_mid_year_start_is_partial():
    nav = pd.Series(
        [100, 105, 110],
        index=pd.to_datetime(["2020-03-15", "2020-12-31", "2021-12-31"]),
        dtype=float,
    )
    yr = m.calendar_year_returns(nav)
    assert yr.loc[2020, "partial"] and not yr.loc[2021, "partial"]
    assert yr.loc[2020, "return"] == pytest.approx(0.05)
    assert m.pct_positive_years(nav) == 1.0  # only 2021 counts


# --------------------------------------------------------------------------- vs benchmark


@pytest.fixture
def bench_and_double(rng):
    rb = rng.normal(0.0003, 0.01, 750)
    return nav_from_returns(rb), nav_from_returns(2 * rb), rb


def test_beta_alpha_correlation_of_2x_portfolio(bench_and_double):
    bench, port, _ = bench_and_double
    assert m.beta(port, bench) == pytest.approx(2.0)
    assert m.correlation(port, bench) == pytest.approx(1.0)
    assert m.alpha(port, bench, rf=0.0) == pytest.approx(0.0, abs=1e-12)


def test_tracking_error_and_information_ratio(bench_and_double):
    bench, port, rb = bench_and_double
    assert m.tracking_error(port, bench) == pytest.approx(np.std(rb, ddof=1) * np.sqrt(252))
    ir = (m.cagr(port) - m.cagr(bench)) / m.tracking_error(port, bench)
    assert m.information_ratio(port, bench) == pytest.approx(ir)
    assert np.isnan(m.information_ratio(bench, bench))


def test_capture_ratios_of_half_exposure(rng):
    rb = rng.normal(0, 0.01, 500)
    up, down = m.capture_ratios(nav_from_returns(0.5 * rb), nav_from_returns(rb))
    assert up == pytest.approx(0.5) and down == pytest.approx(0.5)


def test_correlation_uses_returns_not_levels(rng):
    # Two independent random walks: levels look correlated, returns do not.
    a = nav_from_returns(rng.normal(0.001, 0.01, 2000))
    b = nav_from_returns(rng.normal(0.001, 0.01, 2000))
    assert abs(m.correlation(a, b)) < 0.1
    assert abs(a.corr(b)) > 0.5  # the classic mistake would report this


def test_align_joins_by_date_not_row_position():
    idx = pd.bdate_range("2024-01-01", periods=5)  # Mon..Fri
    a = pd.Series([100, 101, 102, 103, 104], index=idx, dtype=float, name="a")
    b = pd.Series([50, 51, 53, 54], index=idx.delete(2), dtype=float, name="b")  # no Wednesday
    both = m.align(a, b)
    assert len(both) == 4 and idx[2] not in both.index
    r = m.to_returns(both)
    # Thursday's return for `a` spans Tue->Thu, it is not a fake 0% day.
    assert r.loc[idx[3], "a"] == pytest.approx(103 / 101 - 1)
    assert r.loc[idx[1], "b"] == pytest.approx(51 / 50 - 1)


# --------------------------------------------------------------------------- implementation, rolling, summary


def test_turnover_one_way_annualized():
    w = pd.DataFrame(
        {"A": [0.6, 0.4], "B": [0.4, 0.6]},
        index=pd.to_datetime(["2020-01-31", "2021-01-31"]),
    )
    assert m.turnover(w) == pytest.approx(0.2 / (366 / 365.25))


def test_rolling_cagr_of_constant_growth():
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    g = 0.07
    nav = pd.Series((1 + g) ** ((idx - idx[0]).days / 365.25), index=idx)
    roll = m.rolling_cagr(nav, years=3)
    assert roll.index[0] >= idx[0] + pd.Timedelta(days=1096)
    assert np.allclose(roll, g, atol=1e-3)


def test_summary_keys_and_benchmark_fields(bench_and_double):
    bench, port, _ = bench_and_double
    s = m.summary(port, bench, rf=0.02)
    assert set(m.METRIC_LABELS) == set(s.index)
    assert s["rf"] == 0.02
    assert "beta" not in m.summary(port).index


def test_summary_table_uses_common_window():
    idx = pd.bdate_range("2020-01-01", periods=300)
    prices = pd.DataFrame(
        {"old": np.linspace(100, 130, 300), "new": np.linspace(100, 120, 300)}, index=idx
    )
    prices.loc[idx[:100], "new"] = np.nan  # late starter
    tbl = m.summary_table(prices, benchmark="old")
    assert tbl.loc["start", "old"] == idx[100].date() == tbl.loc["start", "new"]
    assert np.isnan(tbl.loc["beta", "old"])  # the benchmark is not compared with itself
    assert np.isfinite(tbl.loc["beta", "new"])


# --------------------------------------------------------------------------- frequency, rolling, episodes


def test_to_frequency_keeps_real_dates_and_cagr():
    idx = pd.bdate_range("2020-01-06", "2020-03-18")
    s = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
    mth = m.to_frequency(s, "monthly")
    assert list(mth.index) == list(pd.to_datetime(["2020-01-06", "2020-01-31", "2020-02-28", "2020-03-18"]))
    assert m.cagr(mth) == pytest.approx(m.cagr(s))
    wk = m.to_frequency(s, "weekly")
    assert (wk.index[1:-1].dayofweek == 4).all()  # Fridays (no holidays in this range)
    assert m.to_frequency(s, "daily") is s


def test_to_frequency_rejects_unknown():
    with pytest.raises(ValueError):
        m.to_frequency(pd.Series(dtype=float), "hourly")


def test_rolling_correlation_of_identical_series_is_one(rng):
    a = nav_from_returns(rng.normal(0, 0.01, 600))
    rc = m.rolling_correlation(a, a, years=1)
    assert len(rc) > 0 and np.allclose(rc, 1.0)


def test_sharpe_standard_error():
    assert m.sharpe_standard_error(0.0, 4) == pytest.approx(0.5)
    assert m.sharpe_standard_error(0.5, 10) == pytest.approx(np.sqrt(1.125 / 10))
    assert np.isnan(m.sharpe_standard_error(1.0, 0))


def test_episode_returns_uses_asof_and_nan_outside():
    s = pd.Series([100.0, 50.0, 80.0], index=pd.to_datetime(["2020-02-18", "2020-03-20", "2020-06-01"]))
    eps = {"crash": ("2020-02-19", "2020-03-23"), "before": ("2019-01-01", "2019-02-01")}
    out = m.episode_returns(s, eps)
    assert out["crash"] == pytest.approx(-0.5)  # last level on or before each date
    assert np.isnan(out["before"])
