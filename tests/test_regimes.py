"""Regimes module: real-time shifting, thresholds, returns by regime, analogs, scorecard, filter; and the Regimes page."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine import regimes as rg


def idx(n=120, start="2000-01-31"):
    return pd.date_range(start, periods=n, freq="ME")


def test_known_at_shifts_by_publication_lag():
    i = idx(5)
    p = pd.DataFrame({"IP_YOY": [1.0, 2, 3, 4, 5], "SLOPE": [1.0, 2, 3, 4, 5]}, index=i)
    k = rg.known_at(p)
    assert np.isnan(k["IP_YOY"].iloc[0]) and k["IP_YOY"].iloc[1] == 1.0  # published a month later
    assert k["SLOPE"].iloc[0] == 1.0  # market data: no lag


def test_threshold_modes_and_labels():
    i = idx(60)
    g = pd.Series(np.r_[np.zeros(30), np.ones(30)], index=i)
    inf = pd.Series(np.r_[np.ones(30), np.zeros(30)], index=i)
    full = rg.threshold_regimes(g, inf, "full")
    assert full.iloc[0] == "B" and full.iloc[-1] == "D"  # below/above the full-sample average of 0.5
    roll = rg.threshold_regimes(g, inf, "rolling", window=12)
    assert roll.iloc[:8].isna().all() and roll.iloc[9:].notna().all()  # needs 80% of the window (a missing month is tolerated)
    with pytest.raises(ValueError):
        rg.threshold_regimes(g, inf, "nope")


def test_full_sample_threshold_uses_the_future():
    i = idx(40)
    g = pd.Series(np.r_[np.full(20, 1.0), np.full(20, 3.0)], index=i)
    inf = pd.Series(1.0, index=i)
    full = rg.threshold_regimes(g, inf, "full")
    exp = rg.threshold_regimes(g, inf, "expanding", window=5)
    assert full.iloc[10] in ("B", "C")  # 1 < full mean 2: 'growth down' only because the future is known
    assert exp.iloc[10] in ("A", "D")  # with data up to then, growth equals its average: 'up'


def test_returns_by_regime_matches_next_month():
    i = idx(6)
    reg = pd.Series(["A", "B", "A", "B", "A", "B"], index=i)
    r = pd.DataFrame({"X": [0.0, 0.1, -0.1, 0.1, -0.1, 0.1]}, index=i)
    nxt = rg.returns_by_regime(r, reg, next_month=True).set_index("regime")
    same = rg.returns_by_regime(r, reg, next_month=False).set_index("regime")
    assert nxt.loc["A", "hit_rate"] == 1.0  # after an A month, X always rose
    assert same.loc["A", "hit_rate"] == 0.0


def test_rotation_targets_full_vs_expanding():
    i = idx(100)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0.01, 0.03, (100, 2)), index=i, columns=["X", "Y"])
    reg = pd.Series(np.where(np.arange(100) % 2, "A", "C"), index=i)
    full = rg.regime_rotation_targets(r, reg, "full", min_months=10)
    exp = rg.regime_rotation_targets(r, reg, "expanding", min_months=10)
    np.testing.assert_allclose(full.sum(axis=1), 1.0)
    np.testing.assert_allclose(exp.sum(axis=1), 1.0)
    assert not np.allclose(full.iloc[0], exp.iloc[0])


def test_nearest_months_respects_gap_and_horizon():
    i = idx(200)
    z = pd.DataFrame({"a": np.sin(np.arange(200) / 10), "b": np.cos(np.arange(200) / 10)}, index=i)
    an = rg.nearest_months(z, i[-1], k=5, gap_months=6, horizon=12)
    assert len(an) == 5
    assert (an["date"] <= i[-1] - pd.offsets.MonthEnd(12)).all()
    d = sorted(an["date"])
    assert all((b_.year - a_.year) * 12 + b_.month - a_.month > 6 for a_, b_ in zip(d, d[1:]))


def test_signal_scorecard_trigger_and_lead():
    i = idx(60)
    flags = pd.Series(0.0, index=i)
    flags.iloc[20:24] = 1.0
    usrec = pd.Series(0.0, index=i)
    usrec.iloc[30:36] = 1.0
    stocks = pd.Series(0.01, index=i)
    sc = rg.signal_scorecard(flags, usrec, stocks, within=24, quiet=12)
    assert len(sc) == 1
    assert sc["recession followed"].iloc[0] and sc["lead (months)"].iloc[0] == 10


def test_apply_filter_moves_risky_weight_to_cash():
    i = idx(3)
    t = pd.DataFrame({"SPY": [1.0, 1.0, 0.5], "AGG": [0.0, 0.0, 0.5]}, index=i)
    on = pd.Series([1.0, 0.0, 0.0], index=i)
    f = rg.apply_filter(t, on, ["SPY"])
    assert f.loc[i[0], "SPY"] == 1.0
    assert f.loc[i[1], "CASH"] == 1.0
    assert f.loc[i[2], "AGG"] == 0.5 and f.loc[i[2], "CASH"] == 0.5
    np.testing.assert_allclose(f.sum(axis=1), 1.0)


def test_real_data_panel_and_monitor():
    panel = rg.macro_panel()
    known = rg.known_at(panel)
    assert panel["IP_YOY"].first_valid_index().year == 1920
    feats = known[["IP_YOY", "CPI_YOY", "UNRATE_CHG12", "SLOPE", "CREDIT"]].dropna()
    z = rg.standardize(feats)
    an = rg.nearest_months(z, z.index[-1], k=10)
    per, summ = rg.analog_outcomes(rg.long_returns()[["US stocks"]], an, 12)
    assert summ.loc["US stocks", "analogs with data"] == 10
    pay = rg.realtime_vs_revised_payrolls()
    assert pay.loc["2024-07-31", "revision"] < -500  # payrolls were first reported far higher than today's data


pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

PAGE = str(Path(__file__).resolve().parents[1] / "apps" / "pages" / "regimes.py")


def test_regimes_page_runs_every_option():
    at = AppTest.from_file(PAGE, default_timeout=180)
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]
    at.button(key="rg_track").click()
    at.run(timeout=180)
    assert any(x.label.startswith("Correlation") for x in at.metric)
    for rule in ("gem", "faber", "top3", "bh"):
        at.selectbox(key="rg_rule").set_value(rule)
        at.run(timeout=180)
        assert not at.exception, (rule, [e.value for e in at.exception])
    for kind in ("payrolls_first", "credit"):
        at.selectbox(key="rg_kind").set_value(kind)
        at.run(timeout=180)
        assert not at.exception, (kind, [e.value for e in at.exception])
    at.radio(key="rg_wuni").set_value("etf")
    at.radio(key="rg_mode").set_value("full")
    at.toggle(key="rg_lag").set_value(False)
    at.radio(key="rg_dataset").set_value("class")
    at.run(timeout=180)
    assert not at.exception, [e.value for e in at.exception]
