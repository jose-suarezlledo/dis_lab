"""Bias checks: each one fires when it should and stays quiet when it should not."""

import numpy as np
import pandas as pd

from engine.checks import performance_checks


def levels(start, end, cols=("A", "B"), seed=0, drift=0.0003, vol=0.01):
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed)
    common = rng.normal(drift, vol, len(idx))
    data = {c: 100 * np.cumprod(1 + 0.8 * common + rng.normal(0, vol * 0.5, len(idx))) for c in cols}
    return pd.DataFrame(data, index=idx)


def titles(checks, severity=None):
    return [c.title for c in checks if severity is None or c.severity == severity]


def test_short_calm_window_warns():
    warns = " ".join(titles(performance_checks(levels("2023-01-02", "2024-12-31")), "warn"))
    assert "Short history" in warns and "No major sell-off" in warns


def test_long_window_lists_episodes():
    out = performance_checks(levels("2005-01-03", "2024-12-31"))
    assert any("Global financial crisis" in t for t in titles(out, "info"))
    assert not any("Short history" in t for t in titles(out))


def test_too_good_to_be_true():
    lv = levels("2005-01-03", "2024-12-31", drift=0.002, vol=0.005)
    warns = titles(performance_checks(lv), "warn")
    assert any("Sharpe" in t for t in warns) and any("CAGR" in t for t in warns)


def test_mixed_currencies_and_calendars():
    lv = levels("2005-01-03", "2024-12-31")
    ccy = {"A": "USD", "B": "EUR"}
    warns = " ".join(titles(performance_checks(lv, listing_currency=ccy), "warn"))
    assert "Mixed currencies" in warns and "closing times" in warns
    out = performance_checks(lv, listing_currency=ccy, display_currency="EUR", freq="weekly")
    assert not any("Mixed currencies" in t or "closing times" in t for t in titles(out, "warn"))


def test_poor_benchmark_fit():
    lv = levels("2005-01-03", "2024-12-31")
    lv["C"] = 100 * np.cumprod(1 + np.random.default_rng(9).normal(0, 0.01, len(lv)))
    warns = titles(performance_checks(lv, benchmark="A"), "warn")
    assert any("poor benchmark for C" in t for t in warns)
    assert not any("poor benchmark for B" in t for t in warns)


def test_zero_cost_rebalancing_warns():
    lv = levels("2005-01-03", "2024-12-31")
    pf = {"name": "My portfolio", "rebalance": "monthly", "cost_bps": 0}
    assert any("zero cost" in t for t in titles(performance_checks(lv, portfolio=pf), "warn"))
    pf["cost_bps"] = 10
    assert not any("zero cost" in t for t in titles(performance_checks(lv, portfolio=pf)))


def test_window_cut_by_youngest_series():
    lv = levels("2010-01-04", "2024-12-31")
    inc = {"A": pd.Timestamp("1995-01-03"), "B": pd.Timestamp("2010-01-04")}
    assert any("because of B" in t for t in titles(performance_checks(lv, inception=inc)))


def test_warnings_come_first():
    sev = [c.severity for c in performance_checks(levels("2023-01-02", "2024-12-31"))]
    assert sev == sorted(sev, key=lambda s: s != "warn")
