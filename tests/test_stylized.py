"""Stylized facts: moments, fat tails, clustering, missing-days arithmetic."""

import numpy as np
import pandas as pd
import pytest

from engine import stylized as st


def nav_from_returns(r, start="2000-01-03"):
    idx = pd.bdate_range(start, periods=len(r) + 1)
    return pd.Series(100 * np.concatenate([[1.0], np.cumprod(1 + np.asarray(r))]), index=idx)


def test_normal_tail_probability():
    assert st.normal_tail_probability(0) == pytest.approx(0.5)
    assert st.normal_tail_probability(3) == pytest.approx(0.0013499, rel=1e-4)


def test_normal_returns_have_no_fat_tails():
    nav = nav_from_returns(np.random.default_rng(0).normal(0.0003, 0.01, 50_000))
    assert abs(st.return_stats(nav)["excess_kurtosis"]) < 0.1
    tails = st.tail_counts(nav, sigmas=(3,))
    assert tails.loc[3, "actual"] == pytest.approx(tails.loc[3, "normal_expected"], rel=0.25)


def test_student_t_returns_have_fat_tails():
    nav = nav_from_returns(0.01 * np.random.default_rng(1).standard_t(3, 50_000) / np.sqrt(3))
    tails = st.tail_counts(nav, sigmas=(4,))
    assert tails.loc[4, "actual"] > 3 * tails.loc[4, "normal_expected"]
    assert st.return_stats(nav)["excess_kurtosis"] > 2


def test_volatility_drag_matches_plus_minus_example():
    nav = nav_from_returns([0.5, -0.5] * 50)  # average +0%, but you lose money
    s = st.return_stats(nav)
    assert s["arith_mean_ann"] == pytest.approx(0.0)
    assert s["cagr"] < 0
    assert s["vol_drag"] > 0


def test_volatility_clustering_shows_in_absolute_returns():
    vol = np.tile(np.repeat([0.005, 0.03], 500), 10)
    nav = nav_from_returns(np.random.default_rng(2).normal(0, 1, len(vol)) * vol)
    assert abs(st.autocorrelation(nav, 5).iloc[0]) < 0.1
    assert st.autocorrelation(nav, 5, absolute=True).iloc[0] > 0.2


def test_missing_days_effect_by_hand():
    nav = nav_from_returns([0.10, -0.10] + [0.0] * 98)
    out = st.missing_days_effect(nav, n=1)
    assert out["missed_best"] < out["all_days"] < out["missed_worst"]
    assert out["missed_both"] == pytest.approx(0.0, abs=1e-12)
