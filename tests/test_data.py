"""Data cache: round trips, no silent filling, Indexa parsing, snapshot integrity."""

import numpy as np
import pandas as pd
import pytest

from engine import data


@pytest.fixture
def tmp_cache(tmp_path):
    idx = pd.bdate_range("2024-01-01", periods=10)
    prices = pd.DataFrame(
        {"OLD": np.linspace(100, 109, 10), "NEW": np.r_[[np.nan] * 4, np.linspace(50, 55, 6)]},
        index=idx,
    )
    data.save_prices(prices, data_dir=tmp_path)
    return tmp_path, prices


def test_price_round_trip_and_manifest(tmp_cache):
    folder, prices = tmp_cache
    back = data.load_prices(["OLD", "NEW"], data_dir=folder)
    pd.testing.assert_frame_equal(back, prices, check_freq=False, check_names=False)
    man = data.load_manifest(folder)
    assert set(man["id"]) == {"OLD", "NEW"}
    assert man.set_index("id").loc["NEW", "first_date"] == "2024-01-05"
    assert data.cached_tickers(folder) == ["NEW", "OLD"]


def test_late_starter_is_not_filled(tmp_cache):
    folder, _ = tmp_cache
    back = data.load_prices(["OLD", "NEW"], data_dir=folder)
    assert back["NEW"].iloc[:4].isna().all()


def test_resaving_replaces_manifest_row(tmp_cache):
    folder, prices = tmp_cache
    data.save_prices(prices[["OLD"]] * 2, data_dir=folder)
    man = data.load_manifest(folder)
    assert (man["id"] == "OLD").sum() == 1


def test_missing_ticker_raises_helpful_error(tmp_cache):
    folder, _ = tmp_cache
    with pytest.raises(KeyError, match="refresh_data"):
        data.load_prices(["NOPE"], data_dir=folder)


def test_load_prices_date_slice(tmp_cache):
    folder, _ = tmp_cache
    df = data.load_prices("OLD", start="2024-01-03", end="2024-01-05", data_dir=folder)
    assert len(df) == 3


def test_fred_round_trip_and_risk_free_rate(tmp_path):
    s = pd.Series([4.0, 5.0, np.nan, 6.0], index=pd.bdate_range("2024-01-01", periods=4), name="DTB3")
    data.save_fred(s, data_dir=tmp_path)
    back = data.load_fred("DTB3", data_dir=tmp_path)
    assert len(back) == 3
    assert data.risk_free_rate("2024-01-01", "2024-01-31", data_dir=tmp_path) == pytest.approx(0.05)


def test_indexa_parser_handles_semicolon_decimal_comma_and_weekends(tmp_path):
    (tmp_path / "indexa").mkdir()
    header = "Fecha;" + ";".join(f'"10-100k€ - {i}"' for i in range(1, 11))
    days = pd.date_range("2016-01-29", "2016-02-29", freq="D")
    rows = [f"{d.date()};" + ";".join(f"{100 + k:.2f}".replace(".", ",") for _ in range(10)) for k, d in enumerate(days)]
    (tmp_path / "indexa" / "indexa_stats_returns_mutual.csv").write_text(
        "\n".join([header, *rows]), encoding="utf-8"
    )
    monthly = data.load_indexa("10-100k€", data_dir=tmp_path)
    assert list(monthly.columns) == list(range(1, 11))
    assert list(monthly.index) == list(pd.to_datetime(["2016-01-31", "2016-02-29"]))
    assert monthly.iloc[0, 0] == pytest.approx(102.0)  # 31 Jan value, decimal comma parsed
    daily = data.load_indexa("10-100k€", freq=None, data_dir=tmp_path)
    assert len(daily) == len(days)


def test_convert_currency(tmp_path):
    idx = pd.bdate_range("2024-01-01", periods=4)
    fx = pd.Series([1.10, 1.20, np.nan, 1.25], index=idx, name="DEXUSEU")
    data.save_fred(fx, data_dir=tmp_path)
    p = pd.DataFrame({"US": [110.0, 120.0, 130.0, 125.0], "EU": [100.0, 100.0, 100.0, 100.0]}, index=idx)
    ccy = {"US": "USD", "EU": "EUR"}
    eur = data.convert_currency(p, "EUR", ccy, data_dir=tmp_path)
    assert eur["US"].iloc[0] == pytest.approx(100.0)  # 110 USD / 1.10
    assert np.isnan(eur["US"].iloc[2])  # no Fed rate that day: missing, never filled
    pd.testing.assert_series_equal(eur["EU"], p["EU"])
    usd = data.convert_currency(p, "USD", ccy, data_dir=tmp_path)
    assert usd["EU"].iloc[1] == pytest.approx(120.0)
    with pytest.raises(KeyError):
        data.convert_currency(p.assign(X=1.0), "EUR", ccy, data_dir=tmp_path)


# --------------------------------------------------------------------------- quality scan (refresh safety net)


def test_suspicious_moves_flags_bad_ticks_not_volatility():
    idx = pd.bdate_range("2024-01-01", periods=6)
    px = pd.DataFrame({"OK": [100, 110, 95, 100, 112, 100],        # ±15%: volatile but believable
                       "BAD": [100, 101, 160, 102, 103, 104],       # a spike and its reversal
                       "BTC": [100, 135, 100, 101, 102, 103]},      # +35%: fine for crypto
                      index=idx)
    out = data.suspicious_moves(px, asset_class={"BTC": "Crypto"})
    assert list(out["ticker"]) == ["BAD", "BAD"]
    assert [d.date().isoformat() for d in out["date"]] == ["2024-01-03", "2024-01-04"]
    # Only the rows a refresh adds are checked when `after` is given
    assert data.suspicious_moves(px[["BAD"]], after={"BAD": idx[3]}).empty
    assert len(data.suspicious_moves(px[["BAD"]], after={"BAD": idx[2]})) == 1  # the return into day 4 is new
    assert not data.suspicious_moves(px, asset_class={"BTC": "Crypto"}, limits={"Crypto": 0.3}).empty


def test_return_revision_ignores_dividend_rescaling():
    idx = pd.bdate_range("2024-01-01", periods=5)
    old = pd.Series([100, 101, 103, 102, 105], index=idx, dtype=float)
    assert data.return_revision(old, old * 0.97) < 1e-12  # a re-adjustment rescales every price
    new = old.copy()
    new.iloc[2] = 110  # a corrected print changes two daily returns
    assert data.return_revision(old, new) > 0.05
    assert data.return_revision(old, old.iloc[-1:]) == 0.0  # no common returns


# --------------------------------------------------------------------------- the real data


def test_fx_snapshot_is_clean():
    fx = data.load_fred(data.FX_SERIES)
    assert fx.pct_change().abs().max() < 0.06  # Yahoo's EURUSD=X had 17% fake daily moves


@pytest.mark.parametrize("folder", [None, "class snapshot"])
def test_cached_data_integrity(folder):
    folder = data.snapshot_dir() if folder else None
    man = data.load_manifest(folder)
    prices = man[man["kind"] == "price"]
    if prices.empty:
        pytest.skip("no price snapshot yet")
    for t in prices["id"]:
        s = data.load_prices(t, data_dir=folder)[t].dropna()
        assert s.index.is_monotonic_increasing and s.index.is_unique, t
        assert (s > 0).all(), t
        row = prices.set_index("id").loc[t]
        assert str(s.index[-1].date()) == row["last_date"], t


def test_class_snapshot_is_frozen_and_latest_is_not_older():
    """The class scenarios read data/snapshots/2026-10-13; data/ is the latest refresh."""
    snap = data.snapshot_dir()
    assert data.CLASS_SNAPSHOT in data.snapshot_tags()
    frozen = data.load_manifest(snap).set_index(["kind", "id"])
    assert frozen.loc[("price", "SPY"), "last_date"] == "2026-10-13"
    assert frozen.loc[("price", "XG7S.DE"), "last_date"] == "2026-10-13"  # S3 tickers were cut to the same day
    latest = data.load_manifest().set_index(["kind", "id"])
    for key in frozen.index:
        assert key in latest.index, key
        assert latest.loc[key, "last_date"] >= frozen.loc[key, "last_date"], key
    # No bad ticks slipped into the latest refresh (the scan the refresh job runs)
    uni = data.load_universe()
    px = data.load_prices([t for t in data.cached_tickers() if t in uni.index])
    bad = data.suspicious_moves(px, asset_class=uni["asset_class"].to_dict(), after={"BTC-USD": "2020-03-13"})
    assert bad.empty, bad


def test_real_indexa_file_if_present():
    path = data.DATA_DIR / "indexa" / "indexa_stats_returns_mutual.csv"
    if not path.exists():
        pytest.skip("Indexa file not in data/indexa")
    df = data.load_indexa("10-100k€")
    assert list(df.columns) == list(range(1, 11))
    assert df.iloc[0].eq(100).all()
