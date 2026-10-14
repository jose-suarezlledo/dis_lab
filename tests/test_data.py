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


# --------------------------------------------------------------------------- the real snapshot


def test_cached_snapshot_integrity():
    man = data.load_manifest()
    prices = man[man["kind"] == "price"]
    if prices.empty:
        pytest.skip("no price snapshot yet")
    for t in prices["id"]:
        s = data.load_prices(t)[t].dropna()
        assert s.index.is_monotonic_increasing and s.index.is_unique, t
        assert (s > 0).all(), t
        row = prices.set_index("id").loc[t]
        assert str(s.index[-1].date()) == row["last_date"], t


def test_real_indexa_file_if_present():
    path = data.DATA_DIR / "indexa" / "indexa_stats_returns_mutual.csv"
    if not path.exists():
        pytest.skip("Indexa file not in data/indexa")
    df = data.load_indexa("10-100k€")
    assert list(df.columns) == list(range(1, 11))
    assert df.iloc[0].eq(100).all()
