"""Data loading and caching.

Class never depends on a live download: every app reads the CSV snapshots in
`data/`, and `scripts/refresh_data.py` is the only thing that downloads.

Layout
------
data/universe.csv           tickers offered in the apps (name, asset class, currency, UCITS flag)
data/prices/<TICKER>.csv    date, close   (adjusted = total-return close, from yfinance)
data/macro/<SERIES>.csv     date, value   (FRED, as published today, not vintage data)
data/indexa/...csv          Indexa Capital published profile returns (base 100)
data/manifest.csv           one row per cached file: source, first/last date, download date

Prices are never forward-filled or zero-filled here: a ticker has NaN before
its first trading day and on days its exchange was closed. Use
`engine.metrics.align` to compare series on their common dates.
"""

from __future__ import annotations

import datetime as dt
import io
import urllib.request
from pathlib import Path
from typing import Iterable

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"

__all__ = [
    "DATA_DIR",
    "load_universe",
    "load_manifest",
    "cached_tickers",
    "load_prices",
    "download_prices",
    "save_prices",
    "load_fred",
    "download_fred",
    "save_fred",
    "risk_free_rate",
    "load_indexa",
]


def _dir(data_dir: Path | str | None) -> Path:
    return Path(data_dir) if data_dir is not None else DATA_DIR


def _file_stem(ticker: str) -> str:
    return ticker.replace("/", "_")


# --------------------------------------------------------------------------- catalogue


def load_universe(data_dir: Path | str | None = None) -> pd.DataFrame:
    """Tickers offered in the apps, indexed by ticker."""
    return pd.read_csv(_dir(data_dir) / "universe.csv", dtype=str).set_index("ticker")


def load_manifest(data_dir: Path | str | None = None) -> pd.DataFrame:
    path = _dir(data_dir) / "manifest.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["kind", "id", "source", "first_date", "last_date", "rows", "downloaded"]
        )
    return pd.read_csv(path, dtype=str)


def _update_manifest(rows: list[dict], data_dir: Path) -> None:
    man = load_manifest(data_dir)
    new = pd.DataFrame(rows).astype(str)
    keys = set(zip(new["kind"], new["id"]))
    keep = man[[(k, i) not in keys for k, i in zip(man["kind"], man["id"])]]
    out = pd.concat([keep, new], ignore_index=True).sort_values(["kind", "id"])
    out.to_csv(data_dir / "manifest.csv", index=False)


def cached_tickers(data_dir: Path | str | None = None) -> list[str]:
    man = load_manifest(data_dir)
    return sorted(man.loc[man["kind"] == "price", "id"])


# --------------------------------------------------------------------------- prices


def load_prices(
    tickers: str | Iterable[str],
    start: str | None = None,
    end: str | None = None,
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Wide table of cached adjusted closes, one column per ticker, union of dates.

    No filling: each column is NaN before its inception and on market holidays.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    folder = _dir(data_dir) / "prices"
    missing = [t for t in tickers if not (folder / f"{_file_stem(t)}.csv").exists()]
    if missing:
        raise KeyError(
            f"not in the data cache: {missing}. Add them to data/universe.csv and run "
            "`py scripts/refresh_data.py`, or fetch them live with download_prices()."
        )
    cols = []
    for t in tickers:
        s = pd.read_csv(folder / f"{_file_stem(t)}.csv", index_col="date", parse_dates=True)
        cols.append(s["close"].rename(t))
    df = pd.concat(cols, axis=1).sort_index()
    df.index.name = "date"
    return df.loc[start:end]


def _windows_ca_bundle() -> str | None:
    """certifi + the Windows root store, as one PEM file (Windows only).

    yfinance downloads through curl, which trusts only certifi's bundle. Antivirus
    HTTPS scanning (e.g. Norton) and corporate proxies re-sign traffic with a root
    certificate that lives in the Windows store, so curl rejects it while browsers
    and Python's own ssl accept it. Merging the two keeps verification on.
    """
    import ssl
    import sys
    import tempfile

    if sys.platform != "win32":
        return None
    import certifi

    path = Path(tempfile.gettempdir()) / "dis_lab_ca_bundle.pem"
    pems = [Path(certifi.where()).read_text(encoding="ascii")]
    for cert, encoding, _trust in ssl.enum_certificates("ROOT"):
        if encoding == "x509_asn":
            pems.append(ssl.DER_cert_to_PEM_cert(cert))
    path.write_text("\n".join(pems), encoding="ascii")
    return str(path)


def download_prices(tickers: str | Iterable[str], start: str | None = None) -> pd.DataFrame:
    """Live download of adjusted (total-return) daily closes from Yahoo Finance.

    Tickers that return no data are dropped and listed in `df.attrs["failed"]`.
    """
    import yfinance as yf  # imported lazily: apps reading the cache don't need it

    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    kwargs = {"start": start} if start else {"period": "max"}
    bundle = _windows_ca_bundle()
    if bundle:
        from curl_cffi import requests as curl_requests

        kwargs["session"] = curl_requests.Session(impersonate="chrome", verify=bundle)
    raw = yf.download(
        tickers, auto_adjust=True, progress=False, threads=True, group_by="column", **kwargs
    )
    close = raw["Close"]
    if isinstance(close, pd.Series):
        close = close.to_frame(tickers[0])
    close = close.reindex(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None).normalize()
    close.index.name = "date"
    close = close.loc[close.index < pd.Timestamp.today().normalize()]  # today's bar may be intraday
    failed = [t for t in tickers if close[t].dropna().empty]
    close = close.drop(columns=failed)
    close.attrs["failed"] = failed
    return close


def save_prices(
    prices: pd.DataFrame,
    source: str = "yfinance (auto_adjust=True)",
    data_dir: Path | str | None = None,
    downloaded: dt.date | None = None,
) -> None:
    """Write one CSV per column into data/prices and record it in the manifest."""
    data_dir = _dir(data_dir)
    folder = data_dir / "prices"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = (downloaded or dt.date.today()).isoformat()
    rows = []
    for t in prices.columns:
        s = prices[t].dropna().rename("close")
        if s.empty:
            continue
        s.index.name = "date"
        s.to_csv(folder / f"{_file_stem(t)}.csv", float_format="%.6f", date_format="%Y-%m-%d")
        rows.append(
            {
                "kind": "price",
                "id": t,
                "source": source,
                "first_date": s.index[0].date(),
                "last_date": s.index[-1].date(),
                "rows": len(s),
                "downloaded": stamp,
            }
        )
    if rows:
        _update_manifest(rows, data_dir)


# --------------------------------------------------------------------------- macro (FRED)


def load_fred(series_id: str, data_dir: Path | str | None = None) -> pd.Series:
    path = _dir(data_dir) / "macro" / f"{series_id}.csv"
    if not path.exists():
        raise KeyError(f"FRED series {series_id} not cached; run `py scripts/refresh_data.py`")
    s = pd.read_csv(path, index_col="date", parse_dates=True)["value"]
    return s.rename(series_id)


def download_fred(series_id: str) -> pd.Series:
    """Latest-vintage series from FRED's public CSV endpoint (no API key needed).

    For real-time (vintage) data use ALFRED with the FRED_API_KEY environment
    variable; this function is for current-vintage snapshots only.
    """
    with urllib.request.urlopen(FRED_CSV_URL.format(series_id=series_id), timeout=60) as resp:
        text = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(text), na_values=["."])
    df.columns = ["date", "value"]
    s = df.set_index(pd.to_datetime(df["date"]))["value"].astype(float).dropna()
    s.index.name = "date"
    return s.rename(series_id)


def save_fred(series: pd.Series, data_dir: Path | str | None = None) -> None:
    data_dir = _dir(data_dir)
    folder = data_dir / "macro"
    folder.mkdir(parents=True, exist_ok=True)
    s = series.dropna().rename("value")
    s.index.name = "date"
    s.to_csv(folder / f"{series.name}.csv", date_format="%Y-%m-%d")
    _update_manifest(
        [
            {
                "kind": "fred",
                "id": series.name,
                "source": "FRED fredgraph.csv (latest vintage)",
                "first_date": s.index[0].date(),
                "last_date": s.index[-1].date(),
                "rows": len(s),
                "downloaded": dt.date.today().isoformat(),
            }
        ],
        data_dir,
    )


def risk_free_rate(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    series_id: str = "DTB3",
    data_dir: Path | str | None = None,
) -> float:
    """Average annual risk-free rate over a window, as a decimal (0.02 = 2%).

    Default: 3-month US Treasury bill, secondary market (FRED DTB3, in percent).
    """
    s = load_fred(series_id, data_dir).loc[start:end]
    if s.empty:
        raise ValueError(f"no {series_id} data between {start} and {end}")
    return float(s.mean() / 100)


# --------------------------------------------------------------------------- Indexa Capital


INDEXA_TIERS = ("<10k€", "10-100k€", ">100k€")


def load_indexa(
    tier: str = "10-100k€",
    freq: str | None = "ME",
    data_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Indexa Capital published returns for risk profiles 1-10 (base 100), EUR.

    The file has one row per *calendar* day, with weekend values repeated, so by
    default it is resampled to month-end (freq="ME"). Pass freq=None for the raw
    daily rows. Columns are the profile numbers 1..10.
    """
    if tier not in INDEXA_TIERS:
        raise ValueError(f"tier must be one of {INDEXA_TIERS}")
    path = _dir(data_dir) / "indexa" / "indexa_stats_returns_mutual.csv"
    raw = pd.read_csv(path, sep=";", decimal=",", index_col="Fecha", parse_dates=True)
    cols = [c for c in raw.columns if c.startswith(f"{tier} - ")]
    df = raw[cols].copy()
    df.columns = [int(c.rsplit("- ", 1)[1]) for c in cols]
    df.index.name = "date"
    df = df.dropna(how="all")
    if freq:
        df = df.resample(freq).last()
    return df
