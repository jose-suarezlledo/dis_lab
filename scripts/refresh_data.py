"""Refresh the data snapshots in data/ (the only script that downloads anything).

    py scripts/refresh_data.py              # every ticker in data/universe.csv + FRED series
    py scripts/refresh_data.py SPY AGG      # just these tickers (must be in universe.csv)

Commit the updated CSVs and data/manifest.csv afterwards, so that the apps on
Streamlit Cloud serve the same snapshot everyone saw in class.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402

FRED_SERIES = {
    "DTB3": "3-month Treasury bill rate, % (risk-free rate)",
    "DGS10": "10-year Treasury constant-maturity yield, %",
}


def main(argv: list[str]) -> int:
    universe = data.load_universe()
    tickers = argv or list(universe.index)
    unknown = [t for t in tickers if t not in universe.index]
    if unknown:
        print(f"Not in data/universe.csv (add them first): {unknown}")
        return 1

    print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    prices = data.download_prices(tickers)
    data.save_prices(prices)
    for t in prices.columns:
        s = prices[t].dropna()
        print(f"  {t:10s} {s.index[0].date()} -> {s.index[-1].date()}  ({len(s)} rows)")
    if prices.attrs.get("failed"):
        print(f"FAILED (no data): {prices.attrs['failed']}")

    if not argv:
        for sid, label in FRED_SERIES.items():
            s = data.download_fred(sid)
            data.save_fred(s)
            print(f"  FRED {sid:6s} {s.index[0].date()} -> {s.index[-1].date()}  {label}")
    return 1 if prices.attrs.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
