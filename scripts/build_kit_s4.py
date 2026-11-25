"""Build the Session 4 (tactical asset allocation) student data kit -> data/kits/taa_data_s4.csv

    py scripts/build_kit_s4.py            # from the latest monthly refresh in data/ (default)
    py scripts/build_kit_s4.py --class    # from the frozen class snapshot (what the slides use)

One CSV of month-end adjusted closes in US dollars for a global, long-history TAA universe,
plus a T-bill total-return index (the "cash" asset) and the 3-month T-bill rate. Students point
the strategy lab they build with their AI assistant at this file: by URL
(https://raw.githubusercontent.com/jose-suarezlledo/dis_lab/main/data/kits/taa_data_s4.csv,
served with permissive CORS headers so a page can fetch it directly) or as an upload.

The kit is rebuilt by the monthly data-refresh job, so the URL version is never more than a
month stale. Blank cells mean the asset did not exist yet; a backtest must not read them as 0%.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402
from engine.portfolio import cash_from_rate  # noqa: E402

ASSETS = {
    "SPY": "US large caps (SPDR S&P 500, from Jan 1993)",
    "QQQ": "US tech / growth (Invesco Nasdaq-100, from Mar 1999)",
    "IWM": "US small caps (iShares Russell 2000, from May 2000)",
    "EFA": "Developed markets ex-US (iShares MSCI EAFE, from Aug 2001)",
    "EEM": "Emerging markets (iShares MSCI EM, from Apr 2003)",
    "AGG": "US investment-grade bonds (iShares Core US Aggregate, from Sep 2003)",
    "IEF": "US Treasuries 7-10y (iShares, from Jul 2002)",
    "TLT": "US Treasuries 20y+ (iShares, from Jul 2002)",
    "GLD": "Gold (SPDR Gold Shares, from Nov 2004)",
    "DBC": "Broad commodities (Invesco DB Commodity Index, from Feb 2006)",
    "VNQ": "US listed real estate (Vanguard Real Estate, from Sep 2004)",
    "VFINX": "US large caps, mutual fund with long history (Vanguard 500 Index, from 1976)",
    "VBMFX": "US bonds, mutual fund with long history (Vanguard Total Bond Market, from 1986)",
    "VUSTX": "Long US Treasuries, mutual fund with long history (Vanguard Long-Term Treasury, from 1986)",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "kits" / "taa_data_s4.csv"


def build(data_dir: Path) -> pd.DataFrame:
    px = data.load_prices(list(ASSETS), data_dir=data_dir)
    monthly = px.resample("ME").last()
    # drop the running month: its last price is not a month-end close
    last_price_day = px.index[-1]
    if monthly.index[-1] > last_price_day:
        monthly = monthly.iloc[:-1]
    monthly = monthly.loc["1986-12-31":]
    rate = data.load_fred("DTB3", data_dir=data_dir)
    tbill = cash_from_rate(rate, monthly.index).rename("TBILL")
    rate_m = rate.resample("ME").last().reindex(monthly.index).ffill().rename("TBILL_RATE_PCT")
    kit = monthly.join([tbill, rate_m], how="left")
    kit.index.name = "date"
    return kit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--class", dest="use_class", action="store_true", help="build from the frozen class snapshot")
    args = ap.parse_args()
    folder = data.snapshot_dir() if args.use_class else data.DATA_DIR
    kit = build(folder)
    man = data.load_manifest(folder)
    last = man.loc[man["kind"] == "price", "last_date"].max()
    header = [
        "# DIS Session 4 student data kit: month-end adjusted closes in US dollars (dividends reinvested).",
        f"# Source: Yahoo Finance via the dis_lab engine (prices to {last}); FRED DTB3 for the T-bill rate.",
        "# Columns: " + "; ".join(f"{k} = {v}" for k, v in ASSETS.items())
        + "; TBILL = value of cash earning the 3-month T-bill rate (base 100); TBILL_RATE_PCT = that rate, % per year.",
        "# Blank = the asset did not exist yet. A backtest must exclude it until it has prices, never read a blank as 0%.",
        "# A signal computed from the prices in row t is known at the close of month t and can only earn the return of month t+1.",
        "# Educational use only. Not investment advice. https://github.com/jose-suarezlledo/dis_lab",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        kit.to_csv(f, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"wrote {OUT} ({len(kit)} rows, {len(kit.columns)} columns, {kit.index[0]:%Y-%m-%d} to {kit.index[-1]:%Y-%m-%d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
