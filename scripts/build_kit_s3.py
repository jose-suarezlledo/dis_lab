"""Build the Session 3 student data kit from the snapshot -> data/kits/robo_data_s3.csv

    py scripts/build_kit_s3.py

One CSV, month-end values in euros, base 100 at each series' first available month, from Dec 2015:
UCITS ETFs a euro investor can buy, Indexa Capital's ten published profiles (fund accounts 10-100k€,
net of Indexa's fees, to Jan 2025), the EUR/USD rate and the two short rates (% per year).
Students feed this file to the robo they build; the Robo-Advisor Lab reads the same snapshot, so the
numbers can be checked against each other. No live download happens here.

The kit is built from the frozen class snapshot (engine.data.CLASS_SNAPSHOT), never from the latest
refresh in data/, so rebuilding it always gives the same file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402

START = "2015-12-31"
ETFS = {
    "IWDA.AS": "World equity (iShares Core MSCI World, unhedged)",
    "IS3N.DE": "Emerging-market equity (iShares Core MSCI EM IMI)",
    "EUNH.DE": "Euro government bonds (iShares Core Euro Govt Bond)",
    "XG7S.DE": "Global government bonds hedged to EUR (Xtrackers)",
    "EUNA.DE": "Global aggregate bonds hedged to EUR (iShares Core, from Nov 2017)",
    "XEON.DE": "Cash at the euro overnight rate (Xtrackers)",
    "4GLD.DE": "Gold (Xetra-Gold)",
    "IWDE.L": "World equity hedged to EUR (iShares MSCI World EUR Hedged)",
    "V60A.DE": "Vanguard LifeStrategy 60 (one-fund 60/40, from Dec 2020)",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "kits" / "robo_data_s3.csv"


def main() -> int:
    snap = data.snapshot_dir()  # the frozen class snapshot, not the latest refresh
    px = data.load_prices(list(ETFS), data_dir=snap).loc[START:]
    monthly = px.resample("ME").last()
    ix = data.load_indexa("10-100k€", freq="ME").loc[START:]
    ix.columns = [f"INDEXA_{p}" for p in ix.columns]
    fx = data.load_fred("DEXUSEU", data_dir=snap).resample("ME").last().loc[START:].rename("EURUSD")
    us = data.load_fred("DTB3", data_dir=snap).resample("ME").last().loc[START:].rename("RATE_USD_3M_PCT")
    eu = data.load_fred(data.EURO_RF_SERIES, data_dir=snap).resample("ME").last().loc[START:].rename("RATE_EUR_3M_PCT")
    kit = monthly.join([ix, fx, us, eu], how="outer").sort_index()
    kit = kit.loc[:monthly.index[-1]]
    for c in list(ETFS) + list(ix.columns):
        first = kit[c].first_valid_index()
        kit[c] = kit[c] / kit.loc[first, c] * 100
    kit.index.name = "date"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# DIS Session 3 student data kit. Month-end values, EUR, base 100 at each series' first month.",
        "# Source: dis_lab snapshot (Yahoo Finance adjusted closes to 13 Oct 2026; FRED DEXUSEU, DTB3, "
        "IR3TIB01EZM156N; Indexa Capital published returns, fund accounts 10-100k EUR, to Jan 2025).",
        "# Columns: " + "; ".join(f"{k} = {v}" for k, v in ETFS.items()) + "; INDEXA_1..INDEXA_10 = Indexa "
        "Capital profiles 1-10 net of its fees; EURUSD = US dollars per euro (Fed H.10); RATE_*_3M_PCT = "
        "3-month rates in % per year (US T-bill; euro interbank, monthly).",
        "# Blank = the series did not exist yet. A real backtest must not treat a blank as 0% return.",
        "# Educational use only. Not investment advice.",
    ]
    with OUT.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(header) + "\n")
        kit.to_csv(f, float_format="%.4f", date_format="%Y-%m-%d")
    print(f"wrote {OUT} ({len(kit)} rows, {len(kit.columns)} columns, {kit.index[0]:%Y-%m-%d} to {kit.index[-1]:%Y-%m-%d})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
