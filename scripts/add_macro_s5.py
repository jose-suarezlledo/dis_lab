"""Add the Session 5 (market regimes) macro series to the data cache, once.

    $env:FRED_API_KEY = "..."      # only needed for the ALFRED first-release vintages
    py scripts/add_macro_s5.py

1. FRED latest-vintage series (keyless endpoint) -> data/macro/ (the latest refresh; scripts/refresh_data.py keeps them
   fresh from now on) and data/snapshots/<class>/macro/ cut at the snapshot date (observations dated on or before it).
2. ALFRED first releases ("what was published the first time") of industrial production and payrolls, through the FRED
   API with output_type=4 -> data/macro/ALFRED_<id>_first.csv (+ snapshot copy). These are frozen: ALFRED's first
   release of a past month never changes, so they are not part of the monthly refresh. The key is read from the
   environment and never written anywhere.
"""

from __future__ import annotations

import io
import json
import os
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from engine import data  # noqa: E402

SERIES = ["INDPRO", "CPIAUCSL", "UNRATE", "PAYEMS", "T10Y2Y", "T10Y3M", "DGS2", "GS10", "TB3MS", "BAA", "BAA10Y",
          "SAHMREALTIME", "USREC", "RECPROUSM156N"]
ALFRED = ["INDPRO", "PAYEMS"]
SNAP_END = "2026-10-13"


def alfred_first_release(series_id: str, key: str) -> pd.Series:
    url = (f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={key}&file_type=json"
           "&output_type=4&realtime_start=1776-07-04&realtime_end=9999-12-31")
    ctx = None
    bundle = data._windows_ca_bundle()
    if bundle:
        import ssl
        ctx = ssl.create_default_context(cafile=bundle)
    with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
        obs = json.loads(r.read().decode("utf-8"))["observations"]
    df = pd.DataFrame(obs)
    df = df[df["value"] != "."]
    s = pd.Series(df["value"].astype(float).to_numpy(), index=pd.to_datetime(df["date"]), name=f"ALFRED_{series_id}_first")
    s.index.name = "date"
    return s.sort_index()


def main() -> int:
    snap = data.snapshot_dir()
    for sid in SERIES:
        try:
            s = data.download_fred(sid)
        except Exception as e:  # noqa: BLE001
            print("FAILED", sid, e)
            continue
        data.save_fred(s)
        data.save_fred(s.loc[:SNAP_END].rename(sid), data_dir=snap)
        print(f"{sid}: {s.index[0].date()} -> {s.index[-1].date()} ({len(s)} obs)")
    key = os.environ.get("FRED_API_KEY")
    if not key:
        print("FRED_API_KEY not set: ALFRED vintages skipped")
        return 0
    for sid in ALFRED:
        s = alfred_first_release(sid, key)
        data.save_fred(s)
        data.save_fred(s.loc[:SNAP_END].rename(s.name), data_dir=snap)
        print(f"{s.name}: {s.index[0].date()} -> {s.index[-1].date()} ({len(s)} obs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
