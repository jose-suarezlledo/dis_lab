"""Refresh the latest data in data/ (the only script that downloads anything).

    py scripts/refresh_data.py                   # every ticker in data/universe.csv + FRED series + Ken French factors
    py scripts/refresh_data.py SPY AGG           # just these tickers (must be in universe.csv)
    py scripts/refresh_data.py --summary out.md  # also append a Markdown report (GitHub Actions job summary)

Safety net, per series: a ticker that returns no data, comes back shorter than
the cached file, or shows a one-day move beyond engine.data.MAX_DAILY_MOVE
keeps its cached file and is flagged in the report. Everything else is saved.
Exit code 0 = all refreshed, 2 = some series kept their old data (read the
report), 1 = a usage error.

The scheduled job in .github/workflows/refresh-data.yml runs this monthly, runs
the tests and commits only if they pass. The frozen class snapshot in
data/snapshots/ is never touched.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import data  # noqa: E402

FRED_SERIES = {
    "DTB3": "3-month Treasury bill rate, % (risk-free rate)",
    "DGS10": "10-year Treasury constant-maturity yield, %",
    "DEXUSEU": "USD per EUR, Fed H.10 noon rate (currency conversion)",
    "IR3TIB01EZM156N": "Euro-area 3-month interbank rate, %, monthly (EUR risk-free rate)",
}
REVISION_NOTE = 0.001  # a past daily return that moved by more than this is worth a note


class Report:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, kind: str, sid: str, status: str, last: object = "", note: str = "") -> None:
        self.rows.append({"kind": kind, "id": sid, "status": status, "last_date": str(last), "note": note})
        flag = "" if status == "refreshed" else " <-- "
        print(f"  {kind:7s} {sid:16s} {status:10s} {str(last):10s} {note}{flag}")

    @property
    def problems(self) -> list[dict]:
        return [r for r in self.rows if r["status"] != "refreshed"]

    def markdown(self) -> str:
        n = len(self.rows)
        head = f"## Data refresh {dt.date.today().isoformat()}\n\n"
        head += (f"**{n - len(self.problems)} of {n} series refreshed.** "
                 + ("No problems." if not self.problems else
                    f"**{len(self.problems)} kept their old data: check them.**") + "\n\n")
        lines = ["| Kind | Series | Status | Last date | Note |", "|---|---|---|---|---|"]
        for r in sorted(self.rows, key=lambda r: (r["status"] == "refreshed", r["kind"], r["id"])):
            lines.append(f"| {r['kind']} | {r['id']} | {r['status']} | {r['last_date']} | {r['note']} |")
        return head + "\n".join(lines) + "\n"


def _old_price(t: str) -> pd.Series | None:
    try:
        return data.load_prices(t)[t].dropna()
    except KeyError:
        return None


def refresh_prices(tickers: list[str], universe: pd.DataFrame, rep: Report) -> None:
    print(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    try:
        new = data.download_prices(tickers)
    except Exception as e:  # network down, Yahoo blocking: keep everything
        for t in tickers:
            old = _old_price(t)
            rep.add("price", t, "failed", "" if old is None else old.index[-1].date(), f"download error: {e}")
        return
    failed = set(new.attrs.get("failed", []))
    asset_class = universe["asset_class"].to_dict()
    good = []
    for t in tickers:
        old = _old_price(t)
        old_last = None if old is None else old.index[-1]
        if t in failed:
            rep.add("price", t, "failed", "" if old is None else old_last.date(), "no data from Yahoo")
            continue
        s = new[t].dropna()
        if old is not None and s.index[-1] < old_last:
            rep.add("price", t, "shorter", old_last.date(), f"download ends {s.index[-1].date()}, before the cached file")
            continue
        moves = data.suspicious_moves(s.to_frame(t), asset_class={t: asset_class.get(t, "")},
                                      after={t: old_last} if old is not None else None)
        if not moves.empty:
            what = ", ".join(f"{d:%Y-%m-%d} {v:+.1%}" for d, v in zip(moves["date"], moves["return"]))
            rep.add("price", t, "suspicious", "" if old is None else old_last.date(), f"one-day move(s) {what}")
            continue
        note = ""
        if old is not None:
            rev = data.return_revision(old, s)
            if rev > REVISION_NOTE:
                note = f"past returns revised by up to {rev:.2%}"
        rep.add("price", t, "refreshed", s.index[-1].date(), note)
        good.append(t)
    if good:
        data.save_prices(new[good])


def refresh_fred(rep: Report) -> None:
    for sid, label in FRED_SERIES.items():
        try:
            old_last = data.load_fred(sid).index[-1]
        except KeyError:
            old_last = None
        try:
            s = data.download_fred(sid)
            if s.empty or (old_last is not None and s.index[-1] < old_last):
                raise ValueError(f"download ends {s.index[-1].date() if len(s) else 'empty'}, before the cached file")
        except Exception as e:
            rep.add("fred", sid, "failed", "" if old_last is None else old_last.date(), str(e))
            continue
        data.save_fred(s)
        rep.add("fred", sid, "refreshed", s.index[-1].date(), label)


def refresh_french(rep: Report) -> None:
    try:
        old_last = data.load_french().index[-1]
    except KeyError:
        old_last = None
    try:
        f = data.download_french()
        if f.empty or (old_last is not None and f.index[-1] < old_last):
            raise ValueError("download ends before the cached file")
    except Exception as e:
        rep.add("factors", "french_us_monthly", "failed", "" if old_last is None else old_last.date(), str(e))
        return
    data.save_french(f)
    rep.add("factors", "french_us_monthly", "refreshed", f.index[-1].date(), ", ".join(f.columns))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tickers", nargs="*", help="only these tickers (default: everything, plus FRED and Ken French)")
    ap.add_argument("--summary", metavar="FILE", help="append a Markdown report to this file")
    args = ap.parse_args(argv)

    universe = data.load_universe()
    tickers = args.tickers or list(universe.index)
    unknown = [t for t in tickers if t not in universe.index]
    if unknown:
        print(f"Not in data/universe.csv (add them first): {unknown}")
        return 1

    rep = Report()
    refresh_prices(tickers, universe, rep)
    if not args.tickers:
        refresh_fred(rep)
        refresh_french(rep)

    print()
    print(rep.markdown())
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as f:
            f.write(rep.markdown())
    return 2 if rep.problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
