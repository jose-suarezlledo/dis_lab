# DIS Lab

Shared engine, data snapshots and browser apps for **Data-driven Investment Strategies** (IESE MBA elective, Prof. José Suárez-Lledó).

> *AI can build you a strategy in a minute. This course teaches you whether to put money behind it.*

Educational material only. Nothing here is investment advice.

## Layout

| Folder | What it holds |
|---|---|
| `engine/` | One tested Python package used by every app and notebook. `data.py` loads the cached snapshots; `metrics.py` holds every performance and risk definition. |
| `data/` | CSV snapshots with download dates (`manifest.csv`). The apps never download live during class. |
| `apps/` | Streamlit apps, one per session tool. |
| `notebooks/` | Optional Colab notebooks (same engine) for students who want to code. |
| `scripts/` | `refresh_data.py`, the only script that downloads data. |
| `tests/` | `py -m pytest` |

## Quick start (Windows)

```powershell
py -m pip install -e ".[download,apps,dev]"
py -m pytest                      # run the tests
py scripts/refresh_data.py        # refresh every snapshot in data/universe.csv
```

```python
from engine import data, metrics

prices = data.load_prices(["SPY", "AGG", "VBINX"])
rf = data.risk_free_rate("2010-01-01", "2024-12-31")       # average 3m T-bill
metrics.summary_table(prices.loc["2010":"2024"], benchmark="SPY", rf=rf)
```

## Metric definitions

Every number in every app comes from `engine/metrics.py`:

- **Returns:** simple close-to-close returns on adjusted (total-return) closes.
- **CAGR:** (End/Start)^(1/years) − 1, with years = calendar days / 365.25.
- **Volatility:** std of periodic returns × √(periods per year). The frequency is inferred from the dates: 252 daily, 52 weekly, 12 monthly.
- **Sharpe:** (CAGR − rf) / volatility. The rf used is always reported.
- **Sortino:** (CAGR − rf) / downside deviation. Downside is measured below rf, over all periods.
- **Max drawdown:** worst fall from a running peak, measured on the NAV path.
- **Time under water:** longest stretch, in calendar days, below a previous peak (flagged if not yet recovered).
- **Calmar:** CAGR / |Max drawdown|.
- **% positive years / worst year:** full calendar years only.
- **Versus a benchmark:**
  - beta and Jensen's alpha (annualized);
  - correlation of **returns** (never of prices);
  - tracking error;
  - information ratio = (CAGR − benchmark CAGR) / tracking error;
  - upside and downside capture.
- **Turnover:** one-way, ½ × Σ|Δweight| per rebalance, annualized.

Series are compared on the dates they share (joined by date, never by row position). Missing data is never filled with a 0% return.

## Data

- **Prices:** Yahoo Finance via `yfinance`, adjusted closes. Adjusted histories change slightly every time dividends are re-adjusted, so the snapshot date in `data/manifest.csv` matters.
- **Macro:** FRED, latest vintage (`DTB3` risk-free rate, `DGS10`).
- **Indexa Capital:** profile returns as published by Indexa (base 100, EUR, net of fees). See `data/README.md`.

To add a ticker:
1. Add a row to `data/universe.csv`.
2. Run `py scripts/refresh_data.py TICKER`.
3. Commit the new CSV and `manifest.csv`.
