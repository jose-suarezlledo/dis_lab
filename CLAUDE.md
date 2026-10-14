# dis_lab: working notes

Course-wide rules are in `../CLAUDE.md`; metric definitions follow its §7. This file covers the code repo only.

**Repo:** https://github.com/jose-suarezlledo/dis_lab (public, `main`). Push only as `jose-suarezlledo`. Check with `gh api user --jq .login`, never the fund's `Global-Gradient` account.

## Status (14 Oct 2026)
- **Engine v0.1 done:**
  - `engine/metrics.py`: all §7 metrics plus benchmark-relative metrics, turnover, rolling CAGR/vol, `summary()` / `summary_table()`.
  - `engine/data.py`: cache, downloaders, `risk_free_rate()`, `load_indexa()`.
- **Data snapshot:** 46 tickers (`data/universe.csv`) to 13 Oct 2026, FRED DTB3/DGS10, Indexa profiles. Yahoo has no 13 Oct bar for the EU listings.
- **Tests:** 42 passing (`py -m pytest`).
- **Not built yet:** backtest engine (weights → NAV with costs, signal shifted one period), bias checks, PSR/DSR, apps, notebooks.

## Decisions taken
- "Annualized return" in Sharpe, Sortino and Calmar is the **CAGR** (geometric, calendar days / 365.25), not the arithmetic mean × 252. Sortino's target is rf, over all periods.
- Information ratio = (CAGR − benchmark CAGR) / tracking error. Alpha = Jensen's, from arithmetic periodic means.
- % positive years / worst year use **full calendar years only**. A series starting 4 Jan has a partial first year, so start a window on the prior year-end to include it.
- Comparisons use **common dates** (inner join on dates, no filling). Cross-market pairs (US vs EU listings, BTC 7-day) therefore get multi-day returns, and daily correlation is understated by close-time asynchrony. That's a teaching point; consider weekly/monthly correlation in the apps.
- The snapshot is CSV per ticker (`date, close` adjusted, in listing currency) plus `manifest.csv`. Downloads drop today's possibly-intraday bar.
- FRED via the keyless `fredgraph.csv` endpoint (latest vintage). ALFRED/vintage data will need `FRED_API_KEY` from the environment.

## Gotchas on the instructor's PC
- **Norton 360 intercepts HTTPS.** curl (inside yfinance) rejects its root certificate. `data._windows_ca_bundle()` merges certifi with the Windows root store, so verification stays on. If yfinance says "possibly delisted; no price data found" for everything, delete `%LOCALAPPDATA%\py-yfinance` (stale cookie) and retry.
- Python is `py` (Anaconda 3.12, pandas 2.2). Code must stay pandas-3-safe; pytest turns FutureWarnings into errors.
- The repo sits inside OneDrive. If OneDrive ever reports conflicts in `.git/`, pause OneDrive sync before git operations.

## Next steps
1. **Session 1 performance & risk analyzer** (Streamlit app, `apps/`), toolkit item #1, reused in S2/S3/S8:
   - pick assets or a weighted portfolio vs a benchmark from the universe;
   - date window, rf from DTB3 (stated);
   - NAV, drawdown and rolling charts; metrics table; calendar-year bars;
   - "what could be wrong here" panel.
   Then deploy on Streamlit Community Cloud and decide how apps import `engine` (requirements `-e .` vs a path shim).
2. Backtest engine + bias checks (needed by S3 robo simulator, S5 TAA).
