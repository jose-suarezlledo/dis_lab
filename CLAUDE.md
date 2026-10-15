# dis_lab: working notes

Course-wide rules are in `../CLAUDE.md`; metric definitions follow its §7. This file covers the code repo only.

**Repo:** https://github.com/jose-suarezlledo/dis_lab (public, `main`). Push only as `jose-suarezlledo`. Check with `gh api user --jq .login`, never the fund's `Global-Gradient` account.

## Status (15 Oct 2026)
- **Engine v0.2:**
  - `engine/metrics.py`: all §7 metrics, benchmark-relative metrics, turnover, rolling CAGR/vol/correlation, `to_frequency` (daily/weekly/monthly), `sharpe_standard_error` (Lo 2002), `STRESS_EPISODES` + `episode_returns`, `summary()` / `summary_table()`.
  - `engine/data.py`: cache, downloaders, `risk_free_rate()`, `convert_currency()` (USD/EUR via Fed DEXUSEU), `load_indexa()`.
  - `engine/portfolio.py`: `fixed_weight_portfolio()` (target weights, drift, period-end rebalancing, costs on traded value, turnover). Seed of the backtest engine.
  - `engine/stylized.py`: return moments, fat tails vs normal (`tail_counts`), autocorrelation of returns and |returns|, volatility drag, missing best/worst days.
  - `engine/checks.py`: `performance_checks()`, the automatic "what could be wrong here" list mapped to the §7 bias checklist.
- **Apps:** `apps/performance_analyzer.py` (toolkit #1) with 5 class scenarios; shared `apps/dis_ui.py` (palette, formats, glossary).
- **Data snapshot:** 46 tickers to 13 Oct 2026, FRED DTB3/DGS10/DEXUSEU/IR3TIB01EZM156N, Indexa profiles.
- **Tests:** 90 passing (`py -m pytest`), including AppTest smoke tests of every scenario × currency × frequency. Also verified under pandas 3.0.6 + Streamlit 1.66 (what Streamlit Cloud installs).
- **Not built yet:** signal-based backtest engine (weights from signals, shifted one period), PSR/DSR, walk-forward, notebooks.

## Decisions taken
- "Annualized return" in Sharpe, Sortino and Calmar is the **CAGR** (geometric, calendar days / 365.25), not the arithmetic mean × 252. Sortino's target is rf, over all periods.
- Information ratio = (CAGR − benchmark CAGR) / tracking error. Alpha = Jensen's, from arithmetic periodic means.
- % positive years / worst year use **full calendar years only**. A series starting 4 Jan has a partial first year, so start a window on the prior year-end to include it.
- Comparisons use **common dates** (inner join on dates, no filling). Cross-market pairs (US vs EU listings, BTC 7-day) therefore get multi-day returns, and daily correlation is understated by close-time asynchrony. That's a teaching point; consider weekly/monthly correlation in the apps.
- The snapshot is CSV per ticker (`date, close` adjusted, in listing currency) plus `manifest.csv`. Downloads drop today's possibly-intraday bar.
- FRED via the keyless `fredgraph.csv` endpoint (latest vintage). ALFRED/vintage data will need `FRED_API_KEY` from the environment.
- **Currency conversion uses FRED `DEXUSEU`, never Yahoo `EURUSD=X`** (bad ticks: 17% fake daily moves in 2008, up to 15% off the Fed rate). Unhedged; dates matched exactly, no filling.
- **EUR risk-free rate:** euro-area 3m interbank (`IR3TIB01EZM156N`, OECD monthly, lags ~9 months). Apps default to it when the currency is EUR, otherwise DTB3; always shown on screen.
- **Portfolios:** rebalance on the last common trading day of each period, never on the sample's last day; cost = Σ|Δw| × bps on NAV; initial purchase free. A portfolio's window starts when every series in the comparison exists, so its turnover/costs cover the same window as the metrics.
- **Frequency:** `to_frequency` keeps real last dates (no relabelling to period end) and the first observation, so CAGR is identical at every frequency.
- **Apps import the engine with a `sys.path` shim** at the top of each app (its own folder + repo root), not `-e .` in requirements. Works locally, on Streamlit Cloud and in AppTest.
- Stress episodes are S&P 500 closing peak → trough (verified against SPY in the snapshot).

## Gotchas on the instructor's PC
- **Norton 360 intercepts HTTPS.** curl (inside yfinance) rejects its root certificate. `data._windows_ca_bundle()` merges certifi with the Windows root store, so verification stays on. If yfinance says "possibly delisted; no price data found" for everything, delete `%LOCALAPPDATA%\py-yfinance` (stale cookie) and retry.
- Python is `py` (Anaconda 3.12, pandas 2.2). Code must stay pandas-3-safe; pytest turns FutureWarnings into errors.
- The repo sits inside OneDrive. If OneDrive ever reports conflicts in `.git/`, pause OneDrive sync before git operations.

## Next steps
1. **Deploy** `apps/performance_analyzer.py` on Streamlit Community Cloud (share.streamlit.io, signed in as `jose-suarezlledo`; repo `jose-suarezlledo/dis_lab`, branch `main`, main file `apps/performance_analyzer.py`, Python 3.12). Record the URL here and in `Sessions/1_.../CLAUDE.md`.
2. Backtest engine: signal → weights shifted one period → `portfolio`-style NAV with costs; bias checks for look-ahead and parameter snooping; PSR/DSR. Needed by S3 robo simulator and S5 TAA.
3. Extend the analyzer for S3 (Indexa profiles as series, hedged vs unhedged) and S8 (PCA/contribution) rather than writing new apps.
