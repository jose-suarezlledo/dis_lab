# dis_lab: working notes

Course-wide rules are in `../CLAUDE.md`; metric definitions follow its §7. This file covers the code repo only.

**Repo:** https://github.com/jose-suarezlledo/dis_lab (public, `main`). Push only as `jose-suarezlledo`. Check with `gh api user --jq .login`, never the fund's `Global-Gradient` account.

## Status (22 Oct 2026)
- **Engine v0.3** (Session 2 additions marked S2):
  - `engine/metrics.py`: all §7 metrics, benchmark-relative metrics, turnover, rolling CAGR/vol/correlation, `to_frequency` (daily/weekly/monthly), `sharpe_standard_error` (Lo 2002), `STRESS_EPISODES` + `episode_returns`, `summary()` / `summary_table()`.
  - `engine/data.py`: cache, downloaders, `risk_free_rate()`, `convert_currency()` (USD/EUR via Fed DEXUSEU), `load_indexa()`.
  - `engine/portfolio.py`: `fixed_weight_portfolio()` (target weights, drift, period-end rebalancing, costs on traded value, turnover). Seed of the backtest engine. S2: `cash_from_rate()` (cash NAV from a quoted rate, e.g. DTB3) and `vol_target_portfolio()` (weight = min(target / trailing vol, cap), set at the close and applied the next day; `lag=0` reproduces look-ahead for teaching).
  - S2: `engine/factors.py`: growth, annualized and rolling returns, decade table and worst drawdown of monthly long-short factor returns. `data.load_french()` / `download_french()`: Ken French US factors (MKT_RF, SMB, HML, RMW, CMA, MOM, RF) since 1926, `data/factors/french_us_monthly.csv`.
  - S2: `metrics.correlation_matrix(levels, on='returns'|'levels')` (levels only to show the bug) and `metrics.down_market_table()`.
  - `engine/stylized.py`: return moments, fat tails vs normal (`tail_counts`), autocorrelation of returns and |returns|, volatility drag, missing best/worst days.
  - `engine/checks.py`: `performance_checks()`, the automatic "what could be wrong here" list mapped to the §7 bias checklist. S2: notes for rule-based products (live record only; pre-launch backtests) and a warning for thematic funds.
- **Apps:** `apps/performance_analyzer.py` (toolkit #1); shared `apps/dis_ui.py` (palette, formats, glossary). Scenarios are grouped by class session (sidebar radio `Class`; `?session=2` in the URL opens Session 2): 5 for S1, 6 for S2 (equal weight, single factors, multi-factor/rotation, alternatives, UCITS factors in EUR, themes). New **Diversification** tab: correlation matrix with a returns / price-levels toggle, and how each series did when the benchmark fell (the heatmap moved there from All metrics).
- **Data snapshot:** 68 tickers to 13 Oct 2026 (S2 added 22 on 21 Oct, downloaded then and cut to the same end date), FRED DTB3/DGS10/DEXUSEU/IR3TIB01EZM156N, Indexa profiles, Ken French factors to Aug 2026.
- **Tests:** 112 passing (22 Oct 2026) (`py -m pytest`), including AppTest smoke tests of every scenario × currency × frequency. Also verified under pandas 3.0.6 + Streamlit 1.66 (what Streamlit Cloud installs).
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
- **Yahoo's Xetra lines of the iShares/Xtrackers world factor ETFs have bad ticks** (±20–30% one-day reversals in 2014, 5–6 Jun 2017, Oct 2025): the snapshot uses the London USD lines (IWMO/IWQU/IWVL/MVOL.L), converted to EUR by the engine. BTAL dropped (−33%/+49% on 29–30 Apr 2015). Run a max-|daily return| scan on every new ticker before adding it.
- Asset classes `Smart beta`, `Factor equity`, `Multi-factor`, `Alternative`, `Thematic` drive the rule-based-product checks.

## Gotchas on the instructor's PC
- **Norton 360 intercepts HTTPS.** curl (inside yfinance) rejects its root certificate. `data._windows_ca_bundle()` merges certifi with the Windows root store, so verification stays on. If yfinance says "possibly delisted; no price data found" for everything, delete `%LOCALAPPDATA%\py-yfinance` (stale cookie) and retry.
- Python is `py` (Anaconda 3.12, pandas 2.2). Code must stay pandas-3-safe; pytest turns FutureWarnings into errors.
- The repo sits inside OneDrive. If OneDrive ever reports conflicts in `.git/`, pause OneDrive sync before git operations.

## Next steps
1. **Deployed** 19 Oct 2026: https://dislab-performance-analyzer-s1.streamlit.app/ (Streamlit Community Cloud, from `main`, `apps/performance_analyzer.py`). Pushing to `main` redeploys it. Free apps sleep after days without traffic: open it ~15 min before class.
2. Backtest engine: signal → weights shifted one period → `portfolio`-style NAV with costs; bias checks for look-ahead and parameter snooping; PSR/DSR. Needed by S3 robo simulator and S5 TAA.
3. S2 (22 Oct 2026): pushed with the Session 2 scenarios; check the live app shows Class: Session 1 by default and `?session=2` works.
4. Extend the analyzer for S3 (Indexa profiles as series, hedged vs unhedged) and S8 (PCA/contribution) rather than writing new apps.
