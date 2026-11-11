# dis_lab: working notes

Course-wide rules are in `../CLAUDE.md`; metric definitions follow its §7. This file covers the code repo only.

**Repo:** https://github.com/jose-suarezlledo/dis_lab (public, `main`). Push only as `jose-suarezlledo`. Check with `gh api user --jq .login`, never the fund's `Global-Gradient` account.

## Status (3 Nov 2026)
- **Engine v0.4** (Session 3 additions marked S3):
  - `engine/metrics.py`: all §7 metrics, benchmark-relative metrics, turnover, rolling CAGR/vol/correlation, `to_frequency`, `sharpe_standard_error` (Lo 2002), `STRESS_EPISODES` + `episode_returns`, `summary()` / `summary_table()`, `correlation_matrix(on='returns'|'levels')`, `down_market_table()`. S3: `time_under_water` no longer trips a FutureWarning on int-named series.
  - `engine/data.py`: cache, downloaders, `risk_free_rate()`, `convert_currency()` (USD/EUR via Fed DEXUSEU), `load_indexa()`, Ken French loaders.
  - `engine/portfolio.py`: `fixed_weight_portfolio()`, `cash_from_rate()`, `vol_target_portfolio()`. S3: `drift_band_portfolio()` (reset to target only when a weight drifts beyond ±band, checked daily/monthly/quarterly/yearly; costs and turnover) and `apply_fee()` (annual fee accrued by calendar days).
  - S3: `engine/fx.py`: `hedged_levels()` (synthetic currency hedge = local NAV × cash_EUR / cash_USD, i.e. local return plus the short-rate carry) and `currency_views()` (local / unhedged / hedged, rebased to 100).
  - S3: `engine/robo.py`: the reference robo. `QUESTIONS` (7 questions in three groups: tolerance, capacity, need), `risk_profile()` (profile = round(min(tolerance, capacity)), need never raises it), `MODEL_PORTFOLIOS` (10 rows × IWDA.AS / IS3N.DE / EUNH.DE / XG7S.DE / XEON.DE; equity = 10% × profile, 95% at 10; cash sleeve in 1–2), `robo_backtest()` (calendar or band rebalancing + costs + fee layer), `goal_simulation()` (joint block bootstrap of monthly returns; profile-based pot pays goals when due vs one bucket per goal; reports P(met), median, 5th percentile, P(all met), equity at start), `aggregate_weights()`, `monthly_returns_from_prices()`.
  - `engine/factors.py`, `engine/stylized.py`, `engine/checks.py` as before. S3: `checks` adds an info note for provider-published robo series (asset class `Robo-advisor`).
- **Apps:**
  - `apps/performance_analyzer.py` (toolkit #1): scenarios per session (`?session=2`, `?session=3`). S3: Indexa's ten profiles appended as series `INDEXA-1..10` (asset class `Robo-advisor`, EUR, fund accounts 10–100k€, business days), four Session 3 scenarios (Indexa's profiles vs DIY 60/40; robo vs LifeStrategy vs DIY; world equity USD / EUR unhedged / EUR hedged with the real hedged ETF IWDE.L; rebalancing never/monthly/yearly).
  - S3: `apps/robo_simulator.py` (toolkit #3, "Robo-Advisor Lab"): tabs 1 profile (questionnaire, override), 2 portfolio (table, editable row), 3 backtest (fee layer, rebalancing incl. drift band, costs, window, benchmark = Indexa same profile / V60A.DE / IWDE.L, "the Excel way" toggle = daily free rebalancing, no fee), 4 goals (goal-based vs profile-based, euro ETFs 2009–26 or long US history 1987–2026, bucket 0 = cash), 5 what could be wrong (the seven traps + engine checks), 6 how it works + AI prompt. **Not yet deployed:** create a second Streamlit Cloud app from `main` with main file `apps/robo_simulator.py`; planned name `dislab-robo-lab` (the deck and docs print that URL; rebuild them if the name differs).
- **Data snapshot:** 73 tickers to 13 Oct 2026 (S3 added IWDE.L, EUNA.DE, XG7S.DE, XEON.DE, V60A.DE on 3 Nov 2026, cut to the same end date), FRED DTB3/DGS10/DEXUSEU/IR3TIB01EZM156N, Indexa profiles, Ken French factors to Aug 2026. S3: `data/kits/robo_data_s3.csv`, the student data file (`py scripts/build_kit_s3.py`).
- **Notebooks:** S3: `notebooks/s3_robo_lab.ipynb` (the optional code track: Excel way vs engine, FX views, goals).
- **Tests:** 150 passing (3 Nov 2026) (`py -m pytest`), including AppTest smoke tests of both apps (every scenario, currencies, frequencies, robo rebalance modes, histories).
- **Not built yet:** signal-based backtest engine (weights from signals, shifted one period), PSR/DSR, walk-forward.

## Decisions taken
- "Annualized return" in Sharpe, Sortino and Calmar is the **CAGR** (geometric, calendar days / 365.25), not the arithmetic mean × 252. Sortino's target is rf, over all periods.
- Information ratio = (CAGR − benchmark CAGR) / tracking error. Alpha = Jensen's, from arithmetic periodic means.
- % positive years / worst year use **full calendar years only**.
- Comparisons use **common dates** (inner join on dates, no filling).
- The snapshot is CSV per ticker (`date, close` adjusted, in listing currency) plus `manifest.csv`. Downloads drop today's possibly-intraday bar.
- FRED via the keyless `fredgraph.csv` endpoint (latest vintage).
- **Currency conversion uses FRED `DEXUSEU`, never Yahoo `EURUSD=X`.** Unhedged; dates matched exactly, no filling.
- **EUR risk-free rate:** euro-area 3m interbank (`IR3TIB01EZM156N`, OECD monthly, lags ~9 months). Apps default to it when the currency is EUR.
- **Portfolios:** rebalance on the last common trading day of each period, never on the sample's last day; cost = Σ|Δw| × bps on NAV; initial purchase free.
- **Frequency:** `to_frequency` keeps real last dates and the first observation, so CAGR is identical at every frequency.
- **Apps import the engine with a `sys.path` shim** at the top of each app.
- Stress episodes are S&P 500 closing peak → trough.
- **Yahoo's Xetra lines of the iShares/Xtrackers world factor ETFs have bad ticks**: the snapshot uses the London USD lines. Run a max-|daily return| scan on every new ticker before adding it (S3 tickers scanned: IWDE.L's 9.8% on 12 Mar 2020 is real).
- Asset classes `Smart beta`, `Factor equity`, `Multi-factor`, `Alternative`, `Thematic` drive the rule-based-product checks; `Robo-advisor` drives the provider-published note.
- **S3 (3 Nov 2026):** the model table's hedged-bond leg is XG7S.DE (from 2013) rather than EUNA.DE (from Nov 2017) so that the Indexa comparison starts in Dec 2015. The robo's fee layer defaults to 0.43% (Indexa's management 0.338% + custody 0.093%, Nov 2026); fund TERs are already inside Yahoo's adjusted prices. Indexa series are appended in the apps, not added to `universe.csv`, so `refresh_data.py` never tries to download them.
- **S3 goal simulation:** joint block bootstrap (12-month blocks) of historical monthly returns, no expected-return assumption; the euro menu (IWDA/EUNH/XEON since Oct 2009) is short and bullish, so the long US history (VFINX/VBMFX since 1987, cash at 0%) is offered as a robustness check. The profile-based pot pays each goal when due (an early goal can be met with money meant for a later one); that is the point, not a bug.

## Gotchas on the instructor's PC
- **Norton 360 intercepts HTTPS.** curl (inside yfinance) rejects its root certificate. `data._windows_ca_bundle()` merges certifi with the Windows root store. If yfinance says "possibly delisted; no price data found" for everything, delete `%LOCALAPPDATA%\py-yfinance` and retry.
- Python is `py` (Anaconda 3.12, pandas 2.2). Code must stay pandas-3-safe; pytest turns FutureWarnings into errors.
- The repo sits inside OneDrive. If OneDrive ever reports conflicts in `.git/`, pause OneDrive sync before git operations.
- Bash heredocs in the Claude Code harness break on non-ASCII characters (€, →, –): write patch scripts with the Write tool and run them with `py`.

## Next steps
0. **Scheduled data refresh (agreed with the instructor on 11 Nov 2026; to build on 12 Nov).** Goal: the apps stay online and current for this and future cohorts.
   - **Instructor first:** GitHub → Settings → Actions → General → Workflow permissions = "Read and write permissions". Confirm the frequency; monthly (first weekday) was recommended to limit repo growth.
   - **Build:**
     - A GitHub Actions workflow runs `scripts/refresh_data.py`, scans every series for bad ticks and runs the tests, and commits only if everything passes. It also has a manual "Run workflow" button.
     - Failed or closed tickers keep their old data and are flagged.
     - The class scenarios read a **frozen snapshot**: S1/S2 13 Oct 2026, S3 its own. That keeps the numbers identical to the slides, because Yahoo revises past adjusted prices.
     - Custom runs on the latest data. An on-screen label says which data the user is seeing.
     - Pin `requirements.txt` to the tested versions.
     - Apply the same to the Robo-Advisor Lab.
   - **After it ships:** every conversation must `git pull` before pushing.
   - **Why pin versions (11 Nov 2026):** Streamlit Cloud installed a newer Streamlit than the local 1.57. A radio whose labels changed with the return frequency then came back empty and crashed the Diversification tab. Fixed in b69a9b4 with fixed labels and a fallback; keep widget labels independent of other settings.
1. **Deploy the Robo-Advisor Lab** on Streamlit Community Cloud (instructor: share.streamlit.io → New app → repo `jose-suarezlledo/dis_lab`, branch `main`, main file `apps/robo_simulator.py`, app URL `dislab-robo-lab`). Then open both apps ~15 min before class (free apps sleep).
2. Backtest engine: signal → weights shifted one period → `portfolio`-style NAV with costs; bias checks for look-ahead and parameter snooping; PSR/DSR. Needed by S5 TAA and S7.
3. Extend the analyzer for S8 (PCA/contribution) rather than writing new apps.
