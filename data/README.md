# Data snapshots

The apps read only these files, so class never depends on Yahoo Finance or FRED being up. Download dates are in `manifest.csv`.

| Path | Content | Source |
|---|---|---|
| `universe.csv` | Tickers offered in the apps, with name, asset class, currency, UCITS flag | Hand-curated |
| `prices/<TICKER>.csv` | `date, close`: adjusted (total-return) daily close in the listing currency | Yahoo Finance (`yfinance`, `auto_adjust=True`) |
| `macro/<SERIES>.csv` | `date, value`, latest vintage (not real-time data): `DTB3` (US 3m T-bill, %), `DGS10`, `DEXUSEU` (USD per EUR, Fed noon rate, used for currency conversion), `IR3TIB01EZM156N` (euro-area 3m interbank rate, %, monthly, OECD; lags by several months) | FRED public CSV endpoint |
| `indexa/indexa_stats_returns_mutual.csv` | Indexa Capital model-portfolio returns: profiles 1–10 × 3 account-size tiers, base 100, EUR, net of fees. `;` separator, decimal comma, one row per **calendar** day (weekends repeated) | Indexa Capital published statistics |

Caveats to teach, not hide:
- **Yahoo adjusted closes are revised** whenever dividends are re-adjusted. Two downloads on different dates won't match to the last decimal.
- **Listing currency matters.** `IWDA.AS` is in EUR and `URTH` in USD. Comparing them mixes asset returns with EUR/USD moves. Convert with `engine.data.convert_currency()`, which uses the Fed's `DEXUSEU`.
- **Yahoo's `EURUSD=X` has bad ticks** (e.g. 17% fake daily moves in Dec 2008, up to 15% off the Fed rate). It stays in the snapshot as a find-the-flaw example, but never use it for conversion.
- **Mutual funds** (`VFINX`, `VBINX`…) carry long histories and are useful for 2000–02 and 2008, but they price once a day at NAV.
- **Bitcoin trades 7 days a week.** Aligning it with ETFs drops weekends, which turns them into multi-day Monday returns.
- **Indexa:** use the month-end series (`engine.data.load_indexa()` default). The repeated weekend values would bias daily volatility down.
