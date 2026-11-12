# Apps

One Streamlit app per session tool, all built on `engine/`. Run locally with `py -m streamlit run apps/<app>.py`.

| App | Toolkit item | Sessions | What it does |
|---|---|---|---|
| `performance_analyzer.py` ([live](https://dislab-performance-analyzer-s1.streamlit.app/)) | #1 Performance & risk analyzer | S1, S2, S3, S8 | Funds/ETFs and an optional fixed-weight portfolio vs a benchmark: growth, drawdowns, all metrics, calendar years, stress episodes, rolling view, diversification, stylized facts of returns, automatic bias checks. Class scenarios per session (`?session=2`, `?session=3`). Session 3 adds Indexa Capital's published profiles as series (`INDEXA-1` … `INDEXA-10`). |
| `robo_simulator.py` (deploy as `dislab-robo-lab`) | #3 Robo / allocation simulator | S3 | The reference robo-advisor: questionnaire (tolerance, capacity, need) → profile → model portfolio of UCITS ETFs → backtest net of trading costs and the robo's fee layer, by calendar or drift-band rebalancing, vs Indexa's published profile → goal-based vs profile-based simulation on bootstrapped history → the seven traps of an AI-built robo. |

Conventions:
- Every number comes from `engine/`. Apps only choose what to show and how to explain it.
- Both apps offer a **Data** switch: the frozen class snapshot (`engine.data.snapshot_dir()`, what the slides were built on; class scenarios select it) or the latest monthly refresh in `data/` (custom runs select it). The common-window line and the "How this works" tab say which one is on screen. Pass the chosen folder as `data_dir` to every loader (`risk_free_rate`, `convert_currency`…), never mix the two.
- Widget labels must not depend on other settings (a label that changes between reruns resets the widget; this crashed a tab on 11 Nov 2026). Use stable option keys with `format_func`.
- `dis_ui.py` holds what apps share: palette, number formats, metric glossary, disclaimer.
- Each app puts its own folder and the repo root on `sys.path`, so it runs the same way locally, on Streamlit Community Cloud and in `tests/test_apps.py`.
- Every app must pass a smoke test in `tests/test_apps.py`.

Deployed on Streamlit Community Cloud from `main`. Main file path: `apps/<app>.py`. Python 3.12. Each app is its own Streamlit Cloud app (same repo, different main file). The Cloud installs exactly the versions pinned in `requirements.txt` (change a pin, run the tests, push). Every push to `main`, including the monthly data-refresh commit, redeploys the apps.
