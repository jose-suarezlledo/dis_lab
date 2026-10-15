# Apps

One Streamlit app per session tool, all built on `engine/`. Run locally with `py -m streamlit run apps/<app>.py`.

| App | Toolkit item | Sessions | What it does |
|---|---|---|---|
| `performance_analyzer.py` | #1 Performance & risk analyzer | S1, S2, S3, S8 | Funds/ETFs and an optional fixed-weight portfolio vs a benchmark: growth, drawdowns, all metrics, calendar years, stress episodes, rolling view, stylized facts of returns, automatic bias checks |

Conventions:
- Every number comes from `engine/`. Apps only choose what to show and how to explain it.
- `dis_ui.py` holds what apps share: palette, number formats, metric glossary, disclaimer.
- Each app puts its own folder and the repo root on `sys.path`, so it runs the same way locally, on Streamlit Community Cloud and in `tests/test_apps.py`.
- Every app must pass a smoke test in `tests/test_apps.py`.

Deployed on Streamlit Community Cloud from `main`. Main file path: `apps/<app>.py`. Python 3.12.
