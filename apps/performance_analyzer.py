"""Performance & Risk Analyzer: DIS toolkit #1 (Sessions 1, 2, 3 and 8).

Pick funds/ETFs (and optionally a fixed-weight portfolio of them), a benchmark and
a window; get every section 7 metric, the charts behind them, the stylized facts
of their returns, and the list of things that could make the numbers misleading.

    py -m streamlit run apps/performance_analyzer.py
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import dis_ui as ui  # noqa: E402
from engine import __version__ as ENGINE_VERSION  # noqa: E402
from engine import checks as chk  # noqa: E402
from engine import data  # noqa: E402
from engine import metrics as m  # noqa: E402
from engine import stylized as sty  # noqa: E402
from engine.portfolio import REBALANCE_FREQUENCIES, fixed_weight_portfolio  # noqa: E402

st.set_page_config(page_title="Performance & Risk Analyzer · DIS", page_icon="📈", layout="wide")

PORTFOLIO = "My portfolio"
CURRENCY_OPTIONS = ["EUR", "USD", "As listed"]
RF_OPTIONS = ["Auto", "US T-bill", "Euro 3m", "Custom"]
MAX_ASSETS = 6


@st.cache_data
def load_all():
    uni = data.load_universe()
    uni = uni[uni["asset_class"] != "FX"]  # exchange rates are not investments here
    prices = data.load_prices(list(uni.index))
    man = data.load_manifest()
    return uni, prices, man


UNI, PRICES, MANIFEST = load_all()
FIRST_DAY = PRICES.index[0].date()
LAST_DAY = PRICES.index[-1].date()

# --------------------------------------------------------------------------- scenarios

SCENARIOS = {
    "Stocks, bonds and a 60/40 mix (since 1987)": dict(
        assets=["VFINX", "VBMFX"], benchmark="VFINX", portfolio=[("VFINX", 60), ("VBMFX", 40)],
        rebalance="monthly", cost=5, start=dt.date(1987, 1, 1), currency="As listed", freq="daily",
        question="Diversification: does a 60/40 mix give up much return for the risk it removes? "
                 "Look at CAGR, max drawdown and time under water side by side.",
    ),
    "Myth: the highest return is the best investment": dict(
        assets=["QQQ", "SPY", "VBINX"], benchmark="SPY", portfolio=None,
        start=dt.date(1999, 3, 10), currency="As listed", freq="daily",
        question="QQQ has the best CAGR since 1999. Would you have held it from 2000 to 2015? "
                 "Check its max drawdown and time under water, then try starting the window in 2009.",
    ),
    "Myth: bonds are safe": dict(
        assets=["TLT", "IEF", "AGG", "SHY", "BIL"], benchmark="AGG", portfolio=None,
        start=dt.date(2007, 6, 1), currency="As listed", freq="daily",
        question="Which of these lost the most, and when? Compare 2008 with 2022 in the stress-episode "
                 "table. What does 'safe' mean for a bond?",
    ),
    "Myth: gold and bitcoin protect you in a crash": dict(
        assets=["GLD", "BTC-USD", "SPY"], benchmark="SPY", portfolio=None,
        start=dt.date(2014, 9, 17), currency="As listed", freq="weekly",
        question="Do gold and bitcoin go up when stocks fall? Check the stress episodes and the rolling "
                 "correlation, not the full-period CAGR.",
    ),
    "Euro investor: one world index, three wrappers": dict(
        assets=["IWDA.AS", "VWCE.DE", "URTH"], benchmark="IWDA.AS", portfolio=None,
        start=dt.date(2019, 7, 23), currency="EUR", freq="weekly",
        question="Same index exposure, different products and currencies. Switch the currency to "
                 "'As listed': what changes, and why?",
    ),
    "Custom (keep my settings)": None,
}
DEFAULT_SCENARIO = next(iter(SCENARIOS))


def apply_scenario() -> None:
    sc = SCENARIOS.get(st.session_state.scenario)
    if sc is None:
        return
    ss = st.session_state
    ss.assets = sc["assets"]
    ss.benchmark = sc["benchmark"]
    ss.pf_on = sc["portfolio"] is not None
    rows = sc["portfolio"] or [("SPY", 60), ("AGG", 40)]
    ss.pf_rows = pd.DataFrame(rows, columns=["Ticker", "Weight %"])
    ss.pop("pf_editor", None)  # reset the editor to the new rows
    ss.rebalance = sc.get("rebalance", "monthly")
    ss.cost_bps = float(sc.get("cost", 5))
    ss.start = sc["start"]
    ss.end = LAST_DAY
    ss.currency = sc["currency"]
    ss.freq = sc["freq"]


if "scenario" not in st.session_state:
    st.session_state.scenario = DEFAULT_SCENARIO
    st.session_state.rf_mode = "Auto"
    st.session_state.rf_custom = 2.0
    st.session_state.colors = {}
    apply_scenario()


def label(t: str) -> str:
    return f"{t} · {UNI.loc[t, 'name']}" if t in UNI.index else t


# --------------------------------------------------------------------------- sidebar

with st.sidebar:
    st.header("Set up the comparison")
    st.selectbox("Scenario", list(SCENARIOS), key="scenario", on_change=apply_scenario,
                 help="Loads a ready-made comparison used in class. Then change anything you like.")
    st.multiselect("Funds / ETFs to compare", list(UNI.index), key="assets", format_func=label,
                   max_selections=MAX_ASSETS,
                   help="Real products, adjusted for dividends (total return). Up to 6.")
    st.selectbox("Benchmark", list(UNI.index), key="benchmark", format_func=label,
                 help="What you compare against: ideally what you would otherwise hold. "
                      "Shown as a grey dashed line.")

    with st.expander("Build a portfolio (optional)", expanded=st.session_state.pf_on):
        st.toggle("Add a fixed-weight portfolio", key="pf_on",
                  help="Hold the weights below, rebalancing back to them periodically.")
        pf_table = st.data_editor(
            st.session_state.pf_rows, key="pf_editor", num_rows="dynamic", hide_index=True,
            width="stretch",
            column_config={
                "Ticker": st.column_config.SelectboxColumn(options=list(UNI.index), required=True),
                "Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=5.0,
                                                          format="%.0f"),
            },
        )
        st.radio("Rebalance", list(REBALANCE_FREQUENCIES), key="rebalance", horizontal=True,
                 help="How often the weights are reset to target (on the last trading day of the "
                      "period). 'none' = buy and hold: the weights drift.")
        st.number_input("Trading cost (bps per trade)", min_value=0.0, max_value=200.0, step=5.0,
                        key="cost_bps",
                        help="Spread + commission + slippage, in basis points of the amount traded "
                             "(10 bps = 0.10%). Charged at every rebalance.")

    st.subheader("Window and units")
    c1, c2 = st.columns(2)
    c1.date_input("From", key="start", min_value=FIRST_DAY, max_value=LAST_DAY,
                  help="The comparison starts when every series has data, so it may start later.")
    c2.date_input("To", key="end", min_value=FIRST_DAY, max_value=LAST_DAY)
    st.radio("Currency", CURRENCY_OPTIONS, key="currency", horizontal=True,
             help="EUR: what a euro investor without currency hedging actually earned. "
                  "'As listed' keeps each product in its trading currency (mixing currencies).")
    st.radio("Return frequency", list(m.FREQUENCIES), key="freq", horizontal=True,
             help="Daily, weekly (Friday) or month-end closes. CAGR is the same; volatility, "
                  "drawdowns and correlations change.")
    st.radio("Risk-free rate", RF_OPTIONS, key="rf_mode", horizontal=True,
             help="Used in Sharpe, Sortino and alpha. Auto = euro 3-month rate for EUR, "
                  "US 3-month T-bill otherwise; averaged over the window.")
    if st.session_state.rf_mode == "Custom":
        st.number_input("Risk-free rate (% per year)", step=0.25, key="rf_custom")

    st.divider()
    st.caption(ui.DISCLAIMER)

# --------------------------------------------------------------------------- build the series

st.title("Performance & Risk Analyzer")
st.caption("DIS toolkit #1 · Arc stages: **Data → Backtest → Robustness** · "
           "every number comes from the shared `dis_lab` engine")
st.markdown("> **The decision:** would you put €100k of your own money in this? "
            "Which number on this page would make you say no?")

sc = SCENARIOS.get(st.session_state.scenario)
if sc is not None:
    st.info(f"**{st.session_state.scenario}.** {sc['question']}")

assets: list[str] = list(st.session_state.assets)
bench: str = st.session_state.benchmark
start, end = pd.Timestamp(st.session_state.start), pd.Timestamp(st.session_state.end)
currency = st.session_state.currency
freq = st.session_state.freq

weights: dict[str, float] = {}
if st.session_state.pf_on:
    tbl = pf_table.dropna()
    tbl = tbl[tbl["Ticker"].isin(UNI.index) & (tbl["Weight %"] > 0)]
    w = tbl.groupby("Ticker")["Weight %"].sum()
    if w.sum() <= 0:
        st.warning("The portfolio has no weights yet: add tickers and weights in the sidebar.")
    else:
        if abs(w.sum() - 100) > 0.01:
            st.warning(f"Portfolio weights add up to {w.sum():.0f}%, so they were scaled to 100%.")
        weights = (w / w.sum()).to_dict()

if not assets and not weights:
    st.info("Pick at least one fund or ETF in the sidebar.")
    st.stop()
if start >= end:
    st.error("'From' must be before 'To'.")
    st.stop()

tickers = list(dict.fromkeys(assets + list(weights) + [bench]))
px = PRICES[tickers].loc[start:end]
if currency != "As listed":
    px = data.convert_currency(px, currency, UNI["currency"].to_dict())
inception = {t: px[t].first_valid_index() for t in tickers}
if any(d is None for d in inception.values()):
    st.error("No data in this window for: " + ", ".join(t for t, d in inception.items() if d is None))
    st.stop()
common_start = max(inception.values())
px = px.loc[common_start:]

series: dict[str, pd.Series] = {t: px[t] for t in assets}
pf = None
if weights:
    pf = fixed_weight_portfolio(px, weights, st.session_state.rebalance, st.session_state.cost_bps,
                                name=PORTFOLIO)
    series[PORTFOLIO] = pf.nav
if bench not in series:
    series[bench] = px[bench]
daily = m.align(*[s.rename(k) for k, s in series.items()])
if len(daily) < 60:
    st.error("Fewer than 60 common trading days in this window: widen it or drop the youngest series.")
    st.stop()
levels = m.to_frequency(daily, freq)
if len(levels) < 24:
    st.error(f"Only {len(levels)} {freq} observations: widen the window or use a higher frequency.")
    st.stop()

w_start, w_end = levels.index[0], levels.index[-1]
years = m.years_between(w_start, w_end)
colors = ui.assign_colors([c for c in levels.columns if c != bench], st.session_state.colors)


def risk_free() -> tuple[float, str]:
    mode = st.session_state.rf_mode
    if mode == "Custom":
        return st.session_state.rf_custom / 100, "custom rate"
    if mode == "Auto":
        mode = "Euro 3m" if currency == "EUR" else "US T-bill"
    if mode == "Euro 3m":
        try:
            return (data.risk_free_rate(w_start, w_end, data.EURO_RF_SERIES),
                    "euro-area 3-month interbank rate (OECD via FRED, monthly), window average")
        except ValueError:
            pass  # window before 1994: fall back to the T-bill
    return data.risk_free_rate(w_start, w_end, "DTB3"), "3-month US T-bill (FRED DTB3), window average"


rf, rf_source = risk_free()
listed = sorted({UNI.loc[t, "currency"] for t in tickers})
shown_ccy = currency if currency != "As listed" else (
    f"as listed ({listed[0]})" if len(listed) == 1 else "as listed (mixed: " + ", ".join(listed) + ")")

st.markdown(
    f"**Common window:** {w_start:%d %b %Y} → {w_end:%d %b %Y} ({years:.1f} years) · "
    f"**{freq.capitalize()}** returns · **Currency:** {shown_ccy} · "
    f"**Risk-free rate:** {rf:.2%} ({rf_source})"
)

checks = chk.performance_checks(
    levels, benchmark=bench, inception={t: inception[t] for t in tickers},
    listing_currency={c: UNI.loc[c, "currency"] for c in levels.columns if c in UNI.index},
    display_currency=None if currency == "As listed" else currency, freq=freq, rf=rf,
    portfolio=None if pf is None else {"name": PORTFOLIO, "rebalance": st.session_state.rebalance,
                                       "cost_bps": st.session_state.cost_bps},
    asset_class={c: UNI.loc[c, "asset_class"] for c in levels.columns if c in UNI.index},
)
warns = [c for c in checks if c.severity == "warn"]
if warns:
    st.warning(f"**{len(warns)} thing(s) to check before you believe these numbers:** "
               + " · ".join(c.title for c in warns) + ". Details in *What could be wrong here*.")

# --------------------------------------------------------------------------- metrics (engine)

table = m.summary_table(levels, benchmark=bench, rf=rf)
table.loc["sharpe_se"] = [m.sharpe_standard_error(table.loc["sharpe", c], years) for c in table.columns]
if pf is not None:
    table.loc["annual_turnover"] = [pf.annual_turnover if c == PORTFOLIO else np.nan for c in table.columns]
    table.loc["annual_cost"] = [pf.annual_cost if c == PORTFOLIO else np.nan for c in table.columns]
SHOWN = ["cagr", "vol", "sharpe", "sharpe_se", "sortino", "max_dd", "tuw_days", "calmar", "pct_pos_years",
         "worst_year", "beta", "alpha", "correlation", "tracking_error", "info_ratio", "up_capture",
         "down_capture", "annual_turnover", "annual_cost"]
SHOWN = [k for k in SHOWN if k in table.index]


def display_table(keys: list[str]) -> pd.DataFrame:
    out = pd.DataFrame({c: [ui.format_metric(k, table.loc[k, c]) for k in keys] for c in table.columns},
                       index=[ui.GLOSSARY[k][0] if k in ui.GLOSSARY else m.METRIC_LABELS[k] for k in keys])
    out.columns = [f"{c} (benchmark)" if c == bench else c for c in out.columns]
    return out


# --------------------------------------------------------------------------- chart helpers


def style(fig: go.Figure, yfmt: str, height: int = 380, log: bool = False, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50 if title else 30, b=10),
        title=dict(text=title, x=0, font=dict(size=15)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif"),
    )
    fig.update_yaxes(tickformat=yfmt, type="log" if log else "linear", gridcolor=ui.GRID_COLOR, zeroline=False)
    fig.update_xaxes(showgrid=False)
    return fig


def lines(df: pd.DataFrame, yfmt: str, hover: str, **kw) -> go.Figure:
    fig = go.Figure()
    for c in df.columns:
        is_b = c == bench
        fig.add_trace(go.Scatter(
            x=df.index, y=df[c], mode="lines", name=f"{c} (benchmark)" if is_b else c,
            line=dict(width=2, color=ui.BENCHMARK_COLOR if is_b else colors[c], dash="dash" if is_b else "solid"),
            hovertemplate=f"%{{y:{hover}}}",
        ))
    return style(fig, yfmt, **kw)


def color_of(c: str) -> str:
    return ui.BENCHMARK_COLOR if c == bench else colors[c]


# --------------------------------------------------------------------------- tabs

tabs = st.tabs(["Growth & drawdowns", "All metrics", "Calendar years & crises", "Rolling view",
                "How returns behave", f"What could be wrong here ({len(warns)})", "How this works"])

with tabs[0]:
    st.dataframe(display_table(["cagr", "vol", "sharpe", "max_dd", "tuw_days"]), width="stretch")
    log = st.toggle("Log scale", value=years > 10,
                    help="On a log scale equal percentage moves look equally tall, so a fall from 200 "
                         "to 100 and from 20 to 10 look alike. Use it for long windows.")
    growth = levels / levels.iloc[0] * 100
    fig = lines(growth, ",.0f", ",.1f", log=log, height=420, title="Growth of 100 invested at the start")
    if log:
        lo, hi = growth.min().min(), growth.max().max()
        ticks = [v * 10**e for e in range(-2, 7) for v in (1, 2, 5) if lo / 1.5 <= v * 10**e <= hi * 1.5]
        fig.update_yaxes(tickvals=ticks)
    st.plotly_chart(fig, width="stretch")
    dd = pd.DataFrame({c: m.drawdown_series(levels[c]) for c in levels.columns})
    st.plotly_chart(lines(dd, ".0%", ".1%", height=300, title="Drawdown: % below the previous high"),
                    width="stretch")
    st.caption("Read both charts together: the top one is what brochures show, the bottom one is what "
               "investors live through.")

with tabs[1]:
    st.dataframe(display_table(SHOWN), width="stretch", height=(len(SHOWN) + 1) * 35 + 3)
    st.caption(f"Risk-free rate {rf:.2%} ({rf_source}). Benchmark-relative rows are blank for the "
               "benchmark itself. % positive years and worst year use full calendar years only.")
    raw = table.copy()
    raw.index = [m.METRIC_LABELS.get(k, ui.GLOSSARY.get(k, (k,))[0]) for k in raw.index]
    d1, d2 = st.columns(2)
    d1.download_button("Download metrics (CSV)", raw.to_csv().encode("utf-8"), "metrics.csv", "text/csv")
    d2.download_button("Download the series (CSV)", levels.to_csv().encode("utf-8"), "series.csv", "text/csv")
    with st.expander("What each metric means, and its trap", expanded=False):
        st.dataframe(ui.glossary_table(SHOWN), hide_index=True, width="stretch")

    if len(levels.columns) > 2:
        names = list(levels.columns)
        corr = pd.DataFrame([[m.correlation(levels[a], levels[b]) if a != b else 1.0 for b in names] for a in names],
                            index=names, columns=names)
        fig = go.Figure(go.Heatmap(z=corr.to_numpy(), x=names, y=names, zmin=-1, zmax=1,
                                   colorscale=ui.DIVERGING_SCALE, text=corr.map(lambda v: f"{v:.2f}").to_numpy(),
                                   texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
                                   xgap=2, ygap=2))
        fig.update_layout(height=120 + 45 * len(names), margin=dict(l=10, r=10, t=50, b=10),
                          title=dict(text=f"Correlation of {freq} returns", x=0, font=dict(size=15)),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue = move together, red = move in opposite directions, grey = unrelated. "
                   "Diversification comes from the grey and red cells.")

with tabs[2]:
    yr = pd.DataFrame({c: m.calendar_year_returns(levels[c])["return"] for c in levels.columns})
    partial = m.calendar_year_returns(levels.iloc[:, 0])["partial"]
    xlab = [f"{y}*" if partial[y] else str(y) for y in yr.index]
    fig = go.Figure()
    for c in yr.columns:
        fig.add_trace(go.Bar(x=xlab, y=yr[c], name=f"{c} (benchmark)" if c == bench else c,
                             marker_color=color_of(c), hovertemplate="%{y:.1%}"))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig = style(fig, ".0%", height=400, title="Calendar-year returns")
    fig.update_xaxes(type="category")  # keeps "2026*" (partial year) as its own bar
    st.plotly_chart(fig, width="stretch")
    st.caption(r"\* partial year (the window starts or ends during it).")
    yt = yr.map(ui.pct)
    yt.index = xlab
    with st.expander("Calendar-year table"):
        st.dataframe(yt, width="stretch")

    st.subheader("Stress episodes")
    ep = pd.DataFrame({c: m.episode_returns(daily[c]) for c in daily.columns})
    ep.index = [f"{k} ({pd.Timestamp(a):%b %Y} → {pd.Timestamp(b):%b %Y})" for k, (a, b) in m.STRESS_EPISODES.items()]
    ep = ep.dropna(how="all")
    if ep.empty:
        st.info("No major sell-off falls inside this window, so this table is empty. That is itself a warning.")
    else:
        st.dataframe(ep.map(ui.pct), width="stretch")
    st.caption("Total return from the S&P 500 closing high to its closing low in each sell-off, on daily data. "
               "An episode is blank when the window does not cover it entirely.")

with tabs[3]:
    win = st.radio("Window", [1, 3, 5], index=1, horizontal=True, format_func=lambda y: f"{y} years",
                   help="Each point summarises the trailing window ending on that date.")
    if years < win + 0.5:
        st.info(f"The common window is too short for {win}-year rolling statistics.")
    else:
        rc = pd.DataFrame({c: m.rolling_cagr(levels[c], win) for c in levels.columns})
        st.plotly_chart(lines(rc, ".0%", ".1%", height=330, title=f"Rolling {win}-year CAGR"), width="stretch")
        st.caption("The return you would have earned if you had invested exactly "
                   f"{win} years before each date. Look at the spread between best and worst starting points.")
        rv = pd.DataFrame({c: m.rolling_vol(levels[c], win) for c in levels.columns})
        st.plotly_chart(lines(rv, ".0%", ".1%", height=300, title=f"Rolling {win}-year volatility"), width="stretch")
        others = [c for c in levels.columns if c != bench]
        if others:
            rcor = pd.DataFrame({c: m.rolling_correlation(levels[c], levels[bench], win) for c in others})
            st.plotly_chart(lines(rcor, ".1f", ".2f", height=300,
                                  title=f"Rolling {win}-year correlation with {bench}"), width="stretch")
            st.caption("Correlation is not a constant. The stock-bond correlation, for example, turned "
                       "positive in 2022, when both fell together.")

with tabs[4]:
    st.markdown("Real returns are not the bell curve of textbooks. Pick one series (daily returns).")
    pick = st.selectbox("Series", list(daily.columns), index=list(daily.columns).index(bench),
                        format_func=lambda c: f"{c} (benchmark)" if c == bench else c)
    s = daily[pick]
    rs = sty.return_stats(s)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(f"Worst day ({rs['worst_date']:%d %b %Y})", ui.pct(rs["worst"]),
              help="The single worst daily return in the window.")
    k2.metric("…in standard deviations", f"{rs['worst_in_sigmas']:.1f} σ",
              help="How many daily standard deviations below the average that day was. A normal "
                   "distribution makes anything beyond 5 σ practically impossible.")
    k3.metric("Excess kurtosis", f"{rs['excess_kurtosis']:.1f}",
              help="0 for a normal distribution. Higher = fatter tails: more extreme days than the bell curve predicts.")
    k4.metric("Average vs CAGR", f"{rs['arith_mean_ann']:.1%} vs {rs['cagr']:.1%}",
              help="Average daily return × 252 versus what you actually earned. The gap is volatility drag: "
                   "+50% then −50% averages 0% but loses 25%.")

    r = m.to_returns(s)
    mu, sd = r.mean(), r.std()
    logy = st.toggle("Log scale for counts (shows the tails)", value=True)
    bins = 120
    width = (r.max() - r.min()) / bins
    xs = np.linspace(r.min(), r.max(), 400)
    normal = len(r) * width * np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=r, xbins=dict(start=r.min(), end=r.max(), size=width), name="Actual days",
                               marker_color=ui.SERIES_COLORS[0], hovertemplate="%{x:.1%}: %{y} days<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=normal, mode="lines", name="Normal distribution, same mean and volatility",
                             line=dict(color=ui.SERIES_COLORS[1], width=2), hoverinfo="skip"))
    fig = style(fig, "", height=360, title=f"Daily returns of {pick}: actual vs bell curve")
    fig.update_xaxes(tickformat=".0%")
    if logy:
        counts, _ = np.histogram(r, bins=bins)
        fig.update_yaxes(type="log", range=[np.log10(0.5), np.log10(counts.max() * 2)])
        normal = np.where(normal < 0.3, np.nan, normal)  # below a fraction of a day the curve is off the chart
        fig.data[1].y = normal
    fig.update_layout(hovermode="closest", bargap=0.05)
    st.plotly_chart(fig, width="stretch")

    tails = sty.tail_counts(s)
    tt = pd.DataFrame({
        "Daily loss worse than": [f"{k:g} σ ({ui.pct(tails.loc[k, 'threshold'])})" for k in tails.index],
        "Actually happened": tails["actual"].astype(int).to_numpy(),
        "Bell curve predicts": [f"{v:.1f}" if v >= 0.1 else f"{v:.3f}" for v in tails["normal_expected"]],
        "Bell curve says: once every": [f"{v:,.0f} years" if v >= 1 else f"{v * 12:.0f} months"
                                        for v in tails["normal_once_every_years"]],
    })
    st.dataframe(tt, hide_index=True, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        ac = pd.DataFrame({"Returns": sty.autocorrelation(s, 20), "Size of returns (|r|)": sty.autocorrelation(s, 20, True)})
        fig = go.Figure()
        for i, c in enumerate(ac.columns):
            fig.add_trace(go.Bar(x=ac.index, y=ac[c], name=c, marker_color=ui.SERIES_COLORS[i],
                                 hovertemplate="lag %{x}: %{y:.2f}"))
        band = 2 / np.sqrt(len(r))
        for y in (band, -band):
            fig.add_hline(y=y, line=dict(color=ui.BENCHMARK_COLOR, width=1, dash="dot"))
        fig.update_layout(barmode="group")
        fig = style(fig, ".2f", height=320, title="Does today predict tomorrow? (autocorrelation)")
        fig.update_xaxes(title="lag (days)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Returns themselves barely predict the next day (bars inside the dotted band = noise). "
                   "Their size does: calm follows calm, storms follow storms. That is volatility clustering.")
    with c2:
        vol1m = m.rolling_vol(s, 21 / 252)
        fig = go.Figure(go.Scatter(x=vol1m.index, y=vol1m, mode="lines", line=dict(color=color_of(pick), width=2),
                                   name="1-month volatility", hovertemplate="%{y:.0%}"))
        st.plotly_chart(style(fig, ".0%", height=320, title="Volatility comes in waves (1-month, annualized)"),
                        width="stretch")

    st.markdown("**\"Don't miss the 10 best days\"**: the full story")
    n_days = st.slider("Number of days", 5, 30, 10, step=5)
    md = sty.missing_days_effect(s, n_days)
    st.dataframe(pd.DataFrame({
        "Scenario": ["Invested every day", f"Missed the {n_days} best days", f"Missed the {n_days} worst days",
                     "Missed both"],
        "CAGR": [ui.pct(md[k], 2) for k in ["all_days", "missed_best", "missed_worst", "missed_both"]],
    }), hide_index=True, width="stretch")
    st.caption("Missing a day = sitting in cash at 0% that day. The best and worst days cluster in the same "
               "turbulent weeks, so nobody misses only the best ones. The real lesson is about volatility "
               "regimes, not about never selling.")

with tabs[5]:
    st.markdown("Automatic checks against the course's **bias checklist**. A warning does not mean the "
                "numbers are wrong; it means you should know why they might be before you act on them.")
    for c in checks:
        box = st.warning if c.severity == "warn" else st.info
        box(f"**{c.title}**  \n{c.detail}  \n*Checklist {c.item}: {chk.CHECKLIST[c.item]}*",
            icon="⚠️" if c.severity == "warn" else "ℹ️")
    with st.expander("The full bias checklist"):
        st.markdown("""
1. **Look-ahead:** did the rule use information it could not have had at the time?
2. **Survivorship / selection:** was the universe or the asset chosen with hindsight?
3. **Overfitting / data snooping:** how many variants were tried before this one?
4. **Costs and implementation:** commissions, spreads, taxes, FX, minimum sizes, can an EU investor buy it?
5. **Benchmark:** compared with buy-and-hold *and* a simple sensible alternative (60/40, equal weight)?
6. **Sanity:** Sharpe above 1.5 or returns far above the asset class? Assume a bug until proven otherwise.
7. **Regime dependence:** does it hold in 2000–02, 2008, 2020 and 2022, or only on average?
""")

with tabs[6]:
    st.markdown(f"""
**What this tool does, in plain English**
1. Takes the daily closing prices of the products you picked, adjusted for dividends and splits
   (so returns are *total* returns, net of the fund's own fees).
2. Keeps only the dates on which **every** series has a price (no gaps filled, no invented 0% days), and
   cuts everything to the dates they share. A young product shortens the window for everyone.
3. Optionally converts every price to one currency with the Federal Reserve's daily EUR/USD rate
   (unhedged: currency moves are part of the return).
4. Builds your portfolio, if you defined one: it holds the target weights, lets them drift with
   prices, resets them on the last trading day of each rebalancing period, and pays the trading cost on
   what it trades.
5. Computes every metric with the course's shared engine (`dis_lab/engine`, version {ENGINE_VERSION}),
   with these definitions: CAGR uses calendar days / 365.25; volatility = standard deviation of returns
   × √(periods per year); Sharpe = (CAGR − risk-free) / volatility; drawdowns are measured on the value path.

**Data:** Yahoo Finance adjusted closes and FRED (US T-bill, euro 3-month rate, EUR/USD), saved as a
snapshot so the tool never depends on a live download. Snapshot taken on
{MANIFEST['downloaded'].max()}; prices up to {LAST_DAY:%d %b %Y}.

**Ask your AI analyst** (any assistant works). Download the metrics CSV, attach it, and paste:
""")
    st.code(
        "You are the sceptical member of my investment committee. Attached are performance metrics for "
        "several funds over the same window. 1) In plain English, what do they say about return, risk and "
        "the worst experience an investor would have had? 2) Using this checklist (look-ahead, "
        "survivorship/selection, overfitting, costs, benchmark, sanity, regime dependence), give me the "
        "three strongest reasons NOT to trust a conclusion drawn from this table. 3) What one extra test "
        "would change your mind? Do not recommend a product.",
        language=None, wrap_lines=True,
    )
    st.caption(ui.DISCLAIMER)
