"""DIS Strategy Lab: toolkit #5 (Session 4, tactical asset allocation; later Sessions 5 and 7).

Build a rules-based strategy (dual momentum, Faber's moving average, GEM), see every
intermediate step, test its robustness, and audit it with the seven traps. The same page
is the reference against which students check the lab their AI assistant built.

    py -m streamlit run apps/strategy_lab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1])]

import dis_ui as ui  # noqa: E402
from engine import __version__ as ENGINE_VERSION  # noqa: E402
from engine import data  # noqa: E402
from engine import metrics as m  # noqa: E402
from engine import robustness as rb  # noqa: E402
from engine import signals as sg  # noqa: E402
from engine.backtest import BacktestResult, cash_returns_from_rate, signal_backtest, trace  # noqa: E402

st.set_page_config(page_title="Strategy Lab · DIS", page_icon="🧪", layout="wide")

REPO = "https://github.com/jose-suarezlledo/dis_lab"
KIT_URL = "https://raw.githubusercontent.com/jose-suarezlledo/dis_lab/main/data/kits/taa_data_s4.csv"
ASSETS = {  # the Session 4 universe (same as the student kit)
    "SPY": "US large caps (S&P 500)", "QQQ": "US tech / growth (Nasdaq-100)", "IWM": "US small caps (Russell 2000)",
    "EFA": "Developed markets ex-US", "EEM": "Emerging markets", "AGG": "US bonds (aggregate)",
    "IEF": "US Treasuries 7-10y", "TLT": "US Treasuries 20y+", "GLD": "Gold", "DBC": "Commodities",
    "VNQ": "US listed real estate", "VFINX": "S&P 500 index fund (since 1976)",
    "VBMFX": "US bond index fund (since 1986)", "VUSTX": "Long Treasury fund (since 1986)",
}
BONDS = ("AGG", "IEF", "TLT", "VBMFX", "VUSTX")
DEFAULT_UNIVERSE = ["SPY", "EFA", "EEM", "AGG", "TLT", "GLD", "DBC", "VNQ"]
LOOKBACKS = [1, 3, 6, 9, 12, "13612W"]
RULES = {"dual": "Dual momentum (Top-N)", "sma": "Moving average (Faber)", "gem": "GEM (Antonacci)", "ew": "Equal weight 1/N"}
ABSOLUTE = {"cash": "Pick must beat T-bills over the lookback", "zero": "Pick must have a positive return", "none": "No filter (relative momentum only)"}
LAGS = {1: "Honest: signal at month-end t, invested during t+1", 0: "Same month: the signal earns its own month (look-ahead)"}
MISSING = {"exclude": "Exclude an asset until it has enough prices", "zero": "Count a missing asset as 0% return (the trap)"}
PRESETS = {
    "custom": "Custom (keep my settings)",
    "gem": "GEM: Antonacci's Global Equities Momentum (2013)",
    "faber": "Faber: 10-month moving average on the S&P 500 (2007)",
    "top3": "Top-3 of 8 by 13612W, vetoed into bonds (Keller)",
    "trap": "The 2025 assignment: Top-1 of 8, same-month signal, no costs",
}
PRESET_SETTINGS = {
    "gem": dict(rule="gem", universe=["SPY", "EFA", "AGG"], cost=10, lag=1, missing="exclude", start=1993),
    "faber": dict(rule="sma", universe=["SPY"], window=10, safe=sg.CASH, cost=10, lag=1, missing="exclude", start=1993),
    "top3": dict(rule="dual", universe=DEFAULT_UNIVERSE, lookback="13612W", top_n=3, absolute="cash", safe="AGG",
                 cost=10, lag=1, missing="exclude", start=1993),
    "trap": dict(rule="dual", universe=DEFAULT_UNIVERSE, lookback=12, top_n=1, absolute="cash", safe="AGG",
                 cost=0, lag=0, missing="zero", start=1993),
}
TRAPS = [
    ("Same-month signal", 1, "The weights decided with month t's closing prices earn month t's return. Shift them one month."),
    ("Missing assets at 0%", 2, "An asset that did not exist yet 'earns' 0%, or is selected with a score of 0. Exclude it until listed."),
    ("No trading costs", 4, "A Top-1 rule trades several times a year; spreads and commissions come out of the return."),
    ("Mixed day counts / no risk-free rate", 6, "CAGR on calendar time, volatility on monthly returns × √12, Sharpe net of T-bills."),
    ("Parameter cherry-picking", 3, "Lookback 12 and Top-1 were chosen after seeing the results. Look at the neighbours."),
    ("Wrong benchmark", 5, "Compare with 60/40 and with equal weight of the same universe, not only with the S&P 500."),
    ("Start-date and rebalance-day luck", 7, "The result depends on the year you start and the day of the month you trade."),
]
DATA_FOLDERS = {"class": data.snapshot_dir(), "latest": data.DATA_DIR}


# --------------------------------------------------------------------------- data


@st.cache_data
def load_all():
    out = {}
    for key, folder in DATA_FOLDERS.items():
        px = data.load_prices(list(ASSETS), data_dir=folder)
        monthly = px.resample("ME").last()
        if monthly.index[-1] > px.index[-1]:  # the running month has no month-end close yet
            monthly = monthly.iloc[:-1]
        rate = data.load_fred("DTB3", data_dir=folder)
        cash = cash_returns_from_rate(rate, monthly.index)
        out[key] = dict(daily=px.loc[: monthly.index[-1]], monthly=monthly, cash=cash, rate=rate,
                        manifest=data.load_manifest(folder))
    return out


DATASETS = load_all()
LAST = {k: v["monthly"].index[-1].date() for k, v in DATASETS.items()}
DATA_LABEL = {"class": f"Class snapshot (month-ends to {LAST['class']:%b %Y})",
              "latest": f"Latest refresh (month-ends to {LAST['latest']:%b %Y})"}


# --------------------------------------------------------------------------- state

DEFAULTS = dict(dataset="class", preset="custom", rule="dual", universe=DEFAULT_UNIVERSE, lookback=12, top_n=1,
                absolute="cash", safe="AGG", window=10, gem_us="SPY", gem_intl="EFA", gem_bonds="AGG", cost=10, lag=1,
                missing="exclude", start=1993, show_ew=True, show_6040=True, log_scale=True)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def apply_preset() -> None:
    p = st.session_state.preset
    if p in PRESET_SETTINGS:
        for k, v in PRESET_SETTINGS[p].items():
            st.session_state[k] = v


def touched() -> None:
    """Any manual change turns the preset selector back to Custom (labels stay fixed)."""
    st.session_state.preset = "custom"


# --------------------------------------------------------------------------- header

st.title("Strategy Lab")
st.caption("DIS toolkit #5 · Arc stages: **Rules → Backtest → Robustness → Implementation** · "
           "every number comes from the shared `dis_lab` engine")
st.markdown("> **The decision:** GEM and Faber's 10-month rule are the most copied tactical rules in the world. "
            "Thirteen years after publication, would you run €100k of your own money on one of them, "
            "and what would make you stop?")
st.radio("Data", list(DATASETS), key="dataset", format_func=DATA_LABEL.get, horizontal=True,
         help="Class snapshot: the frozen data the slides were built on. Latest refresh: the same sources, "
              "updated monthly by a scheduled job (the student kit at the URL below follows this copy).")
D = DATASETS[st.session_state.dataset]
MONTHLY, CASH, DAILY = D["monthly"], D["cash"], D["daily"]

# --------------------------------------------------------------------------- sidebar: the rule

with st.sidebar:
    st.header("The rule")
    st.selectbox("Start from a known rule", list(PRESETS), key="preset", format_func=PRESETS.get, on_change=apply_preset)
    st.radio("Rule", list(RULES), key="rule", format_func=RULES.get, on_change=touched)
    rule = st.session_state.rule
    if rule == "gem":
        c1, c2, c3 = st.columns(3)
        c1.selectbox("US", ["SPY", "VFINX", "QQQ", "IWM"], key="gem_us", on_change=touched)
        c2.selectbox("Abroad", ["EFA", "EEM"], key="gem_intl", on_change=touched)
        c3.selectbox("Bonds", list(BONDS), key="gem_bonds", on_change=touched)
        universe = [st.session_state.gem_us, st.session_state.gem_intl, st.session_state.gem_bonds]
    else:
        st.multiselect("Universe", list(ASSETS), key="universe", format_func=lambda t: f"{t} · {ASSETS[t]}", on_change=touched)
        universe = list(st.session_state.universe)
    if rule == "dual":
        st.select_slider("Momentum lookback (months)", LOOKBACKS, key="lookback", on_change=touched,
                         help="Return over the last N months. 13612W = Keller's weighted mix of 1, 3, 6 and 12 months.")
        st.slider("Hold the Top-N", 1, 4, key="top_n", on_change=touched)
        st.radio("Absolute momentum filter", list(ABSOLUTE), key="absolute", format_func=ABSOLUTE.get, on_change=touched)
    if rule == "sma":
        st.slider("Moving-average window (months)", 3, 15, key="window", on_change=touched)
    if rule in ("dual", "sma"):
        safe_opts = [sg.CASH] + [b for b in BONDS if b in universe]
        if st.session_state.safe not in safe_opts:
            st.session_state.safe = sg.CASH
        st.selectbox("Safe asset (where vetoed picks go)", safe_opts, key="safe", on_change=touched,
                     format_func=lambda t: "CASH · T-bills" if t == sg.CASH else f"{t} · {ASSETS[t]}")
    st.slider("One-way trading cost (bps of value traded)", 0, 50, key="cost", on_change=touched)
    st.header("The timeline")
    st.radio("Signal timing", list(LAGS), key="lag", format_func=LAGS.get, on_change=touched)
    st.radio("Assets that did not exist yet", list(MISSING), key="missing", format_func=MISSING.get, on_change=touched)
    years = list(range(1987, LAST[st.session_state.dataset].year))
    if st.session_state.start not in years:
        st.session_state.start = years[0]
    st.selectbox("Start no earlier than", years, key="start", on_change=touched)
    st.header("Benchmarks")
    st.toggle("60/40 (S&P 500 / US bonds, monthly rebalanced)", key="show_6040")
    st.toggle("Equal weight of the universe", key="show_ew")
    st.toggle("Log scale", key="log_scale")

if not universe:
    st.info("Pick at least one asset in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- compute

S = st.session_state
start_ts = pd.Timestamp(f"{S.start}-01-01")


def build_targets(prices: pd.DataFrame, cash: pd.Series, settings: dict) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Target weights (and the score table that produced them) for the sidebar settings."""
    r = settings["rule"]
    if r == "dual":
        lb = settings["lookback"]
        score = sg.score_13612w(prices) if lb == "13612W" else sg.momentum(prices, int(lb))
        absolute = None if settings["absolute"] == "none" else settings["absolute"]
        w = sg.dual_momentum_weights(prices, lb, settings["top_n"], absolute, cash, settings["safe"], settings["missing"])
        return w, score
    if r == "sma":
        return sg.sma_weights(prices, settings["window"], settings["safe"], settings["missing"]), sg.sma(prices, settings["window"])
    if r == "gem":
        return sg.gem_weights(prices, settings["gem_us"], settings["gem_intl"], settings["gem_bonds"], cash), \
            sg.momentum(prices[[settings["gem_us"], settings["gem_intl"]]], 12)
    return sg.equal_weights(prices, settings["missing"]), None


settings = {k: S[k] for k in ("rule", "lookback", "top_n", "absolute", "safe", "window", "gem_us", "gem_intl", "gem_bonds",
                              "cost", "lag", "missing")}
prices = MONTHLY[universe]
try:
    targets, score = build_targets(prices, CASH, settings)
    targets = targets.loc[start_ts:]
    if targets.empty:
        raise ValueError("no decision date on or after the start year; move the start earlier or shorten the lookback")
    result = signal_backtest(prices, targets, lag=S.lag, cost_bps=S.cost, cash_returns=CASH, missing=S.missing, name="Strategy")
except (ValueError, KeyError) as exc:
    st.error(f"The engine refused this backtest: {exc}")
    st.stop()

nav = result.nav
window = nav.index
rf = data.risk_free_rate(window[0], window[-1], data_dir=DATA_FOLDERS[S.dataset])

benchmarks: dict[str, pd.Series] = {}
spx = MONTHLY["VFINX"].reindex(window).dropna()
benchmarks["S&P 500 (index fund)"] = 100 * spx / spx.iloc[0]
if S.show_6040:
    w6040 = pd.DataFrame({"VFINX": 0.6, "VBMFX": 0.4}, index=window)
    benchmarks["60/40"] = signal_backtest(MONTHLY, w6040, lag=0, cost_bps=S.cost, name="60/40").nav
if S.show_ew and S.rule != "ew" and len(universe) > 1:
    try:
        ew = sg.equal_weights(prices).loc[window[0]:]
        benchmarks["Equal weight"] = signal_backtest(prices, ew, lag=0, cost_bps=S.cost, name="Equal weight").nav
    except ValueError:
        pass
colors = ui.assign_colors(["Strategy"] + list(benchmarks), S.setdefault("colors", {}))

# --------------------------------------------------------------------------- tabs

tabs = st.tabs(["1 · Results", "2 · Glass box", "3 · Robustness", "What could be wrong here", "Build your own", "How this works"])

with tabs[0]:
    rule_text = {
        "dual": f"Each month-end, rank the {len(universe)} assets by their {S.lookback}-month return"
                f"{' (13612W weighted)' if S.lookback == '13612W' else ''}; hold the Top-{S.top_n} at equal weight. "
                + {"cash": "A pick that did not beat T-bills over the lookback goes to the safe asset.",
                   "zero": "A pick with a negative return goes to the safe asset.", "none": "No absolute filter."}[S.absolute],
        "sma": f"Each month-end, hold each asset whose price is above its {S.window}-month average (1/N each); "
               "an asset below its average sends its share to the safe asset.",
        "gem": f"Each month-end: if {S.gem_us} beat T-bills over 12 months, hold whichever of {S.gem_us} / {S.gem_intl} "
               f"rose more over 12 months; otherwise hold {S.gem_bonds}.",
        "ew": "Hold every asset in the universe at 1/N, rebalanced monthly.",
    }[S.rule]
    st.markdown(f"**The rule in plain English.** {rule_text} Weights decided at month-end *t* are held during "
                f"{'month *t+1*' if S.lag == 1 else 'month *t* itself (look-ahead)'}; one-way cost {S.cost} bps on value traded.")
    summ = m.summary(nav, benchmarks.get("S&P 500 (index fund)"), rf)
    k = st.columns(7)
    k[0].metric("CAGR", ui.pct(summ["cagr"]))
    k[1].metric("Volatility", ui.pct(summ["vol"]))
    k[2].metric("Sharpe", ui.num(summ["sharpe"]), help=f"rf = {rf:.2%} (3-month T-bill average over the window)")
    k[3].metric("Max drawdown", ui.pct(summ["max_dd"]))
    k[4].metric("Worst year", ui.pct(summ["worst_year"]))
    k[5].metric("Trades / year", ui.num(result.trades / max(summ["years"], 1e-9), 1))
    k[6].metric("Cost / year", ui.pct(result.annual_cost, 2))
    if S.lag == 0:
        st.error("Look-ahead is ON: the signal earns the month it was computed from. Nobody can trade on a month-end "
                 "close before it exists. Switch 'Signal timing' back to honest before believing any number here.")
    if S.missing == "zero":
        st.warning("Missing assets count as 0%: the backtest 'holds' assets before they were listed.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=nav.index, y=nav, name="Strategy", line=dict(color=colors["Strategy"], width=2.5)))
    for name_, s in benchmarks.items():
        fig.add_trace(go.Scatter(x=s.index, y=s, name=name_, line=dict(color=colors[name_], width=1.5, dash="dot")))
    fig.update_layout(title="Growth of 100", yaxis_type="log" if S.log_scale else "linear", height=420,
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15), hovermode="x unified")
    fig.update_yaxes(gridcolor=ui.GRID_COLOR)
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        dd = go.Figure()
        dd.add_trace(go.Scatter(x=nav.index, y=m.drawdown_series(nav), name="Strategy", fill="tozeroy",
                                line=dict(color=colors["Strategy"])))
        for name_, s in benchmarks.items():
            dd.add_trace(go.Scatter(x=s.index, y=m.drawdown_series(s), name=name_, line=dict(color=colors[name_], dash="dot")))
        dd.update_layout(title="Drawdowns", yaxis_tickformat=".0%", height=320, margin=dict(l=10, r=10, t=40, b=10),
                         legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(dd, width="stretch")
    with c2:
        held = result.held
        al = go.Figure()
        for j, col in enumerate(held.columns):
            if (held[col] > 0).any():
                al.add_trace(go.Scatter(x=held.index, y=held[col], name=col, stackgroup="one",
                                        line=dict(width=0, color=ui.SERIES_COLORS[j % len(ui.SERIES_COLORS)])))
        al.update_layout(title="What the strategy held, month by month", yaxis_tickformat=".0%", height=320,
                         margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(al, width="stretch")

    st.subheader("Metrics on the common window")
    table = pd.DataFrame({"Strategy": summ} | {k_: m.summary(v, benchmarks.get("S&P 500 (index fund)") if k_ != "S&P 500 (index fund)" else None, rf)
                                              for k_, v in benchmarks.items()})
    rows = ["start", "end", "years", "cagr", "vol", "sharpe", "sortino", "max_dd", "tuw_days", "calmar", "pct_pos_years", "worst_year"]
    shown = table.loc[rows].copy()
    shown.index = [m.METRIC_LABELS.get(r, r) for r in rows]
    st.dataframe(shown.apply(lambda col: [ui.format_metric(r, v) for r, v in zip(rows, col)]), width="stretch")
    se = m.sharpe_standard_error(summ["sharpe"], summ["years"])
    st.caption(f"Sharpe {summ['sharpe']:.2f} ± {se:.2f} (one standard error, Lo 2002): differences smaller than about "
               f"{2 * se:.2f} between two strategies on this window are noise. Turnover {result.annual_turnover:.0%} per year one-way.")

    c3, c4 = st.columns([2, 1])
    with c3:
        st.subheader("Calendar years")
        all_navs = {"Strategy": nav} | benchmarks
        cy = pd.DataFrame({k_: m.calendar_year_returns(v)["return"] for k_, v in all_navs.items()})
        partial = m.calendar_year_returns(nav)["partial"]
        cy.index = [f"{y}*" if partial.get(y, False) else str(y) for y in cy.index]
        st.dataframe(cy.T.style.format("{:.1%}", na_rep=""), width="stretch")
        st.caption("* partial year")
    with c4:
        st.subheader("Current signal")
        last_decision = targets.index[-1]
        current = targets.iloc[-1]
        current = current[current > 0]
        st.markdown(f"Decided at the close of **{last_decision:%d %b %Y}**, to hold during the following month:")
        st.dataframe(current.rename("weight").to_frame().style.format("{:.0%}"), width="stretch")
        if score is not None and S.rule in ("dual", "gem"):
            st.caption("Scores on that date: " + ", ".join(f"{a} {v:+.1%}" for a, v in score.loc[last_decision].dropna().sort_values(ascending=False).items()))

with tabs[1]:
    st.markdown(
        "The whole backtest, one table per step, so that any month can be checked by hand. Row *t* of every table is "
        "the month ending on that date. **Scores** use prices up to *t*; **targets** are decided at the close of *t*; "
        f"**held weights** are the targets shifted {S.lag} month{'s' if S.lag != 1 else ''}; the **strategy return** of "
        "month *t* is held weights × asset returns of *t*, minus the cost of moving from last month's drifted weights to the new ones.")
    n_show = st.slider("Months shown (from the end)", 6, 60, 12, key="glass_n")
    steps = {
        "1 · Prices (month-end, adjusted)": prices.loc[result.held.index].tail(n_show),
        "2 · Asset returns (month over month)": result.asset_returns.tail(n_show),
    }
    if score is not None:
        steps["3 · Scores (what the rule ranks)"] = score.reindex(result.held.index).tail(n_show)
        if S.rule == "dual":
            steps["4 · Ranks (1 = best, among assets with a score)"] = score.reindex(result.held.index).rank(axis=1, ascending=False).tail(n_show)
    steps["5 · Target weights decided at the close of each month"] = targets.reindex(result.held.index).tail(n_show)
    steps[f"6 · Weights actually held during the month (targets shifted {S.lag})"] = result.held.tail(n_show)
    steps["7 · Cost paid, strategy return, NAV"] = pd.DataFrame({
        "cost paid": result.costs.reindex(result.held.index).fillna(0.0),
        "strategy return": result.returns, "NAV": result.nav}).tail(n_show)
    for title, df in steps.items():
        with st.expander(title, expanded=title.startswith(("5", "6", "7"))):
            fmt = "{:.2f}" if title.startswith("1") or title.startswith("4") else ("{:.1%}" if not title.startswith("7") else None)
            st.dataframe(df.style.format(fmt, na_rep="") if fmt else df.style.format({"cost paid": "{:.3%}", "strategy return": "{:.2%}", "NAV": "{:.2f}"}),
                         width="stretch")
    st.subheader("Inspect one month")
    pick = st.selectbox("Month", list(result.held.index[::-1]), key="glass_month", format_func=lambda t: f"{t:%b %Y}")
    st.dataframe(trace(result, pick, score).style.format("{:.2%}", na_rep=""), width="stretch")
    full = pd.concat({"price": prices.loc[result.held.index], "target": targets.reindex(result.held.index),
                      "held": result.held, "asset_return": result.asset_returns}, axis=1)
    full[("strategy", "cost")] = result.costs.reindex(result.held.index).fillna(0.0)
    full[("strategy", "return")] = result.returns
    full[("strategy", "nav")] = result.nav
    st.download_button("Download the full trace (CSV)", full.to_csv(float_format="%.6f"), "strategy_trace.csv", "text/csv")

with tabs[2]:
    st.markdown("Four questions for any backtest: how do the **neighbours** of the chosen parameters do, does the result "
                "survive another **rebalance day**, how did it do **after** the rule was published (or after a hold-out "
                "date), and how did it behave in each **crisis**?")
    rf_grid = rf

    if S.rule == "dual":
        st.subheader("Parameter neighbourhood")
        metric = st.selectbox("Colour by", ["sharpe", "cagr", "max_dd", "worst_year"], key="grid_metric",
                              format_func=lambda k_: m.METRIC_LABELS.get(k_, k_))

        def rule_fn(lookback, top_n):
            absolute = None if S.absolute == "none" else S.absolute
            w = sg.dual_momentum_weights(prices, lookback, top_n, absolute, CASH, S.safe, S.missing).loc[start_ts:]
            return signal_backtest(prices, w, lag=S.lag, cost_bps=S.cost, cash_returns=CASH, missing=S.missing).nav

        grid = rb.parameter_grid(rule_fn, {"lookback": LOOKBACKS, "top_n": [1, 2, 3, 4]}, rf=rf_grid)
        grid["lookback"] = grid["lookback"].astype(str)
        piv = grid.pivot(index="lookback", columns="top_n", values=metric).reindex([str(x) for x in LOOKBACKS])
        hm = go.Figure(go.Heatmap(z=piv.to_numpy(dtype=float), x=[f"Top-{c}" for c in piv.columns], y=[str(i) for i in piv.index],
                                  colorscale="RdBu" if metric != "max_dd" else "RdBu_r", text=np.round(piv.to_numpy(dtype=float), 2),
                                  texttemplate="%{text}", colorbar=dict(title=m.METRIC_LABELS.get(metric, metric))))
        hm.update_layout(title=f"{m.METRIC_LABELS.get(metric, metric)} across lookback × Top-N (same universe, costs and timing)",
                         yaxis_title="lookback (months)", height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(hm, width="stretch")
        chosen = grid[(grid["lookback"].astype(str) == str(S.lookback)) & (grid["top_n"] == S.top_n)]
        if not chosen.empty and chosen[metric].notna().all():
            better = (grid[metric] > chosen[metric].iloc[0]).sum() if metric != "max_dd" else (grid[metric] > chosen[metric].iloc[0]).sum()
            st.caption(f"Your cell ranks {int(better) + 1} of {int(grid[metric].notna().sum())} neighbours on {m.METRIC_LABELS.get(metric, metric)}. "
                       f"Spread across the grid: Sharpe {grid['sharpe'].min():.2f} to {grid['sharpe'].max():.2f}. "
                       "If only the best cell works, you found noise, not a strategy.")
    elif S.rule == "sma":
        st.subheader("Window neighbourhood")
        rows_ = []
        for wdw in range(3, 16):
            try:
                w = sg.sma_weights(prices, wdw, S.safe, S.missing).loc[start_ts:]
                nv = signal_backtest(prices, w, lag=S.lag, cost_bps=S.cost, cash_returns=CASH, missing=S.missing).nav
                rows_.append({"window": wdw, "CAGR": m.cagr(nv), "Sharpe": m.sharpe(nv, rf_grid), "Max drawdown": m.max_drawdown(nv)})
            except ValueError:
                pass
        wt = pd.DataFrame(rows_).set_index("window")
        st.dataframe(wt.T.style.format("{:.2f}"), width="stretch")
        st.caption("Faber chose 10 months in 2007. If 8 or 12 give very different results, the rule is fragile.")

    st.subheader("Rebalance-day luck")
    st.markdown("The same rule, with 'month' defined as the 1st, 6th, 11th or 16th trading day instead of the month-end. "
                "Nothing about the strategy changed; only the calendar.")
    if st.button("Run the five calendars (a few seconds)", key="run_luck"):
        daily = DAILY[universe].dropna(how="all")

        def rule_daily(monthly_px: pd.DataFrame) -> pd.Series:
            c = cash_returns_from_rate(D["rate"], monthly_px.index)
            w, _ = build_targets(monthly_px, c, settings)
            w = w.loc[start_ts:]
            return signal_backtest(monthly_px, w, lag=S.lag, cost_bps=S.cost, cash_returns=c, missing=S.missing).nav

        try:
            luck = rb.rebalance_day_luck(daily, rule_daily, days=(0, 5, 10, 15, -1), rf=rf_grid)
            st.dataframe(luck[["cagr", "vol", "sharpe", "max_dd", "worst_year"]].astype(float).style.format("{:.3f}"), width="stretch")
            st.caption(f"CAGR from {luck['cagr'].min():.1%} to {luck['cagr'].max():.1%}, max drawdown from "
                       f"{luck['max_dd'].max():.0%} to {luck['max_dd'].min():.0%}: that range is luck, not skill (Hoffstein et al., 2019).")
        except (ValueError, KeyError) as exc:
            st.warning(f"Could not run every calendar: {exc}")

    st.subheader("Before and after a date")
    default_split = pd.Timestamp("2013-12-31") if nav.index[0] < pd.Timestamp("2010-01-01") < nav.index[-1] else nav.index[len(nav) // 2]
    split = st.date_input("Split date (publication date, or the start of your hold-out)", default_split.date(),
                          min_value=nav.index[0].date(), max_value=nav.index[-1].date(), key="split")
    sp = rb.split_periods(nav, pd.Timestamp(split), rf=rf_grid, labels=("Before", "After"))
    bench_sp = {k_: rb.split_periods(v, pd.Timestamp(split), rf=rf_grid, labels=("Before", "After")) for k_, v in benchmarks.items()}
    comb = pd.concat({"Strategy": sp} | bench_sp, axis=1)
    st.dataframe(comb.loc[["cagr", "vol", "sharpe", "max_dd", "worst_year"]].astype(float).style.format("{:.3f}"), width="stretch")
    st.caption("GEM was published in 2013 (book 2014) and Faber's rule in 2007. McLean & Pontiff (2016): published "
               "predictors lose about 58% of their return after publication.")

    st.subheader("Crises")
    ep = rb.subperiod_table({"Strategy": nav} | benchmarks)
    st.dataframe(ep.style.format("{:.1%}", na_rep="not covered"), width="stretch")
    st.caption("Total return from the S&P 500's closing peak to its closing trough. A trend rule needs time to react: "
               "it helps in long bear markets (2000–02, 2008) and hurts in fast ones (Covid, April 2025).")

with tabs[3]:
    st.markdown("The seven traps of an AI-built tactical backtest, and whether this one has them switched on.")
    status = {
        "Same-month signal": ("on", "Signal timing is set to same month.") if S.lag == 0 else ("off", "Signal lagged one month."),
        "Missing assets at 0%": ("on", "Missing assets count as 0%.") if S.missing == "zero" else ("off", "Assets excluded until listed."),
        "No trading costs": ("on", "Cost is 0 bps.") if S.cost == 0 else ("off", f"{S.cost} bps one-way on value traded."),
        "Mixed day counts / no risk-free rate": ("off", f"Engine definitions: CAGR on calendar days, vol × √12, Sharpe net of rf {rf:.2%}."),
        "Parameter cherry-picking": ("check", "Open Robustness → parameter neighbourhood and read your cell's rank."),
        "Wrong benchmark": ("off", "60/40 and equal weight shown.") if (S.show_6040 and S.show_ew) else ("on", "Switch on 60/40 and equal weight in the sidebar."),
        "Start-date and rebalance-day luck": ("check", "Change 'Start no earlier than' and run the five calendars."),
    }
    for title, item, text in TRAPS:
        state, note = status[title]
        icon = {"on": "🔴", "off": "🟢", "check": "🟡"}[state]
        st.markdown(f"{icon} **{title}** · checklist item {item}  \n{text}  \n*Here:* {note}")
    if summ["sharpe"] > 1.5 or summ["cagr"] > 0.25:
        st.error(f"Sanity (item 6): Sharpe {summ['sharpe']:.2f} / CAGR {summ['cagr']:.1%} on a monthly asset-allocation rule. "
                 "Assume a bug until proven otherwise.")

with tabs[4]:
    st.markdown("### Build your own strategy lab with your AI assistant")
    st.markdown(
        "You are the PM; the assistant is your quant analyst. Paste the prompt below into Claude, ChatGPT, Gemini or "
        "Copilot (free tiers work), iterate until the page opens in your browser and runs on the data, then **audit it** "
        "with the seven traps and compare its numbers with this page. Save the `.html` file: it is yours to keep and change.")
    st.markdown(f"**Data.** Month-end prices in US dollars, refreshed monthly: `{KIT_URL}` (your page can fetch it "
                f"directly; a copy can also be uploaded). Daily prices per ticker: `{REPO}/tree/main/data/prices`.")
    prompt = f"""You are my quant analyst. Build a single-file HTML page (HTML + CSS + plain JavaScript, no build step; Plotly from a CDN is fine) called "My Strategy Lab" that backtests tactical asset allocation rules on monthly data.

DATA. By default fetch {KIT_URL} (it allows cross-origin requests). Also offer a file picker for a local copy. Lines starting with "#" are comments. The header row is: date, then one column per asset (month-end adjusted closes in US dollars), TBILL (the value of cash earning the 3-month T-bill rate, base 100) and TBILL_RATE_PCT. A blank cell means the asset did not exist yet: exclude it from the ranking until it has the prices the rule needs, and never treat a blank as a 0% return.

RULES. (1) Dual momentum: score each asset by its N-month return (N selectable: 1, 3, 6, 9, 12) or by the 13612W score = (12·r1 + 4·r3 + 2·r6 + r12) / 4; hold the top K assets (K selectable, 1 to 4) at equal weight; absolute filter: a pick whose return over the lookback is below TBILL's return over the same window goes to a safe asset instead (selectable: TBILL or a bond column). (2) Faber: hold each asset only when its price is above its 10-month moving average, otherwise its share goes to TBILL; equal weight among the assets that have an average. (3) Benchmarks: buy and hold SPY; 60/40 SPY/AGG rebalanced monthly; equal weight of the chosen universe.

TIMING. The signal computed from the prices in row t decides the weights held during month t+1, so apply them to the next row's returns. Add a clearly labelled switch "same-month (look-ahead, wrong)" that applies them to row t, so that I can see the size of that error.

COSTS. A one-way trading cost in basis points on the value traded each month (default 10), charged whenever the weights change.

OUTPUTS. Growth of 100 for the strategy and the benchmarks (log-scale option); drawdown chart; a metrics table with CAGR (using calendar days / 365.25), annual volatility (standard deviation of monthly returns × √12), Sharpe = (CAGR − TBILL CAGR) / volatility, max drawdown on the value path, worst calendar year, number of trades and turnover; a calendar-year returns table; the current holdings as of the last row; an allocation-over-time chart. Show the parameters next to the results, and a "what could be wrong" box listing every assumption.

GLASS BOX. Keep every intermediate series inspectable: one table with, per month, prices, scores, rank, target weights, held weights, cost, strategy return and NAV, downloadable as CSV.

Start by restating the rules in your own words and listing the design choices you are making; then write the page."""
    st.code(prompt, language="text")
    st.markdown("""**Audit prompt (after it runs).** *"Check your own work: (1) show me, for one month, the prices used for the signal and the returns the weights were applied to, with their dates; (2) cut the data at December 2019, re-run, and confirm that no signal before December 2019 changed; (3) list every place where a missing value could become a zero; (4) tell me how CAGR, volatility and Sharpe are computed, with the day counts."*

**Then compare** your page with this one on the same rule (GEM, Faber, Top-3): the engine's numbers are the reference. Differences of a few tenths of a percent come from data details; differences of several points mean one of the seven traps.""")
    st.markdown(f"Source of this page and of every number in it: [{REPO}]({REPO}) (`engine/signals.py`, `engine/backtest.py`, `engine/robustness.py`).")

with tabs[5]:
    st.markdown(f"""
**Data.** {DATA_LABEL[S.dataset]}. Yahoo Finance adjusted closes (dividends reinvested), sampled on the last trading day of each
month; the running month is dropped. Cash = 3-month T-bill (FRED DTB3) accrued on calendar days / 360. The risk-free rate in the
Sharpe ratio is the T-bill average over the window shown.

**Timeline.** A score computed from prices up to month-end *t* is known at that close. With honest timing the weights it implies
earn the returns of month *t+1*; the trade from last month's drifted weights to the new ones pays the one-way cost on the value traded.
The first position is free. Turnover is one-way (half the value traded).

**Late-starting assets** have no score until they have the prices the rule needs, and are never selected before that (unless the
trap is switched on). A rule that needs 12 months of prices starts 12 months after the youngest asset in the universe lists.

**Benchmarks.** S&P 500 = Vanguard 500 Index Fund (VFINX, since 1976, so every window has it); 60/40 = VFINX / VBMFX rebalanced
monthly at the same cost; equal weight = 1/N of the universe, rebalanced monthly.

**Definitions** (course section 7): CAGR on calendar days / 365.25; volatility = standard deviation of monthly returns × √12;
Sharpe = (CAGR − rf) / volatility; max drawdown on the NAV path; time under water = longest stretch below a previous high.

Engine `dis_lab` v{ENGINE_VERSION}. {ui.DISCLAIMER}
""")
    st.dataframe(ui.glossary_table(["cagr", "vol", "sharpe", "sortino", "max_dd", "tuw_days", "calmar", "worst_year", "sharpe_se"]),
                 width="stretch", hide_index=True)
