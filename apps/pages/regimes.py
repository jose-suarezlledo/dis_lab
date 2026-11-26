"""Strategy Lab · Regimes (Session 5): regime monitor, signal scorecard, regimes and returns, macro filter for a trend
rule, portfolio weather report, real-time vs revised data. Every number comes from engine/regimes.py.

Opens as the second page of apps/strategy_lab.py (Streamlit multipage: apps/pages/), URL .../regimes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[2])]

import dis_ui as ui  # noqa: E402
from engine import __version__ as ENGINE_VERSION  # noqa: E402
from engine import data  # noqa: E402
from engine import metrics as m  # noqa: E402
from engine import regimes as rg  # noqa: E402
from engine import signals as sg  # noqa: E402
from engine.backtest import cash_returns_from_rate, signal_backtest  # noqa: E402

st.set_page_config(page_title="Regimes · Strategy Lab · DIS", page_icon="🧭", layout="wide")

DATA_FOLDERS = {"latest": data.DATA_DIR, "class": data.snapshot_dir()}
FEATURES = {"IP_YOY": "Growth: industrial production, % y/y", "CPI_YOY": "Inflation: consumer prices, % y/y",
            "UNRATE_CHG12": "Labour: unemployment change over 12 months", "SLOPE": "Yield curve: 10y − 3m",
            "CREDIT": "Credit: Baa − 10y spread"}
LONG_ASSETS = ["US stocks", "US 10y Treasuries (approx.)", "Cash (T-bills)"]
FACTOR_ASSETS = ["Value − growth (HML)", "Small − big (SMB)", "Momentum (MOM)"]
QUAD = rg.QUADRANT_NAMES
EXCEL_ETFS = ["IVE", "IVW", "SPHQ", "SPY", "IWM"]
U8 = ["SPY", "EFA", "EEM", "AGG", "TLT", "GLD", "DBC", "VNQ"]


@st.cache_data
def load(key: str):
    folder = DATA_FOLDERS[key]
    panel = rg.macro_panel(folder)
    known = rg.known_at(panel)
    rets = rg.long_returns(folder)
    px = data.load_prices(list(dict.fromkeys(EXCEL_ETFS + U8 + ["VFINX", "VBMFX"])), data_dir=folder)
    monthly = px.resample("ME").last()
    if monthly.index[-1] > px.index[-1]:
        monthly = monthly.iloc[:-1]
    cash = cash_returns_from_rate(data.load_fred("DTB3", folder), monthly.index)
    payrolls = rg.realtime_vs_revised_payrolls(folder)
    return panel, known, rets, monthly, cash, payrolls


@st.cache_data
def track_record(key: str, cols: tuple[str, ...], asset: str, horizon: int, k: int):
    _, known, rets, *_ = load(key)
    ze = rg.standardize(known[list(cols)].dropna(), expanding=True)
    return rg.analog_track_record(ze, rets[asset], horizon, k)


st.title("Regimes")
st.caption("DIS toolkit #4 · Session 5 · Arc stages: **Idea → Data → Rules / Model → Backtest** · every number comes from the shared `dis_lab` engine")
st.markdown("> **The decision:** a fund (or your AI) says it rotates by macro regime and shows Sharpe 0.9 vs 0.6. Would you move €100k of a world index fund "
            "into it, knowing the regime labels were assigned with revised data and full-sample averages?")
st.radio("Data", list(DATA_FOLDERS), key="rg_dataset", horizontal=True,
         format_func={"latest": "Latest refresh (macro to the last published month)", "class": "Class snapshot (to 13 Oct 2026)"}.get)
PANEL, KNOWN, RETS, MONTHLY, CASH, PAY = load(st.session_state.rg_dataset)

tabs = st.tabs(["1 · Regime monitor", "2 · Signal scorecard", "3 · Regimes and returns", "4 · Macro filter on a trend rule",
                "5 · Portfolio weather report", "6 · Real-time vs revised", "Build your own", "How this works"])

# --------------------------------------------------------------------------- 1 regime monitor
with tabs[0]:
    st.markdown("**What happened the last times the economy looked like today?** Five macro readings, each as an investor could read it at the month-end "
                "(publication lags applied), are turned into z-scores; the tool finds the past months closest to the latest one, at least six months apart, "
                "and shows what assets did in the following months. Then it tells you how good this method has been.")
    c1, c2, c3 = st.columns([2, 1, 1])
    feats = c1.multiselect("Readings", list(FEATURES), default=list(FEATURES), format_func=FEATURES.get, key="rg_feats")
    k = c2.slider("Number of analogs", 5, 20, 10, key="rg_k")
    horizon = c3.select_slider("Horizon (months)", [3, 6, 12, 24], 12, key="rg_h")
    if len(feats) < 2:
        st.info("Pick at least two readings.")
    else:
        F = KNOWN[feats].dropna()
        Z = rg.standardize(F)
        last = Z.index[-1]
        now = pd.DataFrame({"reading": [FEATURES[f] for f in feats], "value known at the close of " + f"{last:%b %Y}": F.loc[last].round(2).to_numpy(),
                            "z-score (vs 1953–today)": Z.loc[last].round(2).to_numpy(),
                            "describes the month of": [(last - pd.offsets.MonthEnd(rg.PUB_LAG.get(f, 0))).strftime("%b %Y") for f in feats]})
        st.subheader(f"Today's reading (as known at the close of {last:%B %Y})")
        st.dataframe(now, hide_index=True, width="stretch")
        an = rg.nearest_months(Z, last, k=k, horizon=horizon)
        per, summ = rg.analog_outcomes(RETS[LONG_ASSETS + FACTOR_ASSETS], an, horizon)
        st.subheader(f"The {k} most similar months, and the next {horizon} months")
        show = per.copy()
        show.insert(0, "distance", an.set_index("date")["distance"].round(2))
        for f in feats:
            show.insert(1, FEATURES[f].split(":")[0], F.reindex(per.index)[f].round(1))
        show.index = show.index.strftime("%b %Y")
        st.dataframe(show.style.format({c: "{:.1%}" for c in per.columns}, na_rep="–"), width="stretch")
        s2 = summ.T[LONG_ASSETS + FACTOR_ASSETS]
        st.dataframe(s2.style.format("{:.1%}", subset=pd.IndexSlice[[i for i in s2.index if i != "analogs with data"], :]).format("{:.0f}", subset=pd.IndexSlice[["analogs with data"], :]),
                     width="stretch")
        st.caption("Compare the analogs with 'all months': if the analogs' median and % positive look like the base rate, today's macro picture says little about the next "
                   "year. Bonds before 1986 are approximated from the 10-year yield (carry − 7.5 × change in yield).")

        st.subheader("How good has this method been? (honest backtest)")
        st.markdown("At every month since 1975, using only data available then (expanding z-scores, analogs whose outcomes were already known), predict US stocks' next "
                    f"{horizon} months with the analogs' median, and compare with what happened.")
        if st.button("Run the track record (a few seconds)", key="rg_track"):
            tr = track_record(st.session_state.rg_dataset, tuple(feats), "US stocks", horizon, k)
            hit = float(((tr["predicted"] > 0) == (tr["realized"] > 0)).mean())
            base = float((tr["realized"] > 0).mean())
            corr = float(tr[["predicted", "realized"]].corr().iloc[0, 1])
            q = tr["predicted"].rank(pct=True)
            low, high = tr.loc[q <= 0.2, "realized"].mean(), tr.loc[q >= 0.8, "realized"].mean()
            a, b, c_, d = st.columns(4)
            a.metric("Correlation predicted vs realized", f"{corr:.2f}")
            b.metric("Sign right", f"{hit:.0%}", help="Share of months where the sign of the prediction matched the realized return")
            c_.metric("Base rate (always say 'up')", f"{base:.0%}")
            d.metric("Realized: lowest vs highest fifth of predictions", f"{low:.1%} vs {high:.1%}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=tr["predicted"], y=tr["realized"], mode="markers", marker=dict(size=4, color=ui.SERIES_COLORS[0], opacity=0.5),
                                     text=tr.index.strftime("%b %Y"), name="month"))
            fig.update_layout(height=360, xaxis_title=f"predicted next {horizon}m (analogs' median)", yaxis_title="realized", xaxis_tickformat=".0%",
                              yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
            st.caption("If the dots form a cloud, the monitor is a way to ask good questions about today, not a forecast. Use it to know which past episodes to read about.")

# --------------------------------------------------------------------------- 2 signal scorecard
with tabs[1]:
    st.markdown("**This recession signal just flashed: what is its record?** Each trigger = the first month the signal is on after 12 months off, on data as known at "
                "the time. Recessions are the NBER's dates, which are announced months later: the scorecard is a judgment on history, not a real-time signal.")
    flags = rg.signal_flags(KNOWN)
    rows = []
    for key_, (label, *_r) in rg.SIGNALS.items():
        sc = rg.signal_scorecard(flags[key_], PANEL["USREC"], RETS["US stocks"])
        f_ = flags[key_].dropna()
        early = sc[~sc["already in recession"]]
        rows.append({"signal": label, "on now?": "yes" if f_.iloc[-1] == 1 else "no", "as of": f"{f_.index[-1]:%b %Y}", "since": f"{f_.index[0]:%Y}",
                     "triggers": len(sc), "recession followed (≤ 24m)": int(early["recession followed"].sum()),
                     "false alarms": int((~early["recession followed"]).sum()), "late (already in recession)": int(sc["already in recession"].sum()),
                     "median lead (months)": early["lead (months)"].median(), "stocks next 12m (median)": sc["stocks next 12m"].median()})
    card = pd.DataFrame(rows)
    st.dataframe(card.style.format({"median lead (months)": "{:.0f}", "stocks next 12m (median)": "{:.1%}"}, na_rep="–"), hide_index=True, width="stretch")
    pick = st.selectbox("Look at one signal", list(rg.SIGNALS), format_func=lambda k_: rg.SIGNALS[k_][0], key="rg_sig")
    sc = rg.signal_scorecard(flags[pick], PANEL["USREC"], RETS["US stocks"])
    col = rg.SIGNALS[pick][1]
    series = KNOWN[col] if col != "CREDIT_JUMP" else (KNOWN["CREDIT"] - KNOWN["CREDIT"].rolling(12, min_periods=12).min())
    series = series.dropna()
    fig = go.Figure()
    rec = PANEL["USREC"].reindex(series.index).fillna(0)
    fig.add_trace(go.Scatter(x=series.index, y=np.where(rec == 1, series.max(), np.nan), fill="tozeroy", mode="none", fillcolor="rgba(137,135,129,0.25)", name="NBER recession"))
    fig.add_trace(go.Scatter(x=series.index, y=series, name=rg.SIGNALS[pick][0], line=dict(color=ui.SERIES_COLORS[0])))
    fig.add_hline(y=rg.SIGNALS[pick][3], line_dash="dot", line_color=ui.BENCHMARK_COLOR)
    for t in sc["trigger"]:
        fig.add_vline(x=t, line_color="#e34948", line_width=1)
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    st.dataframe(sc.assign(trigger=sc["trigger"].dt.strftime("%b %Y")).style.format({"stocks next 12m": "{:.1%}", "lead (months)": "{:.0f}"}, na_rep="–"),
                 hide_index=True, width="stretch")

# --------------------------------------------------------------------------- 3 regimes and returns
with tabs[2]:
    st.markdown("**Which box are we in, and what did assets do in each box?** Growth and a second variable are compared with their average; four boxes. "
                "Change how the average is computed and whether publication lags are respected, and watch the labels and the 'returns by regime' move.")
    a, b, c, d = st.columns(4)
    yvar = a.selectbox("Second variable", ["CPI_YOY", "DGS2"], format_func={"CPI_YOY": "Inflation (CPI y/y)", "DGS2": "2-year yield (the 2025 spreadsheet)"}.get, key="rg_y")
    mode = b.radio("Average used as threshold", ["rolling", "expanding", "full"], key="rg_mode",
                   format_func={"rolling": "Rolling 36 months", "expanding": "Expanding (all data up to then)", "full": "Full sample (look-ahead)"}.get)
    lagged = c.toggle("Respect publication lags", True, key="rg_lag")
    nextm = d.toggle("Match regime with NEXT month's returns", True, key="rg_next")
    src = KNOWN if lagged else PANEL
    reg = rg.threshold_regimes(src["IP_YOY"], src[yvar], mode, 36)
    ref = rg.threshold_regimes(KNOWN["IP_YOY"], KNOWN[yvar], "rolling", 36)
    names = {"A": "Growth up, " + ("inflation" if yvar == "CPI_YOY" else "2y yield") + " up", "B": "Growth down, up", "C": "Growth down, down", "D": "Growth up, down"}
    st.caption(f"Current label: **{reg.dropna().iloc[-1]} · {names[reg.dropna().iloc[-1]]}** (as of {reg.dropna().index[-1]:%b %Y}). "
               f"Agreement with the honest version (rolling, lagged): **{rg.label_agreement(reg, ref):.0%}** of months.")
    lab = reg.dropna()
    colors = {"A": "#e34948", "B": "#eda100", "C": "#2a78d6", "D": "#1baf7a"}
    tl = go.Figure()
    for r_ in "ABCD":
        mask = (lab == r_).astype(float)
        tl.add_trace(go.Bar(x=lab.index, y=mask, name=f"{r_} · {names[r_]}", marker_color=colors[r_], marker_line_width=0))
    tl.update_layout(barmode="stack", bargap=0, height=170, yaxis_visible=False, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.4))
    st.plotly_chart(tl, width="stretch")
    tab = rg.returns_by_regime(RETS[LONG_ASSETS + FACTOR_ASSETS], reg, next_month=nextm)
    piv = tab.pivot(index="asset", columns="regime", values="ann_return").reindex(LONG_ASSETS + FACTOR_ASSETS)
    piv.columns = [f"{c_} · {names[c_]}" for c_ in piv.columns]
    st.markdown("**Annualized return by regime**" + (" (regime known at t, return of t+1)" if nextm else " (same month: what most charts show)"))
    def _shade(v):
        if pd.isna(v):
            return ""
        a_ = min(abs(v) / 0.15, 1) * 0.45
        return f"background-color: rgba(42,120,214,{a_:.2f})" if v > 0 else f"background-color: rgba(227,73,72,{a_:.2f})"

    st.dataframe(piv.style.format("{:.1%}").map(_shade), width="stretch")
    counts = tab.drop_duplicates("regime").set_index("regime")["months"]
    st.caption("Months per regime: " + ", ".join(f"{r_} {int(counts.get(r_, 0))}" for r_ in "ABCD") + ".")

    st.subheader("The 2025 spreadsheet's rotation, honestly")
    st.markdown("Factor ETFs (S&P 500 value IVE, growth IVW, quality SPHQ, the S&P 500 SPY, small caps IWM). In each regime, weights proportional to each ETF's average "
                "return in that regime: computed on the **full sample** (the spreadsheet) or only on **months before** each decision (expanding).")
    px5 = MONTHLY[EXCEL_ETFS].dropna()
    r5 = px5.pct_change(fill_method=None).dropna()
    reg5 = rg.threshold_regimes(KNOWN["IP_YOY"], KNOWN["DGS2"], "rolling", 37).reindex(r5.index)
    out = {}
    for md in ("full", "expanding"):
        tg = rg.regime_rotation_targets(r5, reg5, md)
        out[md] = signal_backtest(px5, tg, lag=1, cost_bps=10, name=md).nav
    spy = px5["SPY"].loc[out["expanding"].index[0]:]
    ew = signal_backtest(px5, pd.DataFrame(0.2, index=r5.index, columns=EXCEL_ETFS).loc[out["expanding"].index[0]:], lag=0, cost_bps=10).nav
    navs = {"Rotation, full-sample averages (spreadsheet)": out["full"].loc[out["expanding"].index[0]:], "Rotation, expanding averages (honest)": out["expanding"],
            "S&P 500 (SPY)": 100 * spy / spy.iloc[0], "Equal weight of the 5": ew}
    navs = {k_: 100 * v / v.iloc[0] for k_, v in navs.items()}
    rf = data.risk_free_rate(spy.index[0], spy.index[-1], data_dir=DATA_FOLDERS[st.session_state.rg_dataset])
    st.dataframe(pd.DataFrame({k_: m.summary(v, None, rf)[["start", "cagr", "vol", "sharpe", "max_dd"]] for k_, v in navs.items()}).T
                 .style.format({"cagr": "{:.1%}", "vol": "{:.1%}", "sharpe": "{:.2f}", "max_dd": "{:.1%}"}), width="stretch")
    st.caption("10 bps one-way; weights decided at month-end t, held during t+1; regime from IP growth and the 2-year yield vs their 37-month averages, publication lag applied.")

# --------------------------------------------------------------------------- 4 macro filter
with tabs[3]:
    st.markdown("**Does a macro filter improve your trend rule?** Take a Session 4 rule; when the filter says 'risk off' at a month-end, the risky part of the next "
                "month's portfolio goes to T-bills. Compare with and without, before and after 2014, honestly lagged or not.")
    a, b, c, d = st.columns(4)
    rule = a.selectbox("Rule", ["gem", "faber", "top3", "bh"], key="rg_rule",
                       format_func={"gem": "GEM (SPY/EFA/AGG)", "faber": "Faber 10-month on SPY", "top3": "Top-3 of 8 (13612W)", "bh": "Buy and hold SPY"}.get)
    kind = b.selectbox("Filter", ["curve", "credit", "growth", "payrolls", "payrolls_first", "sahm"], key="rg_kind",
                       format_func={"curve": "Yield curve: risk off when 10y − 3m < 0", "credit": "Credit: risk off when Baa spread above its 12m average",
                                    "growth": "Growth (industrial production, revised): risk off when below its 12m average",
                                    "payrolls": "Growth (payrolls, revised): risk off when below its 12m average",
                                    "payrolls_first": "Growth (payrolls as FIRST PUBLISHED, ALFRED): same rule, real-time data",
                                    "sahm": "Sahm rule: risk off when ≥ 0.5"}.get)
    fmode = c.radio("Average", ["rolling", "full"], key="rg_fmode", format_func={"rolling": "Rolling 12 months", "full": "Full sample (look-ahead)"}.get)
    flag_lag = d.toggle("Respect publication lags", True, key="rg_flag_lag")
    P8 = MONTHLY[list(dict.fromkeys(U8))]
    if rule == "gem":
        base_t = sg.gem_weights(MONTHLY, "SPY", "EFA", "AGG", CASH)
        risky = ["SPY", "EFA"]
    elif rule == "faber":
        base_t = sg.sma_weights(MONTHLY[["SPY"]], 10)
        risky = ["SPY"]
    elif rule == "top3":
        base_t = sg.dual_momentum_weights(P8, "13612W", 3, "cash", CASH, "AGG").loc["2006-02-28":]
        risky = [x for x in U8 if x not in ("AGG", "TLT")]
    else:
        base_t = pd.DataFrame({"SPY": 1.0}, index=MONTHLY["SPY"].dropna().index)
        risky = ["SPY"]
    src = KNOWN if flag_lag else PANEL
    on = rg.risk_on_filter(src, kind, fmode, 12)
    filt_t = rg.apply_filter(base_t, on, risky)
    base_t = base_t.copy()
    if "CASH" not in base_t.columns:
        base_t["CASH"] = 0.0
    start_ = max(base_t.index[0], on.dropna().index[0])
    res_b = signal_backtest(MONTHLY, base_t.loc[start_:], lag=1, cost_bps=10, cash_returns=CASH, name="Rule")
    res_f = signal_backtest(MONTHLY, filt_t.loc[start_:], lag=1, cost_bps=10, cash_returns=CASH, name="Rule + filter")
    rf = data.risk_free_rate(res_b.nav.index[0], res_b.nav.index[-1], data_dir=DATA_FOLDERS[st.session_state.rg_dataset])
    tbl = {}
    for nm, nv in (("Rule", res_b.nav), ("Rule + filter", res_f.nav)):
        for lab_, seg in (("all", nv), ("to 2013", nv.loc[:"2013-12-31"]), ("2014 on", nv.loc["2013-12-31":])):
            if len(seg) > 24:
                s_ = m.summary(100 * seg / seg.iloc[0], None, rf)
                tbl[(nm, lab_)] = s_[["start", "end", "cagr", "vol", "sharpe", "max_dd"]]
    st.dataframe(pd.DataFrame(tbl).T.style.format({"cagr": "{:.1%}", "vol": "{:.1%}", "sharpe": "{:.2f}", "max_dd": "{:.1%}"}), width="stretch")
    off_share = float((on.reindex(filt_t.index) == 0).mean())
    st.caption(f"Risk off in {off_share:.0%} of months. 10 bps one-way. A filter that helps only with full-sample averages or without publication lags is look-ahead, not insight.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=res_b.nav.index, y=res_b.nav, name="Rule", line=dict(color=ui.BENCHMARK_COLOR, dash="dot")))
    fig.add_trace(go.Scatter(x=res_f.nav.index, y=res_f.nav, name="Rule + filter", line=dict(color=ui.SERIES_COLORS[0], width=2.5)))
    fig.update_layout(yaxis_type="log", height=360, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- 5 weather report
with tabs[4]:
    st.markdown("**How does my portfolio behave in each regime and crisis?** Enter an allocation; it is rebalanced monthly. Macro predicts *risk* far better than "
                "return, so this is the regime tool to use every year: which weather hurts you most?")
    universe = st.radio("Universe", ["long", "etf"], horizontal=True, key="rg_wuni",
                        format_func={"long": "Long history (1954–): US stocks, 10y Treasuries, T-bills", "etf": "ETFs (2006–): the Session 4 universe"}.get)
    if universe == "long":
        cols = st.columns(3)
        w = {a_: cols[i].number_input(f"{a_} %", 0, 100, [60, 40, 0][i], 5, key=f"rg_w{i}") / 100 for i, a_ in enumerate(LONG_ASSETS)}
        rets_ = RETS[LONG_ASSETS]
    else:
        cols = st.columns(4)
        defaults = {"SPY": 30, "EFA": 15, "EEM": 5, "AGG": 20, "TLT": 10, "GLD": 10, "DBC": 5, "VNQ": 5}
        w = {a_: cols[i % 4].number_input(f"{a_} %", 0, 100, defaults[a_], 5, key=f"rg_we{a_}") / 100 for i, a_ in enumerate(U8)}
        rets_ = MONTHLY[U8].pct_change(fill_method=None)
    tot = sum(w.values())
    if abs(tot - 1) > 1e-6:
        st.warning(f"Weights add up to {tot:.0%}; they are rescaled to 100%.")
    if tot <= 0:
        st.stop()
    wv = pd.Series(w) / tot
    pr = (rets_[wv.index].dropna() * wv).sum(axis=1)
    reg_w = rg.threshold_regimes(KNOWN["IP_YOY"], KNOWN["CPI_YOY"], "rolling", 36)
    sb = rg.stock_bond_correlation(RETS["US stocks"], RETS["US 10y Treasuries (approx.)"])
    corr_reg = pd.Series(np.where(sb > 0, "Stocks and bonds move together (corr > 0)", "Bonds hedge stocks (corr ≤ 0)"), index=sb.index).where(sb.notna())
    wr = rg.weather_report(pr, reg_w, corr_reg)
    a, b = st.columns(2)
    with a:
        t1 = wr["by_regime"].copy()
        t1.index = [f"{i} · {QUAD[i]}" for i in t1.index]
        st.markdown("**By growth/inflation regime** (known at t, return of t+1)")
        st.dataframe(t1.style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}", "hit_rate": "{:.0%}"}), width="stretch")
        st.markdown("**By stock–bond correlation regime** (36 months, known at t)")
        st.dataframe(wr["by_correlation"].style.format({"ann_return": "{:.1%}", "ann_vol": "{:.1%}", "hit_rate": "{:.0%}"}), width="stretch")
    with b:
        st.markdown("**Crises** (peak to trough of the S&P 500, month-end approximation)")
        st.dataframe(wr["episodes"].style.format("{:.1%}", na_rep="not covered"), width="stretch")
        nav = 100 * (1 + pr).cumprod()
        st.caption(f"Whole period {pr.index[0]:%b %Y} – {pr.index[-1]:%b %Y}: CAGR {m.cagr(nav):.1%}, volatility {m.annualized_vol(nav):.1%}, max drawdown {m.max_drawdown(nav):.1%}.")
    fig = go.Figure(go.Scatter(x=sb.index, y=sb, line=dict(color=ui.SERIES_COLORS[0])))
    fig.add_hline(y=0, line_color=ui.BENCHMARK_COLOR)
    fig.update_layout(title="Stock–bond correlation, rolling 36 months (US stocks vs 10y Treasuries)", height=280, margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- 6 real time vs revised
with tabs[5]:
    st.markdown("**What did investors actually know?** US payrolls as first published (ALFRED) against today's revised numbers. Every backtest on today's FRED data "
                "uses the right-hand line; every investor at the time saw the left-hand one.")
    p = PAY.loc["2015":]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p.index, y=p["first_release"] / 1000, name="As first published", line=dict(color=ui.BENCHMARK_COLOR, dash="dot")))
    fig.add_trace(go.Scatter(x=p.index, y=p["latest"] / 1000, name="As known today", line=dict(color=ui.SERIES_COLORS[0])))
    fig.update_layout(height=320, yaxis_title="millions of jobs", margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    rv = PAY["revision"].loc["2023":]
    st.caption(f"Largest downward revision since 2023: {rv.min():,.0f} thousand jobs ({rv.idxmin():%b %Y}). Latest month: {PAY['revision'].iloc[-1]:,.0f} thousand.")
    st.dataframe(PAY.loc["2023":, ["first_release", "latest", "revision"]].iloc[::3].style.format("{:,.0f}"), width="stretch")

# --------------------------------------------------------------------------- build your own
with tabs[6]:
    st.markdown("### Build your own regime monitor")
    st.markdown("Paste into your AI assistant. The data comes from FRED's public CSV links (no key needed for the latest vintage).")
    st.code("""You are my macro analyst. Build a single-file HTML page (plain JavaScript, Plotly from a CDN) called "My Regime Monitor".

DATA. Fetch these FRED series as CSV from https://fred.stlouisfed.org/graph/fredgraph.csv?id=SERIES (if the browser blocks it, give me a file picker for the downloaded CSVs): INDPRO (industrial production), CPIAUCSL (consumer prices), UNRATE (unemployment), GS10 and TB3MS (10-year and 3-month rates), BAA (Baa corporate yield), USREC (NBER recessions). Put every series at month-end.

REAL TIME. Industrial production, CPI and unemployment for month t are published during month t+1: shift them one month before using them. Never compute an average, a threshold or a z-score with data after the date it is used for.

READINGS. Growth = INDPRO % change over 12 months; inflation = CPIAUCSL % change over 12 months; labour = UNRATE minus its value 12 months earlier; curve = GS10 − TB3MS; credit = BAA − GS10.

MONITOR. Standardize the five readings (z-scores on all history up to the latest month), find the 10 past months most similar to the latest one (Euclidean distance), at least 6 months apart and at least 12 months before today, and list them with their readings.

SCORECARD. For the curve (< 0) and the Sahm rule (3-month average unemployment ≥ 0.5 above its 12-month low), list every trigger since 1960, whether a recession (USREC) started within 24 months, and the lead time.

Show a 'what could be wrong here' box listing every assumption (publication lags, revisions: FRED gives today's revised data, not what was known then).""", language="text")
    st.markdown("**Audit prompt:** *\"Cut the data at December 2007, re-run the monitor for December 2007, and show me that the result uses no data after that date. "
                "Which of your series were revised after 2007, and how would you get the version published at the time?\"* (Answer: ALFRED.)")

with tabs[7]:
    st.markdown(f"""
**Macro data** (FRED, latest vintage unless stated): industrial production INDPRO, CPI CPIAUCSL, unemployment UNRATE, payrolls PAYEMS, Sahm rule real time
SAHMREALTIME, NBER recessions USREC, 10y−2y T10Y2Y, 10y−3m T10Y3M, 2-year DGS2, 10-year GS10, 3-month TB3MS, Baa BAA. Payrolls as first published: ALFRED
(output type 'initial release'). Each series is dated at the end of the month it describes and then shifted by its publication lag
(IP, CPI, unemployment, payrolls, Sahm: 1 month; market yields: 0).

**Assets.** US stocks = Fama-French market return (MKT−RF + RF) and factors HML, SMB, MOM (Ken French's library); T-bills = RF; 10-year Treasuries approximated from
GS10 as yield/12 − 7.5 × change in yield (a standard duration approximation, not a fund). ETFs from Yahoo Finance adjusted closes, month-ends.

**Regimes.** A–D = growth (IP y/y) and the second variable (CPI y/y, or the 2-year yield as in the 2025 spreadsheet) above or below their average; the
average is rolling (36 months), expanding, or full-sample (look-ahead, kept on purpose). Returns by regime match the regime known at the close of month t with
the return of month t+1 unless the switch says otherwise.

**Analogs.** z-scores on all history (the monitor) or expanding (the track record); Euclidean distance; at least 6 months apart; outcomes must be known.

**Signal scorecard.** A trigger is the first month on after 12 months off; 'recession followed' = an NBER recession started within 24 months; 'late' = the
recession had already started (the NBER dates are known only with hindsight).

Engine `dis_lab` v{ENGINE_VERSION}. {ui.DISCLAIMER}
""")
