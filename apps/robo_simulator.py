"""Robo-Advisor Lab: DIS toolkit #3 (Session 3), the reference robo on the shared engine.

Questionnaire → risk profile → model portfolio → backtest net of fees → goals.
Every rule is visible; use it to audit the robo your AI built.

    py -m streamlit run apps/robo_simulator.py
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
from engine import robo  # noqa: E402
from engine.portfolio import drift_band_portfolio  # noqa: E402

st.set_page_config(page_title="Robo-Advisor Lab · DIS", page_icon="🤖", layout="wide")

INDEXA_TIER = "10-100k€"
EXTRA = ["V60A.DE", "IWDE.L", "4GLD.DE", "EUNA.DE"]  # offered as benchmarks / extra menu
LONG_US = {"VFINX": "IWDA.AS", "VBMFX": "EUNH.DE"}  # long-history stand-ins for the goal simulation
INDEXA_FEE = 0.0043  # Indexa's layers on top of the funds' TER: management 0.338% + custody 0.093% (Nov 2026)
TRAPS = [
    ("Hindsight", "Were the model weights, or the start date, chosen after looking at the same data the backtest uses?"),
    ("Free rebalancing", "Daily (or monthly) resets to target at zero cost. Real rebalancing pays spreads and commissions."),
    ("No fees", "The advisory and custody layers, and the funds' TERs, must come out of the return."),
    ("Wrong currency", "Dollar data for a euro investor: the backtest then contains the EUR/USD move or hides it."),
    ("Mixed day counts", "CAGR on calendar years, volatility on 12 monthly returns × √12; never 252 days on monthly data."),
    ("Tolerance only", "A questionnaire that asks only how you feel about losses gives the same portfolio to investors with very different capacity and need."),
    ("Wrong benchmark", "Compare with the profile you would otherwise be sold (Indexa, a 60/40 fund), not with the S&P 500."),
]


@st.cache_data
def load_all():
    uni = data.load_universe()
    menu = list(robo.ASSETS) + EXTRA + list(LONG_US)
    px = data.load_prices(menu)
    ix = data.load_indexa(INDEXA_TIER, freq=None)
    ix = ix[ix.index.dayofweek < 5]  # the file repeats values on weekends
    ix.columns = [f"Indexa {p}" for p in ix.columns]
    man = data.load_manifest()
    return uni, px, ix, man


UNI, PRICES, INDEXA, MANIFEST = load_all()
LAST_DAY = PRICES.index[-1].date()


def label(t: str) -> str:
    return f"{t} · {robo.ASSETS.get(t, UNI.loc[t, 'name'] if t in UNI.index else t)}"


# --------------------------------------------------------------------------- state

if "profile_override" not in st.session_state:
    st.session_state.profile_override = 0  # 0 = use the questionnaire
    st.session_state.colors = {}

st.title("Robo-Advisor Lab")
st.caption("DIS toolkit #3 · Arc stages: **Rules / Model → Backtest → Implementation** · "
           "every number comes from the shared `dis_lab` engine")
st.markdown("> **The decision:** a robo-advisor offers to manage €100k of your savings for about 0.5% a year. "
            "Do you sign up, build the same thing yourself with two ETFs, or hand it to your bank's mixed fund? "
            "And would you trust the robo you built last night?")

tabs = st.tabs(["1 · Your profile", "2 · Your portfolio", "3 · Backtest", "4 · Goals",
                "What could be wrong here", "How this works"])

# --------------------------------------------------------------------------- 1 profile

with tabs[0]:
    st.markdown("Seven questions, three things measured. Most real questionnaires only measure the first.")
    answers: dict[str, int] = {}
    cols = st.columns(3)
    for j, (dim, title) in enumerate(robo.DIMENSIONS.items()):
        with cols[j]:
            st.subheader(title.split(":")[0])
            st.caption(title.split(":", 1)[1].strip())
            for q in robo.QUESTIONS:
                if q.dimension != dim:
                    continue
                opts = [o[0] for o in q.options]
                default = 1 if q.dimension == "need" else min(2, len(opts) - 1)
                choice = st.radio(q.text, opts, index=default, key=f"q_{q.id}")
                answers[q.id] = opts.index(choice)
    rp = robo.risk_profile(answers)
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Risk tolerance", f"{rp.tolerance:.1f} / 10", help="Average of the three tolerance questions.")
    k2.metric("Risk capacity", f"{rp.capacity:.1f} / 10", help="Average of the three capacity questions.")
    k3.metric("Risk need", f"{rp.need:.0f} / 10", help="The return your goal requires, as a risk level.")
    k4.metric("Your profile", f"{rp.profile} / 10", help="The lower of tolerance and capacity, rounded.")
    (st.warning if rp.need_exceeds_profile else st.info)(rp.note)
    st.markdown("**Rule used:** profile = round(min(tolerance, capacity)), clamped to 1–10. Need never raises "
                "the profile. Compare with the 2025 Excel exercise, whose score mixed age, horizon and three "
                "1–5 scales with ad hoc weights and was not clamped: a 25-year-old with 'very stable income' "
                "could land outside 1–10.")
    st.selectbox("Override the profile (to test any row of the table)", list(range(0, 11)),
                 key="profile_override", format_func=lambda p: "Use the questionnaire" if p == 0 else f"Profile {p}")

profile = st.session_state.profile_override or rp.profile

# --------------------------------------------------------------------------- 2 portfolio

with tabs[1]:
    st.markdown(f"**Model portfolio for profile {profile}.** A robo does not optimise per client: it keeps one "
                "table and maps you to a row. Edit the weights below to build your own row (they are scaled to 100%).")
    base = robo.MODEL_PORTFOLIOS.copy()
    show = (base * 100).round(1)
    show.columns = [label(c) for c in show.columns]
    st.dataframe(show.style.format("{:.1f}%").apply(
        lambda r: ["background-color: rgba(42,120,214,0.18)" if r.name == profile else "" for _ in r], axis=1),
        width="stretch")
    st.caption("Equity = 10% × profile (95% for profile 10), split 85/15 world / emerging markets. Bonds split 50/50 "
               "euro government / global hedged to euro. Profiles 1–2 keep a cash sleeve. Indexa's profile 6 is "
               "also 60/40; its bonds are more diversified (inflation-linked, US government and corporate, hedged).")
    row = pd.DataFrame({"Ticker": list(robo.ASSETS), "What it is": list(robo.ASSETS.values()),
                        "Weight %": [round(float(base.loc[profile, t]) * 100, 1) for t in robo.ASSETS]})
    edited = st.data_editor(row, key=f"weights_editor_{profile}", hide_index=True, width="stretch",
                            disabled=["Ticker", "What it is"],
                            column_config={"Weight %": st.column_config.NumberColumn(min_value=0.0, max_value=100.0,
                                                                                     step=5.0, format="%.1f")})
    w = edited.set_index("Ticker")["Weight %"].astype(float)
    w = w[w > 0]
    if w.sum() <= 0:
        st.warning("Put some weight somewhere.")
        st.stop()
    weights = (w / w.sum()).to_dict()
    eq = robo.equity_share(weights)
    st.markdown(f"**Equity share: {eq:.0%}.** " + ("Edited row." if abs(w.sum() - 100) > 0.05 or
                not np.allclose(w.reindex(robo.ASSETS).fillna(0) / 100, base.loc[profile].to_numpy(), atol=1e-3)
                else "Standard row."))
    fig = go.Figure(go.Bar(x=[robo.ASSETS[t] for t in weights], y=list(weights.values()),
                           marker_color=ui.SERIES_COLORS[: len(weights)], hovertemplate="%{y:.0%}"))
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), yaxis_tickformat=".0%", showlegend=False)
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------------------------------- 3 backtest

with tabs[2]:
    c1, c2, c3, c4 = st.columns(4)
    fee_pct = c1.number_input("Robo fee layer (% per year)", 0.0, 3.0, INDEXA_FEE * 100, 0.05, key="fee",
                              help="Advisory + custody, on top of the funds' own TER (already in the prices). "
                                   "Indexa's average is 0.34% + 0.09%; a bank's profiled fund is often 1.5–1.9% all-in.")
    rebalance = c2.selectbox("Rebalancing", list(robo.REBALANCE_CHOICES), index=4, key="rebalance",
                             format_func=lambda v: {"band": "drift band (checked monthly)"}.get(v, v),
                             help="How the weights are reset: by calendar, never (buy and hold), or only when a "
                                  "weight drifts beyond the band, which is what most robos do.")
    band_pts = c3.number_input("Band (weight points)", 1.0, 25.0, 5.0, 1.0, key="band",
                               disabled=rebalance != "band")
    cost_bps = c4.number_input("Trading cost (bps per trade)", 0.0, 100.0, 10.0, 5.0, key="cost_bps",
                               help="Spread + commission + slippage on what is traded. Index funds through "
                                    "Indexa trade at NAV with no spread; ETFs through a broker do not.")
    c5, c6, c7 = st.columns(3)
    start = c5.date_input("From", dt.date(2015, 12, 31), min_value=dt.date(2009, 1, 1), max_value=LAST_DAY, key="start")
    end = c6.date_input("To", LAST_DAY, min_value=dt.date(2009, 1, 1), max_value=LAST_DAY, key="end")
    bench_opts = [f"Indexa {profile}"] + [f"Indexa {p}" for p in range(1, 11) if p != profile] + ["V60A.DE", "IWDE.L"]
    bench_name = c7.selectbox("Benchmark", bench_opts, key="bench",
                              help="Indexa's published return for the same profile number (net of its fees, "
                                   "10–100k€ accounts), Vanguard LifeStrategy 60 (V60A.DE) or hedged world equity.")
    excel_way = st.toggle("Also show 'the Excel way': daily rebalancing, no costs, no fee (find the flaw)",
                          value=False, key="excel_way")

    px = PRICES[list(weights)].loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
    if len(px) < 60:
        st.error("Fewer than 60 common trading days: widen the window.")
        st.stop()
    res = robo.robo_backtest(px, weights, fee=fee_pct / 100, rebalance=rebalance, band=band_pts / 100,
                             cost_bps=cost_bps, name="Your robo")
    series = {"Your robo": res.nav, "Your robo, before fees": res.gross.nav.rename("Your robo, before fees")}
    if excel_way:
        series["The Excel way"] = drift_band_portfolio(px, weights, band=0.0, check="daily", cost_bps=0.0,
                                                       name="The Excel way").nav
    if bench_name.startswith("Indexa"):
        bseries = INDEXA[bench_name]
    else:
        bseries = PRICES[bench_name]
    series[bench_name] = bseries.loc[pd.Timestamp(start):pd.Timestamp(end)].dropna()
    daily = m.align(*[s.rename(k) for k, s in series.items()])
    levels = m.to_frequency(daily, "monthly")
    if len(levels) < 24:
        st.error("Fewer than 24 months in common with the benchmark: widen the window or change it.")
        st.stop()
    w_start, w_end = levels.index[0], levels.index[-1]
    years = m.years_between(w_start, w_end)
    try:
        rf = data.risk_free_rate(w_start, w_end, data.EURO_RF_SERIES)
        rf_src = "euro 3-month rate"
    except ValueError:
        rf, rf_src = 0.0, "none"
    st.markdown(f"**Common window:** {w_start:%d %b %Y} → {w_end:%d %b %Y} ({years:.1f} years) · monthly returns · "
                f"EUR · risk-free {rf:.2%} ({rf_src})")
    if bench_name.startswith("Indexa") and w_end < pd.Timestamp(end) - pd.Timedelta(days=45):
        st.info(f"Indexa's published series ends on {INDEXA.index[-1]:%d %b %Y}, so the comparison stops there.")

    table = m.summary_table(levels, benchmark=bench_name, rf=rf)
    keys = ["cagr", "vol", "sharpe", "max_dd", "tuw_days", "worst_year", "pct_pos_years", "correlation", "tracking_error"]
    keys = [k for k in keys if k in table.index]
    disp = pd.DataFrame({c: [ui.format_metric(k, table.loc[k, c]) for k in keys] for c in table.columns},
                        index=[ui.GLOSSARY[k][0] for k in keys])
    st.dataframe(disp, width="stretch")
    colors = ui.assign_colors([c for c in levels.columns if c != bench_name], st.session_state.colors)

    def lines(df, yfmt, hover, title, height=380):
        fig = go.Figure()
        for c in df.columns:
            is_b = c == bench_name
            fig.add_trace(go.Scatter(x=df.index, y=df[c], mode="lines", name=f"{c} (benchmark)" if is_b else c,
                                     line=dict(width=2, color=ui.BENCHMARK_COLOR if is_b else colors[c],
                                               dash="dash" if is_b else "solid"),
                                     hovertemplate=f"%{{y:{hover}}}"))
        fig.update_layout(height=height, hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                          title=dict(text=title, x=0, font=dict(size=15)),
                          legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0))
        fig.update_yaxes(tickformat=yfmt, gridcolor=ui.GRID_COLOR, zeroline=False)
        fig.update_xaxes(showgrid=False)
        return fig

    growth = levels / levels.iloc[0] * 100_000
    st.plotly_chart(lines(growth, ",.0f", ",.0f", "Growth of €100,000"), width="stretch")
    dd = pd.DataFrame({c: m.drawdown_series(levels[c]) for c in levels.columns})
    st.plotly_chart(lines(dd, ".0%", ".1%", "Drawdown: % below the previous high", 300), width="stretch")

    fee_cost = float(growth["Your robo, before fees"].iloc[-1] - growth["Your robo"].iloc[-1])
    n_reb = len(res.gross.turnover)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Fee layer over the window", f"€{fee_cost:,.0f}", help="What the robo's own fees took from €100,000, "
              "before counting what they bought you (discipline, no spreads, tax-free switches).")
    k2.metric("Rebalances", f"{n_reb}", help="Number of dates on which the portfolio was reset to target.")
    k3.metric("Turnover per year", f"{res.gross.annual_turnover:.1%}", help="One-way share of the portfolio traded per year.")
    k4.metric("Trading cost per year", f"{res.gross.annual_cost:.2%}", help="At the cost per trade you set.")
    if excel_way:
        gap = m.cagr(levels["The Excel way"]) - m.cagr(levels["Your robo"])
        st.warning(f"**The Excel way** (daily rebalancing at zero cost, no fee) shows a CAGR {gap:+.2%} higher than "
                   "your robo. That gap is not return: it is costs and fees that the spreadsheet never charged. "
                   "Most AI-built robos start this way.")
    with st.expander("Calendar years"):
        yr = pd.DataFrame({c: m.calendar_year_returns(levels[c])["return"] for c in levels.columns})
        st.dataframe(yr.map(ui.pct), width="stretch")
    d1, d2 = st.columns(2)
    d1.download_button("Download metrics (CSV)", table.to_csv().encode("utf-8"), "robo_metrics.csv", "text/csv")
    d2.download_button("Download the series (CSV)", levels.to_csv().encode("utf-8"), "robo_series.csv", "text/csv")

# --------------------------------------------------------------------------- 4 goals

with tabs[3]:
    st.markdown("**Profile-based** puts all your money in one portfolio set by your risk profile. **Goal-based** "
                "gives each goal its own bucket, with risk set by that goal's horizon. Same money, same "
                "contributions, same fee, same bootstrapped history for both. Which one reaches your goals more often?")
    g1, g2, g3 = st.columns(3)
    history = g1.radio("History to resample", ["Euro ETFs (2009–2026)", "Long US history (1987–2026)"], key="history",
                       help="Blocks of 12 months are drawn at random from the chosen history. The euro menu is short "
                            "and mostly a bull market; the US series are longer (S&P 500 and US bonds, in dollars) "
                            "and stand in for world equity and euro bonds.")
    sim_fee = g2.number_input("Fee on both approaches (% per year)", 0.0, 3.0, 0.5, 0.1, key="sim_fee")
    n_sims = int(g3.selectbox("Simulations", [500, 2000, 5000], index=1, key="n_sims"))
    default_goals = pd.DataFrame({
        "Goal": ["House deposit", "Retirement"],
        "Money set aside today (€)": [62_000.0, 38_000.0],
        "Target at the horizon (€)": [60_000.0, 500_000.0],
        "Years": [3.0, 30.0],
        "Bucket profile (0–10)": [0, 9],
        "Monthly contribution (€)": [0.0, 300.0],
    })
    goals_tbl = st.data_editor(default_goals, key="goals_editor", num_rows="dynamic", hide_index=True, width="stretch",
                               column_config={"Bucket profile (0–10)": st.column_config.NumberColumn(
                                   min_value=0, max_value=10, step=1, help="0 = all cash; 1–10 = a row of the model table")})
    goals_tbl = goals_tbl.dropna()
    if goals_tbl.empty:
        st.info("Add at least one goal.")
        st.stop()
    menu = ["IWDA.AS", "EUNH.DE", "XEON.DE"]  # the three-asset menu both histories can provide
    if history.startswith("Euro"):
        mr = robo.monthly_returns_from_prices(PRICES[menu])
        hist_note = f"euro ETFs, {mr.index[0]:%b %Y} to {mr.index[-1]:%b %Y}"
    else:
        us = robo.monthly_returns_from_prices(PRICES[list(LONG_US)]).rename(columns=LONG_US)
        us["XEON.DE"] = 0.0  # cash at zero: conservative stand-in
        mr = us[menu]
        hist_note = f"US stocks and bonds in dollars, {mr.index[0]:%b %Y} to {mr.index[-1]:%b %Y}; cash at 0%"

    def bucket(p: int) -> dict[str, float]:
        """Three-asset version of the model row: equity → IWDA, bonds → EUNH, cash → XEON. 0 = all cash."""
        if int(p) == 0:
            return {"IWDA.AS": 0.0, "EUNH.DE": 0.0, "XEON.DE": 1.0}
        row = robo.MODEL_PORTFOLIOS.loc[int(p)]
        return {"IWDA.AS": float(row["IWDA.AS"] + row["IS3N.DE"]),
                "EUNH.DE": float(row["EUNH.DE"] + row["XG7S.DE"]), "XEON.DE": float(row["XEON.DE"])}

    goals = [robo.Goal(str(r["Goal"]), float(r["Money set aside today (€)"]), float(r["Target at the horizon (€)"]),
                       float(r["Years"]), bucket(int(r["Bucket profile (0–10)"])), float(r["Monthly contribution (€)"]))
             for _, r in goals_tbl.iterrows()]
    prof_w = bucket(profile)
    sim = robo.goal_simulation(mr, goals, prof_w, n_sims=n_sims, fee=sim_fee / 100, seed=7)
    t = sim.table.copy()
    out = pd.DataFrame({
        "Chance the goal is met": t["p_met"].map(lambda v: f"{v:.0%}"),
        "Median wealth at the horizon": t["median_wealth"].map(lambda v: f"€{v:,.0f}"),
        "Bad case (5th percentile)": t["p5_wealth"].map(lambda v: f"€{v:,.0f}"),
        "Target": t["target"].map(lambda v: f"€{v:,.0f}"),
    })
    st.dataframe(out, width="stretch")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("All goals met: goal-based", f"{sim.p_all_met['Goal-based']:.0%}")
    k2.metric(f"All goals met: profile {profile} for everything", f"{sim.p_all_met['Profile-based']:.0%}")
    k3.metric("Equity at the start: goal-based", f"{sim.equity_at_start['Goal-based']:.0%}")
    k4.metric("Equity at the start: profile-based", f"{sim.equity_at_start['Profile-based']:.0%}")
    st.caption(f"{sim.n_sims:,} paths of {sim.months} months, blocks of {sim.block} months resampled from {hist_note}. "
               "In the profile-based run each goal is paid out of the single pot when it comes due, so an early goal "
               "can be 'met' with money that was meant for a later one. Nominal euros; no taxes.")
    agg = robo.aggregate_weights(goals)
    st.markdown(f"**Like for like:** the goal-based buckets add up to {robo.equity_share(agg):.0%} equity overall. "
                f"Set the profile override to the row with that equity share (tab 1) to compare at equal total risk.")
    st.markdown("**What the theory says.** Markowitz's frontier is about the whole pot; goal-based investing "
                "(Shefrin & Statman's behavioural portfolio theory, Chhabra 2005, Das–Markowitz–Scheid–Statman 2010) "
                "builds one sub-portfolio per goal with a maximum probability of missing it. Das et al. show the sum "
                "of the goal portfolios is close to mean-variance efficient: the loss in efficiency is small, and what "
                "you gain is risk placed where the horizon can absorb it, and goals you can actually track.")

# --------------------------------------------------------------------------- 5 checks

with tabs[4]:
    st.markdown("The **seven traps** of an AI-built robo, and whether this lab avoids them.")
    for name, text in TRAPS:
        st.markdown(f"- **{name}.** {text}")
    st.divider()
    st.markdown("**Automatic checks on the backtest above** (engine bias checklist):")
    inception = {c: PRICES[c].first_valid_index() for c in weights}
    inception[bench_name] = bseries.first_valid_index()
    checks = chk.performance_checks(levels, benchmark=bench_name, inception=inception,
                                    listing_currency={c: "EUR" for c in levels.columns}, display_currency="EUR",
                                    freq="monthly", rf=rf,
                                    portfolio={"name": "Your robo", "rebalance": rebalance, "cost_bps": cost_bps})
    for c in checks:
        (st.warning if c.severity == "warn" else st.info)(
            f"**{c.title}**  \n{c.detail}  \n*Checklist {c.item}: {chk.CHECKLIST[c.item]}*")
    st.markdown("**This lab's own weak spots:** the model table was designed in 2026, knowing how these assets "
                "did (hindsight in the weights); the backtest starts in Dec 2015 because Indexa's record does; the "
                "goal simulation resamples a short, mostly bullish history; nothing here is net of taxes.")

# --------------------------------------------------------------------------- 6 how

with tabs[5]:
    st.markdown(f"""
**What this tool does, in plain English**
1. Scores your answers in three groups (tolerance, capacity, need) and sets your profile to the lower of
   tolerance and capacity. Need can lower the ambition of a goal, never raise the risk.
2. Maps the profile to a row of a fixed table of UCITS ETFs in euros, the way robos do (no per-client optimiser).
3. Backtests that row on the data snapshot: holds the weights, lets them drift, resets them by calendar or when a
   weight leaves its band, pays the trading cost on what it trades, and then charges the fee layer every day
   (calendar-day accrual). Fund TERs are already inside the prices.
4. Compares it with Indexa Capital's published return for the same profile number (10–100k€ accounts, net of
   Indexa's fees), with the engine's metric definitions (CAGR by calendar days; volatility = std of monthly
   returns × √12; Sharpe = (CAGR − rf) / vol; drawdowns on the value path).
5. Runs your goals two ways on block-bootstrapped history and counts how often each goal is reached.

**Data:** Yahoo Finance adjusted closes (snapshot {MANIFEST['downloaded'].max()}; prices to {LAST_DAY:%d %b %Y}),
FRED (euro 3-month rate), Indexa Capital's published series (Dec 2015 to Jan 2025). Engine `dis_lab` v{ENGINE_VERSION}.

**Ask your AI analyst.** Download the metrics CSV from tab 3, attach it, and paste:
""")
    st.code(
        "You are the sceptical member of my investment committee. Attached are the backtest metrics of a robo-advisor "
        "model portfolio and of a real robo-advisor's published profile over the same window. 1) In plain English, what "
        "did each deliver, and what was the worst experience a client would have had? 2) Using the seven traps "
        "(hindsight, free rebalancing, no fees, wrong currency, mixed day counts, tolerance-only questionnaire, wrong "
        "benchmark), which ones could still affect these numbers? 3) What one change to the robo would you insist on "
        "before putting €100k in it? Do not recommend a product.",
        language=None, wrap_lines=True,
    )
    st.caption(ui.DISCLAIMER)
