"""Interactive dashboard.  Run:  uv run streamlit run dashboard/app.py"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from vrp_alpha.config import DATA_PROCESSED

RES = DATA_PROCESSED / "results"
MARKETS = {"US": "US · S&P 500", "EU": "Europe · Euro Stoxx 50", "UK": "UK · FTSE 100", "IN": "India · NIFTY 50"}
SOURCE = {
    "US": "Synthetic surface from VIX term structure, calibrated to CBOE PUT/BXM/PPUT/BXMD",
    "EU": "Synthetic: US surface scaled by trailing realised-vol ratio (VSTOXX history not free)",
    "UK": "Synthetic: US surface scaled by trailing realised-vol ratio (FTSE IVI history not free)",
    "IN": "Real NSE NIFTY option settlement prices (F&O bhavcopy)",
}
BLUE, ORANGE, AQUA, YELLOW, INK2, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#52514e", "#8a8984"
STRAT_STYLE = {
    "GDE Optimised": dict(color=BLUE, width=2.4),
    "GDE 25-delta": dict(color=ORANGE, width=1.6),
    "GDE ATM": dict(color=AQUA, width=1.6),
    "50/50 Equity/Cash": dict(color=INK2, width=1.3, dash="dot"),
    "Equity (TR)": dict(color=MUTED, width=1.3, dash="dash"),
}

st.set_page_config(page_title="Option Selling Strategy Optimiser", layout="wide")


@st.cache_data
def load():
    return {
        "nav": pd.read_parquet(RES / "nav.parquet"),
        "trades": pd.read_parquet(RES / "trades.parquet"),
        "perf": pd.read_csv(RES / "performance.csv"),
        "episodes": pd.read_csv(RES / "episodes.csv"),
        "stress": pd.read_csv(RES / "stress.csv"),
        "greeks": pd.read_parquet(RES / "greeks.parquet"),
        "sens": pd.read_csv(RES / "sensitivity.csv"),
        "vrp": pd.read_parquet(DATA_PROCESSED / "vrp.parquet"),
        "by_delta": pd.read_csv(DATA_PROCESSED / "prob_by_delta.csv"),
        "scores": pd.read_csv(DATA_PROCESSED / "prob_scores.csv", index_col=0),
        "calib": pd.read_parquet(DATA_PROCESSED / "calibration_series.parquet"),
    }


@st.cache_resource
def live_engine(market: str):
    from vrp_alpha.analytics import vrp
    from vrp_alpha.models.probability import make_fhs
    from vrp_alpha.pricing.surface import get_market

    tables = vrp.load()
    return get_market(market), make_fhs(tables), tables[market]


def _layout(fig, title="", height=380, yfmt=None):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14)), height=height, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.15),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", tickformat=yfmt)
    return fig


D = load()
st.title("Option Selling Strategy Optimiser: picking strikes least likely to be exercised")
st.caption("Put & covered-call writing backtested on US, European, UK and Indian indices (2020–2025)")
market = st.radio("Market", list(MARKETS), format_func=MARKETS.get, horizontal=True)
st.info(f"**Option data:** {SOURCE[market]}", icon="ℹ️")

tabs = st.tabs(["Overview", "VRP monitor", "Exercise probability", "Strike scanner", "Backtest", "Risk & stress"])

perf = D["perf"][D["perf"]["market"] == market].set_index("strategy")

# ---------------------------------------------------------------- overview
with tabs[0]:
    o, b25, mix = perf.loc["GDE Optimised"], perf.loc["GDE 25-delta"], perf.loc["50/50 Equity/Cash"]
    c = st.columns(4)
    c[0].metric("GDE Optimised · CAGR", f"{o['CAGR']:.1%}", f"{(o['CAGR'] - mix['CAGR']) * 100:+.1f} pts vs 50/50")
    c[1].metric("Sharpe", f"{o['Sharpe']:.2f}", f"{o['Sharpe'] - b25['Sharpe']:+.2f} vs fixed 25Δ")
    c[2].metric("Max drawdown", f"{o['MaxDD']:.1%}", f"{(o['MaxDD'] - b25['MaxDD']) * 100:+.1f} pts vs fixed 25Δ")
    c[3].metric("Puts exercised", f"{o['put_exercise_rate']:.0%}",
                f"{(o['put_exercise_rate'] - b25['put_exercise_rate']) * 100:+.0f} pts vs fixed 25Δ", delta_color="inverse")
    nav = D["nav"][D["nav"]["market"] == market]
    fig = go.Figure()
    for s, style in STRAT_STYLE.items():
        d = nav[nav["strategy"] == s]
        fig.add_scatter(x=d["date"], y=d["nav"], name=s, line=style, hovertemplate="%{y:.3f}")
    fig.update_yaxes(type="log")
    st.plotly_chart(_layout(fig, "Growth of 1 (local currency, log scale)", 420), width="stretch")
    show = perf[["CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Alpha_ann", "Beta", "DownCapture",
                 "put_exercise_rate", "call_exercise_rate", "put_cycles_written", "call_cycles_written"]]
    st.dataframe(show.style.format({k: "{:.1%}" for k in ["CAGR", "Vol", "MaxDD", "Alpha_ann", "put_exercise_rate",
                                                           "call_exercise_rate", "put_cycles_written",
                                                           "call_cycles_written"]} | {k: "{:.2f}" for k in
                                                          ["Sharpe", "Sortino", "Beta", "DownCapture"]}, na_rep="—"),
                 width="stretch")

# ---------------------------------------------------------------- VRP
with tabs[1]:
    v = D["vrp"].xs(market)
    fig = go.Figure()
    fig.add_scatter(x=v.index, y=v["iv30"], name="30d ATM implied vol", line=dict(color=BLUE, width=1.8))
    fig.add_scatter(x=v.index, y=v["ens"], name="Forecast vol (HAR+GARCH)", line=dict(color=AQUA, width=1.4))
    fig.add_scatter(x=v.index, y=v["fwd_rv"], name="Realised vol, next 21d (ex-post)", line=dict(color=ORANGE, width=1.2))
    st.plotly_chart(_layout(fig, "Implied vs forecast vs realised volatility", 400, ".0%"), width="stretch")
    fig = go.Figure()
    fig.add_bar(x=v.index, y=v["vrp_expost"], name="IV − realised", marker_color=BLUE)
    st.plotly_chart(_layout(fig, "Ex-post volatility risk premium (vol points)", 300, ".0%"), width="stretch")
    d = v.dropna(subset=["iv30", "fwd_rv"])
    c = st.columns(3)
    c[0].metric("Average IV − realised", f"{d['vrp_expost'].mean() * 100:.1f} vol pts")
    c[1].metric("Days IV > realised", f"{(d['vrp_expost'] > 0).mean():.0%}")
    c[2].metric("Worst day", f"{d['vrp_expost'].min() * 100:.1f} vol pts", str(d["vrp_expost"].idxmin().date()),
                delta_color="off")

# ---------------------------------------------------------------- probability
with tabs[2]:
    st.markdown("Pooled across all four markets: for options sold at each delta, how often were they "
                "actually exercised versus what the market price implied?")
    bd = D["by_delta"]
    cols = st.columns(2)
    for col, side in zip(cols, ["put", "call"]):
        d = bd[bd["side"] == side]
        fig = go.Figure()
        fig.add_scatter(x=d["target_delta"], y=d["p_rn"], name="Market-implied", mode="lines+markers",
                        line=dict(color=ORANGE, width=2), marker=dict(size=8))
        fig.add_scatter(x=d["target_delta"], y=d["p_fhs"], name="Model (FHS)", mode="lines+markers",
                        line=dict(color=BLUE, width=2), marker=dict(size=8))
        fig.add_scatter(x=d["target_delta"], y=d["realised_itm"], name="Actually exercised", mode="markers",
                        marker=dict(color="#0b0b0b", size=9, symbol="diamond"))
        col.plotly_chart(_layout(fig, f"Short {side}s: P(exercise) by delta", 360, ".0%"), width="stretch")
    st.subheader("Out-of-sample scoring (walk-forward 2021-2025, lower is better)")
    st.dataframe(D["scores"].style.format("{:.4f}").format({"n": "{:,.0f}", "itm_rate": "{:.1%}"}),
                 width="stretch")
    st.caption("The ML layer (regularised logistic regression on the structural probabilities) does not beat FHS "
               "out of sample: five years of overlapping monthly outcomes are too few to learn regime corrections "
               "that survive 2021 → 2022 → 2023. Reported as a negative result.")

# ---------------------------------------------------------------- scanner
with tabs[3]:
    st.markdown("Re-runs the optimiser for any date: premium net of costs versus the model's expected payoff "
                "and tail risk. **Score = (edge − λ·CVaR95) / strike**; eligible if score > 0 and "
                "model P(exercise) ≤ cap.")
    with st.spinner("Loading option market (India loads the full NSE chain the first time)…"):
        mk, fhs, vt = live_engine(market)
    dates = vt.dropna(subset=["ens", "iv30"]).index
    c = st.columns([2, 1])
    d = c[0].select_slider("Date", options=list(dates), value=dates[-1], format_func=lambda x: x.strftime("%Y-%m-%d"))
    lam = c[1].number_input("Risk aversion λ", 0.0, 0.5, 0.02, 0.01)
    from vrp_alpha.optimizer.strikes import score_candidates

    e = mk.next_expiry(d, 21)
    st.caption(f"Expiry {e:%Y-%m-%d} · spot {mk.df.at[d, 'spot']:,.1f} · forecast vol {vt.at[d, 'ens']:.1%} · "
               f"30d IV {vt.at[d, 'iv30']:.1%}")
    for is_call in (False, True):
        sc = score_candidates(mk, fhs, d, e, float(vt.at[d, "ens"]), is_call, lam)
        if sc.empty:
            continue
        sc = sc.assign(edge_pct=sc["edge"] / sc["K"], score_bp=sc["score"] * 1e4)
        pick = sc.loc[sc.loc[sc["eligible"], "score"].idxmax()] if sc["eligible"].any() else None
        st.markdown(f"**{'Calls (overwrite)' if is_call else 'Puts (collateralised)'}** — " +
                    (f"pick strike **{pick['K']:,.0f}** (Δ {abs(pick['delta']):.2f}, model P(exercise) "
                     f"{pick['p_phys']:.1%} vs market {pick['p_rn']:.1%})" if pick is not None else
                     "**no strike worth selling** this cycle"))
        st.dataframe(sc[["target_delta", "K", "iv", "price", "p_rn", "p_phys", "edge_pct", "cvar95", "score_bp",
                         "eligible"]].style.format({"K": "{:,.0f}", "iv": "{:.1%}", "price": "{:,.2f}",
                                                    "p_rn": "{:.1%}", "p_phys": "{:.1%}", "edge_pct": "{:+.2%}",
                                                    "cvar95": "{:,.1f}", "score_bp": "{:+.1f}"}),
                     width="stretch", hide_index=True)

# ---------------------------------------------------------------- backtest
with tabs[4]:
    nav = D["nav"][D["nav"]["market"] == market]
    fig = go.Figure()
    for s, style in STRAT_STYLE.items():
        d = nav[nav["strategy"] == s].set_index("date")["nav"]
        fig.add_scatter(x=d.index, y=d / d.cummax() - 1, name=s, line=style, hovertemplate="%{y:.1%}")
    st.plotly_chart(_layout(fig, "Drawdown", 340, ".0%"), width="stretch")
    strat = st.radio("Trade log", ["GDE Optimised", "GDE 25-delta", "GDE ATM"], horizontal=True)
    tr = D["trades"].query("market == @market and strategy == @strat").copy()
    if "skipped" in tr:
        tr["skipped"] = tr["skipped"].fillna(False).astype(bool)
    st.dataframe(tr.drop(columns=["market", "strategy"]), width="stretch", hide_index=True)
    st.subheader("Robustness to λ (GDE Optimised)")
    st.dataframe(D["sens"][D["sens"]["market"] == market].drop(columns="market"), width="stretch",
                 hide_index=True)

# ---------------------------------------------------------------- risk
with tabs[5]:
    g = D["greeks"][D["greeks"]["market"] == market]
    cols = st.columns(2)
    for col, (field, title, fmt) in zip(cols, [("delta_total", "Portfolio delta (equity-equivalent)", ".2f"),
                                               ("vega", "Vega: NAV change per +1 vol pt", ".2%")]):
        fig = go.Figure()
        for s in ["GDE Optimised", "GDE 25-delta", "GDE ATM"]:
            d = g[g["strategy"] == s]
            fig.add_scatter(x=d["date"], y=d[field], name=s, line=STRAT_STYLE[s])
        col.plotly_chart(_layout(fig, title, 320, fmt), width="stretch")
    stt = D["stress"][D["stress"]["market"] == market]
    fig = go.Figure()
    for s in ["Equity (TR)", "50/50 Equity/Cash", "GDE 25-delta", "GDE Optimised"]:
        d = stt[stt["strategy"] == s]
        fig.add_bar(y=d["scenario"], x=d["total_pnl"], name=s, orientation="h",
                    marker_color=STRAT_STYLE[s]["color"], hovertemplate="%{x:.1%}")
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(_layout(fig, "Scenario stress on latest live book (1-day P&L, % NAV)", 420, None)
                    .update_xaxes(tickformat=".0%"), width="stretch")
    st.subheader("Episodes inside the window")
    ep = D["episodes"][D["episodes"]["market"] == market].drop(columns="market").set_index("episode")
    st.dataframe(ep.style.format("{:.1%}"), width="stretch")
