"""Static figures + markdown results tables for the README.

Usage: uv run python -m vrp_alpha.report
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vrp_alpha.config import DATA_PROCESSED, FIGURES, REPORTS

RES = DATA_PROCESSED / "results"
MARKETS = ["US", "EU", "UK", "IN"]
MARKET_NAMES = {"US": "US · S&P 500", "EU": "Europe · Euro Stoxx 50", "UK": "UK · FTSE 100", "IN": "India · NIFTY 50"}

# palette (validated reference instance, light mode); benchmarks are neutral ink
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
STRAT_STYLE = {
    "GDE Optimised": dict(color=BLUE, lw=2.0, ls="-"),
    "GDE 25-delta": dict(color=ORANGE, lw=1.6, ls="-"),
    "GDE ATM": dict(color=AQUA, lw=1.6, ls="-"),
    "50/50 Equity/Cash": dict(color=INK2, lw=1.3, ls=":"),
    "Equity (TR)": dict(color=MUTED, lw=1.3, ls="--"),
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "legend.fontsize": 9, "text.color": INK,
})


def _save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_calibration() -> None:
    s = pd.read_parquet(DATA_PROCESSED / "calibration_series.parquet")
    names = {"PUT": "PUT · 1m ATM put-write", "BXM": "BXM · 1m ATM buy-write",
             "PPUT": "PPUT · 5% OTM protective put", "BXMD": "BXMD · 30-delta buy-write"}
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for ax, b in zip(axes.flat, names):
        ax.plot((1 + s[("actual", b)]).cumprod(), color=INK2, lw=1.6, ls="--", label="CBOE index (actual)")
        ax.plot((1 + s[("model", b)]).cumprod(), color=BLUE, lw=2.0, label="Replicated on synthetic surface")
        ax.axvline(pd.Timestamp("2023-01-01"), color=MUTED, lw=0.8)
        ax.set_title(names[b], loc="left")
    axes[0, 0].legend(loc="upper left")
    axes[0, 0].text(pd.Timestamp("2023-02-01"), axes[0, 0].get_ylim()[0] * 1.02, "out-of-sample →", color=INK2, fontsize=8)
    fig.suptitle("The synthetic US surface reproduces real CBOE option-strategy returns", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "01_calibration.png")


def fig_vrp() -> None:
    v = pd.read_parquet(DATA_PROCESSED / "vrp.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, MARKETS):
        d = v.xs(m).dropna(subset=["iv30"])
        ax.plot(d.index, d["iv30"] * 100, color=BLUE, lw=1.4, label="30d implied vol (ATM)")
        ax.plot(d.index, d["fwd_rv"] * 100, color=ORANGE, lw=1.1, label="Realised vol over next 21d")
        ax.set_title(MARKET_NAMES[m], loc="left")
        ax.set_ylabel("vol %")
    axes[0, 0].legend(loc="upper right")
    fig.suptitle("Implied volatility usually sits above the volatility that follows", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    _save(fig, "02_vrp.png")


def fig_exercise() -> None:
    b = pd.read_csv(DATA_PROCESSED / "prob_by_delta.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, side in zip(axes, ["put", "call"]):
        d = b[b["side"] == side]
        x = d["target_delta"]
        ax.plot(x, d["p_rn"] * 100, color=ORANGE, lw=2, marker="o", ms=5, label="Market-implied P(exercise)")
        ax.plot(x, d["p_fhs"] * 100, color=BLUE, lw=2, marker="o", ms=5, label="Model P(exercise) · FHS")
        ax.plot(x, d["realised_itm"] * 100, color=INK, lw=0, marker="D", ms=6, label="Actually exercised")
        ax.plot([0, 0.5], [0, 50], color=GRID, lw=1, zorder=0)
        ax.set_xlabel("|delta| at sale")
        ax.set_title(f"Short {side}s · 1-month · 4 markets · 2020-25", loc="left")
    axes[0].set_ylabel("% of options finishing in the money")
    axes[0].legend(loc="upper left")
    fig.suptitle("Puts are exercised far less often than the market implies; calls more often", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "03_exercise_probability.png")


def fig_nav() -> None:
    nav = pd.read_parquet(RES / "nav.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for ax, m in zip(axes.flat, MARKETS):
        for strat, st in STRAT_STYLE.items():
            d = nav[(nav["market"] == m) & (nav["strategy"] == strat)]
            ax.plot(d["date"], d["nav"], label=strat, **st)
        ax.set_title(MARKET_NAMES[m], loc="left")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:.1f}x"))
    axes[0, 0].legend(loc="upper left")
    fig.suptitle("Growth of 1 unit of local currency · Oct 2020 – Dec 2025", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "04_nav.png")


def fig_stress() -> None:
    st = pd.read_csv(RES / "stress.csv")
    strats = ["Equity (TR)", "50/50 Equity/Cash", "GDE 25-delta", "GDE Optimised"]
    colors = [MUTED, INK2, ORANGE, BLUE]
    scen = st["scenario"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharey=True)
    for ax, m in zip(axes.flat, MARKETS):
        d = st[st["market"] == m]
        width = 0.2
        y = np.arange(len(scen))
        for i, (s, c) in enumerate(zip(strats, colors)):
            vals = d[d["strategy"] == s].set_index("scenario").reindex(scen)["total_pnl"] * 100
            ax.barh(y + (i - 1.5) * width, vals, height=width * 0.9, color=c, label=s)
        ax.set_yticks(y, scen)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.invert_yaxis()
        ax.set_title(MARKET_NAMES[m], loc="left")
        ax.set_xlabel("1-day P&L, % of NAV")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Scenario stress on the latest live book (spot shock + implied-vol multiplier)", x=0.01,
                 ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    _save(fig, "05_stress.png")


def fig_greeks() -> None:
    g = pd.read_parquet(RES / "greeks.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    colors = dict(zip(MARKETS, [BLUE, ORANGE, AQUA, "#eda100"]))
    for m in MARKETS:
        d = g[(g["market"] == m) & (g["strategy"] == "GDE Optimised")]
        axes[0].plot(d["date"], d["delta_total"], color=colors[m], lw=1.3, label=m)
        axes[1].plot(d["date"], d["vega"] * 100, color=colors[m], lw=1.3, label=m)
    axes[0].set_title("Portfolio delta (equity-equivalent exposure)", loc="left")
    axes[1].set_title("Vega · % of NAV per +1 vol point", loc="left")
    axes[0].legend(ncol=4, loc="upper left")
    fig.suptitle("GDE Optimised: risk exposures through time (weekly)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    _save(fig, "06_greeks.png")


def _pct(x) -> str:
    return "" if pd.isna(x) else f"{x * 100:.1f}%"


def tables() -> str:
    perf = pd.read_csv(RES / "performance.csv")
    out = ["# Results\n", "_Generated by `python -m vrp_alpha.report`. Backtest: Oct 2020 – Dec 2025, "
           "local currency, monthly rolls, costs included._\n"]

    out.append("## Strategy performance\n")
    for m in MARKETS:
        d = perf[perf["market"] == m]
        out.append(f"### {MARKET_NAMES[m]}\n")
        out.append("| Strategy | CAGR | Vol | Sharpe | Sortino | Max DD | Alpha (ann.) | Beta | Down-capture | Put exercise rate | Put avg Δ |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for _, r in d.iterrows():
            out.append(
                f"| {r['strategy']} | {_pct(r['CAGR'])} | {_pct(r['Vol'])} | {r['Sharpe']:.2f} | {r['Sortino']:.2f} | "
                f"{_pct(r['MaxDD'])} | {_pct(r['Alpha_ann'])} | {r['Beta']:.2f} | {r['DownCapture']:.2f} | "
                f"{_pct(r.get('put_exercise_rate'))} | {'' if pd.isna(r.get('put_avg_delta')) else f'{r.get('put_avg_delta'):.2f}'} |")
        out.append("")

    ep = pd.read_csv(RES / "episodes.csv")
    out.append("## Stress episodes inside the window (total return over the episode)\n")
    cols = ["Equity (TR)", "50/50 Equity/Cash", "GDE ATM", "GDE 25-delta", "GDE Optimised"]
    out.append("| Market | Episode | " + " | ".join(cols) + " |")
    out.append("|---|---|" + "---:|" * len(cols))
    for _, r in ep.iterrows():
        out.append(f"| {r['market']} | {r['episode']} | " + " | ".join(_pct(r[c]) for c in cols) + " |")
    out.append("")

    sc = pd.read_csv(DATA_PROCESSED / "prob_scores.csv", index_col=0)
    out.append("## Exercise-probability models (walk-forward, 2021-2025, lower is better)\n")
    out.append("| Market | n | Realised ITM rate | Brier: market-implied | Brier: FHS | Brier: ML | Log-loss: market-implied | Log-loss: FHS | Log-loss: ML |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for m, r in sc.iterrows():
        out.append(f"| {m} | {int(r['n'])} | {_pct(r['itm_rate'])} | {r['brier_rn']:.4f} | **{r['brier_fhs']:.4f}** | "
                   f"{r['brier_ml']:.4f} | {r['logloss_rn']:.4f} | **{r['logloss_fhs']:.4f}** | {r['logloss_ml']:.4f} |")
    out.append("")

    bd = pd.read_csv(DATA_PROCESSED / "prob_by_delta.csv")
    out.append("## Exercise frequency by delta (all markets pooled)\n")
    out.append("| Side | Target Δ | n | Market-implied P(ITM) | FHS P(ITM) | Actually ITM | Avg premium (% spot) | Avg payoff (% spot) | Realised edge (% spot) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in bd.iterrows():
        out.append(f"| {r['side']} | {r['target_delta']:.2f} | {int(r['n'])} | {_pct(r['p_rn'])} | {_pct(r['p_fhs'])} | "
                   f"**{_pct(r['realised_itm'])}** | {r['avg_premium_pct'] * 100:.2f}% | {r['avg_payoff_pct'] * 100:.2f}% | "
                   f"{r['realised_edge_pct'] * 100:+.2f}% |")
    out.append("")

    cal = pd.read_csv(DATA_PROCESSED / "calibration_stats.csv")
    cal.columns = ["sample", "bench"] + list(cal.columns[2:])
    out.append("## Surface calibration vs CBOE benchmarks\n")
    out.append("| Sample | Benchmark | Monthly corr | Tracking error (ann.) | Model CAGR | Actual CAGR |")
    out.append("|---|---|---:|---:|---:|---:|")
    for _, r in cal.iterrows():
        out.append(f"| {r['sample']} | {r['bench']} | {r['corr']:.3f} | {_pct(r['tracking_err_ann'])} | "
                   f"{_pct(r['model_cagr'])} | {_pct(r['actual_cagr'])} |")
    out.append("")

    att = pd.read_csv(RES / "attribution.csv")
    out.append("## Attribution: where does the optimiser's edge come from?\n")
    out.append("| Market | Variant | CAGR | Vol | Sharpe | Max DD | Put exercise rate |")
    out.append("|---|---|---:|---:|---:|---:|---:|")
    for _, r in att.iterrows():
        out.append(f"| {r['market']} | {r['variant']} | {_pct(r['CAGR'])} | {_pct(r['Vol'])} | {r['Sharpe']:.2f} | "
                   f"{_pct(r['MaxDD'])} | {_pct(r['put_exercise_rate'])} |")
    out.append("")

    sens = pd.read_csv(RES / "sensitivity.csv")
    out.append("## Robustness: optimiser risk-aversion λ (GDE Optimised)\n")
    out.append("| Market | λ | CAGR | Sharpe | Max DD | Put exercise rate | Put avg Δ | Cycles with a put written |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in sens.iterrows():
        out.append(f"| {r['market']} | {r['lambda']:.2f} | {_pct(r['CAGR'])} | {r['Sharpe']:.2f} | {_pct(r['MaxDD'])} | "
                   f"{_pct(r['put_exercise_rate'])} | {r['put_avg_delta']:.2f} | {_pct(r['put_cycles_written'])} |")
    out.append("")

    st = pd.read_csv(RES / "stress.csv")
    piv = st.pivot_table(index=["market", "scenario"], columns="strategy", values="total_pnl", sort=False)
    out.append("## Scenario stress (1-day P&L, % of NAV, latest live book)\n")
    cols = [c for c in ["Equity (TR)", "50/50 Equity/Cash", "GDE ATM", "GDE 25-delta", "GDE Optimised"] if c in piv]
    out.append("| Market | Scenario | " + " | ".join(cols) + " |")
    out.append("|---|---|" + "---:|" * len(cols))
    for (m, s), r in piv.iterrows():
        out.append(f"| {m} | {s} | " + " | ".join(_pct(r[c]) for c in cols) + " |")
    out.append("")

    rec = pd.read_parquet(RES / "recommendations.parquet")
    out.append("## Live output: strike scan on the last date in the data\n")
    for m in MARKETS:
        d = rec[rec["market"] == m]
        if d.empty:
            continue
        out.append(f"**{MARKET_NAMES[m]}** — as of {d['date'].iloc[0]:%Y-%m-%d}, expiry {d['expiry'].iloc[0]:%Y-%m-%d}\n")
        out.append("| Side | Target Δ | Strike | IV | Premium | Market P(ITM) | Model P(ITM) | Edge (% strike) | Score (bp) | Eligible |")
        out.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|")
        best = {s: g.loc[g.loc[g["eligible"], "score"].idxmax()]["K"] if g["eligible"].any() else None
                for s, g in d.groupby("side")}
        for _, r in d.iterrows():
            mark = " ◀ pick" if best.get(r["side"]) == r["K"] else ""
            out.append(f"| {r['side']} | {r['target_delta']:.2f} | {r['K']:,.0f} | {_pct(r['iv'])} | {r['price']:,.2f} | "
                       f"{_pct(r['p_rn'])} | {_pct(r['p_phys'])} | {r['edge'] / r['K'] * 100:+.2f}% | {r['score'] * 1e4:+.1f} | "
                       f"{'✓' if r['eligible'] else '—'}{mark} |")
        out.append("")
    return "\n".join(out)


def run() -> None:
    fig_calibration()
    fig_vrp()
    fig_exercise()
    fig_nav()
    fig_stress()
    fig_greeks()
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "results.md").write_text(tables())
    print("report ->", REPORTS)


if __name__ == "__main__":
    run()
