"""Performance and risk metrics, all in local currency against the local cash rate."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPISODES = {
    "2022 bear market": ("2022-01-03", "2022-10-12"),
    "Aug-2024 vol shock": ("2024-07-31", "2024-08-05"),
    "2025 tariff drawdown": ("2025-02-19", "2025-04-08"),
}


def _rf_daily(nav: pd.Series, r: pd.Series) -> pd.Series:
    rr = r.reindex(r.index.union(nav.index)).ffill().reindex(nav.index)
    dt = nav.index.to_series().diff().dt.days.fillna(0) / 365
    return np.expm1(rr * dt)


def performance(nav: pd.Series, bench: pd.Series, r: pd.Series) -> dict:
    """nav/bench: daily NAV series (same calendar); r: continuous short rate."""
    ret = nav.pct_change().dropna()
    b = bench.reindex(nav.index).pct_change().dropna()
    rf = _rf_daily(nav, r).reindex(ret.index)
    ex, bex = ret - rf, b - rf
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    ann = np.sqrt(252)
    dd = nav / nav.cummax() - 1
    downside = ex[ex < 0].std() * ann
    beta = np.cov(ex, bex)[0, 1] / bex.var()
    alpha = (ex.mean() - beta * bex.mean()) * 252
    m = nav.resample("ME").last().pct_change().dropna()
    bm = bench.reindex(nav.index).resample("ME").last().pct_change().dropna()
    up, down = bm > 0, bm < 0
    tail = ret.quantile(0.05)
    return {
        "CAGR": nav.iloc[-1] ** (1 / yrs) - 1,
        "Vol": ret.std() * ann,
        "Sharpe": ex.mean() / ex.std() * ann,
        "Sortino": ex.mean() * 252 / downside,
        "MaxDD": dd.min(),
        "Calmar": (nav.iloc[-1] ** (1 / yrs) - 1) / abs(dd.min()),
        "CVaR95_daily": ret[ret <= tail].mean(),
        "Beta": beta,
        "Alpha_ann": alpha,
        "UpCapture": m[up].mean() / bm[up].mean(),
        "DownCapture": m[down].mean() / bm[down].mean(),
    }


def option_stats(trades: pd.DataFrame, nav: pd.Series) -> dict:
    if trades.empty or "side" not in trades:
        return {}
    live = trades[trades.get("skipped", pd.Series(False, index=trades.index)).ne(True)]
    out = {}
    for side in ("put", "call"):
        g = live[live["side"] == side]
        n_cycles = trades[trades["side"] == side]["entry"].nunique()
        out[f"{side}_exercise_rate"] = g["exercised"].mean() if len(g) else np.nan
        out[f"{side}_avg_delta"] = g["delta"].mean() if len(g) else np.nan
        out[f"{side}_cycles_written"] = len(g) / n_cycles if n_cycles else np.nan
        nav_at = nav.reindex(g["entry"]).values if len(g) else np.array([])
        prem = (g["qty"] * g["premium_net"]).values / nav_at if len(g) else np.array([0.0])
        yrs = (nav.index[-1] - nav.index[0]).days / 365.25
        out[f"{side}_premium_yield_ann"] = np.nansum(prem) / yrs
    return out


def episodes(navs: dict[str, pd.Series]) -> pd.DataFrame:
    rows = {}
    for name, (a, b) in EPISODES.items():
        rows[name] = {s: n.asof(pd.Timestamp(b)) / n.asof(pd.Timestamp(a)) - 1 for s, n in navs.items()}
    return pd.DataFrame(rows).T
