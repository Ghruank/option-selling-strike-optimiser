"""Scenario stress tests on a live GDE book.

The study window deliberately starts after the COVID crash, so the backtest never
experiences a -34% month. Stress tests close that gap without using pre-window data:
each scenario shocks spot and scales implied vols (sticky-strike), advances one day and
revalues the equity sleeve, cash and short options.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_alpha.config import load_config
from vrp_alpha.pricing import bs
from vrp_alpha.pricing.surface import OptionMarket, year_frac


def book_at(trades: pd.DataFrame, date) -> pd.DataFrame:
    """Option legs open at `date` (entered on or before, expiring after)."""
    if trades.empty or "K" not in trades:
        return trades
    t = trades[trades["K"].notna()]
    d = pd.Timestamp(date)
    return t[(t["entry"] <= d) & (t["expiry"] > d)]


def stress_book(mk: OptionMarket, legs: pd.DataFrame, date, nav: float, equity_value: float) -> pd.DataFrame:
    date = pd.Timestamp(date)
    row = mk.df.loc[date]
    S, r, q = row["spot"], row["r"], row["q"]
    rows = []
    for sc in load_config()["stress"]["scenarios"]:
        S1 = S * (1 + sc["spot"])
        eq_pnl = equity_value * sc["spot"]
        opt_pnl = 0.0
        for _, lg in legs.iterrows():
            is_call = lg["side"] == "call"
            T = year_frac(date, lg["expiry"])
            px = mk.mark(date, lg["expiry"], lg["K"], is_call)
            iv = bs.implied_vol(px, S, lg["K"], T, r, q, is_call)
            iv = iv if np.isfinite(iv) else mk.atm_iv(date, T * 365)
            new = bs.price(S1, lg["K"], max(T - 1 / 365, 0), r, q, min(iv * sc["vol_mult"], 3.0), is_call)
            opt_pnl -= lg["qty"] * (new - px)
        rows.append({"scenario": sc["name"], "spot_shock": sc["spot"], "vol_mult": sc["vol_mult"],
                     "equity_pnl": eq_pnl / nav, "options_pnl": opt_pnl / nav,
                     "total_pnl": (eq_pnl + opt_pnl) / nav})
    return pd.DataFrame(rows)
