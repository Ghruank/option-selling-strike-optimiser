"""GDE-style backtest: equity sleeve with call overwriting + cash-collateralised puts.

Each monthly cycle (expiry e0 -> e1), starting from NAV0:
  * equity sleeve  w * NAV0 in the index (total return: price + dividend yield)
  * short calls    coverage * equity units (covered)
  * cash sleeve    (1 - w) * NAV0 earning the local short rate, fully securing
                   n_put = (1 - w) * NAV0 / K_put short puts
  * premiums (net of friction) are added to cash; options are held to expiry and
    cash-settled at intrinsic value. Daily NAV marks the options to market.

Strike rules: ("delta", d) fixed target delta, ("opt",) the optimiser, None = no leg.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vrp_alpha.config import load_config
from vrp_alpha.models.probability import FHS
from vrp_alpha.optimizer.strikes import choose, friction
from vrp_alpha.pricing import bs
from vrp_alpha.pricing.surface import OptionMarket, year_frac


@dataclass
class Strategy:
    name: str
    equity_weight: float
    put_rule: tuple | None = None
    call_rule: tuple | None = None
    call_coverage: float = 1.0
    risk_aversion: float | None = None  # override for sensitivity runs


def default_strategies() -> list[Strategy]:
    w = load_config()["portfolio"]["equity_weight"]
    return [
        Strategy("Equity (TR)", 1.0),
        Strategy("50/50 Equity/Cash", w),
        Strategy("GDE ATM", w, ("delta", 0.50), ("delta", 0.50)),
        Strategy("GDE 25-delta", w, ("delta", 0.25), ("delta", 0.25)),
        Strategy("GDE Optimised", w, ("opt",), ("opt",)),
    ]


@dataclass
class Leg:
    is_call: bool
    K: float
    qty: float
    premium: float
    info: dict = field(default_factory=dict)


def _open_leg(mk, fhs, vt, rule, d, e, is_call, qty_fn, risk_aversion=None) -> Leg | None:
    if rule is None:
        return None
    if rule[0] == "delta":
        sel = mk.select(d, e, rule[1], is_call)
        if sel is None:
            return None
        sel["premium_net"] = sel["price"] - float(friction(sel["price"], mk.name))
    else:
        sel = choose(mk, fhs, d, e, float(vt.at[d, "ens"]), is_call, risk_aversion)
        if sel is None:
            return None
    qty = qty_fn(sel["K"])
    return Leg(is_call, float(sel["K"]), qty, float(sel["premium_net"]), sel)


def run_backtest(mk: OptionMarket, strat: Strategy, fhs: FHS, vt: pd.DataFrame,
                 start: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = load_config()
    start = pd.Timestamp(start or cfg["backtest_start"])
    exps = mk.settled_expiries[mk.settled_expiries >= start]
    df = mk.df
    nav = 1.0
    daily, trades = [], []
    for e0, e1 in zip(exps[:-1], exps[1:]):
        S0, r, q = df.at[e0, "spot"], df.at[e0, "r"], df.at[e0, "q"]
        w = strat.equity_weight
        units = w * nav / S0
        cash = (1 - w) * nav
        put = _open_leg(mk, fhs, vt, strat.put_rule, e0, e1, False, lambda K: (1 - w) * nav / K,
                        strat.risk_aversion)
        call = _open_leg(mk, fhs, vt, strat.call_rule, e0, e1, True, lambda K: strat.call_coverage * units,
                         strat.risk_aversion)
        legs = [leg for leg in (put, call) if leg is not None]
        cash += sum(leg.qty * leg.premium for leg in legs)

        last_mark = {id(leg): leg.info.get("price", leg.premium) for leg in legs}
        for t in df.index[(df.index > e0) & (df.index <= e1)]:
            tau = year_frac(e0, t)
            eq = units * df.at[t, "spot"] * np.exp(q * tau)
            cash_t = cash * np.exp(r * tau)
            opt = 0.0
            for leg in legs:
                m = mk.mark(t, e1, leg.K, leg.is_call)
                if np.isnan(m):
                    m = last_mark[id(leg)]
                last_mark[id(leg)] = m
                opt += leg.qty * m
            daily.append({"date": t, "nav": eq + cash_t - opt, "equity": eq, "cash": cash_t,
                          "short_options": opt})
        nav = daily[-1]["nav"]

        ST = mk.settlement(e1)
        for leg in legs:
            payoff = max(ST - leg.K, 0) if leg.is_call else max(leg.K - ST, 0)
            trades.append({
                "entry": e0, "expiry": e1, "side": "call" if leg.is_call else "put",
                "K": leg.K, "S0": S0, "ST": ST, "qty": leg.qty, "premium_net": leg.premium,
                "iv": leg.info.get("iv"), "delta": abs(leg.info.get("delta", np.nan)),
                "p_phys": leg.info.get("p_phys"), "exercised": payoff > 0, "payoff": payoff,
                "pnl_pct_nav": leg.qty * (leg.premium - payoff) / (daily[-1]["nav"] or 1),
            })
        for side, rule in (("put", strat.put_rule), ("call", strat.call_rule)):
            if rule is not None and not any(tr["entry"] == e0 and tr["side"] == side for tr in trades):
                trades.append({"entry": e0, "expiry": e1, "side": side, "skipped": True})
    nav_df = pd.DataFrame(daily).set_index("date")
    start_row = pd.DataFrame({"nav": [1.0]}, index=[exps[0]])
    nav_df = pd.concat([start_row, nav_df])
    return nav_df, pd.DataFrame(trades)


def book_greeks(mk: OptionMarket, legs_df: pd.DataFrame, date, nav: float) -> dict:
    """Net Greeks of the short option book (as fraction of NAV) at a date."""
    row = mk.df.loc[pd.Timestamp(date)]
    S, r, q = row["spot"], row["r"], row["q"]
    tot = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    for _, lg in legs_df.iterrows():
        T = year_frac(date, lg["expiry"])
        if T <= 0:
            continue
        is_call = lg["side"] == "call"
        px = mk.mark(date, lg["expiry"], lg["K"], is_call)
        iv = bs.implied_vol(px, S, lg["K"], T, r, q, is_call)
        if not np.isfinite(iv):
            continue
        g = bs.greeks(S, lg["K"], T, r, q, iv, is_call)
        tot["delta"] -= lg["qty"] * g["delta"] * S / nav        # dollar delta / NAV
        tot["gamma"] -= lg["qty"] * g["gamma"] * S * S / 100 / nav  # delta change per 1% move
        tot["vega"] -= lg["qty"] * g["vega"] / 100 / nav        # NAV change per 1 vol pt
        tot["theta"] -= lg["qty"] * g["theta"] / 365 / nav      # NAV change per day
    return tot
