"""Strike selection: maximise risk-adjusted premium capture under the physical measure.

For each candidate strike (target deltas from config) on the chosen expiry:

    premium_net = market premium - trading friction
    edge        = premium_net * e^{rT} - E_phys[payoff]          (index points)
    score       = (edge - lambda * CVaR95_phys[payoff]) / K       (per unit of collateral)

subject to P_phys(exercise) <= max_itm_prob. If no candidate has a positive score the
side is left unwritten for the cycle - the tool also says when *not* to sell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_alpha.config import load_config
from vrp_alpha.models.probability import FHS, candidate_options
from vrp_alpha.pricing import bs
from vrp_alpha.pricing.surface import OptionMarket


def friction(premium, market: str) -> np.ndarray:
    c = load_config()["costs"]
    return np.maximum(np.asarray(premium) * c["premium_haircut"], c["min_points"][market])


def score_candidates(mk: OptionMarket, fhs: FHS, date, expiry, sigma_fcst: float,
                     is_call: bool, risk_aversion: float | None = None) -> pd.DataFrame:
    cfg = load_config()["optimizer"]
    lam = cfg["risk_aversion"] if risk_aversion is None else risk_aversion
    deltas = cfg["call_deltas"] if is_call else cfg["put_deltas"]
    cand = candidate_options(mk, date, expiry, deltas)
    if cand.empty:
        return cand
    cand = cand[cand["is_call"] == is_call].copy()
    row = mk.df.loc[pd.Timestamp(date)]
    S, r, q = row["spot"], row["r"], row["q"]
    T = float(cand["T"].iloc[0])
    ph = fhs.evaluate(date, S, cand["K"].values, T, r, q, sigma_fcst, is_call)
    cand["premium_net"] = cand["price"] - friction(cand["price"], mk.name)
    cand["p_rn"] = bs.prob_itm_rn(S, cand["K"].values, T, r, q, cand["iv"].values, is_call)
    cand["p_phys"] = ph["p_itm"]
    cand["e_payoff"] = ph["e_payoff"]
    cand["cvar95"] = ph["cvar95"]
    cand["edge"] = cand["premium_net"] * np.exp(r * T) - cand["e_payoff"]
    cand["score"] = (cand["edge"] - lam * cand["cvar95"]) / cand["K"]
    cand["eligible"] = (cand["p_phys"] <= cfg["max_itm_prob"]) & (cand["score"] > 0)
    return cand.reset_index(drop=True)


def choose(mk: OptionMarket, fhs: FHS, date, expiry, sigma_fcst: float, is_call: bool,
           risk_aversion: float | None = None) -> dict | None:
    cand = score_candidates(mk, fhs, date, expiry, sigma_fcst, is_call, risk_aversion)
    if cand.empty or not cand["eligible"].any():
        return None
    return cand.loc[cand.loc[cand["eligible"], "score"].idxmax()].to_dict()
