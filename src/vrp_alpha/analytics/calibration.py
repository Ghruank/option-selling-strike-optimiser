"""Calibrate the synthetic US surface to real CBOE strategy benchmarks.

Without free historical SPX chains, the surface is only as credible as its ability
to reproduce real, traded outcomes. CBOE publishes daily values of rules-based
option-writing indices that settle against actual SPX option prices:

    PUT   - sell 1m ATM put, fully cash-collateralised     -> pins the ATM level
    BXM   - long SPX, sell 1m ATM call                     -> ATM level (call side)
    PPUT  - long SPX, buy 1m 5% OTM put                    -> pins the put wing
    BXMD  - long SPX, sell 1m 30-delta call                -> pins the call wing

We replicate each index cycle-by-cycle on the synthetic surface and choose the three
surface parameters that minimise the squared monthly-return gap. Fit on 2020-2022,
validate out of sample on 2023-2025, then refit on the full window for production.
"""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vrp_alpha.config import DATA_PROCESSED
from vrp_alpha.data.market import load_benchmarks
from vrp_alpha.pricing.surface import PARAMS_FILE, SurfaceParams, SyntheticSurface, year_frac

BENCHES = ["PUT", "BXM", "PPUT", "BXMD"]
SPLIT = "2023-01-01"


def replicate(surface: SyntheticSurface) -> pd.DataFrame:
    """Model monthly returns of each benchmark, one row per roll cycle."""
    df = surface.df
    rows = []
    exps = surface.settled_expiries
    for e0, e1 in zip(exps[:-1], exps[1:]):
        S0, S1 = df.at[e0, "spot"], df.at[e1, "spot"]
        r, q, T = df.at[e0, "r"], df.at[e0, "q"], year_frac(e0, e1)
        growth, div = np.exp(r * T), np.exp(q * T)
        put_atm = surface.mark(e0, e1, S0, False)
        call_atm = surface.mark(e0, e1, S0, True)
        K_pp = 0.95 * S0
        put_5 = surface.mark(e0, e1, K_pp, False)
        c30 = surface.select(e0, e1, 0.30, True)
        rows.append({
            "start": e0, "end": e1,
            "PUT": ((S0 + put_atm) * growth - max(S0 - S1, 0)) / S0 - 1,
            "BXM": (S1 * div - max(S1 - S0, 0)) / (S0 - call_atm) - 1,
            "PPUT": (S1 * div + max(K_pp - S1, 0)) / (S0 + put_5) - 1,
            "BXMD": (S1 * div - max(S1 - c30["K"], 0)) / (S0 - c30["price"]) - 1,
        })
    return pd.DataFrame(rows).set_index("end")


def actual_returns(expiries: pd.DatetimeIndex) -> pd.DataFrame:
    b = load_benchmarks()[BENCHES]
    b = b.reindex(b.index.union(expiries)).ffill().reindex(expiries)
    return b.pct_change().iloc[1:]


def _loss(x, surface, actual, mask):
    surface.params = SurfaceParams(*x)
    model = replicate(surface)
    diff = (model[BENCHES] - actual[BENCHES]).loc[mask]
    return float((diff**2).sum().sum())


def fit(surface: SyntheticSurface, actual: pd.DataFrame, mask) -> SurfaceParams:
    x0 = [0.88, 0.40, -0.29]
    res = minimize(_loss, x0, args=(surface, actual, mask), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-7, "maxiter": 300})
    return SurfaceParams(*res.x)


def tracking_stats(model: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for b in BENCHES:
        m, a = model[b], actual[b]
        out[b] = {
            "corr": m.corr(a),
            "tracking_err_ann": (m - a).std() * np.sqrt(12),
            "mean_gap_ann": (m - a).mean() * 12,
            "model_cagr": (1 + m).prod() ** (12 / len(m)) - 1,
            "actual_cagr": (1 + a).prod() ** (12 / len(a)) - 1,
        }
    return pd.DataFrame(out).T


def run() -> dict:
    surface = SyntheticSurface("US", SurfaceParams.load())
    actual = actual_returns(surface.settled_expiries)
    in_sample = actual.index < SPLIT

    p_is = fit(surface, actual, in_sample)
    surface.params = p_is
    model = replicate(surface)
    oos = tracking_stats(model.loc[~in_sample], actual.loc[~in_sample])
    ins = tracking_stats(model.loc[in_sample], actual.loc[in_sample])

    p_full = fit(surface, actual, np.ones(len(actual), bool))
    surface.params = p_full
    model_full = replicate(surface)
    PARAMS_FILE.write_text(json.dumps(asdict(p_full), indent=2))

    report = {
        "params_in_sample": asdict(p_is),
        "params_full": asdict(p_full),
        "in_sample": ins,
        "out_of_sample": oos,
        "full": tracking_stats(model_full, actual),
    }
    series = pd.concat({"model": model_full[BENCHES], "actual": actual[BENCHES]}, axis=1)
    series.to_parquet(DATA_PROCESSED / "calibration_series.parquet")
    pd.concat({k: report[k] for k in ["in_sample", "out_of_sample", "full"]}).to_csv(
        DATA_PROCESSED / "calibration_stats.csv")
    return report


if __name__ == "__main__":
    rep = run()
    for k, v in rep.items():
        print(f"\n== {k}")
        print(v.round(4).to_string() if isinstance(v, pd.DataFrame) else v)
