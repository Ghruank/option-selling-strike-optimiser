"""Exercise probability: risk-neutral vs physical vs machine-learned.

The original question - "which options will never be exercised?" - cannot be answered
with certainty, but it can be answered in probability. Three estimates per option:

  P_rn   market-implied (risk-neutral) N(-d2) using the option's own implied vol.
  P_fhs  physical, Filtered Historical Simulation: standardised past 1m returns
         (pooled across markets, only those observable at the date) rescaled by the
         ensemble vol forecast, shrunk toward a fat-tailed Student-t prior because the
         post-COVID window contains no full crash.
  P_ml   regularised logistic layer over the above, trained walk-forward on outcomes.

Because implied vol carries a risk premium, P_rn systematically overstates the chance
of exercise. The gap between P_rn and the physical estimates is the edge the strike
optimizer monetises.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from vrp_alpha.config import DATA_PROCESSED, load_config
from vrp_alpha.models.volforecast import H
from vrp_alpha.pricing import bs
from vrp_alpha.pricing.surface import OptionMarket, year_frac

DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
T_DOF = 4
PRIOR_WEIGHT = 500  # pseudo-observations of the Student-t prior
_T_SAMPLE = stats.t.ppf((np.arange(2000) + 0.5) / 2000, T_DOF) / np.sqrt(T_DOF / (T_DOF - 2))


# --------------------------------------------------------------------------- FHS
def standardised_residuals(vrp_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """z = (H-day log return) / (ex-ante ensemble vol * sqrt(H/252)), with the date it
    becomes observable (t + H trading days). Pooled across markets."""
    frames = []
    for m, df in vrp_tables.items():
        logp = df["ret"].cumsum()
        fwd = logp.shift(-H) - logp
        z = fwd / (df["ens"] * np.sqrt(H / 252))
        avail = pd.Series(df.index, index=df.index).shift(-H)
        frames.append(pd.DataFrame({"market": m, "z": z, "available": avail}).dropna())
    return pd.concat(frames).sort_values("available")


class FHS:
    """Physical return distribution at a given date (no look-ahead)."""

    def __init__(self, resid: pd.DataFrame, erp: float):
        self.resid = resid
        self.erp = erp

    def sample(self, date) -> tuple[np.ndarray, np.ndarray]:
        emp = self.resid.loc[self.resid["available"] <= pd.Timestamp(date), "z"].values
        emp = (emp - emp.mean()) / emp.std() if len(emp) > 30 else np.empty(0)
        n = len(emp)
        w_emp = n / (n + PRIOR_WEIGHT)
        z = np.concatenate([emp, _T_SAMPLE])
        w = np.concatenate([np.full(n, w_emp / max(n, 1)), np.full(len(_T_SAMPLE), (1 - w_emp) / len(_T_SAMPLE))])
        return z, w

    def evaluate(self, date, S, K, T, r, q, sigma, is_call) -> dict[str, np.ndarray]:
        """P(ITM), E[payoff] and CVaR95 of payoff under the physical measure, per strike."""
        z, w = self.sample(date)
        K = np.atleast_1d(np.asarray(K, float))
        m = (r - q + self.erp - 0.5 * sigma**2) * T
        ST = S * np.exp(m + sigma * np.sqrt(T) * z)                 # (n,)
        pay = np.maximum(ST[None, :] - K[:, None], 0) if is_call else np.maximum(K[:, None] - ST[None, :], 0)
        p_itm = ((pay > 0) * w).sum(axis=1)
        e_pay = (pay * w).sum(axis=1)
        # CVaR95: weighted mean of the worst 5% payoffs
        order = np.argsort(-pay, axis=1)
        pw = w[order]
        cum = np.cumsum(pw, axis=1)
        tail = np.minimum(pw, np.maximum(0.05 - (cum - pw), 0))
        cvar = (np.take_along_axis(pay, order, axis=1) * tail).sum(axis=1) / 0.05
        return {"p_itm": p_itm, "e_payoff": e_pay, "cvar95": cvar}


# --------------------------------------------------------------------------- dataset
def candidate_options(mk: OptionMarket, date, expiry, deltas=DELTAS) -> pd.DataFrame:
    """Listed (or synthetic) OTM options nearest to each target delta."""
    rows = []
    for is_call in (False, True):
        sm = mk.smile(date, expiry, is_call)
        if sm.empty:
            continue
        for d in deltas:
            row = sm.iloc[(sm["delta"].abs() - d).abs().argmin()].to_dict()
            row.update(is_call=is_call, target_delta=d)
            rows.append(row)
    out = pd.DataFrame(rows)
    return out.drop_duplicates(["is_call", "K"]) if not out.empty else out


def build_dataset(mk: OptionMarket, vt: pd.DataFrame, fhs: FHS) -> pd.DataFrame:
    """One row per (date, option): features, three probabilities and the realised outcome."""
    last = mk.dates[-1]
    recs = []
    for d in vt.dropna(subset=["ens", "iv30"]).index:
        e = mk.next_expiry(d, 21)
        if e is None:
            continue
        cand = candidate_options(mk, d, e)
        if cand.empty:
            continue
        row = mk.df.loc[d]
        S, r, q = row["spot"], row["r"], row["q"]
        f = vt.loc[d]
        for is_call, g in cand.groupby("is_call"):
            T = g["T"].values
            ph = fhs.evaluate(d, S, g["K"].values, T[0], r, q, f["ens"], is_call)
            p_rn = bs.prob_itm_rn(S, g["K"].values, T, r, q, g["iv"].values, is_call)
            settled = e <= last
            ST = mk.settlement(e) if settled else np.nan
            itm = (ST > g["K"].values) if is_call else (ST < g["K"].values)
            payoff = np.maximum(ST - g["K"].values, 0) if is_call else np.maximum(g["K"].values - ST, 0)
            lm = np.log(g["K"].values / g["F"].values)
            recs.append(pd.DataFrame({
                "market": mk.name, "date": d, "expiry": e, "is_call": is_call,
                "target_delta": g["target_delta"].values, "delta": g["delta"].abs().values,
                "K": g["K"].values, "S": S, "F": g["F"].values, "T": T, "iv": g["iv"].values,
                "premium": g["price"].values, "r": r,
                "p_rn": p_rn, "p_fhs": ph["p_itm"], "e_payoff_fhs": ph["e_payoff"], "cvar95_fhs": ph["cvar95"],
                # features (all known at date d)
                "otm_dist_fcst": np.abs(lm) / (f["ens"] * np.sqrt(T)),
                "otm_dist_iv": np.abs(lm) / (g["iv"].values * np.sqrt(T)),
                "iv_over_fcst": g["iv"].values / f["ens"],
                "rv_over_fcst": f["rv21"] / f["ens"],
                "term": f["term"], "mom21": f["mom21"], "mom63": f["mom63"], "dd252": f["dd252"],
                # outcome
                "settled": settled, "ST": ST,
                "itm": np.where(settled, itm, np.nan), "payoff": payoff,
            }))
    return pd.concat(recs, ignore_index=True)


# --------------------------------------------------------------------------- ML
# A regularised logistic layer on top of the structural estimates. A deeper model
# (gradient boosting on raw features) was tried and overfit badly across regimes:
# 2021 (melt-up) -> 2022 (bear) -> 2023-25 (bull) shifts the base rates, and five years
# of overlapping monthly outcomes (~70 independent expiries per market) is too little
# data to learn stable corrections. The logistic layer is kept as the ML benchmark.
def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _X(df: pd.DataFrame) -> pd.DataFrame:
    call = df["is_call"].astype(float)
    l_fhs = _logit(df["p_fhs"])
    return pd.DataFrame({
        "l_fhs": l_fhs,
        "l_rn": _logit(df["p_rn"]),
        "log_iv_over_fcst": np.log(df["iv_over_fcst"]),
        "term": df["term"].fillna(1.0),
        "call": call,
        "call_x_l_fhs": call * l_fhs,
    })


def _model() -> LogisticRegression:
    return LogisticRegression(C=0.1, max_iter=1000)


def walk_forward_ml(ds: pd.DataFrame, first_test_year: int = 2021) -> pd.Series:
    """Yearly walk-forward: train only on options whose expiry is before the test year."""
    p = pd.Series(np.nan, index=ds.index)
    X = _X(ds)
    for year in range(first_test_year, ds["date"].dt.year.max() + 1):
        cutoff = pd.Timestamp(f"{year}-01-01")
        train = ds["settled"] & (ds["expiry"] < cutoff)
        test = ds["date"].dt.year == year
        if train.sum() < 2000 or not test.any():
            continue
        clf = _model().fit(X[train], ds.loc[train, "itm"].astype(int))
        p[test] = clf.predict_proba(X[test])[:, 1]
    return p


def evaluate(ds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scores on rows where all three probabilities exist and the outcome is known."""
    ev = ds[ds["settled"] & ds["p_ml"].notna()]
    scores = {}
    for name, g in list(ev.groupby("market")) + [("ALL", ev)]:
        y = g["itm"].astype(int)
        scores[name] = {
            "n": len(g), "itm_rate": y.mean(),
            **{f"brier_{k}": brier_score_loss(y, g[f"p_{k}"].clip(1e-4, 1 - 1e-4)) for k in ["rn", "fhs", "ml"]},
            **{f"logloss_{k}": log_loss(y, g[f"p_{k}"].clip(1e-4, 1 - 1e-4), labels=[0, 1]) for k in ["rn", "fhs", "ml"]},
        }
    by_delta = (
        ds[ds["settled"]]
        .assign(side=lambda x: np.where(x["is_call"], "call", "put"))
        .groupby(["side", "target_delta"])
        .agg(n=("itm", "size"), realised_itm=("itm", "mean"), p_rn=("p_rn", "mean"),
             p_fhs=("p_fhs", "mean"), p_ml=("p_ml", "mean"),
             avg_premium_pct=("premium", lambda s: (s / ds.loc[s.index, "S"]).mean()),
             avg_payoff_pct=("payoff", lambda s: (s / ds.loc[s.index, "S"]).mean()))
    )
    by_delta["realised_edge_pct"] = by_delta["avg_premium_pct"] - by_delta["avg_payoff_pct"]
    return pd.DataFrame(scores).T, by_delta


def make_fhs(vrp_tables: dict[str, pd.DataFrame]) -> FHS:
    return FHS(standardised_residuals(vrp_tables), load_config()["optimizer"]["erp"])


def run(markets: dict[str, OptionMarket], vrp_tables: dict[str, pd.DataFrame], rebuild: bool = True) -> pd.DataFrame:
    path = DATA_PROCESSED / "prob_dataset.parquet"
    if rebuild or not path.exists():
        fhs = make_fhs(vrp_tables)
        ds = pd.concat([build_dataset(mk, vrp_tables[m], fhs) for m, mk in markets.items()], ignore_index=True)
    else:
        ds = pd.read_parquet(path)
    ds["p_ml"] = walk_forward_ml(ds)
    ds.to_parquet(path)
    scores, by_delta = evaluate(ds)
    scores.to_csv(DATA_PROCESSED / "prob_scores.csv")
    by_delta.to_csv(DATA_PROCESSED / "prob_by_delta.csv")
    return ds
