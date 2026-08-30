"""Volatility risk premium monitor: implied vs forecast vs realised volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from vrp_alpha.config import DATA_PROCESSED
from vrp_alpha.models.volforecast import forecast_table
from vrp_alpha.pricing.surface import OptionMarket


def vrp_table(mk: OptionMarket) -> pd.DataFrame:
    """Daily ATM 30d IV, term slope, vol forecasts and the ex-ante / ex-post VRP."""
    vt = forecast_table(mk.df["spot"])
    iv30 = pd.Series({d: mk.atm_iv(d, 30) for d in mk.dates})
    iv90 = pd.Series({d: mk.atm_iv(d, 90) for d in mk.dates})
    df = vt.join(pd.DataFrame({"iv30": iv30, "term": iv90 / iv30}))
    df["vrp_exante"] = df["iv30"] - df["ens"]        # tradable signal
    df["vrp_expost"] = df["iv30"] - df["fwd_rv"]     # what was actually harvested
    spot = mk.df["spot"]
    df["mom21"] = np.log(spot / spot.shift(21))
    df["mom63"] = np.log(spot / spot.shift(63))
    df["dd252"] = spot / spot.rolling(252, min_periods=1).max() - 1
    if "INDIAVIX" in mk.df:
        df["indiavix"] = mk.df["INDIAVIX"] / 100
    return df


def summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = {}
    for m, df in tables.items():
        d = df.dropna(subset=["iv30", "fwd_rv"])
        rows[m] = {
            "avg_iv30": d["iv30"].mean(),
            "avg_realised": d["fwd_rv"].mean(),
            "avg_vrp_vol_pts": d["vrp_expost"].mean(),
            "pct_days_vrp_positive": (d["vrp_expost"] > 0).mean(),
            "avg_var_premium": (d["iv30"] ** 2 - d["fwd_rv"] ** 2).mean(),
            "iv_over_rv_median": (d["iv30"] / d["fwd_rv"]).median(),
            "worst_vrp_vol_pts": d["vrp_expost"].min(),
            "worst_vrp_date": d["vrp_expost"].idxmin().date(),
        }
    return pd.DataFrame(rows).T


def save(tables: dict[str, pd.DataFrame]) -> None:
    pd.concat(tables, names=["market", "date"]).to_parquet(DATA_PROCESSED / "vrp.parquet")
    summary(tables).to_csv(DATA_PROCESSED / "vrp_summary.csv")


def load() -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(DATA_PROCESSED / "vrp.parquet")
    return {m: df.xs(m) for m in df.index.get_level_values(0).unique()}
