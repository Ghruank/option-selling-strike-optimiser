"""Option markets behind one interface.

* NSEChain         (India)       - real NIFTY option settlement prices from NSE bhavcopy.
* SyntheticSurface (US, EU, UK)  - no free historical chains exist, so options are priced
  off a volatility surface built from the VIX term structure (VIX9D/VIX/VIX3M) and a
  parametric smile. Its three parameters are calibrated so that replicating the CBOE
  PUT, PPUT and BXMD benchmark indices reproduces their *actual* published returns
  (see analytics/calibration.py). EU and UK scale the US surface by the trailing ratio
  of local to US realised volatility, because VSTOXX / FTSE IVI history is not free.

Common interface (all dates are trading days, T in calendar years):
    expiries, next_expiry(date), forward(date, expiry), smile(date, expiry, is_call),
    select(date, expiry, abs_delta, is_call), mark(date, expiry, K, is_call),
    atm_iv(date, days), settlement(expiry)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from vrp_alpha.config import DATA_PROCESSED
from vrp_alpha.data.market import load_market
from vrp_alpha.pricing import bs

PARAMS_FILE = DATA_PROCESSED / "surface_params.json"
DEFAULT_PARAMS = {"c_atm": 0.88, "put_slope": 0.40, "call_slope": -0.29}
PUT_POWER, CALL_CURV = 0.8, 0.15  # fixed shape constants; slopes are calibrated
SMILE_FLOOR, SMILE_CAP = 0.55, 3.0


def year_frac(d0, d1) -> float:
    return max((pd.Timestamp(d1) - pd.Timestamp(d0)).days, 0) / 365.0


def third_friday_expiries(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Monthly expiries: third Friday, moved to the prior trading day on holidays."""
    months = pd.period_range(dates[0], dates[-1], freq="M")
    out = []
    for m in months:
        first = m.start_time
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)
        prior = dates[dates <= fri]
        if len(prior) and (fri - prior[-1]).days < 5:
            out.append(prior[-1])
    return pd.DatetimeIndex(out)


def ewma_vol(returns: pd.Series, lam: float = 0.97) -> pd.Series:
    """Annualised RiskMetrics EWMA volatility (uses returns up to and including t)."""
    var = (returns**2).ewm(alpha=1 - lam, adjust=False).mean()
    return np.sqrt(252 * var)


class OptionMarket:
    name: str
    df: pd.DataFrame
    expiries: pd.DatetimeIndex

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.df.index

    def next_expiry(self, date, min_days: int = 21) -> pd.Timestamp | None:
        later = self.expiries[self.expiries >= pd.Timestamp(date) + pd.Timedelta(days=min_days)]
        return later[0] if len(later) else None

    @property
    def settled_expiries(self) -> pd.DatetimeIndex:
        return self.expiries[self.expiries <= self.dates[-1]]

    def settlement(self, expiry) -> float:
        return float(self.df["spot"].asof(pd.Timestamp(expiry)))

    def select(self, date, expiry, abs_delta: float, is_call: bool) -> dict | None:
        sm = self.smile(date, expiry, is_call)
        if sm.empty:
            return None
        row = sm.iloc[(sm["delta"].abs() - abs_delta).abs().argmin()]
        return row.to_dict()


# --------------------------------------------------------------------------- synthetic
@dataclass
class SurfaceParams:
    c_atm: float  # ATM implied vol as a fraction of the VIX-family (variance-swap) level
    put_slope: float   # put wing:  1 + put_slope * |z|^0.8
    call_slope: float  # call wing: 1 + call_slope * z + 0.15 z^2

    @classmethod
    def load(cls) -> "SurfaceParams":
        if PARAMS_FILE.exists():
            return cls(**json.loads(PARAMS_FILE.read_text()))
        return cls(**DEFAULT_PARAMS)


class SyntheticSurface(OptionMarket):
    TENORS = np.array([9.0, 30.0, 93.0])

    def __init__(self, name: str, params: SurfaceParams | None = None):
        self.name = name
        self.params = params or SurfaceParams.load()
        df = load_market(name)
        df["vol_scale"] = 1.0 if name == "US" else self._vol_scale(df)
        self.df = df
        # extend the calendar ~4 months past the data so the latest date has listed expiries
        future = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=90)
        self.expiries = third_friday_expiries(df.index.union(future))

    @staticmethod
    def _vol_scale(df: pd.DataFrame) -> pd.Series:
        """Trailing ratio of local to US realised vol (EWMA, smoothed over ~6 months)."""
        us = load_market("US")["spot"]
        local_vol = ewma_vol(np.log(df["spot"]).diff().dropna())
        us_vol = ewma_vol(np.log(us).diff().dropna()).shift(1)  # US close is after EU/UK close
        us_vol = us_vol.reindex(us_vol.index.union(local_vol.index)).ffill().reindex(local_vol.index)
        ratio = (local_vol / us_vol).rolling(126, min_periods=21).median()
        return ratio.reindex(df.index).bfill().clip(0.6, 1.6)

    def atm_iv(self, date, days: float = 30.0) -> float:
        row = self.df.loc[pd.Timestamp(date)]
        levels = np.array([row["VIX9D"], row["VIX"], row["VIX3M"]], dtype=float) / 100.0
        ok = np.isfinite(levels)
        tenors, levels = self.TENORS[ok], levels[ok]
        tv = levels**2 * tenors  # total variance is linear-interpolable in time
        d = float(np.clip(days, tenors[0], tenors[-1]))
        vol = np.sqrt(np.interp(d, tenors, tv) / d)
        return float(self.params.c_atm * vol * row["vol_scale"])

    def iv(self, date, T: float, K, F: float, atm: float | None = None):
        atm = self.atm_iv(date, T * 365) if atm is None else atm
        z = np.log(np.asarray(K, float) / F) / (atm * np.sqrt(max(T, 1 / 365)))
        mult = np.where(
            z < 0,
            1.0 + self.params.put_slope * np.abs(z) ** PUT_POWER,
            1.0 + self.params.call_slope * z + CALL_CURV * z**2,
        )
        return atm * np.clip(mult, SMILE_FLOOR, SMILE_CAP)

    def forward(self, date, expiry) -> float:
        row = self.df.loc[pd.Timestamp(date)]
        return float(row["spot"] * np.exp((row["r"] - row["q"]) * year_frac(date, expiry)))

    def smile(self, date, expiry, is_call: bool) -> pd.DataFrame:
        row = self.df.loc[pd.Timestamp(date)]
        T = year_frac(date, expiry)
        F = self.forward(date, expiry)
        atm = self.atm_iv(date, T * 365)
        z = np.arange(-4.0, 3.0001, 0.02)
        K = F * np.exp(z * atm * np.sqrt(T))
        K = K[(K >= F) if is_call else (K <= F)]
        iv = self.iv(date, T, K, F, atm)
        px = bs.price(row["spot"], K, T, row["r"], row["q"], iv, is_call)
        delta = bs.greeks(row["spot"], K, T, row["r"], row["q"], iv, is_call)["delta"]
        return pd.DataFrame({"K": K, "iv": iv, "price": px, "delta": delta, "T": T, "F": F})

    def mark(self, date, expiry, K: float, is_call: bool) -> float:
        date = pd.Timestamp(date)
        row = self.df.loc[date]
        T = year_frac(date, expiry)
        if T <= 0:
            return float(max(row["spot"] - K, 0) if is_call else max(K - row["spot"], 0))
        iv = self.iv(date, T, K, self.forward(date, expiry))
        return float(bs.price(row["spot"], K, T, row["r"], row["q"], iv, is_call))


# --------------------------------------------------------------------------- NSE (real)
class NSEChain(OptionMarket):
    """Real NIFTY monthly options. Prices are NSE daily settlement prices."""

    def __init__(self, min_contracts: float = 1.0):
        self.name = "IN"
        self.df = load_market("IN")
        chain = pd.read_parquet(DATA_PROCESSED / "nifty_chain.parquet")
        chain = chain[chain["date"].isin(self.df.index)].copy()
        # contractual expiry can fall on an exchange holiday (e.g. 2023-06-29): NSE then
        # expires the contract on the previous trading day
        trading = self.df.index
        uniq = pd.DatetimeIndex(chain["expiry"].unique())
        actual = {e: (trading[trading <= e][-1] if e <= trading[-1] and e not in trading else e) for e in uniq}
        chain["expiry"] = chain["expiry"].map(actual)
        fut = chain[chain["kind"] == "FUT"]
        self.expiries = pd.DatetimeIndex(sorted(fut["expiry"].unique()))
        self._fwd = fut.set_index(["date", "expiry"])["settle"]

        opt = chain[(chain["kind"] == "OPT") & chain["expiry"].isin(fut["expiry"].unique())].copy()
        opt = opt.join(self._fwd.rename("F"), on=["date", "expiry"])
        opt = opt.join(self.df[["spot", "r"]], on="date").dropna(subset=["F", "r"])
        opt["T"] = (opt["expiry"] - opt["date"]).dt.days / 365.0
        opt = opt[opt["T"] > 0]
        opt["is_call"] = opt["opt_type"] == "CE"
        # Black-76 on the futures price: BSM with S=F and q=r
        opt["iv"] = bs.implied_vol(
            opt["settle"].values, opt["F"].values, opt["strike"].values, opt["T"].values,
            opt["r"].values, opt["r"].values, opt["is_call"].values,
        )
        opt["delta"] = np.nan
        ok = opt["iv"].notna()
        g = bs.greeks(opt.loc[ok, "F"].values, opt.loc[ok, "strike"].values, opt.loc[ok, "T"].values,
                      opt.loc[ok, "r"].values, opt.loc[ok, "r"].values, opt.loc[ok, "iv"].values,
                      opt.loc[ok, "is_call"].values)
        opt.loc[ok, "delta"] = g["delta"]
        opt["traded"] = opt["contracts"] >= min_contracts
        self.opt = opt
        self._by_key = opt.set_index(["date", "expiry", "strike", "is_call"]).sort_index()
        self._settle = dict(zip(zip(opt["date"], opt["expiry"], opt["strike"], opt["is_call"]), opt["settle"]))

    def forward(self, date, expiry) -> float:
        try:
            return float(self._fwd.loc[(pd.Timestamp(date), pd.Timestamp(expiry))])
        except KeyError:
            row = self.df.loc[pd.Timestamp(date)]
            return float(row["spot"] * np.exp((row["r"] - row["q"]) * year_frac(date, expiry)))

    @lru_cache(maxsize=4096)
    def _slice(self, date, expiry) -> pd.DataFrame:
        try:
            return self._by_key.loc[(pd.Timestamp(date), pd.Timestamp(expiry))]
        except KeyError:
            return pd.DataFrame()

    def smile(self, date, expiry, is_call: bool) -> pd.DataFrame:
        s = self._slice(pd.Timestamp(date), pd.Timestamp(expiry))
        if s.empty:
            return s
        s = s.xs(is_call, level="is_call").reset_index()
        F = self.forward(date, expiry)
        s = s[s["traded"] & s["iv"].notna() & ((s["strike"] >= F) if is_call else (s["strike"] <= F))]
        return s.rename(columns={"strike": "K", "settle": "price"})[["K", "iv", "price", "delta", "T", "F"]]

    def mark(self, date, expiry, K: float, is_call: bool) -> float:
        date, expiry = pd.Timestamp(date), pd.Timestamp(expiry)
        spot = float(self.df["spot"].asof(date))
        if date >= expiry:
            return max(spot - K, 0.0) if is_call else max(K - spot, 0.0)
        px = self._settle.get((date, expiry, K, is_call))
        if px is not None:
            return float(px)
        # contract missing that day: reprice with IV of the nearest listed strike
        s = self._slice(date, expiry)
        if s.empty:
            return np.nan
        s = s.reset_index()
        s = s[s["iv"].notna()]
        near = s.iloc[(s["strike"] - K).abs().argmin()]
        T = year_frac(date, expiry)
        r = float(self.df.loc[date, "r"])
        F = self.forward(date, expiry)
        return float(bs.price(F, K, T, r, r, near["iv"], is_call))

    def atm_iv(self, date, days: float = 30.0) -> float:
        """ATM IV interpolated in total variance across listed monthly expiries."""
        date = pd.Timestamp(date)
        pts = []
        for exp in self.expiries[self.expiries > date + pd.Timedelta(days=4)][:3]:
            s = self._slice(date, exp)
            if s.empty:
                continue
            s = s.reset_index()
            s = s[s["iv"].notna() & s["traded"]]
            if s.empty:
                continue
            F = self.forward(date, exp)
            near = s.iloc[(s["strike"] - F).abs().argsort()[:4]]  # 2 strikes x call/put
            pts.append((year_frac(date, exp) * 365, float(near["iv"].median())))
        if not pts:
            return np.nan
        t, v = np.array(pts).T
        tv = v**2 * t
        d = float(np.clip(days, t[0], t[-1]))
        return float(np.sqrt(np.interp(d, t, tv) / d))


def get_market(name: str, **kw) -> OptionMarket:
    return NSEChain(**kw) if name == "IN" else SyntheticSurface(name, **kw)
