"""Walk-forward volatility forecasts (the 'physical-measure' volatility).

Two standard models, both refit on an expanding window every REFIT days using only
data available at the forecast date:

  * HAR-RV (Corsi 2009) on daily squared returns - daily / weekly / monthly components
  * GJR-GARCH(1,1) with Student-t innovations    - leverage effect + fat tails

The production forecast is the equal-weight average of the two variances (forecast
combination is more robust than either model alone). Horizon: next H trading days.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from arch import arch_model

H = 21          # forecast horizon, trading days (~1 month)
MIN_OBS = 120   # burn-in before the first forecast
REFIT = 21


def _har_design(r2: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "d": r2,
        "w": r2.rolling(5).mean(),
        "m": r2.rolling(22).mean(),
    })


def har_forecast(returns: pd.Series) -> pd.Series:
    r2 = returns**2
    X = _har_design(r2)
    y = r2[::-1].rolling(H).mean()[::-1].shift(-1)  # mean r^2 over t+1..t+H
    out = pd.Series(np.nan, index=returns.index)
    beta = None
    for i, t in enumerate(returns.index):
        if i < MIN_OBS:
            continue
        if beta is None or i % REFIT == 0:
            # rows whose target window ended by t (no look-ahead)
            train = pd.concat([X, y.rename("y")], axis=1).iloc[: i - H].dropna()
            A = np.column_stack([np.ones(len(train)), train[["d", "w", "m"]].values])
            beta = np.linalg.lstsq(A, train["y"].values, rcond=None)[0]
        x = X.iloc[i]
        if x.isna().any():
            continue
        f = beta[0] + beta[1:] @ x.values
        out.iloc[i] = max(f, 0.25 * x["m"])  # floor: OLS on r^2 can go negative
    return np.sqrt(252 * out)


def garch_forecast(returns: pd.Series) -> pd.Series:
    r = 100 * returns
    out = pd.Series(np.nan, index=returns.index)
    params = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(MIN_OBS, len(r)):
            hist = r.iloc[: i + 1]
            am = arch_model(hist, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
            if params is None or i % REFIT == 0:
                params = am.fit(disp="off", show_warning=False).params
            fc = am.fix(params).forecast(horizon=H, reindex=False)
            out.iloc[i] = fc.variance.values[-1].mean() / 1e4
    return np.sqrt(252 * out)


def forecast_table(spot: pd.Series) -> pd.DataFrame:
    """Per-date annualised vols: trailing RV, HAR, GARCH, ensemble and ex-post forward RV."""
    ret = np.log(spot).diff().dropna()
    har = har_forecast(ret)
    garch = garch_forecast(ret)
    df = pd.DataFrame({
        "ret": ret,
        "rv21": np.sqrt(252 * (ret**2).rolling(21).mean()),
        "har": har,
        "garch": garch,
    })
    df["ens"] = np.sqrt(0.5 * (df["har"] ** 2 + df["garch"] ** 2))
    # ex-post realised vol over the next H days: evaluation / labels only, never a feature
    df["fwd_rv"] = np.sqrt(252 * (ret**2)[::-1].rolling(H).mean()[::-1].shift(-1))
    return df


def qlike(forecast_var: pd.Series, realised_var: pd.Series) -> float:
    ok = forecast_var.notna() & realised_var.notna() & (realised_var > 0)
    ratio = realised_var[ok] / forecast_var[ok]
    return float((ratio - np.log(ratio) - 1).mean())
