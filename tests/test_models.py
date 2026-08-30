import numpy as np
import pandas as pd
import pytest

from vrp_alpha.config import DATA_PROCESSED
from vrp_alpha.models.probability import FHS
from vrp_alpha.pricing.surface import third_friday_expiries


def _fhs(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    resid = pd.DataFrame({"market": "X", "z": rng.standard_t(5, n),
                          "available": pd.Timestamp("2020-01-01")})
    return FHS(resid, erp=0.0)


def test_fhs_put_probability_monotone_and_cvar_dominates_mean():
    K = np.linspace(80, 100, 11)
    out = _fhs().evaluate("2021-01-01", 100.0, K, 30 / 365, 0.02, 0.0, 0.2, False)
    assert np.all(np.diff(out["p_itm"]) >= 0)            # higher put strike -> more likely exercised
    assert np.all(out["cvar95"] >= out["e_payoff"] - 1e-12)
    assert np.all((out["p_itm"] >= 0) & (out["p_itm"] <= 1))


def test_fhs_ignores_residuals_not_yet_observable():
    resid = pd.DataFrame({"market": "X", "z": [-50.0] * 100, "available": pd.Timestamp("2030-01-01")})
    fhs = FHS(resid, erp=0.0)
    z, w = fhs.sample("2021-01-01")
    assert len(z) == 2000 and abs(w.sum() - 1) < 1e-9     # prior only: future data invisible


def test_third_friday_expiries():
    dates = pd.bdate_range("2024-01-01", "2024-06-30")
    exp = third_friday_expiries(dates)
    assert list(exp.strftime("%Y-%m-%d")) == ["2024-01-19", "2024-02-16", "2024-03-15",
                                              "2024-04-19", "2024-05-17", "2024-06-21"]
    # Good-Friday style holiday: third Friday missing -> previous trading day
    exp = third_friday_expiries(dates.drop(pd.Timestamp("2024-03-15")))
    assert pd.Timestamp("2024-03-14") in exp


@pytest.mark.skipif(not (DATA_PROCESSED / "vrp.parquet").exists(), reason="pipeline data not built")
def test_equity_only_backtest_tracks_total_return():
    from vrp_alpha.analytics import vrp
    from vrp_alpha.backtest.engine import Strategy, run_backtest
    from vrp_alpha.models.probability import make_fhs
    from vrp_alpha.pricing.surface import get_market

    mk = get_market("US")
    tables = vrp.load()
    nav, trades = run_backtest(mk, Strategy("eq", 1.0), make_fhs(tables), tables["US"])
    s = mk.df["spot"]
    start, end = nav.index[0], nav.index[-1]
    expected = s[end] / s[start] * np.exp(mk.df["q"].iloc[0] * (end - start).days / 365)
    assert trades.empty
    assert abs(nav["nav"].iloc[-1] / expected - 1) < 1e-9
