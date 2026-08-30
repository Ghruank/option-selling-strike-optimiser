"""End-to-end pipeline. Usage:  uv run python -m vrp_alpha.pipeline [--download]

Stages: data -> surface calibration -> VRP monitor -> exercise probabilities ->
backtests (+ risk-aversion sensitivity) -> Greeks / stress / live recommendations.
Every artefact lands in data/processed/ and is read by the report and dashboard.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from vrp_alpha.analytics import calibration, vrp
from vrp_alpha.backtest.engine import Strategy, book_greeks, default_strategies, run_backtest
from vrp_alpha.config import DATA_PROCESSED, load_config
from vrp_alpha.data import market as market_data
from vrp_alpha.data import nse
from vrp_alpha.models import probability
from vrp_alpha.optimizer.strikes import score_candidates
from vrp_alpha.pricing.surface import get_market
from vrp_alpha.risk import metrics, stress

MARKETS = ["US", "EU", "UK", "IN"]
OUT = DATA_PROCESSED / "results"
SENSITIVITY_LAMBDAS = [0.0, 0.01, 0.02, 0.05, 0.10]


def _stage(msg: str) -> float:
    print(f"\n=== {msg}", flush=True)
    return time.time()


ATTRIBUTION = [
    ("GDE 25-delta", ("delta", 0.25), ("delta", 0.25)),
    ("Puts 25-delta, no calls", ("delta", 0.25), None),
    ("Puts optimised, calls 25-delta", ("opt",), ("delta", 0.25)),
    ("Puts optimised, no calls", ("opt",), None),
    ("GDE Optimised", ("opt",), ("opt",)),
]


def run_attribution(markets, fhs, tables) -> pd.DataFrame:
    """Separate the two decisions the optimiser makes: put strike and whether to overwrite."""
    w = load_config()["portfolio"]["equity_weight"]
    nav_all = pd.read_parquet(OUT / "nav.parquet")
    rows = {}
    for m, mk in markets.items():
        bench = nav_all.query("market == @m and strategy == 'Equity (TR)'").set_index("date")["nav"]
        for name, put_rule, call_rule in ATTRIBUTION:
            nav, tr = run_backtest(mk, Strategy(name, w, put_rule, call_rule), fhs, tables[m])
            p = metrics.performance(nav["nav"], bench, mk.df["r"])
            rows[(m, name)] = {**{k: p[k] for k in ["CAGR", "Vol", "Sharpe", "MaxDD"]},
                               "put_exercise_rate": metrics.option_stats(tr, nav["nav"]).get("put_exercise_rate")}
    df = pd.DataFrame(rows).T
    df.index.names = ["market", "variant"]
    df.to_csv(OUT / "attribution.csv")
    return df


def run(download: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()

    if download or not (DATA_PROCESSED / "index_closes.parquet").exists():
        _stage("download market data")
        market_data.download()
    if download or not (DATA_PROCESSED / "nifty_chain.parquet").exists():
        _stage("download NSE bhavcopy (slow, cached per day)")
        nse.download()
        nse.build_chain()

    _stage("calibrate synthetic surface to CBOE benchmarks")
    rep = calibration.run()
    print(rep["out_of_sample"].round(4).to_string())

    _stage("volatility risk premium monitor")
    markets = {m: get_market(m) for m in MARKETS}
    tables = {m: vrp.vrp_table(mk) for m, mk in markets.items()}
    vrp.save(tables)
    print(vrp.summary(tables).round(4).to_string())

    _stage("exercise probabilities: risk-neutral vs FHS vs ML")
    probability.run(markets, tables)
    print(pd.read_csv(DATA_PROCESSED / "prob_scores.csv", index_col=0).round(4).to_string())
    fhs = probability.make_fhs(tables)

    _stage("backtests")
    navs, trades, perf, opt_stats, eps = [], [], {}, {}, {}
    for m, mk in markets.items():
        r = mk.df["r"]
        runs = {s.name: run_backtest(mk, s, fhs, tables[m]) for s in default_strategies()}
        bench = runs["Equity (TR)"][0]["nav"]
        for name, (nav, tr) in runs.items():
            navs.append(nav.assign(market=m, strategy=name))
            if not tr.empty:
                trades.append(tr.assign(market=m, strategy=name))
            perf[(m, name)] = metrics.performance(nav["nav"], bench, r)
            opt_stats[(m, name)] = metrics.option_stats(tr, nav["nav"])
        eps[m] = metrics.episodes({k: v[0]["nav"] for k, v in runs.items()})
        print(m, "done", flush=True)
    pd.concat(navs).rename_axis("date").reset_index().to_parquet(OUT / "nav.parquet")
    pd.concat(trades, ignore_index=True).to_parquet(OUT / "trades.parquet")
    perf_df = pd.DataFrame(perf).T.join(pd.DataFrame(opt_stats).T)
    perf_df.index.names = ["market", "strategy"]
    perf_df.to_csv(OUT / "performance.csv")
    pd.concat(eps, names=["market", "episode"]).to_csv(OUT / "episodes.csv")
    print(perf_df[["CAGR", "Vol", "Sharpe", "MaxDD", "Alpha_ann", "put_exercise_rate"]].round(3).to_string())

    _stage("risk-aversion sensitivity (GDE Optimised)")
    w = cfg["portfolio"]["equity_weight"]
    sens = {}
    for m, mk in markets.items():
        bench = pd.read_parquet(OUT / "nav.parquet").query("market == @m and strategy == 'Equity (TR)'")
        bench = bench.set_index("date")["nav"]
        for lam in SENSITIVITY_LAMBDAS:
            s = Strategy(f"lambda={lam}", w, ("opt",), ("opt",), risk_aversion=lam)
            nav, tr = run_backtest(mk, s, fhs, tables[m])
            p = metrics.performance(nav["nav"], bench, mk.df["r"])
            sens[(m, lam)] = {**{k: p[k] for k in ["CAGR", "Vol", "Sharpe", "MaxDD"]},
                              **metrics.option_stats(tr, nav["nav"])}
    sens_df = pd.DataFrame(sens).T
    sens_df.index.names = ["market", "lambda"]
    sens_df.to_csv(OUT / "sensitivity.csv")
    print(sens_df[["CAGR", "Sharpe", "MaxDD", "put_exercise_rate", "put_avg_delta"]].round(3).to_string())

    _stage("attribution: put strike selection vs call overwrite decision")
    print(run_attribution(markets, fhs, tables).round(3).to_string())

    _stage("Greeks timeline, stress tests, live recommendations")
    nav_all = pd.read_parquet(OUT / "nav.parquet")
    tr_all = pd.read_parquet(OUT / "trades.parquet")
    greeks_rows, stress_rows, recs = [], [], []
    for m, mk in markets.items():
        for strat in ["GDE ATM", "GDE 25-delta", "GDE Optimised"]:
            nv = nav_all.query("market == @m and strategy == @strat").set_index("date")
            tr = tr_all.query("market == @m and strategy == @strat")
            for d in nv.index[nv.index.weekday == 4]:
                legs = stress.book_at(tr, d)
                g = book_greeks(mk, legs, d, nv.at[d, "nav"])
                g["delta_total"] = g["delta"] + nv.at[d, "equity"] / nv.at[d, "nav"]
                greeks_rows.append({"market": m, "strategy": strat, "date": d, **g})
            last_entry = tr["entry"].max()
            legs = stress.book_at(tr, last_entry)
            st = stress.stress_book(mk, legs, last_entry, nv["nav"].asof(last_entry),
                                    w * nv["nav"].asof(last_entry))
            stress_rows.append(st.assign(market=m, strategy=strat, book_date=last_entry))
        # equity-only reference
        st = stress.stress_book(mk, pd.DataFrame(), mk.dates[-1], 1.0, 1.0)
        stress_rows.append(st.assign(market=m, strategy="Equity (TR)", book_date=mk.dates[-1]))
        st = stress.stress_book(mk, pd.DataFrame(), mk.dates[-1], 1.0, w)
        stress_rows.append(st.assign(market=m, strategy="50/50 Equity/Cash", book_date=mk.dates[-1]))

        d = mk.dates[-1]
        e = mk.next_expiry(d, 21)
        for is_call in (False, True):
            sc = score_candidates(mk, fhs, d, e, float(tables[m].at[d, "ens"]), is_call)
            recs.append(sc.assign(market=m, date=d, expiry=e, side="call" if is_call else "put"))
    pd.DataFrame(greeks_rows).to_parquet(OUT / "greeks.parquet")
    pd.concat(stress_rows, ignore_index=True).to_csv(OUT / "stress.csv", index=False)
    pd.concat(recs, ignore_index=True).to_parquet(OUT / "recommendations.parquet")
    print("pipeline complete ->", OUT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true", help="re-download all market data")
    ap.add_argument("--attribution-only", action="store_true", help="rerun only the attribution stage")
    args = ap.parse_args()
    if args.attribution_only:
        mks = {m: get_market(m) for m in MARKETS}
        tbl = vrp.load()
        print(run_attribution(mks, probability.make_fhs(tbl), tbl).round(3).to_string())
    else:
        run(download=args.download)
