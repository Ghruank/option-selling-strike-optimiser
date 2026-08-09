"""Free market data: index levels, volatility indices, short rates, CBOE benchmarks.

Sources
  * Yahoo Finance  - index closes (^GSPC, ^STOXX50E, ^FTSE, ^NSEI) and vol indices
                     (^VIX9D, ^VIX, ^VIX3M, ^SKEW, ^INDIAVIX)
  * FRED           - DTB3 (US 3m T-bill), EUR short-term rate, SONIA
  * RBI            - policy repo rate schedule (hard-coded, public record)
  * CBOE           - PUT / BXM / BXMD / PPUT strategy-benchmark index histories
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from vrp_alpha.config import DATA_PROCESSED, load_config

VOL_TICKERS = {"VIX9D": "^VIX9D", "VIX": "^VIX", "VIX3M": "^VIX3M", "SKEW": "^SKEW", "INDIAVIX": "^INDIAVIX"}
CBOE_BENCHMARKS = ["PUT", "BXM", "BXMD", "PPUT"]

# RBI policy repo rate (effective date, %). Source: RBI monetary policy statements.
RBI_REPO = [
    ("2020-03-27", 4.40), ("2020-05-22", 4.00), ("2022-05-04", 4.40), ("2022-06-08", 4.90),
    ("2022-08-05", 5.40), ("2022-09-30", 5.90), ("2022-12-07", 6.25), ("2023-02-08", 6.50),
    ("2025-02-07", 6.25), ("2025-04-09", 6.00), ("2025-06-06", 5.50), ("2025-12-05", 5.25),
]


def _yahoo_closes(tickers: dict[str, str], start: str, end: str) -> pd.DataFrame:
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(list(tickers.values()), start=start, end=end_excl, auto_adjust=False, progress=False)
    closes = raw["Close"].rename(columns={v: k for k, v in tickers.items()})
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes[list(tickers)]


def _fred(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(io.StringIO(requests.get(url, timeout=90).text), na_values=".")
    return df.set_index(pd.to_datetime(df.iloc[:, 0])).iloc[:, 1].astype(float).rename(series_id)


def _rbi_repo(index: pd.DatetimeIndex) -> pd.Series:
    s = pd.Series({pd.Timestamp(d): r for d, r in RBI_REPO})
    return s.reindex(s.index.union(index)).ffill().reindex(index).rename("RBI_REPO")


def _cboe(name: str) -> pd.Series:
    url = f"https://cdn.cboe.com/api/global/us_indices/daily_prices/{name}_History.csv"
    df = pd.read_csv(io.StringIO(requests.get(url, timeout=60).text))
    return df.set_index(pd.to_datetime(df["DATE"], format="%m/%d/%Y"))[name].astype(float)


def download() -> None:
    cfg = load_config()
    start, end = cfg["start"], cfg["end"]
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    idx = _yahoo_closes({m: c["index"] for m, c in cfg["markets"].items()}, start, end)
    vol = _yahoo_closes(VOL_TICKERS, start, end)
    idx.to_parquet(DATA_PROCESSED / "index_closes.parquet")
    vol.to_parquet(DATA_PROCESSED / "vol_indices.parquet")

    rates = {}
    for m, c in cfg["markets"].items():
        sid = c["rate_series"]
        s = _rbi_repo(pd.bdate_range(start, end)) if sid == "RBI_REPO" else _fred(sid)
        rates[m] = s.loc[start:end]
    pd.DataFrame(rates).to_parquet(DATA_PROCESSED / "rates.parquet")

    bench = pd.DataFrame({b: _cboe(b) for b in CBOE_BENCHMARKS}).loc[start:end]
    bench["SPX"] = idx["US"]
    bench.to_parquet(DATA_PROCESSED / "cboe_benchmarks.parquet")
    print("market data:", {k: len(v) for k, v in [("index", idx), ("vol", vol), ("bench", bench)]})


def load_market(market: str) -> pd.DataFrame:
    """Daily frame on the market's own trading calendar.

    Columns: spot, r (cont. comp. decimal), q, plus US vol indices lagged one US
    session (so EU/UK/IN closes never see a VIX print from later the same day),
    and INDIAVIX for India.
    """
    cfg = load_config()
    mcfg = cfg["markets"][market]
    idx = pd.read_parquet(DATA_PROCESSED / "index_closes.parquet")[market].dropna()
    rates = pd.read_parquet(DATA_PROCESSED / "rates.parquet")[market]
    vol = pd.read_parquet(DATA_PROCESSED / "vol_indices.parquet")

    df = pd.DataFrame({"spot": idx})
    r_pct = rates.reindex(rates.index.union(df.index)).ffill().reindex(df.index)
    df["r"] = np.log1p(r_pct / 100.0)
    df["q"] = mcfg["div_yield"]

    us_vol = vol[["VIX9D", "VIX", "VIX3M", "SKEW"]].dropna(subset=["VIX"])
    if market != "US":
        us_vol = us_vol.shift(1)
    us_vol = us_vol.reindex(us_vol.index.union(df.index)).ffill().reindex(df.index)
    df = df.join(us_vol)
    if market == "IN":
        iv = vol["INDIAVIX"].dropna()
        df["INDIAVIX"] = iv.reindex(iv.index.union(df.index)).ffill().reindex(df.index)
    return df.dropna(subset=["spot", "r", "VIX"])


def load_benchmarks() -> pd.DataFrame:
    return pd.read_parquet(DATA_PROCESSED / "cboe_benchmarks.parquet")


if __name__ == "__main__":
    download()
