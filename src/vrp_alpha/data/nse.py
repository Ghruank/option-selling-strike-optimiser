"""NSE F&O bhavcopy loader: real end-of-day NIFTY option chains.

NSE publishes a free daily bhavcopy (settlement file) for the F&O segment.
Two formats cover the study window:
  * legacy  (up to 2024-07-05): archives.nseindia.com/content/historical/DERIVATIVES/...
  * UDiFF   (2024 onwards):     nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_...
Only NIFTY index options and futures are kept and normalised to one schema.
"""

from __future__ import annotations

import io
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date

import pandas as pd
import requests

from vrp_alpha.config import DATA_PROCESSED, DATA_RAW, load_config

RAW_DIR = DATA_RAW / "nse"
UDIFF_FROM = date(2024, 7, 8)
HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)", "Accept": "*/*"}
COLUMNS = ["date", "kind", "expiry", "strike", "opt_type", "close", "settle", "contracts", "oi"]


def _legacy_url(d: date) -> str:
    mon = d.strftime("%b").upper()
    return (
        f"https://archives.nseindia.com/content/historical/DERIVATIVES/"
        f"{d.year}/{mon}/fo{d.strftime('%d')}{mon}{d.year}bhav.csv.zip"
    )


def _udiff_url(d: date) -> str:
    return (
        "https://nsearchives.nseindia.com/content/fo/"
        f"BhavCopy_NSE_FO_0_0_0_{d.strftime('%Y%m%d')}_F_0000.csv.zip"
    )


def _parse_legacy(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    raw = raw[(raw["SYMBOL"] == "NIFTY") & raw["INSTRUMENT"].isin(["OPTIDX", "FUTIDX"])]
    return pd.DataFrame(
        {
            "date": pd.Timestamp(d),
            "kind": raw["INSTRUMENT"].map({"OPTIDX": "OPT", "FUTIDX": "FUT"}),
            "expiry": pd.to_datetime(raw["EXPIRY_DT"], format="%d-%b-%Y"),
            "strike": raw["STRIKE_PR"].astype(float),
            "opt_type": raw["OPTION_TYP"].str.strip(),
            "close": raw["CLOSE"].astype(float),
            "settle": raw["SETTLE_PR"].astype(float),
            "contracts": raw["CONTRACTS"].astype(float),
            "oi": raw["OPEN_INT"].astype(float),
        }
    )


def _parse_udiff(raw: pd.DataFrame, d: date) -> pd.DataFrame:
    raw = raw[(raw["TckrSymb"] == "NIFTY") & raw["FinInstrmTp"].isin(["IDO", "IDF"])]
    return pd.DataFrame(
        {
            "date": pd.Timestamp(d),
            "kind": raw["FinInstrmTp"].map({"IDO": "OPT", "IDF": "FUT"}),
            "expiry": pd.to_datetime(raw["XpryDt"]),
            "strike": raw["StrkPric"].fillna(0).astype(float),
            "opt_type": raw["OptnTp"].fillna("XX").str.strip(),
            "close": raw["ClsPric"].astype(float),
            "settle": raw["SttlmPric"].astype(float),
            "contracts": raw["TtlTradgVol"].astype(float),
            "oi": raw["OpnIntrst"].astype(float),
        }
    )


def fetch_day(d: date, session: requests.Session, retries: int = 4) -> str:
    """Download and cache one trading day. Returns a status string."""
    out = RAW_DIR / f"{d:%Y%m%d}.parquet"
    if out.exists():
        return "cached"
    candidates = [(_udiff_url, _parse_udiff)] if d >= UDIFF_FROM else [
        (_legacy_url, _parse_legacy),
        (_udiff_url, _parse_udiff),
    ]
    for url_fn, parse in candidates:
        for attempt in range(retries):
            try:
                r = session.get(url_fn(d), headers=HEADERS, timeout=30)
            except requests.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 404:
                break
            if r.status_code != 200:
                time.sleep(2 ** attempt)
                continue
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                raw = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
            raw.columns = raw.columns.str.strip()
            parse(raw, d)[COLUMNS].to_parquet(out, index=False)
            return "ok"
    return "missing"  # weekend-adjacent holiday or not published


def download(start: str | None = None, end: str | None = None, workers: int = 4) -> None:
    cfg = load_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range(start or cfg["start"], end or cfg["end"]).date
    session = requests.Session()
    with ThreadPoolExecutor(workers) as pool:
        statuses = list(pool.map(lambda d: fetch_day(d, session), days))
    counts = pd.Series(statuses).value_counts().to_dict()
    print(f"NSE bhavcopy: {len(days)} business days -> {counts}")


def build_chain() -> pd.DataFrame:
    """Concatenate cached days into one NIFTY chain table (options + futures)."""
    files = sorted(RAW_DIR.glob("*.parquet"))
    chain = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    chain = chain.sort_values(["date", "kind", "expiry", "opt_type", "strike"])
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    chain.to_parquet(DATA_PROCESSED / "nifty_chain.parquet", index=False)
    return chain


if __name__ == "__main__":
    download()
    build_chain()
